[PYTHON_TO_RUST_PORTING_GUIDE.md](https://github.com/user-attachments/files/35444747/PYTHON_TO_RUST_PORTING_GUIDE.md)
# rusted-ports
port to rust?  look no further
# Moving a Python codebase to Rust: a working guide for AI agents

This guide is written for an AI coding agent (and the people directing it) that has been asked to
move Python code to Rust. Load it as project instructions, an `AGENTS.md` section, or an agent
definition. It is drawn from three sets of ports in one project between March and November 2026:

| When | What was ported | Kind of port |
|---|---|---|
| March 2026 | A bioinformatics toolkit ("rusted rotifer"): about ten PyO3 crates, 872 native tests at its July snapshot | Hot functions moved under Python. Python stays the caller and keeps a fallback. |
| May to June 2026 | A GPU driver layer: 20 files (about 10,300 lines) ported from C, a daemon of about 32,000 lines, and a small daemon that replaced a Python bridge | Systems code where Rust owns a device. |
| November 2026 | A whole GPU compute path: launch rules, a run loop, an HMM fit, a profile search, a run sequencer and a native launch service; about 47,000 lines of Rust and 12,700 lines of tests in under three days | The Python path replaced end to end. Python stays as the oracle and for host tooling. |

It complements Anthropic's post [Migrating our Messages API to Rust](https://www.anthropic.com/research/migrating-api-to-rust)
(November 25, 2026). That post covers a stateless web service checked against live traffic. This
guide covers what that post does not: numerical code, the Python/Rust boundary, GPU kernels,
compiler toolchains, and systems that hold state or hardware. The last section maps one onto the
other.

## Fill this in before the agent starts

An agent given this guide needs these facts about your project. Leave none blank; "not known" is an
answer the agent can act on, a guess is not.

| Fact | Your project |
|---|---|
| The reference implementation (path, version, how to run it) | |
| What it must match: bytes, values within a tolerance, or decisions only | |
| The tolerance for each numeric output, and the reason for each | |
| Inputs the reference cannot be run on (private data, live hardware, billing) | |
| Commands the agent may never run (anything that opens hardware, formats the whole tree, deploys) | |
| The test commands, and which tests open real devices or services | |
| What stays in Python, and why | |
| Who decides whether a difference is a bug or an intended change | |

<!-- shared:start -->

## Decide what to port

Port because a measurement says a specific piece is the problem: a profile, a timeout record, a
deployment constraint. "Rust is faster" is not a measurement.

Good candidates:
- Tight loops over data. In the toolkit port, a triple-nested alignment loop ran 188 to 238 times
  faster in Rust than in Python.
- Process and device lifecycle: code that has to hold a device, wait on child processes, survive
  signals and never be half-updated.
- Rules that guard something expensive, once they have stopped changing.

Leave in Python, and say so in the plan:
- I/O-bound code, glue, argument parsing, plotting, one-off scripts.
- Code already calling an optimised library. A 512-wide matrix multiply was half as fast in Rust
  as in NumPy, because NumPy was already calling the platform BLAS.
- Code whose cost is at the boundary. A record parser was half as fast in Rust once its output had
  to be converted back to Python dictionaries.
- Code still being changed weekly. A port freezes it.
- Safety checks that have one home. In the GPU project the host-safety gates stayed in one Python
  script that the Rust sequencer runs and never reimplements: two copies of a safety rule drift,
  and the copy that drifts is the one nobody is watching. If that script cannot run, the Rust side
  refuses to proceed.

Write the "stays in Python" list down with a reason per item. A port with no such list tends to
grow until it stalls.

## Define "correct" before writing Rust

The Python code is the oracle. The port is correct when it does what Python does, including what
Python does by accident. Where the two differ, a named person decides whether that is a bug to keep,
a bug to fix in both, or an intended change. The agent does not decide that alone, and it never
guesses what Python does: it runs Python and looks.

If the reference cannot be run where you are working, you may still plan from reading the code.
Label every statement about behaviour as read, not run, and put running the reference first in the
steps you hand to someone else.

Build the checks as a ladder. Each rung is cheaper and faster than the one above it, and each
catches what the one below cannot.

1. **Decision corpus.** Write a Python script that drives the reference over shape grids, limit
   boundaries and deliberately bad inputs, and dumps every decision (accept or refuse, and the
   computed arguments). The Rust code must match every line. Two rule sets were ported this way,
   against 12,298 and 28,210 recorded cases; in the second, 18,994 of the cases were refusals.
   Refusals matter most: a rule port that accepts what Python refused is the dangerous kind of wrong.
2. **Golden outputs made by Python itself.** Run the reference end to end on small inputs, commit
   its output directory, and require byte equality for every array and file. Exclude by name the
   fields that differ per run (timestamps, timings). An exclusion list that grows is a warning.
3. **Python as a live oracle.** For logic that builds commands, plans or records, have a test call
   both sides and compare word for word: a Rust test that runs the Python function in a subprocess,
   or a Python test that calls the reference and the real wrapper. Import pure functions only;
   never call an entry point that acts. This rung also catches a later change on the Python side.
4. **The real service code on an emulated device.** Put the production code path behind an
   emulated backend with fault and refusal injection, and give the emulated device an identity the
   real safety checks would refuse, so a test can never reach real hardware by mistake.
5. **Bit identity across backends.** If the same program runs on a CPU emulator, a second GPU API
   and the target device, pick one as the oracle (the CPU emulator) and compare the others launch
   by launch.
6. **The real system equals the emulator's hash.** A real run passes only when its final state has
   the hash the emulator produced for the same inputs. This turns "it ran without errors" into a
   check with one right answer.

Not every rung applies to every port. A single numeric function behind a binding has rungs 1 to 3
and rung 5 in the form "every implementation against one oracle". Say which rungs do not apply and
why, so a missing rung is a decision and not an omission.

Rules that keep the ladder honest:
- A test that is skipped because a fixture is missing is reported as skipped. Many test runners
  count it as a pass, and then it proves nothing on that checkout.
- Never weaken an assertion to make a ported test pass. If the assertion was wrong, say so and get
  the decision recorded.
- A parity test must call the real wrapper. The toolkit's fallback-parity tests compared the raw
  extension with private copies of the Python code, never the wrappers users call, so a wrapper
  that passed the wrong arguments was not caught.
- Keep machine paths out of fixtures. Generated fixtures leaked absolute paths in 34 places; the
  generator now fails on any home-directory or temporary-directory path.

## Numerical parity

Decide per output whether you are claiming identical bits or a tolerance, and write down which.
If you are given a tolerance without its reason, proceed, and report whether the code can meet
something tighter or cannot meet it at all.

Bit identity is achievable when you control the order of operations:
- Mirror the reference operation for operation in the same precision: same order of multiplies and
  adds, same reduction tree, same polynomial coefficients.
- Disable fused multiply-add on both sides. A fused `a*b+c` rounds once instead of twice and
  differs by up to one unit in the last place. Treat a one-ulp difference as a finding, never as
  noise: it is usually a contraction or a rounding rule.
- Fix summation order. Parallel sums use a fixed tree from a fixed starting value; every other sum
  is one ascending loop.

Hidden numerics that had to be reproduced to match Python byte for byte:
- **Random streams.** Matching a fit required NumPy's seeding, its PCG64 generator and the exact
  algorithm behind `choice(..., replace=False)`. "Same seed" means nothing across libraries.
- **BLAS summation order.** The Python reference itself gave different bits under two NumPy builds
  (one on Accelerate, one on OpenBLAS). The port models both and the test names which one it is
  matching. Check whether your reference is one reference before you chase a difference.
- **Float formatting.** Python's `repr` of a float and NumPy's float32 rounding had to be
  reproduced so that log files matched.
- **Precision of the fallback.** If the Rust path casts to f32 and the Python fallback computes in
  f64, the two paths are not the same function. Either make them the same or state the tolerance.
- **Ties and choices.** Where the output is a choice (a traceback, an argmax), define the tie set
  from the reference's own values before comparing, require an exact match outside it, and record
  the reference's order of preference inside it. A reference that chooses with float `==` makes
  the choice depend on exact bits: match the bits, or get a decision on a tie rule.

Where bit identity is not reachable (an eigen-solver, a library you do not control), assert a
stated tolerance with its reason, for example "1 to 2 ulp from a different summation order,
checked to 1e-14 relative". Do not write "approximately equal" without a number.

Classic parity bugs found by comparing against Python: an affine-gap boundary off by one extension,
a traceback that chose by float `==` in Python and by a stored argmax in Rust, two index layouts
for the same alphabet, and a lowercase character treated as a gap on one side only.

## The Python/Rust boundary

When Rust is called from Python (PyO3 and maturin), most of the work and most of the bugs are at
the boundary. Budget as much time for it as for the Rust itself.

- **Count the crossings.** Anthropic's first attempt moved only an output parser to Rust; each
  token then crossed the boundary three times and the saving was 8 to 10 percent. Move a whole
  stage, or pass a batch across once.
- **Lock the versions together.** The PyO3 crate, the NumPy crate and maturin are upgraded as one
  change. Build in the same Python environment you run in; the wrong interpreter fails at import
  with a missing-symbol error.
- **Release the GIL for CPU work.** Extract Python data into Rust types, release the lock, compute,
  re-acquire to build the result. Holding it blocks every other Python thread.
- **Do not assume contiguous arrays.** `as_slice().unwrap()` panics on a sliced or transposed NumPy
  array. Convert to standard layout or return an error.
- **Return plain Python types** (numbers, strings, lists, dicts, arrays). A custom class forces the
  fallback to construct the same class.
- **Keep the fallback honest.** A wrapper that tries Rust and falls back to Python on any exception
  will hide a broken Rust path for months. Fall back only on "extension not installed", log every
  fallback, and give users one switch that turns the Rust path off. Catch the import with a broad
  exception class, because an ABI mismatch is not always an `ImportError`.
- **Route by measurement.** Keep a table of which functions go to Rust always, never, or only
  below or above a size, and fill it from benchmarks.
- **Test the extension both ways.** Drive it from the Python test runner, and make the native test
  runner work too: build without the extension-module feature, point the build at the interpreter,
  and set the library search path to that interpreter's library directory.
- **Parallelise above a threshold** found by measurement. Task dispatch has a cost, and small
  batches are slower in parallel.
- **Leave stdin, terminal detection and output formatting in Python.** Users pipe output into other
  tools; a changed column or float format breaks them silently.

When Rust calls a C++ library that Python also wraps: write a thin C shim, declare `extern "C"`
functions over opaque handles, and link against the same shared library file that the Python wheel
loads. That is what makes bit-for-bit parity with the Python binding possible. Then find out what
the Python binding does for you without saying so. A lazy tensor graph driven from Rust ran at 152
to 176 ms per step until the port called the library's graph compiler, as the Python decorator
does; compiled, it ran at 44 ms against Python's 50 on the same machine and day.

## GPU kernels

A kernel is not ported alone. Treat one kernel set as a group of files that change together, and
refuse to install it unless every member is present:

| Member | What it is for |
|---|---|
| Rust kernel source | The code compiled for the device. |
| Executable specification in NumPy | The same arithmetic, operation for operation, in the kernel's precision. This is the reference. |
| CPU mock | Decodes the same argument bytes the device would receive and runs the specification, refusing launches the device would get wrong. |
| The kernel source compiled for the host | The same Rust source built against a mock device library and run on threads. |
| Validating launcher | The only supported way to launch. A kernel cannot see buffer sizes, so the launcher derives block and grid, checks ranges, extents and aliasing, and refuses the rest. |
| ABI golden table | Parameter offsets and sizes decoded from the compiled binary, pinned in a test. |
| Hash pins and a build manifest | Hashes of every input and output and the toolchain versions. The loader refuses a binary that does not match its pin. |
| Device bit-compare | A zero-tolerance comparison on the device, rehearsed first on the mock. |

That gives three implementations (Rust on the CPU, the NumPy specification, the device) that must
agree. Practices that made this work:

- **Build without contraction and lint the output.** Compile so multiplies and adds stay separate
  instructions with an explicit rounding mode, then scan the generated assembly and fail the build
  on any fused instruction or any unmarked floating-point operation the assembler might fuse.
- **Hash inputs before the first build step and re-check after every step.** An input edited
  mid-build deletes the outputs. There is deliberately no switch for reusing an old intermediate.
- **Port the launcher's rules with a decision corpus** (rung 1 above). In the native path the
  service does not check buffer sizes either, so the rules are the only guard against an
  out-of-bounds write on the device.
- **Prove the comparison can fail.** Inject a one-ulp error and require every kernel's check to
  fail.
- **Add a static check for stale reads.** A mock cannot see a read of last step's values, because
  they look plausible. A check that every byte read was written earlier in the same step can.
- **Run the emulator as the oracle for long runs.** Making the emulator's thread blocks run as
  coroutines brought a production-shaped check from not finishing to seconds per step, which is
  what made rung 6 practical.
- **Expect each backend to have its own concurrency bug.** One GPU API needed a barrier that made
  a whole group reach it together (a kernel hung about one run in ten without it); another needed
  an explicit barrier between launches that share a buffer. The CPU oracle found both.

## Compiler and toolchain work

For a project that compiles Rust to something other than a host binary (here: Rust to MIR, to an
intermediate dialect, to LLVM IR, to GPU assembly):

- **Dump every stage.** Have one command that prints the compiler's input IR, each intermediate
  form before and after each pass, and the final assembly. Dump before verification, so a failed
  verification still shows what was produced. The first question in any code-generation bug is
  which stage first has the wrong thing.
- **Pin the toolchain and record it.** Pin the nightly by date in a toolchain file and keep every
  crate that uses compiler-internal APIs on exactly that date; those APIs change between
  nightlies. Do not put a nightly date in the manifest's minimum-version field, which accepts only
  stable version numbers. Record the back-end assembler's version and hash in the build manifest.
- **Keep the compiler-internal crate out of the main workspace.** It needs different flags and a
  different build flow, and one workspace-wide command would otherwise fail everywhere.
- **Know which optimisations change meaning for your target.** One MIR pass duplicated barrier
  calls and had to be disabled for kernel builds. Minimum versions matter too: newer target
  intrinsics were rejected by the previous major version of the back end.
- **Gate platform-only crates.** Keep them in the tree behind target checks and in one exclude list
  that the build file owns. Trust the build file, not the prose: here the written list had six
  crates and the build file nine.
- **Make "is the toolchain healthy" a host-only command.** A health check that also contacts a
  device or a daemon fails for unrelated reasons and gets ignored.

## Systems that hold state or hardware

A web service can be shadowed and switched over request by request. A long run that holds a device
cannot. What worked instead:

- **Mixed operation by design.** The Rust sequencer writes the same record files, with the same
  schema and exit codes, as the Python one, so a run started by either can be continued by the
  other. That is the rollback: the Python path keeps working on the same directory.
- **Explicit takeover.** Continuing a Python-started run with the Rust runner needs a flag, and the
  takeover is recorded in the run's own records. In the other direction the Python runner refuses a
  directory the Rust one has taken. A lock file keeps both from running at once.
- **Bind a resume to what it resumes.** A key derived from the constants and the launch list ties a
  checkpoint to the program that wrote it. Records from the old runner lack the key and fall back
  to checkpoint hashes.
- **Hash your own binary.** The sequencer hashes its executable at start and before every segment
  and stops if it changed. A rebuild during a multi-hour run would otherwise swap the code under
  it. Run a copy named by its hash, in a directory no build writes to.
- **Signals and children.** While a device is held, record signals and act on them at a safe point.
  Never signal or kill a child that holds a device; judge it from the files it leaves. One review
  found that a signal during preflight was cleared and the device started anyway.
- **Never let logging kill the supervisor.** `println!` panics when the terminal has gone. Route
  output through one function that drops failed writes, or a closed terminal ends a supervisor
  whose child is still running.
- **Fail closed on missing evidence.** An unreadable health reading is not a clean one. Two reviews
  found the opposite in first drafts.
- **Separate a refusal from a fault.** An API that returns only a code made every failure look like
  a device fault; an out-of-memory refusal would have locked out a healthy device. One side-effect
  free question to the service after a failed call tells the two apart.
- **Write records durably.** Sync before rename, and treat a record that could not be written as an
  error with its own exit code.
- **Set timeouts across layers on purpose.** The client, the library and the service had 60, 30
  and 90 seconds, each chosen alone. A layer that gives up before the one below it has answered
  turns a slow reply into a fault.
- **A port does not fix the layer below.** The Rust path lost the hardware link at the same points
  the Python path had. Do not credit or blame the port for what the hardware does, and do not
  claim a stability gain that was not measured under matched conditions.

## Tooling traps

- **Do not format the whole tree.** A whole-tree format rewrote 11 files outside the change,
  including kernel sources whose hashes were pinned (some were untracked, so version control could
  not restore them). Format only the files you edited.
- **Find the tests that open real hardware or services before running a suite.** Three integration
  tests in one crate opened a real device with no gate, and a full run of a script test suite
  executed a real recovery command. Search the tests for the open call first, and run named files.
- **Check what is in a commit.** A commit limited to named files swept in someone else's
  uncommitted edits to one of those files. Read the diff of every listed file first.
- **Check workspace membership.** New crates that are members only through an uncommitted manifest
  build on your machine and nowhere else.
- **Do not rebuild a binary a run is using.** A supervisor that starts its worker by path will pick
  up the new one mid-run.
- **Clear stale bytecode after a rename**, and benchmark release builds only.
- **Suspect your own documentation.** A porting guide written during the toolkit port stated two
  things about Rust that were wrong (that a once-initialised cell is poisoned for good by a panic
  in its initialiser, and that worker threads deadlock on the GIL). Check a claim about the
  language against the language's documentation before it becomes a rule.

## Review and measurement

- **Use reviewers that did not write the code**, more than one, and require every one to say yes.
  Give them the code and the question, not your conclusion. In the last week of this work most
  first review rounds drew at least one no, and three drew a no from all three reviewers. Each no
  was a real defect: an option that had no effect because it was set after the process it
  configured had started, a fix that would not have saved the run it was written for, a lock
  released while the device was still in use.
- **A verifier tells you outputs matched, not that the change is fit to ship.** Anthropic's post
  makes the same point. A fix that makes the comparison pass by copying the expected value is a
  pass and a defect.
- **Benchmark honestly.** State the baseline (a pure-Python loop is not a NumPy baseline), the
  thread counts on both sides, the input sizes, and where Rust lost. The toolkit's own suite of
  126 cases gave a geometric mean of 6.8 times with Rust ahead in 94; its matrix multiply and
  small set-similarity cases were slower than NumPy. Its preregistered validation protocol has
  not been run, so those are development figures, and the guide says so here on purpose.
- **Do not compare runs that differ in more than the language.** No controlled Python-versus-Rust
  comparison exists for the device path in this project: dates, flags and segment lengths
  differed, and the two were in the same range. Report that as "not measured".
- **Write each bug down when it is fixed**, with its cause. This guide exists because that was
  done.

## Working procedure

1. Collect the project facts listed at the top of this document. If the reference, the tolerance
   or the forbidden commands are missing, ask before writing code. If you cannot ask, do the work
   that does not depend on the answer and return the open questions first.
2. Run the reference on a small input and keep its output outside the source tree. Send its caches
   elsewhere too: running a reference can write into its own directory. If it cannot be run, say
   so first and continue only as a reading-only plan.
3. List what will be ported, what stays, and why. Get that list agreed by whoever asked for the
   port.
4. Build rung 1 or 2 of the ladder for the first unit before writing its Rust.
5. Port the smallest unit that has a complete check. Port what Python does. Where you cannot tell
   what Python does, run it; do not infer it from its name or its comments.
6. Make the check pass without touching the check. Add a test that fails on the old Rust and passes
   on the new for every bug fixed.
7. Climb the ladder as units join: subprocess oracle, emulated service, cross-backend, then the
   real system. Stop before any step that needs hardware, a deployment or someone else's data, and
   hand the exact command to the person who runs it.
8. Before cutover, show mixed operation: a run started by each side and continued by the other, or
   the documented reason it is not possible.
9. Format only the files you edited, run named test files, read the diff of every file in the
   commit.
10. Ask for independent review, fix what it finds, and ask again until every reviewer says yes.

## What to return

Give the result first, then the evidence:

- What was ported, what was deliberately left, and what is not done.
- For each parity claim: the rung, the number of cases, the result, and whether it is bit identity
  or a tolerance (with the tolerance and its reason).
- Every difference from Python that remains, with who has to decide on it. If nobody was named,
  propose someone and list it as open.
- Tests that were skipped, and why.
- What you ran yourself, apart from what you inferred or were told.
- Performance only with its baseline, sizes and build mode, and the cases where Rust was slower.
- Any step you were not allowed to run, or that needs hardware or a deployment, as the exact
  command for the person who will run it. Never run it yourself.

<!-- shared:end -->

## How this complements Anthropic's post

The post's central claim is that once writing code is cheap, verification is the bottleneck. Every
section above is a way of building a verifier for a kind of code the post does not discuss.

| In the post (a web service) | Here (numerical, GPU and device code) |
|---|---|
| The Python API is the oracle, and it keeps changing | The Python runner is the oracle; a subprocess-oracle test catches later Python changes, and the safety rules that must not fork stay in Python |
| The Rust API passed about 98% of its tests while about a quarter of production inputs matched | Tests are one rung; a decision corpus of tens of thousands of cases and the emulator's hash on the real system are the others |
| Checkpoints: compare at each step so the first difference names the code | Compare launch by launch across backends; dump every compiler stage |
| Fingerprints: the shape of a difference, since request text could not be read | A hash of the final state; a one-ulp difference treated as a finding with a cause |
| More than 100 weakened assertions in ported tests | Parity tests that never called the real wrapper; skipped tests counted as passes |
| "Port exactly what Python did"; an agent then found seven more guesses | Reproduce the random stream, the BLAS order and the float formatting; run Python, never infer it |
| A verifier cannot tell whether a fix is good enough to ship | Independent reviewers who must all say yes |
| A merge check: a Python change needs a matching Rust change or an exemption | Shared record schemas, takeover flags and a lock, so either side can continue a run |
| Shadowed traffic, then moving users a few at a time | No shadowing is possible with one device: mixed operation, emulated service, binary self-hash |
| Not covered: numerics, the binding layer, GPU kernels, compilers, hardware | The sections above |

What this project did not have and the post does: production traffic as a verifier, and a measured
before-and-after under matched conditions. If you have those, use them; they are stronger than
anything here.

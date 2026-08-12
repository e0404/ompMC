---
applyTo: "ucodes/omc_python/**/*.cpp,ucodes/omc_python/**/*.h,ucodes/omc_python/ompmc/**/*.py"
---

This is the nanobind-based Python extension. `omc_python.cpp` is the only C++
translation unit in the whole repository — the physics core stays C99 on
purpose. Like the MEX host, this should be a thin translator with no physics
logic of its own; see [AGENTS.md](../../AGENTS.md).

When reviewing:

- **The GIL must be released around any OpenMP-parallel engine call.** Only
  the binding glue and the progress callback should touch the GIL — check
  that a new engine invocation or callback path doesn't hold it across the
  transport loop (that would serialize what should be a parallel run) or
  release it around code that actually touches Python objects.
- **Optional parameters need explicit `.none()` annotation.** In nanobind, a
  parameter does not accept Python `None` unless annotated
  `"name"_a.none()`; a missing annotation fails at the call site with a
  generic "incompatible function arguments" error that gives no hint what's
  wrong. Check any new optional/nullable-looking parameter for this.
- **`int verbose_flag` (or any other `extern` global the C core declares)
  must be defined exactly once by the host**, not the core — a missing
  definition is a link failure, not a runtime bug, but is easy to introduce
  when adding a new host-side translation unit.
- **`nb::ndarray<const T>` vs. writable arrays.** Confirm the constness of a
  new ndarray parameter matches whether the engine writes through it.
- **No `exit()`/`abort()` reachable from Python-callable code.** Failures must
  go through `omcFail()`/`omcLog()`, whose Python sink uses an exception, not
  a process-killing call — an uncaught `exit()` here kills the whole Python
  interpreter mid-session (this has happened before this boundary was
  cleaned up).
- **Cross-check numerical changes against the MATLAB path**, not just Python
  tests in isolation. This project maintains byte-level cross-validation
  between the Python and MEX outputs on the same fixture (`mex_reference.mat`
  in `ucodes/omc_python/tests/`) specifically to catch the two hosts drifting
  apart; a PR that changes shared engine behavior should not leave that
  comparison untouched or unexplained.
- Prefer extending the shared engine (`src/omc_engine_*.h`) over adding
  Python-only logic, unless the change is genuinely about the Python API
  (numpy conversion, GIL handling, packaging).

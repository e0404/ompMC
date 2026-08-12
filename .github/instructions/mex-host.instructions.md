---
applyTo: "ucodes/omc_matrad/**/*.c,ucodes/omc_matrad/**/*.h"
---

This is the MATLAB/Octave MEX host. It should be a thin translator —
`mxArray` in, sparse `dij`/dense cube out — with no physics logic of its own;
see [AGENTS.md](../../AGENTS.md). When reviewing:

- **No `exit()`, `printf`, or process-killing error handling.** All failures
  must go through `omcFail()`/`omcLog()` so the installed MEX sink can turn
  them into `mexErrMsgIdAndTxt()`-style errors instead of crashing MATLAB.
  A bad input file or malformed struct field must never take down the whole
  MATLAB session.
- **`mxArray` memory ownership.** Memory handed to MATLAB (e.g. via
  `mxSetPr`) must come from `mxMalloc`/MATLAB's allocator, not a
  plain `malloc` the engine owns — check both directions of any buffer that
  crosses the MEX boundary for who allocates and who frees it.
- **Required-field / input validation.** New `mcOpt`/geometry/source struct
  fields read from MATLAB should go through the existing required-field
  validation helper rather than being read unchecked — a missing or
  wrong-typed field should produce a clear `omcFail()` message, not a segfault
  or silent wrong value.
- **Precedence rules for redundant inputs** (e.g. spectrum vs. spectrumFile
  vs. monoEnergy) are meaningful and were chosen deliberately, with the loser
  announced via `omcLog()`. Don't collapse or silently change that precedence.
- **Column-major vs. row-major.** MATLAB is column-major; the C core is
  row-major. A new array crossing this boundary needs an explicit
  transpose/reordering step, not an assumption that a `memcpy` is safe — check
  whether one was needed and whether it was applied consistently (this
  project has a documented case where a rotation matrix needed transposing on
  the way in but a transmission matrix, already the right layout, did not).
- Prefer extending the shared engine (`src/omc_engine_*.h`) over adding
  MEX-only logic, unless the change is genuinely about the MEX/MATLAB API
  (memory ownership, error reporting, struct parsing).

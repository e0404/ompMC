## What this repository is

ompMC is a Monte Carlo photon/electron transport engine for radiotherapy dose
calculation: a C99 physics core (`src/`) shared by three host programs — a
command-line binary (`ucodes/omc_dosxyz`), a MATLAB/Octave MEX file
(`ucodes/omc_matrad`), and a Python extension (`ucodes/omc_python`, C++
binding via nanobind). It is OpenMP-parallelized and used for real treatment
planning, so silent numerical or physics regressions are the main risk, not
crashes.

Read [AGENTS.md](../AGENTS.md) before reviewing: it lists patterns in this
codebase that look wrong but are intentional. Do not flag those as bugs.

## How to review here

This codebase has a documented history of subtle, hard-to-spot correctness
bugs surviving into commits: De Morgan errors in rejection sampling, off-by-one
errors in the last bin of a lookup table, a variance divisor swapped in a
batch-statistics formula, an epsilon/clamp bug at a geometric boundary, and a
Windows-only data-corruption bug from opening a binary file in text mode.
Apply extra scrutiny — treat as **high priority** — to any change touching:

- **`src/*.c` / `src/*.h`** — the physics core. See
  `.github/instructions/core-physics.instructions.md` for the specific
  invariants (RNG draw order, byte-reproducibility, `exit()` discipline,
  variance-reduction unbiasedness) and use the `code-review` agent skill's
  checklist when reviewing changes here.
- **Random number consumption.** Any change to the number, order, or
  conditionality of `setRandom()`/random-draw calls per history is a
  correctness bug even if it compiles and runs: it desynchronizes the RNG
  stream from every downstream draw in that history and silently shifts the
  physics. This has caused a real regression before (spectrum sampling drawing
  randoms unconditionally instead of only for a polyenergetic source).
- **Anything reachable from inside an `omp parallel` region.** `exit()` may
  only be called from the three sites that are documented as safe
  (`initStack`, `checkStackSpace`, `photon`'s stack-overflow path); everywhere
  else, error handling must go through `omcFail()`/`omcLog()`
  (`src/omc_host.h`), which the host's installed sink can turn into a
  `longjmp`/exception instead of killing the whole process. Flag any new
  `exit()`, `abort()`, unguarded `printf`/`mexErrMsgIdAndTxt`, or call into a
  MATLAB/Python API from worker-thread-reachable code.
- **Variance-reduction techniques** (Woodcock tracking, electron range
  rejection, Russian roulette, and anything similar). These must stay provably
  unbiased. A PR introducing or changing one should show — in the description,
  a test, or a comment — how unbiasedness was checked (typically a multi-seed
  statistical comparison against the unmodified path), not just that it runs
  faster. "It's faster" without a bias check is not enough to approve.
- **Cross-platform build/runtime differences.** MSVC's OpenMP is C89/2.0-only
  in C mode and rejects a loop-variable declared inside `#pragma omp for`;
  binary data files (`pegs4/`, `data/`, `.egsphant`, phase-space files) must be
  opened with `"rb"`/`"wb"`, never text mode, or Windows silently corrupts them
  via CRLF translation.
- **The engine/host boundary** (`src/omc_engine_dij.*`, `src/omc_engine_cube.*`,
  `src/omc_engine_forward.*`, `src/omc_geom.*`, `src/omc_source*.*`). This
  layer exists specifically so `omc_matrad` and `omc_python` cannot drift
  apart. A change to one host that isn't reflected in the other, or that
  duplicates logic instead of extending the shared engine, should be flagged.

## General expectations

- This is C99 in `src/` and the two C-based user codes; only the Python
  binding TU (`ucodes/omc_python/omc_python.cpp`) is C++. Don't suggest C++-only
  constructs (RAII, STL containers, exceptions) for `src/` or the C user codes.
- Favor correctness and clarity over cleverness in numerically sensitive code;
  a suggestion to "simplify" a formula in `src/` should come with a note on
  whether it changes the computed value, not just the source length.
- Comment style in this repo explains *why*, not *what* — don't ask for
  comments that restate what the code obviously does.
- Don't request changes purely for style/formatting unless they conflict with
  the surrounding file's existing convention.
- MCP tools: the GitHub MCP server is useful here (checking CI status, prior
  related PRs/issues). The Playwright MCP server is not relevant — this
  project has no web UI.

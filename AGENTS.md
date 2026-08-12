# Agent context for ompMC

This file exists to stop reviewers (human or AI) from "fixing" things that are
deliberate. If you're reviewing a diff and something here looks odd, this is
why.

## Patterns that look like bugs but aren't

- **`exit(EXIT_FAILURE)` still exists in three places in `src/ompmc.c`**
  (`initStack`, `checkStackSpace`, and inside `photon()`'s stack-overflow
  path) even though the rest of the codebase converted 33 of 36 former
  `exit()` sites to `omcFail()` so a bad input file doesn't kill the whole
  host process (MATLAB, Python). These three are reachable from inside an
  `omp parallel` region; a `longjmp` from a worker thread into the master
  thread's `jmp_buf` (what `omcFail()` does for the Python host) would be
  undefined behavior, so they intentionally stay as a hard process abort.
  Don't convert them without solving that problem first.

- **The dosxyz-style cube engine's uncertainty output is *relative* sigma with
  a `0.9999999` sentinel for empty voxels**, while the Dij (beamlet) engine
  reports *variance of the mean*. This looks like an inconsistency between
  `src/omc_engine_cube.*`/`src/omc_score.*` and `src/omc_engine_dij.*` — it
  isn't. The two engines report genuinely different statistics because
  they're consumed differently downstream (a `.3ddose`-style file vs. an
  optimizer's dose-influence matrix). Don't "unify" them.

- **The RNG is Philox4x32-10, not a conventional stream RNG**, keyed on a
  fixed `rng seeds` value with the *global history index* placed in the
  counter's high 64 bits (`src/omc_random.*`). This is what makes runs
  byte-identical regardless of thread count and repeatable from any history
  index — it's the reason a pure code-layout refactor of the physics core can
  be verified with `cmp` against a pre-change binary. Don't replace it with
  something that seeds per-thread or per-batch instead of per-history.

- **`omcSpectrumSample()` must draw zero random numbers for a monoenergetic
  source and exactly the historical count for a polyenergetic one.** A
  refactor that makes the random draw unconditional (even if mathematically
  equivalent for the monoenergetic case) desynchronizes every later draw in
  that history from the RNG stream and silently changes the computed dose.
  This has happened once already during the Python-interface work.

- **Electron range rejection and electron Russian roulette
  (`vrt.esave`/`e_rr`/`f_rr`) are off by default**, not because they're
  unvalidated but because they were measured unbiased-but-inefficient at
  clinical settings (3 mm voxels, `ecut` 0.521): a 4–8% time saving traded for
  17–131% higher slice-variance, i.e. a net efficiency *loss*. Don't enable
  them by default or suggest that as a "quick win" without new benchmarking
  data.

- **The phase-space (`phsp`) reader's `$CHECKSUM` is the binary file's size in
  bytes**, not `recordLength × particle count` — this matches the IAEA
  reference tool (`iaea_check_file_size_byte_order`), even though deriving it
  from the record count looks like the more "obvious" formula and is wrong.
  Similarly, a header's `$PARTICLES` count is treated as informational (a
  mismatch is logged, not fatal) because real published files have been found
  to over-count by one record.

- **The IAEA phase-space reader currently only supports `$BYTE_ORDER 1234`**
  (little-endian) and refuses big-endian files cleanly. This is a known,
  deliberate scope limit, not an oversight — don't ask for silent best-effort
  handling of the untested path.

- **The collimator's transmission mask attenuates by weight by default and
  draws no random numbers**, keeping a collimated run comparable
  history-by-history with the open-field run. `OMC_MODIFIER_ROULETTE` is an
  opt-in alternative that does draw randoms. Don't suggest roulette as the
  default.

## Architectural intent worth knowing

- `src/` is deliberately plain C99, not C++, even though the tree also
  contains one C++ translation unit (`ucodes/omc_python/omc_python.cpp`, the
  nanobind binding). The physics stays C on purpose; only the binding layer is
  allowed to be C++.
- The engines in `src/omc_engine_*.h` exist so `omc_matrad` (MATLAB) and
  `omc_python` cannot drift into two implementations of the same physics —
  each host is meant to be a thin translator (`mxArray`/numpy in, sparse
  matrix or dense cube out) with no physics logic of its own. A host-specific
  fix that isn't traceable to a real host-only concern (memory ownership,
  MATLAB/Python API quirks) usually belongs in the shared engine instead.
- Error/log reporting goes through `omcLog()`/`omcFail()`
  (`src/omc_host.h`), never `printf`/`mexErrMsgIdAndTxt` directly, so each
  host can install its own sink. Both are documented as master-thread-only —
  they are not safe to call from inside a parallel region except at the three
  sites noted above.

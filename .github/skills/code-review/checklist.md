# ompMC review checklist

Work through the items relevant to the diff. Each one names a bug class this
project has actually shipped or explicitly guarded against — treat a "yes" as
something to flag, not necessarily block, but always surface with a concrete
explanation of the failure mode.

## Randomness and reproducibility

- [ ] Does the diff change how many times, in what order, or under what
      condition `setRandom()` (or a helper that calls it) is invoked per
      history? Even a mathematically-equivalent reordering shifts every
      subsequent draw in that history.
- [ ] If the diff touches `src/omc_random.*`, does it preserve per-history
      keying (so `omc_dosxyz` output stays byte-identical run-to-run
      regardless of thread count)?
- [ ] Does the PR say how a numerically-sensitive change was validated —
      exact `cmp` against a pre-change binary (valid for `omc_dosxyz`), or a
      statistical comparison with a stated sample size and result (required
      for the non-byte-reproducible matRad MEX path)? "It builds and the
      smoke test passes" is not sufficient evidence for a physics change.

## Control flow and threading

- [ ] Does the diff add any `exit()`, `abort()`, unguarded `printf`, or a
      MATLAB/Python API call in code reachable from inside an
      `omp parallel` region, outside the three documented exceptions
      (`initStack`, `checkStackSpace`, `photon`'s stack-overflow path)?
- [ ] Does new error-handling code go through `omcFail()`/`omcLog()`
      (`src/omc_host.h`) rather than a host-specific call, so all three hosts
      handle it correctly?
- [ ] Does a new `#pragma omp for` loop declare its loop variable inline?
      (Fails to compile under MSVC's C-mode OpenMP, which is 2.0-only.)

## Physics/numerical logic

- [ ] In any new or changed boolean condition (especially rejection sampling
      or a boundary/region check), is a negation distributed correctly —
      i.e., is `!(a && b)` written as `!a || !b`, not `!a && !b`?
- [ ] In any new or changed loop over an energy/angle table, is the last bin
      handled the same way as the others (no off-by-one truncation or
      duplicate copy)?
- [ ] In any new or changed variance/normalization formula
      (`src/omc_score.*`), is the divisor the one the statistics actually
      call for — batch count vs. history count vs. particle count are easy to
      swap and each compiles fine?
- [ ] Does a new epsilon/clamp at a geometric or energy boundary look like it
      fixes the underlying condition, or does it just mask a boundary case
      (e.g., silently pushing a particle that landed exactly on a voxel face)?
- [ ] If a hot-path cache or memoization is added or modified, are all
      invalidation points covered — including at medium/voxel boundaries?
- [ ] If a hot struct/table layout changes, were all access sites updated
      consistently? A partial update compiles and produces wrong physics
      silently.
- [ ] If a variance-reduction technique (Woodcock tracking, range rejection,
      Russian roulette, splitting) is added or changed, does the PR
      demonstrate it stays unbiased, not just faster?

## Language/memory-safety boundaries

- [ ] MEX (`ucodes/omc_matrad`): does any buffer crossing the MEX boundary
      have a clear, correct owner (`mxMalloc` vs. the engine's own
      allocator), in both directions?
- [ ] MEX: does a new array crossing the MATLAB/C boundary need a
      column-major/row-major transpose, and if so, is it applied — and only
      where actually needed (not every array needs it; check per-array, not
      by analogy)?
- [ ] Python (`ucodes/omc_python`): is the GIL released around the
      OpenMP-parallel engine call and held only in the binding glue/progress
      callback?
- [ ] Python: does every new optional/nullable parameter that should accept
      `None` carry the nanobind `.none()` annotation?
- [ ] Python: does a new `nb::ndarray<...>` parameter's constness match
      whether the engine writes through it?
- [ ] Does a data file that is binary (PEGS4, `.egsphant`, phase-space files)
      get opened with `"rb"`/`"wb"`, not text mode?

## Architecture

- [ ] Does new logic belong in the shared engine (`src/omc_engine_*.h`,
      `src/omc_geom.*`, `src/omc_source*.*`) rather than being duplicated or
      added host-only in `omc_matrad.c` or `omc_python.cpp`? A host file
      should only translate its native types in and out.
- [ ] If shared engine behavior changed, were both the MATLAB and Python test
      suites (or at least their cross-validation fixture) considered, not
      just the one host the PR author was working in?

---
applyTo: "src/**/*.c,src/**/*.h"
---

This is the shared transport engine (`ompmc_core`) linked by all three user
codes. It is C99, OpenMP-parallelized, and numerically sensitive: a change
that "looks" behavior-preserving can silently shift the computed dose. See
[AGENTS.md](../../AGENTS.md) for specific patterns that look wrong but are
intentional — check it before flagging something here.

When reviewing a diff in this directory:

1. **Count and order of random draws.** If the diff touches anything that
   calls `setRandom()` (directly or via a sampling helper like
   `omcSpectrumSample()`), check whether the number of draws per history or
   their order/conditionality changed. A change here desynchronizes the RNG
   stream for the rest of that history even when each individual draw is
   still mathematically correct in isolation. Ask for a byte-identical (or
   documented statistically-equivalent) validation against a pre-change build
   when this is touched.
2. **`exit()`/`abort()` reachability.** Error paths must go through
   `omcFail()`/`omcLog()` (`src/omc_host.h`) unless the call site is one of
   the three documented `exit()` sites reachable from a parallel region
   (`initStack`, `checkStackSpace`, `photon`'s stack-overflow path — see
   AGENTS.md). A new direct `exit()`, `abort()`, or unguarded `printf` inside
   transport code (`photon()`, `electron()`, the interaction routines, or
   anything called from inside an `omp parallel` region) is a bug: it would
   kill the MATLAB/Python host process instead of failing gracefully, or
   (worse) run from a worker thread where `omcLog`/`omcFail` are documented as
   unsafe.
3. **Logical/comparison operators in rejection sampling and boundary checks.**
   This codebase has previously shipped a De Morgan's-law error in a
   rejection condition and an off-by-one in a lookup table's last bin. Read
   inverted conditions (`!(a && b)` vs `!a || !b`), boundary comparisons
   (`<` vs `<=`), and loop bounds in newly touched physics formulas carefully
   rather than assuming they're correct because the surrounding code is
   dense.
4. **Divisors and normalization constants in scoring/variance code**
   (`omc_score.*`). A swapped or misapplied divisor in a batch-variance or
   normalization formula produces plausible-looking but wrong numbers — it
   won't crash or look obviously broken in a spot check.
5. **Epsilon guards and clamps at geometric/energy boundaries.** Check that a
   changed or added epsilon/clamp doesn't just mask a boundary bug (e.g. a
   particle landing exactly on a voxel face or `z` bound) rather than fix it.
6. **Cache/memoization correctness.** Several hot-path caches exist (e.g.
   per-medium cross-section lookups, voxel-index memoization). If a diff adds
   or touches one, verify the invalidation points are complete — a stale
   cache read at a medium/voxel boundary is the kind of bug that only shows
   up as a small statistical bias, not a crash.
7. **Variance-reduction techniques (Woodcock tracking, range rejection,
   Russian roulette, splitting).** Must remain provably unbiased. Expect the
   PR to state how unbiasedness was checked, not just report a speedup.
8. **Windows/MSVC-specific C constraints.** MSVC's OpenMP in C mode is
   2.0-only and rejects a loop variable declared inside `#pragma omp for` —
   flag any new `omp for` loop that declares its loop variable inline. Binary
   files (PEGS4 data, `.egsphant`, phase-space files) must be opened with
   `"rb"`/`"wb"` — a text-mode open of binary data has previously corrupted
   Windows builds silently (CRLF translation and 0x1A-as-EOF).
9. **Struct/table layout changes.** This code has been through deliberate
   layout changes for cache locality (e.g. interleaving coefficient pairs).
   If a diff changes a hot struct or table layout, check every access site
   was updated consistently — a partial update compiles fine and produces
   subtly wrong physics.

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

- **The pencil source's Gaussian spreads draw no random numbers when their
  sigma is zero**, for exactly the reason `omcSpectrumSample()` above draws
  none for a monoenergetic source: the stream is indexed per history, so
  drawing a Box-Muller pair and multiplying it by zero would shift every later
  draw in that history and silently change the dose of every beam that never
  asked for a spread. `tests/test_omc_source_pencil.c` pins the draw count of
  each combination; if you add a third blur, add its count there too.

- **A round beam cannot tell you which transverse frame you perturbed it in.**
  The pencil source's divergence is applied in the phantom's own x-y plane
  (`tiltDirection()`), not in a basis built perpendicular to the direction:
  for a beam along +z the obvious construction comes out as (y, -x), which
  pairs the x position with the y angle. Rotating a round, uncorrelated
  distribution changes nothing, so that bug passed every spot and divergence
  test and was only exposed by a `correlation`, which is stated per axis and
  so makes the frame observable. If you add anything else per-axis to a
  source, test it with something that is not round.

- **A measured dose profile is wider than the `spotSigma` that produced it,
  and that is not a bug.** Deposition is the incident fluence convolved with
  however far the radiation carries the energy, and convolution adds second
  moments: `sigma_dep^2 = sigma_src^2 + K`. K is large at low photon energies
  — about 8.8 cm² for 100 keV in water, the diffuse scattered-photon halo —
  so a 1 cm spot can deposit like a 3 cm one. Verified by fitting sigma_dep^2
  against sigma_src^2 over a range of widths: the slope is 1, which is what
  says the source width itself is right. Don't "correct" the source for it.

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

- **`howfar()`, `hownear()` and `regionIndex()` branch on a global,
  `struct Geom::mode`, on every call** rather than dispatching through a
  function pointer set once at initialization. On the face of it a table of
  pointers is the tidier answer, and it is the wrong one here: these are
  called once per electron step, and link time optimization — which
  `CMakeLists.txt` turns on largely for their sake — can inline a direct call
  and cannot inline an indirect one. A pointer table would therefore tax the
  rectilinear geometry, which is every existing user code, to buy the cylinder
  something it does not need. The mode cannot change during a run, so the
  branch predicts perfectly after the first call.

- **The cylindrical geometry has no struct of its own.** It borrows
  `struct Geom`, carrying rings in `isize` and depth slabs in `ksize` with
  `jsize` pinned to 1, so that the region numbering `1 + ir + iz*nr` is
  literally the rectilinear `1 + ix + iy*isize + iz*isize*jsize` with the y
  index held at zero. That is what lets `initRegions()`, `struct Score`,
  `ausgab()` and the region memo in `omc_utilities.h` serve a cylinder without
  a line of change or a second code path. Giving it its own struct would mean
  a second version of each of those. `omcGeomCylInit()` sets `jsize` itself
  rather than asking the host for it, because no host should have to know
  about the rectilinear grid it is borrowed from.

- **`omcGeomDetectSpacing()` sets `geometry.mode = OMC_GEOM_CARTESIAN` as a
  side effect**, which looks unrelated to detecting spacing. It is where the
  reset belongs: every loader of a voxel grid already calls it, so no host can
  forget, and a resident host (a MEX file, a Python module) that ran a
  cylinder and then a cube would otherwise transport the cube through the
  cylinder. The Python test suite runs both orders in one process for exactly
  this reason.

- **The radial scorer reports relative sigma with the `0.9999999` sentinel,
  like the cube engine and unlike the Dij engine.** That is the same
  deliberate split noted above, and `omcScoreToRadial()` is a near-copy of
  `omcScoreToCube()` on purpose: everything but the mass of a region — an
  annulus rather than a box — has to stay identical, so that a reader
  comparing an r-z result against a cube one never has to wonder which
  convention is in play.

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

## Release process

Day-to-day work merges into `develop`. `master` carries releases only, and is
what a `v*` tag is cut from.

1. **Branch.** `rc/<version>` off `develop` — `rc/0.3.0`, not `release/0.3.0`.
   The release candidate is what gets reviewed, so `develop` stays open for
   new work while it is.
2. **PR, carrying no version at all.** Open it into `master` straight away,
   with none of the step below in it. What is under review is the content
   going out, and that reads perfectly well without a version number on it.
3. **Bump, once that is approved.** One commit, carrying the version and the
   changelog entry that says what it is: a version without its entry describes
   nothing, and an entry without its version belongs to no release.

   The version lives in exactly one place, the `project(ompMC VERSION ...)`
   call in `CMakeLists.txt` — `pyproject.toml` (through scikit-build-core's
   regex provider), `docs/conf.py` and the C code's `OMPMC_VERSION_STRING` all
   read it from there, so nothing else is edited, and `CITATION.cff` carries no
   version field on purpose. `CHANGELOG.md` is Keep a Changelog: rename
   `## [Unreleased]` to `## [<version>] - <date>`, open a fresh empty
   `Unreleased`, and update the two link references at the bottom of the file.
4. **Merge, then tag.** Only tag once the PR is merged. Merge `master` back
   into `develop` afterwards, so the two do not drift.

**The bump waits for the approval.** Step 3 is mechanical, and it is tempting
to have it done before anyone looks — but a release branch that carries a
version before the release is agreed gets two things wrong. The date
in `## [<version>] - <date>` becomes the day the branch was cut rather than the
day the release went out, and review and CI are asynchronous enough for those
to differ by days. And the number itself is a claim ahead of its evidence: if
the review changes what ships, because something turns out to be breaking or
because something gets pulled, it can be the wrong number heading the wrong
list of changes.

**A tag is a publication, not a bookmark.** Pushing `v<version>` triggers two
irreversible things, so it is the last step rather than a way to mark a
commit:

- `wheels.yml` publishes `ompmc <version>` to PyPI through trusted publishing.
  PyPI never lets a version number be reused, so a mistake here is permanent.
- `release.yml` packages the per-platform zips `build.yml` already built and
  attaches them to a GitHub Release.

`release.yml` reacts to `build.yml` finishing via `workflow_run`, and GitHub
evaluates such a workflow's definition **only from the default branch**. It
therefore cannot package a tag pushed before the version of `release.yml` that
handles it reached `master` — which is the real reason step 4 tags after the
merge rather than before. Its `workflow_dispatch` input re-packages an existing
tag if that ordering is ever got wrong.

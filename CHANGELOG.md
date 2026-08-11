# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `omc_phsp`, a reader for IAEA format phase space files
  (`.IAEAheader`/`.IAEAphsp`), such as the ones published at
  <https://www-nds.iaea.org/phsp/>. It reads the header on its own or the
  whole file into memory, and hands the particles back either by index or by
  popping them one at a time. How many particles there are is counted from
  the file rather than believed from the header, which is what it takes to
  read the published datasets; a file that marks no histories is reported,
  since nothing drawing from one can group the particles it holds.
- `omc_source_phsp`, a source that starts histories from a phase space read
  by `omc_phsp`. It turns and moves the particles into the phantom's
  coordinate system, carries them to the face they enter it by, and puts one
  on the stack per history -- either replaying the file in order or drawing
  from it at random, in both cases decided by the history index alone so a
  run does not depend on how its histories were scheduled. A history whose
  particle misses the phantom, or is a neutron or proton, produces nothing
  and says so, which is why it returns a value the caller has to check before
  showering.
- `omc_collimator`, something in the beam's way: a `struct OmcBeamModifier`
  the forward engine applies between the source and the shower, and
  `struct OmcApertureMask`, a transmission mask on a plane. It works by back
  projection, so it composes with any source -- which is what makes it
  possible to cut a field out of a phase space recorded above the jaws, as
  the IAEA ones are. A single open cell is a rectangular field. It attenuates
  by weight rather than by roulette and draws no random numbers, so putting a
  collimator in the beam leaves every history's random stream where it was.
- `omc_source`, one interface every source of primary particles fills in. A
  source now answers only "which particle starts this history, and where is it
  going"; carrying it to the phantom, finding its voxel and counting the
  energy it brought are `omcSourcePlace()`'s job and the engine's, done once
  for everyone instead of once per source.

### Changed

- `omcCalcForward()` takes a beam modifier as well, `NULL` for an open beam.
  A particle it stops is stopped before being carried to the phantom, so a
  blocked history costs one plane intersection rather than a shower, and
  `struct OmcForwardSummary::blocked` reports how many there were.
- `omcCalcForward()` takes a `struct OmcSource` rather than beamlets and
  weights, so the same engine runs a fluence map or a phase space without
  knowing which. Weighted beamlets are dressed as one with
  `omcBeamletHistoriesAsSource()`, a phase space with
  `omcPhspSamplerAsSource()`, and `omcCalcForwardPhsp()` is gone. What the
  result means -- the dose for the weights given, or the dose per history --
  comes from the source too.
- `struct OmcForwardOptions` no longer carries the charge or the source
  geometry, and `struct OmcForwardSummary` no longer carries how the histories
  were shared out among beamlets; both belong to the source, and the latter is
  asked of it with `omcBeamletHistoriesStats()`.
- Beamlet particles now start at the source point and are flown to the phantom
  like everyone else's, instead of being walked backwards from the aperture.
  The ray is the same one, so the dose is unchanged: over the matRad fixture
  the total agrees to 5.1e-16 and per voxel to 1.2e-10, which is the size of
  the last-ulp differences a compiler change already makes.
- This changelog.
- Release packaging workflow (`release.yml`): on a `v*` tag, packages
  build.yml's binaries into per-platform zips (`omc_dosxyz` + the MATLAB MEX
  file per platform/toolchain, the Octave MEX file per platform/ABI bucket),
  each bundled with its data files and published to a GitHub Release.

### Changed

- build.yml's artifact collection now sorts binaries into per-target
  subfolders and vendors DLLs per binary, based on each binary's actual
  import table, instead of a blanket per-job DLL list -- the MinGW-built
  MATLAB MEX file and `omc_dosxyz` do not need the same runtime DLLs.

## [0.2.0] - 2026-08-06

First tagged release of this fork.

### Added

- Python interface (`pip install ompmc`) via nanobind, with prebuilt wheels for
  Linux (x86_64/aarch64), macOS (x86_64/arm64) and Windows, and a source
  distribution fallback.
- `forward_beamlet` dose-calculation mode (`mcOpt.mode` in MATLAB, `calc_forward`
  in Python): direct dose for a weighted field without building the full
  dose-influence matrix first.
- Documentation site (Sphinx, Doxygen/Breathe, sphinxcontrib-matlabdomain)
  covering the C, MATLAB, and Python APIs.
- `CITATION.cff`.
- Code coverage reporting via Codecov.
- Dependabot configuration for GitHub Actions and pip dependencies.

### Changed

- Repository restructured into `src/`, `ucodes/`, `tests/` (from a single-file
  `matRad_ompInterface.c` layout).
- CI build matrix extended to Windows (MSVC and MinGW), Linux x64/ARM64, and
  macOS x64/ARM64, with unit tests run via CTest on every push.

[Unreleased]: https://github.com/e0404/ompMC/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/e0404/ompMC/releases/tag/v0.2.0

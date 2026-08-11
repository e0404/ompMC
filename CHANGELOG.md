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
  the IAEA ones are. A single open cell is a rectangular field. By default it
  attenuates by weight and draws no random numbers at all, so putting a
  collimator in the beam leaves every history's random stream where it was;
  roulette is available instead, see below.
- `omc_source`, one interface every source of primary particles fills in. A
  source now answers only "which particle starts this history, and where is it
  going"; carrying it to the phantom, finding its voxel and counting the
  energy it brought are `omcSourcePlace()`'s job and the engine's, done once
  for everyone instead of once per source.
- A choice of how a collimator's transmission is paid for, as
  `struct OmcBeamModifier::apply` and `omcBeamModifierApply()`. The default,
  `OMC_MODIFIER_WEIGHT`, multiplies the particle's weight by the fraction and
  transports it regardless. `OMC_MODIFIER_ROULETTE` lets it through with that
  probability at full weight instead, which is where the time goes behind
  thick leaves: a 2% leaf costs a shower one history in fifty rather than
  every one of them, at the price of one random number and more noise per
  history. A cell that is fully open or fully shut is decided without
  drawing, so an all-or-nothing aperture -- a jaw, which is most of the use --
  leaves every random stream exactly where the weight mode does. Deciding how
  to spend the fraction is the engine's, in one place; `transmission()` stays
  a pure function that draws nothing whichever mode is in force.
- Phase spaces and collimators are reachable from both host interfaces, which
  is what makes them usable without writing C.

  In Python: `ompmc.PhaseSpaceSource` (the file, and the rotation and
  translation carrying it into the phantom's coordinate system),
  `ompmc.ApertureMask` with `ApertureMask.rectangle()` for the common single
  opening, `ompmc.RunSummary`, and `ompmc.calc_forward_phsp()`.
  `ompmc.calc_forward()` takes a `collimator` too.

  In MATLAB: `mcOpt.mode = 'forward_phsp'`, fed by `mcSrc.phaseSpace`
  (`file`, `order`, `first`, `rotation`, `translation`), and `mcSrc.collimator`
  (`z`, `x0`, `y0`, `dx`, `dy`, `transmission`, `outside`, `roulette`), which
  both forward modes accept.
- A third, optional output from both MATLAB forward modes: a struct of
  `nHistories`, `nStarted`, `nBlocked` and `energyFraction`. It matters most
  for a phase space, which is recorded wherever the original simulation
  scored it rather than aimed at your phantom, so most histories starting
  nothing is the normal case and these numbers are what tell it apart from a
  transform that is wrong.
- This changelog.
- Release packaging workflow (`release.yml`): on a `v*` tag, packages
  build.yml's binaries into per-platform zips (`omc_dosxyz` + the MATLAB MEX
  file per platform/toolchain, the Octave MEX file per platform/ABI bucket),
  each bundled with its data files and published to a GitHub Release.

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
- `omcCalcForward()` applies a modifier through `omcBeamModifierApply()`
  rather than multiplying the weight by `omcBeamModifierTransmission()`
  itself, which is what lets the choice between weight and roulette be the
  modifier's to declare. A modifier that returns a fraction that is not a
  number is now treated as having stopped the particle rather than being
  multiplied into its weight.
- The MATLAB interface accepts a third output argument. Mode `'dij'` refuses
  it -- a beamlet that started nothing comes back as a column of zeros, which
  says so already.
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

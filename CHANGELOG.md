# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

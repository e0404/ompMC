# Building ompMC

ompMC is built with [CMake](https://cmake.org/) (3.20 or newer). Two user codes
are produced:

| Target       | Kind                | Output                                          |
|--------------|---------------------|-------------------------------------------------|
| `omc_dosxyz` | command line binary | `build/bin/omc_dosxyz[.exe]`                     |
| `omc_matrad` | MATLAB MEX file     | `build/bin/omc_matrad.mexw64`, `.mexa64`, `.mexmaca64`, … |

Both link against the `ompmc_core` static library built from `src/`.

## Quick start

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

The configuration summary printed at the end of the configure step states
whether OpenMP and MATLAB were picked up.

## Options

| Option | Default | Meaning |
|---|---|---|
| `OMPMC_BUILD_DOSXYZ` | `ON` | Build the `omc_dosxyz` command line user code |
| `OMPMC_BUILD_MATRAD_MEX` | `AUTO` | Build the MEX file. `AUTO` skips it when no MATLAB is found, `ON` makes a missing MATLAB a hard error, `OFF` never builds it |
| `OMPMC_WITH_OPENMP` | `ON` | Multi threaded execution. Falls back to a serial build with a warning if no OpenMP runtime is available |
| `OMPMC_NATIVE_TUNING` | `OFF` | Add `-mtune=native`. Do not use for binaries you intend to distribute |

## Selecting the MATLAB installation

The MEX file is compiled by the CMake toolchain itself, against the MATLAB
headers in `extern/include` and the import libraries in `extern/lib`. MATLAB is
never started, so no license is required just to build.

CMake's `FindMatlab` module looks in the usual places (the Windows registry,
`/usr/local/MATLAB/R*`, `/Applications/MATLAB_R*.app`). If your MATLAB lives
somewhere else, or if several versions are installed, point the build at one:

```sh
cmake -S . -B build -DMatlab_ROOT_DIR="/path/to/MATLAB/R2024b"
```

`Matlab_ROOT_DIR` is the directory that contains `bin/`, `extern/` and
`toolbox/` — the value MATLAB reports as `matlabroot`.

## Platform notes

**Windows / MSVC.** MSVC ships no `<getopt.h>`, so `omc_dosxyz` is built with
the small replacement in [src/compat/](src/compat/). CMake detects this
automatically; on platforms with a POSIX `getopt_long()` the system header is
used unchanged. MSVC implements OpenMP 2.0, which is all ompMC uses. The
binaries need `vcomp140.dll` from the Visual C++ redistributable, which any
machine running MATLAB already has.

**Windows / MinGW-w64.** Works with the MinGW import libraries MATLAB ships in
`extern/lib/win64/mingw64`. The resulting binaries need `libgomp-1.dll`,
`libgcc_s_seh-1.dll` and `libwinpthread-1.dll` from your MinGW installation to
run outside of the build environment.

**macOS.** Apple's clang needs a separate OpenMP runtime:

```sh
brew install libomp
```

The build finds the Homebrew keg on its own. For a non-Homebrew libomp, pass
`-DOpenMP_ROOT=/path/to/libomp`. Binaries link against `libomp.dylib` by
absolute path, so libomp has to be present on the machine that runs them.

**Linux.** Nothing special; the OpenMP runtime comes with GCC.

## Running omc_dosxyz

`-i` takes the input file path *without* the `.inp` extension, and the paths
inside the input file are resolved relative to the current working directory:

```sh
./build/bin/omc_dosxyz -i ucodes/omc_dosxyz/smoke_test -o smoke_test
```

run from the repository root writes `output/smoke_test.3ddose`. This is the
short run the CI uses as a build smoke test; `ucodes/omc_dosxyz/input_file.inp`
is the full example, meant to be run from `ucodes/omc_dosxyz/`.

## Continuous integration

[.github/workflows/build.yml](.github/workflows/build.yml) builds and smoke
tests every push on Windows x64 (MSVC and MinGW), Linux x64, Linux ARM64, macOS
x64 and macOS ARM64, and uploads the binaries as workflow artifacts. The MEX
file is built everywhere except Linux ARM64, for which MathWorks publishes no
MATLAB release.

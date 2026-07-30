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

## OpenMP and MATLAB

MATLAB loads its own OpenMP runtime (`libiomp5`, plus `libmwompwrapper`) at
startup, so a MEX file compiled with `-fopenmp` or `/openmp` puts a *second*
OpenMP runtime into the process — `libgomp` for GCC/MinGW, `vcomp140` for MSVC,
`libomp` for clang. Measured on Windows with MATLAB R2025b, the two runtimes
coexist happily and parallel regions produce correct results with the full
thread count.

What is *not* safe is unloading the MEX file afterwards. Once a parallel region
has run, the OpenMP worker threads outlive the MEX file, and dropping the last
reference to the runtime — via `clear mex` or simply by quitting MATLAB — takes
it down while those threads are still alive. With `vcomp140` this crashes MATLAB
with an access violation, every time. `omc_matrad` therefore calls `mexLock()`
on entry. The consequence for development is that **a rebuilt MEX file is only
picked up after restarting MATLAB**.

[test_omc_matrad_mex.m](ucodes/omc_matrad/test_omc_matrad_mex.m) covers this. It
runs a small dose calculation from `test_fixture.mat` — inputs captured from
matRad's `matRad_PhotonOmpMCEngine` for a BOXPHANTOM photon plan, 25 beamlets on
a 48×48×48 dose grid — and then releases the MEX file. Removing the `mexLock()`
call makes that test take MATLAB down with an access violation, so the crash
cannot come back unnoticed. Run it locally with:

```matlab
addpath('build/bin'); addpath('ucodes/omc_matrad');
test_omc_matrad_mex
```

Only structural properties of the result are asserted (shape, sparsity, finite
non-negative dose, every beamlet scoring). ompMC seeds its RNG per thread, so
the numbers depend on the thread count and are not comparable across machines.

Do not try to remove the duplicate by linking the MEX file against MATLAB's own
`libiomp5`: MATLAB routes OpenMP through `libmwompwrapper`, and a MEX file
linked straight to `libiomp5` crashes inside `__kmp_launch_worker` on its first
parallel region.

On macOS, Homebrew's `libomp` and MATLAB's `libiomp5` are the same LLVM runtime,
and libomp's duplicate detection may abort with `OMP: Error #15`. Setting
`KMP_DUPLICATE_LIB_OK=TRUE` before starting MATLAB is the usual escape hatch.

Note also that MSVC implements OpenMP 2.0, which requires the loop variable of a
`#pragma omp parallel for` to be declared *outside* the `for` statement. The
current sources comply; `for (int i = 0; ...)` under an OpenMP pragma would
break the MSVC build with error C3015.

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

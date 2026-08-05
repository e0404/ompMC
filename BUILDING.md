# Building ompMC

ompMC is built with [CMake](https://cmake.org/) (3.20 or newer). Each user code
in [ucodes/](ucodes/) becomes one target:

| Target       | Kind                  | Output                                          |
|--------------|-----------------------|-------------------------------------------------|
| `omc_dosxyz` | command line binary   | `build/bin/omc_dosxyz[.exe]`                     |
| `omc_matrad` | MATLAB MEX file       | `build/bin/omc_matrad.mexw64`, `.mexa64`, `.mexmaca64`, … |
| `_ompmc`     | Python extension      | installed into the `ompmc` package, see [The Python extension](#the-python-extension) |

They all link against the `ompmc_core` static library built from `src/`.

`ompmc.c` calls four functions it does not define — `ausgab()` for scoring and
`howfar()`, `hownear()`, `regionIndex()` for the geometry. All four live in the
core library now, in [src/omc_score.c](src/omc_score.c) and
[src/omc_geom.c](src/omc_geom.c), so every user code transports through the same
rectilinear voxel phantom and only has to *fill* `struct Geom` from whatever it
reads: an `.egsphant` file, cubes handed over by MATLAB, or arrays from another
host. Shared code reports through `omcLog()`/`omcFail()`
([src/omc_host.h](src/omc_host.h)) rather than `printf()` or
`mexErrMsgIdAndTxt()`, and each host installs the sinks that give those meaning;
both are called on the master thread only, never from inside a parallel region.

The dose calculation itself is a library function too. There are three engines,
differing only in where the particles start and how the result comes back:

| Engine | Source | Result |
| --- | --- | --- |
| [src/omc_engine_dij.h](src/omc_engine_dij.h) | beamlet apertures at isocentre | one sparse column per beamlet, through a callback |
| [src/omc_engine_forward.h](src/omc_engine_forward.h) | the same beamlets, weighted | dense dose and uncertainty cubes |
| [src/omc_engine_cube.h](src/omc_engine_cube.h) | point source behind a collimator | dense dose and uncertainty cubes |

The two halves each engine is built from are shared rather than repeated, which
is what keeps them from drifting apart:
[src/omc_source_beamlet.h](src/omc_source_beamlet.h) starts a history on a
beamlet aperture, for the Dij and forward engines both, and `omcScoreToCube()`
in [src/omc_score.h](src/omc_score.h) turns accumulated energy into a dense cube
for the forward and cube engines both — the air threshold, the empty-voxel
convention and the batch variance therefore have one definition each.

All three take their energies from [src/omc_spectrum.h](src/omc_spectrum.h),
which turns a `.spectrum` file, a histogram handed over by the host, or a single
energy into the same sampling tables.

A user code is then only a translator: `omc_matrad.c` converts `mxArray`s into
those structs and appends the columns it gets back to a MATLAB sparse matrix,
`omc_dosxyz.c` reads an input file and writes a `.3ddose`,
[ucodes/omc_python/omc_python.cpp](ucodes/omc_python/omc_python.cpp) does the same for numpy arrays,
and nothing about any of those hosts reaches the engines.

## The Python extension

```sh
pip install .          # or: pip install -e . for a development install
```

scikit-build-core drives the same CMake project with `OMPMC_BUILD_PYTHON=ON`,
which is the only configuration that needs a C++ compiler — the binding is the
only C++ in the tree, so `enable_language(CXX)` sits inside that option rather
than in `project()`. nanobind must be importable by the interpreter being built
for; the build asks it for its CMake package directory.

On Windows build the extension with the same compiler CPython uses (MSVC).
Mixing toolchains between the MEX file and the extension is not wrong, but it
does change the last bits of a dose: a MinGW-built MEX file and an MSVC-built
extension agree on the sparsity pattern exactly and on the total dose to 3e-16,
while individual voxels differ by up to 2e-10 because their math libraries round
`log`/`exp` differently. Built with the same compiler they agree to 6e-16.

That difference is also why the test suite's `test_matches_mex`, which holds the
extension against a stored MEX result, is marked `mex`: it is a regression test
against one particular build, not a portability test. `pytest -m "not mex"`
skips it, which is what the wheel CI does.

### Wheels

`pip install .` compiles for the interpreter it is run with. Redistributable
wheels come out of [.github/workflows/wheels.yml](.github/workflows/wheels.yml),
which runs cibuildwheel with the `[tool.cibuildwheel]` configuration in
`pyproject.toml`; `pipx run cibuildwheel --platform <os>` reproduces it locally.

Two things about a wheel differ from a local build:

- **The OpenMP runtime travels with it.** auditwheel copies `libgomp` in on
  Linux and delocate copies Homebrew's `libomp.dylib` in on macOS. On Windows
  MSVC's `/openmp` links `vcomp140.dll`, which is not part of Windows but of
  the Visual C++ redistributable, so the repair step is told to vendor it
  explicitly (`delvewheel repair --add-dll vcomp140.dll`).
- **One wheel serves many Pythons.** `wheel.py-api = "cp312"` turns on
  nanobind's `STABLE_ABI`, so the 3.12 build is tagged `abi3` and loads under
  every later Python too. scikit-build-core signals this to CMake through
  `SKBUILD_SABI_COMPONENT`, which is why `find_package(Python ...)` interpolates
  that variable — without the `Development.SABIModule` component nanobind
  silently builds a version specific module instead. On 3.9 to 3.11, where the
  stable ABI is not usable, scikit-build-core ignores the setting and emits one
  wheel per version.

The macOS wheels are tagged for the macOS release of the runner that built them
(15.0 for x86-64, 14.0 for arm64) rather than for something older. The bundled
`libomp.dylib` is a Homebrew bottle built for that release, and dyld refuses to
load a library built for a newer system than the one running, so a lower tag
would promise support the wheel does not have. Older macOS installs build from
the source distribution.

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
| `OMPMC_BUILD_MATRAD_OCT` | `AUTO` | Build the same user code as a GNU Octave `.mex`. `AUTO` skips it when no Octave is found, `ON` makes a missing Octave a hard error, `OFF` never builds it. See [GNU Octave](#gnu-octave) |
| `OMPMC_WITH_OPENMP` | `ON` | Multi threaded execution. Falls back to a serial build with a warning if no OpenMP runtime is available |
| `OMPMC_WITH_OPENLIBM` | `OFF` | Fetch [openlibm](https://github.com/JuliaMath/openlibm) (MIT licensed) at configure time and resolve the `log`/`exp`/`sin`/`cos` calls from it, statically. Recommended for MinGW GCC, whose bundled software math routines are several times slower than the UCRT ones MSVC uses — the transport samples `-log(rng)` for every photon flight segment. Measured ~15% faster overall on MinGW; pointless with MSVC or glibc. Needs CMake 3.25+ and network access at configure time |
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

## GNU Octave

Octave implements the same MEX C API, so the exact same `omc_matrad.c` is built
a second time against Octave's headers and dropped next to the MATLAB one as
`omc_matrad.mex`. The two extensions do not collide — MATLAB only looks for
`.mexw64`/`.mexa64`/`.mexmaca64`, Octave only for `.mex` — so one `build/bin` can
serve both, and `test_omc_matrad_mex.m` runs unchanged under either.

Octave ships no CMake package, so [cmake/FindOctave.cmake](cmake/FindOctave.cmake)
locates it through `octave-config`. Anything on `PATH` is found automatically, as
are the usual Windows install locations; otherwise point the build at one:

```sh
cmake -S . -B build -DOctave_ROOT="/path/to/octave"
```

Two things changed in Octave 10, and both are detected rather than assumed:

* **The MEX entry points moved** out of `liboctinterp` into their own
  `liboctmex`. Octave 9 and older have no such library, so where a library has
  to be linked at all the build falls back to `liboctinterp`.
* **Octave 10 refuses to load a `.mex`** that does not say which `liboctmex` ABI
  it was built against, failing with *"No SOVERSION found in .mex file
  function"*. `mkoctfile` supplies that by generating a one-line stub, so the
  CMake build generates the same one from
  [ucodes/omc_matrad/omc_mex_soversion.c.in](ucodes/omc_matrad/omc_mex_soversion.c.in). `octave-config` does
  not report the number, so it is read out of the `liboctmex` library name.
  Releases predating the check get no stub.

A `.mex` is loaded into a process that already provides the MEX symbols, so on
Linux and macOS nothing is linked against it at all — exactly what `mkoctfile`
does. Windows PE cannot leave symbols undefined, so there the Octave libraries
really are on the link line (`liboctmex` from Octave 10, `liboctinterp` plus
`liboctave` before it), and **MSVC cannot build the Octave MEX file**: Octave
ships MinGW import libraries. Use MinGW GCC or clang on Windows; with
`OMPMC_BUILD_MATRAD_OCT=AUTO` the MSVC build just skips it.

The Windows build tolerates a C runtime mismatch: the Octave 10.3 installer is
built against the UCRT, while MinGW GCC usually targets `msvcrt.dll`, and both
runtimes then live in the process. Nothing crosses that boundary here — the MEX
file frees only what it allocated, and every `mxArray` goes through Octave's own
allocator — so the combination works. llvm-mingw's UCRT toolchain matches
Octave's runtime outright.

On Windows, do not put Octave's `bin` on `PATH` to be found. It is an MSYS2
tree that also carries `gcc`, `g++`, `ld`, `cmake` and `ctest`, so it will
shadow the toolchain you meant to build with. Pass `-DOctave_ROOT=` instead;
nothing else needs Octave on `PATH`, since the build and the test both refer to
it by absolute path.

Tested against Octave 6.4, 8.4, 10.3 and 11.3 on Windows and Octave 8.4 on
Linux — both sides of the Octave 10 changes above. All of them return the same
`dij` as the MATLAB MEX file, to the last digit.

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

Performance note: MinGW **GCC** binaries run about twice as slow as the other
Windows toolchains on this code — GCC emulates thread-local storage on Windows
with a function call per access, and the RNG and particle stack are thread
local. [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) (clang for the
same MinGW target, non-proprietary) uses native TLS and comes within ~20% of
MSVC. Unzip a release, then configure with
`-DCMAKE_C_COMPILER=<llvm-mingw>/bin/x86_64-w64-mingw32-clang.exe` and
`-DOMPMC_WITH_OPENLIBM=ON`; the binaries need `libomp.dll` from the toolchain's
`bin/` next to them. Either way, build with `OMPMC_WITH_OPENLIBM` — MinGW's
bundled math routines are several times slower than the UCRT ones.

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
OpenMP runtime into the process — `libgomp` for GCC/MinGW, `vcomp140` for MSVC.
Measured on Windows with MATLAB R2025b, those two coexist with MATLAB's
runtime — they share no symbol names — and parallel regions produce correct
results with the full thread count.

**Clang on Windows is the exception, in a good way**: clang emits the same
`__kmpc_*` calls that MATLAB's Intel runtime implements (LLVM's `libomp` is a
fork of it), and MATLAB ships the import library `bin/win64/libiomp5md.lib`
right next to the DLL. The build therefore links the MEX file against MATLAB's
own runtime automatically when compiling with clang — one OpenMP runtime in
the process, no `libomp.dll` to ship, no risk of Intel's duplicate-runtime
abort ("OMP: Error #15"). One newer runtime entry point clang emits that
MATLAB's runtime predates is provided as a documented no-op shim
([omc_kmp_compat.c](ucodes/omc_matrad/omc_kmp_compat.c)). Verified against
MATLAB R2025b. GCC cannot do the same: it emits `GOMP_*` calls and Intel's
Windows runtime has no GOMP compatibility layer.

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
non-negative dose, every beamlet scoring). The counter-based RNG makes results
independent of the thread count, but the exact numbers still differ across
platforms with the last-ulp rounding of the math library, so the test does not
pin them.

macOS needs its own arrangement, and it differs by architecture.

On **Apple Silicon**, MATLAB ships an LLVM OpenMP runtime of its own at
`MATLAB.app/bin/maca64/libomp.dylib` — the very runtime clang targets — so a MEX
file linked against Homebrew's `libomp.dylib` puts two copies of the *same*
runtime into the process. They export the same symbols, so calls cross between
them: a worker thread started by one ends up in the other's code operating on
thread state it does not own. The observed failure is `OMP: Error #179 Function
pthread_mutex_init failed` followed by a segmentation fault in
`__kmp_suspend_64`, with both dylibs visible in the stack trace.

The build therefore links **no** OpenMP runtime into the MEX file on macOS. Its
OpenMP symbols are left undefined (`-undefined dynamic_lookup`) and bind to
MATLAB's copy when the MEX file is loaded. Homebrew's libomp is still needed at
build time for `omp.h`, and `omc_dosxyz` — which runs in its own process, with
no MATLAB around — keeps linking it normally. A side benefit is that the MEX
file carries no absolute path into a particular MATLAB or Homebrew tree.

Note that `KMP_DUPLICATE_LIB_OK=TRUE`, the usual advice for duplicate OpenMP
runtimes, is not a fix here: it only silences the duplicate-runtime check, it
does not stop the two runtimes from calling into each other.

On **Intel macOS** that same arrangement does not work, because Intel MATLAB
brings no OpenMP runtime into the process for the MEX file to bind to:
`bin/maci64` contains neither `libomp.dylib` nor `libiomp5.dylib`, only the
`libmwompwrapper` shim, which defines none of the `__kmpc_*` entry points. A MEX
file built the Apple Silicon way fails to load outright, with an unresolved
`__kmpc_dispatch_deinit` — an entry point clang emits for `schedule(dynamic)`
and `schedule(guided)` loops.

There the MEX file gets a private runtime instead, linked from Homebrew's static
`libomp.a`. Nothing can collide with it: `matlab_add_mex` passes an
`-exported_symbols_list` that exports only `mexFunction`, so the runtime stays
invisible to the rest of the process, and its internal calls are resolved at
link time rather than through the flat namespace. `libomp.a` is C++ internally,
so `libc++` is linked alongside it.

The build picks between the two by asking the MATLAB installation which runtime
it ships, not by looking at the architecture, so it keeps working if MathWorks
changes what a release contains. The configuration summary reports the outcome
as `OpenMP runtime` under `omc_matrad`.

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

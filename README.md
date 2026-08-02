# ompMC

> The original repository is **[edoerner/ompMC](https://github.com/edoerner/ompMC)** by Edgardo Doerner.
> This repository is a fork under further development, aimed at integration into the
> treatment planning toolkits **[matRad](https://github.com/e0404/matRad)** (`e0404/matRad`)
> and **[pyRadPlan](https://github.com/e0404/pyRadPlan)** (`e0404/pyRadPlan`).

ompMC is an OpenMP-parallelized, CPU-based Monte Carlo code for coupled photon–electron
transport in voxelized geometries. Its physics is a C re-implementation of the EGSnrc
condensed-history transport algorithms, restricted to the interactions that matter for
megavoltage photon beams, and specialized for the one geometry a treatment planning
system needs: a rectilinear dose grid of user-defined materials and densities.

The point of the code is **beamlet-based Monte Carlo treatment planning**. Rather than
producing a single dose distribution, the `omc_matrad` user code transports histories for
many beamlets in one run and returns the dose-influence matrix (Dij) — one sparse column
per beamlet — that an optimizer needs for fluence-map optimization. Dose is scored per
beamlet and per batch, so a matching variance matrix comes out alongside it. Everything
runs on ordinary multi-core CPUs; no GPU, no cluster, no external EGSnrc installation.

## Citing ompMC

If you use this code, please cite the work it is based on:

- E. Doerner and P. Caprile,
  *Technical Note: Parallel implementation of the EGSnrc Monte Carlo simulation of ionizing
  radiation transport using OpenMP*,
  Medical Physics **44**(12), 6672–6677 (2017).
  [doi:10.1002/mp.12642](https://doi.org/10.1002/mp.12642)

- E. Doerner and P. Caprile,
  *Technical Note: An hybrid parallel implementation for EGSnrc Monte Carlo user codes*,
  Medical Physics **45**(8), 3969–3973 (2018).
  [doi:10.1002/mp.13033](https://doi.org/10.1002/mp.13033)

- E. Doerner, C. Rebolledo and V. Gomez,
  *Monte Carlo modelling of photon transport using Heterogeneous Computing*,
  Journal of Physics: Conference Series **1043**, 012062 (2018).
  [doi:10.1088/1742-6596/1043/1/012062](https://doi.org/10.1088/1742-6596/1043/1/012062)

## User codes

| Target       | Kind                | What it does |
|--------------|---------------------|--------------|
| `omc_dosxyz` | command line binary | DOSXYZnrc-style standalone dose calculation on an `.egsphant` phantom, driven by a plain-text input file. Writes a `.3ddose` file. |
| `omc_matrad` | MATLAB / Octave MEX file | Beamlet dose-influence matrix for matRad. Takes density and material cubes, geometry, source and option structs; returns sparse `dij` (and optionally its variance). The same source builds against MATLAB (`.mexw64`/`.mexa64`/…) and GNU Octave (`.mex`); see [BUILDING.md](BUILDING.md#gnu-octave). |

Both link against `ompmc_core`, the transport library built from [src/](src/):

- [src/ompmc.c](src/ompmc.c) — physics: media and PEGS4 data, photon and electron transport,
  Compton, Rayleigh, pair/triplet, photoelectric, Møller, Bhabha, bremsstrahlung, annihilation,
  multiple scattering
- [src/omc_random.c](src/omc_random.c) — random number generation
- [src/omc_score.c](src/omc_score.c) — dose and variance scoring
- [src/omc_utilities.c](src/omc_utilities.c) — input-file parsing and small helpers

## Building

See **[BUILDING.md](BUILDING.md)** for the full story — CMake options, how the MATLAB
installation is located, and the platform-specific handling of the OpenMP runtime inside a
MEX file (which is genuinely fiddly on macOS).

The short version:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

CMake 3.20+ is required. OpenMP and MATLAB are both optional: without OpenMP the build falls
back to serial execution with a warning, and without MATLAB the MEX target is simply skipped.
The configure step prints a summary of what was found.

## Running `omc_dosxyz`

`-i` takes the input file path *without* the `.inp` extension; paths inside the input file are
resolved relative to the current working directory. From the repository root:

```sh
./build/bin/omc_dosxyz -i ucodes/omc_dosxyz/smoke_test -o smoke_test
```

writes `output/smoke_test.3ddose`. [ucodes/omc_dosxyz/input_file.inp](ucodes/omc_dosxyz/input_file.inp)
is the fuller example, meant to be run from `ucodes/omc_dosxyz/`.

Input files are grouped into sections of `key = value` lines:

```
# start source definition
mono energy = 20.0
spectrum file = ./../../spectra/mohan6.spectrum
charge = 0
collimator bounds = -2.5 2.5 -2.5 2.5
ssd = 90.0
# end source definition

# start MC control
ncase  = 100000
nbatch = 10
rng seeds = 97 33
# stop MC control

# start geometry
method of input = phantom
phantom file = ./../../phantoms/WATER.egsphant
# stop geometry

# start MC transport
global ecut = 0.521
global pcut = 0.010
pegs file = ./../../pegs4/521icru.pegs4dat
pgs4form file = ./../../pegs4/pgs4form.dat
# stop MC transport

# start VRT
nsplit = 20
# stop VRT

# start ompMC environment
data folder = ./../../data/
output folder = ./../../output/
# stop ompMC environment
```

## Using `omc_matrad` from MATLAB

```matlab
addpath('build/bin');
[dij, dijVar] = omc_matrad(cubeRho, cubeMatIx, mcGeo, mcSrc, mcOpt);
```

- `cubeRho` — 3D `double` cube of mass densities
- `cubeMatIx` — 3D `int32` cube of material indices into `mcGeo.material`
- `mcGeo` — dose grid: `material`, `xBounds`, `yBounds`, `zBounds`
- `mcSrc` — beamlet source: `nBixels`, `iBeam`, source position and, per beamlet, the corner
  and two edge vectors of its aperture
- `mcOpt` — run settings: `nHistories`, `nBatches`, `nSplit`, `charge`, `global_ecut`,
  `global_pcut`, `randomSeeds`, `pegsFile`, `pgs4formFile`, `dataFolder`, `outputFolder`, and
  optionally `spectrum`, `spectrumFile`, `monoEnergy`, `sourceGeometry` (`'point'` or
  `'gaussian'`), `sourceGaussianWidth`, `relDoseThreshold`, `verbose`, `progressCallback`, and
  the variance-reduction keys below

`charge` picks the source particle: `-1` for electrons, `0` for photons, `+1` for positrons.

The source spectrum can either be read from a `.spectrum` file (`spectrumFile`, default
`./spectra/mohan6.spectrum`) or passed in directly as `mcOpt.spectrum`, a struct holding the
same information:

| Field | Meaning |
| --- | --- |
| `energy` | upper energy of each bin in MeV, strictly ascending vector |
| `fluence` | relative number of particles per bin, same length, non-negative |
| `eMin` | lower energy of the first bin in MeV, optional, default `0` |
| `mode` | `0` for counts per bin (default), `1` for counts per MeV |

```matlab
mcOpt.spectrum = struct('energy', [1; 2; 3], 'fluence', [0.2; 0.5; 0.3]);
```

Within a bin the energy is sampled uniformly, as it is for a spectrum read from file.

`monoEnergy` is the third way: a single kinetic energy in MeV, used for every source particle.

The three are tried in order — `spectrum`, then `spectrumFile`, then `monoEnergy` — and whichever
loses is announced rather than silently dropped. Giving none of them uses `spectra/mohan6.spectrum`.

`progressCallback`, if given, is a function handle called with a single scalar in `[0,1]` once
per batch and once per finished beamlet; it replaces the built-in `waitbar` and owns any
handle/window lifecycle itself, e.g. `mcOpt.progressCallback = @(p) waitbar(p, h, msg);`. Without
it, a `waitbar` is shown automatically when `verbose >= 2`.

Both outputs are sparse, with one column per beamlet and one row per dose-grid voxel; the
second output is only computed if requested. Entries below `relDoseThreshold` (relative to the
beamlet maximum) are dropped.

**The MEX file calls `mexLock()` on entry** and cannot be unloaded — unloading an OpenMP-using
MEX file after a parallel region has run crashes MATLAB. In practice this means a rebuilt MEX
file is only picked up after restarting MATLAB. [BUILDING.md](BUILDING.md) explains why in
detail.

## Using ompMC from Python

```sh
pip install .
```

which compiles the extension for the interpreter it is run with; a C++ compiler and a working
OpenMP runtime are all it needs. Prebuilt wheels for Linux, macOS and Windows come out of the
`wheels` workflow and carry their own OpenMP runtime, so they need neither.

The wheel bundles the cross section data, PEGS files and spectra, so nothing has to be pointed at
the source tree. Two calculations are available, sharing the same phantom, physics and spectra:

```python
import numpy as np, ompmc

n = 32
lateral, depth = np.linspace(-8.0, 8.0, n + 1), np.linspace(0.0, 16.0, n + 1)
geometry = ompmc.Geometry(
    lateral, lateral, depth, ["H2O521ICRU"],
    density=np.full((n, n, n), 1.0, order="F"),
    material=np.ones((n, n, n), dtype=np.int32, order="F"),
)

# One dense dose cube from a collimated beam
dose, uncertainty = ompmc.calc_cube(
    geometry,
    ompmc.CollimatedSource(ssd=100.0, x_min=-2, x_max=2, y_min=-2, y_max=2),
    ompmc.Spectrum.monoenergetic(6.0),
    n_histories=100_000, n_batches=10,
)

# ... or one sparse column per beamlet, as scipy.sparse.csc_array
dij = ompmc.calc_dij(geometry, beamlet_source, ompmc.Spectrum.default(),
                     n_histories=100_000, progress=lambda p: print(f"{p:.0%}"))
```

- **Cubes must be Fortran ordered.** The transport indexes voxels with the first axis varying
  fastest, so a C ordered cube would be a silently transposed phantom; it is rejected instead.
- Material indices count from 1, matching matRad's `cubeMatIx`; 0 means vacuum.
- `progress` is called with the fraction finished; returning `False` stops the run, as does Ctrl-C.
- The GIL is released for the whole calculation, so the OpenMP threads run at full speed. The
  engines keep their state in globals, so one calculation runs at a time per process: use
  `multiprocessing`, not threads.
- `ompmc.Physics(...)` carries the cut-offs, seeds, splitting factor and the variance-reduction
  keys below.

## Variance reduction

| Key (input file / `mcOpt`) | Effect |
|---|---|
| `nsplit` / `nSplit` | Uniform photon splitting at the source. `> 1` enables it. |
| `esave` | Electron range rejection: electrons whose residual CSDA range cannot carry them out of the current voxel are terminated below this total energy (MeV). `0` or absent disables it. |
| `e_rr`, `f_rr` | Unbiased Russian roulette of newly created electrons below total energy `e_rr` (MeV), with survival probability `1/f_rr`. Both must be set (`f_rr > 1`) to take effect. |

Photon transport uses Woodcock (delta) tracking, so photon steps are not stopped at voxel
boundaries.

## Data files

| Directory | Contents |
|---|---|
| [data/](data/) | XCOM photon cross sections, multiple-scattering and spin-effect data |
| [pegs4/](pegs4/) | PEGS4 material data (`521icru`, `700icru`) and the `pgs4form` bremsstrahlung form factors |
| [phantoms/](phantoms/) | Example `.egsphant` phantoms: `WATER`, `TG119`, `PROSTATE` |
| [spectra/](spectra/) | Example photon spectra: `mohan6`, `var_6MV`, `250` |

## Tests

Unit tests are built by default (`OMPMC_BUILD_TESTS=ON`) and registered with CTest:

```sh
ctest --test-dir build --output-on-failure
```

This covers the transport helpers and media data ([tests/](tests/)) plus a short `omc_dosxyz`
smoke run. When the Octave MEX file was built, `ctest` also drives it through the MEX-side test
below. The same test runs unchanged in MATLAB, which needs a MATLAB session:

```matlab
addpath('build/bin'); addpath('ucodes/omc_matrad');
test_omc_matrad_mex
```

[.github/workflows/build.yml](.github/workflows/build.yml) builds and smoke tests every push on
Windows x64 (MSVC and MinGW), Linux x64, Linux ARM64, macOS x64 and macOS ARM64.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
Copyright (C) 2018 Edgardo Doerner (edoerner@fis.puc.cl).

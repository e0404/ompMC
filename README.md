# ompMC

[![Build](https://github.com/e0404/ompMC/actions/workflows/build.yml/badge.svg)](https://github.com/e0404/ompMC/actions/workflows/build.yml)
[![Docs](https://github.com/e0404/ompMC/actions/workflows/docs.yml/badge.svg)](https://github.com/e0404/ompMC/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/e0404/ompMC/branch/master/graph/badge.svg)](https://codecov.io/gh/e0404/ompMC)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

[![C](https://img.shields.io/badge/C-A8B9CC?logo=c&logoColor=white)](src/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![MATLAB](https://img.shields.io/badge/MATLAB-0076A8?logo=mathworks&logoColor=white)](ucodes/omc_matrad/)

[![Windows](https://img.shields.io/badge/Windows-0078D6?logo=windows&logoColor=white)](.github/workflows/build.yml)
[![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)](.github/workflows/build.yml)
[![macOS](https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=white)](.github/workflows/build.yml)

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
| `omc_dosrz` | command line binary | DOSRZnrc-style dose in a homogeneous cylinder, scored by radial ring and depth slab — the shape a pencil-beam dose distribution wants. Takes a parallel pencil beam, a point source at an SSD, or an IAEA phase-space file. Writes a `.rzdose` file. |
| `omc_matrad` | MATLAB / Octave MEX file | Dose for matRad. Takes density and material cubes, geometry, source and option structs, and returns either a sparse beamlet dose-influence matrix `dij` or, with `mcOpt.mode = 'forward_beamlet'` or `'forward_phsp'`, a dense dose cube — of one weighted field, or of the particles of an IAEA phase-space file. The same source builds against MATLAB (`.mexw64`/`.mexa64`/…) and GNU Octave (`.mex`); see [BUILDING.md](BUILDING.md#gnu-octave). |

All link against `ompmc_core`, the transport library built from [src/](src/):

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

## Running `omc_dosrz`

Same command line, same input-file syntax, a different phantom: one homogeneous cylinder
about the beam axis, scored into radial rings and depth slabs rather than voxels. From the
repository root:

```sh
./build/bin/omc_dosrz -i ucodes/omc_dosrz/smoke_test -o smoke_rz
```

writes `output/smoke_rz.rzdose`. [ucodes/omc_dosrz/input_file.inp](ucodes/omc_dosrz/input_file.inp)
is the fuller example, meant to be run from `ucodes/omc_dosrz/`.

The cylinder is described in the input file — there is no phantom file for it — and the
source is one of three:

```
# start source definition
# 'pencil' : a parallel beam of no width on the axis
# 'point'  : a point source at 'ssd', illuminating a disc of 'field radius'
# 'phsp'   : an IAEA phase space file
source type = pencil
mono energy = 6.0
charge = 0
# Optional: widen either delta into a Gaussian. 'spot sigma' (cm) is the beam
# width where it meets the front face, 'divergence sigma' (rad) its angular
# spread. Left out, both are 0 -- the delta itself, drawing no random numbers.
# spot sigma = 0.15
# divergence sigma = 0.01
# stop source definition

# start geometry
medium = H2O521ICRU
medium density = 1.0
cylinder radius = 5.0
radial bins = 10
cylinder depth = 10.0
depth bins = 20
# stop geometry
```

Either axis can be given its boundaries in full instead — `radial bin edges` and
`depth bin edges`, ascending, the radial list starting at 0 — which is how to put fine rings
on the beam and coarse ones out where the dose has gone.

The `.rzdose` file is the `.3ddose` layout with the axis it does not have removed: the ring
and slab counts, the ring boundaries, the depth boundaries, then the dose and its relative
uncertainty with the ring running fastest. Dose is in Gy **per incident history** — not the
dose per unit fluence `omc_dosxyz` reports, there being no field for a pencil beam to have a
fluence over.

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

### `mcOpt.mode` — dose-influence matrix or forward dose

| `mcOpt.mode` | Returns |
| --- | --- |
| `'dij'` (default) | `[dij, dijVar]` — one sparse column per beamlet |
| `'forward_beamlet'` | `[dose, relUnc, summary]` — one dense cube, the size of `cubeRho` |
| `'forward_phsp'` | the same, from an IAEA phase-space file instead of beamlets |

`'forward_beamlet'` computes the dose of a whole weighted field in one go. The collimation is
given as one weight per beamlet in `mcSrc.bixelWeights`, a non-negative vector of length
`nBixels`: a blocked beamlet gets `0`, an open one its fluence, a partly transmitting one a
fraction of it. This is the fluence map matRad already optimises, so no new geometry is needed.

```matlab
mcOpt.mode = 'forward_beamlet';
mcSrc.bixelWeights = w;                 % from matRad_fluenceOptimization
[dose, relUnc] = omc_matrad(cubeRho, cubeMatIx, mcGeo, mcSrc, mcOpt);
```

The result is what `dij*w` would have been, in Gy for exactly those weights — doubling every
weight doubles the dose — but it is reached directly instead of through the matrix. Histories go
to the beamlets in proportion to their weight, so a blocked beamlet costs nothing and the run
time no longer grows with `nBixels`.

Two things change meaning in this mode:

- **`nHistories` counts the whole calculation**, not one beamlet. Switching a `dij` run over
  unchanged therefore divides the statistics by `nBixels`; multiply it by `nBixels` to keep them.
- **`relDoseThreshold` does nothing.** It prunes columns of a sparse matrix, and there is no
  matrix here. Note the flip side when comparing the two modes: it is the `dij` result that is
  pruned, so set it to `0` for a like-for-like comparison.

`relUnc` is the relative uncertainty per voxel, `0.9999999` where nothing was deposited — the
convention `omc_dosxyz` writes into a `.3ddose` file. `mcOpt.outputDose = 0` asks for mean
deposited energy instead of Gy.

The weights modulate **fluence, not spectrum**: a leaf transmitting 2% starts 2% of the
particles, with the spectrum unhardened. Attenuation in the collimator, its scatter and the beam
hardening that goes with it are not modelled. The mode is named for its source model rather than
its output, which is what lets `'forward_phsp'` sit next to it.

The third output, `summary`, describes what became of the histories: `nHistories`, `nStarted`
(how many put a particle into the phantom), `nBlocked` (how many the collimator stopped) and
`energyFraction`. Mode `'dij'` has no equivalent and refuses it — a beamlet that started nothing
comes back as a column of zeros, which says so already.

### `mcOpt.mode = 'forward_phsp'` — starting from a phase space

A phase-space file records everything that crossed a plane in an earlier simulation of a
treatment head; the sets published at <https://www-nds.iaea.org/phsp/> are the output of full
models of real linacs. Starting histories from those particles is the difference between
modelling the beam and describing it. There is no spectrum and there are no beamlets: the file
carries the energy, position and direction of every particle it holds.

```matlab
mcOpt.mode = 'forward_phsp';
mcSrc.phaseSpace = struct('file', 'Varian_TrueBeam6MV_01');
[dose, relUnc, summary] = omc_matrad(cubeRho, cubeMatIx, mcGeo, mcSrc, mcOpt);
```

| `mcSrc.phaseSpace` | Meaning |
| --- | --- |
| `file` | base name of the `.IAEAheader`/`.IAEAphsp` pair, with or without either extension |
| `order` | `'replay'` (default) walks the file in order and draws no random numbers; `'random'` picks a particle per history, at one random number each |
| `first` | particle the replay starts at, default `0` |
| `rotation` | `3x3` rotation carrying the phase space into the phantom's coordinate system, default `eye(3)` |
| `translation` | `1x3` offsets in cm, applied after `rotation`, default zeros |

The whole file goes into memory, so it costs about its size on disk — gigabytes for a published
dataset. And it was recorded wherever the original simulation scored it, not aimed at your
phantom, so **it is normal for most histories to start nothing**; `summary.nStarted` is what
tells that apart from a transform that is wrong.

**One particle per history.** A phase space records which particles a single original history
left behind, and those are correlated. Drawing them one at a time still gets the dose right on
average, but the uncertainty a run reports comes out smaller than the truth by however much they
are correlated.

### `mcSrc.collimator` — something in the beam's way

Optional in either forward mode. A transmission grid on a plane, applied by back projection from
wherever the source put the particle — so it composes with either source and does not care which
side of the plane the particle started on. That is what lets a field be cut out of a phase space
recorded *above* the jaws, as the published ones are.

```matlab
% A 10 x 10 cm field at the 100 cm isocentre, from a jaw at 40 cm
mcSrc.collimator = struct('z', 40, 'x0', -2, 'y0', -2, 'dx', 4, 'dy', 4, ...
                          'transmission', 1);
```

| Field | Meaning |
| --- | --- |
| `z` | the plane the mask sits on, in cm |
| `x0`, `y0` | lower corner of the grid, in cm |
| `dx`, `dy` | cell size in cm, both positive |
| `transmission` | matrix of fractions in `[0,1]`, one per cell, x down the rows; a scalar `1` is a rectangular aperture |
| `outside` | what gets through beside the grid, default `0` — what a field stop does |
| `roulette` | see below, default `false` |

With beamlets this is usually unnecessary, the collimation already being in `bixelWeights`; it
earns its place where the weights cannot say what is wanted, such as a block cutting across
beamlets or a leaf that transmits.

`roulette` chooses how a partly transmitting cell is paid for, and both give the same dose in
the mean. `false` multiplies the particle's weight by the fraction and transports it regardless,
so a 2% leaf costs a full shower for a fiftieth of the dose — but **draws no random numbers at
all**, leaving every history's random stream where it would have been with the beam open, so a
collimated run stays comparable history by history with the open one it came from. `true` lets
the particle through with that probability at full weight instead, spending the time where the
dose is, at the price of one random number and more noise per history. Cells that are fully open
or fully shut are decided without drawing either way, so an all-or-nothing aperture behaves
identically under both.

This is a **mask, not a collimator**: it attenuates and blocks, but does not scatter and does not
harden the spectrum of what it lets through — good for the fluence, poor for the penumbra.

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
the source tree. Four calculations are available, sharing the same phantom and physics:

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

# ... or the dense cube of a whole weighted field, which is dij @ weights
# computed directly. A blocked beamlet weighs 0 and costs nothing.
dose, uncertainty = ompmc.calc_forward(
    geometry, beamlet_source, weights, ompmc.Spectrum.default(),
    n_histories=100_000,
)

# ... or the same from a linac's own particles, with a 10 x 10 cm field at
# 100 cm cut out of them by a jaw at 40 cm. No spectrum: the file carries one.
dose, uncertainty, summary = ompmc.calc_forward_phsp(
    geometry,
    ompmc.PhaseSpaceSource("Varian_TrueBeam6MV_01"),
    n_histories=1_000_000,
    collimator=ompmc.ApertureMask.rectangle(40.0, -2.0, 2.0, -2.0, 2.0),
)
```

and the r-z dose of a pencil beam, which takes a cylinder rather than a voxel phantom:

```python
cylinder = ompmc.CylinderGeometry(
    r_bounds=np.linspace(0.0, 5.0, 21),
    z_bounds=np.linspace(0.0, 20.0, 41),
    material="H2O700ICRU",
    density=1.0,
)

dose, uncertainty, summary = ompmc.calc_radial(
    cylinder, ompmc.PencilBeamSource(), n_histories=1_000_000)

depth_dose_on_axis = dose[0, :]
```

- **Cubes must be Fortran ordered.** The transport indexes voxels with the first axis varying
  fastest, so a C ordered cube would be a silently transposed phantom; it is rejected instead.
- Material indices count from 1, matching matRad's `cubeMatIx`; 0 means vacuum.
- `progress` is called with the fraction finished; returning `False` stops the run, as does Ctrl-C.
- `calc_forward` is the Python side of `mcOpt.mode = 'forward_beamlet'` above, with the same two
  caveats: `n_histories` counts the whole calculation rather than one beamlet, and the weights
  modulate fluence rather than spectrum. It takes a `collimator=` too.
- `calc_forward_phsp` is the Python side of `mcOpt.mode = 'forward_phsp'`, and carries the same
  warnings: the whole file goes into memory, one particle starts each history so the reported
  uncertainty is optimistic, and most histories starting nothing is normal — the file was
  recorded wherever the original simulation scored it, not aimed at your phantom. It returns a
  third value, a `RunSummary` of `n_histories`, `n_started`, `n_blocked` and `energy_fraction`,
  which is what tells that apart from a transform that is wrong.
- `calc_radial` is the Python side of `omc_dosrz`, and the only one that does not take a
  `Geometry`: rings and depth slabs are its regions, and `dose` comes back shaped
  `(n_rings, n_slabs)`. It takes a `PencilBeamSource` — parallel, or a point source with
  `ssd=` — or a `PhaseSpaceSource`. Its dose is per incident history rather than per unit
  fluence; there is no field for a pencil beam to have a fluence over.
- `PencilBeamSource(spot_sigma=..., divergence_sigma=...)` widens the beam from a delta into
  a Gaussian in position, in angle, or both — drawn independently, so it is a blurred pencil
  rather than a beam with emittance. Either left out draws no random numbers, so a plain
  pencil is unaffected by their existence.
- `ompmc.ApertureMask` is something in the beam's way, applied by back projection so it composes
  with either source. `roulette=True` spends a partly transmitting cell as a survival probability
  at full weight rather than as a weight multiplier — cheaper behind thick leaves, noisier, and
  it draws a random number where the default draws none.
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

This covers the transport helpers, the two geometries and media data ([tests/](tests/)) plus
short `omc_dosxyz` and `omc_dosrz` smoke runs. When the Octave MEX file was built, `ctest` also
drives it through the MEX-side test below. The same test runs unchanged in MATLAB, which needs a MATLAB session:

```matlab
addpath('build/bin'); addpath('ucodes/omc_matrad');
test_omc_matrad_mex
```

[.github/workflows/build.yml](.github/workflows/build.yml) builds and smoke tests every push on
Windows x64 (MSVC and MinGW), Linux x64, Linux ARM64, macOS x64 and macOS ARM64.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a history of changes.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
Copyright (C) 2018-2026 Edgardo Doerner and Niklas Wahl.

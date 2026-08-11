# ompMC

ompMC is an OpenMP-parallelized, CPU-based Monte Carlo code for coupled
photon-electron transport in voxelized geometries, built for beamlet-based
treatment planning: `omc_matrad` transports histories for many beamlets in one
run and returns the dose-influence matrix (Dij) an optimizer needs for
fluence-map optimization, and the Python interface gives the same
calculations to any other host.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Getting started
:link: getting-started/installation
:link-type: doc
Build ompMC and its Python extension.
:::

:::{grid-item-card} {octicon}`package` Python API
:link: python-api/index
:link-type: doc
`Geometry`, `Spectrum`, `calc_dij`, `calc_cube`, `calc_forward`,
`calc_forward_phsp`.
:::

:::{grid-item-card} {octicon}`file-code` C API
:link: c-api/index
:link-type: doc
The engine headers a new host embeds ompMC through.
:::

:::{grid-item-card} {octicon}`terminal` MATLAB / Octave
:link: matlab-api/index
:link-type: doc
The `omc_matrad` MEX interface for matRad.
:::

:::{grid-item-card} {octicon}`mark-github` Source
:link: https://github.com/e0404/ompMC
The repository, issue tracker and full README.
:::
::::

```{toctree}
:hidden:
:caption: Getting started

getting-started/installation
```

```{toctree}
:hidden:
:caption: Python

python-api/index
```

```{toctree}
:hidden:
:caption: C

c-api/index
```

```{toctree}
:hidden:
:caption: MATLAB / Octave

matlab-api/index
```

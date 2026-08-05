# C API

`ompmc_core` is a C99 static library shared by every user code -- the
`omc_dosxyz` command line binary, the `omc_matrad` MATLAB/Octave MEX file, and
the `_ompmc` Python extension. A new host reaches it through ten headers,
grouped below the way {doc}`../getting-started/installation` introduces them.

The transport physics itself (`ompmc.c`/`ompmc.h`, roughly 6,000 lines of
photon and electron interaction code) is not part of this reference: it has no
public entry points a host calls directly, and is meant to be read as source
rather than browsed as an API.

```{toctree}
:maxdepth: 2

engines
geometry-and-sources
infrastructure
```

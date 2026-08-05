# Installation

The Python package installs the ordinary way:

```sh
pip install ompmc
```

which compiles the extension for the interpreter it is run with; a C++
compiler and a working OpenMP runtime are all it needs. Prebuilt wheels for
Linux, macOS and Windows carry their own OpenMP runtime, so they need
neither.

Building the C library, `omc_dosxyz` and the MATLAB/Octave MEX file uses
CMake directly. What follows is the build guide from the repository root,
included here so it has one copy. A few of its relative links (to `src/`,
`LICENSE`, and similar files outside `docs/`) resolve on GitHub but not from
this site; the [repository on GitHub](https://github.com/e0404/ompMC) is the
canonical place to browse those.
so it has one copy. A few of its relative links (to `src/`, `LICENSE`, and
similar files outside `docs/`) resolve on GitHub but not from this site; the
[repository on GitHub](https://github.com/e0404/ompMC) is the canonical place
to browse those.

```{include} ../../BUILDING.md
:relative-docs: docs/
:relative-images:
:start-line: 1
```

# MATLAB / Octave interface

`omc_matrad` is a MATLAB/Octave MEX file built from
[ucodes/omc_matrad/omc_matrad.c](https://github.com/e0404/ompMC/blob/master/ucodes/omc_matrad/omc_matrad.c) --
see {doc}`../getting-started/installation` for how it's built, and note the
`mexLock()` caveat there (a rebuilt MEX file needs a MATLAB restart).

A MEX file carries no MATLAB source for autodoc to read, so the reference
below comes from a same-named `.m` file holding only a help comment --
the usual way to document a MEX file. It deliberately is **not** shipped
next to the real `omc_matrad` MEX file, though: `addpath` prepends by
default, and a same-named `.m` file anywhere later on the path than
`build/bin` would shadow the compiled MEX file, so `help omc_matrad` in an
actual MATLAB session prints only its default one-liner, not this page.

```matlab
addpath('build/bin');
[dij, dijVar] = omc_matrad(cubeRho, cubeMatIx, mcGeo, mcSrc, mcOpt);
```

```{eval-rst}
.. mat:module:: .
```

## Reference

```{eval-rst}
.. mat:autofunction:: omc_matrad
```

## Progress callback

`mcOpt.progressCallback`, if given, replaces the built-in `waitbar`. It must
be a function handle taking a single scalar in `[0, 1]`;
[recordProgressCallback.m](https://github.com/e0404/ompMC/blob/master/ucodes/omc_matrad/recordProgressCallback.m)
is what the test suite uses to record progress into a variable instead of a
window:

```{literalinclude} ../../ucodes/omc_matrad/recordProgressCallback.m
:language: matlab
```

# MATLAB / Octave interface

`omc_matrad` is a MATLAB/Octave MEX file built from
[ucodes/omc_matrad/omc_matrad.c](https://github.com/e0404/ompMC/blob/master/ucodes/omc_matrad/omc_matrad.c) --
see {doc}`../getting-started/installation` for how it's built, and note the
`mexLock()` caveat there (a rebuilt MEX file needs a MATLAB restart).

A MEX file carries no MATLAB source for `help`/`doc` to read, so it ships
with a same-named `.m` file containing only a help comment -- MATLAB always
executes the MEX file and only reads documentation from the `.m` file. That
stub is what the reference below is generated from, so it is also what
`help omc_matrad` prints in MATLAB itself.

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
be a function handle taking a single scalar in `[0, 1]`; the example below is
what the test suite uses to record progress into a variable instead of a
window.

```{eval-rst}
.. mat:autofunction:: recordProgressCallback
```

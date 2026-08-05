# Dose engines

The three ways to run a calculation, differing only in where particles start
and how the result comes back. All three share the phantom
({doc}`geometry-and-sources`), the spectrum and the transport itself, and are
singletons: one calculation at a time per process.

## Dose-influence matrix (Dij)

```{doxygenfile} omc_engine_dij.h
```

## Forward dose from weighted beamlets

```{doxygenfile} omc_engine_forward.h
```

## Dose cube from a collimated beam

```{doxygenfile} omc_engine_cube.h
```

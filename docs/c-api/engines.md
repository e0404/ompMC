# Dose engines

The four ways to run a calculation, differing only in where particles start,
what shape the phantom is and how the result comes back. All of them share the
phantom ({doc}`geometry-and-sources`), the spectrum and the transport itself,
and are singletons: one calculation at a time per process.

## Dose-influence matrix (Dij)

```{doxygenfile} omc_engine_dij.h
```

## Forward dose from weighted beamlets

```{doxygenfile} omc_engine_forward.h
```

## Dose cube from a collimated beam

```{doxygenfile} omc_engine_cube.h
```

## Radial (r-z) dose in a cylinder

```{doxygenfile} omc_engine_radial.h
```

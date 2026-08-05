# Python API

The `ompmc` package wraps the compiled `_ompmc` extension: dataclasses
describe the phantom, source and physics, and three functions run a
calculation against them, sharing the same phantom, physics and source
spectra.

```{list-table}
:header-rows: 1

* - Function
  - Result
* - {py:func}`ompmc.calc_dij`
  - One sparse column of dose per beamlet -- the dose-influence matrix a
    treatment planning system optimizes against.
* - {py:func}`ompmc.calc_forward`
  - Dose everywhere in the phantom from a whole weighted set of beamlets, in
    one go -- what `calc_dij(...) @ weights` would give.
* - {py:func}`ompmc.calc_cube`
  - Dose everywhere in the phantom from a single collimated beam.
```

## Quickstart

```{literalinclude} ../../README.md
:language: python
:start-after: "```python"
:end-before: "```"
:dedent:
```

```{note}
Cubes must be Fortran ordered (see {py:class}`~ompmc.Geometry` below): the
transport indexes voxels with the first axis varying fastest, so a C-ordered
cube would describe a transposed phantom -- it is rejected rather than
silently copied.
```

## Phantom, sources and physics

```{eval-rst}
.. autoclass:: ompmc.Geometry
   :members:

.. autoclass:: ompmc.Spectrum
   :members:

.. autoclass:: ompmc.BeamletSource
   :members:

.. autoclass:: ompmc.CollimatedSource
   :members:

.. autoclass:: ompmc.Physics
   :members:
```

## Calculations

```{eval-rst}
.. autofunction:: ompmc.calc_dij

.. autofunction:: ompmc.calc_forward

.. autofunction:: ompmc.calc_cube
```

## Utilities

```{eval-rst}
.. autofunction:: ompmc.data_path

.. data:: ompmc.MAX_MEDIA

   Maximum number of distinct media a :class:`Geometry` may reference.
```

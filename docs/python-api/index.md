# Python API

The `ompmc` package wraps the compiled `_ompmc` extension: dataclasses
describe the phantom, source and physics, and five functions run a
calculation against them, sharing the same physics.

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
* - {py:func}`ompmc.calc_forward_phsp`
  - The same, from the particles of an IAEA phase space file rather than from
    a spectrum through an aperture.
* - {py:func}`ompmc.calc_cube`
  - Dose everywhere in the phantom from a single collimated beam.
* - {py:func}`ompmc.calc_radial`
  - Dose in a cylinder, binned into rings and depth slabs rather than voxels
    -- what a pencil beam distribution wants. Takes a
    {py:class}`~ompmc.CylinderGeometry` rather than a
    {py:class}`~ompmc.Geometry`.
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

.. autoclass:: ompmc.CylinderGeometry
   :members:

.. autoclass:: ompmc.Spectrum
   :members:

.. autoclass:: ompmc.BeamletSource
   :members:

.. autoclass:: ompmc.CollimatedSource
   :members:

.. autoclass:: ompmc.PencilBeamSource
   :members:

.. autoclass:: ompmc.PhaseSpaceSource
   :members:

.. autoclass:: ompmc.Physics
   :members:
```

## Collimation

A {py:class}`~ompmc.ApertureMask` is something in the beam's way, applied
between the source and the shower. It works by back projection from wherever
the source put the particle, so it composes with any source and does not care
which side of its plane the particle started on -- which is what makes it
possible to cut a field out of a phase space recorded *above* the jaws, as
the published ones are.

With beamlets it is usually unnecessary: the collimation is already in the
weights {py:func}`~ompmc.calc_forward` takes. It is worth reaching for where
the weights cannot say what is wanted -- a block cutting across beamlets, or
a leaf that transmits -- and for a phase space, which has no weights to put
it in.

```{note}
This is a **mask, not a collimator**. It attenuates and blocks; it does not
scatter, and it does not harden the spectrum of what it lets through. A
particle that would have scattered off a leaf edge into the field is simply
gone. Good for the fluence, poor for the penumbra -- the same simplification
the beamlet weights make.
```

`roulette` chooses how a partly transmitting cell is paid for, and both
choices give the same dose in the mean:

```{list-table}
:header-rows: 1
:widths: 15 85

* - `roulette`
  - What a cell transmitting 2% costs
* - `False` (default)
  - The particle keeps 2% of its weight and is transported in full, so a
    shower runs for a fiftieth of the dose. In return **nothing is drawn**:
    every history's random stream is exactly where it would have been with
    the beam open, so a collimated run stays comparable, history by history,
    with the open one it came from.
* - `True`
  - The particle gets through one time in fifty at full weight and is stopped
    the other forty nine, so the time goes where the dose is. The price is
    one random number per history and more noise per history -- it pays when
    the leaves are thick and costs when they are nearly open.
```

Cells that are fully open or fully shut are decided without drawing either
way, so an all-or-nothing aperture -- a jaw, which is most of the use --
behaves identically under both.

```{eval-rst}
.. autoclass:: ompmc.ApertureMask
   :members:
```

## Calculations

```{eval-rst}
.. autofunction:: ompmc.calc_dij

.. autofunction:: ompmc.calc_forward

.. autofunction:: ompmc.calc_forward_phsp

.. autoclass:: ompmc.RunSummary

.. autofunction:: ompmc.calc_cube

.. autofunction:: ompmc.calc_radial
```

## Utilities

```{eval-rst}
.. autofunction:: ompmc.data_path

.. data:: ompmc.MAX_MEDIA

   Maximum number of distinct media a :class:`Geometry` may reference.
```

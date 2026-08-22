"""ompMC - OpenMP parallel Monte Carlo photon and electron transport.

Five calculations are available, sharing the same physics:

``calc_dij``
    One sparse column of dose per beamlet, the dose influence matrix a
    treatment planning system optimizes against.
``calc_forward``
    Dose everywhere in the phantom from a whole weighted set of beamlets at
    once -- what ``calc_dij(...) @ weights`` would give, computed directly.
``calc_forward_phsp``
    The same, from the particles of an IAEA phase space file rather than
    from a spectrum through an aperture.
``calc_cube``
    Dose everywhere in the phantom from a single collimated beam.
``calc_radial``
    Dose in a cylinder, binned into rings and depth slabs rather than voxels
    -- what a pencil beam distribution wants, since the dose around a narrow
    beam is exactly what a rectilinear grid is worst at. Takes a
    ``CylinderGeometry`` rather than a ``Geometry``.

The first four take the phantom as numpy arrays::

    import numpy as np, ompmc

    n = 32
    bounds = np.linspace(-8.0, 8.0, n + 1)
    density = np.full((n, n, n), 1.0, order="F")
    material = np.ones((n, n, n), dtype=np.int32, order="F")

    geometry = ompmc.Geometry(bounds, bounds, bounds, ["H2O521ICRU"], density,
                              material)

Note the ``order="F"``: the transport indexes voxels with the first axis
varying fastest, so a C-ordered cube would describe a transposed phantom. It is
rejected rather than silently copied.

Material indices count from 1, matching matRad's ``cubeMatIx``; 0 means vacuum.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import _ompmc

__all__ = [
    "Geometry",
    "CylinderGeometry",
    "Spectrum",
    "BeamletSource",
    "CollimatedSource",
    "PencilBeamSource",
    "PhaseSpaceSource",
    "ApertureMask",
    "Physics",
    "RunSummary",
    "calc_dij",
    "calc_forward",
    "calc_forward_phsp",
    "calc_cube",
    "calc_radial",
    "data_path",
    "__version__",
]

__version__ = _ompmc.__version__

MAX_MEDIA = _ompmc.MAX_MEDIA
"""Maximum number of distinct media a :class:`Geometry` may reference."""


def data_path() -> Path:
    """Directory holding the cross section data, PEGS files and spectra.

    Set ``OMPMC_DATA_PATH`` to override it; otherwise the copy shipped inside
    the package is used, falling back to the source tree when running from a
    checkout that was not installed.

    Returns
    -------
    pathlib.Path
        Directory containing the ``data``, ``pegs4`` and ``spectra``
        subdirectories.

    Raises
    ------
    FileNotFoundError
        If no such directory can be found and ``OMPMC_DATA_PATH`` is not set.
    """
    override = os.environ.get("OMPMC_DATA_PATH")
    if override:
        return Path(override)

    # An installed wheel carries the files inside the package; running from a
    # checkout finds them further up, wherever the package happens to sit in
    # the source tree.
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if all((candidate / name).is_dir()
               for name in ("data", "pegs4", "spectra")):
            return candidate

    raise FileNotFoundError(
        "Cannot find the ompMC data files. Set OMPMC_DATA_PATH to the "
        "directory holding 'data', 'pegs4' and 'spectra'."
    )


def _as_bounds(values, name: str) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a vector of at least two boundaries")
    if not np.all(np.diff(array) > 0):
        raise ValueError(f"{name} must be strictly ascending")
    return array


def _as_triples(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3), got {array.shape}")
    # Fortran order puts each component in one contiguous block, which is the
    # layout the engine reads.
    return np.asfortranarray(array)


@dataclass
class Geometry:
    """The voxel phantom: where the boundaries are and what is in each voxel.

    Parameters
    ----------
    x_bounds, y_bounds, z_bounds : array_like
        Strictly ascending voxel boundaries along each axis, in cm. ``n + 1``
        values describe ``n`` voxels along that axis.
    materials : sequence of str
        Medium names, matching entries in the PEGS file. `material` below
        indexes into this list, starting from 1; at most :data:`MAX_MEDIA`
        are supported.
    density : numpy.ndarray
        Fortran-ordered ``float64`` cube of mass densities in g/cm^3, shaped
        ``(len(x_bounds) - 1, len(y_bounds) - 1, len(z_bounds) - 1)``.
    material : numpy.ndarray
        Fortran-ordered ``int32`` cube of the same shape, indexing
        `materials` from 1; 0 means vacuum.

    Raises
    ------
    ValueError
        If a bounds vector is not strictly ascending, `materials` is empty
        or longer than :data:`MAX_MEDIA`, the cubes are not shaped like the
        bounds describe, `density` is negative anywhere, or `material` holds
        an index outside ``0 .. len(materials)``.
    TypeError
        If `density` or `material` do not have the required dtype.

    Notes
    -----
    The transport indexes voxels with the first axis varying fastest, so a
    C-ordered cube would describe a transposed phantom -- it is rejected
    rather than silently copied. Use ``np.asfortranarray(...)``.
    """

    x_bounds: np.ndarray
    y_bounds: np.ndarray
    z_bounds: np.ndarray
    materials: Sequence[str]
    density: np.ndarray
    material: np.ndarray

    def __post_init__(self) -> None:
        self.x_bounds = _as_bounds(self.x_bounds, "x_bounds")
        self.y_bounds = _as_bounds(self.y_bounds, "y_bounds")
        self.z_bounds = _as_bounds(self.z_bounds, "z_bounds")

        self.materials = [str(name) for name in self.materials]
        if not 1 <= len(self.materials) <= MAX_MEDIA:
            raise ValueError(
                f"between 1 and {MAX_MEDIA} materials are supported, "
                f"got {len(self.materials)}"
            )

        self.density = self._check_cube(self.density, np.float64, "density")
        self.material = self._check_cube(self.material, np.int32, "material")

        if np.any(self.density < 0.0):
            raise ValueError("density must not be negative")

        lo, hi = int(self.material.min()), int(self.material.max())
        if lo < 0 or hi > len(self.materials):
            raise ValueError(
                f"material indices run from {lo} to {hi}, outside 0 (vacuum) "
                f"to {len(self.materials)}"
            )

        shape = self.density.shape
        expected = (self.x_bounds.size - 1, self.y_bounds.size - 1,
                    self.z_bounds.size - 1)
        if shape != expected:
            raise ValueError(
                f"the cubes are {shape} but the boundaries describe {expected}"
            )

    @staticmethod
    def _check_cube(values, dtype, name: str) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 3:
            raise ValueError(f"{name} must be a three dimensional cube")
        if array.dtype != dtype:
            raise TypeError(f"{name} must have dtype {np.dtype(dtype).name}, "
                            f"got {array.dtype}")
        if not array.flags.f_contiguous:
            raise ValueError(
                f"{name} must be Fortran ordered: the transport indexes voxels "
                f"with the first axis varying fastest, so a C ordered cube "
                f"would describe a transposed phantom. Use "
                f"np.asfortranarray({name})."
            )
        return array

    @property
    def shape(self) -> tuple[int, int, int]:
        """Voxel count along each axis, ``(nx, ny, nz)``."""
        return self.density.shape

    @property
    def n_voxels(self) -> int:
        """Total number of voxels, ``nx * ny * nz``."""
        return int(self.density.size)


@dataclass
class CylinderGeometry:
    """A homogeneous cylinder, binned into rings and depth slabs.

    The phantom :func:`calc_radial` transports in. It sits about the z axis
    with its front face at ``z_bounds[0]`` and depth running along +z, which
    is the direction the beams in :class:`PencilBeamSource` travel.

    Rings rather than voxels because of what a narrow beam does to a
    rectilinear grid: dose falls by orders of magnitude over the first few
    millimetres off the axis, so following it needs voxels far finer than the
    rest of the phantom will ever need. A ring is the natural bin for it, and
    -- being the transport's own region rather than a sum over voxels
    afterwards -- comes with an uncertainty the batch statistics can actually
    speak for.

    Parameters
    ----------
    r_bounds : array_like
        Ring boundaries in cm, ascending, starting at 0. There is no hollow
        middle: the innermost ring reaches the axis.
    z_bounds : array_like
        Depth slab boundaries in cm, ascending.
    material : str
        Name of the PEGS medium the cylinder is made of, e.g.
        ``"H2O700ICRU"``.
    density : float, optional
        Mass density in g/cm^3. Left out, the medium's own PEGS density is
        used.

    Raises
    ------
    ValueError
        If either boundary list does not ascend, if `r_bounds` does not start
        at the axis, or if `density` is not positive.

    Examples
    --------
    Fine rings on the beam and coarse ones out where the dose has gone::

        geometry = ompmc.CylinderGeometry(
            r_bounds=[0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 5.0],
            z_bounds=np.linspace(0.0, 20.0, 41),
            material="H2O700ICRU",
            density=1.0,
        )
    """

    r_bounds: np.ndarray
    z_bounds: np.ndarray
    material: str
    density: float | None = None

    def __post_init__(self) -> None:
        self.r_bounds = _as_bounds(self.r_bounds, "r_bounds")
        self.z_bounds = _as_bounds(self.z_bounds, "z_bounds")

        # Region 0 already means "outside the phantom", so there is nothing
        # left for a hole in the middle to be.
        if self.r_bounds[0] != 0.0:
            raise ValueError(
                f"r_bounds must start at the axis, r = 0, not "
                f"{self.r_bounds[0]!r}: a cylinder with a hole in it is not "
                f"something ompMC can transport")

        if not isinstance(self.material, str) or not self.material:
            raise ValueError("material must be a non-empty PEGS medium name")

        if self.density is not None:
            self.density = float(self.density)
            if not self.density > 0.0:
                raise ValueError(
                    f"density must be positive, got {self.density!r}")

    @property
    def shape(self) -> tuple[int, int]:
        """Rings and depth slabs, as ``(n_rings, n_slabs)``."""
        return (self.r_bounds.size - 1, self.z_bounds.size - 1)

    @property
    def n_regions(self) -> int:
        """Number of scoring regions, i.e. rings times slabs."""
        return (self.r_bounds.size - 1)*(self.z_bounds.size - 1)


@dataclass
class Spectrum:
    """The energy distribution of the source particles.

    Build one with :meth:`from_file`, :meth:`from_histogram`,
    :meth:`monoenergetic` or :meth:`default`; there is no public constructor.
    """

    _payload: dict

    @classmethod
    def from_file(cls, path) -> "Spectrum":
        """Read an EGSnrc style ``.spectrum`` file.

        Parameters
        ----------
        path : str or os.PathLike
            Path to the spectrum file.

        Returns
        -------
        Spectrum

        Raises
        ------
        FileNotFoundError
            If `path` does not exist.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No spectrum file at {path}")
        return cls({"file": str(path)})

    @classmethod
    def from_histogram(cls, energy, fluence, e_min: float = 0.0,
                       per_mev: bool = False) -> "Spectrum":
        """A histogram spectrum.

        Parameters
        ----------
        energy : array_like
            Upper edge of each bin, in MeV; finite and strictly ascending.
        fluence : array_like
            Relative number of particles per bin, same length as `energy`;
            non-negative, summing to a positive, finite value.
        e_min : float, optional
            Lower edge of the first bin, in MeV. Must be less than
            ``energy[0]``.
        per_mev : bool, optional
            If true, `fluence` holds counts per MeV rather than counts per
            bin.

        Returns
        -------
        Spectrum

        Raises
        ------
        ValueError
            If the shapes of `energy` and `fluence` disagree, `energy` is
            not finite and strictly ascending above `e_min`, or `fluence`
            is not non-negative and finite-summing to a positive value.
        """
        energy = np.ascontiguousarray(energy, dtype=np.float64)
        fluence = np.ascontiguousarray(fluence, dtype=np.float64)

        if energy.ndim != 1 or energy.size == 0:
            raise ValueError("energy must be a non-empty vector")
        if energy.shape != fluence.shape:
            raise ValueError(
                f"energy has {energy.size} entries but fluence has {fluence.size}"
            )
        if not np.all(np.isfinite(energy)) or not np.all(np.diff(energy) > 0):
            raise ValueError("energy must be finite and strictly ascending")
        if e_min < 0.0 or energy[0] <= e_min:
            raise ValueError(
                f"the first bin ends at {energy[0]} MeV, which must be above "
                f"its lower edge e_min = {e_min} MeV"
            )
        if np.any(fluence < 0.0) or not np.isfinite(fluence.sum()) or \
                fluence.sum() <= 0.0:
            raise ValueError(
                "fluence must be non-negative and sum to a positive, finite value"
            )

        return cls({
            "energy": energy,
            "fluence": fluence,
            "e_min": float(e_min),
            "mode": 1 if per_mev else 0,
        })

    @classmethod
    def monoenergetic(cls, energy: float) -> "Spectrum":
        """Every particle starts with the same kinetic energy.

        Parameters
        ----------
        energy : float
            Kinetic energy in MeV; positive and finite.

        Returns
        -------
        Spectrum

        Raises
        ------
        ValueError
            If `energy` is not positive and finite.
        """
        energy = float(energy)
        if not energy > 0.0 or not np.isfinite(energy):
            raise ValueError(f"energy is {energy} MeV, it must be positive "
                             f"and finite")
        return cls({"mono_energy": energy})

    @classmethod
    def default(cls) -> "Spectrum":
        """The 6 MV bremsstrahlung spectrum shipped with ompMC.

        Returns
        -------
        Spectrum
        """
        return cls.from_file(data_path() / "spectra" / "mohan6.spectrum")


@dataclass
class BeamletSource:
    """Beamlet apertures at isocentre, one sparse Dij column each.

    Parameters
    ----------
    i_beam : array_like of int
        Index of the beam each beamlet belongs to, counting from 0.
    source : array_like
        Shape ``(n_beams, 3)``, the xyz source position of each beam.
    corner : array_like
        Shape ``(n_beamlets, 3)``, the corner of each beamlet's aperture
        rectangle.
    side1, side2 : array_like
        Shape ``(n_beamlets, 3)``, the two edge vectors spanning the
        aperture rectangle from `corner`.

    Raises
    ------
    ValueError
        If `i_beam` is empty, `corner`/`side1`/`side2` do not each have one
        row per beamlet, or `i_beam` references a beam outside `source`.
    """

    i_beam: np.ndarray
    source: np.ndarray
    corner: np.ndarray
    side1: np.ndarray
    side2: np.ndarray

    def __post_init__(self) -> None:
        self.i_beam = np.ascontiguousarray(self.i_beam, dtype=np.int32)
        if self.i_beam.ndim != 1 or self.i_beam.size == 0:
            raise ValueError("i_beam must be a non-empty vector")

        self.source = _as_triples(self.source, "source")
        self.corner = _as_triples(self.corner, "corner")
        self.side1 = _as_triples(self.side1, "side1")
        self.side2 = _as_triples(self.side2, "side2")

        n = self.i_beam.size
        for name in ("corner", "side1", "side2"):
            if getattr(self, name).shape[0] != n:
                raise ValueError(
                    f"{name} has {getattr(self, name).shape[0]} rows but there "
                    f"are {n} beamlets"
                )

        if self.i_beam.min() < 0 or self.i_beam.max() >= self.source.shape[0]:
            raise ValueError(
                f"i_beam runs from {self.i_beam.min()} to {self.i_beam.max()}, "
                f"outside the {self.source.shape[0]} beams given"
            )

    @property
    def n_beamlets(self) -> int:
        """Number of beamlets, ``len(i_beam)``."""
        return int(self.i_beam.size)


@dataclass
class CollimatedSource:
    """A point source at `ssd` behind a rectangular opening on the surface.

    Parameters
    ----------
    ssd : float
        Distance from the source to the phantom surface, in cm; positive.
    x_min, x_max, y_min, y_max : float
        Bounds of the rectangular opening on the phantom surface, in cm.

    Raises
    ------
    ValueError
        If `ssd` is not positive, or the opening has negative width in
        either direction.
    """

    ssd: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        if not self.ssd > 0.0:
            raise ValueError(f"ssd is {self.ssd} cm, it must be positive")
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("the collimator opening has negative width")


@dataclass
class PencilBeamSource:
    """A beam down the axis of a :class:`CylinderGeometry`.

    Two beams, distinguished by whether an `ssd` is given. Without one it is a
    parallel pencil of no width, every particle entering at r = 0 travelling
    along +z, which is what a dose kernel is defined for. With one it is a
    point source that far upstream of the front face, illuminating a disc on
    it -- what a real machine looks like.

    Either delta a real beam does not have can be widened into a Gaussian,
    independently of the other: `spot_sigma` gives it a width, and
    `divergence_sigma` gives it an angular spread. Both default to a delta,
    and a delta draws no random numbers, so a beam that asks for neither is
    exactly the beam it would have been without them.

    A beam with both can also be given a `correlation` between the two, which
    is what moves its waist off the front face. :meth:`focused` says the same
    thing the way beam data usually comes: where the waist is and how narrow
    it is there.

    Parameters
    ----------
    ssd : float, optional
        Distance from the point source to the front face, in cm. Left out,
        the beam is a parallel pencil instead.
    field_radius : float, optional
        Radius of the disc illuminated on the front face, in cm. Only
        meaningful with an `ssd`; left out, the whole face is illuminated.
    spot_sigma : float, optional
        Standard deviation of the starting position, in cm, spread as a round
        two-dimensional Gaussian across the beam. A parallel pencil is defined
        on the front face and starts its particles there, so this is the width
        of the beam where it enters; a point source starts its particles on
        its focal spot, so this is the size of that. Left out, the beam has no
        width.
    divergence_sigma : float, optional
        Standard deviation of the direction, in **radians**, spread as a round
        two-dimensional Gaussian about the nominal one. Left out, the beam
        does not diverge.
    correlation : float, optional
        How strongly where a particle starts predicts where it is going, from
        -1 to 1. Negative converges onto a waist inside the phantom, positive
        has already passed its waist upstream, and the default of 0 puts the
        waist on the front face. The same correlation applies in both
        transverse planes, which is what keeps the beam round.

    Raises
    ------
    ValueError
        If `ssd`, `field_radius`, `spot_sigma` or `divergence_sigma` is not
        positive, if a `field_radius` is given without an `ssd`, or if a
        `correlation` is outside [-1, 1] or given without both a `spot_sigma`
        and a `divergence_sigma` for it to relate.

    Warnings
    --------
    The point source spreads its particles evenly over the disc it
    illuminates -- uniform fluence on the entrance plane, the same convention
    :class:`CollimatedSource` follows. That is not an isotropic point source,
    whose fluence would fall off with the inverse square across the field, and
    the difference shows at short SSD.

    See Also
    --------
    focused : The same beam, described by where its waist is.

    Notes
    -----
    The width at a distance `s` downstream of where the beam is specified is

    .. math::

        \\sigma^2(s) = \\sigma^2 + 2 s \\rho \\sigma \\sigma'
                       + s^2 \\sigma'^2

    for `spot_sigma` :math:`\\sigma`, `divergence_sigma` :math:`\\sigma'` and
    `correlation` :math:`\\rho` -- the transport then widens it further, since
    what a detector at depth sees is that convolved with the scattering
    kernel. Without a correlation the beam only ever gets wider, which makes
    it a blurred pencil rather than a beam with emittance.

    A spot wide enough to reach past the edge of the cylinder will put some
    particles outside it, and a parallel one that starts outside never enters.
    Those histories still count towards the fluence the result is divided by;
    :class:`RunSummary` reports how many of them there were.

    Examples
    --------
    The kernel case, a 4 cm field at 100 cm, and a beam of finite emittance::

        pencil = ompmc.PencilBeamSource()
        machine = ompmc.PencilBeamSource(ssd=100.0, field_radius=4.0)
        real = ompmc.PencilBeamSource(spot_sigma=0.15, divergence_sigma=0.01)
    """

    ssd: float | None = None
    field_radius: float | None = None
    spot_sigma: float | None = None
    divergence_sigma: float | None = None
    correlation: float = 0.0

    def __post_init__(self) -> None:
        if self.ssd is not None:
            self.ssd = float(self.ssd)
            if not self.ssd > 0.0:
                raise ValueError(f"ssd must be positive, got {self.ssd!r}")

        if self.field_radius is not None:
            if self.ssd is None:
                raise ValueError(
                    "field_radius only means something for a point source; "
                    "give an ssd as well, or leave both out for a parallel "
                    "pencil beam")

            self.field_radius = float(self.field_radius)
            if not self.field_radius > 0.0:
                raise ValueError(
                    f"field_radius must be positive, got "
                    f"{self.field_radius!r}")

        for name in ("spot_sigma", "divergence_sigma"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not value > 0.0:
                    raise ValueError(
                        f"{name} must be positive, got {value!r}; leave it "
                        f"out for a beam with no spread at all")
                setattr(self, name, value)

        self.correlation = float(self.correlation)
        if not -1.0 <= self.correlation <= 1.0:
            raise ValueError(
                f"correlation must lie between -1 and 1, got "
                f"{self.correlation!r}")

        # The core ignores a correlation it cannot apply. Refusing it here
        # instead, because in Python it is far more likely to be a beam that
        # was meant to have a waist and quietly did not.
        if self.correlation != 0.0 and (self.spot_sigma is None
                                        or self.divergence_sigma is None):
            raise ValueError(
                "correlation relates where a particle starts to where it is "
                "going, so it needs both a spot_sigma and a "
                "divergence_sigma; give both, or leave the correlation out")

    @classmethod
    def focused(cls, waist_sigma: float, divergence_sigma: float,
                waist_depth: float, **kwargs) -> "PencilBeamSource":
        """Build a beam from where it is narrowest.

        The mirror of the constructor: beam data is usually quoted as a waist
        somewhere and an angular spread, rather than as the width on the
        phantom surface and a correlation. This converts the one into the
        other -- the returned beam has exactly `waist_sigma` at
        `waist_depth`.

        Parameters
        ----------
        waist_sigma : float
            Width at the waist, in cm. Must be positive.
        divergence_sigma : float
            Angular spread, in **radians**. Must be positive.
        waist_depth : float
            How far past the front face the waist sits, in cm. Positive is
            inside the phantom, 0 puts it on the face, and negative puts it
            upstream. Note that this is measured from the surface for both
            beams, including a point source specified by its `ssd`.
        **kwargs
            Passed on to the constructor -- `ssd` and `field_radius`.

        Returns
        -------
        PencilBeamSource
            A beam with the `spot_sigma` and `correlation` that put the waist
            there.

        Raises
        ------
        ValueError
            If `waist_sigma` or `divergence_sigma` is not positive.

        Examples
        --------
        A beam that comes to a 1 mm waist 5 cm into the phantom::

            beam = ompmc.PencilBeamSource.focused(0.1, 0.02, 5.0)

        Notes
        -----
        There is no waist a real divergence cannot reach: the correlation
        this produces always comes out inside [-1, 1].
        """
        waist_sigma = float(waist_sigma)
        divergence_sigma = float(divergence_sigma)

        if not waist_sigma > 0.0:
            raise ValueError(
                f"waist_sigma must be positive, got {waist_sigma!r}")
        if not divergence_sigma > 0.0:
            raise ValueError(
                f"divergence_sigma must be positive, got "
                f"{divergence_sigma!r}")

        # var(s) is smallest at s = -rho sigma / sigma', where it is
        # sigma^2 (1 - rho^2); solving both for sigma and rho gives this.
        drift = float(waist_depth)*divergence_sigma
        spot_sigma = math.hypot(waist_sigma, drift)

        return cls(spot_sigma=spot_sigma,
                   divergence_sigma=divergence_sigma,
                   correlation=-drift/spot_sigma, **kwargs)

    @property
    def waist(self) -> tuple[float, float]:
        """Where the beam is narrowest, as ``(sigma, depth)`` in cm.

        The inverse of :meth:`focused`, and 0 depth for any beam that was not
        given a correlation. Depth is measured from the front face, so a
        negative one is a beam that is already spreading when it arrives.
        """
        if not self.spot_sigma or not self.divergence_sigma:
            return (self.spot_sigma or 0.0, 0.0)

        rho = self.correlation

        return (self.spot_sigma*math.sqrt(1.0 - rho*rho),
                -rho*self.spot_sigma/self.divergence_sigma)

    @property
    def _payload(self) -> dict:
        return {
            "kind": 1 if self.ssd is not None else 0,
            "ssd": float(self.ssd) if self.ssd is not None else 0.0,
            # 0 is how the core spells "the whole front face"
            "field_radius": (float(self.field_radius)
                             if self.field_radius is not None else 0.0),
            # and how it spells "no spread", which draws no random numbers
            "spot_sigma": (float(self.spot_sigma)
                           if self.spot_sigma is not None else 0.0),
            "divergence_sigma": (float(self.divergence_sigma)
                                 if self.divergence_sigma is not None
                                 else 0.0),
            "correlation": float(self.correlation),
        }


@dataclass
class PhaseSpaceSource:
    """Particles read from an IAEA phase space file, one per history.

    A phase space is a record of every particle that crossed a plane in some
    earlier simulation of a treatment head -- the ones published at
    https://www-nds.iaea.org/phsp/ are the output of full models of real
    linacs. Using one starts histories from the machine's own particles
    rather than from a spectrum through an aperture, which is the difference
    between modelling the beam and describing it.

    The file is a pair, ``name.IAEAheader`` and ``name.IAEAphsp``; `path` is
    either the shared base name or either half of it.

    Parameters
    ----------
    path : str or os.PathLike
        Base name of the dataset, with or without the extension.
    order : str, optional
        ``"replay"`` (the default) walks the file in order from `first`,
        wrapping at the end, and draws no random numbers. ``"random"`` picks
        a particle per history, costing one random number, which is worth it
        when a run is much shorter than the file and a contiguous stretch of
        it would sample only one part of the beam.
    first : int, optional
        Particle the replay starts at. Ignored when `order` is
        ``"random"``.
    rotation : array_like, optional
        ``3x3`` rotation carrying the phase space's coordinate system into
        the phantom's, applied to positions and directions alike. Defaults
        to no rotation.
    translation : array_like, optional
        Three offsets in cm, added after `rotation`. Defaults to no
        translation.

    Raises
    ------
    ValueError
        If `order` is not one of the two names, `first` is negative,
        `rotation` is not ``3x3`` or is not a rotation, or `translation` is
        not a three vector.

    Notes
    -----
    ompMC transports photons, electrons and positrons; a neutron or proton
    in the file starts no history and is counted as one that produced
    nothing.
    """

    path: str | os.PathLike
    order: str = "replay"
    first: int = 0
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None

    _ORDERS = {"replay": 0, "random": 1}

    def __post_init__(self) -> None:
        if self.order not in self._ORDERS:
            raise ValueError(
                f"order is {self.order!r}, expected 'replay' or 'random'"
            )
        if self.first < 0:
            raise ValueError(f"first is {self.first}, it cannot be negative")

        if self.rotation is None:
            self.rotation = np.eye(3)
        self.rotation = np.ascontiguousarray(self.rotation, dtype=np.float64)
        if self.rotation.shape != (3, 3):
            raise ValueError(
                f"rotation has shape {self.rotation.shape}, expected (3, 3)"
            )

        # A matrix that is not a rotation would stretch the directions it
        # turns, and those have to stay unit vectors. The determinant alone
        # does not settle it -- a shear like [[1,1,0],[0,1,0],[0,0,1]] has
        # determinant 1 and still stretches, and a matrix holding a NaN
        # passes any comparison asked of it. What makes a rotation is
        # orthonormal rows, with the determinant then telling a rotation
        # from a reflection. The core checks the same thing; catching it
        # here says so in Python terms.
        if not np.all(np.isfinite(self.rotation)):
            raise ValueError("rotation holds values that are not finite")
        if not np.allclose(self.rotation @ self.rotation.T, np.eye(3),
                           atol=1e-6):
            raise ValueError(
                "rotation is not orthonormal: its rows have to be unit "
                "vectors at right angles to each other, or it would stretch "
                "the directions it turns"
            )
        if not np.isclose(np.linalg.det(self.rotation), 1.0, atol=1e-6):
            raise ValueError(
                f"rotation has determinant "
                f"{float(np.linalg.det(self.rotation)):.6g}, and a rotation "
                f"has 1; -1 with orthonormal rows is a reflection"
            )

        if self.translation is None:
            self.translation = np.zeros(3)
        self.translation = np.ascontiguousarray(self.translation,
                                                dtype=np.float64).ravel()
        if self.translation.size != 3:
            raise ValueError(
                f"translation has {self.translation.size} entries, expected 3"
            )

    @property
    def _payload(self) -> dict:
        return {
            "path": str(self.path),
            "order": self._ORDERS[self.order],
            "first": int(self.first),
            # Row major, which is how the core reads the nine values
            "rotation": [float(v) for v in self.rotation.ravel(order="C")],
            "translation": [float(v) for v in self.translation],
        }


@dataclass
class ApertureMask:
    """Something in the beam's way: a transmission grid on a plane.

    The mask sits on the plane ``z = z`` of the phantom's coordinate system,
    where a jaw or a leaf bank would be, and cell ``(i, j)`` holds the
    fraction of a particle crossing it that gets through -- 1 open, 0 shut,
    anything between for a leaf that transmits.

    It works by back projection from wherever the source put the particle,
    so it composes with any source and does not care which side of the plane
    the particle started on. That is what makes it possible to cut a field
    out of a phase space recorded above the jaws, as the IAEA ones are.

    Parameters
    ----------
    z : float
        The plane the mask sits on, in cm.
    x0, y0 : float
        Lower corner of the grid, in cm.
    dx, dy : float
        Cell size in cm, both positive.
    transmission : array_like
        ``(nx, ny)`` fractions in ``[0, 1]``, the first axis along x.
    outside : float, optional
        What gets through beside the grid, in ``[0, 1]``. Zero -- what a
        field stop does -- is the default.
    roulette : bool, optional
        How a partly transmitting cell is paid for. False multiplies the
        particle's weight by the fraction and transports it regardless, so a
        2% leaf costs a full shower for a fiftieth of the dose but draws no
        random numbers at all. True lets the particle through with that
        probability at full weight instead, spending the time on the
        particles that matter at the price of one random number and more
        noise per history. Cells that are fully open or fully shut are
        decided without drawing either way, so an all-or-nothing aperture
        behaves identically under both.

    Raises
    ------
    ValueError
        If `transmission` is not a non-empty 2-D grid of values in
        ``[0, 1]``, the cells are not positive, or `outside` is outside
        ``[0, 1]``.

    Notes
    -----
    This is a mask, not a collimator: it attenuates and blocks, but does not
    scatter and does not harden the spectrum of what it lets through. Good
    for the fluence, poor for the penumbra -- the same simplification the
    beamlet weights of :func:`calc_forward` make.
    """

    z: float
    x0: float
    y0: float
    dx: float
    dy: float
    transmission: np.ndarray
    outside: float = 0.0
    roulette: bool = False

    def __post_init__(self) -> None:
        # Fortran order puts x contiguous, which is the layout the core reads
        self.transmission = np.asfortranarray(self.transmission,
                                              dtype=np.float64)
        if self.transmission.ndim != 2 or self.transmission.size == 0:
            raise ValueError(
                f"transmission must be a non-empty (nx, ny) grid, got shape "
                f"{self.transmission.shape}"
            )
        if not np.all(np.isfinite(self.transmission)):
            raise ValueError("transmission holds values that are not finite")
        if self.transmission.min() < 0.0 or self.transmission.max() > 1.0:
            raise ValueError(
                f"transmission runs from {self.transmission.min():.6g} to "
                f"{self.transmission.max():.6g}, and a fraction has to be "
                f"between 0 and 1"
            )

        if not (self.dx > 0.0 and self.dy > 0.0):
            raise ValueError(
                f"the cells are {self.dx} by {self.dy} cm, and both have to "
                f"be positive"
            )
        if not 0.0 <= self.outside <= 1.0:
            raise ValueError(
                f"outside is {self.outside}, and a fraction has to be between "
                f"0 and 1"
            )

    @classmethod
    def rectangle(cls, z: float, x_min: float, x_max: float, y_min: float,
                  y_max: float, *, outside: float = 0.0,
                  roulette: bool = False) -> "ApertureMask":
        """One open cell: a rectangular field, which is most of the use.

        Parameters
        ----------
        z : float
            The plane the aperture sits on, in cm.
        x_min, x_max, y_min, y_max : float
            The opening, in cm. Note these are at `z`, not at isocentre: an
            opening of ``w`` at ``z`` grows to ``w * iso / z`` there.
        outside : float, optional
            What gets through beyond the opening; 0 by default.
        roulette : bool, optional
            As in the constructor. Makes no difference to an opening that is
            fully open and fully shut outside it.

        Returns
        -------
        ApertureMask

        Raises
        ------
        ValueError
            If the opening has zero or negative width in either direction.
        """
        if not (x_max > x_min and y_max > y_min):
            raise ValueError(
                f"the opening runs from {x_min} to {x_max} across and "
                f"{y_min} to {y_max} up, and both have to be positive"
            )

        return cls(z=z, x0=x_min, y0=y_min, dx=x_max - x_min,
                   dy=y_max - y_min, transmission=np.ones((1, 1)),
                   outside=outside, roulette=roulette)

    @property
    def shape(self) -> tuple[int, int]:
        """Cells along x and y."""
        return (int(self.transmission.shape[0]),
                int(self.transmission.shape[1]))

    @property
    def _payload(self) -> dict:
        return {
            "z": float(self.z),
            "x0": float(self.x0),
            "y0": float(self.y0),
            "dx": float(self.dx),
            "dy": float(self.dy),
            "transmission": self.transmission,
            "outside": float(self.outside),
            "roulette": bool(self.roulette),
        }


@dataclass
class RunSummary:
    """What became of the histories a run asked for."""

    n_histories: int
    """Histories actually run -- the number asked for, rounded down to a whole
    number of batches."""

    n_started: int
    """Histories that put a particle into the phantom."""

    n_blocked: int
    """Histories a collimator stopped, before the particle was carried
    anywhere. The rest -- ``n_histories - n_started - n_blocked`` -- got past
    the collimator but missed the phantom, or offered a particle ompMC does
    not transport."""

    energy_fraction: float
    """The fraction of the energy that entered the phantom which stayed in
    it; the rest left through a face."""


@dataclass
class Physics:
    """Transport parameters and where the interaction data lives.

    Parameters
    ----------
    pegs_file, pgs4form_file, data_folder, output_folder : str, os.PathLike or None, optional
        Override the corresponding file or directory; each defaults to the
        matching path under :func:`data_path`.
    global_ecut, global_pcut : float, optional
        Global electron and photon transport cut-offs, in MeV.
    n_split : int, optional
        Photon splitting factor at the source; ``1`` disables splitting.
    seeds : tuple of int, optional
        The two seeds of the Philox4x32-10 random number generator.
    esave : float or None, optional
        Electron range-rejection threshold, in MeV; ``None`` disables it.
    e_rr, f_rr : float or None, optional
        Russian roulette threshold, in MeV, and survival factor. Both must
        be set (`f_rr` > 1) to take effect.
    """

    pegs_file: str | os.PathLike | None = None
    pgs4form_file: str | os.PathLike | None = None
    data_folder: str | os.PathLike | None = None
    output_folder: str | os.PathLike | None = None

    global_ecut: float = 0.7
    global_pcut: float = 0.01
    n_split: int = 20
    seeds: tuple[int, int] = (97, 33)

    # Variance reduction, off unless set
    esave: float | None = None
    e_rr: float | None = None
    f_rr: float | None = None

    def input_items(self) -> dict[str, str]:
        """The key/value pairs the core library reads its configuration from.

        Returns
        -------
        dict of str to str
        """
        root = data_path()

        pegs = self.pegs_file or root / "pegs4" / "700icru.pegs4dat"
        pgs4form = self.pgs4form_file or root / "pegs4" / "pgs4form.dat"
        data = self.data_folder or root / "data"
        output = self.output_folder or root / "output"

        items = {
            "global ecut": f"{self.global_ecut:.10g}",
            "global pcut": f"{self.global_pcut:.10g}",
            "nsplit": str(int(self.n_split)),
            "rng seeds": f"{int(self.seeds[0])} {int(self.seeds[1])}",
            "pegs file": str(pegs),
            "pgs4form file": str(pgs4form),
            # The core appends file names to these, so they need the separator
            "data folder": os.path.join(str(data), ""),
            "output folder": os.path.join(str(output), ""),
        }

        for key, value in (("esave", self.esave), ("e_rr", self.e_rr),
                           ("f_rr", self.f_rr)):
            if value is not None:
                items[key] = f"{float(value):.10g}"

        return items


def _check_run(n_histories: int, n_batches: int, charge: int) -> None:
    if n_batches < 2:
        raise ValueError(
            f"n_batches is {n_batches}; at least 2 are needed for the "
            f"uncertainty estimate"
        )
    if n_histories < 1:
        raise ValueError(f"n_histories is {n_histories}, it must be positive")
    if charge not in (-1, 0, 1):
        raise ValueError(
            f"charge is {charge}, expected -1 for electrons, 0 for photons or "
            f"+1 for positrons"
        )


def calc_dij(
    geometry: Geometry,
    source: BeamletSource,
    spectrum: Spectrum | None = None,
    physics: Physics | None = None,
    *,
    n_histories: int = 10_000,
    n_batches: int = 10,
    charge: int = 0,
    rel_dose_threshold: float = 0.01,
    gaussian_source: bool = False,
    source_width: float = 0.2123,
    variance: bool = False,
    progress: Callable[[float], bool | None] | None = None,
    verbosity: int = 0,
):
    """Calculate the dose influence matrix, one sparse column per beamlet.

    Parameters
    ----------
    geometry : Geometry
        The voxel phantom.
    source : BeamletSource
        The beamlets to calculate a column for.
    spectrum : Spectrum, optional
        Source energy spectrum. Defaults to :meth:`Spectrum.default`.
    physics : Physics, optional
        Transport parameters and data file locations. Defaults to
        ``Physics()``.
    n_histories : int, optional
        Histories simulated per beamlet.
    n_batches : int, optional
        Statistical batches per beamlet, at least 2, needed for the
        uncertainty estimate.
    charge : int, optional
        Source particle: ``-1`` electrons, ``0`` photons, ``1`` positrons.
    rel_dose_threshold : float, optional
        Voxels below this fraction of a beamlet's maximum dose are dropped
        from its column. In ``[0, 1)``.
    gaussian_source : bool, optional
        Spread the starting point over the collimator plane instead of a
        point source, softening the penumbra.
    source_width : float, optional
        Standard deviation of the Gaussian source, in cm. Ignored unless
        `gaussian_source` is true.
    variance : bool, optional
        Also return the variance of the mean, per voxel.
    progress : callable, optional
        Called with the fraction finished, in ``[0, 1]``, once per batch and
        once per beamlet. Returning ``False`` stops the calculation.
    verbosity : int, optional
        Log level passed to the engine.

    Returns
    -------
    scipy.sparse.csc_array
        Shape ``(geometry.n_voxels, source.n_beamlets)``, dose in Gy per
        incident particle. When `variance` is true, a ``(dose, variance)``
        pair of such arrays instead.

    Raises
    ------
    ValueError
        If `n_batches`, `n_histories`, `charge` or `rel_dose_threshold` are
        out of range.
    KeyboardInterrupt
        If `progress` returned false, or Ctrl-C was pressed, stopping the
        calculation before every beamlet was reported.
    """
    from scipy.sparse import csc_array

    _check_run(n_histories, n_batches, charge)
    if not 0.0 <= rel_dose_threshold < 1.0:
        raise ValueError(
            f"rel_dose_threshold is {rel_dose_threshold}, it must be in [0, 1)"
        )

    spectrum = spectrum or Spectrum.default()
    physics = physics or Physics()

    options = {
        "n_histories": int(n_histories),
        "n_batches": int(n_batches),
        "charge": int(charge),
        "rel_dose_threshold": float(rel_dose_threshold),
        "gaussian_source": bool(gaussian_source),
        "source_width": float(source_width),
        "want_variance": bool(variance),
    }

    data, indices, indptr, variance_data, beamlets_done = _ompmc.calc_dij(
        geometry.density, geometry.material, geometry.x_bounds,
        geometry.y_bounds, geometry.z_bounds, list(geometry.materials),
        source.i_beam, source.source, source.corner, source.side1,
        source.side2, options, physics.input_items(), spectrum._payload,
        progress, int(verbosity),
    )

    if beamlets_done != source.n_beamlets:
        raise KeyboardInterrupt(
            f"the calculation was stopped after {beamlets_done} of "
            f"{source.n_beamlets} beamlets"
        )

    shape = (geometry.n_voxels, source.n_beamlets)
    dij = csc_array((data, indices, indptr), shape=shape)

    if not variance:
        return dij

    return dij, csc_array((variance_data, indices, indptr), shape=shape)


def calc_forward(
    geometry: Geometry,
    source: BeamletSource,
    weights,
    spectrum: Spectrum | None = None,
    physics: Physics | None = None,
    *,
    n_histories: int = 10_000,
    n_batches: int = 10,
    charge: int = 0,
    gaussian_source: bool = False,
    source_width: float = 0.2123,
    output_dose: bool = True,
    collimator: ApertureMask | None = None,
    progress: Callable[[float], bool | None] | None = None,
    verbosity: int = 0,
):
    """Calculate the dose of a whole weighted set of beamlets, in one cube.

    This is what ``calc_dij(...) @ weights`` would give, computed directly.
    `weights` is where the collimation comes in: a blocked beamlet gets 0, an
    open one its fluence, a partly transmitting one a fraction of it.
    Histories go to the beamlets in proportion to their weight, so a blocked
    beamlet costs nothing and the run time no longer grows with the number of
    beamlets.

    Parameters
    ----------
    geometry : Geometry
        The voxel phantom.
    source : BeamletSource
        The weighted beamlets.
    weights : array_like
        One finite, non-negative value per beamlet, summing to a finite,
        positive total. Modulates fluence, not spectrum: a beamlet at 0.02
        starts 2% of the particles, with the spectrum unhardened.
        Attenuation in a collimator, its scatter and the beam hardening that
        goes with it are not modelled.
    spectrum : Spectrum, optional
        Source energy spectrum. Defaults to :meth:`Spectrum.default`.
    physics : Physics, optional
        Transport parameters and data file locations. Defaults to
        ``Physics()``.
    n_histories : int, optional
        Histories simulated over the whole calculation -- unlike
        :func:`calc_dij`, this does not count per beamlet, so multiply it by
        `source.n_beamlets` to keep the same statistics as a `calc_dij` run.
    n_batches : int, optional
        Statistical batches, at least 2, needed for the uncertainty
        estimate.
    charge : int, optional
        Source particle: ``-1`` electrons, ``0`` photons, ``1`` positrons.
    gaussian_source : bool, optional
        Spread the starting point over the collimator plane instead of a
        point source, softening the penumbra.
    source_width : float, optional
        Standard deviation of the Gaussian source, in cm. Ignored unless
        `gaussian_source` is true.
    output_dose : bool, optional
        If true, the dose is in Gy for exactly these weights -- doubling
        them doubles it. If false, the mean deposited energy is returned
        instead.
    collimator : ApertureMask, optional
        Something in the beam's way, on top of the weights. Usually
        unnecessary -- with beamlets the collimation is already in the
        weights -- and worth reaching for only where the weights cannot say
        what is wanted, such as a block cutting across beamlets or a leaf
        that transmits.
    progress : callable, optional
        Called with the fraction finished, in ``[0, 1]``, once per batch.
        Returning ``False`` stops the calculation.
    verbosity : int, optional
        Log level passed to the engine.

    Returns
    -------
    dose : numpy.ndarray
        Cube shaped like the phantom.
    uncertainty : numpy.ndarray
        Cube shaped like the phantom, the relative uncertainty of `dose`,
        and 0.9999999 where nothing was deposited.

    Raises
    ------
    ValueError
        If `n_batches`, `n_histories` or `charge` are out of range, `weights`
        does not have one entry per beamlet, holds a negative or non-finite
        value, or sums to zero or a non-finite value.
    KeyboardInterrupt
        If `progress` returned false, or Ctrl-C was pressed, before any
        result was available -- the batches are averaged, so a run stopped
        partway through has no result to return.

    Notes
    -----
    There is no ``rel_dose_threshold`` as in :func:`calc_dij`: it prunes
    columns of a sparse matrix, and there is no matrix here. Worth
    remembering when comparing the two, since it is the `calc_dij` result
    that gets pruned.
    """
    _check_run(n_histories, n_batches, charge)

    weights = np.ascontiguousarray(weights, dtype=np.float64).ravel()

    if weights.size != source.n_beamlets:
        raise ValueError(
            f"weights has {weights.size} entries but there are "
            f"{source.n_beamlets} beamlets"
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights holds values that are not finite")
    if np.any(weights < 0.0):
        raise ValueError(
            f"{int(np.sum(weights < 0.0))} of the weights are negative"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        total_weight = weights.sum()
    if not np.isfinite(total_weight):
        raise ValueError("weights must sum to a finite value")
    if not total_weight > 0.0:
        raise ValueError("every weight is zero, so there is nothing to "
                         "calculate")

    spectrum = spectrum or Spectrum.default()
    physics = physics or Physics()

    options = {
        "n_histories": int(n_histories),
        "n_batches": int(n_batches),
        "charge": int(charge),
        "gaussian_source": bool(gaussian_source),
        "source_width": float(source_width),
        "output_dose": bool(output_dose),
    }

    (dose, uncertainty, completed,
     _nhist, _nsampled, _nweighted, _kept, _fraction,
     _blocked) = _ompmc.calc_forward(
        geometry.density, geometry.material, geometry.x_bounds,
        geometry.y_bounds, geometry.z_bounds, list(geometry.materials),
        source.i_beam, source.source, source.corner, source.side1,
        source.side2, weights, options, physics.input_items(),
        spectrum._payload,
        collimator._payload if collimator is not None else None,
        progress, int(verbosity),
    )

    if not completed:
        # The batches are averaged, so a run that stopped partway through is a
        # dose with no meaning; there is nothing to hand back.
        raise KeyboardInterrupt(
            "the calculation was stopped before any result was available")

    shape = geometry.shape
    return (dose.reshape(shape, order="F"),
            uncertainty.reshape(shape, order="F"))


def calc_forward_phsp(
    geometry: Geometry,
    source: PhaseSpaceSource,
    physics: Physics | None = None,
    *,
    n_histories: int = 10_000,
    n_batches: int = 10,
    output_dose: bool = True,
    collimator: ApertureMask | None = None,
    progress: Callable[[float], bool | None] | None = None,
    verbosity: int = 0,
):
    """Calculate the dose from the particles of a phase space file.

    One particle of the file starts each history, moved into the phantom's
    coordinate system by the source's transform and then carried to whatever
    face of the phantom it enters by. There is no `spectrum` argument: the
    file carries the energy of every particle it holds, which is most of the
    reason for using one.

    The whole file is read into memory, so it costs about its own size on
    disk -- gigabytes for a published dataset.

    Parameters
    ----------
    geometry : Geometry
        The voxel phantom.
    source : PhaseSpaceSource
        The file, and where it sits relative to the phantom.
    physics : Physics, optional
        Transport parameters and data file locations. Defaults to
        ``Physics()``.
    n_histories : int, optional
        Histories simulated. A history whose particle misses the phantom, or
        which the collimator stops, still counts as one -- see
        :class:`RunSummary`.
    n_batches : int, optional
        Statistical batches, at least 2, needed for the uncertainty
        estimate.
    output_dose : bool, optional
        If true, dose per history in Gy. If false, the mean deposited
        energy.
    collimator : ApertureMask, optional
        Something in the beam's way. The published phase spaces are recorded
        above the jaws, field independent on purpose, so this is how a field
        gets cut out of one.
    progress : callable, optional
        Called with the fraction finished, in ``[0, 1]``, once per batch.
        Returning ``False`` stops the calculation.
    verbosity : int, optional
        Log level passed to the engine.

    Returns
    -------
    dose : numpy.ndarray
        Cube shaped like the phantom.
    uncertainty : numpy.ndarray
        Cube shaped like the phantom, the relative uncertainty of `dose`,
        and 0.9999999 where nothing was deposited.
    summary : RunSummary
        What became of the histories. Worth looking at here in a way it is
        not for beamlets: a phase space is recorded wherever the original
        simulation scored it, not aimed at this phantom, so it is normal for
        most histories to start nothing.

    Raises
    ------
    ValueError
        If `n_batches` or `n_histories` are out of range.
    RuntimeError
        If the file cannot be read, does not match its header, or holds no
        particles.
    KeyboardInterrupt
        If `progress` returned false, or Ctrl-C was pressed, before any
        result was available.

    Warnings
    --------
    One particle per history. A phase space records which particles a single
    original history left behind, and those are correlated. Drawing them one
    at a time still gets the dose right on average, but the uncertainty a run
    reports comes out smaller than the truth by however much they are
    correlated.

    Examples
    --------
    Cutting a 10 x 10 cm field at 100 cm out of a phase space scored at the
    top of the jaws::

        source = ompmc.PhaseSpaceSource("Varian_TrueBeam6MV_01")
        jaw = ompmc.ApertureMask.rectangle(40.0, -2.0, 2.0, -2.0, 2.0)

        dose, unc, summary = ompmc.calc_forward_phsp(
            geometry, source, n_histories=1_000_000, collimator=jaw)
    """
    _check_run(n_histories, n_batches, 0)

    physics = physics or Physics()

    options = {
        "n_histories": int(n_histories),
        "n_batches": int(n_batches),
        "output_dose": bool(output_dose),
    }

    (dose, uncertainty, completed, nhist, started, blocked,
     fraction) = _ompmc.calc_forward_phsp(
        geometry.density, geometry.material, geometry.x_bounds,
        geometry.y_bounds, geometry.z_bounds, list(geometry.materials),
        source._payload, options, physics.input_items(),
        collimator._payload if collimator is not None else None,
        progress, int(verbosity),
    )

    if not completed:
        raise KeyboardInterrupt(
            "the calculation was stopped before any result was available")

    shape = geometry.shape
    return (dose.reshape(shape, order="F"),
            uncertainty.reshape(shape, order="F"),
            RunSummary(int(nhist), int(started), int(blocked),
                       float(fraction)))


def calc_cube(
    geometry: Geometry,
    source: CollimatedSource,
    spectrum: Spectrum | None = None,
    physics: Physics | None = None,
    *,
    n_histories: int = 10_000,
    n_batches: int = 10,
    charge: int = 0,
    output_dose: bool = True,
    progress: Callable[[float], bool | None] | None = None,
    verbosity: int = 0,
):
    """Calculate the dose everywhere in the phantom from one collimated beam.

    Parameters
    ----------
    geometry : Geometry
        The voxel phantom.
    source : CollimatedSource
        The point source and its collimator opening.
    spectrum : Spectrum, optional
        Source energy spectrum. Defaults to :meth:`Spectrum.default`.
    physics : Physics, optional
        Transport parameters and data file locations. Defaults to
        ``Physics()``.
    n_histories : int, optional
        Histories simulated.
    n_batches : int, optional
        Statistical batches, at least 2, needed for the uncertainty
        estimate.
    charge : int, optional
        Source particle: ``-1`` electrons, ``0`` photons, ``1`` positrons.
    output_dose : bool, optional
        If true, the dose is in Gy per incident fluence. If false, the mean
        deposited energy is returned instead.
    progress : callable, optional
        Called with the fraction finished, in ``[0, 1]``, once per batch.
        Returning ``False`` stops the calculation.
    verbosity : int, optional
        Log level passed to the engine.

    Returns
    -------
    dose : numpy.ndarray
        Cube shaped like the phantom.
    uncertainty : numpy.ndarray
        Cube shaped like the phantom, the relative uncertainty of `dose`,
        and 0.9999999 where nothing was deposited.

    Raises
    ------
    ValueError
        If `n_batches`, `n_histories` or `charge` are out of range.
    """
    _check_run(n_histories, n_batches, charge)

    spectrum = spectrum or Spectrum.default()
    physics = physics or Physics()

    options = {
        "n_histories": int(n_histories),
        "n_batches": int(n_batches),
        "charge": int(charge),
        "output_dose": bool(output_dose),
        "ssd": float(source.ssd),
        "x_min": float(source.x_min),
        "x_max": float(source.x_max),
        "y_min": float(source.y_min),
        "y_max": float(source.y_max),
    }

    dose, uncertainty, _nhist, _fraction = _ompmc.calc_cube(
        geometry.density, geometry.material, geometry.x_bounds,
        geometry.y_bounds, geometry.z_bounds, list(geometry.materials),
        options, physics.input_items(), spectrum._payload, progress,
        int(verbosity),
    )

    shape = geometry.shape
    return (dose.reshape(shape, order="F"),
            uncertainty.reshape(shape, order="F"))


def calc_radial(
    geometry: CylinderGeometry,
    source: PencilBeamSource | PhaseSpaceSource,
    spectrum: Spectrum | None = None,
    physics: Physics | None = None,
    *,
    n_histories: int = 10_000,
    n_batches: int = 10,
    charge: int = 0,
    output_dose: bool = True,
    collimator: ApertureMask | None = None,
    progress: Callable[[float], bool | None] | None = None,
    verbosity: int = 0,
):
    """Calculate the dose in a cylinder, by radial ring and depth slab.

    The r-z counterpart of :func:`calc_forward`, and what a pencil beam dose
    distribution wants: the rings are the transport's own regions, so each one
    gets its uncertainty from the batch statistics directly rather than from
    summing correlated voxels afterwards.

    Any source will do. :class:`PencilBeamSource` gives the two beams that
    shine down the axis; :class:`PhaseSpaceSource` replays a file, moved into
    the phantom's coordinate system by its own transform.

    Parameters
    ----------
    geometry : CylinderGeometry
        The cylinder, its rings and its depth slabs.
    source : PencilBeamSource or PhaseSpaceSource
        Where the particles come from.
    spectrum : Spectrum, optional
        Energies for a :class:`PencilBeamSource`, defaulting to
        :meth:`Spectrum.default`. Must not be given with a
        :class:`PhaseSpaceSource`, which carries its own energies.
    physics : Physics, optional
        Transport parameters and data file locations. Defaults to
        ``Physics()``.
    n_histories : int, optional
        Histories simulated. A history whose particle misses the phantom, or
        which the collimator stops, still counts as one.
    n_batches : int, optional
        Statistical batches, at least 2, needed for the uncertainty estimate.
    charge : int, optional
        0 for photons, -1 for electrons, +1 for positrons. Ignored for a
        phase space, which carries its own particle types.
    output_dose : bool, optional
        If true, dose per incident history in Gy. If false, the mean
        deposited energy.
    collimator : ApertureMask, optional
        Something in the beam's way.
    progress : callable, optional
        Called with the fraction finished, in ``[0, 1]``, once per batch.
        Returning ``False`` stops the calculation.
    verbosity : int, optional
        Log level passed to the engine.

    Returns
    -------
    dose : numpy.ndarray
        Shaped ``(n_rings, n_slabs)``, in Gy per incident history.
    uncertainty : numpy.ndarray
        Shaped ``(n_rings, n_slabs)``, the relative uncertainty of `dose`, and
        0.9999999 where nothing was deposited.
    summary : RunSummary
        What became of the histories.

    Raises
    ------
    ValueError
        If `n_batches`, `n_histories` or `charge` are out of range, or if a
        `spectrum` is given together with a :class:`PhaseSpaceSource`.
    RuntimeError
        If the geometry or the source is one the engine cannot run.
    KeyboardInterrupt
        If `progress` returned false, or Ctrl-C was pressed, before any result
        was available.

    Notes
    -----
    The result is the dose one incident particle delivers, not the dose per
    unit fluence :func:`calc_cube` reports -- there is no field for a pencil
    beam to have a fluence over.

    Examples
    --------
    The depth dose on the axis of a 6 MV photon pencil in water::

        geometry = ompmc.CylinderGeometry(
            r_bounds=np.linspace(0.0, 5.0, 21),
            z_bounds=np.linspace(0.0, 20.0, 41),
            material="H2O700ICRU", density=1.0)

        dose, unc, summary = ompmc.calc_radial(
            geometry, ompmc.PencilBeamSource(), n_histories=1_000_000)

        depth_dose_on_axis = dose[0, :]
    """
    from_phsp = isinstance(source, PhaseSpaceSource)

    _check_run(n_histories, n_batches, 0 if from_phsp else charge)

    if from_phsp and spectrum is not None:
        raise ValueError(
            "a phase space carries the energy of every particle it holds, so "
            "it takes no spectrum")

    if not from_phsp:
        spectrum = spectrum or Spectrum.default()

    physics = physics or Physics()

    options = {
        "n_histories": int(n_histories),
        "n_batches": int(n_batches),
        "charge": int(charge),
        "output_dose": bool(output_dose),
    }

    (dose, uncertainty, completed, nhist, started, blocked,
     fraction) = _ompmc.calc_radial(
        geometry.r_bounds, geometry.z_bounds, geometry.material,
        # 0 is how the core spells "whatever the PEGS data says it weighs"
        float(geometry.density) if geometry.density is not None else 0.0,
        source._payload, options, physics.input_items(),
        None if from_phsp else spectrum._payload,
        collimator._payload if collimator is not None else None,
        progress, int(verbosity),
    )

    if not completed:
        raise KeyboardInterrupt(
            "the calculation was stopped before any result was available")

    shape = geometry.shape
    return (dose.reshape(shape, order="F"),
            uncertainty.reshape(shape, order="F"),
            RunSummary(int(nhist), int(started), int(blocked),
                       float(fraction)))

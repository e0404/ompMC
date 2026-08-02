"""ompMC - OpenMP parallel Monte Carlo photon and electron transport.

Two calculations are available, sharing the same phantom, physics and source
spectra:

``calc_dij``
    One sparse column of dose per beamlet, the dose influence matrix a
    treatment planning system optimizes against.
``calc_cube``
    Dose everywhere in the phantom from a single collimated beam.

Both take the phantom as numpy arrays::

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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import _ompmc

__all__ = [
    "Geometry",
    "Spectrum",
    "BeamletSource",
    "CollimatedSource",
    "Physics",
    "calc_dij",
    "calc_cube",
    "data_path",
    "__version__",
]

__version__ = _ompmc.__version__

MAX_MEDIA = _ompmc.MAX_MEDIA


def data_path() -> Path:
    """Directory holding the cross section data, PEGS files and spectra.

    Set ``OMPMC_DATA_PATH`` to override it; otherwise the copy shipped inside
    the package is used, falling back to the source tree when running from a
    checkout that was not installed.
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
    """The voxel phantom: where the boundaries are and what is in each voxel."""

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
        return self.density.shape

    @property
    def n_voxels(self) -> int:
        return int(self.density.size)


@dataclass
class Spectrum:
    """The energy distribution of the source particles.

    Build one with :meth:`from_file`, :meth:`from_histogram` or
    :meth:`monoenergetic`.
    """

    _payload: dict

    @classmethod
    def from_file(cls, path) -> "Spectrum":
        """Read an EGSnrc style ``.spectrum`` file."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No spectrum file at {path}")
        return cls({"file": str(path)})

    @classmethod
    def from_histogram(cls, energy, fluence, e_min: float = 0.0,
                       per_mev: bool = False) -> "Spectrum":
        """A histogram: ``energy`` holds the upper edge of each bin in MeV."""
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
        """Every particle starts with the same kinetic energy, in MeV."""
        energy = float(energy)
        if not energy > 0.0 or not np.isfinite(energy):
            raise ValueError(f"energy is {energy} MeV, it must be positive "
                             f"and finite")
        return cls({"mono_energy": energy})

    @classmethod
    def default(cls) -> "Spectrum":
        """The 6 MV bremsstrahlung spectrum shipped with ompMC."""
        return cls.from_file(data_path() / "spectra" / "mohan6.spectrum")


@dataclass
class BeamletSource:
    """Beamlet apertures at isocentre, one sparse Dij column each.

    ``source`` holds one xyz row per beam, the other three one row per beamlet:
    the corner of the aperture rectangle and the two edge vectors spanning it.
    ``i_beam`` says which beam each beamlet belongs to, counting from 0.
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
        return int(self.i_beam.size)


@dataclass
class CollimatedSource:
    """A point source at ``ssd`` behind a rectangular opening on the surface."""

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
class Physics:
    """Transport parameters and where the interaction data lives."""

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
        """The key/value pairs the core library reads its configuration from."""
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

    Returns a ``scipy.sparse.csc_array`` of shape ``(n_voxels, n_beamlets)`` in
    Gy per incident particle, or a ``(dose, variance)`` pair when ``variance``
    is true. Voxels below ``rel_dose_threshold`` of the beamlet maximum are
    dropped from the column.

    ``progress`` is called with the fraction finished, in [0, 1], once per
    batch and once per beamlet. Returning ``False`` stops the calculation and
    raises ``KeyboardInterrupt``; Ctrl-C does the same.
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

    Returns ``(dose, uncertainty)``, both cubes shaped like the phantom. The
    dose is in Gy per incident fluence unless ``output_dose`` is false, in
    which case it is the mean deposited energy. The uncertainty is relative,
    and 0.9999999 wherever nothing was deposited.

    ``progress`` works as it does for :func:`calc_dij`, called once per batch.
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

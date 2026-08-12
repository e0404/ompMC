"""Shared fixtures for the ompMC Python tests."""

import struct
from pathlib import Path

import numpy as np
import pytest

import ompmc

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "ucodes" / "omc_matrad" / "test_fixture.mat"


@pytest.fixture
def water_phantom():
    """A small block of water, 0.5 cm voxels, with the beam entering along z."""
    n = 16
    lateral = np.linspace(-4.0, 4.0, n + 1)
    depth = np.linspace(0.0, 8.0, n + 1)

    return ompmc.Geometry(
        lateral, lateral, depth,
        ["H2O521ICRU"],
        np.full((n, n, n), 1.0, order="F"),
        np.ones((n, n, n), dtype=np.int32, order="F"),
    )


@pytest.fixture
def water_physics():
    """PEGS data matching the water phantom, with a low enough electron cutoff."""
    return ompmc.Physics(
        pegs_file=ompmc.data_path() / "pegs4" / "521icru.pegs4dat",
        global_ecut=0.521,
        global_pcut=0.01,
    )


@pytest.fixture
def phsp_beam(tmp_path):
    """An IAEA phase space pair, written here rather than committed.

    Ten 6 MeV photons a centimetre above the `water_phantom` block, spread
    over the middle of it and heading straight in along +z. Every variable is
    stored, which makes the record 29 bytes: the type byte, the energy, then
    x, y, z, u, v and the weight. W is never stored in this format -- it is
    reconstructed from u and v, with the sign of the type byte -- so its
    flag adds no bytes.

    Returns
    -------
    str
        The base name of the pair, which is what
        :class:`ompmc.PhaseSpaceSource` takes.
    """
    n = 10
    record_length = 1 + 4 + 4*5 + 4

    records = bytearray()
    for i in range(n):
        records.append(1)                       # a photon, w > 0
        records += struct.pack(
            "<7f",
            -6.0,                               # negative: opens a history
            -2.0 + 4.0*(i % 5)/4.0,             # x
            -2.0 + 4.0*(i % 3)/2.0,             # y
            -1.0,                               # z, a centimetre above
            0.0, 0.0,                           # u, v, so w is +1
            1.0,                                # weight
        )

    assert len(records) == n*record_length

    stem = tmp_path / "beam"
    stem.with_suffix(".IAEAphsp").write_bytes(bytes(records))
    stem.with_suffix(".IAEAheader").write_text(f"""$IAEA_INDEX:
0
// Written by the ompMC Python tests. Not a recording of anything.

$FILE_TYPE:
0

$CHECKSUM:
{n*record_length}

$RECORD_CONTENTS:
1     // X is stored ?
1     // Y is stored ?
1     // Z is stored ?
1     // U is stored ?
1     // V is stored ?
1     // W is stored ?
1     // Weight is stored ?
0     // Extra floats stored ?
0     // Extra longs stored ?

$RECORD_LENGTH:
{record_length}

$BYTE_ORDER:
1234

$ORIG_HISTORIES:
{n}

$PARTICLES:
{n}

$PHOTONS:
{n}
""")

    return str(stem)


@pytest.fixture(scope="session")
def matrad_fixture():
    """The inputs captured from matRad that the MEX smoke test also uses."""
    scipy_io = pytest.importorskip("scipy.io")

    if not FIXTURE.is_file():
        pytest.skip(f"{FIXTURE} is not in this checkout")

    raw = scipy_io.loadmat(FIXTURE, struct_as_record=False, squeeze_me=True)

    geo = raw["mcGeo"]
    src = raw["mcSrc"]
    opt = raw["mcOpt"]

    geometry = ompmc.Geometry(
        np.asarray(geo.xBounds, dtype=np.float64).ravel(),
        np.asarray(geo.yBounds, dtype=np.float64).ravel(),
        np.asarray(geo.zBounds, dtype=np.float64).ravel(),
        # matRad's material table is one row per medium: name, then the CT
        # number and density range it covers. Only the names are ompMC's
        # business, which is what the MEX file reads too.
        [str(row[0]).strip() for row in np.asarray(geo.material)],
        np.asfortranarray(raw["cubeRho"], dtype=np.float64),
        np.asfortranarray(raw["cubeMatIx"].astype(np.int32)),
    )

    def triples(x, y, z):
        return np.column_stack([np.atleast_1d(np.asarray(v, dtype=np.float64))
                                for v in (x, y, z)])

    source = ompmc.BeamletSource(
        # matRad counts beams from 1
        np.atleast_1d(np.asarray(src.iBeam, dtype=np.int32)).ravel() - 1,
        triples(src.xSource, src.ySource, src.zSource),
        triples(src.xCorner, src.yCorner, src.zCorner),
        triples(src.xSide1, src.ySide1, src.zSide1),
        triples(src.xSide2, src.ySide2, src.zSide2),
    )

    physics = ompmc.Physics(
        pegs_file=ompmc.data_path() / "pegs4" / "700icru.pegs4dat",
        global_ecut=float(opt.global_ecut),
        global_pcut=float(opt.global_pcut),
        n_split=int(opt.nSplit),
        seeds=tuple(int(s) for s in np.asarray(opt.randomSeeds).ravel()),
    )

    return {
        "geometry": geometry,
        "source": source,
        "physics": physics,
        "spectrum": ompmc.Spectrum.from_file(
            ompmc.data_path() / "spectra" / "mohan6.spectrum"),
        "n_histories": int(opt.nHistories),
        "n_batches": int(opt.nBatches),
        "charge": int(opt.charge),
        "rel_dose_threshold": float(opt.relDoseThreshold),
        "gaussian_source": str(opt.sourceGeometry).strip() == "gaussian",
        "source_width": float(opt.sourceGaussianWidth),
    }

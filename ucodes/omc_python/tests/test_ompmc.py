"""Tests for the ompMC Python interface.

Only structural and physical properties are checked. ompMC seeds its random
number generator per history, but threads accumulate energy into a voxel in
whatever order they finish, so the last bits of a dose are not reproducible
across runs and nothing here compares against stored numbers -- except
test_matches_mex, which holds the two interfaces against each other.
"""

from pathlib import Path

import numpy as np
import pytest

import ompmc


def test_version_and_data():
    assert ompmc.__version__.count(".") == 2
    root = ompmc.data_path()
    assert (root / "data" / "msnew.data").is_file()
    assert (root / "pegs4" / "700icru.pegs4dat").is_file()
    assert (root / "spectra" / "mohan6.spectrum").is_file()


class TestGeometryValidation:
    """The cube layout has to be checked up front: a C ordered cube would be a
    silently transposed phantom rather than an error."""

    def make(self, **overrides):
        n = 4
        bounds = np.linspace(-1.0, 1.0, n + 1)
        kwargs = dict(
            x_bounds=bounds, y_bounds=bounds, z_bounds=bounds,
            materials=["H2O521ICRU"],
            density=np.full((n, n, n), 1.0, order="F"),
            material=np.ones((n, n, n), dtype=np.int32, order="F"),
        )
        kwargs.update(overrides)
        return ompmc.Geometry(**kwargs)

    def test_accepts_a_valid_phantom(self):
        geometry = self.make()
        assert geometry.shape == (4, 4, 4)
        assert geometry.n_voxels == 64

    def test_rejects_c_ordered_cubes(self):
        with pytest.raises(ValueError, match="Fortran"):
            self.make(density=np.full((4, 4, 4), 1.0))

    def test_rejects_wrong_dtype(self):
        with pytest.raises(TypeError, match="int32"):
            self.make(material=np.ones((4, 4, 4), dtype=np.int64, order="F"))

    def test_rejects_material_index_out_of_range(self):
        material = np.ones((4, 4, 4), dtype=np.int32, order="F")
        material[0, 0, 0] = 5
        with pytest.raises(ValueError, match="material indices"):
            self.make(material=material)

    def test_rejects_mismatched_bounds(self):
        with pytest.raises(ValueError, match="boundaries describe"):
            self.make(z_bounds=np.linspace(-1.0, 1.0, 6))

    def test_rejects_descending_bounds(self):
        with pytest.raises(ValueError, match="ascending"):
            self.make(x_bounds=np.array([1.0, 0.5, 0.0, -0.5, -1.0]))


class TestSpectrumValidation:

    def test_from_histogram(self):
        spectrum = ompmc.Spectrum.from_histogram([1.0, 2.0, 3.0],
                                                 [0.2, 0.5, 0.3])
        assert spectrum._payload["mode"] == 0

    def test_rejects_descending_energies(self):
        with pytest.raises(ValueError, match="ascending"):
            ompmc.Spectrum.from_histogram([3.0, 2.0, 1.0], [1.0, 1.0, 1.0])

    def test_rejects_negative_fluence(self):
        with pytest.raises(ValueError, match="non-negative"):
            ompmc.Spectrum.from_histogram([1.0, 2.0], [1.0, -1.0])

    def test_rejects_empty_spectrum(self):
        with pytest.raises(ValueError, match="non-empty"):
            ompmc.Spectrum.from_histogram([], [])

    def test_rejects_nonpositive_mono_energy(self):
        with pytest.raises(ValueError, match="positive"):
            ompmc.Spectrum.monoenergetic(0.0)

    def test_rejects_missing_file(self):
        with pytest.raises(FileNotFoundError):
            ompmc.Spectrum.from_file("no_such.spectrum")


class TestCalcCube:

    def test_deposits_dose_with_a_buildup_region(self, water_phantom,
                                                 water_physics):
        source = ompmc.CollimatedSource(ssd=100.0, x_min=-2.0, x_max=2.0,
                                        y_min=-2.0, y_max=2.0)

        dose, uncertainty = ompmc.calc_cube(
            water_phantom, source, ompmc.Spectrum.monoenergetic(6.0),
            water_physics, n_histories=4000, n_batches=4)

        assert dose.shape == water_phantom.shape
        assert uncertainty.shape == water_phantom.shape
        assert np.all(np.isfinite(dose)) and np.all(dose >= 0.0)
        assert dose.max() > 0.0

        # 6 MeV photons deposit little at the surface and build up over the
        # first centimetres; the entrance plane must be well below the peak.
        depth = dose.sum(axis=(0, 1))
        assert depth[0] < 0.7*depth.max()

        # The beam is 4 cm wide in a 8 cm phantom, so the corners stay cold
        assert dose[0, 0, :].max() < 0.1*dose.max()

        # Voxels that received nothing carry the .3ddose convention
        assert uncertainty[dose == 0.0].min() == pytest.approx(0.9999999)

    def test_mean_energy_output(self, water_phantom, water_physics):
        source = ompmc.CollimatedSource(ssd=100.0, x_min=-2.0, x_max=2.0,
                                        y_min=-2.0, y_max=2.0)

        energy, _ = ompmc.calc_cube(
            water_phantom, source, ompmc.Spectrum.monoenergetic(6.0),
            water_physics, n_histories=2000, n_batches=2, output_dose=False)

        assert energy.max() > 0.0

    def test_rejects_a_single_batch(self, water_phantom, water_physics):
        source = ompmc.CollimatedSource(ssd=100.0, x_min=-1.0, x_max=1.0,
                                        y_min=-1.0, y_max=1.0)
        with pytest.raises(ValueError, match="at least 2"):
            ompmc.calc_cube(water_phantom, source, n_batches=1)


class TestCalcDij:

    def test_matches_the_matrad_fixture_structurally(self, matrad_fixture):
        f = matrad_fixture

        dij = ompmc.calc_dij(
            f["geometry"], f["source"], f["spectrum"], f["physics"],
            n_histories=f["n_histories"], n_batches=f["n_batches"],
            charge=f["charge"], rel_dose_threshold=f["rel_dose_threshold"],
            gaussian_source=f["gaussian_source"],
            source_width=f["source_width"])

        n_voxels = f["geometry"].n_voxels
        assert dij.shape == (n_voxels, f["source"].n_beamlets)
        assert dij.nnz > 0

        dose = dij.data
        assert np.all(np.isfinite(dose)) and np.all(dose >= 0.0)

        # Every beamlet has to deposit something
        per_beamlet = np.asarray(dij.sum(axis=0)).ravel()
        assert np.all(per_beamlet > 0.0)

        # A CSC matrix holds ascending row indices within each column
        for k in range(dij.shape[1]):
            rows = dij.indices[dij.indptr[k]:dij.indptr[k + 1]]
            assert np.all(np.diff(rows) > 0)

    def test_variance_output(self, matrad_fixture):
        f = matrad_fixture

        dij, variance = ompmc.calc_dij(
            f["geometry"], f["source"], f["spectrum"], f["physics"],
            n_histories=2000, n_batches=4, variance=True)

        assert variance.shape == dij.shape
        assert variance.nnz == dij.nnz
        assert np.all(variance.data >= 0.0)

    def test_electrons_deposit_somewhere_else_than_photons(self, matrad_fixture):
        f = matrad_fixture

        photons = ompmc.calc_dij(f["geometry"], f["source"], f["spectrum"],
                                 f["physics"], n_histories=2000, n_batches=2)
        electrons = ompmc.calc_dij(f["geometry"], f["source"], f["spectrum"],
                                   f["physics"], n_histories=2000, n_batches=2,
                                   charge=-1)

        shared = (photons.astype(bool).multiply(electrons.astype(bool))).nnz
        union = photons.nnz + electrons.nnz - shared
        assert shared/union < 0.5

    def test_rejects_a_bad_charge(self, matrad_fixture):
        f = matrad_fixture
        with pytest.raises(ValueError, match="charge"):
            ompmc.calc_dij(f["geometry"], f["source"], charge=2)


class TestCalcForward:
    """The forward mode has to be dij @ w, computed directly."""

    def weights(self, n):
        """Every other beamlet open, with one that only partly transmits."""
        w = np.zeros(n)
        w[::2] = 1.0
        w[1] = 0.25
        return w

    def test_matches_dij_times_weights(self, matrad_fixture):
        f = matrad_fixture
        n = f["source"].n_beamlets
        w = self.weights(n)

        # The reference must not be pruned -- relDoseThreshold drops low dose
        # voxels the forward cube keeps -- and the forward run needs n times
        # the histories, because it counts the whole calculation rather than
        # one beamlet.
        dij = ompmc.calc_dij(
            f["geometry"], f["source"], f["spectrum"], f["physics"],
            n_histories=2000, n_batches=5, rel_dose_threshold=0.0)
        reference = (dij @ w).reshape(f["geometry"].shape, order="F")

        dose, uncertainty = ompmc.calc_forward(
            f["geometry"], f["source"], w, f["spectrum"], f["physics"],
            n_histories=2000*n, n_batches=5)

        assert dose.shape == f["geometry"].shape
        assert np.all(np.isfinite(dose)) and np.all(dose >= 0.0)
        assert dose.max() > 0.0

        # The total is the compiler and scheduling independent part, and the
        # tightest thing to compare. The two runs draw different random
        # streams, so what is left is the statistical spread of the run.
        total = abs(dose.sum() - reference.sum())/reference.sum()
        assert total < 0.02, f"total dose differs from dij @ w by {total:.3g}"

        # Totals agreeing would survive a cube with permuted axes, so check
        # where the dose actually went as well. Loose on purpose: a single
        # profile bin is far noisier than the total.
        for axis in range(3):
            others = tuple(a for a in range(3) if a != axis)
            got = dose.sum(axis=others)
            want = reference.sum(axis=others)

            hot = want > 0.05*want.max()
            profile = np.abs(got[hot] - want[hot])/want[hot]
            assert profile.max() < 0.15, (
                f"the profile along axis {axis} differs from dij @ w by "
                f"{profile.max():.3g}")

        # Same convention omc_dosxyz writes into a .3ddose
        assert uncertainty.shape == dose.shape
        assert np.all((uncertainty >= 0.0) & (uncertainty <= 1.0))
        assert uncertainty[dose == 0.0].min() == pytest.approx(0.9999999)

    def test_weights_scale_the_dose(self, matrad_fixture):
        """The dose is in Gy for the weights given, not per history."""
        f = matrad_fixture
        w = self.weights(f["source"].n_beamlets)

        common = dict(n_histories=4000, n_batches=4)

        single, _ = ompmc.calc_forward(f["geometry"], f["source"], w,
                                       f["spectrum"], f["physics"], **common)
        double, _ = ompmc.calc_forward(f["geometry"], f["source"], 2.0*w,
                                       f["spectrum"], f["physics"], **common)

        # Doubling every weight doubles the fluence, and the histories are
        # handed out in the same proportions, so this holds far tighter than
        # the statistics: the two runs sample identically.
        assert double.sum()/single.sum() == pytest.approx(2.0, rel=1e-12)

    def test_progress_and_cancellation(self, matrad_fixture):
        f = matrad_fixture
        w = np.ones(f["source"].n_beamlets)
        seen = []

        ompmc.calc_forward(f["geometry"], f["source"], w, f["spectrum"],
                           f["physics"], n_histories=2000, n_batches=4,
                           progress=lambda p: seen.append(p))

        assert seen == sorted(seen)
        assert seen[-1] == pytest.approx(1.0)

        with pytest.raises(KeyboardInterrupt):
            ompmc.calc_forward(f["geometry"], f["source"], w, f["spectrum"],
                               f["physics"], n_histories=2000, n_batches=4,
                               progress=lambda p: False)

    @pytest.mark.parametrize("bad, match", [
        ("length", "beamlets"),
        ("negative", "negative"),
        ("zero", "every weight is zero"),
        ("nan", "finite"),
        ("infinite", "finite"),
        ("overflow", "finite"),
    ])
    def test_rejects_unusable_weights(self, matrad_fixture, bad, match):
        f = matrad_fixture
        n = f["source"].n_beamlets

        w = {
            "length": np.ones(n + 1),
            "negative": -np.ones(n),
            "zero": np.zeros(n),
            "nan": np.full(n, np.nan),
            "infinite": np.full(n, np.inf),
            "overflow": np.full(n, np.finfo(np.float64).max),
        }[bad]

        with pytest.raises(ValueError, match=match):
            ompmc.calc_forward(f["geometry"], f["source"], w)


class TestPhaseSpaceSourceValidation:

    def test_defaults_to_the_identity_transform(self):
        source = ompmc.PhaseSpaceSource("beam")
        assert np.array_equal(source.rotation, np.eye(3))
        assert np.array_equal(source.translation, np.zeros(3))
        assert source._payload["order"] == 0

    def test_accepts_random_order(self):
        assert ompmc.PhaseSpaceSource("beam", order="random")._payload[
            "order"] == 1

    def test_rejects_an_unknown_order(self):
        with pytest.raises(ValueError, match="replay"):
            ompmc.PhaseSpaceSource("beam", order="shuffle")

    def test_rejects_a_negative_start(self):
        with pytest.raises(ValueError, match="negative"):
            ompmc.PhaseSpaceSource("beam", first=-1)

    def test_rejects_a_matrix_that_is_not_a_rotation(self):
        # A matrix that is not a rotation would stretch the directions it
        # turns, and those have to stay unit vectors.
        with pytest.raises(ValueError, match="determinant"):
            ompmc.PhaseSpaceSource("beam", rotation=2.0*np.eye(3))

    def test_rejects_a_misshaped_transform(self):
        with pytest.raises(ValueError, match=r"\(3, 3\)"):
            ompmc.PhaseSpaceSource("beam", rotation=np.eye(4))
        with pytest.raises(ValueError, match="expected 3"):
            ompmc.PhaseSpaceSource("beam", translation=[1.0, 2.0])

    def test_a_real_rotation_goes_through_row_major(self):
        # A quarter turn about z, so the payload has to read across rows.
        rotation = np.array([[0.0, -1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0]])
        source = ompmc.PhaseSpaceSource("beam", rotation=rotation,
                                        translation=[1.0, 2.0, 3.0])

        assert source._payload["rotation"] == [0.0, -1.0, 0.0,
                                               1.0, 0.0, 0.0,
                                               0.0, 0.0, 1.0]
        assert source._payload["translation"] == [1.0, 2.0, 3.0]


class TestApertureMaskValidation:

    def test_a_rectangle_is_one_open_cell(self):
        mask = ompmc.ApertureMask.rectangle(40.0, -2.0, 2.0, -3.0, 3.0)

        assert mask.shape == (1, 1)
        assert mask.transmission[0, 0] == 1.0
        assert (mask.x0, mask.y0, mask.dx, mask.dy) == (-2.0, -3.0, 4.0, 6.0)
        assert mask.outside == 0.0

    def test_rejects_a_rectangle_with_no_width(self):
        with pytest.raises(ValueError, match="positive"):
            ompmc.ApertureMask.rectangle(40.0, 2.0, 2.0, -3.0, 3.0)

    def test_rejects_transmission_outside_zero_to_one(self):
        # Above one would quietly multiply the dose rather than fail.
        with pytest.raises(ValueError, match="between 0 and 1"):
            ompmc.ApertureMask(z=1.0, x0=0.0, y0=0.0, dx=1.0, dy=1.0,
                               transmission=[[1.5]])
        with pytest.raises(ValueError, match="between 0 and 1"):
            ompmc.ApertureMask(z=1.0, x0=0.0, y0=0.0, dx=1.0, dy=1.0,
                               transmission=[[-0.5]])

    def test_rejects_cells_with_no_size(self):
        with pytest.raises(ValueError, match="positive"):
            ompmc.ApertureMask(z=1.0, x0=0.0, y0=0.0, dx=0.0, dy=1.0,
                               transmission=[[1.0]])

    def test_rejects_an_out_of_range_outside(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            ompmc.ApertureMask(z=1.0, x0=0.0, y0=0.0, dx=1.0, dy=1.0,
                               transmission=[[1.0]], outside=2.0)

    def test_rejects_a_grid_that_is_not_two_dimensional(self):
        with pytest.raises(ValueError, match="non-empty"):
            ompmc.ApertureMask(z=1.0, x0=0.0, y0=0.0, dx=1.0, dy=1.0,
                               transmission=[1.0, 1.0])

    def test_the_grid_keeps_x_first(self):
        mask = ompmc.ApertureMask(z=1.0, x0=0.0, y0=0.0, dx=1.0, dy=1.0,
                                  transmission=[[1.0, 0.0], [0.5, 0.0]])
        assert mask.shape == (2, 2)
        # Fortran order, so the flat layout has x running fastest, which is
        # what the core reads.
        assert list(mask._payload["transmission"].ravel(order="F")) == [
            1.0, 0.5, 0.0, 0.0]


class TestCalcForwardPhsp:
    """The phase space engine, driven from a file the test writes itself.

    Ten photons a centimetre above the water block, spread across the middle
    of it and heading straight in, which is enough to tell dose from no dose
    and a blocked beam from an open one.
    """

    def test_deposits_dose_with_a_buildup_region(self, phsp_beam,
                                                 water_phantom, water_physics):
        source = ompmc.PhaseSpaceSource(phsp_beam)

        dose, uncertainty, summary = ompmc.calc_forward_phsp(
            water_phantom, source, water_physics,
            n_histories=4000, n_batches=4)

        assert dose.shape == water_phantom.shape
        assert np.all(np.isfinite(dose)) and np.all(dose >= 0.0)
        assert dose.max() > 0.0

        assert summary.n_histories == 4000
        assert summary.n_started == 4000       # all of them are aimed in
        assert summary.n_blocked == 0
        assert 0.0 < summary.energy_fraction < 1.0

        # Photons build up over the first centimetres.
        depth = dose.sum(axis=(0, 1))
        assert depth[0] < 0.9*depth.max()

    def test_a_missing_file_is_an_exception(self, water_phantom,
                                            water_physics):
        with pytest.raises(RuntimeError):
            ompmc.calc_forward_phsp(
                water_phantom, ompmc.PhaseSpaceSource("no_such_phase_space"),
                water_physics, n_histories=100, n_batches=2)

    def test_a_transform_can_aim_the_beam_away(self, phsp_beam, water_phantom,
                                               water_physics):
        # A half turn about x turns the particles round, and the translation
        # puts them back above the tank afterwards -- the turn alone would
        # carry z = -1 to z = +1, which is inside it. Facing away from the
        # phantom from outside it, nothing starts and nothing is deposited.
        away = ompmc.PhaseSpaceSource(
            phsp_beam,
            rotation=np.array([[1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0],
                               [0.0, 0.0, -1.0]]),
            translation=[0.0, 0.0, -2.0])

        dose, _, summary = ompmc.calc_forward_phsp(
            water_phantom, away, water_physics, n_histories=400, n_batches=2)

        assert summary.n_started == 0
        assert dose.max() == 0.0

    def test_a_shut_collimator_stops_everything(self, phsp_beam,
                                                water_phantom, water_physics):
        shut = ompmc.ApertureMask(z=-0.5, x0=-50.0, y0=-50.0, dx=100.0,
                                  dy=100.0, transmission=[[0.0]], outside=0.0)

        dose, _, summary = ompmc.calc_forward_phsp(
            water_phantom, ompmc.PhaseSpaceSource(phsp_beam), water_physics,
            n_histories=400, n_batches=2, collimator=shut)

        assert summary.n_blocked == 400
        assert summary.n_started == 0
        assert dose.max() == 0.0

    def test_an_open_collimator_changes_nothing(self, phsp_beam,
                                                water_phantom, water_physics):
        wide = ompmc.ApertureMask.rectangle(-0.5, -50.0, 50.0, -50.0, 50.0)

        source = ompmc.PhaseSpaceSource(phsp_beam)
        kwargs = dict(n_histories=2000, n_batches=2)

        bare, _, bare_summary = ompmc.calc_forward_phsp(
            water_phantom, source, water_physics, **kwargs)
        through, _, open_summary = ompmc.calc_forward_phsp(
            water_phantom, source, water_physics, collimator=wide, **kwargs)

        assert open_summary.n_blocked == 0
        assert open_summary.n_started == bare_summary.n_started

        # An open mask draws no random numbers and changes no weight, so the
        # two runs are the same run; only the order threads accumulated in
        # can differ.
        assert through.sum() == pytest.approx(bare.sum(), rel=1e-9)

    def test_a_half_transmitting_collimator_halves_the_dose(
            self, phsp_beam, water_phantom, water_physics):
        half = ompmc.ApertureMask(z=-0.5, x0=-50.0, y0=-50.0, dx=100.0,
                                  dy=100.0, transmission=[[0.5]], outside=0.5)

        source = ompmc.PhaseSpaceSource(phsp_beam)
        kwargs = dict(n_histories=2000, n_batches=2)

        bare, _, _ = ompmc.calc_forward_phsp(
            water_phantom, source, water_physics, **kwargs)
        attenuated, _, summary = ompmc.calc_forward_phsp(
            water_phantom, source, water_physics, collimator=half, **kwargs)

        # Weight, not roulette: every history still transports.
        assert summary.n_blocked == 0
        assert attenuated.sum() == pytest.approx(0.5*bare.sum(), rel=1e-9)

    def test_roulette_stops_half_and_keeps_the_dose(self, phsp_beam,
                                                    water_phantom,
                                                    water_physics):
        half = ompmc.ApertureMask(z=-0.5, x0=-50.0, y0=-50.0, dx=100.0,
                                  dy=100.0, transmission=[[0.5]], outside=0.5,
                                  roulette=True)

        source = ompmc.PhaseSpaceSource(phsp_beam)
        kwargs = dict(n_histories=8000, n_batches=4)

        bare, _, _ = ompmc.calc_forward_phsp(
            water_phantom, source, water_physics, **kwargs)
        played, _, summary = ompmc.calc_forward_phsp(
            water_phantom, source, water_physics, collimator=half, **kwargs)

        # Where the saving is: half the histories never reach a shower.
        assert 0.45 < summary.n_blocked/summary.n_histories < 0.55
        assert summary.n_blocked + summary.n_started == summary.n_histories

        # And the same dose, up to the noise roulette adds.
        assert played.sum() == pytest.approx(0.5*bare.sum(), rel=0.05)

    def test_a_multi_cell_grid_keeps_its_orientation(self, phsp_beam,
                                                     water_phantom,
                                                     water_physics):
        """Two cells across x, the negative one open. If the grid reached the
        engine transposed or flipped, the dose would land on the wrong side of
        the tank -- which nothing about a single cell mask can tell you."""
        halved = ompmc.ApertureMask(
            z=-0.5, x0=-50.0, y0=-50.0, dx=50.0, dy=100.0,
            transmission=[[1.0], [0.0]],        # (nx, ny) = (2, 1)
            outside=0.0)

        assert halved.shape == (2, 1)

        dose, _, summary = ompmc.calc_forward_phsp(
            water_phantom, ompmc.PhaseSpaceSource(phsp_beam), water_physics,
            n_histories=4000, n_batches=4, collimator=halved)

        # The fixture puts particles at x of -2, -1, 0, 1 and 2 in turn, so
        # the four in ten at a negative x are the ones that get through.
        assert summary.n_blocked/summary.n_histories == pytest.approx(0.6,
                                                                      abs=0.02)

        n = water_phantom.shape[0]
        assert dose[:n//2].sum() > 10.0*dose[n//2:].sum()

    def test_random_order_also_works(self, phsp_beam, water_phantom,
                                     water_physics):
        source = ompmc.PhaseSpaceSource(phsp_beam, order="random")

        dose, _, summary = ompmc.calc_forward_phsp(
            water_phantom, source, water_physics, n_histories=2000,
            n_batches=2)

        assert summary.n_started == 2000
        assert dose.max() > 0.0

    def test_rejects_a_single_batch(self, phsp_beam, water_phantom):
        with pytest.raises(ValueError, match="at least 2"):
            ompmc.calc_forward_phsp(water_phantom,
                                    ompmc.PhaseSpaceSource(phsp_beam),
                                    n_batches=1)


def test_calc_forward_takes_a_collimator(matrad_fixture):
    """Beamlets collimate through their weights, but a mask still applies --
    for a block cutting across them, or a leaf that transmits."""
    f = matrad_fixture
    weights = np.ones(f["source"].n_beamlets)

    # A plane the beamlets all cross, shut everywhere.
    shut = ompmc.ApertureMask(z=0.0, x0=-500.0, y0=-500.0, dx=1000.0,
                              dy=1000.0, transmission=[[0.0]], outside=0.0)

    dose, _ = ompmc.calc_forward(
        f["geometry"], f["source"], weights, f["spectrum"], f["physics"],
        n_histories=2000, n_batches=2, collimator=shut)

    assert dose.max() == 0.0


class TestProgress:

    def test_reports_monotonic_progress(self, matrad_fixture):
        f = matrad_fixture
        seen = []

        ompmc.calc_dij(f["geometry"], f["source"], f["spectrum"], f["physics"],
                       n_histories=1000, n_batches=2,
                       progress=lambda p: seen.append(p))

        assert seen, "the progress callback was never called"
        assert all(0.0 <= p <= 1.0 for p in seen)
        assert seen == sorted(seen)
        assert seen[-1] == pytest.approx(1.0)

    def test_returning_false_stops_the_calculation(self, matrad_fixture):
        f = matrad_fixture
        calls = []

        def stop_after_two(fraction):
            calls.append(fraction)
            return len(calls) < 2

        with pytest.raises(KeyboardInterrupt):
            ompmc.calc_dij(f["geometry"], f["source"], f["spectrum"],
                           f["physics"], n_histories=1000, n_batches=2,
                           progress=stop_after_two)

        # It stopped early rather than running every beamlet
        assert len(calls) < 2*f["source"].n_beamlets

    def test_an_exception_in_the_callback_surfaces(self, matrad_fixture):
        f = matrad_fixture

        def explode(fraction):
            raise ZeroDivisionError("from the callback")

        with pytest.raises(ZeroDivisionError, match="from the callback"):
            ompmc.calc_dij(f["geometry"], f["source"], f["spectrum"],
                           f["physics"], n_histories=1000, n_batches=2,
                           progress=explode)


def test_engine_errors_become_python_exceptions(water_phantom):
    """omcFail() in the C code has to arrive as an exception, not a crash."""
    physics = ompmc.Physics(pegs_file="no_such_file.pegs4dat")
    source = ompmc.CollimatedSource(ssd=100.0, x_min=-1.0, x_max=1.0,
                                    y_min=-1.0, y_max=1.0)

    with pytest.raises(RuntimeError):
        ompmc.calc_cube(water_phantom, source,
                        ompmc.Spectrum.monoenergetic(1.0), physics,
                        n_histories=100, n_batches=2)


REFERENCE = Path(__file__).resolve().parent / "mex_reference.mat"


@pytest.mark.mex
@pytest.mark.skipif(not REFERENCE.is_file(),
                    reason="no MEX reference; regenerate it with "
                           "ucodes/omc_matrad/export_reference.m")
def test_matches_mex(matrad_fixture):
    """The Python and MATLAB interfaces must agree on the same inputs.

    Both drive the same engine with the same per-history random streams, so
    the only thing that may differ is the order in which threads accumulated
    energy into a voxel -- which changes the last bits of a dose and nothing
    else. The sparsity pattern has to be identical.

    This is the test that keeps the two interfaces from drifting apart, and
    the reason the engines were pulled out of the user codes at all.

    Marked "mex" so it can be deselected: the stored reference comes from one
    particular MEX build on one machine, so the sparsity pattern is only
    guaranteed against a build sharing its math library. The wheel CI, which
    builds on four other toolchains, runs -m "not mex" for that reason.
    """
    scipy_io = pytest.importorskip("scipy.io")

    f = matrad_fixture
    reference = scipy_io.loadmat(REFERENCE)

    dij = ompmc.calc_dij(
        f["geometry"], f["source"], f["spectrum"], f["physics"],
        n_histories=f["n_histories"], n_batches=f["n_batches"],
        charge=f["charge"], rel_dose_threshold=f["rel_dose_threshold"],
        gaussian_source=f["gaussian_source"], source_width=f["source_width"])

    assert dij.shape == tuple(np.asarray(reference["shape"]).ravel())
    np.testing.assert_array_equal(dij.indptr,
                                  np.asarray(reference["indptr"]).ravel())
    np.testing.assert_array_equal(dij.indices,
                                  np.asarray(reference["indices"]).ravel())

    values = np.asarray(reference["values"]).ravel()

    # The total is the compiler independent part: it agrees to machine
    # precision no matter how the two were built.
    total = abs(dij.data.sum() - values.sum())/values.sum()
    assert total < 1e-12, f"total dose differs by {total:.3e}"

    # Individual voxels are looser, because the two builds may not share a
    # math library. Measured on this fixture: 6e-16 when the MEX file and the
    # extension are both built with MSVC, 2e-10 when the MEX is MinGW built
    # and the extension MSVC built -- last-ulp differences in log/exp move
    # interaction points a little and redistribute dose between voxels
    # without changing the total.
    relative = np.abs(dij.data - values)/np.abs(values)
    assert relative.max() < 1e-8, (
        f"largest relative difference to the MEX result is {relative.max():.3e}")

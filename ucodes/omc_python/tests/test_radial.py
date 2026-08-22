"""The r-z dose calculation: the cylinder, the beams and what comes back.

The physics assertions here are the broad shapes, the same ones the C engine
test checks -- a photon beam builds up and falls off, dose falls away from the
axis. What this file is really watching is the Python side of it: that the
validation catches what it should before anything is transported, that the
result comes back shaped and ordered the way it says, and that a process can
run a cylinder and a cube one after the other without the first leaking into
the second.
"""

import numpy as np
import pytest

import ompmc


@pytest.fixture
def water_cylinder():
    """A water cylinder, 5 cm across in 10 rings and 10 cm deep in 20 slabs."""
    return ompmc.CylinderGeometry(
        r_bounds=np.linspace(0.0, 5.0, 11),
        z_bounds=np.linspace(0.0, 10.0, 21),
        material="H2O521ICRU",
        density=1.0,
    )


class TestCylinderGeometryValidation:

    def test_reports_its_shape(self, water_cylinder):
        assert water_cylinder.shape == (10, 20)
        assert water_cylinder.n_regions == 200

    def test_rejects_a_hole_in_the_middle(self):
        with pytest.raises(ValueError, match="axis"):
            ompmc.CylinderGeometry(
                r_bounds=[0.5, 1.0, 2.0], z_bounds=[0.0, 1.0],
                material="H2O521ICRU")

    def test_rejects_bounds_that_do_not_ascend(self):
        with pytest.raises(ValueError, match="ascending"):
            ompmc.CylinderGeometry(
                r_bounds=[0.0, 2.0, 1.0], z_bounds=[0.0, 1.0],
                material="H2O521ICRU")

        with pytest.raises(ValueError, match="ascending"):
            ompmc.CylinderGeometry(
                r_bounds=[0.0, 1.0], z_bounds=[0.0, 2.0, 1.0],
                material="H2O521ICRU")

    def test_rejects_a_geometry_with_no_bins(self):
        with pytest.raises(ValueError, match="two boundaries"):
            ompmc.CylinderGeometry(
                r_bounds=[0.0], z_bounds=[0.0, 1.0], material="H2O521ICRU")

    def test_rejects_a_density_that_is_not_positive(self):
        with pytest.raises(ValueError, match="density"):
            ompmc.CylinderGeometry(
                r_bounds=[0.0, 1.0], z_bounds=[0.0, 1.0],
                material="H2O521ICRU", density=0.0)

    def test_rejects_an_empty_material(self):
        with pytest.raises(ValueError, match="material"):
            ompmc.CylinderGeometry(
                r_bounds=[0.0, 1.0], z_bounds=[0.0, 1.0], material="")

    def test_a_density_of_none_means_the_pegs_one(self):
        geometry = ompmc.CylinderGeometry(
            r_bounds=[0.0, 1.0], z_bounds=[0.0, 1.0], material="H2O521ICRU")
        assert geometry.density is None


class TestPencilBeamSourceValidation:

    def test_a_bare_source_is_a_parallel_pencil(self):
        source = ompmc.PencilBeamSource()
        assert source._payload["kind"] == 0

    def test_an_ssd_makes_it_a_point_source(self):
        source = ompmc.PencilBeamSource(ssd=100.0)
        assert source._payload["kind"] == 1
        assert source._payload["ssd"] == 100.0
        # 0 is how the core spells "the whole front face"
        assert source._payload["field_radius"] == 0.0

    def test_rejects_an_ssd_that_is_not_positive(self):
        with pytest.raises(ValueError, match="ssd"):
            ompmc.PencilBeamSource(ssd=-1.0)

        with pytest.raises(ValueError, match="ssd"):
            ompmc.PencilBeamSource(ssd=0.0)

    def test_rejects_a_field_radius_without_a_source_to_have_it(self):
        with pytest.raises(ValueError, match="field_radius"):
            ompmc.PencilBeamSource(field_radius=3.0)

    def test_rejects_a_field_radius_that_is_not_positive(self):
        with pytest.raises(ValueError, match="field_radius"):
            ompmc.PencilBeamSource(ssd=100.0, field_radius=-2.0)

    def test_a_bare_source_has_no_spread(self):
        payload = ompmc.PencilBeamSource()._payload
        assert payload["spot_sigma"] == 0.0
        assert payload["divergence_sigma"] == 0.0

    def test_carries_the_spreads_through(self):
        payload = ompmc.PencilBeamSource(
            spot_sigma=0.15, divergence_sigma=0.01)._payload
        assert payload["spot_sigma"] == 0.15
        assert payload["divergence_sigma"] == 0.01
        assert payload["kind"] == 0        # still a parallel pencil

    @pytest.mark.parametrize("name", ["spot_sigma", "divergence_sigma"])
    def test_rejects_a_spread_that_is_not_positive(self, name):
        with pytest.raises(ValueError, match=name):
            ompmc.PencilBeamSource(**{name: -0.1})

        with pytest.raises(ValueError, match=name):
            ompmc.PencilBeamSource(**{name: 0.0})

    def test_a_bare_source_is_uncorrelated(self):
        assert ompmc.PencilBeamSource()._payload["correlation"] == 0.0

    def test_carries_a_correlation_through(self):
        payload = ompmc.PencilBeamSource(
            spot_sigma=0.5, divergence_sigma=0.06,
            correlation=-0.6)._payload
        assert payload["correlation"] == -0.6

    @pytest.mark.parametrize("rho", [-1.5, 1.5])
    def test_rejects_a_correlation_outside_the_unit_interval(self, rho):
        with pytest.raises(ValueError, match="correlation"):
            ompmc.PencilBeamSource(spot_sigma=0.5, divergence_sigma=0.06,
                                   correlation=rho)

    @pytest.mark.parametrize("kwargs", [
        {},                                 # neither width nor divergence
        {"spot_sigma": 0.5},                # nothing to point
        {"divergence_sigma": 0.06},         # nothing to point it from
    ])
    def test_rejects_a_correlation_with_nothing_to_relate(self, kwargs):
        # The core would quietly ignore it. Here it is far more likely to be
        # a beam that was meant to have a waist and silently did not.
        with pytest.raises(ValueError, match="correlation"):
            ompmc.PencilBeamSource(correlation=-0.6, **kwargs)

    def test_the_ends_of_the_interval_are_allowed(self):
        for rho in (-1.0, 1.0):
            ompmc.PencilBeamSource(spot_sigma=0.5, divergence_sigma=0.06,
                                   correlation=rho)


class TestPencilBeamWaist:

    def test_a_focused_beam_has_its_waist_where_it_was_asked_for(self):
        beam = ompmc.PencilBeamSource.focused(0.1, 0.02, 5.0)

        sigma, depth = beam.waist
        assert sigma == pytest.approx(0.1)
        assert depth == pytest.approx(5.0)

    def test_it_is_wider_at_the_surface_than_at_the_waist(self):
        beam = ompmc.PencilBeamSource.focused(0.1, 0.02, 5.0)

        # sqrt(0.1^2 + (5*0.02)^2), the drift added in quadrature
        assert beam.spot_sigma == pytest.approx(np.hypot(0.1, 0.1))
        assert beam.spot_sigma > beam.waist[0]
        assert beam.correlation < 0.0        # it converges

    @pytest.mark.parametrize("depth", [-4.0, 0.0, 2.5, 20.0])
    def test_the_round_trip_holds_wherever_the_waist_is(self, depth):
        beam = ompmc.PencilBeamSource.focused(0.2, 0.03, depth)

        sigma, back = beam.waist
        assert sigma == pytest.approx(0.2)
        assert back == pytest.approx(depth)
        assert -1.0 <= beam.correlation <= 1.0

    def test_a_waist_on_the_face_is_an_ordinary_uncorrelated_beam(self):
        beam = ompmc.PencilBeamSource.focused(0.2, 0.03, 0.0)

        assert beam.correlation == 0.0
        assert beam.spot_sigma == pytest.approx(0.2)

    def test_an_uncorrelated_beam_is_narrowest_where_it_starts(self):
        beam = ompmc.PencilBeamSource(spot_sigma=0.4, divergence_sigma=0.02)
        assert beam.waist == (0.4, 0.0)

    def test_a_beam_with_no_spread_has_a_waist_of_none(self):
        assert ompmc.PencilBeamSource().waist == (0.0, 0.0)

    def test_it_passes_the_rest_of_the_beam_on(self):
        beam = ompmc.PencilBeamSource.focused(0.1, 0.02, 5.0,
                                              ssd=100.0, field_radius=4.0)
        assert beam._payload["kind"] == 1
        assert beam._payload["ssd"] == 100.0
        assert beam._payload["field_radius"] == 4.0

    @pytest.mark.parametrize("args", [
        (0.0, 0.02, 5.0), (-0.1, 0.02, 5.0),        # waist_sigma
        (0.1, 0.0, 5.0), (0.1, -0.02, 5.0),         # divergence_sigma
    ])
    def test_rejects_a_beam_it_cannot_focus(self, args):
        with pytest.raises(ValueError):
            ompmc.PencilBeamSource.focused(*args)


class TestCalcRadialValidation:

    def test_rejects_too_few_batches(self, water_cylinder):
        with pytest.raises(ValueError):
            ompmc.calc_radial(water_cylinder, ompmc.PencilBeamSource(),
                              n_histories=100, n_batches=1)

    def test_rejects_a_charge_it_cannot_transport(self, water_cylinder):
        with pytest.raises(ValueError):
            ompmc.calc_radial(water_cylinder, ompmc.PencilBeamSource(),
                              n_histories=100, n_batches=2, charge=5)

    def test_rejects_a_spectrum_given_with_a_phase_space(self, water_cylinder,
                                                         phsp_beam):
        with pytest.raises(ValueError, match="no spectrum"):
            ompmc.calc_radial(
                water_cylinder, ompmc.PhaseSpaceSource(phsp_beam),
                spectrum=ompmc.Spectrum.monoenergetic(6.0),
                n_histories=100, n_batches=2)


class TestCalcRadial:

    def test_returns_rings_by_slabs(self, water_cylinder, water_physics):
        dose, uncertainty, summary = ompmc.calc_radial(
            water_cylinder, ompmc.PencilBeamSource(),
            ompmc.Spectrum.monoenergetic(6.0), water_physics,
            n_histories=2000, n_batches=4)

        assert dose.shape == water_cylinder.shape
        assert uncertainty.shape == water_cylinder.shape
        assert dose.flags.f_contiguous

        assert summary.n_histories == 2000
        assert summary.n_started == 2000
        assert summary.n_blocked == 0
        assert 0.0 < summary.energy_fraction < 1.0

    def test_deposits_dose_with_a_buildup_region(self, water_cylinder,
                                                 water_physics):
        dose, _unc, _summary = ompmc.calc_radial(
            water_cylinder, ompmc.PencilBeamSource(),
            ompmc.Spectrum.monoenergetic(6.0), water_physics,
            n_histories=40_000, n_batches=4)

        on_axis = dose[0, :]
        kmax = int(np.argmax(on_axis))

        assert on_axis[kmax] > 0.0
        # Below the surface, and well before the back of the cylinder
        assert 0 < kmax < water_cylinder.shape[1]//2
        assert on_axis[-1] < on_axis[kmax]

    def test_dose_falls_away_from_the_axis(self, water_cylinder,
                                           water_physics):
        dose, _unc, _summary = ompmc.calc_radial(
            water_cylinder, ompmc.PencilBeamSource(),
            ompmc.Spectrum.monoenergetic(6.0), water_physics,
            n_histories=40_000, n_batches=4)

        kmax = int(np.argmax(dose[0, :]))

        assert dose[0, kmax] > dose[2, kmax] > dose[5, kmax]
        assert dose[-1, kmax] < 0.05*dose[0, kmax]

    def test_empty_regions_carry_the_sentinel(self, water_physics):
        # A metre of water and a beam too soft to cross it: the far slabs are
        # a dozen mean free paths down, so nothing reaches them and they have
        # to say so rather than report a confident zero. The cylinder is much
        # bigger than the one the other tests use because the default physics
        # splits photons, which fills a small phantom surprisingly well.
        deep = ompmc.CylinderGeometry(
            r_bounds=np.linspace(0.0, 50.0, 11),
            z_bounds=np.linspace(0.0, 100.0, 21),
            material="H2O521ICRU", density=1.0)

        dose, uncertainty, _summary = ompmc.calc_radial(
            deep, ompmc.PencilBeamSource(),
            ompmc.Spectrum.monoenergetic(0.1), water_physics,
            n_histories=400, n_batches=2)

        assert np.any(dose == 0.0)
        assert uncertainty[dose == 0.0].min() == pytest.approx(0.9999999)
        assert uncertainty[dose == 0.0].max() == pytest.approx(0.9999999)

        # And the ones that did get dose report a real uncertainty. Not a
        # smaller one: a region reached in only one of two batches has a
        # relative uncertainty of exactly 1, which is a number the statistics
        # arrived at rather than a stand-in for not having any.
        assert np.all(uncertainty[dose > 0.0] > 0.0)

    def test_a_gaussian_spot_broadens_the_dose(self, water_cylinder,
                                               water_physics):
        def entrance_profile(source):
            dose, _unc, _summary = ompmc.calc_radial(
                water_cylinder, source, ompmc.Spectrum.monoenergetic(6.0),
                water_physics, n_histories=20_000, n_batches=4,
                output_dose=False)      # energy, so ring volumes drop out
            entrance = dose[:, 0]
            return entrance[0]/entrance.sum()

        delta = entrance_profile(ompmc.PencilBeamSource())
        widened = entrance_profile(ompmc.PencilBeamSource(spot_sigma=1.5))

        # A delta pencil puts nearly everything in the innermost ring; a
        # centimetre and a half of spot spreads it over most of them.
        assert delta > 0.8
        assert widened < 0.3

    def test_a_spot_that_overruns_the_cylinder_loses_histories(
            self, water_cylinder, water_physics):
        # Those histories happened and still count towards the fluence, so
        # the summary is where they show rather than going quietly missing.
        _dose, _unc, summary = ompmc.calc_radial(
            water_cylinder, ompmc.PencilBeamSource(spot_sigma=4.0),
            ompmc.Spectrum.monoenergetic(6.0), water_physics,
            n_histories=4000, n_batches=4)

        assert summary.n_histories == 4000
        assert summary.n_started < 4000

    def test_a_divergent_beam_runs(self, water_cylinder, water_physics):
        dose, _unc, summary = ompmc.calc_radial(
            water_cylinder, ompmc.PencilBeamSource(divergence_sigma=0.05),
            ompmc.Spectrum.monoenergetic(6.0), water_physics,
            n_histories=4000, n_batches=4)

        assert summary.n_started == 4000
        assert dose.sum() > 0.0

    def test_a_focused_beam_is_narrowest_at_its_waist(self, water_cylinder,
                                                      water_physics):
        # The one thing a correlation does that nothing else can: converge.
        # A spot and a divergence on their own only ever widen with depth, so
        # the beam this is measured against is the same width and divergence
        # with the correlation taken out.
        #
        # 1 MeV rather than the 6 MeV elsewhere in this file: at 6 MeV the
        # secondary electrons carry the energy far enough sideways to blur a
        # waist this narrow away.
        def narrowest_slab(source):
            dose, _unc, _summary = ompmc.calc_radial(
                water_cylinder, source, ompmc.Spectrum.monoenergetic(1.0),
                water_physics, n_histories=20_000, n_batches=4,
                output_dose=False)      # energy, so ring volumes drop out

            # How much of each slab is on the axis. Per slab, so the beam
            # being attenuated on its way down cancels and what is left is
            # only how wide it is.
            on_axis = dose[0, :]/dose.sum(axis=0)
            return int(np.argmax(on_axis))

        # Down to a millimetre 5 cm in, which the half centimetre slabs put
        # at slab 10. Converging hard, so the waist beats the scatter blur.
        focused = ompmc.PencilBeamSource.focused(0.1, 0.4, 5.0)
        blurred = ompmc.PencilBeamSource(
            spot_sigma=focused.spot_sigma, divergence_sigma=0.4)

        assert narrowest_slab(blurred) <= 1
        assert 7 <= narrowest_slab(focused) <= 13

    def test_an_ssd_source_runs_too(self, water_cylinder, water_physics):
        dose, _unc, summary = ompmc.calc_radial(
            water_cylinder, ompmc.PencilBeamSource(ssd=100.0, field_radius=4.0),
            ompmc.Spectrum.monoenergetic(6.0), water_physics,
            n_histories=4000, n_batches=4)

        assert summary.n_started == 4000
        assert dose.sum() > 0.0

    def test_an_electron_beam_runs_too(self, water_cylinder, water_physics):
        dose, _unc, summary = ompmc.calc_radial(
            water_cylinder, ompmc.PencilBeamSource(),
            ompmc.Spectrum.monoenergetic(6.0), water_physics,
            n_histories=4000, n_batches=4, charge=-1)

        # An electron beam stopping in the cylinder leaves most of its energy
        # there, unlike a photon beam of the same energy.
        assert summary.energy_fraction > 0.5
        assert dose.sum() > 0.0

    def test_a_phase_space_drives_it(self, water_cylinder, water_physics,
                                     phsp_beam):
        dose, _unc, summary = ompmc.calc_radial(
            water_cylinder, ompmc.PhaseSpaceSource(phsp_beam),
            physics=water_physics, n_histories=4000, n_batches=4)

        assert summary.n_started == 4000
        assert dose.sum() > 0.0

    def test_output_dose_false_gives_energy(self, water_cylinder,
                                            water_physics):
        dose, _unc, _summary = ompmc.calc_radial(
            water_cylinder, ompmc.PencilBeamSource(),
            ompmc.Spectrum.monoenergetic(6.0), water_physics,
            n_histories=2000, n_batches=4, output_dose=False)

        energy, _unc2, _s2 = ompmc.calc_radial(
            water_cylinder, ompmc.PencilBeamSource(),
            ompmc.Spectrum.monoenergetic(6.0), water_physics,
            n_histories=2000, n_batches=4, output_dose=True)

        # Different quantities, so different numbers; both positive somewhere.
        assert dose.sum() > 0.0
        assert energy.sum() > 0.0
        assert not np.allclose(dose, energy)

    def test_progress_is_reported_and_can_stop_the_run(self, water_cylinder,
                                                       water_physics):
        seen = []

        def watch(fraction):
            seen.append(fraction)
            return True

        ompmc.calc_radial(
            water_cylinder, ompmc.PencilBeamSource(),
            ompmc.Spectrum.monoenergetic(6.0), water_physics,
            n_histories=400, n_batches=4, progress=watch)

        assert len(seen) == 4
        assert seen[-1] == pytest.approx(1.0)

        with pytest.raises(KeyboardInterrupt):
            ompmc.calc_radial(
                water_cylinder, ompmc.PencilBeamSource(),
                ompmc.Spectrum.monoenergetic(6.0), water_physics,
                n_histories=400, n_batches=4, progress=lambda f: False)

    def test_a_bad_material_is_reported_rather_than_transported(
            self, water_physics):
        geometry = ompmc.CylinderGeometry(
            r_bounds=[0.0, 1.0], z_bounds=[0.0, 1.0],
            material="NOT_A_MEDIUM", density=1.0)

        with pytest.raises(RuntimeError):
            ompmc.calc_radial(geometry, ompmc.PencilBeamSource(),
                              ompmc.Spectrum.monoenergetic(6.0), water_physics,
                              n_histories=100, n_batches=2)


class TestGeometryModeDoesNotLeak:
    """A process runs one calculation at a time, but not only one calculation.

    The shape of the phantom is a global the transport reads, so a cylinder
    left behind by an earlier call would silently be what a later cube run
    transports in. Each entry point declares the shape it wants, and this is
    what says so.
    """

    def test_a_cube_after_a_cylinder(self, water_cylinder, water_phantom,
                                     water_physics):
        ompmc.calc_radial(
            water_cylinder, ompmc.PencilBeamSource(),
            ompmc.Spectrum.monoenergetic(6.0), water_physics,
            n_histories=400, n_batches=2)

        dose, _unc = ompmc.calc_cube(
            water_phantom, ompmc.CollimatedSource(90.0, -1.0, 1.0, -1.0, 1.0),
            ompmc.Spectrum.monoenergetic(6.0), water_physics,
            n_histories=400, n_batches=2)

        assert dose.shape == water_phantom.shape
        assert dose.sum() > 0.0

    def test_a_cylinder_after_a_cube(self, water_cylinder, water_phantom,
                                     water_physics):
        ompmc.calc_cube(
            water_phantom, ompmc.CollimatedSource(90.0, -1.0, 1.0, -1.0, 1.0),
            ompmc.Spectrum.monoenergetic(6.0), water_physics,
            n_histories=400, n_batches=2)

        dose, _unc, _summary = ompmc.calc_radial(
            water_cylinder, ompmc.PencilBeamSource(),
            ompmc.Spectrum.monoenergetic(6.0), water_physics,
            n_histories=400, n_batches=2)

        assert dose.shape == water_cylinder.shape
        assert dose.sum() > 0.0

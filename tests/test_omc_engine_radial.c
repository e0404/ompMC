/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations

 The radial engine end to end: a beam down the axis of a water cylinder,
 through the transport, into an r-z dose distribution. Everything below the
 engine is real -- real cross sections, real water, real showers -- so this
 has to run from the repository root where the data and pegs4 folders live.
 CTest is told to do that.

 The physics assertions are deliberately the broad ones, the shapes that no
 amount of geometry confusion can fake: a photon beam builds up to a maximum
 below the surface and falls away past it, an electron beam stops somewhere
 near its range, and dose falls off away from the axis. Anything tighter would
 be a test of the cross section tables rather than of this engine, and would
 fail for the wrong reasons.
*****************************************************************************/

/* Before anything can include setjmp.h; see the comment in
 tests/test_omc_phsp.c for why MinGW's SEH-unwinding longjmp() is not what
 this harness wants. */
#if defined(__MINGW32__)
    #define __USE_MINGW_SETJMP_NON_SEH 1
#endif

#include "omc_engine_radial.h"
#include "omc_geom.h"
#include "omc_geom_cyl.h"
#include "omc_host.h"
#include "omc_phsp.h"
#include "omc_source_pencil.h"
#include "omc_source_phsp.h"
#include "omc_spectrum.h"
#include "omc_utilities.h"
#include "ompmc.h"

#include <math.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int verbose_flag = 0;

#if defined(_MSC_VER)
    __declspec(thread) extern struct Stack stack;
#else
    extern struct Stack stack;
    #pragma omp threadprivate(stack)
#endif

extern struct Media media;
extern struct Pegs pegs_data;

/*******************************************************************************
* Assertion harness
*******************************************************************************/
static int tests_run = 0;
static int tests_failed = 0;
static const char *current_test = "";

#define CHECK(cond)                                                           \
    do {                                                                      \
        if (!(cond)) {                                                        \
            printf("  FAIL %s:%d in %s: %s\n",                                \
                   __FILE__, __LINE__, current_test, #cond);                  \
            tests_failed++;                                                   \
        }                                                                     \
    } while (0)

#define CHECK_CLOSE(got, want, tol)                                           \
    do {                                                                      \
        double _g = (got), _w = (want);                                       \
        if (!(fabs(_g - _w) <= (tol))) {                                      \
            printf("  FAIL %s:%d in %s: %s == %.17g, expected %.17g "         \
                   "(tol %g)\n", __FILE__, __LINE__, current_test,            \
                   #got, _g, _w, (double)(tol));                              \
            tests_failed++;                                                   \
        }                                                                     \
    } while (0)

#define RUN(fn)                                                               \
    do {                                                                      \
        current_test = #fn;                                                   \
        tests_run++;                                                          \
        int _before = tests_failed;                                           \
        printf("%-54s ", #fn);                                                \
        fflush(stdout);                                                       \
        fn();                                                                 \
        printf("%s\n", tests_failed == _before ? "ok" : "FAILED");            \
    } while (0)

static void silentLog(int level, const char *message, void *user) {
    (void)level; (void)message; (void)user;
}

static void quietHost(void) {
    struct OmcHost quiet = {silentLog, NULL, NULL};
    omcSetHost(&quiet);
}

static jmp_buf fail_jmp;
static char fail_id[128];
static int fail_seen;
static int fail_armed;

static void catchingFail(const char *id, const char *message, void *user) {

    (void)user;
    snprintf(fail_id, sizeof(fail_id), "%s", id != NULL ? id : "");
    fail_seen = 1;

    if (!fail_armed) {
        printf("\n  FAIL %s: unexpected failure %s\n         %s\n",
               current_test, id != NULL ? id : "(no id)",
               message != NULL ? message : "");
        fflush(stdout);
        exit(EXIT_FAILURE);
    }

    fail_armed = 0;
    longjmp(fail_jmp, 1);
}

static void catchingHost(void) {

    fail_seen = 0;
    fail_armed = 0;
    fail_id[0] = '\0';

    struct OmcHost catcher = {silentLog, catchingFail, NULL};
    omcSetHost(&catcher);
}

#define EXPECT_FAIL(id, call)                                                 \
    do {                                                                      \
        fail_seen = 0;                                                        \
        fail_id[0] = '\0';                                                    \
        fail_armed = 1;                                                       \
        if (setjmp(fail_jmp) == 0) {                                          \
            call;                                                             \
        }                                                                     \
        fail_armed = 0;                                                       \
        if (!fail_seen) {                                                     \
            printf("  FAIL %s:%d in %s: %s did not fail, expected %s\n",      \
                   __FILE__, __LINE__, current_test, #call, (id));            \
            tests_failed++;                                                   \
        }                                                                     \
        else if (strcmp(fail_id, (id)) != 0) {                                \
            printf("  FAIL %s:%d in %s: %s failed with %s, expected %s\n",    \
                   __FILE__, __LINE__, current_test, #call, fail_id, (id));   \
            tests_failed++;                                                   \
        }                                                                     \
    } while (0)

/*******************************************************************************
* A water cylinder, and the physics to transport in it
*
* 5 cm across in ten half centimetre rings and 10 cm deep in twenty half
* centimetre slabs, its front face at z = 0 so a beam from negative z shines
* straight into it.
*******************************************************************************/

#define NRAD 10
#define NZR 20
#define NREG (NRAD*NZR)

static double rb[NRAD + 1], zb[NZR + 1];
static int medIndices[NREG];
static double medDensities[NREG];

static void setInput(int i, const char *key, const char *value) {
    snprintf(input_items[i].key, sizeof(input_items[i].key), "%s", key);
    snprintf(input_items[i].value, sizeof(input_items[i].value), "%s", value);
}

/*! @p density in every region, or 0 to leave it to the PEGS file. */
static void setUpWaterCylinderAt(double density) {

    setInput(0, "pegs file", "./pegs4/700icru.pegs4dat");
    setInput(1, "pgs4form file", "./pegs4/pgs4form.dat");
    setInput(2, "data folder", "./data/");
    setInput(3, "rng seeds", "97 33");
    setInput(4, "global ecut", "0.700");
    setInput(5, "global pcut", "0.010");
    setInput(6, "nsplit", "1");
    setInput(7, "esave", "2.0");
    input_idx = 8;

    for (int i = 0; i <= NRAD; i++) rb[i] = 0.5*i;
    for (int k = 0; k <= NZR; k++) zb[k] = 0.5*k;

    memset(&geometry, 0, sizeof(geometry));

    geometry.isize = NRAD;
    geometry.ksize = NZR;
    geometry.rbounds = rb;
    geometry.zbounds = zb;

    for (int i = 0; i < NREG; i++) {
        /* EGS counts media from 1; 0 would be vacuum, and a cylinder of
         vacuum absorbs nothing, which is a confusing way to find this out. */
        medIndices[i] = 1;
        medDensities[i] = density;
    }

    geometry.med_indices = medIndices;
    geometry.med_densities = medDensities;

    omcStageRegion(0, 0, 0, 0);
    omcGeomCylInit();

    media.nmed = 1;
    snprintf(media.med_names[0], 60, "%s", "H2O700ICRU");

    quietHost();

    initMediaData();
    initRegions();
    initVrt();
}

static void setUpWaterCylinder(void) {
    setUpWaterCylinderAt(1.0);
}

static void tearDownWaterCylinder(void) {

    cleanPhoton();
    cleanRayleigh();
    cleanPair();
    cleanElectron();
    cleanMscat();
    cleanSpin();
    cleanRegions();

    geometry.med_indices = NULL;
    geometry.med_densities = NULL;
    geometry.rbounds = NULL;
    geometry.zbounds = NULL;
    geometry.mode = OMC_GEOM_CARTESIAN;

    omcSetHost(NULL);
}

static struct OmcRadialOptions optionsFor(int nhist) {

    struct OmcRadialOptions opt;
    memset(&opt, 0, sizeof(opt));

    opt.nhist = nhist;
    opt.nbatch = 4;
    opt.outputDose = 1;

    return opt;
}

/* dose[ir + iz*nr], which is how the engine lays it out. */
static double at(const double *dose, int ir, int iz) {
    return dose[ir + iz*NRAD];
}

/* The index of the largest entry of a depth profile. */
static int argmax(const double *profile, int n) {

    int best = 0;
    for (int k = 0; k < n; k++) {
        if (profile[k] > profile[best]) {
            best = k;
        }
    }

    return best;
}

/*******************************************************************************
* The tests
*******************************************************************************/

/* A 6 MeV photon pencil into water has to build up to a maximum below the
 surface and fall away past it. */
static void test_a_photon_pencil_builds_up_and_falls_off(void) {

    setUpWaterCylinder();

    struct OmcSpectrum spectrum;
    omcSpectrumMonoenergetic(&spectrum, 6.0);

    struct OmcPencilSource pencil;
    memset(&pencil, 0, sizeof(pencil));
    pencil.kind = OMC_PENCIL_PARALLEL;
    pencil.spectrum = &spectrum;
    pencil.charge = 0;

    struct OmcSource source;
    omcPencilSourceAsSource(&pencil, &source);

    struct OmcRadialOptions opt = optionsFor(40000);
    struct OmcForwardSummary summary;

    double *dose = malloc(NREG*sizeof(double));
    double *unc = malloc(NREG*sizeof(double));

    int finished = omcCalcRadial(&opt, &source, NULL, dose, unc, NULL,
                                 &summary);

    CHECK(finished == 1);
    CHECK(summary.nhist == 40000);
    CHECK(summary.started == 40000);        /* all of them are aimed in */
    CHECK(summary.blocked == 0);
    CHECK(summary.energyFraction > 0.0 && summary.energyFraction < 1.0);

    /* The depth dose on the axis, which for a pencil beam is the innermost
     ring rather than an average over neighbouring voxels. */
    double depth[NZR];
    for (int k = 0; k < NZR; k++) {
        depth[k] = at(dose, 0, k);
    }

    int kmax = argmax(depth, NZR);

    CHECK(depth[kmax] > 0.0);

    /* Below the surface rather than in the very first slab, and well before
     the back of the cylinder. */
    CHECK(kmax > 0);
    CHECK(kmax < NZR/2);

    /* And falling by the far end. */
    CHECK(depth[NZR - 1] < depth[kmax]);

    /* Every region has an uncertainty, and the ones that got dose have a
     believable one. */
    CHECK(unc[0 + kmax*NRAD] > 0.0 && unc[0 + kmax*NRAD] < 0.5);

    omcSpectrumFree(&spectrum);
    free(dose);
    free(unc);
    tearDownWaterCylinder();
}

/* A density of zero does not mean a cylinder of nothing. initRegions() reads
 it as "whatever the PEGS file says this medium weighs" and sets rhof to 1,
 which is how a host with no density of its own to impose says so -- and both
 hosts of this engine let one be left out, so it is the ordinary case and not
 a corner.

 Read literally it is below the air threshold, and the scorer would call every
 region of the phantom air: a whole run of zeros and empty-region
 uncertainties out of transport that went perfectly well.

 Asserted against the same cylinder with that density written out in full.
 rhof is 1 either way -- rho/rho exactly, in the second case -- so this is the
 same transport twice and the only question is whether the scorer weighed the
 rings the same. To the last few digits and not bit for bit, for the reason
 test_a_rerun_gives_the_same_answer() gives: two runs accumulate their atomics
 in whatever order the threads arrive, and no two runs of anything here agree
 more closely than that. */
static void test_a_density_left_to_pegs_is_not_mistaken_for_air(void) {

    double *fromPegs = malloc(NREG*sizeof(double));
    double *fromPegsUnc = malloc(NREG*sizeof(double));
    double *spelledOut = malloc(NREG*sizeof(double));
    double pegsDensity;

    for (int sentinel = 1; sentinel >= 0; sentinel--) {

        /* The first pass has to be the sentinel one: it is what reports the
         density the second pass then states explicitly. */
        setUpWaterCylinderAt(sentinel ? 0.0 : pegsDensity);

        if (sentinel) {
            pegsDensity = pegs_data.rho[0];
            CHECK(pegsDensity > 0.044);         /* the premise: not air */
        }

        struct OmcSpectrum spectrum;
        omcSpectrumMonoenergetic(&spectrum, 6.0);

        struct OmcPencilSource pencil;
        memset(&pencil, 0, sizeof(pencil));
        pencil.kind = OMC_PENCIL_PARALLEL;
        pencil.spectrum = &spectrum;
        pencil.charge = 0;

        struct OmcSource source;
        omcPencilSourceAsSource(&pencil, &source);

        struct OmcRadialOptions opt = optionsFor(20000);
        struct OmcForwardSummary summary;

        CHECK(omcCalcRadial(&opt, &source,
                            NULL, sentinel ? fromPegs : spelledOut,
                            sentinel ? fromPegsUnc : NULL,
                            NULL, &summary) == 1);

        omcSpectrumFree(&spectrum);
        tearDownWaterCylinder();
    }

    /* Real dose, not a phantom reported as empty */
    double onAxis = 0.0;
    for (int k = 0; k < NZR; k++) {
        onAxis += at(fromPegs, 0, k);
    }
    CHECK(onAxis > 0.0);

    /* and real uncertainties with it, rather than the empty-region sentinel
     everywhere -- which is the shape the bug took: dose zeroed and 0.9999999
     written over the statistics that had been collected. */
    int empty = 0;
    for (int i = 0; i < NREG; i++) {
        if (fromPegsUnc[i] >= 0.9999999) {
            empty++;
        }
    }
    CHECK(empty < NREG);

    double peak = 0.0;
    for (int i = 0; i < NREG; i++) {
        if (fromPegs[i] > peak) {
            peak = fromPegs[i];
        }
    }
    CHECK(peak > 0.0);

    for (int i = 0; i < NREG; i++) {
        if (fromPegs[i] > 0.01*peak) {
            CHECK_CLOSE(fromPegs[i], spelledOut[i], 1e-9*fromPegs[i]);
        }
    }

    free(fromPegs);
    free(fromPegsUnc);
    free(spelledOut);
}

/* The whole point of scoring in rings: dose falls away from the axis of a
 pencil beam, and does so steeply. */
static void test_dose_falls_off_away_from_the_axis(void) {

    setUpWaterCylinder();

    struct OmcSpectrum spectrum;
    omcSpectrumMonoenergetic(&spectrum, 6.0);

    struct OmcPencilSource pencil;
    memset(&pencil, 0, sizeof(pencil));
    pencil.kind = OMC_PENCIL_PARALLEL;
    pencil.spectrum = &spectrum;
    pencil.charge = 0;

    struct OmcSource source;
    omcPencilSourceAsSource(&pencil, &source);

    struct OmcRadialOptions opt = optionsFor(40000);
    struct OmcForwardSummary summary;

    double *dose = malloc(NREG*sizeof(double));
    double *unc = malloc(NREG*sizeof(double));

    CHECK(omcCalcRadial(&opt, &source, NULL, dose, unc, NULL, &summary) == 1);

    /* At the depth of the maximum on the axis */
    double depth[NZR];
    for (int k = 0; k < NZR; k++) {
        depth[k] = at(dose, 0, k);
    }
    int kmax = argmax(depth, NZR);

    /* Not asserted ring by ring -- the outer ones are noisy -- but across
     enough of a gap that noise cannot account for it. */
    CHECK(at(dose, 0, kmax) > at(dose, 2, kmax));
    CHECK(at(dose, 2, kmax) > at(dose, 5, kmax));
    CHECK(at(dose, 9, kmax) < 0.05*at(dose, 0, kmax));

    omcSpectrumFree(&spectrum);
    free(dose);
    free(unc);
    tearDownWaterCylinder();
}

/* A 10 MeV electron beam has a range in water of about 5 cm, and what is
 past it is the bremsstrahlung tail rather than the beam. */
static void test_an_electron_pencil_stops_near_its_range(void) {

    setUpWaterCylinder();

    struct OmcSpectrum spectrum;
    omcSpectrumMonoenergetic(&spectrum, 10.0);

    struct OmcPencilSource pencil;
    memset(&pencil, 0, sizeof(pencil));
    pencil.kind = OMC_PENCIL_PARALLEL;
    pencil.spectrum = &spectrum;
    pencil.charge = -1;

    struct OmcSource source;
    omcPencilSourceAsSource(&pencil, &source);

    struct OmcRadialOptions opt = optionsFor(20000);
    struct OmcForwardSummary summary;

    double *dose = malloc(NREG*sizeof(double));
    double *unc = malloc(NREG*sizeof(double));

    CHECK(omcCalcRadial(&opt, &source, NULL, dose, unc, NULL, &summary) == 1);
    CHECK(summary.started == 20000);

    /* An electron beam stopping in the phantom leaves nearly all of its
     energy there. */
    CHECK(summary.energyFraction > 0.5);
    CHECK(summary.energyFraction <= 1.0 + 1e-9);

    double depth[NZR];
    for (int k = 0; k < NZR; k++) {
        double sum = 0.0;
        for (int ir = 0; ir < NRAD; ir++) {
            sum += at(dose, ir, k);
        }
        depth[k] = sum;
    }

    int kmax = argmax(depth, NZR);

    CHECK(depth[kmax] > 0.0);
    CHECK(kmax < NZR/2);                    /* it stops in the first half */

    /* Past the practical range, only the tail is left. z = 6 cm is slab 12. */
    CHECK(depth[NZR - 1] < 0.05*depth[kmax]);

    omcSpectrumFree(&spectrum);
    free(dose);
    free(unc);
    tearDownWaterCylinder();
}

/* The point source illuminates a field rather than a point, so more of what
 it deposits near the surface lands off the axis. */
static void test_an_ssd_source_spreads_the_beam(void) {

    setUpWaterCylinder();

    struct OmcSpectrum spectrum;
    omcSpectrumMonoenergetic(&spectrum, 6.0);

    double offAxisFraction[2];

    for (int kind = 0; kind < 2; kind++) {

        struct OmcPencilSource pencil;
        memset(&pencil, 0, sizeof(pencil));
        pencil.spectrum = &spectrum;
        pencil.charge = 0;

        if (kind == 0) {
            pencil.kind = OMC_PENCIL_PARALLEL;
        }
        else {
            pencil.kind = OMC_PENCIL_SSD;
            pencil.ssd = 100.0;
            pencil.fieldRadius = 4.0;
        }

        struct OmcSource source;
        omcPencilSourceAsSource(&pencil, &source);

        struct OmcRadialOptions opt = optionsFor(20000);
        struct OmcForwardSummary summary;

        double *dose = malloc(NREG*sizeof(double));

        CHECK(omcCalcRadial(&opt, &source, NULL, dose, NULL, NULL,
                            &summary) == 1);
        CHECK(summary.started == 20000);

        /* Energy rather than dose, so that the ring volumes do not have to
         be undone: dose is per unit mass and the outer rings are large. */
        double onAxis = 0.0, total = 0.0;
        for (int k = 0; k < 4; k++) {
            for (int ir = 0; ir < NRAD; ir++) {
                double mass = M_PI*(rb[ir+1]*rb[ir+1] - rb[ir]*rb[ir])*
                              (zb[k+1] - zb[k]);
                double energy = at(dose, ir, k)*mass;

                total += energy;
                if (ir == 0) {
                    onAxis += energy;
                }
            }
        }

        CHECK(total > 0.0);
        offAxisFraction[kind] = 1.0 - onAxis/total;

        free(dose);
    }

    /* A pencil puts its entrance dose on the axis; a field spreads it. */
    CHECK(offAxisFraction[1] > offAxisFraction[0]);

    omcSpectrumFree(&spectrum);
    tearDownWaterCylinder();
}

/* Widening the source has to widen the dose. The entrance slab is where it
 shows cleanest: down there the beam has not yet been broadened by scatter, so
 nearly all of what a delta pencil deposits is in the innermost ring. */
static void test_a_gaussian_spot_broadens_the_dose(void) {

    setUpWaterCylinder();

    struct OmcSpectrum spectrum;
    omcSpectrumMonoenergetic(&spectrum, 6.0);

    double onAxisFraction[2];

    for (int widened = 0; widened < 2; widened++) {

        struct OmcPencilSource pencil;
        memset(&pencil, 0, sizeof(pencil));
        pencil.kind = OMC_PENCIL_PARALLEL;
        pencil.spectrum = &spectrum;
        pencil.charge = 0;
        pencil.spotSigma = widened ? 1.5 : 0.0;

        struct OmcSource source;
        omcPencilSourceAsSource(&pencil, &source);

        struct OmcRadialOptions opt = optionsFor(20000);
        opt.outputDose = 0;             /* energy, so ring volumes drop out */
        struct OmcForwardSummary summary;

        double *dose = malloc(NREG*sizeof(double));

        CHECK(omcCalcRadial(&opt, &source, NULL, dose, NULL, NULL,
                            &summary) == 1);

        if (!widened) {
            CHECK(summary.started == 20000);
        }
        else {
            /* A spot wide enough to matter spills over the edge of the
             cylinder, and a particle that starts beyond the barrel
             travelling parallel to it never enters. Those histories happened
             and still count towards the fluence the result is divided by --
             they are beam that missed, not beam that was never there. A
             Gaussian of 1.5 cm against a radius of 5 loses about 0.4%. */
            CHECK(summary.started < 20000);
            CHECK(summary.started > 19000);
        }

        double onAxis = 0.0, total = 0.0;
        for (int ir = 0; ir < NRAD; ir++) {
            total += at(dose, ir, 0);
            if (ir == 0) {
                onAxis += at(dose, ir, 0);
            }
        }

        CHECK(total > 0.0);
        onAxisFraction[widened] = onAxis/total;

        free(dose);
    }

    /* A delta pencil puts nearly everything in the first ring; a centimetre
     and a half of spot spreads it over most of them. */
    CHECK(onAxisFraction[0] > 0.8);
    CHECK(onAxisFraction[1] < 0.3);

    omcSpectrumFree(&spectrum);
    tearDownWaterCylinder();
}

/* Divergence widens the beam too, but with depth rather than at the surface:
 the particles all enter on the axis and fan out from there. */
static void test_a_divergent_beam_widens_with_depth(void) {

    setUpWaterCylinder();

    struct OmcSpectrum spectrum;
    omcSpectrumMonoenergetic(&spectrum, 6.0);

    struct OmcPencilSource pencil;
    memset(&pencil, 0, sizeof(pencil));
    pencil.kind = OMC_PENCIL_PARALLEL;
    pencil.spectrum = &spectrum;
    pencil.charge = 0;
    pencil.divergenceSigma = 0.2;       /* wide, so it shows over 10 cm */

    struct OmcSource source;
    omcPencilSourceAsSource(&pencil, &source);

    struct OmcRadialOptions opt = optionsFor(40000);
    opt.outputDose = 0;
    struct OmcForwardSummary summary;

    double *dose = malloc(NREG*sizeof(double));

    CHECK(omcCalcRadial(&opt, &source, NULL, dose, NULL, NULL, &summary) == 1);

    /* The fraction still on the axis, at the front of the cylinder and at
     the back of it */
    double frac[2];
    int slabs[2] = {0, NZR - 1};

    for (int end = 0; end < 2; end++) {
        double onAxis = 0.0, total = 0.0;
        for (int ir = 0; ir < NRAD; ir++) {
            total += at(dose, ir, slabs[end]);
            if (ir == 0) {
                onAxis += at(dose, ir, slabs[end]);
            }
        }
        CHECK(total > 0.0);
        frac[end] = onAxis/total;
    }

    CHECK(frac[1] < frac[0]);

    free(dose);
    omcSpectrumFree(&spectrum);
    tearDownWaterCylinder();
}

/* The claim a correlation actually makes, made about dose rather than about
 the particles the source hands out: a converging beam is narrowest somewhere
 INSIDE the phantom, at the depth it was told to focus, and wider again past
 it. Nothing else the source can do produces that shape -- both a spot and a
 divergence only ever widen with depth -- so the beam it is measured against
 is the very same width and divergence with the correlation taken out.

 1 MeV rather than the 6 MeV the tests above use, because at 6 MeV the
 secondary electrons carry the energy a couple of centimetres sideways and
 that blur is comparable to the waist itself. */
static void test_a_converging_beam_focuses_inside_the_phantom(void) {

    setUpWaterCylinder();

    struct OmcSpectrum spectrum;
    omcSpectrumMonoenergetic(&spectrum, 1.0);

    /* Down to a millimetre at 5 cm, half way along the cylinder, which the
     0.5 cm slabs put at slab 10. Converging hard: the waist has to beat the
     scatter blur to be visible at all. */
    const double waistDepth = 5.0;
    const double divergence = 0.4;

    double spotSigma, correlation;
    omcPencilWaist(0.1, divergence, waistDepth, &spotSigma, &correlation);

    int kmax[2];

    for (int converging = 0; converging < 2; converging++) {

        struct OmcPencilSource pencil;
        memset(&pencil, 0, sizeof(pencil));
        pencil.kind = OMC_PENCIL_PARALLEL;
        pencil.spectrum = &spectrum;
        pencil.charge = 0;
        pencil.spotSigma = spotSigma;
        pencil.divergenceSigma = divergence;
        pencil.correlation = converging ? correlation : 0.0;

        struct OmcSource source;
        omcPencilSourceAsSource(&pencil, &source);

        struct OmcRadialOptions opt = optionsFor(40000);
        opt.outputDose = 0;             /* energy, so ring volumes drop out */
        struct OmcForwardSummary summary;

        double *dose = malloc(NREG*sizeof(double));

        CHECK(omcCalcRadial(&opt, &source, NULL, dose, NULL, NULL,
                            &summary) == 1);

        /* How much of each slab's energy is on the axis. Per slab, so that
         the beam being attenuated on its way down cancels out and what is
         left is only how wide it is. */
        double onAxis[NZR];
        for (int k = 0; k < NZR; k++) {
            double total = 0.0;
            for (int ir = 0; ir < NRAD; ir++) {
                total += at(dose, ir, k);
            }
            CHECK(total > 0.0);
            onAxis[k] = at(dose, 0, k)/total;
        }

        kmax[converging] = argmax(onAxis, NZR);

        if (converging) {
            /* Narrower at the waist than at either end -- it converges, and
             then it comes apart again. */
            CHECK(onAxis[10] > 2.0*onAxis[0]);
            CHECK(onAxis[10] > 2.0*onAxis[NZR - 1]);
        }

        free(dose);
    }

    /* Uncorrelated, the beam is at its narrowest where it starts. The same
     beam told to focus is narrowest around where it was told to. */
    CHECK(kmax[0] <= 1);
    CHECK(kmax[1] >= 7 && kmax[1] <= 13);

    omcSpectrumFree(&spectrum);
    tearDownWaterCylinder();
}

/*******************************************************************************
* A phase space built in memory, the same way test_omc_forward_phsp.c does it
*******************************************************************************/

#define RECORD_LENGTH 29
#define MAX_RECORDS 64

struct Made {
    struct OmcPhsp phsp;
    unsigned char bytes[MAX_RECORDS*RECORD_LENGTH];
};

static void putLeFloat(unsigned char *dst, float value) {

    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));

    dst[0] = (unsigned char)(bits & 0xFF);
    dst[1] = (unsigned char)((bits >> 8) & 0xFF);
    dst[2] = (unsigned char)((bits >> 16) & 0xFF);
    dst[3] = (unsigned char)((bits >> 24) & 0xFF);
}

/* n photons of the given energy, close to the axis, heading into the cylinder
 from a centimetre above its front face. */
static void makeBeam(struct Made *made, int n, double energy) {

    memset(made, 0, sizeof(*made));

    made->phsp.header.fileType = 0;
    made->phsp.header.byteOrder = 1234;
    made->phsp.header.recordLength = RECORD_LENGTH;
    made->phsp.header.particles = (unsigned long long)n;
    made->phsp.header.checksum = (unsigned long long)n*RECORD_LENGTH;
    made->phsp.nRecords = (unsigned long long)n;
    made->phsp.newHistories = (unsigned long long)n;
    made->phsp.raw = made->bytes;
    made->phsp.cursor = 0;

    for (int i = 0; i < OMC_PHSP_NVARIABLES; i++) {
        made->phsp.header.stored[i] = 1;
    }

    for (int i = 0; i < n; i++) {
        unsigned char *at_ = made->bytes + (size_t)i*RECORD_LENGTH;

        at_[0] = (unsigned char)(OMC_PHSP_PHOTON & 0xFF);
        putLeFloat(at_ + 1, (float)-energy);        /* opens a history */
        putLeFloat(at_ + 5, (float)(-1.0 + 2.0*(double)(i % 5)/4.0));
        putLeFloat(at_ + 9, (float)(-1.0 + 2.0*(double)(i % 3)/2.0));
        putLeFloat(at_ + 13, (float)-1.0);          /* a centimetre above */
        putLeFloat(at_ + 17, 0.0f);                 /* u */
        putLeFloat(at_ + 21, 0.0f);                 /* v, so w is +1 */
        putLeFloat(at_ + 25, 1.0f);                 /* weight */
    }
}

/* A phase space is a source like any other, so the radial engine takes one
 without knowing what it is. */
static void test_a_phase_space_drives_the_radial_engine(void) {

    setUpWaterCylinder();

    struct Made made;
    makeBeam(&made, 16, 6.0);

    struct OmcPhspSampler sampler;
    memset(&sampler, 0, sizeof(sampler));
    sampler.phsp = &made.phsp;
    sampler.order = OMC_PHSP_REPLAY;
    sampler.first = 0;
    omcPhspTransformIdentity(&sampler.transform);

    struct OmcSource source;
    omcPhspSamplerAsSource(&sampler, &source);

    struct OmcRadialOptions opt = optionsFor(20000);
    struct OmcForwardSummary summary;

    double *dose = malloc(NREG*sizeof(double));
    double *unc = malloc(NREG*sizeof(double));

    CHECK(omcCalcRadial(&opt, &source, NULL, dose, unc, NULL, &summary) == 1);
    CHECK(summary.started == 20000);
    CHECK(summary.energyFraction > 0.0 && summary.energyFraction < 1.0);

    double depth[NZR];
    for (int k = 0; k < NZR; k++) {
        double sum = 0.0;
        for (int ir = 0; ir < NRAD; ir++) {
            sum += at(dose, ir, k);
        }
        depth[k] = sum;
    }

    int kmax = argmax(depth, NZR);

    CHECK(depth[kmax] > 0.0);
    CHECK(kmax > 0);                        /* it builds up, like any beam */
    CHECK(depth[NZR - 1] < depth[kmax]);

    free(dose);
    free(unc);
    tearDownWaterCylinder();
}

/* A region nothing reached carries the same 0.9999999 the cube engine and the
 .3ddose format use, rather than a zero that would read as a perfect
 measurement. */
static void test_regions_nothing_reached_carry_the_sentinel(void) {

    setUpWaterCylinder();

    struct OmcSpectrum spectrum;
    omcSpectrumMonoenergetic(&spectrum, 1.0);

    struct OmcPencilSource pencil;
    memset(&pencil, 0, sizeof(pencil));
    pencil.kind = OMC_PENCIL_PARALLEL;
    pencil.spectrum = &spectrum;
    pencil.charge = 0;

    struct OmcSource source;
    omcPencilSourceAsSource(&pencil, &source);

    /* Few enough histories that the far corner of the cylinder cannot have
     been reached: a 1 MeV pencil on the axis, and the outermost ring at the
     back is 5 cm across the beam and 10 cm down it. */
    struct OmcRadialOptions opt = optionsFor(400);
    struct OmcForwardSummary summary;

    double *dose = malloc(NREG*sizeof(double));
    double *unc = malloc(NREG*sizeof(double));

    CHECK(omcCalcRadial(&opt, &source, NULL, dose, unc, NULL, &summary) == 1);

    int empty = 0;
    for (int i = 0; i < NREG; i++) {
        if (dose[i] == 0.0) {
            empty++;
            CHECK_CLOSE(unc[i], 0.9999999, 1e-12);
        }
        else {
            CHECK(unc[i] > 0.0);
        }
    }

    CHECK(empty > 0);

    omcSpectrumFree(&spectrum);
    free(dose);
    free(unc);
    tearDownWaterCylinder();
}

/* Which histories a run does and what they draw depends on the history index
 alone, so a rerun agrees to within the reordering of the atomic
 accumulation -- not bit for bit, but far closer than the statistics. */
static void test_a_rerun_gives_the_same_answer(void) {

    setUpWaterCylinder();

    struct OmcSpectrum spectrum;
    omcSpectrumMonoenergetic(&spectrum, 6.0);

    struct OmcPencilSource pencil;
    memset(&pencil, 0, sizeof(pencil));
    pencil.kind = OMC_PENCIL_PARALLEL;
    pencil.spectrum = &spectrum;
    pencil.charge = 0;

    struct OmcSource source;
    omcPencilSourceAsSource(&pencil, &source);

    struct OmcRadialOptions opt = optionsFor(20000);

    double *first = malloc(NREG*sizeof(double));
    double *again = malloc(NREG*sizeof(double));
    struct OmcForwardSummary s1, s2;

    CHECK(omcCalcRadial(&opt, &source, NULL, first, NULL, NULL, &s1) == 1);
    CHECK(omcCalcRadial(&opt, &source, NULL, again, NULL, NULL, &s2) == 1);

    /* The bookkeeping is exact */
    CHECK(s1.nhist == s2.nhist);
    CHECK(s1.started == s2.started);
    CHECK(s1.blocked == s2.blocked);
    CHECK_CLOSE(s1.energyFraction, s2.energyFraction,
                1e-9*fabs(s1.energyFraction));

    /* And so, to the last few digits, is the dose wherever there is enough
     of it for the comparison to mean anything. */
    double peak = 0.0;
    for (int i = 0; i < NREG; i++) {
        if (first[i] > peak) {
            peak = first[i];
        }
    }

    for (int i = 0; i < NREG; i++) {
        if (first[i] > 0.01*peak) {
            CHECK_CLOSE(first[i], again[i], 1e-9*first[i]);
        }
    }

    omcSpectrumFree(&spectrum);
    free(first);
    free(again);
    tearDownWaterCylinder();
}

static void test_the_engine_refuses_what_it_cannot_run(void) {

    setUpWaterCylinder();
    catchingHost();

    struct OmcSpectrum spectrum;
    omcSpectrumMonoenergetic(&spectrum, 6.0);

    struct OmcPencilSource pencil;
    memset(&pencil, 0, sizeof(pencil));
    pencil.kind = OMC_PENCIL_PARALLEL;
    pencil.spectrum = &spectrum;
    pencil.charge = 0;

    struct OmcSource source;
    omcPencilSourceAsSource(&pencil, &source);

    double *dose = malloc(NREG*sizeof(double));

    /* The batch variance divides by nbatch - 1 */
    {
        struct OmcRadialOptions opt = optionsFor(1000);
        opt.nbatch = 1;
        EXPECT_FAIL("ompMC:radial:tooFewBatches",
            omcCalcRadial(&opt, &source, NULL, dose, NULL, NULL, NULL));
    }

    /* A source that cannot make a particle */
    {
        struct OmcRadialOptions opt = optionsFor(1000);
        struct OmcSource empty;
        memset(&empty, 0, sizeof(empty));
        EXPECT_FAIL("ompMC:radial:noSource",
            omcCalcRadial(&opt, &empty, NULL, dose, NULL, NULL, NULL));
    }

    /* And a phantom that is not a cylinder, which has no rings to score in */
    {
        struct OmcRadialOptions opt = optionsFor(1000);
        geometry.mode = OMC_GEOM_CARTESIAN;
        EXPECT_FAIL("ompMC:radial:notCylindrical",
            omcCalcRadial(&opt, &source, NULL, dose, NULL, NULL, NULL));
        geometry.mode = OMC_GEOM_CYLINDRICAL;
    }

    omcSpectrumFree(&spectrum);
    free(dose);
    tearDownWaterCylinder();
}

/*******************************************************************************/

int main(void) {

    setvbuf(stdout, NULL, _IONBF, 0);

    printf("test_omc_engine_radial\n");

    RUN(test_a_photon_pencil_builds_up_and_falls_off);
    RUN(test_a_density_left_to_pegs_is_not_mistaken_for_air);
    RUN(test_dose_falls_off_away_from_the_axis);
    RUN(test_an_electron_pencil_stops_near_its_range);
    RUN(test_an_ssd_source_spreads_the_beam);
    RUN(test_a_gaussian_spot_broadens_the_dose);
    RUN(test_a_divergent_beam_widens_with_depth);
    RUN(test_a_converging_beam_focuses_inside_the_phantom);
    RUN(test_a_phase_space_drives_the_radial_engine);
    RUN(test_regions_nothing_reached_carry_the_sentinel);
    RUN(test_a_rerun_gives_the_same_answer);
    RUN(test_the_engine_refuses_what_it_cannot_run);

    printf("\n%d tests, %d failures\n", tests_run, tests_failed);

    return tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

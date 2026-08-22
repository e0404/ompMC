/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations

 The two sources that shine down the axis of a cylinder: a parallel pencil,
 which is the one a dose kernel is defined for, and a point at a given SSD,
 which is what a real machine looks like.

 Nothing here transports. A source's whole job is to answer "which particle
 starts this history, and where is it going", so these tests ask it that and
 check the answer -- including how many random numbers it took to give it,
 which matters more than it looks. The random stream is indexed per history,
 so a source that draws a different number of values depending on what it was
 asked shifts every later draw in that history and changes the answer of an
 otherwise identical run.
*****************************************************************************/

/* Before anything can include setjmp.h; see the comment in
 tests/test_omc_phsp.c for why MinGW's SEH-unwinding longjmp() is not what
 this harness wants. */
#if defined(__MINGW32__)
    #define __USE_MINGW_SETJMP_NON_SEH 1
#endif

#include "omc_geom.h"
#include "omc_geom_cyl.h"
#include "omc_host.h"
#include "omc_random.h"
#include "omc_source.h"
#include "omc_source_pencil.h"
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
        fn();                                                                 \
        printf("%s\n", tests_failed == _before ? "ok" : "FAILED");            \
    } while (0)

static void silentLog(int level, const char *message, void *user) {
    (void)level; (void)message; (void)user;
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
* A cylinder for the sources to shine into: 5 cm across, 10 cm deep, its front
* face at z = 0.
*******************************************************************************/

#define NR 5
#define NZC 10

static double rb[NR + 1], zbc[NZC + 1];
static int cylMedIndices[NR*NZC];
static double cylMedDensities[NR*NZC];

static struct OmcSpectrum mono;

static void setUpCylinder(void) {

    for (int i = 0; i <= NR; i++) rb[i] = 1.0*i;
    for (int k = 0; k <= NZC; k++) zbc[k] = 1.0*k;

    memset(&geometry, 0, sizeof(geometry));

    geometry.isize = NR;
    geometry.ksize = NZC;
    geometry.rbounds = rb;
    geometry.zbounds = zbc;

    for (int i = 0; i < NR*NZC; i++) {
        cylMedIndices[i] = 1;
        cylMedDensities[i] = 1.0;
    }
    geometry.med_indices = cylMedIndices;
    geometry.med_densities = cylMedDensities;

    omcStageRegion(0, 0, 0, 0);
    omcGeomCylInit();

    omcSpectrumMonoenergetic(&mono, 6.0);
}

static struct OmcPencilSource pencilParallel(void) {

    struct OmcPencilSource pencil;
    memset(&pencil, 0, sizeof(pencil));

    pencil.kind = OMC_PENCIL_PARALLEL;
    pencil.spectrum = &mono;
    pencil.charge = 0;

    return pencil;
}

static struct OmcPencilSource pencilSsd(double ssd, double fieldRadius) {

    struct OmcPencilSource pencil;
    memset(&pencil, 0, sizeof(pencil));

    pencil.kind = OMC_PENCIL_SSD;
    pencil.spectrum = &mono;
    pencil.charge = 0;
    pencil.ssd = ssd;
    pencil.fieldRadius = fieldRadius;

    return pencil;
}

/*******************************************************************************
* The tests
*******************************************************************************/

static void test_a_parallel_pencil_starts_on_the_axis_above_the_face(void) {

    setUpCylinder();

    struct OmcPencilSource pencil = pencilParallel();
    struct OmcSource source;
    omcPencilSourceAsSource(&pencil, &source);

    struct OmcSourceParticle particle;
    memset(&particle, 0, sizeof(particle));

    setRandomHistory(0);
    CHECK(source.sample(&source, 0, 0, &particle) == 1);

    /* On the axis, which is what makes it a pencil */
    CHECK_CLOSE(particle.x, 0.0, 1e-15);
    CHECK_CLOSE(particle.y, 0.0, 1e-15);

    /* Upstream of the front face, not on it: a particle handed to
     omcSourcePlace() already sitting on the surface would skip the build up
     region it should have travelled through. */
    CHECK(particle.z < geometry.zbounds[0]);

    /* Straight down the axis */
    CHECK_CLOSE(particle.u, 0.0, 1e-15);
    CHECK_CLOSE(particle.v, 0.0, 1e-15);
    CHECK_CLOSE(particle.w, 1.0, 1e-15);

    CHECK_CLOSE(particle.weight, 1.0, 1e-15);
    CHECK_CLOSE(particle.energy, 6.0, 1e-15);
    CHECK(particle.charge == 0);
}

static void test_the_charge_is_the_one_it_was_given(void) {

    setUpCylinder();

    for (int charge = -1; charge <= 1; charge++) {
        struct OmcPencilSource pencil = pencilParallel();
        pencil.charge = charge;

        struct OmcSource source;
        omcPencilSourceAsSource(&pencil, &source);

        struct OmcSourceParticle particle;
        memset(&particle, 0, sizeof(particle));

        setRandomHistory(3);
        CHECK(source.sample(&source, 3, 0, &particle) == 1);
        CHECK(particle.charge == charge);
    }
}

/* The point source spreads the beam over a disc on the front face. Every
 particle has to come from the same point, be aimed at that disc, and carry a
 direction that is genuinely a unit vector -- the transport takes that as
 given and never renormalizes. */
static void test_an_ssd_source_fills_the_field_it_was_given(void) {

    setUpCylinder();

    double ssd = 100.0;
    double fieldRadius = 3.0;

    struct OmcPencilSource pencil = pencilSsd(ssd, fieldRadius);
    struct OmcSource source;
    omcPencilSourceAsSource(&pencil, &source);

    double rmax = 0.0;
    int outside = 0;

    for (uint64_t ihist = 0; ihist < 4000; ihist++) {

        struct OmcSourceParticle particle;
        memset(&particle, 0, sizeof(particle));

        setRandomHistory(ihist);
        CHECK(source.sample(&source, ihist, (int)ihist, &particle) == 1);

        /* All from the one point, an SSD above the face */
        CHECK_CLOSE(particle.x, 0.0, 1e-15);
        CHECK_CLOSE(particle.y, 0.0, 1e-15);
        CHECK_CLOSE(particle.z, geometry.zbounds[0] - ssd, 1e-12);

        double norm = sqrt(particle.u*particle.u + particle.v*particle.v +
                           particle.w*particle.w);
        CHECK_CLOSE(norm, 1.0, 1e-12);
        CHECK(particle.w > 0.0);

        /* Where the ray meets the front face */
        double t = (geometry.zbounds[0] - particle.z)/particle.w;
        double hx = particle.x + t*particle.u;
        double hy = particle.y + t*particle.v;
        double r = sqrt(hx*hx + hy*hy);

        if (r > fieldRadius + 1e-9) {
            outside++;
        }
        if (r > rmax) {
            rmax = r;
        }
    }

    CHECK(outside == 0);

    /* And it fills the field rather than hugging the axis: r = R*sqrt(rnno)
     is the sampling that spreads particles evenly over the disc, so the
     largest of four thousand should be within a whisker of the edge. */
    CHECK(rmax > 0.97*fieldRadius);
}

/* Left at zero, the field is the whole front face. */
static void test_a_field_radius_of_zero_means_the_whole_face(void) {

    setUpCylinder();

    struct OmcPencilSource pencil = pencilSsd(50.0, 0.0);
    struct OmcSource source;
    omcPencilSourceAsSource(&pencil, &source);

    double rmax = 0.0;
    double radius = geometry.rbounds[geometry.isize];

    for (uint64_t ihist = 0; ihist < 4000; ihist++) {

        struct OmcSourceParticle particle;
        memset(&particle, 0, sizeof(particle));

        setRandomHistory(ihist);
        source.sample(&source, ihist, (int)ihist, &particle);

        double t = (geometry.zbounds[0] - particle.z)/particle.w;
        double hx = particle.x + t*particle.u;
        double hy = particle.y + t*particle.v;
        double r = sqrt(hx*hx + hy*hy);

        CHECK(r <= radius + 1e-9);
        if (r > rmax) {
            rmax = r;
        }
    }

    CHECK(rmax > 0.97*radius);
}

/* What a source draws must depend on the history index and nothing else, or
 the answer starts depending on how OpenMP handed the histories out. */
static void test_sampling_depends_only_on_the_history(void) {

    setUpCylinder();

    struct OmcPencilSource pencil = pencilSsd(80.0, 2.0);
    struct OmcSource source;
    omcPencilSourceAsSource(&pencil, &source);

    for (uint64_t ihist = 0; ihist < 32; ihist++) {

        struct OmcSourceParticle first, again;
        memset(&first, 0, sizeof(first));
        memset(&again, 0, sizeof(again));

        setRandomHistory(ihist);
        source.sample(&source, ihist, 0, &first);

        /* The same history, reached from somewhere else entirely */
        setRandomHistory(ihist + 1000);
        setRandomHistory(ihist);
        source.sample(&source, ihist, 17, &again);

        CHECK(memcmp(&first, &again, sizeof(first)) == 0);
    }
}

/* How many random numbers a source takes is part of its contract, not an
 implementation detail: change it and every run that ever used it gives a
 different answer. The parallel pencil with a monoenergetic spectrum has
 nothing to draw for, and takes none; the point source takes exactly two, for
 the radius and the azimuth. */
static void test_the_draw_count_is_what_it_says(void) {

    setUpCylinder();

    struct OmcSourceParticle particle;

    /* Parallel, monoenergetic: nothing drawn at all */
    {
        struct OmcPencilSource pencil = pencilParallel();
        struct OmcSource source;
        omcPencilSourceAsSource(&pencil, &source);

        setRandomHistory(11);
        double straightAway = setRandom();

        setRandomHistory(11);
        memset(&particle, 0, sizeof(particle));
        source.sample(&source, 11, 0, &particle);
        double afterSampling = setRandom();

        CHECK(afterSampling == straightAway);
    }

    /* SSD, monoenergetic: two, and the same two whatever the field is */
    {
        struct OmcPencilSource pencil = pencilSsd(90.0, 1.0);
        struct OmcSource source;
        omcPencilSourceAsSource(&pencil, &source);

        setRandomHistory(11);
        setRandom();
        setRandom();
        double third = setRandom();

        setRandomHistory(11);
        memset(&particle, 0, sizeof(particle));
        source.sample(&source, 11, 0, &particle);
        double afterSampling = setRandom();

        CHECK(afterSampling == third);
    }
}

static void test_prepare_reports_the_dose_per_history(void) {

    setUpCylinder();

    struct OmcPencilSource pencil = pencilParallel();
    struct OmcSource source;
    omcPencilSourceAsSource(&pencil, &source);

    source.prepare(&source, 500);

    /* Nothing rescales a batch, and what comes out is divided by how many
     histories there were: dose per incident particle. */
    CHECK_CLOSE(source.batchScale, 1.0, 1e-15);
    CHECK_CLOSE(source.incidentFluence, 500.0, 1e-15);
}

static void test_check_refuses_what_it_cannot_sample(void) {

    setUpCylinder();
    catchingHost();

    struct OmcSource source;

    {
        struct OmcPencilSource pencil = pencilParallel();
        pencil.spectrum = NULL;
        omcPencilSourceAsSource(&pencil, &source);
        EXPECT_FAIL("ompMC:pencil:noSpectrum", source.check(&source));
    }

    {
        struct OmcPencilSource pencil = pencilParallel();
        pencil.charge = 3;
        omcPencilSourceAsSource(&pencil, &source);
        EXPECT_FAIL("ompMC:pencil:badCharge", source.check(&source));
    }

    {
        struct OmcPencilSource pencil = pencilSsd(-1.0, 0.0);
        omcPencilSourceAsSource(&pencil, &source);
        EXPECT_FAIL("ompMC:pencil:badSsd", source.check(&source));
    }

    {
        struct OmcPencilSource pencil = pencilSsd(0.0, 0.0);
        omcPencilSourceAsSource(&pencil, &source);
        EXPECT_FAIL("ompMC:pencil:badSsd", source.check(&source));
    }

    {
        struct OmcPencilSource pencil = pencilSsd(100.0, -2.0);
        omcPencilSourceAsSource(&pencil, &source);
        EXPECT_FAIL("ompMC:pencil:badFieldRadius", source.check(&source));
    }

    {
        struct OmcPencilSource pencil = pencilParallel();
        pencil.kind = (enum OmcPencilKind)7;
        omcPencilSourceAsSource(&pencil, &source);
        EXPECT_FAIL("ompMC:pencil:badKind", source.check(&source));
    }

    /* It aims down the axis of a cylinder, and there is no axis in a
     rectilinear phantom for it to aim down. */
    {
        struct OmcPencilSource pencil = pencilParallel();
        omcPencilSourceAsSource(&pencil, &source);

        geometry.mode = OMC_GEOM_CARTESIAN;
        EXPECT_FAIL("ompMC:pencil:notCylindrical", source.check(&source));
        geometry.mode = OMC_GEOM_CYLINDRICAL;
    }

    /* And the one that has to pass, so that the checks above are refusing
     something rather than everything */
    {
        struct OmcPencilSource pencil = pencilSsd(100.0, 2.0);
        omcPencilSourceAsSource(&pencil, &source);
        source.check(&source);
        CHECK(1);
    }

    omcSetHost(NULL);
}

/*******************************************************************************/

int main(void) {

    setvbuf(stdout, NULL, _IONBF, 0);

    /* The generator seeds itself from the input table like everything else
     in ompMC, so there has to be one even though nothing here is read from a
     file. */
    snprintf(input_items[0].key, BUFFER_SIZE, "%s", "rng seeds");
    snprintf(input_items[0].value, BUFFER_SIZE, "%s", "97 33");
    input_idx = 1;

    initRandom();

    /* The core announces the spectrum it built and the generator it seeded;
     neither is what this file is watching. */
    {
        struct OmcHost quiet = {silentLog, NULL, NULL};
        omcSetHost(&quiet);
    }

    printf("test_omc_source_pencil\n");

    RUN(test_a_parallel_pencil_starts_on_the_axis_above_the_face);
    RUN(test_the_charge_is_the_one_it_was_given);
    RUN(test_an_ssd_source_fills_the_field_it_was_given);
    RUN(test_a_field_radius_of_zero_means_the_whole_face);
    RUN(test_sampling_depends_only_on_the_history);
    RUN(test_the_draw_count_is_what_it_says);
    RUN(test_prepare_reports_the_dose_per_history);
    RUN(test_check_refuses_what_it_cannot_sample);

    omcSpectrumFree(&mono);
    cleanRandom();

    printf("\n%d tests, %d failures\n", tests_run, tests_failed);

    return tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

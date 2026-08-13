/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations

 Unit tests for the beamlet source, omc_source_beamlet: the particle one
 history starts with, and how a batch of histories is shared out among
 beamlets that each asked for a different share of the fluence.

 This is the source the matRad interface uses, so most of what it does has
 until now only been exercised through a whole dose calculation. Nothing here
 transports anything or needs a phantom: a source's job ends at a struct
 OmcSourceParticle, and everything after that is omcSourcePlace()'s. So this
 needs no data files and runs from wherever CTest starts it. The same source
 driving a real calculation is checked in test_omc_forward_phsp.c.
*****************************************************************************/

/* Before anything can include setjmp.h; see the comment in
 tests/test_omc_phsp.c for why MinGW's SEH-unwinding longjmp() is not what
 this harness wants. */
#if defined(__MINGW32__)
    #define __USE_MINGW_SETJMP_NON_SEH 1
#endif

#include "omc_host.h"
#include "omc_random.h"
#include "omc_source.h"
#include "omc_source_beamlet.h"
#include "omc_spectrum.h"
#include "omc_utilities.h"

#include <math.h>
#include <setjmp.h>
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
        printf("%-58s ", #fn);                                                \
        fn();                                                                 \
        printf("%s\n", tests_failed == _before ? "ok" : "FAILED");            \
    } while (0)

static void silentLog(int level, const char *message, void *user) {
    (void)level; (void)message; (void)user;
}

static jmp_buf fail_jmp;
static char fail_id[128];
static int fail_seen;
static int fail_armed;          /* is there a live setjmp() to come back to? */

static void catchingFail(const char *id, const char *message, void *user) {

    (void)user;
    snprintf(fail_id, sizeof(fail_id), "%s", id != NULL ? id : "");
    fail_seen = 1;

    /* A failure nobody was expecting. There is no live setjmp() to return
     to, and jumping into a frame that has already returned is undefined
     behaviour -- in practice a crash with nothing printed, which is a
     miserable way to be told that a call went wrong. Say what happened and
     stop instead. */
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

static void installFailCatcher(void) {

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

#define EXPECT_OK(call)                                                       \
    do {                                                                      \
        fail_seen = 0;                                                        \
        fail_armed = 1;                                                       \
        if (setjmp(fail_jmp) == 0) {                                          \
            call;                                                             \
        }                                                                     \
        fail_armed = 0;                                                       \
        if (fail_seen) {                                                      \
            printf("  FAIL %s:%d in %s: %s failed with %s\n",                 \
                   __FILE__, __LINE__, current_test, #call, fail_id);         \
            tests_failed++;                                                   \
        }                                                                     \
    } while (0)

/*******************************************************************************
* Fixtures
*
* One beam whose source sits a metre above the origin, and NBEAMLETS square
* centimetre beamlets laid side by side along x at z = 0, together covering x
* from -NBEAMLETS/2 to +NBEAMLETS/2 and y from -0.5 to 0.5. A beamlet is a
* corner plus two edge vectors, and a sampled point is
*
*     corner + r1*side1 + r2*side2,   r1, r2 uniform on [0,1)
*
* so beamlet i covers x from i - NBEAMLETS/2 to i + 1 - NBEAMLETS/2.
*******************************************************************************/

#define NBEAMLETS 4
#define SOURCE_Z (-100.0)

static int ibeam[NBEAMLETS];
static double xsource[1], ysource[1], zsource[1];
static double xcorner[NBEAMLETS], ycorner[NBEAMLETS], zcorner[NBEAMLETS];
static double xside1[NBEAMLETS], yside1[NBEAMLETS], zside1[NBEAMLETS];
static double xside2[NBEAMLETS], yside2[NBEAMLETS], zside2[NBEAMLETS];

static struct OmcBeamletSource beamlets;
static struct OmcSpectrum spectrum;

static void setUpBeamlets(void) {

    xsource[0] = 0.0;
    ysource[0] = 0.0;
    zsource[0] = SOURCE_Z;

    for (int i = 0; i < NBEAMLETS; i++) {
        ibeam[i] = 0;

        xcorner[i] = (double)i - 0.5*NBEAMLETS;
        ycorner[i] = -0.5;
        zcorner[i] = 0.0;

        xside1[i] = 1.0;  yside1[i] = 0.0;  zside1[i] = 0.0;
        xside2[i] = 0.0;  yside2[i] = 1.0;  zside2[i] = 0.0;
    }

    beamlets.nbeamlets = NBEAMLETS;
    beamlets.ibeam = ibeam;
    beamlets.xsource = xsource;
    beamlets.ysource = ysource;
    beamlets.zsource = zsource;
    beamlets.xcorner = xcorner;
    beamlets.ycorner = ycorner;
    beamlets.zcorner = zcorner;
    beamlets.xside1 = xside1;
    beamlets.yside1 = yside1;
    beamlets.zside1 = zside1;
    beamlets.xside2 = xside2;
    beamlets.yside2 = yside2;
    beamlets.zside2 = zside2;

    omcSpectrumMonoenergetic(&spectrum, 6.0);
}

static struct OmcBeamletSampler samplerFixture(void) {

    struct OmcBeamletSampler sampler;
    memset(&sampler, 0, sizeof(sampler));

    sampler.source = &beamlets;
    sampler.spectrum = &spectrum;
    sampler.charge = 0;
    sampler.geometry = OMC_SOURCE_POINT;
    sampler.gaussianWidth = 0.0;

    return sampler;
}

static struct OmcBeamletHistories historiesFixture(const double *weights) {

    struct OmcBeamletHistories histories;
    memset(&histories, 0, sizeof(histories));

    histories.sampler = samplerFixture();
    histories.weights = weights;

    return histories;
}

/* Where a particle's line crosses z = 0, which for this fixture is the plane
 the beamlets lie in. */
static void crossing(const struct OmcSourceParticle *p, double *x, double *y) {

    double t = (0.0 - p->z)/p->w;

    *x = p->x + t*p->u;
    *y = p->y + t*p->v;
}

/*******************************************************************************
* One history's particle
*******************************************************************************/

/* The shape of the model: a particle leaves the source point and flies
 through the beamlet it belongs to. Both halves matter -- a particle that
 started on the aperture instead would skip a metre of divergence, and one
 aimed anywhere else would not be that beamlet's. */
static void test_a_particle_starts_at_the_source_and_flies_through_its_beamlet(
        void) {

    struct OmcBeamletSampler sampler = samplerFixture();

    for (int i = 0; i < NBEAMLETS; i++) {
        for (int h = 0; h < 200; h++) {
            struct OmcSourceParticle p;
            memset(&p, 0, sizeof(p));

            setRandomHistory((uint64_t)(h + 1000*i));
            CHECK(omcBeamletProduce(&sampler, i, 1.0, &p) == 1);

            /* Started at the source, exactly. */
            CHECK(p.x == 0.0);
            CHECK(p.y == 0.0);
            CHECK(p.z == SOURCE_Z);

            /* Aimed by a unit vector, into the half space the beamlets are
             in. */
            CHECK_CLOSE(p.u*p.u + p.v*p.v + p.w*p.w, 1.0, 1e-12);
            CHECK(p.w > 0.0);

            /* And through its own beamlet, not its neighbour's. */
            double x, y;
            crossing(&p, &x, &y);

            CHECK(x >= xcorner[i] && x <= xcorner[i] + 1.0);
            CHECK(y >= -0.5 && y <= 0.5);
        }
    }
}

/* The aperture is sampled rather than tested against, which is the whole
 reason this source exists: every particle it makes is inside the beamlet, so
 none is thrown away. That only pays if the sampling covers the rectangle
 evenly, so check both edges of it get used and the mean lands in the middle. */
static void test_the_aperture_is_sampled_evenly(void) {

    struct OmcBeamletSampler sampler = samplerFixture();

    const int n = 4000;
    int quadrant[4] = {0, 0, 0, 0};
    double sumx = 0.0, sumy = 0.0;

    for (int h = 0; h < n; h++) {
        struct OmcSourceParticle p;
        memset(&p, 0, sizeof(p));

        setRandomHistory((uint64_t)h);
        omcBeamletProduce(&sampler, 1, 1.0, &p);

        double x, y;
        crossing(&p, &x, &y);

        /* Beamlet 1 covers x in [-1,0] and y in [-0.5,0.5]. */
        double cx = x + 0.5;
        double cy = y;

        sumx += cx;
        sumy += cy;

        quadrant[(cx > 0.0) + 2*(cy > 0.0)]++;
    }

    /* Centred, to within the standard error of a uniform over four thousand
     draws, which is about 0.005 in these units. */
    CHECK_CLOSE(sumx/n, 0.0, 0.02);
    CHECK_CLOSE(sumy/n, 0.0, 0.02);

    /* And no corner of the rectangle left out. */
    for (int q = 0; q < 4; q++) {
        CHECK(quadrant[q] > n/8);
    }
}

/* What the caller asked for comes back untouched: the charge, the weight and
 the energy the spectrum gave. */
static void test_the_particle_carries_what_it_was_asked_to(void) {

    struct OmcBeamletSampler sampler = samplerFixture();
    struct OmcSourceParticle p;

    memset(&p, 0, sizeof(p));
    setRandomHistory(7);
    omcBeamletProduce(&sampler, 0, 0.25, &p);

    CHECK(p.charge == 0);
    CHECK_CLOSE(p.energy, 6.0, 1e-15);
    CHECK_CLOSE(p.weight, 0.25, 1e-15);

    /* Electrons are what the charge is for. */
    sampler.charge = -1;
    memset(&p, 0, sizeof(p));
    setRandomHistory(7);
    omcBeamletProduce(&sampler, 0, 3.0, &p);

    CHECK(p.charge == -1);
    CHECK_CLOSE(p.weight, 3.0, 1e-15);
}

/* A spectrum with more than one energy in it is drawn from per history, and
 within the bounds it was built with. */
static void test_the_energy_comes_from_the_spectrum(void) {

    double upper[3] = {1.0, 2.0, 3.0};
    double counts[3] = {1.0, 2.0, 1.0};

    struct OmcSpectrum spread;
    omcSpectrumFromHistogram(&spread, upper, counts, 3, 0.0,
                             OMC_SPECTRUM_COUNTS_PER_BIN);

    struct OmcBeamletSampler sampler = samplerFixture();
    sampler.spectrum = &spread;

    double lowest = 1.0e30, highest = -1.0e30;

    for (int h = 0; h < 2000; h++) {
        struct OmcSourceParticle p;
        memset(&p, 0, sizeof(p));

        setRandomHistory((uint64_t)h);
        omcBeamletProduce(&sampler, 0, 1.0, &p);

        if (p.energy < lowest)  lowest = p.energy;
        if (p.energy > highest) highest = p.energy;
    }

    CHECK(lowest >= 0.0);
    CHECK(highest <= 3.0);
    CHECK(highest - lowest > 1.0);       /* it really did spread */

    omcSpectrumFree(&spread);
}

/* A gaussian source spreads where the particle starts over the collimator
 plane, which is what softens the penumbra. It does not move the point on the
 aperture the particle is aimed through -- that stays inside the beamlet -- so
 what changes is the direction, not the target. */
static void test_a_gaussian_source_spreads_the_start_and_not_the_target(void) {

    struct OmcBeamletSampler sampler = samplerFixture();
    sampler.geometry = OMC_SOURCE_GAUSSIAN;
    sampler.gaussianWidth = 0.3;

    const int n = 4000;
    double sumx = 0.0, sumy = 0.0, sumxx = 0.0, sumyy = 0.0;
    int moved = 0;

    for (int h = 0; h < n; h++) {
        struct OmcSourceParticle p;
        memset(&p, 0, sizeof(p));

        setRandomHistory((uint64_t)h);
        omcBeamletProduce(&sampler, 1, 1.0, &p);

        /* Spread across the plane, but not along the beam: the two plane
         vectors of this fixture are x and y. */
        CHECK(p.z == SOURCE_Z);

        if (p.x != 0.0 || p.y != 0.0) {
            moved++;
        }

        sumx += p.x;  sumxx += p.x*p.x;
        sumy += p.y;  sumyy += p.y*p.y;

        /* Still aimed through its own beamlet. */
        double x, y;
        crossing(&p, &x, &y);

        CHECK(x >= xcorner[1] && x <= xcorner[1] + 1.0);
        CHECK(y >= -0.5 && y <= 0.5);
    }

    CHECK(moved == n);

    /* Centred on the source point, with the width that was asked for. Three
     standard errors of the mean is about 0.014, and of the standard deviation
     about 0.010. */
    CHECK_CLOSE(sumx/n, 0.0, 0.03);
    CHECK_CLOSE(sumy/n, 0.0, 0.03);
    CHECK_CLOSE(sqrt(sumxx/n), 0.3, 0.03);
    CHECK_CLOSE(sqrt(sumyy/n), 0.3, 0.03);
}

/* A source geometry that is neither falls back to the point source rather
 than reading uninitialised memory. It cannot happen through the public
 interface -- check() refuses it before the histories start -- and that is
 exactly why the backstop needs a test of its own: nothing else reaches it,
 and failing from inside the parallel loop is not an option. */
static void test_an_unknown_geometry_falls_back_to_the_point_source(void) {

    struct OmcBeamletSampler point = samplerFixture();
    struct OmcBeamletSampler strange = samplerFixture();

    strange.geometry = (enum OmcSourceGeometry)99;

    for (int h = 0; h < 50; h++) {
        struct OmcSourceParticle a, b;
        memset(&a, 0, sizeof(a));
        memset(&b, 0, sizeof(b));

        setRandomHistory((uint64_t)h);
        omcBeamletProduce(&point, 2, 1.0, &a);

        setRandomHistory((uint64_t)h);
        omcBeamletProduce(&strange, 2, 1.0, &b);

        /* The same particle, down to the last bit: the fallback draws no
         random numbers of its own either. */
        CHECK(a.x == b.x && a.y == b.y && a.z == b.z);
        CHECK(a.u == b.u && a.v == b.v && a.w == b.w);
    }
}

/*******************************************************************************
* Sharing a batch out among the beamlets
*******************************************************************************/

/* Everything wrong with the beamlets is caught before the histories start,
 on the master thread, because omcFail() from inside the parallel loop would
 call the host from a place it cannot expect to be called from. */
static void test_the_source_looks_over_what_it_was_given(void) {

    double weights[NBEAMLETS] = {1.0, 1.0, 1.0, 1.0};
    struct OmcSource source;

    struct OmcBeamletHistories good = historiesFixture(weights);
    omcBeamletHistoriesAsSource(&good, &source);
    EXPECT_OK(source.check(&source));

    struct OmcBeamletHistories strange = historiesFixture(weights);
    strange.sampler.geometry = (enum OmcSourceGeometry)99;
    omcBeamletHistoriesAsSource(&strange, &source);
    EXPECT_FAIL("ompMC:forward:invalidSourceGeometry", source.check(&source));

    struct OmcBeamletHistories none = historiesFixture(weights);
    none.sampler.source = NULL;
    omcBeamletHistoriesAsSource(&none, &source);
    EXPECT_FAIL("ompMC:forward:noBeamlets", source.check(&source));

    struct OmcBeamletSource empty = beamlets;
    empty.nbeamlets = 0;

    struct OmcBeamletHistories emptied = historiesFixture(weights);
    emptied.sampler.source = &empty;
    omcBeamletHistoriesAsSource(&emptied, &source);
    EXPECT_FAIL("ompMC:forward:noBeamlets", source.check(&source));

    struct OmcBeamletHistories unweighted = historiesFixture(NULL);
    omcBeamletHistoriesAsSource(&unweighted, &source);
    EXPECT_FAIL("ompMC:forward:invalidWeight", source.check(&source));
}

/* How many histories each beamlet gets, and that every one of the batch goes
 to somebody. A beamlet asking for three times the fluence of another gets
 three times the histories. */
static void test_the_histories_are_shared_out_by_weight(void) {

    double weights[NBEAMLETS] = {3.0, 1.0, 0.0, 4.0};
    const int nperbatch = 800;

    struct OmcBeamletHistories histories = historiesFixture(weights);
    struct OmcSource source;
    omcBeamletHistoriesAsSource(&histories, &source);

    EXPECT_OK(source.prepare(&source, nperbatch));

    int count[NBEAMLETS] = {0, 0, 0, 0};

    for (int h = 0; h < nperbatch; h++) {
        struct OmcSourceParticle p;
        memset(&p, 0, sizeof(p));

        setRandomHistory((uint64_t)h);
        CHECK(source.sample(&source, (uint64_t)h, h, &p) == 1);

        /* Which beamlet it came from, read back off where it is aimed. */
        double x, y;
        crossing(&p, &x, &y);

        int i = (int)floor(x + 0.5*NBEAMLETS);
        if (i < 0) i = 0;
        if (i > NBEAMLETS - 1) i = NBEAMLETS - 1;

        count[i]++;
    }

    /* 3 : 1 : 0 : 4 out of eight, over eight hundred histories. The shares
     come out whole here, so this is exact rather than approximate. */
    CHECK(count[0] == 300);
    CHECK(count[1] == 100);
    CHECK(count[2] == 0);
    CHECK(count[3] == 400);

    /* And the batch is accounted for down to the last history. */
    CHECK(count[0] + count[1] + count[2] + count[3] == nperbatch);

    /* What the run is worth per batch: the fluence asked for, shared over the
     histories that carry it. */
    CHECK_CLOSE(source.batchScale, 8.0/nperbatch, 1e-15);
    CHECK_CLOSE(source.incidentFluence, 1.0, 1e-15);

    struct OmcBeamletStats stats;
    omcBeamletHistoriesStats(&histories, &stats);

    CHECK(stats.nweighted == 3);        /* the zero one never asked */
    CHECK(stats.nsampled == 3);
    CHECK_CLOSE(stats.totalWeight, 8.0, 1e-15);
    CHECK_CLOSE(stats.sampledWeight, 8.0, 1e-15);

    source.release(&source);
}

/* The shares almost never come out whole, and what is left over rides on the
 particle weight rather than on the counts -- so a batch always carries the
 fluence that was asked for, however the rounding fell. Weights that are not
 in any simple ratio are the point of this one. */
static void test_the_rounding_rides_on_the_weight_not_on_the_counts(void) {

    double weights[NBEAMLETS] = {0.37, 1.91, 0.08, 2.64};
    const int nperbatch = 501;

    struct OmcBeamletHistories histories = historiesFixture(weights);
    struct OmcSource source;
    omcBeamletHistoriesAsSource(&histories, &source);

    EXPECT_OK(source.prepare(&source, nperbatch));

    double carried[NBEAMLETS] = {0.0, 0.0, 0.0, 0.0};
    double total = 0.0;

    for (int h = 0; h < nperbatch; h++) {
        struct OmcSourceParticle p;
        memset(&p, 0, sizeof(p));

        setRandomHistory((uint64_t)h);
        source.sample(&source, (uint64_t)h, h, &p);

        double x, y;
        crossing(&p, &x, &y);

        int i = (int)floor(x + 0.5*NBEAMLETS);
        if (i < 0) i = 0;
        if (i > NBEAMLETS - 1) i = NBEAMLETS - 1;

        carried[i] += p.weight;
        total += p.weight;

        /* Near one, never far from it: the physical scale rides on the batch
         instead, so the transport keeps seeing the weights it always has. */
        CHECK(p.weight > 0.5 && p.weight < 2.0);
    }

    /* Each beamlet's share of the batch is its share of the fluence, exactly
     -- that is what the weight is there to fix up. */
    double sum = 0.0;
    for (int i = 0; i < NBEAMLETS; i++) {
        sum += weights[i];
    }

    for (int i = 0; i < NBEAMLETS; i++) {
        CHECK_CLOSE(carried[i]/total, weights[i]/sum, 1e-12);
    }

    /* And a batch is nperbatch histories' worth of fluence. */
    CHECK_CLOSE(total, (double)nperbatch, 1e-9);
    CHECK_CLOSE(total*source.batchScale, sum, 1e-12);

    source.release(&source);
}

/* A beamlet whose share rounds to nothing is dropped rather than given a
 history it has not earned -- and the dropping is reported rather than
 hidden, which is what the stats are for. */
static void test_a_beamlet_too_weak_for_a_history_is_dropped_and_reported(
        void) {

    /* Two hundred histories to share, and the third beamlet wants under a
     two-hundredth of the fluence -- but enough of it that leaving it out is
     worth warning about. */
    double weights[NBEAMLETS] = {1.0, 1.0, 0.005, 1.0};
    const int nperbatch = 200;

    struct OmcBeamletHistories histories = historiesFixture(weights);
    struct OmcSource source;
    omcBeamletHistoriesAsSource(&histories, &source);

    EXPECT_OK(source.prepare(&source, nperbatch));

    struct OmcBeamletStats stats;
    omcBeamletHistoriesStats(&histories, &stats);

    CHECK(stats.nweighted == 4);        /* all four asked */
    CHECK(stats.nsampled == 3);         /* three were heard */
    CHECK(stats.sampledWeight < stats.totalWeight);

    source.release(&source);
}

/* The answer outlives the working memory it was worked out in: release()
 gives that back before the engine returns, and the caller only gets to ask
 afterwards. */
static void test_the_stats_survive_the_run_that_produced_them(void) {

    double weights[NBEAMLETS] = {1.0, 2.0, 3.0, 4.0};

    struct OmcBeamletHistories histories = historiesFixture(weights);
    struct OmcSource source;
    omcBeamletHistoriesAsSource(&histories, &source);

    /* Before the run there is nothing to say, and saying zero is better than
     saying whatever the stack held. */
    struct OmcBeamletStats before;
    omcBeamletHistoriesStats(&histories, &before);

    CHECK(before.nweighted == 0);
    CHECK(before.nsampled == 0);
    CHECK(before.totalWeight == 0.0);
    CHECK(before.sampledWeight == 0.0);

    EXPECT_OK(source.prepare(&source, 400));
    source.release(&source);

    struct OmcBeamletStats after;
    omcBeamletHistoriesStats(&histories, &after);

    CHECK(after.nweighted == 4);
    CHECK(after.nsampled == 4);
    CHECK_CLOSE(after.totalWeight, 10.0, 1e-15);
    CHECK_CLOSE(after.sampledWeight, 10.0, 1e-15);

    /* And giving it back twice is not a second free of the same memory. */
    source.release(&source);
}

/* Weights that cannot be shared out are refused, with which one it was. */
static void test_weights_that_make_no_sense_are_refused(void) {

    struct OmcSource source;

    double negative[NBEAMLETS] = {1.0, -1.0, 1.0, 1.0};
    struct OmcBeamletHistories a = historiesFixture(negative);
    omcBeamletHistoriesAsSource(&a, &source);
    EXPECT_FAIL("ompMC:forward:invalidWeight", source.prepare(&source, 100));

    double notANumber[NBEAMLETS] = {1.0, 1.0, 1.0, 0.0};
    notANumber[3] = nan("");
    struct OmcBeamletHistories b = historiesFixture(notANumber);
    omcBeamletHistoriesAsSource(&b, &source);
    EXPECT_FAIL("ompMC:forward:invalidWeight", source.prepare(&source, 100));

    /* Each of these is finite; their sum is not, and every share below is a
     fraction of that sum. */
    double huge[NBEAMLETS] = {1.0e308, 1.0e308, 1.0e308, 1.0e308};
    struct OmcBeamletHistories c = historiesFixture(huge);
    omcBeamletHistoriesAsSource(&c, &source);
    EXPECT_FAIL("ompMC:forward:invalidWeight", source.prepare(&source, 100));

    /* Nothing asked for is not a calculation of nothing, it is a mistake. */
    double silent[NBEAMLETS] = {0.0, 0.0, 0.0, 0.0};
    struct OmcBeamletHistories d = historiesFixture(silent);
    omcBeamletHistoriesAsSource(&d, &source);
    EXPECT_FAIL("ompMC:forward:noWeight", source.prepare(&source, 100));
}

/* One beamlet is the case the whole batch goes to, and the case the binary
 search over the offsets has the least room to get right. */
static void test_a_single_beamlet_takes_the_whole_batch(void) {

    struct OmcBeamletSource one = beamlets;
    one.nbeamlets = 1;

    double weight[1] = {2.5};

    struct OmcBeamletHistories histories = historiesFixture(weight);
    histories.sampler.source = &one;

    struct OmcSource source;
    omcBeamletHistoriesAsSource(&histories, &source);

    EXPECT_OK(source.check(&source));
    EXPECT_OK(source.prepare(&source, 64));

    for (int h = 0; h < 64; h++) {
        struct OmcSourceParticle p;
        memset(&p, 0, sizeof(p));

        setRandomHistory((uint64_t)h);
        CHECK(source.sample(&source, (uint64_t)h, h, &p) == 1);

        double x, y;
        crossing(&p, &x, &y);

        CHECK(x >= xcorner[0] && x <= xcorner[0] + 1.0);
        CHECK_CLOSE(p.weight, 1.0, 1e-12);
    }

    CHECK_CLOSE(source.batchScale, 2.5/64.0, 1e-15);

    source.release(&source);
}

/* More beamlets than histories: every history still belongs to somebody, and
 nobody gets two. This is the case the cumulative walk is most likely to
 misplace, and the case a fluence map of many small spots actually hits. */
static void test_more_beamlets_than_histories_still_adds_up(void) {

    enum { MANY = 64 };

    static int manyBeam[MANY];
    static double manyCorner[MANY], manyOther[MANY], manyZero[MANY];
    static double manySide1[MANY], manySide2[MANY];
    static double weights[MANY];

    for (int i = 0; i < MANY; i++) {
        manyBeam[i] = 0;
        manyCorner[i] = (double)i - 0.5*MANY;
        manyOther[i] = -0.5;
        manyZero[i] = 0.0;
        manySide1[i] = 1.0;
        manySide2[i] = 1.0;
        weights[i] = 1.0;
    }

    struct OmcBeamletSource many;
    many.nbeamlets = MANY;
    many.ibeam = manyBeam;
    many.xsource = xsource;
    many.ysource = ysource;
    many.zsource = zsource;
    many.xcorner = manyCorner;
    many.ycorner = manyOther;
    many.zcorner = manyZero;
    many.xside1 = manySide1;
    many.yside1 = manyZero;
    many.zside1 = manyZero;
    many.xside2 = manyZero;
    many.yside2 = manySide2;
    many.zside2 = manyZero;

    struct OmcBeamletHistories histories = historiesFixture(weights);
    histories.sampler.source = &many;

    struct OmcSource source;
    omcBeamletHistoriesAsSource(&histories, &source);

    const int nperbatch = 40;           /* fewer histories than beamlets */

    EXPECT_OK(source.prepare(&source, nperbatch));

    int seen[MANY];
    memset(seen, 0, sizeof(seen));

    for (int h = 0; h < nperbatch; h++) {
        struct OmcSourceParticle p;
        memset(&p, 0, sizeof(p));

        setRandomHistory((uint64_t)h);
        CHECK(source.sample(&source, (uint64_t)h, h, &p) == 1);

        double x, y;
        crossing(&p, &x, &y);

        int i = (int)floor(x + 0.5*MANY);
        CHECK(i >= 0 && i < MANY);

        if (i >= 0 && i < MANY) {
            seen[i]++;
        }
    }

    /* Sixty four beamlets asking for the same thing and forty histories to
     give: forty of them get one each and the rest get none, rather than a few
     of them getting two while others starve. */
    int given = 0;
    for (int i = 0; i < MANY; i++) {
        CHECK(seen[i] <= 1);
        given += seen[i];
    }

    CHECK(given == nperbatch);

    struct OmcBeamletStats stats;
    omcBeamletHistoriesStats(&histories, &stats);

    CHECK(stats.nweighted == MANY);
    CHECK(stats.nsampled == nperbatch);
    CHECK_CLOSE(stats.totalWeight, (double)MANY, 1e-12);
    CHECK_CLOSE(stats.sampledWeight, (double)nperbatch, 1e-12);

    source.release(&source);
}

int main(void) {

    /* Unbuffered, so a test that brings the process down still leaves behind
     the list of the ones that got that far. */
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("ompMC beamlet source tests\n\n");

    omcSetInputValue("rng seeds", "97 33");
    initRandom();

    installFailCatcher();
    setUpBeamlets();

    RUN(test_a_particle_starts_at_the_source_and_flies_through_its_beamlet);
    RUN(test_the_aperture_is_sampled_evenly);
    RUN(test_the_particle_carries_what_it_was_asked_to);
    RUN(test_the_energy_comes_from_the_spectrum);
    RUN(test_a_gaussian_source_spreads_the_start_and_not_the_target);
    RUN(test_an_unknown_geometry_falls_back_to_the_point_source);
    RUN(test_the_source_looks_over_what_it_was_given);
    RUN(test_the_histories_are_shared_out_by_weight);
    RUN(test_the_rounding_rides_on_the_weight_not_on_the_counts);
    RUN(test_a_beamlet_too_weak_for_a_history_is_dropped_and_reported);
    RUN(test_the_stats_survive_the_run_that_produced_them);
    RUN(test_weights_that_make_no_sense_are_refused);
    RUN(test_a_single_beamlet_takes_the_whole_batch);
    RUN(test_more_beamlets_than_histories_still_adds_up);

    omcSetHost(NULL);
    cleanRandom();

    printf("\n%d test groups, %d check failures\n", tests_run, tests_failed);

    return tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

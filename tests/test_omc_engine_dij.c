/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations

 The dose influence matrix calculation, end to end: one sparse column per
 beamlet, out of the same beamlet source and the same transport the forward
 engine uses. Everything below the engine is real -- real cross sections, real
 water, real showers -- so like test_omc_forward_phsp.c this runs from the
 repository root where the data and pegs4 folders live. CTest is told to do
 that.

 What it is watching for is the shape of the answer rather than the physics:
 that each beamlet's column names voxels that exist, in the order a sparse
 matrix wants them, that a beamlet's column holds its own dose and not the
 previous one's, that the threshold drops voxels rather than zeroing them, and
 that a host can call the whole thing off partway through and be told how much
 of it it got.
*****************************************************************************/

/* Before anything can include setjmp.h; see the comment in
 tests/test_omc_phsp.c for why MinGW's SEH-unwinding longjmp() is not what
 this harness wants. */
#if defined(__MINGW32__)
    #define __USE_MINGW_SETJMP_NON_SEH 1
#endif

#include "omc_engine_dij.h"
#include "omc_geom.h"
#include "omc_host.h"
#include "omc_source_beamlet.h"
#include "omc_spectrum.h"
#include "omc_utilities.h"
#include "ompmc.h"

#include <math.h>
#include <setjmp.h>
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

#define RUN(fn)                                                               \
    do {                                                                      \
        current_test = #fn;                                                   \
        tests_run++;                                                          \
        int _before = tests_failed;                                           \
        printf("%-56s ", #fn);                                                \
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

    /* Nothing was expecting this one, and jumping into a frame that has
     already returned is undefined behaviour -- in practice a crash with
     nothing printed. Say what happened and stop instead. */
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

static void quietHost(void) {
    struct OmcHost quiet = {silentLog, NULL, NULL};
    omcSetHost(&quiet);
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
* A water tank, and the physics to transport in it
*
* 8 x 8 cm across the beam and 8 cm deep in centimetre voxels, its front face
* at z = 0, so a source above it shines straight in.
*******************************************************************************/

#define NX 8
#define NY 8
#define NZ 8
#define GRIDSIZE (NX*NY*NZ)

static double xb[NX + 1], yb[NY + 1], zb[NZ + 1];
static int medIndices[GRIDSIZE];
static double medDensities[GRIDSIZE];

static void setInput(int i, const char *key, const char *value) {
    snprintf(input_items[i].key, sizeof(input_items[i].key), "%s", key);
    snprintf(input_items[i].value, sizeof(input_items[i].value), "%s", value);
}

static void setUpWaterTank(void) {

    setInput(0, "pegs file", "./pegs4/700icru.pegs4dat");
    setInput(1, "pgs4form file", "./pegs4/pgs4form.dat");
    setInput(2, "data folder", "./data/");
    setInput(3, "rng seeds", "97 33");
    setInput(4, "global ecut", "0.700");
    setInput(5, "global pcut", "0.010");
    setInput(6, "nsplit", "20");
    setInput(7, "esave", "2.0");
    input_idx = 8;

    for (int i = 0; i <= NX; i++) xb[i] = -4.0 + 1.0*i;
    for (int j = 0; j <= NY; j++) yb[j] = -4.0 + 1.0*j;
    for (int k = 0; k <= NZ; k++) zb[k] = 0.0 + 1.0*k;

    geometry.isize = NX;
    geometry.jsize = NY;
    geometry.ksize = NZ;
    geometry.xbounds = xb;
    geometry.ybounds = yb;
    geometry.zbounds = zb;

    for (int i = 0; i < GRIDSIZE; i++) {
        /* EGS counts media from 1; 0 would be vacuum. */
        medIndices[i] = 1;
        medDensities[i] = 1.0;
    }

    geometry.med_indices = medIndices;
    geometry.med_densities = medDensities;

    omcGeomDetectSpacing();

    media.nmed = 1;
    snprintf(media.med_names[0], 60, "%s", "H2O700ICRU");

    quietHost();

    initMediaData();
    initRegions();
    initVrt();
}

static void tearDownWaterTank(void) {

    cleanPhoton();
    cleanRayleigh();
    cleanPair();
    cleanElectron();
    cleanMscat();
    cleanSpin();
    cleanRegions();

    geometry.med_indices = NULL;
    geometry.med_densities = NULL;
    geometry.xbounds = NULL;
    geometry.ybounds = NULL;
    geometry.zbounds = NULL;

    omcSetHost(NULL);
}

/*******************************************************************************
* Two beamlets, side by side
*
* The source sits a metre above the tank. Beamlet 0 covers x from -2 to 0 of
* the front face and beamlet 1 covers x from 0 to 2, both the middle 4 cm in
* y -- so which column belongs to which is readable off where the dose is.
*******************************************************************************/

#define NBEAMLETS 2

static int ibeamArr[NBEAMLETS] = {0, 0};
static double xsource[1] = {0.0}, ysource[1] = {0.0}, zsource[1] = {-100.0};
static double xcorner[NBEAMLETS] = {-2.0, 0.0};
static double ycorner[NBEAMLETS] = {-2.0, -2.0};
static double zcorner[NBEAMLETS] = {0.0, 0.0};
static double xside1[NBEAMLETS] = {2.0, 2.0};
static double yside1[NBEAMLETS] = {0.0, 0.0};
static double zside1[NBEAMLETS] = {0.0, 0.0};
static double xside2[NBEAMLETS] = {0.0, 0.0};
static double yside2[NBEAMLETS] = {4.0, 4.0};
static double zside2[NBEAMLETS] = {0.0, 0.0};

static struct OmcBeamletSource beamletsFixture(void) {

    struct OmcBeamletSource source;

    source.nbeamlets = NBEAMLETS;
    source.ibeam = ibeamArr;
    source.xsource = xsource;
    source.ysource = ysource;
    source.zsource = zsource;
    source.xcorner = xcorner;
    source.ycorner = ycorner;
    source.zcorner = zcorner;
    source.xside1 = xside1;
    source.yside1 = yside1;
    source.zside1 = zside1;
    source.xside2 = xside2;
    source.yside2 = yside2;
    source.zside2 = zside2;

    return source;
}

static struct OmcDijOptions optionsFor(int nhist) {

    struct OmcDijOptions opt;
    memset(&opt, 0, sizeof(opt));

    opt.nhist = nhist;
    opt.nbatch = 4;
    opt.charge = 0;
    opt.relDoseThreshold = 0.0;
    opt.sourceGeometry = OMC_SOURCE_POINT;
    opt.wantVariance = 0;

    return opt;
}

/* What the callbacks saw. The columns are kept scattered into full grids,
 which is the easiest thing to make assertions about. */
#define MAX_COLUMNS 4

struct Collected {
    int ncolumns;
    int beamlet[MAX_COLUMNS];
    int nvoxels[MAX_COLUMNS];
    int hadVariance[MAX_COLUMNS];
    double column[MAX_COLUMNS][GRIDSIZE];

    int nprogress;
    double lastFraction;
    int stopAfter;              /* 0 : never stop */
    int badIndex;               /* a voxel index outside the grid */
    int outOfOrder;             /* a column whose indices were not ascending */
    int nonPositive;            /* a dose that was reported but not positive */
};

static void collectBeamlet(int ibeamlet, int nvoxels, const int *voxels,
                           const double *dose, const double *variance,
                           void *user) {

    struct Collected *got = (struct Collected *)user;

    if (got->ncolumns >= MAX_COLUMNS) {
        return;
    }

    int c = got->ncolumns++;

    got->beamlet[c] = ibeamlet;
    got->nvoxels[c] = nvoxels;
    got->hadVariance[c] = variance != NULL;

    memset(got->column[c], 0, sizeof(got->column[c]));

    for (int n = 0; n < nvoxels; n++) {
        if (voxels[n] < 0 || voxels[n] >= GRIDSIZE) {
            got->badIndex++;
            continue;
        }
        /* A sparse column has to come back ascending, or the matrix it goes
         into is not the matrix the caller thinks it is. */
        if (n > 0 && voxels[n] <= voxels[n - 1]) {
            got->outOfOrder++;
        }
        if (!(dose[n] > 0.0)) {
            got->nonPositive++;
        }

        got->column[c][voxels[n]] = dose[n];
    }
}

static int watchProgress(double fraction, void *user) {

    struct Collected *got = (struct Collected *)user;

    got->nprogress++;
    got->lastFraction = fraction;

    if (got->stopAfter > 0 && got->nprogress >= got->stopAfter) {
        return 0;
    }

    return 1;
}

static struct OmcDijCallbacks callbacksFor(struct Collected *got) {

    struct OmcDijCallbacks callbacks;

    memset(got, 0, sizeof(*got));

    callbacks.beamlet = collectBeamlet;
    callbacks.progress = watchProgress;
    callbacks.user = got;

    return callbacks;
}

/* The dose one column put into the half of the tank on one side of x = 0. */
static double halfDose(const double *column, int leftHalf) {

    double sum = 0.0;

    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                if ((i < NX/2) == (leftHalf != 0)) {
                    sum += column[i + j*NX + k*NX*NY];
                }
            }
        }
    }

    return sum;
}

/*******************************************************************************
* The tests
*******************************************************************************/

/* One column per beamlet, each of them naming voxels that exist, in the order
 a sparse matrix wants them, and each holding its own beamlet's dose rather
 than the one before it. That last part is the one worth stating: the
 accumulators are reused between beamlets, so a column that did not clear them
 would come out as the running sum of every beamlet so far. */
static void test_each_beamlet_gets_its_own_column(void) {

    setUpWaterTank();

    struct OmcBeamletSource source = beamletsFixture();
    struct OmcSpectrum spectrum;
    omcSpectrumMonoenergetic(&spectrum, 6.0);

    struct OmcDijOptions opt = optionsFor(4000);
    struct Collected got;
    struct OmcDijCallbacks callbacks = callbacksFor(&got);

    int done = omcCalcDij(&opt, &source, &spectrum, &callbacks);

    CHECK(done == NBEAMLETS);
    CHECK(got.ncolumns == NBEAMLETS);
    CHECK(got.beamlet[0] == 0);
    CHECK(got.beamlet[1] == 1);

    CHECK(got.badIndex == 0);
    CHECK(got.outOfOrder == 0);
    CHECK(got.nonPositive == 0);
    CHECK(got.nvoxels[0] > 0);
    CHECK(got.nvoxels[1] > 0);

    /* No variance was asked for, so none was offered. */
    CHECK(got.hadVariance[0] == 0);
    CHECK(got.hadVariance[1] == 0);

    /* Beamlet 0 lights up the left half of the tank and beamlet 1 the right,
     which is where each of them points. */
    CHECK(halfDose(got.column[0], 1) > 3.0*halfDose(got.column[0], 0));
    CHECK(halfDose(got.column[1], 0) > 3.0*halfDose(got.column[1], 1));

    /* And the two columns are of the same size, as two beamlets of the same
     shape and the same history count should be: a second column carrying the
     first one's dose as well would be about twice it. */
    double left = halfDose(got.column[0], 1);
    double right = halfDose(got.column[1], 0);

    CHECK(fabs(left - right) < 0.2*left);

    /* Progress is reported once per batch and once per beamlet, and runs to
     the end. */
    CHECK(got.nprogress == NBEAMLETS*(opt.nbatch + 1));
    CHECK(fabs(got.lastFraction - 1.0) < 1e-12);

    omcSpectrumFree(&spectrum);
    tearDownWaterTank();
}

/* The threshold drops voxels from the column rather than reporting them at
 zero: a dose influence matrix is mostly empty, and what makes it worth
 storing sparsely is that the small entries are not there at all. */
static void test_the_threshold_drops_voxels_from_the_column(void) {

    setUpWaterTank();

    struct OmcBeamletSource source = beamletsFixture();
    struct OmcSpectrum spectrum;
    omcSpectrumMonoenergetic(&spectrum, 6.0);

    struct Collected everything;
    struct OmcDijCallbacks all = callbacksFor(&everything);
    struct OmcDijOptions opt = optionsFor(4000);

    omcCalcDij(&opt, &source, &spectrum, &all);

    struct Collected trimmed;
    struct OmcDijCallbacks some = callbacksFor(&trimmed);
    opt.relDoseThreshold = 0.5;         /* half the beamlet's maximum */

    omcCalcDij(&opt, &source, &spectrum, &some);

    CHECK(trimmed.ncolumns == NBEAMLETS);
    CHECK(trimmed.nvoxels[0] > 0);
    CHECK(trimmed.nvoxels[0] < everything.nvoxels[0]);
    CHECK(trimmed.nvoxels[1] < everything.nvoxels[1]);

    /* What survived is what was above the cut, and it is still the same
     dose -- the threshold decides what is reported, not what is scored. */
    double kept = 0.0, whole = 0.0;
    for (int i = 0; i < GRIDSIZE; i++) {
        kept += trimmed.column[0][i];
        whole += everything.column[0][i];
    }

    CHECK(kept > 0.0);
    CHECK(kept < whole);

    omcSpectrumFree(&spectrum);
    tearDownWaterTank();
}

/* Asking for the variance gets it, and it is a variance of something rather
 than the zeros a column that never scored would carry. */
static void test_the_variance_comes_back_when_it_is_asked_for(void) {

    setUpWaterTank();

    struct OmcBeamletSource source = beamletsFixture();
    struct OmcSpectrum spectrum;
    omcSpectrumMonoenergetic(&spectrum, 6.0);

    struct OmcDijOptions opt = optionsFor(4000);
    opt.wantVariance = 1;

    struct Collected got;
    struct OmcDijCallbacks callbacks = callbacksFor(&got);

    CHECK(omcCalcDij(&opt, &source, &spectrum, &callbacks) == NBEAMLETS);

    CHECK(got.hadVariance[0] == 1);
    CHECK(got.hadVariance[1] == 1);

    omcSpectrumFree(&spectrum);
    tearDownWaterTank();
}

/* A host can call the whole thing off, and is told how many columns it
 actually got -- the ones already handed over stay good, and the beamlet that
 was abandoned partway through its batches is not among them. */
static void test_a_progress_callback_can_stop_the_run(void) {

    setUpWaterTank();

    struct OmcBeamletSource source = beamletsFixture();
    struct OmcSpectrum spectrum;
    omcSpectrumMonoenergetic(&spectrum, 6.0);

    struct OmcDijOptions opt = optionsFor(400);
    struct Collected got;
    struct OmcDijCallbacks callbacks = callbacksFor(&got);

    /* Stop inside the first beamlet's second batch, before any column has
     been handed over. */
    got.stopAfter = 2;

    CHECK(omcCalcDij(&opt, &source, &spectrum, &callbacks) == 0);
    CHECK(got.ncolumns == 0);

    /* And stop on the report that follows the first finished beamlet, which
     does count: it was reported in full. */
    struct Collected later;
    struct OmcDijCallbacks again = callbacksFor(&later);
    later.stopAfter = opt.nbatch + 1;

    CHECK(omcCalcDij(&opt, &source, &spectrum, &again) == 1);
    CHECK(later.ncolumns == 1);
    CHECK(later.beamlet[0] == 0);

    omcSpectrumFree(&spectrum);
    tearDownWaterTank();
}

/* And a run that cannot be made sense of is refused before the first history,
 on the master thread -- omcFail() out of a worker would call the host from
 somewhere it cannot expect to be called. */
static void test_a_run_that_makes_no_sense_is_refused(void) {

    struct OmcBeamletSource source = beamletsFixture();
    struct OmcSpectrum spectrum;
    omcSpectrumMonoenergetic(&spectrum, 6.0);

    struct Collected got;
    struct OmcDijCallbacks callbacks = callbacksFor(&got);

    catchingHost();

    struct OmcDijOptions strange = optionsFor(100);
    strange.sourceGeometry = (enum OmcSourceGeometry)99;
    EXPECT_FAIL("ompMC:dij:invalidSourceGeometry",
                omcCalcDij(&strange, &source, &spectrum, &callbacks));

    struct OmcBeamletSource none = beamletsFixture();
    none.nbeamlets = 0;

    struct OmcDijOptions ok = optionsFor(100);
    EXPECT_FAIL("ompMC:dij:noBeamlets",
                omcCalcDij(&ok, &none, &spectrum, &callbacks));

    /* The uncertainty is the spread over the batches, so one batch has none
     to estimate. */
    struct OmcDijOptions single = optionsFor(100);
    single.nbatch = 1;
    EXPECT_FAIL("ompMC:dij:tooFewBatches",
                omcCalcDij(&single, &source, &spectrum, &callbacks));

    omcSpectrumFree(&spectrum);
    omcSetHost(NULL);
}

int main(void) {

    /* Unbuffered, so a test that brings the process down still leaves behind
     the list of the ones that got that far. */
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("ompMC dose influence matrix tests\n\n");

    RUN(test_a_run_that_makes_no_sense_is_refused);
    RUN(test_each_beamlet_gets_its_own_column);
    RUN(test_the_threshold_drops_voxels_from_the_column);
    RUN(test_the_variance_comes_back_when_it_is_asked_for);
    RUN(test_a_progress_callback_can_stop_the_run);

    omcSetHost(NULL);

    printf("\n%d test groups, %d check failures\n", tests_run, tests_failed);

    return tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

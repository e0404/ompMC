/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations

 The phase space forward calculation, end to end: particles out of a phase
 space, through the transport, into a dose cube. Everything below the engine is
 real -- real cross sections, real water, real showers -- so unlike the source
 tests in test_omc_source_phsp.c this one has to run from the repository root
 where the data and pegs4 folders live. CTest is told to do that.

 What it is really watching for is the wiring rather than the physics: that a
 history whose particle misses the phantom is skipped rather than showered on
 whatever the stack held from last time, that the histories which produce
 nothing still count towards the fluence the result is divided by, and that a
 phase space pointed away from the phantom gives nothing rather than something.
 The physics gets one check of its own, a broad one: a 6 MV-ish beam into water
 has to build up to a maximum below the surface and fall away after it.
*****************************************************************************/

#include "omc_collimator.h"
#include "omc_engine_forward.h"
#include "omc_geom.h"
#include "omc_host.h"
#include "omc_phsp.h"
#include "omc_source_phsp.h"
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
        printf("%-50s ", #fn);                                                \
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

/*******************************************************************************
* A water tank, and the physics to transport in it
*
* 10 x 10 cm across the beam and 10 cm deep in half centimetre slabs, its front
* face at z = 0 so a phase space at negative z shines straight into it.
*******************************************************************************/

#define NX 10
#define NY 10
#define NZ 20
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

    for (int i = 0; i <= NX; i++) xb[i] = -5.0 + 1.0*i;
    for (int j = 0; j <= NY; j++) yb[j] = -5.0 + 1.0*j;
    for (int k = 0; k <= NZ; k++) zb[k] = 0.0 + 0.5*k;

    geometry.isize = NX;
    geometry.jsize = NY;
    geometry.ksize = NZ;
    geometry.xbounds = xb;
    geometry.ybounds = yb;
    geometry.zbounds = zb;

    for (int i = 0; i < GRIDSIZE; i++) {
        /* EGS counts media from 1; 0 would be vacuum, and a tank of vacuum
         absorbs nothing, which is a confusing way to find this out. */
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
* A phase space built in memory, the same way test_omc_source_phsp.c does it
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

/* n photons of the given energy, spread across the field, all heading along
 +z into the tank from a centimetre above it. @p aimAway turns them round. */
static void makeBeam(struct Made *made, int n, double energy, int aimAway) {

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
        unsigned char *at = made->bytes + (size_t)i*RECORD_LENGTH;

        /* A negative type is how the format says w points backwards. */
        at[0] = (unsigned char)((aimAway ? -OMC_PHSP_PHOTON
                                         : OMC_PHSP_PHOTON) & 0xFF);
        putLeFloat(at + 1, (float)-energy);     /* opens a history */
        putLeFloat(at + 5, (float)(-2.0 + 4.0*(double)(i % 5)/4.0));
        putLeFloat(at + 9, (float)(-2.0 + 4.0*(double)(i % 3)/2.0));
        putLeFloat(at + 13, (float)-1.0);       /* a centimetre above */
        putLeFloat(at + 17, 0.0f);              /* u */
        putLeFloat(at + 21, 0.0f);              /* v, so w is +-1 */
        putLeFloat(at + 25, 1.0f);              /* weight */
    }
}

static struct OmcForwardOptions optionsFor(int nhist) {

    struct OmcForwardOptions opt;
    memset(&opt, 0, sizeof(opt));

    opt.nhist = nhist;
    opt.nbatch = 4;
    opt.outputDose = 0;         /* deposited energy, no density scaling */

    return opt;
}

/* The phase space, dressed as a source the engine will take. */
static struct OmcPhspSampler samplerFor(struct OmcPhsp *phsp) {

    struct OmcPhspSampler sampler;
    memset(&sampler, 0, sizeof(sampler));

    sampler.phsp = phsp;
    sampler.order = OMC_PHSP_REPLAY;
    sampler.first = 0;
    omcPhspTransformIdentity(&sampler.transform);

    return sampler;
}

/*******************************************************************************
* The tests
*******************************************************************************/

/* A 6 MeV photon beam into water has to build up to a maximum below the
 surface and fall away past it, which is the one thing about a photon depth
 dose curve that no amount of geometry confusion can fake. */
static void test_a_photon_beam_builds_up_and_falls_off(void) {

    setUpWaterTank();

    struct Made made;
    makeBeam(&made, 16, 6.0, 0);

    struct OmcForwardOptions opt = optionsFor(20000);
    struct OmcForwardSummary summary;

    double *dose = malloc(GRIDSIZE*sizeof(double));
    double *unc = malloc(GRIDSIZE*sizeof(double));

    struct OmcPhspSampler sampler = samplerFor(&made.phsp);
    struct OmcSource source;
    omcPhspSamplerAsSource(&sampler, &source);

    int finished = omcCalcForward(&opt, &source, NULL, dose, unc, NULL, &summary);

    CHECK(finished == 1);
    CHECK(summary.nhist == 20000);
    CHECK(summary.started == 20000);        /* all of them are aimed in */
    CHECK(summary.energyFraction > 0.0 && summary.energyFraction < 1.0);

    /* The depth dose down the middle of the tank. */
    double depth[NZ];
    for (int k = 0; k < NZ; k++) {
        double sum = 0.0;
        for (int j = NY/2 - 1; j <= NY/2; j++) {
            for (int i = NX/2 - 1; i <= NX/2; i++) {
                sum += dose[i + j*NX + k*NX*NY];
            }
        }
        depth[k] = sum/4.0;
    }

    int kmax = 0;
    for (int k = 0; k < NZ; k++) {
        if (depth[k] > depth[kmax]) {
            kmax = k;
        }
    }

    CHECK(depth[kmax] > 0.0);

    /* Below the surface rather than in the very first slab, and well before
     the back of the tank. */
    CHECK(kmax > 0);
    CHECK(kmax < NZ/2);

    /* And falling by the far end. */
    CHECK(depth[NZ - 1] < depth[kmax]);

    free(dose);
    free(unc);
    tearDownWaterTank();
}

/* Every particle pointing away from the tank leaves it with no dose in it,
 and the engine saying as much rather than showering whatever the stack held
 from the history before. */
static void test_a_beam_aimed_away_deposits_nothing(void) {

    setUpWaterTank();

    struct Made made;
    makeBeam(&made, 16, 6.0, 1);        /* turned round */

    struct OmcForwardOptions opt = optionsFor(4000);
    struct OmcForwardSummary summary;

    double *dose = malloc(GRIDSIZE*sizeof(double));

    struct OmcPhspSampler sampler = samplerFor(&made.phsp);
    struct OmcSource source;
    omcPhspSamplerAsSource(&sampler, &source);

    int finished = omcCalcForward(&opt, &source, NULL, dose, NULL, NULL, &summary);

    CHECK(finished == 1);
    CHECK(summary.started == 0);
    CHECK(summary.energyFraction == 0.0);

    double total = 0.0;
    for (int i = 0; i < GRIDSIZE; i++) {
        total += dose[i];
    }
    CHECK(total == 0.0);

    free(dose);
    tearDownWaterTank();
}

/* The result is per history, and the histories that produced nothing are part
 of that count: they are fluence the phase space stands for that happened to
 miss. So a phase space of which half misses gives half the dose per history
 of one where all of it arrives, rather than the same. */
static void test_histories_that_miss_still_count(void) {

    setUpWaterTank();

    struct Made all;
    struct Made half;

    makeBeam(&all, 16, 6.0, 0);
    makeBeam(&half, 16, 6.0, 0);

    /* Turn every other particle of the second one round. */
    for (int i = 1; i < 16; i += 2) {
        half.bytes[(size_t)i*RECORD_LENGTH] =
            (unsigned char)((-OMC_PHSP_PHOTON) & 0xFF);
    }

    struct OmcForwardOptions opt = optionsFor(20000);
    struct OmcForwardSummary summaryAll;
    struct OmcForwardSummary summaryHalf;

    double *doseAll = malloc(GRIDSIZE*sizeof(double));
    double *doseHalf = malloc(GRIDSIZE*sizeof(double));

    struct OmcPhspSampler samplerAll = samplerFor(&all.phsp);
    struct OmcPhspSampler samplerHalf = samplerFor(&half.phsp);
    struct OmcSource sourceAll;
    struct OmcSource sourceHalf;

    omcPhspSamplerAsSource(&samplerAll, &sourceAll);
    omcPhspSamplerAsSource(&samplerHalf, &sourceHalf);

    omcCalcForward(&opt, &sourceAll, NULL, doseAll, NULL, NULL, &summaryAll);
    omcCalcForward(&opt, &sourceHalf, NULL, doseHalf, NULL, NULL, &summaryHalf);

    CHECK(summaryAll.started == 20000);
    CHECK(summaryHalf.started == 10000);

    double totalAll = 0.0, totalHalf = 0.0;
    for (int i = 0; i < GRIDSIZE; i++) {
        totalAll += doseAll[i];
        totalHalf += doseHalf[i];
    }

    CHECK(totalAll > 0.0);
    CHECK(totalHalf > 0.0);

    /* Half the particles arriving, so half the energy per history. Loose
     bounds: the two runs draw different particles, so this is about the
     factor being a half rather than a one. */
    double ratio = totalHalf/totalAll;
    if (!(ratio > 0.4 && ratio < 0.6)) {
        printf("  FAIL %s: half the beam gave a ratio of %.4f, expected "
               "about 0.5\n", current_test, ratio);
        tests_failed++;
    }

    free(doseAll);
    free(doseHalf);
    tearDownWaterTank();
}

/*******************************************************************************
* The collimator
*
* The mask sits at z = -0.5, between where the phase space puts its particles
* and the front face of the tank, which is where a jaw would be.
*******************************************************************************/

/* A mask that lets everything through has to leave the run alone. It is
 worth its own test because the mask is asked about every history, and a
 lookup that quietly stopped particles at a cell edge, or that drew a random
 number, would show up here and nowhere else. */
static void test_an_open_mask_changes_nothing(void) {

    setUpWaterTank();

    struct Made made;
    makeBeam(&made, 16, 6.0, 0);

    struct OmcForwardOptions opt = optionsFor(20000);
    struct OmcPhspSampler sampler = samplerFor(&made.phsp);
    struct OmcSource source;
    omcPhspSamplerAsSource(&sampler, &source);

    double *bare = malloc(GRIDSIZE*sizeof(double));
    double *masked = malloc(GRIDSIZE*sizeof(double));
    struct OmcForwardSummary bareSummary;
    struct OmcForwardSummary maskedSummary;

    omcCalcForward(&opt, &source, NULL, bare, NULL, NULL, &bareSummary);

    double open[1] = {1.0};
    struct OmcApertureMask mask;
    memset(&mask, 0, sizeof(mask));
    mask.z = -0.5;
    mask.x0 = -50.0;
    mask.y0 = -50.0;
    mask.dx = 100.0;
    mask.dy = 100.0;
    mask.nx = 1;
    mask.ny = 1;
    mask.transmission = open;
    mask.outside = 1.0;

    struct OmcBeamModifier modifier;
    omcApertureMaskAsModifier(&mask, &modifier);

    omcCalcForward(&opt, &source, &modifier, masked, NULL, NULL,
                   &maskedSummary);

    CHECK(maskedSummary.blocked == 0);
    CHECK(maskedSummary.started == bareSummary.started);

    double bareTotal = 0.0, maskedTotal = 0.0;
    for (int i = 0; i < GRIDSIZE; i++) {
        bareTotal += bare[i];
        maskedTotal += masked[i];
    }

    CHECK(bareTotal > 0.0);

    /* The same particles down the same random streams, so what is left
     between the two totals is the order the threads added them up in. */
    double diff = fabs(maskedTotal - bareTotal)/bareTotal;
    if (!(diff < 1.0e-9)) {
        printf("  FAIL %s: an open mask moved the dose by %.3g\n",
               current_test, diff);
        tests_failed++;
    }

    free(bare);
    free(masked);
    tearDownWaterTank();
}

/* A shut one stops the lot, and says so rather than quietly returning an
 empty cube. */
static void test_a_shut_mask_stops_everything(void) {

    setUpWaterTank();

    struct Made made;
    makeBeam(&made, 16, 6.0, 0);

    struct OmcForwardOptions opt = optionsFor(4000);
    struct OmcPhspSampler sampler = samplerFor(&made.phsp);
    struct OmcSource source;
    omcPhspSamplerAsSource(&sampler, &source);

    double shut[1] = {0.0};
    struct OmcApertureMask mask;
    memset(&mask, 0, sizeof(mask));
    mask.z = -0.5;
    mask.x0 = -50.0;
    mask.y0 = -50.0;
    mask.dx = 100.0;
    mask.dy = 100.0;
    mask.nx = 1;
    mask.ny = 1;
    mask.transmission = shut;
    mask.outside = 0.0;

    struct OmcBeamModifier modifier;
    omcApertureMaskAsModifier(&mask, &modifier);

    double *dose = malloc(GRIDSIZE*sizeof(double));
    struct OmcForwardSummary summary;

    int finished = omcCalcForward(&opt, &source, &modifier, dose, NULL, NULL,
                                  &summary);

    CHECK(finished == 1);
    CHECK(summary.started == 0);
    CHECK(summary.blocked == 4000);

    double total = 0.0;
    for (int i = 0; i < GRIDSIZE; i++) {
        total += dose[i];
    }
    CHECK(total == 0.0);

    free(dose);
    tearDownWaterTank();
}

/* And one that half transmits gives half the dose from the same number of
 histories: the weight is what the transmission scales, not the count. That
 is the whole difference between attenuating a beam and thinning it, and the
 reason a nearly shut leaf costs as much to transport as an open one. */
static void test_a_half_transmitting_mask_halves_the_dose(void) {

    setUpWaterTank();

    struct Made made;
    makeBeam(&made, 16, 6.0, 0);

    struct OmcForwardOptions opt = optionsFor(20000);
    struct OmcPhspSampler sampler = samplerFor(&made.phsp);
    struct OmcSource source;
    omcPhspSamplerAsSource(&sampler, &source);

    double *bare = malloc(GRIDSIZE*sizeof(double));
    double *half = malloc(GRIDSIZE*sizeof(double));
    struct OmcForwardSummary bareSummary;
    struct OmcForwardSummary halfSummary;

    omcCalcForward(&opt, &source, NULL, bare, NULL, NULL, &bareSummary);

    double leaky[1] = {0.5};
    struct OmcApertureMask mask;
    memset(&mask, 0, sizeof(mask));
    mask.z = -0.5;
    mask.x0 = -50.0;
    mask.y0 = -50.0;
    mask.dx = 100.0;
    mask.dy = 100.0;
    mask.nx = 1;
    mask.ny = 1;
    mask.transmission = leaky;
    mask.outside = 0.5;

    struct OmcBeamModifier modifier;
    omcApertureMaskAsModifier(&mask, &modifier);

    omcCalcForward(&opt, &source, &modifier, half, NULL, NULL, &halfSummary);

    /* Every history still happened and still transported. */
    CHECK(halfSummary.blocked == 0);
    CHECK(halfSummary.started == bareSummary.started);

    double bareTotal = 0.0, halfTotal = 0.0;
    for (int i = 0; i < GRIDSIZE; i++) {
        bareTotal += bare[i];
        halfTotal += half[i];
    }

    CHECK(bareTotal > 0.0);

    double ratio = halfTotal/bareTotal;
    if (!(fabs(ratio - 0.5) < 1.0e-9)) {
        printf("  FAIL %s: half transmission gave a ratio of %.9g, expected "
               "0.5\n", current_test, ratio);
        tests_failed++;
    }

    free(bare);
    free(half);
    tearDownWaterTank();
}

/* The same half transmitting mask played as roulette instead: half the
 histories are stopped outright and the rest transport at full weight, which
 has to come to the same dose. It is the cheaper of the two here -- half the
 showers -- and the noisier, so it is held to a percent rather than to the
 exact ratio the weight mode gives. */
static void test_roulette_gives_the_same_dose_more_cheaply(void) {

    setUpWaterTank();

    struct Made made;
    makeBeam(&made, 16, 6.0, 0);

    struct OmcForwardOptions opt = optionsFor(40000);
    struct OmcPhspSampler sampler = samplerFor(&made.phsp);
    struct OmcSource source;
    omcPhspSamplerAsSource(&sampler, &source);

    double *bare = malloc(GRIDSIZE*sizeof(double));
    double *played = malloc(GRIDSIZE*sizeof(double));
    struct OmcForwardSummary bareSummary;
    struct OmcForwardSummary playedSummary;

    omcCalcForward(&opt, &source, NULL, bare, NULL, NULL, &bareSummary);

    double leaky[1] = {0.5};
    struct OmcApertureMask mask;
    memset(&mask, 0, sizeof(mask));
    mask.z = -0.5;
    mask.x0 = -50.0;
    mask.y0 = -50.0;
    mask.dx = 100.0;
    mask.dy = 100.0;
    mask.nx = 1;
    mask.ny = 1;
    mask.transmission = leaky;
    mask.outside = 0.5;

    struct OmcBeamModifier modifier;
    omcApertureMaskAsModifier(&mask, &modifier);
    modifier.apply = OMC_MODIFIER_ROULETTE;

    omcCalcForward(&opt, &source, &modifier, played, NULL, NULL,
                   &playedSummary);

    /* Where the saving comes from: half the histories never reach a shower,
     unlike the weight mode where every one of them does. */
    CHECK(playedSummary.blocked > 0);
    CHECK(playedSummary.started + playedSummary.blocked
          == (unsigned long long)playedSummary.nhist);

    double stopped = (double)playedSummary.blocked/(double)playedSummary.nhist;
    CHECK(stopped > 0.45 && stopped < 0.55);

    double bareTotal = 0.0, playedTotal = 0.0;
    for (int i = 0; i < GRIDSIZE; i++) {
        bareTotal += bare[i];
        playedTotal += played[i];
    }

    CHECK(bareTotal > 0.0);

    /* And the dose is the same one, up to the noise roulette adds. */
    double ratio = playedTotal/bareTotal;
    if (!(fabs(ratio - 0.5) < 0.02)) {
        printf("  FAIL %s: roulette gave a ratio of %.9g, expected 0.5 to "
               "within the statistics\n", current_test, ratio);
        tests_failed++;
    }

    free(bare);
    free(played);
    tearDownWaterTank();
}

int main(void) {

    /* Unbuffered, so a test that brings the process down still leaves behind
     the list of the ones that got that far. */
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("ompMC phase space forward calculation tests\n\n");

    RUN(test_a_photon_beam_builds_up_and_falls_off);
    RUN(test_a_beam_aimed_away_deposits_nothing);
    RUN(test_histories_that_miss_still_count);
    RUN(test_an_open_mask_changes_nothing);
    RUN(test_a_shut_mask_stops_everything);
    RUN(test_a_half_transmitting_mask_halves_the_dose);
    RUN(test_roulette_gives_the_same_dose_more_cheaply);

    omcSetHost(NULL);

    printf("\n%d test groups, %d check failures\n", tests_run, tests_failed);

    return tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

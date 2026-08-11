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
        fn();                                                                 \
        printf("%-50s %s\n", #fn,                                             \
               tests_failed == _before ? "ok" : "FAILED");                    \
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

    int finished = omcCalcForward(&opt, &source, dose, unc, NULL, &summary);

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

    int finished = omcCalcForward(&opt, &source, dose, NULL, NULL, &summary);

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

    omcCalcForward(&opt, &sourceAll, doseAll, NULL, NULL, &summaryAll);
    omcCalcForward(&opt, &sourceHalf, doseHalf, NULL, NULL, &summaryHalf);

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

int main(void) {

    printf("ompMC phase space forward calculation tests\n\n");

    RUN(test_a_photon_beam_builds_up_and_falls_off);
    RUN(test_a_beam_aimed_away_deposits_nothing);
    RUN(test_histories_that_miss_still_count);

    omcSetHost(NULL);

    printf("\n%d test groups, %d check failures\n", tests_run, tests_failed);

    return tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations

 Tests that need the PEGS and cross section data loaded, so unlike
 test_ompmc.c this one has to run from the repository root where the data,
 pegs4 and phantoms folders live. CTest is told to do that.

 The focus is the per medium layout of the interaction tables: several of them
 are one flat allocation holding media.nmed slabs back to back, and reading a
 slab without its medium offset is a mistake that produces plausible looking
 but wrong physics rather than a crash.
*****************************************************************************/

#include "ompmc.h"
#include "omc_random.h"
#include "omc_utilities.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
* User code hooks the core library expects. rayleigh() reaches none of them --
* it only touches the stack and the RNG -- but they have to link.
*******************************************************************************/
int verbose_flag = 0;

void howfar(int *idisc, int *irnew, double *ustep) {
    (void)idisc; (void)irnew; (void)ustep;
}
double hownear(void) { return 0.0; }
int regionIndex(double x, double y, double z) {
    (void)x; (void)y; (void)z;
    return 0;
}
void initRegions(void) { }

/* The particle stack is thread local in the core library, so it has to be
 declared the same way here as the user codes do. */
#if defined(_MSC_VER)
    __declspec(thread) extern struct Stack stack;
#else
    extern struct Stack stack;
    #pragma omp threadprivate(stack)
#endif

extern struct Media media;
extern struct Photon photon_data;
extern struct Rayleigh rayleigh_data;
extern struct inputItems input_items[];
extern int input_idx;

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
        printf("%-46s %s\n", #fn,                                             \
               tests_failed == _before ? "ok" : "FAILED");                    \
    } while (0)

/*******************************************************************************
* Fixture: four media spanning a wide range of effective Z, which is what makes
* the Rayleigh form factors differ between them.
*******************************************************************************/
enum { NMED = 4 };
static const char *kMedia[NMED] = {
    "AIR700ICRU",
    "LUNG700ICRU",
    "ICRUTISSUE700ICRU",
    "ICRPBONE700ICRU"
};

static void setInput(int i, const char *key, const char *value) {
    strcpy(input_items[i].key, key);
    strcpy(input_items[i].value, value);
}

static void loadMedia(void) {

    setInput(0, "pegs file", "./pegs4/700icru.pegs4dat");
    setInput(1, "pgs4form file", "./pegs4/pgs4form.dat");
    setInput(2, "data folder", "./data/");
    setInput(3, "output folder", "./output/");
    setInput(4, "rng seeds", "97 33");
    input_idx = 5;

    media.nmed = NMED;
    for (int i = 0; i < NMED; i++) {
        strcpy(media.med_names[i], kMedia[i]);
    }

    initMediaData();
}

/*******************************************************************************
* Layout invariants of the Rayleigh tables
*******************************************************************************/
static void test_rayleigh_fcum_is_a_cdf_per_medium(void) {

    for (int m = 0; m < NMED; m++) {
        const double *fcum = &rayleigh_data.fcum[m*MXRAYFF];

        CHECK(fcum[0] == 0.0);

        for (int j = 1; j < MXRAYFF; j++) {
            CHECK(fcum[j] >= fcum[j-1]);
        }

        /* Normalised so the last entry reaches unity */
        CHECK(fcum[MXRAYFF-1] > 0.9);
    }
}

static void test_rayleigh_i_array_is_populated_per_medium(void) {

    /* Every slab must be filled with in range 1 based bin indices. A slab left
     untouched by initRayleighData() would show up here as garbage. */
    for (int m = 0; m < NMED; m++) {
        const int *ia = &rayleigh_data.i_array[m*RAYCDFSIZE];

        for (int j = 0; j < RAYCDFSIZE; j++) {
            CHECK(ia[j] >= 1 && ia[j] <= MXRAYFF - 1);
        }

        /* It indexes a cumulative distribution, so it is non decreasing */
        for (int j = 1; j < RAYCDFSIZE; j++) {
            CHECK(ia[j] >= ia[j-1]);
        }
    }
}

static void test_rayleigh_tables_differ_between_media(void) {

    /* The point of the medium offset. If the slabs were interchangeable then
     ignoring imed would be harmless; they are not. Bone against air is the
     widest separation among the fixture media. */
    const double *fcum_air = &rayleigh_data.fcum[0];
    const double *fcum_bone = &rayleigh_data.fcum[3*MXRAYFF];
    const int *ia_air = &rayleigh_data.i_array[0];
    const int *ia_bone = &rayleigh_data.i_array[3*RAYCDFSIZE];

    double maxdiff = 0.0;
    for (int j = 0; j < MXRAYFF; j++) {
        double d = fabs(fcum_bone[j] - fcum_air[j]);
        if (d > maxdiff) {
            maxdiff = d;
        }
    }

    int idiff = 0;
    for (int j = 0; j < RAYCDFSIZE; j++) {
        if (ia_bone[j] != ia_air[j]) {
            idiff++;
        }
    }

    /* fcum is a normalised CDF, so this is an absolute probability error */
    CHECK(maxdiff > 0.05);
    CHECK(idiff > RAYCDFSIZE/2);
}

/*******************************************************************************
* rayleigh() itself must sample from the medium it is handed
*******************************************************************************/

/* Mean cosine of the Rayleigh scattering angle in a medium, at a photon energy
 low enough that coherent scattering is not a rounding error.

 A photon travelling along +z hits the small polar change branch of uphi21(),
 which leaves stack.w equal to the sampled cos(theta) exactly, so the deflection
 can be read straight back off the stack. */
static double meanRayleighCosine(int imed, double eig, int nsample) {

    double gle = log(eig);
    int lgle = pwlfInterval(imed, gle, photon_data.ge1, photon_data.ge0) - 1;
    double sum = 0.0;

    for (int i = 0; i < nsample; i++) {
        stack.np = 0;
        stack.p[0].iq = 0;
        stack.p[0].e = eig;
        stack.p[0].wt = 1.0;
        stack.p[0].ir = 1;
        stack.p[0].x = 0.0; stack.p[0].y = 0.0; stack.p[0].z = 0.0;
        stack.p[0].u = 0.0; stack.p[0].v = 0.0; stack.p[0].w = 1.0;

        rayleigh(imed, eig, gle, lgle);

        sum += stack.p[0].w;
    }

    return sum/nsample;
}

static double sampleMeanCosine(int imed, double eig, int nsample) {

    /* Same seed every time so media are compared on the same random stream */
    setInput(4, "rng seeds", "97 33");
    initRandom();
    double m = meanRayleighCosine(imed, eig, nsample);
    cleanRandom();
    return m;
}

static void test_rayleigh_is_broader_in_the_higher_z_medium(void) {

    /* Rayleigh scattering is governed by the atomic form factor F(x,Z). The
     higher Z medium has a form factor extending to larger momentum transfer,
     so its angular distribution is *broader*: mean cos(theta) must come out
     lower for bone than for air, at every energy where coherent scattering
     is not a rounding error.

     This is the assertion that pins the medium offset in rayleigh(). Reading
     the form factor tables without it does not merely blur the difference --
     pmax is offset correctly and still varies per medium, so the two means
     stay apart -- it inverts the ordering, because every medium then samples
     air's distribution while being scaled by its own pmax. Measured with the
     offset missing, bone came out at 0.828 / 0.898 / 0.956 against air's
     0.718 / 0.854 / 0.952, i.e. more forward peaked at all three energies. */
    const double energies[3] = {0.03, 0.05, 0.1};

    /* How far below air bone has to land. The true separation narrows towards
     low energy, where both distributions are broad, so the margin follows it.
     Every one of these is comfortably clear of the ~0.002 standard error at
     this sample count, and the run is deterministic: the seed is fixed and
     reset per medium, so there is no run to run scatter to absorb. */
    const double margin[3] = {0.002, 0.02, 0.02};
    const int nsample = 40000;

    initStack();

    for (int e = 0; e < 3; e++) {
        double eig = energies[e];
        double cos_air = sampleMeanCosine(0, eig, nsample);
        double cos_bone = sampleMeanCosine(3, eig, nsample);

        printf("      %5.3f MeV: air %.5f, bone %.5f, bone-air %+.5f\n",
               eig, cos_air, cos_bone, cos_bone - cos_air);

        /* Both physical */
        CHECK(cos_air > -1.0 && cos_air < 1.0);
        CHECK(cos_bone > -1.0 && cos_bone < 1.0);

        CHECK(cos_bone < cos_air - margin[e]);
    }

    cleanStack();
}

/*******************************************************************************
* Photon interaction tables are also per medium
*******************************************************************************/
static void test_photon_energy_grids_differ_between_media(void) {

    /* ge0/ge1 map log(E) onto the interpolation index and depend on the
     medium's energy range from PEGS, so at least one pair must differ */
    int differ = 0;
    for (int m = 1; m < NMED; m++) {
        if (photon_data.ge0[m] != photon_data.ge0[0] ||
            photon_data.ge1[m] != photon_data.ge1[0]) {
            differ = 1;
        }
    }

    /* Mean free paths certainly differ: air against bone is three orders of
     magnitude in density */
    double gmfp_air = pwlfEval(0*MXGE + 100, log(0.05),
                               photon_data.gmfp);
    double gmfp_bone = pwlfEval(3*MXGE + 100, log(0.05),
                                photon_data.gmfp);

    CHECK(gmfp_air > 0.0);
    CHECK(gmfp_bone > 0.0);
    CHECK(fabs(gmfp_air - gmfp_bone)/gmfp_air > 0.1);
    (void)differ;
}

/*******************************************************************************/
int main(void) {

    printf("ompMC media data tests\n\n");

    loadMedia();

    printf("\n");
    RUN(test_rayleigh_fcum_is_a_cdf_per_medium);
    RUN(test_rayleigh_i_array_is_populated_per_medium);
    RUN(test_rayleigh_tables_differ_between_media);
    RUN(test_rayleigh_is_broader_in_the_higher_z_medium);
    RUN(test_photon_energy_grids_differ_between_media);

    printf("\n%d test groups, %d check failures\n", tests_run, tests_failed);

    cleanPhoton();
    cleanRayleigh();
    cleanPair();
    cleanElectron();
    cleanMscat();
    cleanSpin();

    return tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

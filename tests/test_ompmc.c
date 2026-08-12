/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations

 Unit tests for the self-contained pieces of the transport core: the piecewise
 linear interpolation helpers, the cubic spline used for the spin data, the
 RANMAR random number generator, the azimuthal angle sampler and the voxel
 geometry helpers shared by the user codes.

 The heavy physics routines are not covered here; they need the PEGS and cross
 section data loaded and are exercised end to end by the omc_dosxyz smoke test
 instead.
*****************************************************************************/

#include "ompmc.h"
#include "omc_engine_cube.h"
#include "omc_geom.h"
#include "omc_host.h"
#include "omc_random.h"
#include "omc_score.h"
#include "omc_utilities.h"

#include <math.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
* ompmc.c is compiled into the same library as the code under test and refers
* to this, so it has to exist even though most tests never reach it. The
* geometry hooks it also refers to -- howfar(), hownear(), regionIndex() --
* used to be stubbed out here; they now come from omc_geom.c in the library
* itself, and no test reaches them. ausgab() likewise comes from omc_score.c
* and is exercised directly below.
*******************************************************************************/
int verbose_flag = 0;

/* The particle stack is thread local in the core library, so it has to be
 declared the same way here as the user codes do. */
#if defined(_MSC_VER)
    __declspec(thread) extern struct Stack stack;
#else
    extern struct Stack stack;
    #pragma omp threadprivate(stack)
#endif

/*******************************************************************************
* Minimal assertion harness
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
        fn();                                                                 \
        printf("%-42s %s\n", #fn,                                             \
               tests_failed == _before ? "ok" : "FAILED");                    \
    } while (0)

/*******************************************************************************
* Piecewise linear interpolation helpers (ompmc.c)
*
* The energy grids are stored as coef1*log(E) + coef0, evaluated by pwlfEval()
* and truncated to the containing interval by pwlfInterval(). Everything that
* reads a cross section or stopping power goes through these two.
*******************************************************************************/
static void test_pwlf_eval_is_affine(void) {

    /* Interleaved {slope, intercept} pairs */
    double coef[6] = {2.0, 1.0, -1.5, 4.0, 0.0, 7.0};

    CHECK_CLOSE(pwlfEval(0, 3.0, coef), 7.0, 0.0);
    CHECK_CLOSE(pwlfEval(1, 2.0, coef), 1.0, 0.0);

    /* Index 2 has zero slope, so it is constant in lvar */
    CHECK_CLOSE(pwlfEval(2, -100.0, coef), 7.0, 0.0);
    CHECK_CLOSE(pwlfEval(2,  100.0, coef), 7.0, 0.0);
}

static void test_pwlf_interval_truncates(void) {

    double coef1[1] = {4.0};
    double coef0[1] = {0.5};
    double coef[2] = {4.0, 0.5};

    /* 4*lvar + 0.5, truncated towards zero */
    CHECK(pwlfInterval(0, 1.0,  coef1, coef0) == 4);
    CHECK(pwlfInterval(0, 1.2,  coef1, coef0) == 5);
    CHECK(pwlfInterval(0, 0.0,  coef1, coef0) == 0);

    /* pwlfInterval must agree with truncating what pwlfEval returns; the
     transport code relies on the two staying consistent */
    for (int i = 0; i < 50; i++) {
        double lvar = -3.0 + 0.17*i;
        CHECK(pwlfInterval(0, lvar, coef1, coef0) ==
              (int)pwlfEval(0, lvar, coef));
    }
}

/*******************************************************************************
* Cubic spline (ompmc.c), used to resample the spin rejection data
*******************************************************************************/
static void test_spline_interpolates_nodes(void) {

    enum { N = 9 };
    double x[N], f[N], a[N], b[N], c[N], d[N];

    for (int i = 0; i < N; i++) {
        x[i] = (double)i;
        f[i] = sin(0.4*i) + 0.1*i;
    }

    setSpline(x, f, a, b, c, d, N);

    /* A spline must reproduce the tabulated values at the nodes */
    for (int i = 0; i < N - 1; i++) {
        CHECK_CLOSE(spline(x[i], x, a, b, c, d, N), f[i], 1.0E-12);
    }
}

static void test_spline_matches_smooth_function(void) {

    enum { N = 33 };
    double x[N], f[N], a[N], b[N], c[N], d[N];

    for (int i = 0; i < N; i++) {
        x[i] = -2.0 + 4.0*i/(N - 1);
        f[i] = exp(-x[i]*x[i]);
    }

    setSpline(x, f, a, b, c, d, N);

    /* Between the nodes a cubic spline over a smooth function should track it
     to far better than the node spacing */
    for (int i = 0; i < 40; i++) {
        double s = -1.9 + 3.8*i/39.0;
        CHECK_CLOSE(spline(s, x, a, b, c, d, N), exp(-s*s), 1.0E-4);
    }
}

/*******************************************************************************
* heap_sort (ompmc.c), used when building the photon cross section tables
*******************************************************************************/
static void test_heap_sort_orders_values_and_indices(void) {

    enum { N = 7 };
    double values[N] = {3.5, -1.0, 9.25, 0.0, 3.5, 100.0, -7.5};
    double original[N];
    int indices[N];

    memcpy(original, values, sizeof(values));
    heap_sort(N, values, indices);

    for (int i = 1; i < N; i++) {
        CHECK(values[i-1] <= values[i]);
    }

    /* indices are 1 based and must still point at the value that moved */
    for (int i = 0; i < N; i++) {
        CHECK(indices[i] >= 1 && indices[i] <= N);
        CHECK_CLOSE(original[indices[i] - 1], values[i], 0.0);
    }
}

/*******************************************************************************
* Philox4x32-10 random number generator (omc_random.c)
*******************************************************************************/

/* initRandom() reads its seeds through getInputValue(), so stage them the way
 the user codes do rather than parsing a file. */
static void seedRandom(const char *seeds) {
    strcpy(input_items[0].key, "rng seeds");
    strcpy(input_items[0].value, seeds);
    input_idx = 1;      /* one pair, in slot 0 */
    initRandom();
    setRandomHistory(0);
}

/* Published Philox4x32-10 test vectors from the Random123 known-answer
 tests, so the round function is checked against the reference and not just
 against itself */
static void test_philox_matches_reference_vectors(void) {

    uint32_t out[4];

    const uint32_t zeros[4] = {0u, 0u, 0u, 0u};
    const uint32_t zero_key[2] = {0u, 0u};
    philox4x32(zeros, zero_key, out);
    CHECK(out[0] == 0x6627e8d5u && out[1] == 0xe169c58du &&
          out[2] == 0xbc57ac4cu && out[3] == 0x9b00dbd8u);

    const uint32_t ones[4] = {0xffffffffu, 0xffffffffu,
                              0xffffffffu, 0xffffffffu};
    const uint32_t ones_key[2] = {0xffffffffu, 0xffffffffu};
    philox4x32(ones, ones_key, out);
    CHECK(out[0] == 0x408f276du && out[1] == 0x41c83b0eu &&
          out[2] == 0xa20bc7c6u && out[3] == 0x6d5451fdu);

    const uint32_t pi_ctr[4] = {0x243f6a88u, 0x85a308d3u,
                                0x13198a2eu, 0x03707344u};
    const uint32_t pi_key[2] = {0xa4093822u, 0x299f31d0u};
    philox4x32(pi_ctr, pi_key, out);
    CHECK(out[0] == 0xd16cfe09u && out[1] == 0x94fdccebu &&
          out[2] == 0x5001e420u && out[3] == 0x24126ea1u);
}

static void test_random_stays_in_unit_interval(void) {

    seedRandom("97 33");

    /* Well past the four-value refill boundary so the wrap is covered; the
     endpoints are excluded by construction */
    for (int i = 0; i < 4096 + 3; i++) {
        double r = setRandom();
        CHECK(r > 0.0 && r < 1.0);
    }

    cleanRandom();
}

static void test_random_is_reproducible_for_a_seed(void) {

    enum { N = 512 };
    double first[N];

    seedRandom("97 33");
    for (int i = 0; i < N; i++) {
        first[i] = setRandom();
    }
    cleanRandom();

    seedRandom("97 33");
    for (int i = 0; i < N; i++) {
        CHECK_CLOSE(setRandom(), first[i], 0.0);
    }
    cleanRandom();
}

static void test_random_differs_between_seeds(void) {

    enum { N = 64 };
    double first[N];
    int identical = 1;

    seedRandom("97 33");
    for (int i = 0; i < N; i++) {
        first[i] = setRandom();
    }
    cleanRandom();

    seedRandom("1802 9373");
    for (int i = 0; i < N; i++) {
        if (setRandom() != first[i]) {
            identical = 0;
        }
    }
    cleanRandom();

    CHECK(!identical);
}

static void test_random_history_streams_are_independent(void) {

    enum { N = 37 };    /* not a multiple of the block size on purpose */
    double first[N];
    int identical = 1;

    seedRandom("97 33");

    setRandomHistory(42);
    for (int i = 0; i < N; i++) {
        first[i] = setRandom();
    }

    /* A different history must give a different stream */
    setRandomHistory(43);
    for (int i = 0; i < N; i++) {
        if (setRandom() != first[i]) {
            identical = 0;
        }
    }
    CHECK(!identical);

    /* Returning to a history replays its stream exactly, no matter what ran
     in between -- this is what makes results scheduling-independent */
    setRandomHistory(42);
    for (int i = 0; i < N; i++) {
        CHECK_CLOSE(setRandom(), first[i], 0.0);
    }

    /* Histories whose indices only differ in the high 32 bits must also be
     distinct streams */
    setRandomHistory(42u + (1ULL << 32));
    identical = 1;
    for (int i = 0; i < N; i++) {
        if (setRandom() != first[i]) {
            identical = 0;
        }
    }
    CHECK(!identical);

    cleanRandom();
}

static void test_random_mean_is_plausible(void) {

    enum { N = 200000 };
    double sum = 0.0;

    seedRandom("97 33");
    for (int i = 0; i < N; i++) {
        sum += setRandom();
    }
    cleanRandom();

    /* Standard error of the mean of N uniforms is 1/sqrt(12N) ~ 6.5E-4 here,
     so five sigma is a very loose bound that still catches a broken stream */
    CHECK_CLOSE(sum/N, 0.5, 5.0/sqrt(12.0*N));
}

/*******************************************************************************
* Azimuthal angle sampling (ompmc.c)
*******************************************************************************/
static void test_azimuthal_angle_is_on_the_unit_circle(void) {

    seedRandom("97 33");

    double sum_cos = 0.0;
    double sum_sin = 0.0;
    enum { N = 20000 };

    for (int i = 0; i < N; i++) {
        double costhe, sinthe;
        selectAzimuthalAngle(&costhe, &sinthe);

        CHECK_CLOSE(costhe*costhe + sinthe*sinthe, 1.0, 1.0E-12);
        sum_cos += costhe;
        sum_sin += sinthe;
    }

    /* Uniform in phi means both first moments average to zero */
    CHECK_CLOSE(sum_cos/N, 0.0, 0.05);
    CHECK_CLOSE(sum_sin/N, 0.0, 0.05);

    cleanRandom();
}

/*******************************************************************************
* Voxel geometry helpers (omc_utilities.h)
*******************************************************************************/
static void test_decode_region_round_trips(void) {

    const int imax = 7, jmax = 5, kmax = 3;

    for (int iz = 0; iz < kmax; iz++) {
        for (int iy = 0; iy < jmax; iy++) {
            for (int ix = 0; ix < imax; ix++) {
                int irl = 1 + ix + iy*imax + iz*imax*jmax;
                int dx, dy, dz;

                omcDecodeRegion(irl, imax, jmax, &dx, &dy, &dz);

                CHECK(dx == ix);
                CHECK(dy == iy);
                CHECK(dz == iz);
            }
        }
    }
}

static void test_decode_region_matches_the_original_formula(void) {

    /* Guards the two division rewrite against the three division form the
     user codes used before */
    const int imax = 13, jmax = 11, kmax = 9;

    for (int irl = 1; irl <= imax*jmax*kmax; irl++) {
        int ix, iy, iz;
        omcDecodeRegion(irl, imax, jmax, &ix, &iy, &iz);

        int ijmax = imax*jmax;
        int rx = (irl - 1)%imax;
        int rz = (irl - 1 - rx)/ijmax;
        int ry = ((irl - 1 - rx) - rz*ijmax)/imax;

        CHECK(ix == rx);
        CHECK(iy == ry);
        CHECK(iz == rz);
    }
}

static void test_decode_region_handles_a_single_voxel_axis(void) {

    int ix, iy, iz;

    /* imax == 1 makes the first division degenerate */
    omcDecodeRegion(4, 1, 2, &ix, &iy, &iz);
    CHECK(ix == 0);
    CHECK(iy == 1);
    CHECK(iz == 1);
}

static void test_find_voxel_index_matches_linear_scan(void) {

    enum { N = 16 };
    double bounds[N + 1];

    for (int i = 0; i <= N; i++) {
        bounds[i] = -4.0 + 0.5*i;
    }

    for (int t = 0; t < 400; t++) {
        double pos = -4.05 + 8.1*t/399.0;

        /* The linear scan initHistory() used, kept in range the way the
         clamping in initHistory() guarantees */
        int expect = 0;
        while (expect < N - 1 && bounds[expect+1] < pos) {
            expect++;
        }

        CHECK(omcFindVoxelIndex(bounds, N, pos) == expect);
    }
}

static void test_find_voxel_index_on_boundaries_and_outside(void) {

    double bounds[5] = {0.0, 1.0, 2.0, 3.0, 4.0};

    /* A position exactly on an internal boundary belongs to the lower voxel,
     matching bounds[i+1] >= pos */
    CHECK(omcFindVoxelIndex(bounds, 4, 0.0) == 0);
    CHECK(omcFindVoxelIndex(bounds, 4, 1.0) == 0);
    CHECK(omcFindVoxelIndex(bounds, 4, 1.5) == 1);
    CHECK(omcFindVoxelIndex(bounds, 4, 3.0) == 2);
    CHECK(omcFindVoxelIndex(bounds, 4, 4.0) == 3);

    /* Outside clamps instead of walking off the end of bounds[] */
    CHECK(omcFindVoxelIndex(bounds, 4, -10.0) == 0);
    CHECK(omcFindVoxelIndex(bounds, 4,  10.0) == 3);

    /* Degenerate single voxel grid */
    CHECK(omcFindVoxelIndex(bounds, 1, 0.5) == 0);
    CHECK(omcFindVoxelIndex(bounds, 1, 99.0) == 0);
}

/*******************************************************************************
* Energy scoring (omc_score.c)
*******************************************************************************/

/* Deposit edep in region irl the way the transport code does, through the
 particle stack. */
static void depositAt(int irl, double edep, double wt) {

    stack.np = 0;
    stack.p[0].ir = irl;
    stack.p[0].wt = wt;

    ausgab(edep);
}

static void test_score_accumulates_repeated_deposits(void) {

    initScore(10);
    initStack();

    /* Interleaved, the way an electron crossing back and forth would */
    depositAt(3, 1.0, 1.0);
    depositAt(3, 2.0, 1.0);
    depositAt(3, 4.0, 1.0);
    depositAt(5, 0.5, 1.0);
    depositAt(5, 0.25, 1.0);
    depositAt(3, 8.0, 1.0);

    CHECK_CLOSE(score.endep[3], 15.0, 1.0E-12);
    CHECK_CLOSE(score.endep[5], 0.75, 1.0E-12);

    /* Nothing else may have been written */
    for (int i = 0; i <= 10; i++) {
        if (i != 3 && i != 5) {
            CHECK(score.endep[i] == 0.0);
        }
    }

    cleanStack();
    cleanScore();
}

static void test_score_applies_the_particle_weight(void) {

    initScore(4);
    initStack();

    depositAt(2, 3.0, 2.5);

    CHECK_CLOSE(score.endep[2], 7.5, 1.0E-12);

    cleanStack();
    cleanScore();
}

static void test_score_tracks_touched_voxels_only(void) {

    initScore(100);
    initStack();

    depositAt(42, 1.0, 1.0);
    depositAt(7, 2.0, 1.0);
    depositAt(42, 3.0, 1.0);   // already recorded, must not be listed twice

    CHECK(score.beam_count == 2);

    accumEndep(0.5);

    CHECK_CLOSE(score.accum_endep[42], 2.0, 1.0E-12);
    CHECK_CLOSE(score.accum_endep2[42], 4.0, 1.0E-12);
    CHECK_CLOSE(score.accum_endep[7], 1.0, 1.0E-12);

    /* The batch is consumed and its grid entries cleared, while the voxels
     stay on the beamlet's list for the batches still to come */
    CHECK(score.endep[42] == 0.0);
    CHECK(score.endep[7] == 0.0);
    CHECK(score.beam_count == 2);

    /* A second batch that reaches neither voxel must leave the totals alone */
    accumEndep(0.5);
    CHECK_CLOSE(score.accum_endep[42], 2.0, 1.0E-12);
    CHECK_CLOSE(score.accum_endep[7], 1.0, 1.0E-12);

    cleanStack();
    cleanScore();
}

static void test_score_beam_voxels_are_sorted_and_unique(void) {

    /* The sparse column built from this list needs ascending row indices, so
     deposit in descending order and across several batches to make sure the
     ordering comes from the sort and not from the insertion order. */
    const int order[6] = {90, 12, 55, 3, 55, 71};

    initScore(100);
    initStack();

    for (int i = 0; i < 6; i++) {
        depositAt(order[i], 1.0, 1.0);
            accumEndep(1.0);
    }

    const int *list;
    int n = scoreBeamVoxels(&list);

    CHECK(n == 5);      // 55 appears twice in the input

    for (int i = 1; i < n; i++) {
        CHECK(list[i-1] < list[i]);     // ascending and duplicate free
    }

    if (n == 5) {
        CHECK(list[0] == 3);
        CHECK(list[4] == 90);
    }

    cleanStack();
    cleanScore();
}

static void test_reset_beam_score_clears_both_accumulators(void) {

    initScore(20);
    initStack();

    depositAt(9, 4.0, 1.0);
    accumEndep(1.0);

    CHECK(score.accum_endep[9] != 0.0);
    CHECK(score.accum_endep2[9] != 0.0);

    resetBeamScore();

    /* accum_endep2 used to survive this, so a beamlet's variance leaked into
     the next one */
    CHECK(score.accum_endep[9] == 0.0);
    CHECK(score.accum_endep2[9] == 0.0);
    CHECK(score.beam_count == 0);

    /* A following beamlet starts from zero */
    depositAt(9, 1.0, 1.0);
    accumEndep(1.0);
    CHECK_CLOSE(score.accum_endep[9], 1.0, 1.0E-12);
    CHECK_CLOSE(score.accum_endep2[9], 1.0, 1.0E-12);

    cleanStack();
    cleanScore();
}

/*******************************************************************************
* A host whose sinks the tests can see
*
* omcFail() must not return, so a test that wants to reach one has to give it
* somewhere to go. omc_host.h names longjmp() as how an embedded host ends a
* failure without taking the process with it, which is what the Python module
* does; this is the same thing on a smaller scale.
*******************************************************************************/

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

/*******************************************************************************
* Input item table (omc_utilities.c)
*
* input_idx is the NUMBER of pairs stored, in slots 0..input_idx-1, however
* the table was filled. The tests below pin that down at both ends: the table
* used to be filled two different ways -- the file parser counting to the last
* index, everything else counting pairs -- which agreed only when there were
* at least two of them, and papered over the difference with a lookup that
* scanned one slot past the end.
*******************************************************************************/

/* Write an input deck next to the test executable. The name is given to
 parseInputFile() without the extension, the way the user codes pass -i. */
static int writeDeck(const char *stem, const char *contents) {

    char path[256];
    snprintf(path, sizeof(path), "%s%s", stem, INPUT_EXT);

    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        printf("  FAIL cannot write %s\n", path);
        tests_failed++;
        return 0;
    }
    fputs(contents, fp);
    fclose(fp);

    return 1;
}

static void removeDeck(const char *stem) {

    char path[256];
    snprintf(path, sizeof(path), "%s%s", stem, INPUT_EXT);
    remove(path);
}

static void writeAndParse(const char *stem, const char *contents) {

    if (!writeDeck(stem, contents)) {
        return;
    }

    /* parseInputFile() appends from wherever input_idx already stands, so the
     table has to start empty for the count to mean anything. */
    omcClearInputValues();

    char stem_buf[256];
    snprintf(stem_buf, sizeof(stem_buf), "%s", stem);
    parseInputFile(stem_buf);

    removeDeck(stem);
}

/* The regression test for the "check to see if anything got parsed" early
 return: one pair used to leave input_idx == 0, which that guard could not
 tell apart from an empty table, so every lookup against a one line deck
 failed. */
static void test_input_value_finds_a_single_parsed_pair(void) {

    writeAndParse("test_input_one", "ncase = 4242\n");

    CHECK(input_idx == 1);

    char value[BUFFER_SIZE] = "";
    CHECK(getInputValue(value, "ncase") == 1);
    CHECK(strcmp(value, "4242") == 0);

    omcClearInputValues();
}

static void test_input_value_finds_every_parsed_pair(void) {

    writeAndParse("test_input_many",
                  "# a comment line, skipped\n"
                  "ncase = 20000\n"
                  "\n"
                  "  global ecut =  0.700  \n"
                  "charge = -1\n");

    /* Three pairs, in slots 0, 1 and 2 */
    CHECK(input_idx == 3);

    char value[BUFFER_SIZE];

    /* Including the last pair, which the scan has to reach */
    CHECK(getInputValue(value, "charge") == 1);
    CHECK(strcmp(value, "-1") == 0);

    /* And the first */
    CHECK(getInputValue(value, "ncase") == 1);
    CHECK(strcmp(value, "20000") == 0);

    /* Keys and values are stored trimmed */
    CHECK(getInputValue(value, "global ecut") == 1);
    CHECK(strcmp(value, "0.700") == 0);

    /* A short key must not answer for a longer one it is a substring of */
    CHECK(getInputValue(value, "ecut") == 0);
    CHECK(getInputValue(value, "nbatch") == 0);

    omcClearInputValues();
}

/* An empty table has to answer "not found" rather than read a stale or
 uninitialised slot -- that is the job the removed guard was doing badly. */
static void test_input_value_on_an_empty_table(void) {

    omcClearInputValues();

    char value[BUFFER_SIZE] = "untouched";
    CHECK(getInputValue(value, "ncase") == 0);
    CHECK(strcmp(value, "untouched") == 0);
}

/* Programmatic filling follows the same convention, so a single pair set that
 way is equally findable, and the two ways of filling can be mixed. */
static void test_set_input_value_round_trips(void) {

    omcClearInputValues();

    omcSetInputValue("ncase", "10");

    /* The first pair set on a cleared table goes into slot 0. Pre-incrementing
     instead left that slot permanently empty, costing one of INPUT_PAIRS. */
    CHECK(input_idx == 1);
    CHECK(strcmp(input_items[0].key, "ncase") == 0);

    char value[BUFFER_SIZE] = "";
    CHECK(getInputValue(value, "ncase") == 1);
    CHECK(strcmp(value, "10") == 0);

    /* Setting a known key replaces rather than appends */
    int idx_before = input_idx;
    omcSetInputValue("ncase", "20");
    CHECK(input_idx == idx_before);
    CHECK(getInputValue(value, "ncase") == 1);
    CHECK(strcmp(value, "20") == 0);

    omcSetInputValue("charge", "0");
    CHECK(getInputValue(value, "charge") == 1);
    CHECK(strcmp(value, "0") == 0);
    CHECK(getInputValue(value, "ncase") == 1);
    CHECK(strcmp(value, "20") == 0);

    omcClearInputValues();
    CHECK(getInputValue(value, "ncase") == 0);
    CHECK(getInputValue(value, "charge") == 0);
}

/* The table has to hold the INPUT_PAIRS it advertises, all of them reachable.
 Skipping slot 0 quietly made the real capacity one less. */
static void test_set_input_value_fills_the_whole_table(void) {

    omcClearInputValues();

    char key[BUFFER_SIZE], value[BUFFER_SIZE];
    for (int i = 0; i < INPUT_PAIRS; i++) {
        snprintf(key, sizeof(key), "key%d", i);
        snprintf(value, sizeof(value), "%d", i);
        omcSetInputValue(key, value);
    }

    CHECK(input_idx == INPUT_PAIRS);

    /* Every one of them still findable, the first and the last included */
    for (int i = 0; i < INPUT_PAIRS; i++) {
        snprintf(key, sizeof(key), "key%d", i);
        char expect[BUFFER_SIZE];
        snprintf(expect, sizeof(expect), "%d", i);

        value[0] = '\0';
        CHECK(getInputValue(value, key) == 1);
        CHECK(strcmp(value, expect) == 0);
    }

    /* A full table still replaces rather than overflowing */
    omcSetInputValue("key0", "replaced");
    CHECK(input_idx == INPUT_PAIRS);
    CHECK(getInputValue(value, "key0") == 1);
    CHECK(strcmp(value, "replaced") == 0);

    omcClearInputValues();
}

/* A cleared table is empty, not "one empty pair" -- the distinction the count
 exists to make. */
static void test_clear_input_values_empties_the_table(void) {

    omcSetInputValue("ncase", "10");
    omcSetInputValue("charge", "0");
    CHECK(input_idx > 0);

    omcClearInputValues();
    CHECK(input_idx == 0);

    /* And the next pair set starts again from slot 0 rather than after the
     pairs that are gone */
    omcSetInputValue("charge", "-1");
    CHECK(input_idx == 1);
    CHECK(strcmp(input_items[0].key, "charge") == 0);

    char value[BUFFER_SIZE];
    CHECK(getInputValue(value, "ncase") == 0);
    CHECK(getInputValue(value, "charge") == 1);
    CHECK(strcmp(value, "-1") == 0);

    omcClearInputValues();
}

/* A deck with more pairs than the table holds used to be written straight
 past the end of input_items. It has to stop at the edge and say so. */
static void test_parse_input_file_rejects_an_overfull_deck(void) {

    /* One line per slot, plus a few the table cannot take */
    char deck[(INPUT_PAIRS + 4)*24];
    size_t used = 0;
    for (int i = 0; i < INPUT_PAIRS + 4; i++) {
        used += (size_t)snprintf(deck + used, sizeof(deck) - used,
                                 "key%d = %d\n", i, i);
    }

    if (!writeDeck("test_input_overfull", deck)) {
        return;
    }

    omcClearInputValues();
    installFailCatcher();

    char stem[256];
    snprintf(stem, sizeof(stem), "%s", "test_input_overfull");

    fail_armed = 1;
    if (setjmp(fail_jmp) == 0) {
        parseInputFile(stem);
        CHECK(!"parseInputFile() accepted more pairs than the table holds");
    }
    fail_armed = 0;

    omcSetHost(NULL);

    CHECK(fail_seen == 1);
    CHECK(strcmp(fail_id, "ompMC:input:tooManyItems") == 0);

    /* Filled to the brim and not one past it */
    CHECK(input_idx == INPUT_PAIRS);

    /* What did fit is intact, so the message is about a full table rather
     than about whatever the overflow had already scribbled over */
    char value[BUFFER_SIZE];
    CHECK(getInputValue(value, "key0") == 1);
    CHECK(strcmp(value, "0") == 0);

    removeDeck("test_input_overfull");
    omcClearInputValues();
}

/* The same edge from the programmatic side */
static void test_set_input_value_rejects_one_pair_too_many(void) {

    omcClearInputValues();

    char key[BUFFER_SIZE];
    for (int i = 0; i < INPUT_PAIRS; i++) {
        snprintf(key, sizeof(key), "key%d", i);
        omcSetInputValue(key, "1");
    }
    CHECK(input_idx == INPUT_PAIRS);

    installFailCatcher();

    fail_armed = 1;
    if (setjmp(fail_jmp) == 0) {
        omcSetInputValue("one too many", "1");
        CHECK(!"omcSetInputValue() accepted a pair past INPUT_PAIRS");
    }
    fail_armed = 0;

    omcSetHost(NULL);

    CHECK(fail_seen == 1);
    CHECK(strcmp(fail_id, "ompMC:input:tooManyItems") == 0);
    CHECK(input_idx == INPUT_PAIRS);

    omcClearInputValues();
}

/* A deck parsed from file and then overridden programmatically, which is what
 a host does when it takes a deck and changes one setting. */
static void test_set_input_value_appends_after_a_parsed_file(void) {

    writeAndParse("test_input_mixed", "ncase = 100\n");

    omcSetInputValue("charge", "-1");

    char value[BUFFER_SIZE];
    CHECK(getInputValue(value, "ncase") == 1);
    CHECK(strcmp(value, "100") == 0);
    CHECK(getInputValue(value, "charge") == 1);
    CHECK(strcmp(value, "-1") == 0);

    /* Replacing the parsed pair works too */
    omcSetInputValue("ncase", "200");
    CHECK(getInputValue(value, "ncase") == 1);
    CHECK(strcmp(value, "200") == 0);

    omcClearInputValues();
}

/*******************************************************************************
* SSD source field indices (omc_engine_cube.c)
*
* omcSsdSourceInit() turns the collimator rectangle into the voxel index range
* it covers. The upper index search used to start one below the lower index,
* which reads xbounds[-1] whenever the field reaches the edge of the phantom --
* the ordinary case, since the rectangle is clamped to the phantom first.
*******************************************************************************/

enum { FIELD_NX = 8, FIELD_NY = 6 };

/* Bounds arrays with one guard element in front, so that xbounds[-1] is a
 defined read with a value chosen to be caught rather than tolerated: it is
 above every field edge used below, so the pre-fix loop would find its first
 condition false straight away and leave the upper index at -1. */
static double field_xstore[FIELD_NX + 2];
static double field_ystore[FIELD_NY + 2];

#define FIELD_GUARD 1.0e30

static void setUpFieldGeometry(void) {

    field_xstore[0] = FIELD_GUARD;
    for (int i = 0; i <= FIELD_NX; i++) {
        field_xstore[i + 1] = -4.0 + 1.0*i;     /* -4 .. 4, 1 cm voxels */
    }

    field_ystore[0] = FIELD_GUARD;
    for (int j = 0; j <= FIELD_NY; j++) {
        field_ystore[j + 1] = -3.0 + 1.0*j;     /* -3 .. 3, 1 cm voxels */
    }

    geometry.isize = FIELD_NX;
    geometry.jsize = FIELD_NY;
    geometry.xbounds = &field_xstore[1];
    geometry.ybounds = &field_ystore[1];

    /* omcSsdSourceInit() logs the ranges it found; the test does not need to
     see them. */
    struct OmcHost quiet = {silentLog, NULL, NULL};
    omcSetHost(&quiet);
}

static void tearDownFieldGeometry(void) {

    omcSetHost(NULL);
    geometry.xbounds = NULL;
    geometry.ybounds = NULL;
    geometry.isize = 0;
    geometry.jsize = 0;

    /* The guards must still be intact: nothing may write through xbounds[-1] */
    CHECK(field_xstore[0] == FIELD_GUARD);
    CHECK(field_ystore[0] == FIELD_GUARD);
}

/* The regression test. A field flush against the low edge of the phantom
 leaves the lower index at 0, which is where the old seed of "lower index - 1"
 went out of bounds. */
static void test_ssd_source_field_at_the_low_phantom_edge(void) {

    setUpFieldGeometry();

    struct OmcSsdSource src = {0};
    src.ssd = 90.0;
    src.xinl = -4.0;    /* exactly xbounds[0] */
    src.xinu = -2.0;    /* end of the second voxel */
    src.yinl = -3.0;    /* exactly ybounds[0] */
    src.yinu = -1.0;

    omcSsdSourceInit(&src);

    CHECK(src.ixinl == 0);
    CHECK(src.ixinu == 1);
    CHECK(src.iyinl == 0);
    CHECK(src.iyinu == 1);

    CHECK_CLOSE(src.xsize, 2.0, 1.0e-12);
    CHECK_CLOSE(src.ysize, 2.0, 1.0e-12);

    tearDownFieldGeometry();
}

/* A field that starts below the phantom is clamped to it, and lands in the
 same place as one starting exactly on the edge. */
static void test_ssd_source_field_is_clamped_to_the_phantom(void) {

    setUpFieldGeometry();

    struct OmcSsdSource src = {0};
    src.xinl = -50.0;
    src.xinu =  50.0;
    src.yinl = -50.0;
    src.yinu =  50.0;

    omcSsdSourceInit(&src);

    /* The whole phantom, and no index past the last voxel */
    CHECK(src.ixinl == 0);
    CHECK(src.ixinu == FIELD_NX - 1);
    CHECK(src.iyinl == 0);
    CHECK(src.iyinu == FIELD_NY - 1);

    CHECK_CLOSE(src.xinl, -4.0, 1.0e-12);
    CHECK_CLOSE(src.xinu,  4.0, 1.0e-12);
    CHECK_CLOSE(src.xsize, 8.0, 1.0e-12);
    CHECK_CLOSE(src.ysize, 6.0, 1.0e-12);

    tearDownFieldGeometry();
}

/* Away from the edge the result is unchanged from what the old seed produced,
 which is the other half of the fix being a no-op there. */
static void test_ssd_source_field_inside_the_phantom(void) {

    setUpFieldGeometry();

    struct OmcSsdSource src = {0};
    src.xinl = -1.5;    /* inside voxel 2, spanning -2 .. -1 */
    src.xinu =  2.5;    /* inside voxel 6, spanning  2 ..  3 */
    src.yinl = -0.5;    /* inside voxel 2, spanning -1 ..  0 */
    src.yinu =  1.5;    /* inside voxel 4, spanning  1 ..  2 */

    omcSsdSourceInit(&src);

    CHECK(src.ixinl == 2);
    CHECK(src.ixinu == 6);
    CHECK(src.iyinl == 2);
    CHECK(src.iyinu == 4);

    tearDownFieldGeometry();
}

/* A zero width rectangle is the documented way to ask for a pencil beam, and
 an upper edge below the lower one is normalised to one. */
static void test_ssd_source_pencil_beam(void) {

    setUpFieldGeometry();

    struct OmcSsdSource src = {0};
    src.xinl = -4.0;
    src.xinu = -4.0;    /* zero width against the low edge */
    src.yinl =  0.5;
    src.yinu = -7.0;    /* upper below lower, and below the phantom */

    omcSsdSourceInit(&src);

    CHECK(src.ixinl == 0);
    CHECK(src.ixinu == 0);
    CHECK(src.ixinu >= src.ixinl);

    CHECK(src.iyinu >= src.iyinl);
    CHECK_CLOSE(src.xsize, 0.0, 1.0e-12);
    CHECK_CLOSE(src.ysize, 0.0, 1.0e-12);

    tearDownFieldGeometry();
}

/*******************************************************************************
* Klein-Nishina total cross section (ompmc.c)
*******************************************************************************/
static void test_kn_sigma0_is_positive_and_falls_with_energy(void) {

    double previous = kn_sigma0(0.05);
    CHECK(previous > 0.0);

    /* The Compton cross section per electron decreases monotonically above a
     few tens of keV */
    for (double e = 0.1; e < 20.0; e *= 1.5) {
        double sigma = kn_sigma0(e);
        CHECK(sigma > 0.0);
        CHECK(sigma < previous);
        previous = sigma;
    }
}

/*******************************************************************************/
int main(void) {

    /* Unbuffered, so a test that brings the process down still leaves behind
     the list of the ones that got that far. */
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("ompMC unit tests\n\n");

    RUN(test_pwlf_eval_is_affine);
    RUN(test_pwlf_interval_truncates);

    RUN(test_spline_interpolates_nodes);
    RUN(test_spline_matches_smooth_function);

    RUN(test_heap_sort_orders_values_and_indices);

    RUN(test_philox_matches_reference_vectors);
    RUN(test_random_stays_in_unit_interval);
    RUN(test_random_is_reproducible_for_a_seed);
    RUN(test_random_differs_between_seeds);
    RUN(test_random_history_streams_are_independent);
    RUN(test_random_mean_is_plausible);

    RUN(test_azimuthal_angle_is_on_the_unit_circle);

    RUN(test_decode_region_round_trips);
    RUN(test_decode_region_matches_the_original_formula);
    RUN(test_decode_region_handles_a_single_voxel_axis);
    RUN(test_find_voxel_index_matches_linear_scan);
    RUN(test_find_voxel_index_on_boundaries_and_outside);

    RUN(test_score_accumulates_repeated_deposits);
    RUN(test_score_applies_the_particle_weight);
    RUN(test_score_tracks_touched_voxels_only);
    RUN(test_score_beam_voxels_are_sorted_and_unique);
    RUN(test_reset_beam_score_clears_both_accumulators);

    RUN(test_input_value_finds_a_single_parsed_pair);
    RUN(test_input_value_finds_every_parsed_pair);
    RUN(test_input_value_on_an_empty_table);
    RUN(test_set_input_value_round_trips);
    RUN(test_set_input_value_fills_the_whole_table);
    RUN(test_clear_input_values_empties_the_table);
    RUN(test_parse_input_file_rejects_an_overfull_deck);
    RUN(test_set_input_value_rejects_one_pair_too_many);
    RUN(test_set_input_value_appends_after_a_parsed_file);

    RUN(test_ssd_source_field_at_the_low_phantom_edge);
    RUN(test_ssd_source_field_is_clamped_to_the_phantom);
    RUN(test_ssd_source_field_inside_the_phantom);
    RUN(test_ssd_source_pencil_beam);

    RUN(test_kn_sigma0_is_positive_and_falls_with_energy);

    printf("\n%d test groups, %d check failures\n", tests_run, tests_failed);

    return tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

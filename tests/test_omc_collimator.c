/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations

 Unit tests for the beam modifier and the aperture mask, omc_collimator: where
 a particle crosses the plane the mask sits on, what the cell it crossed lets
 through, and what happens at the edges of the grid.

 Nothing here transports anything, so it needs no data files and runs from
 wherever CTest starts it. The mask driving a real calculation is checked in
 test_omc_forward_phsp.c.
*****************************************************************************/

#include "omc_collimator.h"
#include "omc_host.h"
#include "omc_random.h"
#include "omc_source.h"
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
        fn();                                                                 \
        printf("%-52s %s\n", #fn,                                             \
               tests_failed == _before ? "ok" : "FAILED");                    \
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
* A 2 x 2 mask of 1 cm cells at z = 10, its lower corner at (-1,-1), so it
* covers x and y from -1 to 1. The cells are open, shut, half open and shut
* again, which is enough to tell a lookup that transposes x and y from one
* that does not.
*******************************************************************************/

static double cells[4] = {
    1.0, 0.0,       /* y in [-1,0): x in [-1,0) open, x in [0,1) shut */
    0.5, 0.0        /* y in [0,1):  x in [-1,0) half, x in [0,1) shut */
};

static struct OmcApertureMask maskFixture(void) {

    struct OmcApertureMask mask;
    memset(&mask, 0, sizeof(mask));

    mask.z = 10.0;
    mask.x0 = -1.0;
    mask.y0 = -1.0;
    mask.dx = 1.0;
    mask.dy = 1.0;
    mask.nx = 2;
    mask.ny = 2;
    mask.transmission = cells;
    mask.outside = 0.0;

    return mask;
}

/* A particle at the origin heading straight down the z axis, which crosses
 the mask plane at (0,0) unless it is aimed elsewhere. */
static struct OmcSourceParticle straight(void) {

    struct OmcSourceParticle p;
    memset(&p, 0, sizeof(p));

    p.charge = 0;
    p.energy = 6.0;
    p.x = 0.0;
    p.y = 0.0;
    p.z = 0.0;
    p.u = 0.0;
    p.v = 0.0;
    p.w = 1.0;
    p.weight = 1.0;

    return p;
}

/*******************************************************************************
* The tests
*******************************************************************************/

/* Nothing in the way lets everything through, and is what an engine given no
 modifier sees. */
static void test_no_modifier_lets_everything_through(void) {

    struct OmcSourceParticle p = straight();

    CHECK_CLOSE(omcBeamModifierTransmission(NULL, &p), 1.0, 1e-15);
}

/* The particle is nowhere near the mask when it is asked about: what decides
 is where its line crosses the plane the mask sits on, ten centimetres away.
 A lookup using the particle's own x and y would put this one in the open
 cell instead of the shut one. */
static void test_the_crossing_point_decides_not_the_position(void) {

    struct OmcApertureMask mask = maskFixture();
    struct OmcBeamModifier modifier;
    omcApertureMaskAsModifier(&mask, &modifier);

    struct OmcSourceParticle p = straight();

    /* Starts in the open cell at (-0.5,-0.5) but travels out to (+0.5,-0.5),
     which is shut. */
    p.x = -0.5;
    p.y = -0.5;
    p.u = 0.1;                  /* 10 cm of travel moves it a centimetre */
    p.w = 1.0;

    CHECK_CLOSE(omcBeamModifierTransmission(&modifier, &p), 0.0, 1e-15);

    /* And the other way round: starts shut, ends open. */
    p.x = 0.5;
    p.u = -0.1;

    CHECK_CLOSE(omcBeamModifierTransmission(&modifier, &p), 1.0, 1e-15);
}

/* Which cell is which, including that x runs fastest rather than y. */
static void test_each_cell_lets_its_own_fraction_through(void) {

    struct OmcApertureMask mask = maskFixture();
    struct OmcBeamModifier modifier;
    omcApertureMaskAsModifier(&mask, &modifier);

    struct OmcSourceParticle p = straight();
    p.z = 10.0;                 /* on the plane already, so x and y are it */

    p.x = -0.5; p.y = -0.5;
    CHECK_CLOSE(omcBeamModifierTransmission(&modifier, &p), 1.0, 1e-15);

    p.x = 0.5;  p.y = -0.5;
    CHECK_CLOSE(omcBeamModifierTransmission(&modifier, &p), 0.0, 1e-15);

    p.x = -0.5; p.y = 0.5;
    CHECK_CLOSE(omcBeamModifierTransmission(&modifier, &p), 0.5, 1e-15);

    p.x = 0.5;  p.y = 0.5;
    CHECK_CLOSE(omcBeamModifierTransmission(&modifier, &p), 0.0, 1e-15);
}

/* Beside the grid is what the field stop does, which is normally to stop
 everything -- but a mask covering only part of a beam can say otherwise. */
static void test_outside_the_grid_takes_the_outside_value(void) {

    struct OmcApertureMask mask = maskFixture();
    struct OmcBeamModifier modifier;
    omcApertureMaskAsModifier(&mask, &modifier);

    struct OmcSourceParticle p = straight();
    p.z = 10.0;
    p.x = 50.0;
    p.y = 0.0;

    CHECK_CLOSE(omcBeamModifierTransmission(&modifier, &p), 0.0, 1e-15);

    mask.outside = 0.25;
    CHECK_CLOSE(omcBeamModifierTransmission(&modifier, &p), 0.25, 1e-15);

    /* Just beyond the far edge is outside; just inside it is not. The cast
     that truncates towards zero used to fold the first cell below the corner
     onto the first one above it, so the low edge is worth its own check. */
    mask.outside = 0.0;

    p.x = -1.0 - 1e-9;  p.y = -0.5;
    CHECK_CLOSE(omcBeamModifierTransmission(&modifier, &p), 0.0, 1e-15);

    p.x = -1.0 + 1e-9;
    CHECK_CLOSE(omcBeamModifierTransmission(&modifier, &p), 1.0, 1e-15);

    p.x = 1.0 - 1e-9;   p.y = -0.5;
    CHECK_CLOSE(omcBeamModifierTransmission(&modifier, &p), 0.0, 1e-15);
}

/* A mask the source put the particle BELOW still works: the crossing point
 of a straight line does not care which way along it you look, which is what
 lets a phase space recorded under the jaws be collimated by them. */
static void test_a_mask_behind_the_particle_still_decides(void) {

    struct OmcApertureMask mask = maskFixture();
    struct OmcBeamModifier modifier;
    omcApertureMaskAsModifier(&mask, &modifier);

    struct OmcSourceParticle p = straight();

    /* Twenty centimetres past the mask, still on the line that went through
     the open cell at (-0.5,-0.5): from (-0.5,-0.5,10) onwards along +z. */
    p.x = -0.5;
    p.y = -0.5;
    p.z = 30.0;
    p.u = 0.0;
    p.v = 0.0;
    p.w = 1.0;

    CHECK_CLOSE(omcBeamModifierTransmission(&modifier, &p), 1.0, 1e-15);
}

/* One travelling along the plane never crosses it. */
static void test_a_particle_parallel_to_the_plane_is_stopped(void) {

    struct OmcApertureMask mask = maskFixture();
    struct OmcBeamModifier modifier;
    omcApertureMaskAsModifier(&mask, &modifier);

    struct OmcSourceParticle p = straight();
    p.x = -0.5;
    p.y = -0.5;
    p.z = 10.0;
    p.u = 1.0;
    p.v = 0.0;
    p.w = 0.0;

    CHECK_CLOSE(omcBeamModifierTransmission(&modifier, &p), 0.0, 1e-15);
}

/* A single open cell is a rectangular field, which is the shape most of the
 use of this will take. */
static void test_one_cell_is_a_rectangular_aperture(void) {

    double open = 1.0;
    struct OmcApertureMask mask;
    memset(&mask, 0, sizeof(mask));

    mask.z = 50.0;
    mask.x0 = -2.5;
    mask.y0 = -3.0;
    mask.dx = 5.0;              /* a 5 x 6 cm field */
    mask.dy = 6.0;
    mask.nx = 1;
    mask.ny = 1;
    mask.transmission = &open;
    mask.outside = 0.0;

    struct OmcBeamModifier modifier;
    omcApertureMaskAsModifier(&mask, &modifier);

    struct OmcSourceParticle p = straight();
    p.z = 50.0;

    p.x = 0.0;   p.y = 0.0;
    CHECK_CLOSE(omcBeamModifierTransmission(&modifier, &p), 1.0, 1e-15);

    p.x = 2.4;   p.y = 2.9;
    CHECK_CLOSE(omcBeamModifierTransmission(&modifier, &p), 1.0, 1e-15);

    p.x = 2.6;   p.y = 0.0;
    CHECK_CLOSE(omcBeamModifierTransmission(&modifier, &p), 0.0, 1e-15);

    p.x = 0.0;   p.y = -3.1;
    CHECK_CLOSE(omcBeamModifierTransmission(&modifier, &p), 0.0, 1e-15);
}

/* THE PROPERTY THAT MATTERS MOST: the mask draws no random numbers, so
 putting a collimator in the beam leaves every history's random stream where
 it was. A collimated run and the open run it came from can then be held
 against each other history by history, and the whole point of indexing the
 generator per history survives. */
static void test_the_mask_draws_no_random_numbers(void) {

    struct OmcApertureMask mask = maskFixture();
    struct OmcBeamModifier modifier;
    omcApertureMaskAsModifier(&mask, &modifier);

    struct OmcSourceParticle p = straight();
    p.z = 10.0;
    p.x = -0.5;
    p.y = -0.5;

    /* What the stream gives without a mask in the way. */
    setRandomHistory(12345);
    double a1 = setRandom();
    double a2 = setRandom();

    /* And with one asked in between. */
    setRandomHistory(12345);
    double b1 = setRandom();
    omcBeamModifierTransmission(&modifier, &p);
    double b2 = setRandom();

    CHECK(a1 == b1);
    CHECK(a2 == b2);
}

/* Applying it, rather than only asking about it: the weight mode spends the
 fraction on the weight and transports everything. */
static void test_weight_mode_scales_the_weight_and_keeps_the_particle(void) {

    struct OmcApertureMask mask = maskFixture();
    struct OmcBeamModifier modifier;
    omcApertureMaskAsModifier(&mask, &modifier);

    CHECK(modifier.apply == OMC_MODIFIER_WEIGHT);   /* what it defaults to */

    struct OmcSourceParticle p = straight();
    p.z = 10.0;

    /* The half open cell: through, at half the weight. */
    p.x = -0.5; p.y = 0.5; p.weight = 4.0;
    CHECK(omcBeamModifierApply(&modifier, &p) == 1);
    CHECK_CLOSE(p.weight, 2.0, 1e-15);

    /* The open one: through, untouched. */
    p.x = -0.5; p.y = -0.5; p.weight = 4.0;
    CHECK(omcBeamModifierApply(&modifier, &p) == 1);
    CHECK_CLOSE(p.weight, 4.0, 1e-15);

    /* The shut one: stopped. */
    p.x = 0.5; p.y = -0.5; p.weight = 4.0;
    CHECK(omcBeamModifierApply(&modifier, &p) == 0);

    /* And nothing in the way at all. */
    p.weight = 4.0;
    CHECK(omcBeamModifierApply(NULL, &p) == 1);
    CHECK_CLOSE(p.weight, 4.0, 1e-15);
}

/* Roulette spends the fraction on whether the particle survives instead, and
 leaves the weight of the ones that do alone -- that is the whole point, since
 a survivor at full weight is worth a full shower. */
static void test_roulette_keeps_the_weight_of_what_survives(void) {

    struct OmcApertureMask mask = maskFixture();
    struct OmcBeamModifier modifier;
    omcApertureMaskAsModifier(&mask, &modifier);
    modifier.apply = OMC_MODIFIER_ROULETTE;

    int survived = 0;

    for (int i = 0; i < 2000; i++) {
        struct OmcSourceParticle p = straight();
        p.z = 10.0;
        p.x = -0.5; p.y = 0.5;      /* the half open cell */
        p.weight = 4.0;

        setRandomHistory((uint64_t)i);

        if (omcBeamModifierApply(&modifier, &p)) {
            survived++;
            CHECK_CLOSE(p.weight, 4.0, 1e-15);
        }
    }

    /* Half of 2000, and three standard deviations of a fair coin over that
     many throws is about 67. */
    CHECK(survived > 900 && survived < 1100);
}

/* The two modes agree on the dose: what roulette carries through in survivors
 at full weight is what the weight mode carries through in everything at a
 fraction of it. Held to a percent, which two thousand throws support. */
static void test_roulette_and_weight_carry_the_same_weight_through(void) {

    struct OmcApertureMask mask = maskFixture();

    /* A fraction that is neither a half nor anything else the arithmetic
     could arrive at by accident. */
    double cell[4] = {0.3, 0.3, 0.3, 0.3};
    mask.transmission = cell;

    struct OmcBeamModifier weighted, rouletted;
    omcApertureMaskAsModifier(&mask, &weighted);
    omcApertureMaskAsModifier(&mask, &rouletted);
    rouletted.apply = OMC_MODIFIER_ROULETTE;

    const int n = 20000;
    double weightSum = 0.0, rouletteSum = 0.0;

    for (int i = 0; i < n; i++) {
        struct OmcSourceParticle a = straight(), b = straight();
        a.z = b.z = 10.0;
        a.x = b.x = -0.5;
        a.y = b.y = -0.5;

        setRandomHistory((uint64_t)i);
        if (omcBeamModifierApply(&weighted, &a)) {
            weightSum += a.weight;
        }

        setRandomHistory((uint64_t)i);
        if (omcBeamModifierApply(&rouletted, &b)) {
            rouletteSum += b.weight;
        }
    }

    CHECK_CLOSE(weightSum/n, 0.3, 1e-12);
    CHECK_CLOSE(rouletteSum/n, 0.3, 0.02);
}

/* What roulette costs, and what it does not. A cell that is fully open or
 fully shut is decided without drawing, so an all-or-nothing aperture -- a jaw,
 which is most of the use of this -- leaves every random stream exactly where
 the weight mode does, and a run behind one can still be compared with the
 open run it came from history by history. Only a partly transmitting cell
 spends a random number. */
static void test_roulette_draws_only_where_it_has_to(void) {

    struct OmcApertureMask mask = maskFixture();
    struct OmcBeamModifier modifier;
    omcApertureMaskAsModifier(&mask, &modifier);
    modifier.apply = OMC_MODIFIER_ROULETTE;

    struct OmcSourceParticle p = straight();
    p.z = 10.0;

    /* What the stream gives when nothing has touched it. */
    setRandomHistory(4242);
    double a1 = setRandom();
    double a2 = setRandom();

    /* The open cell: nothing drawn, so the next number is unchanged. */
    p.x = -0.5; p.y = -0.5;
    setRandomHistory(4242);
    double b1 = setRandom();
    CHECK(omcBeamModifierApply(&modifier, &p) == 1);
    CHECK(b1 == a1);
    CHECK(setRandom() == a2);

    /* The shut one: stopped, and again nothing drawn. */
    p.x = 0.5; p.y = -0.5;
    setRandomHistory(4242);
    CHECK(setRandom() == a1);
    CHECK(omcBeamModifierApply(&modifier, &p) == 0);
    CHECK(setRandom() == a2);

    /* Nothing in the way at all draws nothing either. */
    setRandomHistory(4242);
    CHECK(setRandom() == a1);
    CHECK(omcBeamModifierApply(NULL, &p) == 1);
    CHECK(setRandom() == a2);

    /* The half open one: one number spent, so what follows has moved on. */
    p.x = -0.5; p.y = 0.5;
    setRandomHistory(4242);
    CHECK(setRandom() == a1);
    omcBeamModifierApply(&modifier, &p);
    CHECK(setRandom() != a2);
}

/* And the weight mode never draws, whatever the cell -- the property the
 whole no-random-numbers claim rests on. */
static void test_the_weight_mode_draws_nothing_at_any_cell(void) {

    struct OmcApertureMask mask = maskFixture();
    struct OmcBeamModifier modifier;
    omcApertureMaskAsModifier(&mask, &modifier);

    setRandomHistory(777);
    double a1 = setRandom();
    double a2 = setRandom();

    for (int cell = 0; cell < 4; cell++) {
        struct OmcSourceParticle p = straight();
        p.z = 10.0;
        p.x = (cell % 2) ? 0.5 : -0.5;
        p.y = (cell / 2) ? 0.5 : -0.5;

        setRandomHistory(777);
        CHECK(setRandom() == a1);
        omcBeamModifierApply(&modifier, &p);
        CHECK(setRandom() == a2);
    }
}

static void test_check_rejects_an_unusable_mask(void) {

    struct OmcApertureMask good = maskFixture();
    struct OmcBeamModifier modifier;

    omcApertureMaskAsModifier(&good, &modifier);
    EXPECT_OK(modifier.check(&modifier));

    struct OmcApertureMask empty = maskFixture();
    empty.nx = 0;
    omcApertureMaskAsModifier(&empty, &modifier);
    EXPECT_FAIL("ompMC:collimator:emptyMask", modifier.check(&modifier));

    struct OmcApertureMask thin = maskFixture();
    thin.dx = 0.0;
    omcApertureMaskAsModifier(&thin, &modifier);
    EXPECT_FAIL("ompMC:collimator:badCellSize", modifier.check(&modifier));

    struct OmcApertureMask nothing = maskFixture();
    nothing.transmission = NULL;
    omcApertureMaskAsModifier(&nothing, &modifier);
    EXPECT_FAIL("ompMC:collimator:noTransmission", modifier.check(&modifier));

    /* A transmission above one would quietly multiply the dose. */
    double tooMuch[4] = {1.0, 1.5, 1.0, 1.0};
    struct OmcApertureMask amplifying = maskFixture();
    amplifying.transmission = tooMuch;
    omcApertureMaskAsModifier(&amplifying, &modifier);
    EXPECT_FAIL("ompMC:collimator:badTransmission", modifier.check(&modifier));

    struct OmcApertureMask leaky = maskFixture();
    leaky.outside = -0.5;
    omcApertureMaskAsModifier(&leaky, &modifier);
    EXPECT_FAIL("ompMC:collimator:badTransmission", modifier.check(&modifier));
}

int main(void) {

    /* Unbuffered, so a test that brings the process down still leaves behind
     the list of the ones that got that far. */
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("ompMC collimator tests\n\n");

    omcSetInputValue("rng seeds", "97 33");
    initRandom();

    installFailCatcher();

    RUN(test_no_modifier_lets_everything_through);
    RUN(test_the_crossing_point_decides_not_the_position);
    RUN(test_each_cell_lets_its_own_fraction_through);
    RUN(test_outside_the_grid_takes_the_outside_value);
    RUN(test_a_mask_behind_the_particle_still_decides);
    RUN(test_a_particle_parallel_to_the_plane_is_stopped);
    RUN(test_one_cell_is_a_rectangular_aperture);
    RUN(test_the_mask_draws_no_random_numbers);
    RUN(test_weight_mode_scales_the_weight_and_keeps_the_particle);
    RUN(test_roulette_keeps_the_weight_of_what_survives);
    RUN(test_roulette_and_weight_carry_the_same_weight_through);
    RUN(test_roulette_draws_only_where_it_has_to);
    RUN(test_the_weight_mode_draws_nothing_at_any_cell);
    RUN(test_check_rejects_an_unusable_mask);

    omcSetHost(NULL);

    printf("\n%d test groups, %d check failures\n", tests_run, tests_failed);

    return tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

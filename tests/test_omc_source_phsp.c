/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations

 Unit tests for the phase space source, omc_source_phsp: which particle of the
 file a history draws, where the particle ends up in the phantom, and which
 histories produce nothing at all.

 The phase spaces here are built in memory rather than read from a file. The
 reader has its own tests; what these are about is what happens to a particle
 between coming out of one and going on the stack, so they fill a struct
 OmcPhsp by hand and leave the file format out of it.
*****************************************************************************/

#include "omc_geom.h"
#include "omc_host.h"
#include "omc_phsp.h"
#include "omc_random.h"
#include "omc_score.h"
#include "omc_source_phsp.h"
#include "omc_utilities.h"
#include "ompmc.h"

#include <math.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
* Minimal assertion harness, the same one the other test files use
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
        printf("%-52s ", #fn);                                                \
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
* A phantom to aim at
*
* 8 x 6 x 4 one centimetre voxels, so x runs -4 to 4, y -3 to 3 and z 0 to 4.
* A phase space particle comes from above, i.e. from z below 0 heading towards
* positive z, which is where an IAEA phase space would put a beam.
*******************************************************************************/

#define NX 8
#define NY 6
#define NZ 4

static double xbounds[NX + 1];
static double ybounds[NY + 1];
static double zbounds[NZ + 1];

static void setUpPhantom(void) {

    for (int i = 0; i <= NX; i++) {
        xbounds[i] = -4.0 + 1.0*i;
    }
    for (int j = 0; j <= NY; j++) {
        ybounds[j] = -3.0 + 1.0*j;
    }
    for (int k = 0; k <= NZ; k++) {
        zbounds[k] = 0.0 + 1.0*k;
    }

    geometry.isize = NX;
    geometry.jsize = NY;
    geometry.ksize = NZ;
    geometry.xbounds = xbounds;
    geometry.ybounds = ybounds;
    geometry.zbounds = zbounds;

    omcGeomDetectSpacing();
}

static void tearDownPhantom(void) {

    geometry.xbounds = NULL;
    geometry.ybounds = NULL;
    geometry.zbounds = NULL;
    geometry.isize = 0;
    geometry.jsize = 0;
    geometry.ksize = 0;
}

/*******************************************************************************
* Phase spaces built in memory
*
* struct OmcPhsp holds the records packed the way the file does, so the way to
* build one without a file is to pack them the same way. Everything is stored
* here, which makes a 29 byte record.
*******************************************************************************/

#define TEST_RECORD_LENGTH 29
#define MAX_TEST_RECORDS 8

struct Made {
    struct OmcPhsp phsp;
    unsigned char bytes[MAX_TEST_RECORDS*TEST_RECORD_LENGTH];
};

static void putLeFloat(unsigned char *dst, float value) {

    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));

    dst[0] = (unsigned char)(bits & 0xFF);
    dst[1] = (unsigned char)((bits >> 8) & 0xFF);
    dst[2] = (unsigned char)((bits >> 16) & 0xFF);
    dst[3] = (unsigned char)((bits >> 24) & 0xFF);
}

/* One particle to put in a made up phase space. */
struct Made1 {
    int type;
    double energy;
    double x, y, z;
    double u, v;                /* w comes out of these, as in a real file */
    int backwards;              /* 1 : w points the other way */
    double weight;
};

static void makePhsp(struct Made *made, const struct Made1 *particles,
                     unsigned long long n) {

    memset(made, 0, sizeof(*made));

    made->phsp.header.fileType = 0;
    made->phsp.header.byteOrder = 1234;
    made->phsp.header.recordLength = TEST_RECORD_LENGTH;
    made->phsp.header.particles = n;
    made->phsp.header.checksum = n*TEST_RECORD_LENGTH;
    made->phsp.nRecords = n;
    made->phsp.newHistories = n;        /* every particle its own history */
    made->phsp.raw = made->bytes;
    made->phsp.cursor = 0;

    for (int i = 0; i < OMC_PHSP_NVARIABLES; i++) {
        made->phsp.header.stored[i] = 1;
    }

    for (unsigned long long i = 0; i < n; i++) {
        unsigned char *at = made->bytes + i*TEST_RECORD_LENGTH;
        const struct Made1 *p = &particles[i];
        int type = p->backwards ? -p->type : p->type;

        at[0] = (unsigned char)(type & 0xFF);
        /* A negative energy opens a history, and every particle here does. */
        putLeFloat(at + 1, (float)-p->energy);
        putLeFloat(at + 5, (float)p->x);
        putLeFloat(at + 9, (float)p->y);
        putLeFloat(at + 13, (float)p->z);
        putLeFloat(at + 17, (float)p->u);
        putLeFloat(at + 21, (float)p->v);
        putLeFloat(at + 25, (float)p->weight);
    }
}

/* A 1 MeV photon a centimetre above the middle of the phantom, going straight
 down the beam axis into it. */
static struct Made1 straightDown(void) {

    struct Made1 p;
    memset(&p, 0, sizeof(p));

    p.type = OMC_PHSP_PHOTON;
    p.energy = 1.0;
    p.x = 0.5;
    p.y = 0.5;
    p.z = -1.0;
    p.u = 0.0;
    p.v = 0.0;                  /* so w is +1 */
    p.weight = 1.0;

    return p;
}

/* Draw a history and put it in the phantom, which is what an engine does with
 a source: the source makes the particle, omcSourcePlace() carries it to the
 phantom and decides whether it ever gets there.

 @return 1 if there is a particle on the stack to shower. */
static int sampleAndPlace(const struct OmcPhspSampler *sampler, uint64_t ihist,
                          double weight) {

    struct OmcSourceParticle particle;

    return omcPhspProduce(sampler, ihist, weight, &particle) &&
           omcSourcePlace(&particle);
}

static struct OmcPhspSampler samplerFor(const struct OmcPhsp *phsp) {

    struct OmcPhspSampler sampler;
    memset(&sampler, 0, sizeof(sampler));

    sampler.phsp = phsp;
    sampler.order = OMC_PHSP_REPLAY;
    sampler.first = 0;
    omcPhspTransformIdentity(&sampler.transform);

    return sampler;
}

/*******************************************************************************
* The transform
*******************************************************************************/

static void test_identity_transform_leaves_a_particle_alone(void) {

    struct OmcPhspTransform transform;

    omcPhspTransformIdentity(&transform);

    CHECK_CLOSE(transform.rotation[0], 1.0, 1e-15);
    CHECK_CLOSE(transform.rotation[4], 1.0, 1e-15);
    CHECK_CLOSE(transform.rotation[8], 1.0, 1e-15);
    CHECK_CLOSE(transform.rotation[1], 0.0, 1e-15);
    CHECK_CLOSE(transform.rotation[3], 0.0, 1e-15);
    CHECK_CLOSE(transform.translation[0], 0.0, 1e-15);
    CHECK_CLOSE(transform.translation[2], 0.0, 1e-15);

    setUpPhantom();

    struct Made1 one = straightDown();
    struct Made made;
    makePhsp(&made, &one, 1);

    struct OmcPhspSampler sampler = samplerFor(&made.phsp);

    CHECK(sampleAndPlace(&sampler, 0, 1.0) == 1);

    /* Straight down the axis, so it enters at the top face keeping x and y. */
    CHECK_CLOSE(stack.p[0].x, 0.5, 1e-6);
    CHECK_CLOSE(stack.p[0].y, 0.5, 1e-6);
    CHECK_CLOSE(stack.p[0].z, 0.0, 1e-9);
    CHECK_CLOSE(stack.p[0].w, 1.0, 1e-9);

    tearDownPhantom();
}

/* The phase space is recorded in the coordinate system of the machine that
 made it, and the phantom has its own. A translation is what usually carries
 one to the other, so the particle has to be moved and the direction left
 alone. */
static void test_translation_moves_the_particle_not_its_direction(void) {

    setUpPhantom();

    struct Made1 one = straightDown();
    one.x = 0.0;
    one.y = 0.0;

    struct Made made;
    makePhsp(&made, &one, 1);

    struct OmcPhspSampler sampler = samplerFor(&made.phsp);
    sampler.transform.translation[0] = 2.5;
    sampler.transform.translation[1] = -1.5;

    CHECK(sampleAndPlace(&sampler, 0, 1.0) == 1);

    CHECK_CLOSE(stack.p[0].x, 2.5, 1e-6);
    CHECK_CLOSE(stack.p[0].y, -1.5, 1e-6);

    /* Still heading straight down the beam. */
    CHECK_CLOSE(stack.p[0].u, 0.0, 1e-9);
    CHECK_CLOSE(stack.p[0].v, 0.0, 1e-9);
    CHECK_CLOSE(stack.p[0].w, 1.0, 1e-9);

    tearDownPhantom();
}

/* And a rotation turns both. This one is a quarter turn about z, which sends
 x to y and y to -x. */
static void test_rotation_turns_the_particle_and_its_direction(void) {

    setUpPhantom();

    struct Made1 one = straightDown();
    one.x = 2.0;
    one.y = 0.0;
    one.u = 0.6;                /* so w is 0.8, still going into the phantom */
    one.v = 0.0;

    struct Made made;
    makePhsp(&made, &one, 1);

    struct OmcPhspSampler sampler = samplerFor(&made.phsp);

    /* x -> y, y -> -x, z -> z */
    double quarterTurn[9] = {0.0, -1.0, 0.0,
                             1.0,  0.0, 0.0,
                             0.0,  0.0, 1.0};
    memcpy(sampler.transform.rotation, quarterTurn, sizeof(quarterTurn));

    CHECK(sampleAndPlace(&sampler, 0, 1.0) == 1);

    /* The direction turned with it: u went to v. */
    CHECK_CLOSE(stack.p[0].u, 0.0, 1e-9);
    CHECK_CLOSE(stack.p[0].v, 0.6, 1e-6);
    CHECK_CLOSE(stack.p[0].w, 0.8, 1e-6);

    /* It started at (2,0,-1), which the turn sends to (0,2,-1), and from
     there 1/0.8 of a step down the direction reaches the top face. */
    CHECK_CLOSE(stack.p[0].x, 0.0, 1e-6);
    CHECK_CLOSE(stack.p[0].y, 2.0 + 0.6/0.8, 1e-6);
    CHECK_CLOSE(stack.p[0].z, 0.0, 1e-9);

    tearDownPhantom();
}

/* A matrix that is not a rotation would stretch the directions it turns, and
 they have to come out unit vectors. Caught before the histories start. */
static void test_a_transform_that_is_not_a_rotation_is_refused(void) {

    setUpPhantom();

    struct Made1 one = straightDown();
    struct Made made;
    makePhsp(&made, &one, 1);

    struct OmcPhspSampler sampler = samplerFor(&made.phsp);
    sampler.transform.rotation[0] = 2.0;        /* scales x */

    EXPECT_FAIL("ompMC:phspSource:notARotation", omcPhspSourceCheck(&sampler));

    tearDownPhantom();
}

/* The determinant on its own does not settle it, and these are the three ways
 a matrix gets past that check without being a rotation. */
static void test_the_rotation_check_is_not_only_the_determinant(void) {

    setUpPhantom();

    struct Made1 one = straightDown();
    struct Made made;
    makePhsp(&made, &one, 1);

    /* A shear. Determinant exactly 1, and it still stretches what it turns:
     (0,1,0) comes out (1,1,0), which is no longer a unit vector. */
    struct OmcPhspSampler shear = samplerFor(&made.phsp);
    shear.transform.rotation[1] = 1.0;

    EXPECT_FAIL("ompMC:phspSource:notARotation", omcPhspSourceCheck(&shear));

    /* A reflection. Orthonormal rows, determinant -1, and it turns a right
     handed coordinate system into a left handed one. */
    struct OmcPhspSampler mirror = samplerFor(&made.phsp);
    mirror.transform.rotation[0] = -1.0;

    EXPECT_FAIL("ompMC:phspSource:notARotation", omcPhspSourceCheck(&mirror));

    /* And a matrix holding a NaN, which used to pass because every
     comparison against NaN is false, including the one that was meant to
     turn it away. */
    struct OmcPhspSampler broken = samplerFor(&made.phsp);
    broken.transform.rotation[4] = nan("");     /* 0.0/0.0 is a compile
                                                   error on MSVC */

    EXPECT_FAIL("ompMC:phspSource:notARotation", omcPhspSourceCheck(&broken));

    tearDownPhantom();
}

/*******************************************************************************
* Getting into the phantom
*******************************************************************************/

/* The particle is recorded well outside the phantom and has to be carried to
 the face it goes in by, not dropped where it was found. */
static void test_particle_is_carried_to_the_phantom_surface(void) {

    setUpPhantom();

    struct Made1 one = straightDown();
    one.x = -1.5;
    one.y = 2.5;
    one.z = -50.0;              /* half a metre above the phantom */

    struct Made made;
    makePhsp(&made, &one, 1);

    struct OmcPhspSampler sampler = samplerFor(&made.phsp);

    CHECK(sampleAndPlace(&sampler, 0, 1.0) == 1);

    CHECK_CLOSE(stack.p[0].z, 0.0, 1e-9);
    CHECK_CLOSE(stack.p[0].x, -1.5, 1e-6);
    CHECK_CLOSE(stack.p[0].y, 2.5, 1e-6);

    /* And it knows which voxel that is: x = -1.5 is the third along x,
     y = 2.5 the sixth along y, z = 0 the first along z. */
    CHECK(stack.p[0].ir == 1 + 2 + 5*NX + 0*NX*NY);

    tearDownPhantom();
}

/* A particle coming in from the side enters through a side face. */
static void test_particle_entering_from_the_side(void) {

    setUpPhantom();

    struct Made1 one = straightDown();
    one.x = -10.0;
    one.y = 0.5;
    one.z = 2.5;                /* level with the middle of the phantom */
    one.u = 1.0;                /* straight along +x, so v and w are 0 */
    one.v = 0.0;

    struct Made made;
    makePhsp(&made, &one, 1);

    struct OmcPhspSampler sampler = samplerFor(&made.phsp);

    CHECK(sampleAndPlace(&sampler, 0, 1.0) == 1);

    CHECK_CLOSE(stack.p[0].x, -4.0, 1e-9);
    CHECK_CLOSE(stack.p[0].y, 0.5, 1e-6);
    CHECK_CLOSE(stack.p[0].z, 2.5, 1e-6);
    CHECK(stack.p[0].ir == 1 + 0 + 3*NX + 2*NX*NY);

    tearDownPhantom();
}

/* One already inside stays where it is rather than being pushed backwards to
 a face it never crossed. */
static void test_particle_already_inside_stays_put(void) {

    setUpPhantom();

    struct Made1 one = straightDown();
    one.x = 1.5;
    one.y = -0.5;
    one.z = 2.5;

    struct Made made;
    makePhsp(&made, &one, 1);

    struct OmcPhspSampler sampler = samplerFor(&made.phsp);

    CHECK(sampleAndPlace(&sampler, 0, 1.0) == 1);

    CHECK_CLOSE(stack.p[0].x, 1.5, 1e-6);
    CHECK_CLOSE(stack.p[0].y, -0.5, 1e-6);
    CHECK_CLOSE(stack.p[0].z, 2.5, 1e-6);

    tearDownPhantom();
}

/* A phase space particle is recorded wherever the simulation that made it
 scored one, so plenty of them are pointing nowhere near the phantom. Those
 histories produce nothing, and the engine has to be told so rather than
 showering whatever the stack happened to hold. */
static void test_particle_missing_the_phantom_produces_nothing(void) {

    setUpPhantom();

    struct Made1 wide = straightDown();
    wide.x = 40.0;              /* well beyond the phantom in x */
    wide.y = 0.0;
    wide.z = -1.0;              /* going straight down, so it never comes back */

    struct Made made;
    makePhsp(&made, &wide, 1);

    struct OmcPhspSampler sampler = samplerFor(&made.phsp);

    CHECK(sampleAndPlace(&sampler, 0, 1.0) == 0);

    tearDownPhantom();
}

/* And one heading away from it produces nothing either, however well aimed
 the line it travels on would be if it were reversed. */
static void test_particle_heading_away_produces_nothing(void) {

    setUpPhantom();

    struct Made1 away = straightDown();
    away.x = 0.5;
    away.y = 0.5;
    away.z = -1.0;
    away.backwards = 1;         /* w = -1: upwards, away from the phantom */

    struct Made made;
    makePhsp(&made, &away, 1);

    struct OmcPhspSampler sampler = samplerFor(&made.phsp);

    CHECK(sampleAndPlace(&sampler, 0, 1.0) == 0);

    tearDownPhantom();
}

/*******************************************************************************
* What lands on the stack
*******************************************************************************/

/* The charges ompMC works in, and the rest energy a charged particle carries
 on top of the kinetic energy the file records. */
static void test_particle_types_become_charges(void) {

    setUpPhantom();

    struct Made1 three[3];
    for (int i = 0; i < 3; i++) {
        three[i] = straightDown();
        three[i].energy = 2.0;
    }
    three[0].type = OMC_PHSP_PHOTON;
    three[1].type = OMC_PHSP_ELECTRON;
    three[2].type = OMC_PHSP_POSITRON;

    struct Made made;
    makePhsp(&made, three, 3);

    struct OmcPhspSampler sampler = samplerFor(&made.phsp);

    CHECK(sampleAndPlace(&sampler, 0, 1.0) == 1);
    CHECK(stack.p[0].iq == 0);
    CHECK_CLOSE(stack.p[0].e, 2.0, 1e-6);

    CHECK(sampleAndPlace(&sampler, 1, 1.0) == 1);
    CHECK(stack.p[0].iq == -1);
    CHECK_CLOSE(stack.p[0].e, 2.0 + RM, 1e-6);

    CHECK(sampleAndPlace(&sampler, 2, 1.0) == 1);
    CHECK(stack.p[0].iq == 1);
    CHECK_CLOSE(stack.p[0].e, 2.0 + RM, 1e-6);

    tearDownPhantom();
}

/* A phase space may hold particles ompMC has no physics for. That does not
 make the file unreadable, it makes those histories empty. */
static void test_neutrons_and_protons_produce_nothing(void) {

    setUpPhantom();

    struct Made1 two[2];
    two[0] = straightDown();
    two[1] = straightDown();
    two[0].type = OMC_PHSP_NEUTRON;
    two[1].type = OMC_PHSP_PROTON;

    struct Made made;
    makePhsp(&made, two, 2);

    struct OmcPhspSampler sampler = samplerFor(&made.phsp);

    CHECK(sampleAndPlace(&sampler, 0, 1.0) == 0);
    CHECK(sampleAndPlace(&sampler, 1, 1.0) == 0);

    tearDownPhantom();
}

/* The weight in the file and the one the caller passes both count. */
static void test_weights_multiply(void) {

    setUpPhantom();

    struct Made1 one = straightDown();
    one.weight = 0.25;

    struct Made made;
    makePhsp(&made, &one, 1);

    struct OmcPhspSampler sampler = samplerFor(&made.phsp);

    CHECK(sampleAndPlace(&sampler, 0, 1.0) == 1);
    CHECK_CLOSE(stack.p[0].wt, 0.25, 1e-9);

    CHECK(sampleAndPlace(&sampler, 0, 4.0) == 1);
    CHECK_CLOSE(stack.p[0].wt, 1.0, 1e-9);

    tearDownPhantom();
}

/*******************************************************************************
* Which particle a history draws
*******************************************************************************/

/* Replaying walks the file in order and wraps at the end of it, so a run
 longer than the file uses every particle equally often. */
static void test_replay_walks_the_file_in_order(void) {

    setUpPhantom();

    struct Made1 three[3];
    for (int i = 0; i < 3; i++) {
        three[i] = straightDown();
        three[i].energy = 1.0 + i;
    }

    struct Made made;
    makePhsp(&made, three, 3);

    struct OmcPhspSampler sampler = samplerFor(&made.phsp);

    for (uint64_t ihist = 0; ihist < 7; ihist++) {
        CHECK(sampleAndPlace(&sampler, ihist, 1.0) == 1);
        CHECK_CLOSE(stack.p[0].e, 1.0 + (double)(ihist % 3), 1e-6);
    }

    /* And it can be started anywhere in the file. */
    sampler.first = 2;
    CHECK(sampleAndPlace(&sampler, 0, 1.0) == 1);
    CHECK_CLOSE(stack.p[0].e, 3.0, 1e-6);
    CHECK(sampleAndPlace(&sampler, 1, 1.0) == 1);
    CHECK_CLOSE(stack.p[0].e, 1.0, 1e-6);

    tearDownPhantom();
}

/* Which particle a history draws depends on the history index and nothing
 else. That is what keeps a run from depending on how OpenMP handed the
 histories out, and it means the source must not be walking a cursor of its
 own: asking for them out of order has to give the same answers. */
static void test_the_draw_depends_only_on_the_history_index(void) {

    setUpPhantom();

    struct Made1 four[4];
    for (int i = 0; i < 4; i++) {
        four[i] = straightDown();
        four[i].energy = 1.0 + i;
    }

    struct Made made;
    makePhsp(&made, four, 4);

    struct OmcPhspSampler sampler = samplerFor(&made.phsp);

    const uint64_t scrambled[6] = {3, 0, 2, 1, 3, 0};

    for (int i = 0; i < 6; i++) {
        CHECK(sampleAndPlace(&sampler, scrambled[i], 1.0) == 1);
        CHECK_CLOSE(stack.p[0].e, 1.0 + (double)scrambled[i], 1e-6);
    }

    /* The read position omcPhspNext() uses is shared by every thread, so the
     source has to leave it exactly where it found it. */
    CHECK(made.phsp.cursor == 0);

    tearDownPhantom();
}

/* Drawing at random gives the same particle for the same history, because the
 random stream is indexed by the history rather than carried along. */
static void test_random_draws_are_reproducible_per_history(void) {

    setUpPhantom();

    struct Made1 four[4];
    for (int i = 0; i < 4; i++) {
        four[i] = straightDown();
        four[i].energy = 1.0 + i;
    }

    struct Made made;
    makePhsp(&made, four, 4);

    struct OmcPhspSampler sampler = samplerFor(&made.phsp);
    sampler.order = OMC_PHSP_RANDOM;

    double drawn[5];
    int seen[4] = {0};

    for (uint64_t ihist = 0; ihist < 5; ihist++) {
        setRandomHistory(ihist);
        CHECK(sampleAndPlace(&sampler, ihist, 1.0) == 1);
        drawn[ihist] = stack.p[0].e;

        int which = (int)(drawn[ihist] - 1.0 + 0.5);
        CHECK(which >= 0 && which < 4);
        if (which >= 0 && which < 4) {
            seen[which]++;
        }
    }

    /* Ask the same histories again, in the other order. */
    for (int ihist = 4; ihist >= 0; ihist--) {
        setRandomHistory((uint64_t)ihist);
        CHECK(sampleAndPlace(&sampler, (uint64_t)ihist, 1.0) == 1);
        CHECK_CLOSE(stack.p[0].e, drawn[ihist], 1e-12);
    }

    tearDownPhantom();
}

/*******************************************************************************
* Checking the sampler over
*******************************************************************************/

static void test_check_refuses_an_empty_or_confused_sampler(void) {

    setUpPhantom();

    struct Made1 one = straightDown();
    struct Made made;
    makePhsp(&made, &one, 1);

    struct OmcPhspSampler empty = samplerFor(&made.phsp);
    empty.phsp = NULL;
    EXPECT_FAIL("ompMC:phspSource:noParticles", omcPhspSourceCheck(&empty));

    struct OmcPhspSampler odd = samplerFor(&made.phsp);
    odd.order = (enum OmcPhspOrder)7;
    EXPECT_FAIL("ompMC:phspSource:badOrder", omcPhspSourceCheck(&odd));

    struct OmcPhspSampler good = samplerFor(&made.phsp);
    EXPECT_OK(omcPhspSourceCheck(&good));

    /* A caller who skips the check and hands over an empty phase space gets
     empty histories rather than a division by zero in the wrap. */
    struct Made none;
    makePhsp(&none, NULL, 0);

    struct OmcPhspSampler nothing = samplerFor(&none.phsp);
    CHECK(sampleAndPlace(&nothing, 0, 1.0) == 0);

    /* And without a phantom there is nothing to work out an entry point in. */
    tearDownPhantom();
    EXPECT_FAIL("ompMC:phspSource:noGeometry", omcPhspSourceCheck(&good));
}

int main(void) {

    /* Unbuffered, so a test that brings the process down still leaves behind
     the list of the ones that got that far. */
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("ompMC phase space source tests\n\n");

    /* The source puts particles on the thread local stack, which the library
     allocates per thread. This test runs on one. */
    initStack();

    /* initRandom() takes its key from the input table, which no deck filled
     here, so put one there. */
    omcSetInputValue("rng seeds", "97 33");
    initRandom();
    setRandomHistory(0);

    installFailCatcher();

    RUN(test_identity_transform_leaves_a_particle_alone);
    RUN(test_translation_moves_the_particle_not_its_direction);
    RUN(test_rotation_turns_the_particle_and_its_direction);
    RUN(test_a_transform_that_is_not_a_rotation_is_refused);
    RUN(test_the_rotation_check_is_not_only_the_determinant);

    RUN(test_particle_is_carried_to_the_phantom_surface);
    RUN(test_particle_entering_from_the_side);
    RUN(test_particle_already_inside_stays_put);
    RUN(test_particle_missing_the_phantom_produces_nothing);
    RUN(test_particle_heading_away_produces_nothing);

    RUN(test_particle_types_become_charges);
    RUN(test_neutrons_and_protons_produce_nothing);
    RUN(test_weights_multiply);

    RUN(test_replay_walks_the_file_in_order);
    RUN(test_the_draw_depends_only_on_the_history_index);
    RUN(test_random_draws_are_reproducible_per_history);

    RUN(test_check_refuses_an_empty_or_confused_sampler);

    omcSetHost(NULL);

    printf("\n%d test groups, %d check failures\n", tests_run, tests_failed);

    return tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

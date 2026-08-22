/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations

 The cylindrical (r-z) geometry on its own: which region a point falls in, how
 far a step may go before it leaves one, and how near the boundaries are.

 Nothing here transports. howfar(), hownear() and regionIndex() read the
 geometry and the particle stack and nothing else -- no cross sections, no
 PEGS data, no media -- so the fixture is a handful of hand-placed particles
 and this test needs no working directory of its own.

 The two things worth watching are the ones a rectilinear geometry never had
 to answer. A ray can leave a ring through the surface it came in by, so the
 radial distance is a quadratic and not a subtraction; and a ray heading
 inwards can miss the inner cylinder altogether, passing the axis on one side,
 which shows up as a negative discriminant that has to mean "no crossing"
 rather than a NaN.
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
#include "omc_source.h"
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
        printf("%-52s ", #fn);                                                \
        fn();                                                                 \
        printf("%s\n", tests_failed == _before ? "ok" : "FAILED");            \
    } while (0)

static void silentLog(int level, const char *message, void *user) {
    (void)level; (void)message; (void)user;
}

/* For the tests that are about a geometry being refused rather than one
 working. omcFail() does not return, so the only way back is a longjmp to the
 caller that armed one. */
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
* A water-sized cylinder: 5 cm across in five 1 cm rings, 10 cm deep in ten
* 1 cm slabs, front face at z = 0.
*
* The numbers are chosen so that ring and slab boundaries land on integers and
* the expected distances can be read off by hand.
*******************************************************************************/

#define NR 5
#define NZC 10
#define NREGIONS (NR*NZC)

static double rb[NR + 1], zbc[NZC + 1];
static int cylMedIndices[NREGIONS];
static double cylMedDensities[NREGIONS];

/* The region memo in omc_geom_cache is keyed on the region number alone, so a
 test that rebuilds the geometry under the same region numbers has to clear
 it. Staging region 0 -- which is outside the geometry and never looked up --
 is how the memo is spelled "empty". */
static void resetGeomCache(void) {
    omcStageRegion(0, 0, 0, 0);
}

static void setUpCylinder(void) {

    for (int i = 0; i <= NR; i++) rb[i] = 1.0*i;
    for (int k = 0; k <= NZC; k++) zbc[k] = 1.0*k;

    memset(&geometry, 0, sizeof(geometry));

    geometry.isize = NR;
    geometry.ksize = NZC;
    geometry.rbounds = rb;
    geometry.zbounds = zbc;

    for (int i = 0; i < NREGIONS; i++) {
        cylMedIndices[i] = 1;
        cylMedDensities[i] = 1.0;
    }
    geometry.med_indices = cylMedIndices;
    geometry.med_densities = cylMedDensities;

    resetGeomCache();
    omcGeomCylInit();
}

/* Put a particle in the geometry by hand. The region is worked out from the
 position, which is what the transport would have done. */
static void place(double x, double y, double z, double u, double v, double w) {

    stack.np = 0;
    stack.p[0].x = x;
    stack.p[0].y = y;
    stack.p[0].z = z;
    stack.p[0].u = u;
    stack.p[0].v = v;
    stack.p[0].w = w;
    stack.p[0].wt = 1.0;
    stack.p[0].iq = 0;
    stack.p[0].e = 1.0;
    stack.p[0].ir = regionIndex(x, y, z);

    resetGeomCache();
}

/* howfar() as the transport calls it: a step long enough that any boundary
 truncates it, and irnew only meaningful if it did. */
static void step(double *ustep, int *irnew, int *idisc) {

    *ustep = 1.0E10;
    *irnew = -1;
    *idisc = 0;

    howfar(idisc, irnew, ustep);
}

/* The textbook quadratic, deliberately written the way the implementation is
 not: this is what the rationalized roots in omc_geom_cyl.c have to agree
 with. */
static double outerRoot(double x, double y, double u, double v, double radius) {

    double a = u*u + v*v;
    double b = 2.0*(x*u + y*v);
    double c = x*x + y*y - radius*radius;
    double disc = b*b - 4.0*a*c;

    return (-b + sqrt(disc))/(2.0*a);
}

static double innerRoot(double x, double y, double u, double v, double radius) {

    double a = u*u + v*v;
    double b = 2.0*(x*u + y*v);
    double c = x*x + y*y - radius*radius;
    double disc = b*b - 4.0*a*c;

    if (disc < 0.0) {
        return -1.0;            /* the ray passes the axis on one side */
    }

    return (-b - sqrt(disc))/(2.0*a);
}

/*******************************************************************************
* The tests
*******************************************************************************/

static void test_region_index_maps_rings_and_slabs(void) {

    setUpCylinder();

    /* On the axis, in the first slab */
    CHECK(regionIndex(0.0, 0.0, 0.5) == 1);

    /* r = 2.5 is ring 2, z = 3.5 is slab 3 */
    CHECK(regionIndex(2.5, 0.0, 3.5) == 1 + 2 + 3*NR);
    CHECK(regionIndex(0.0, 2.5, 3.5) == 1 + 2 + 3*NR);
    CHECK(regionIndex(-2.5, 0.0, 3.5) == 1 + 2 + 3*NR);

    /* Only r matters, not where on the ring: (3,4) is r = 5 exactly, which
     counts as inside and clamps into the outermost ring. */
    CHECK(regionIndex(3.0, 4.0, 2.5) == 1 + (NR - 1) + 2*NR);

    /* Outside, in each of the three ways there are */
    CHECK(regionIndex(5.5, 0.0, 2.5) == 0);
    CHECK(regionIndex(0.0, 0.0, -0.1) == 0);
    CHECK(regionIndex(0.0, 0.0, 10.1) == 0);
}

static void test_cyl_init_refuses_a_geometry_it_cannot_transport(void) {

    catchingHost();

    /* A hollow cylinder: region 0 means "outside", so there is nowhere for
     the hole to be. */
    setUpCylinder();
    rb[0] = 0.5;
    EXPECT_FAIL("ompMC:geomCyl:badRadialBounds", omcGeomCylInit());
    rb[0] = 0.0;

    setUpCylinder();
    rb[3] = rb[2];
    EXPECT_FAIL("ompMC:geomCyl:badRadialBounds", omcGeomCylInit());
    rb[3] = 3.0;

    setUpCylinder();
    zbc[4] = zbc[3] - 1.0;
    EXPECT_FAIL("ompMC:geomCyl:badDepthBounds", omcGeomCylInit());
    zbc[4] = 4.0;

    setUpCylinder();
    geometry.isize = 0;
    EXPECT_FAIL("ompMC:geomCyl:badGrid", omcGeomCylInit());

    setUpCylinder();
    geometry.ksize = 0;
    EXPECT_FAIL("ompMC:geomCyl:badGrid", omcGeomCylInit());

    setUpCylinder();
    geometry.rbounds = NULL;
    EXPECT_FAIL("ompMC:geomCyl:badRadialBounds", omcGeomCylInit());

    omcSetHost(NULL);
}

/* The azimuthal index is the geometry's own business: a host says how many
 rings and how many slabs, and never has to know that they are carried in the
 rectilinear grid's i and k. */
static void test_cyl_init_takes_the_azimuthal_index_out_of_play(void) {

    setUpCylinder();

    CHECK(geometry.jsize == 1);
    CHECK(geometry.mode == OMC_GEOM_CYLINDRICAL);

    /* Whatever a host left there */
    geometry.jsize = 17;
    omcGeomCylInit();
    CHECK(geometry.jsize == 1);
}

static void test_howfar_outward_along_a_diameter(void) {

    setUpCylinder();

    double ustep;
    int irnew, idisc;

    /* From the axis, straight out: one centimetre to the first ring
     boundary, and the neighbour is the next ring up. */
    place(0.0, 0.0, 0.5, 1.0, 0.0, 0.0);
    step(&ustep, &irnew, &idisc);
    CHECK(idisc == 0);
    CHECK_CLOSE(ustep, 1.0, 1e-12);
    CHECK(irnew == 2);

    /* Halfway through the outermost ring, still heading out: half a
     centimetre to the barrel, and past it is outside. */
    place(4.5, 0.0, 0.5, 1.0, 0.0, 0.0);
    step(&ustep, &irnew, &idisc);
    CHECK_CLOSE(ustep, 0.5, 1e-12);
    CHECK(irnew == 0);

    /* Direction on the ring makes no difference */
    place(0.0, -4.5, 0.5, 0.0, -1.0, 0.0);
    step(&ustep, &irnew, &idisc);
    CHECK_CLOSE(ustep, 0.5, 1e-12);
    CHECK(irnew == 0);

    /* A particle already outside is discarded rather than stepped */
    stack.np = 0;
    stack.p[0].ir = 0;
    step(&ustep, &irnew, &idisc);
    CHECK(idisc == 1);
}

static void test_howfar_inward_and_the_chord_that_misses(void) {

    setUpCylinder();

    double ustep;
    int irnew, idisc;

    /* Heading in from the middle of ring 3: half a centimetre to r = 3, and
     the neighbour is the ring below. */
    place(3.5, 0.0, 0.5, -1.0, 0.0, 0.0);
    step(&ustep, &irnew, &idisc);
    CHECK_CLOSE(ustep, 0.5, 1e-12);
    CHECK(irnew == 1 + 2);

    /* Tangentially, so b = 0 and the inner cylinder is not approached at
     all: the step runs to the outer surface of the same ring, and past it is
     the ring above. */
    place(3.5, 0.0, 0.5, 0.0, 1.0, 0.0);
    step(&ustep, &irnew, &idisc);
    CHECK_CLOSE(ustep, sqrt(16.0 - 12.25), 1e-12);
    CHECK(irnew == 1 + 4);

    /* Heading inwards, but on a chord that passes the axis outside r = 3.
     The inner cylinder is never reached -- the discriminant goes negative --
     and the particle leaves through the outer surface instead. */
    double u = -0.3;
    double v = sqrt(1.0 - u*u);

    CHECK(innerRoot(3.5, 0.0, u, v, 3.0) < 0.0);    /* the premise */

    place(3.5, 0.0, 0.5, u, v, 0.0);
    step(&ustep, &irnew, &idisc);
    CHECK_CLOSE(ustep, outerRoot(3.5, 0.0, u, v, 4.0), 1e-12);
    CHECK(irnew == 1 + 4);
}

/* A ray along the axis crosses no ring boundary at all. The radial quadratic
 degenerates -- a = 0 -- and dividing by it would be an infinity where the
 answer is "never". */
static void test_howfar_along_the_axis_crosses_no_ring(void) {

    setUpCylinder();

    double ustep;
    int irnew, idisc;

    place(0.0, 0.0, 0.5, 0.0, 0.0, 1.0);
    step(&ustep, &irnew, &idisc);
    CHECK_CLOSE(ustep, 0.5, 1e-12);
    CHECK(irnew == 1 + NR);                     /* the next slab down */

    /* Off the axis but still parallel to it: same answer, no ring crossing */
    place(2.5, 0.0, 0.5, 0.0, 0.0, 1.0);
    step(&ustep, &irnew, &idisc);
    CHECK_CLOSE(ustep, 0.5, 1e-12);
    CHECK(irnew == 1 + 2 + NR);

    /* Out of the back face */
    place(2.5, 0.0, 9.5, 0.0, 0.0, 1.0);
    step(&ustep, &irnew, &idisc);
    CHECK_CLOSE(ustep, 0.5, 1e-12);
    CHECK(irnew == 0);

    /* And out of the front one */
    place(2.5, 0.0, 0.5, 0.0, 0.0, -1.0);
    step(&ustep, &irnew, &idisc);
    CHECK_CLOSE(ustep, 0.5, 1e-12);
    CHECK(irnew == 0);
}

static void test_howfar_oblique_agrees_with_the_quadratic(void) {

    setUpCylinder();

    double ustep;
    int irnew, idisc;

    /* Oblique in all three, from ring 1 outwards. Mostly across the beam, so
     that the depth face stays further off than the ring surface and the
     radial root is what truncates. */
    double u = 0.96, v = 0.0, w = 0.28;

    place(1.5, 0.0, 5.5, u, v, w);
    step(&ustep, &irnew, &idisc);
    CHECK_CLOSE(ustep, outerRoot(1.5, 0.0, u, v, 2.0), 1e-12);
    CHECK(irnew == 1 + 2 + 5*NR);

    /* Obliquely inwards, onto a chord that does reach the inner surface */
    u = -0.8; v = 0.0; w = 0.6;
    place(1.5, 0.0, 4.5, u, v, w);
    {
        double expected = innerRoot(1.5, 0.0, u, v, 1.0);
        CHECK(expected > 0.0);                  /* it does reach r = 1 */
        step(&ustep, &irnew, &idisc);
        CHECK_CLOSE(ustep, expected, 1e-12);
        CHECK(irnew == 1 + 0 + 4*NR);
    }

    /* A step that the z face wins */
    place(2.5, 0.0, 0.95, 0.1, 0.0, sqrt(1.0 - 0.01));
    step(&ustep, &irnew, &idisc);
    CHECK_CLOSE(ustep, 0.05/sqrt(1.0 - 0.01), 1e-12);
    CHECK(irnew == 1 + 2 + NR);
}

static void test_hownear_is_the_smallest_of_four(void) {

    setUpCylinder();

    /* Dead centre of a ring and a slab: every one of the four is half a
     centimetre away. */
    place(2.5, 0.0, 0.5, 0.0, 0.0, 1.0);
    CHECK_CLOSE(hownear(), 0.5, 1e-12);

    /* Nearer the outer surface of the ring than anything else */
    place(2.9, 0.0, 0.5, 0.0, 0.0, 1.0);
    CHECK_CLOSE(hownear(), 0.1, 1e-12);

    /* Nearer the inner one */
    place(2.05, 0.0, 0.5, 0.0, 0.0, 1.0);
    CHECK_CLOSE(hownear(), 0.05, 1e-12);

    /* Nearer a z face */
    place(2.5, 0.0, 0.02, 0.0, 0.0, 1.0);
    CHECK_CLOSE(hownear(), 0.02, 1e-12);

    /* On the axis there is no inner surface to be near, and no division by r
     to raise an infinity. The first ring boundary is a centimetre away, so
     the z faces are what is nearest -- the point being that the missing
     inner surface contributes neither a spurious zero nor a NaN. */
    place(0.0, 0.0, 0.5, 0.0, 0.0, 1.0);
    CHECK_CLOSE(hownear(), 0.5, 1e-12);

    place(0.0, 0.0, 5.5, 0.0, 0.0, 1.0);
    CHECK_CLOSE(hownear(), 0.5, 1e-12);

    /* Squeeze the slab so that the ring surface is what wins on the axis */
    zbc[5] = 5.45;
    resetGeomCache();
    omcGeomCylInit();
    place(0.0, 0.0, 5.5, 0.0, 0.0, 1.0);
    CHECK_CLOSE(hownear(), 0.05, 1e-12);
    zbc[5] = 5.0;

    /* Off the axis but still in the first ring, which has no inner bound */
    place(0.9, 0.0, 5.5, 0.0, 0.0, 1.0);
    CHECK_CLOSE(hownear(), 0.1, 1e-12);

    /* Outside, hownear() is zero whatever the position says */
    stack.np = 0;
    stack.p[0].ir = 0;
    CHECK_CLOSE(hownear(), 0.0, 1e-15);
}

/* The one that would catch an off-by-one between the two: walk a ray across
 the geometry, and at every boundary howfar() truncates at, check that the
 region it says the particle is entering is the region the point actually
 falls in. */
static void test_howfar_and_region_index_agree(void) {

    setUpCylinder();

    uint32_t seed = 20240822u;

    for (int iray = 0; iray < 200; iray++) {

        /* A start and a direction, deterministic so a failure is
         reproducible */
        seed = seed*1664525u + 1013904223u;
        double rstart = 4.9*((double)(seed >> 8)/16777216.0);
        seed = seed*1664525u + 1013904223u;
        double phi = 6.283185307179586*((double)(seed >> 8)/16777216.0);
        seed = seed*1664525u + 1013904223u;
        double zstart = 0.05 + 9.9*((double)(seed >> 8)/16777216.0);

        seed = seed*1664525u + 1013904223u;
        double costh = -1.0 + 2.0*((double)(seed >> 8)/16777216.0);
        seed = seed*1664525u + 1013904223u;
        double psi = 6.283185307179586*((double)(seed >> 8)/16777216.0);

        double sinth = sqrt(1.0 - costh*costh);

        double x = rstart*cos(phi);
        double y = rstart*sin(phi);
        double z = zstart;
        double u = sinth*cos(psi);
        double v = sinth*sin(psi);
        double w = costh;

        for (int nstep = 0; nstep < 64; nstep++) {

            place(x, y, z, u, v, w);
            int irl = stack.p[0].ir;

            if (irl == 0) {
                break;
            }

            double ustep;
            int irnew, idisc;
            step(&ustep, &irnew, &idisc);

            CHECK(idisc == 0);
            CHECK(irnew >= 0);
            CHECK(ustep > 0.0 || ustep == 0.0);

            /* Halfway along the step the particle must still be in the
             region it started in: a step truncated too early would show up
             here, and so would one truncated in the wrong place. */
            double mx = x + 0.5*ustep*u;
            double my = y + 0.5*ustep*v;
            double mz = z + 0.5*ustep*w;

            if (ustep > 1e-6) {
                CHECK(regionIndex(mx, my, mz) == irl);
                resetGeomCache();
            }

            /* And just past the end of it, in the region howfar() named.
             The nudge has to be small against the step, or a thin ring
             would be stepped over. */
            double eps = 1e-9;
            x += (ustep + eps)*u;
            y += (ustep + eps)*v;
            z += (ustep + eps)*w;

            if (ustep > 1e-6) {
                int actual = regionIndex(x, y, z);
                resetGeomCache();
                CHECK(actual == irnew);
                if (actual != irnew) {
                    printf("\n    ray %d step %d: howfar said %d, point is "
                           "in %d (r = %.17g, z = %.17g)\n",
                           iray, nstep, irnew, actual, sqrt(x*x + y*y), z);
                    return;         /* one report is enough */
                }
            }

            if (irnew == 0) {
                break;
            }
        }
    }
}

static void test_source_place_carries_a_particle_into_the_cylinder(void) {

    setUpCylinder();

    struct OmcSourceParticle particle;
    memset(&particle, 0, sizeof(particle));
    particle.charge = 0;
    particle.energy = 6.0;
    particle.weight = 1.0;

    /* Straight down the axis from a centimetre above the front face */
    particle.x = 0.0; particle.y = 0.0; particle.z = -1.0;
    particle.u = 0.0; particle.v = 0.0; particle.w = 1.0;

    CHECK(omcSourcePlace(&particle) == 1);
    CHECK(stack.p[0].ir == 1);
    CHECK_CLOSE(stack.p[0].z, 0.0, 1e-9);
    CHECK(stack.p[0].z >= 0.0);                 /* inside, not on the face */
    CHECK_CLOSE(stack.p[0].e, 6.0, 1e-12);

    /* Aimed away from it: the ray never reaches the cylinder */
    particle.w = -1.0;
    CHECK(omcSourcePlace(&particle) == 0);

    /* Parallel to the axis but outside the barrel */
    particle.x = 6.0; particle.z = -1.0; particle.w = 1.0;
    CHECK(omcSourcePlace(&particle) == 0);

    /* In through the barrel, halfway down */
    particle.x = 8.0; particle.y = 0.0; particle.z = 5.5;
    particle.u = -1.0; particle.v = 0.0; particle.w = 0.0;
    CHECK(omcSourcePlace(&particle) == 1);
    CHECK(stack.p[0].ir == 1 + (NR - 1) + 5*NR);
    CHECK(stack.p[0].x <= 5.0);

    /* Already inside: it starts where it is */
    particle.x = 1.5; particle.y = 0.0; particle.z = 3.5;
    particle.u = 0.0; particle.v = 0.0; particle.w = 1.0;
    CHECK(omcSourcePlace(&particle) == 1);
    CHECK(stack.p[0].ir == 1 + 1 + 3*NR);
    CHECK_CLOSE(stack.p[0].z, 3.5, 1e-12);

    /* A charged particle picks up its rest mass on the way in, the same as
     in a rectilinear phantom */
    particle.charge = -1;
    CHECK(omcSourcePlace(&particle) == 1);
    CHECK_CLOSE(stack.p[0].e, 6.0 + RM, 1e-12);
}

/*******************************************************************************
* The rectilinear geometry, unchanged
*
* The whole point of dispatching on a mode rather than linking one geometry or
* the other is that omc_dosxyz and the two hosts keep the geometry they had.
* This is the cheap check that the branch did not disturb it; the expensive
* one is the smoke test's dose file, which has to stay byte for byte what it
* was.
*******************************************************************************/

#define CX 4
#define CY 3
#define CZ 2

static double cxb[CX + 1], cyb[CY + 1], czb[CZ + 1];

static void test_the_rectilinear_geometry_is_untouched(void) {

    for (int i = 0; i <= CX; i++) cxb[i] = -2.0 + 1.0*i;
    for (int j = 0; j <= CY; j++) cyb[j] = -1.5 + 1.0*j;
    for (int k = 0; k <= CZ; k++) czb[k] = 0.0 + 0.5*k;

    memset(&geometry, 0, sizeof(geometry));

    geometry.isize = CX;
    geometry.jsize = CY;
    geometry.ksize = CZ;
    geometry.xbounds = cxb;
    geometry.ybounds = cyb;
    geometry.zbounds = czb;

    resetGeomCache();
    omcGeomDetectSpacing();

    /* A loader that never heard of the cylinder still gets the rectilinear
     geometry, because detecting the spacing is what every one of them
     already does. */
    CHECK(geometry.mode == OMC_GEOM_CARTESIAN);

    CHECK(regionIndex(-1.5, -1.0, 0.25) == 1);
    CHECK(regionIndex(0.5, 0.0, 0.75) == 1 + 2 + 1*CX + 1*CX*CY);
    CHECK(regionIndex(-2.5, 0.0, 0.25) == 0);
    CHECK(regionIndex(0.0, 0.0, 1.25) == 0);

    place(-1.5, -1.0, 0.25, 1.0, 0.0, 0.0);
    CHECK_CLOSE(hownear(), 0.25, 1e-12);

    double ustep;
    int irnew, idisc;
    step(&ustep, &irnew, &idisc);
    CHECK_CLOSE(ustep, 0.5, 1e-12);
    CHECK(irnew == 2);

    geometry.xbounds = NULL;
    geometry.ybounds = NULL;
    geometry.zbounds = NULL;
}

/*******************************************************************************/

int main(void) {

    setvbuf(stdout, NULL, _IONBF, 0);

    /* howfar() and hownear() read the particle stack, so there has to be
     one, even though nothing here transports. */
    initStack();

    printf("test_omc_geom_cyl\n");

    RUN(test_region_index_maps_rings_and_slabs);
    RUN(test_cyl_init_refuses_a_geometry_it_cannot_transport);
    RUN(test_cyl_init_takes_the_azimuthal_index_out_of_play);
    RUN(test_howfar_outward_along_a_diameter);
    RUN(test_howfar_inward_and_the_chord_that_misses);
    RUN(test_howfar_along_the_axis_crosses_no_ring);
    RUN(test_howfar_oblique_agrees_with_the_quadratic);
    RUN(test_hownear_is_the_smallest_of_four);
    RUN(test_howfar_and_region_index_agree);
    RUN(test_source_place_carries_a_particle_into_the_cylinder);
    RUN(test_the_rectilinear_geometry_is_untouched);

    cleanStack();

    printf("\n%d tests, %d failures\n", tests_run, tests_failed);

    return tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

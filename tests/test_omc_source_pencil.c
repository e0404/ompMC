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

/* A parallel pencil blurred in position, in angle, or in both. */
static struct OmcPencilSource pencilGaussian(double spot, double divergence) {

    struct OmcPencilSource pencil;
    memset(&pencil, 0, sizeof(pencil));

    pencil.kind = OMC_PENCIL_PARALLEL;
    pencil.spectrum = &mono;
    pencil.charge = 0;
    pencil.spotSigma = spot;
    pencil.divergenceSigma = divergence;

    return pencil;
}

/* Where a particle crosses the front face of the cylinder, which is the plane
 the beam is actually specified on. */
static void atEntrance(const struct OmcSourceParticle *particle,
                       double *x, double *y) {

    double t = (geometry.zbounds[0] - particle->z)/particle->w;

    *x = particle->x + t*particle->u;
    *y = particle->y + t*particle->v;
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

static void test_a_parallel_pencil_starts_on_the_axis_at_the_face(void) {

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

    /* On the front face, which is where a beam with no source point is
     defined: it has nowhere upstream to be. Not INSIDE the phantom, which is
     what would skip the build up region. */
    CHECK_CLOSE(particle.z, geometry.zbounds[0], 1e-15);

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

/*******************************************************************************
* The Gaussian blurs
*
* A real pencil beam is neither a point nor perfectly parallel. Either delta
* can be widened into a Gaussian, and the two are independent: a spot size
* without divergence, a divergence without spot size, or both.
*******************************************************************************/

/* Sample a lot of histories and report the moments of whatever the callback
 pulls out of each particle. */
static void moments(const struct OmcSource *source,
                    void (*pick)(const struct OmcSourceParticle *,
                                 double *, double *),
                    int n, double *meanA, double *meanB,
                    double *sdA, double *sdB, double *covAB) {

    double sa = 0.0, sb = 0.0, saa = 0.0, sbb = 0.0, sab = 0.0;

    for (int i = 0; i < n; i++) {
        struct OmcSourceParticle particle;
        memset(&particle, 0, sizeof(particle));

        setRandomHistory((uint64_t)i);
        source->sample(source, (uint64_t)i, i, &particle);

        double a, b;
        pick(&particle, &a, &b);

        sa += a;  sb += b;
        saa += a*a;  sbb += b*b;  sab += a*b;
    }

    double na = (double)n;

    *meanA = sa/na;
    *meanB = sb/na;
    *sdA = sqrt(saa/na - (*meanA)*(*meanA));
    *sdB = sqrt(sbb/na - (*meanB)*(*meanB));
    *covAB = sab/na - (*meanA)*(*meanB);
}

static void pickEntrance(const struct OmcSourceParticle *particle,
                         double *x, double *y) {
    atEntrance(particle, x, y);
}

/* The projected angles. The direction is built as (a, b, 1) normalized, so
 u/w and v/w recover exactly what was sampled. */
static void pickAngles(const struct OmcSourceParticle *particle,
                       double *a, double *b) {
    *a = particle->u/particle->w;
    *b = particle->v/particle->w;
}

#define NSAMPLES 40000

static void test_a_spot_sigma_widens_the_beam_where_it_enters(void) {

    setUpCylinder();

    double sigma = 0.3;
    struct OmcPencilSource pencil = pencilGaussian(sigma, 0.0);
    struct OmcSource source;
    omcPencilSourceAsSource(&pencil, &source);

    double mx, my, sx, sy, cov;
    moments(&source, pickEntrance, NSAMPLES, &mx, &my, &sx, &sy, &cov);

    /* Centred on the axis */
    CHECK(fabs(mx) < 0.06*sigma);
    CHECK(fabs(my) < 0.06*sigma);

    /* As wide as it was asked to be */
    CHECK_CLOSE(sx, sigma, 0.05*sigma);
    CHECK_CLOSE(sy, sigma, 0.05*sigma);

    /* Round, not elliptical or tilted. This matters more here than it would
     in a voxel phantom: the rings have no azimuthal binning, so a source
     that was wider in x than in y would be averaged away silently rather
     than showing up in the result. */
    CHECK(fabs(cov) < 0.06*sigma*sigma);

    /* Still parallel: the spot is a position blur and nothing else */
    for (uint64_t ihist = 0; ihist < 32; ihist++) {
        struct OmcSourceParticle particle;
        memset(&particle, 0, sizeof(particle));

        setRandomHistory(ihist);
        source.sample(&source, ihist, 0, &particle);

        CHECK_CLOSE(particle.u, 0.0, 1e-15);
        CHECK_CLOSE(particle.v, 0.0, 1e-15);
        CHECK_CLOSE(particle.w, 1.0, 1e-15);
    }
}

/* Why the waist arithmetic is a parallel-pencil thing, pinned as the fact it
 rests on: a point source's focal spot does not move its beam on the front
 face at all.

 The particle leaves the spot aimed at a point on the illuminated disc, so it
 arrives at that point whatever the spot did to where it set off -- start at
 spot, travel the whole SSD along (aim - spot)/|..| and the spot cancels
 exactly. What the spot blurs for a point source is the DIRECTION, and only
 through the direction the width further downstream.

 The aim is drawn before the spot is, so the two beams below see the same aim
 point for the same history and the entrance positions can be compared one for
 one rather than as distributions. */
static void test_a_focal_spot_does_not_move_a_point_source_on_the_face(void) {

    setUpCylinder();

    struct OmcPencilSource sharp = pencilSsd(100.0, 3.0);
    struct OmcPencilSource blurred = pencilSsd(100.0, 3.0);
    blurred.spotSigma = 0.8;            /* a huge focal spot, on purpose */

    struct OmcSource a, b;
    omcPencilSourceAsSource(&sharp, &a);
    omcPencilSourceAsSource(&blurred, &b);

    for (uint64_t ihist = 0; ihist < 64; ihist++) {
        struct OmcSourceParticle pa, pb;
        memset(&pa, 0, sizeof(pa));
        memset(&pb, 0, sizeof(pb));

        setRandomHistory(ihist);
        a.sample(&a, ihist, 0, &pa);

        setRandomHistory(ihist);
        b.sample(&b, ihist, 0, &pb);

        /* They start a long way apart */
        double startX = fabs(pa.x - pb.x) + fabs(pa.y - pb.y);

        double xa, ya, xb, yb;
        atEntrance(&pa, &xa, &ya);
        atEntrance(&pb, &xb, &yb);

        /* and arrive at the same place regardless */
        CHECK_CLOSE(xa, xb, 1e-12);
        CHECK_CLOSE(ya, yb, 1e-12);

        if (ihist == 0) {
            CHECK(startX > 0.0);
        }
    }
}

static void test_a_divergence_sigma_spreads_the_direction(void) {

    setUpCylinder();

    double sigma = 0.02;
    struct OmcPencilSource pencil = pencilGaussian(0.0, sigma);
    struct OmcSource source;
    omcPencilSourceAsSource(&pencil, &source);

    double ma, mb, sa, sb, cov;
    moments(&source, pickAngles, NSAMPLES, &ma, &mb, &sa, &sb, &cov);

    CHECK(fabs(ma) < 0.06*sigma);
    CHECK(fabs(mb) < 0.06*sigma);
    CHECK_CLOSE(sa, sigma, 0.05*sigma);
    CHECK_CLOSE(sb, sigma, 0.05*sigma);
    CHECK(fabs(cov) < 0.06*sigma*sigma);

    /* Every direction is still a unit vector, which the transport takes as
     given and never renormalizes */
    for (uint64_t ihist = 0; ihist < 64; ihist++) {
        struct OmcSourceParticle particle;
        memset(&particle, 0, sizeof(particle));

        setRandomHistory(ihist);
        source.sample(&source, ihist, 0, &particle);

        double norm = sqrt(particle.u*particle.u + particle.v*particle.v +
                           particle.w*particle.w);
        CHECK_CLOSE(norm, 1.0, 1e-12);
        CHECK(particle.w > 0.0);
    }
}

/* Divergence must tilt the beam without moving it. A parallel pencil starts
 on the front face, so there is no distance over which a diverging particle
 could drift sideways -- but that is a property of where it starts, and this
 is what would catch it being emitted from somewhere upstream again without
 the position being corrected for the tilt. */
static void test_divergence_does_not_displace_the_beam(void) {

    setUpCylinder();

    struct OmcPencilSource pencil = pencilGaussian(0.0, 0.05);
    struct OmcSource source;
    omcPencilSourceAsSource(&pencil, &source);

    for (uint64_t ihist = 0; ihist < 500; ihist++) {
        struct OmcSourceParticle particle;
        memset(&particle, 0, sizeof(particle));

        setRandomHistory(ihist);
        source.sample(&source, ihist, 0, &particle);

        /* No spot size, so every particle has to cross the front face
         exactly on the axis however much it diverges */
        double x, y;
        atEntrance(&particle, &x, &y);

        CHECK_CLOSE(x, 0.0, 1e-12);
        CHECK_CLOSE(y, 0.0, 1e-12);
    }
}

/* Both at once, still independent of each other. */
static void test_spot_and_divergence_compose(void) {

    setUpCylinder();

    double spot = 0.2, divergence = 0.03;
    struct OmcPencilSource pencil = pencilGaussian(spot, divergence);
    struct OmcSource source;
    omcPencilSourceAsSource(&pencil, &source);

    double mx, my, sx, sy, cov;
    moments(&source, pickEntrance, NSAMPLES, &mx, &my, &sx, &sy, &cov);

    /* The spot alone decides the width at the face; the divergence adds
     nothing there, only downstream. */
    CHECK_CLOSE(sx, spot, 0.05*spot);
    CHECK_CLOSE(sy, spot, 0.05*spot);

    double ma, mb, sa, sb, covAngle;
    moments(&source, pickAngles, NSAMPLES, &ma, &mb, &sa, &sb, &covAngle);

    CHECK_CLOSE(sa, divergence, 0.05*divergence);
    CHECK_CLOSE(sb, divergence, 0.05*divergence);

    /* And they are drawn independently, so position and angle are
     uncorrelated -- this is a blurred pencil, not a beam with emittance. */
    double sxa = 0.0;
    for (int i = 0; i < NSAMPLES; i++) {
        struct OmcSourceParticle particle;
        memset(&particle, 0, sizeof(particle));

        setRandomHistory((uint64_t)i);
        source.sample(&source, (uint64_t)i, i, &particle);

        double x, y, a, b;
        atEntrance(&particle, &x, &y);
        pickAngles(&particle, &a, &b);

        sxa += x*a;
    }
    CHECK(fabs(sxa/NSAMPLES) < 0.06*spot*divergence);
}

/*******************************************************************************
* Correlation, i.e. where the waist is
*
* Position and angle drawn independently put the narrowest part of the beam at
* the phantom surface, which is only one of the beams a machine can make. A
* correlation between them moves it: the width at distance s downstream is
*
*     var(s) = sigma^2 + 2 s rho sigma sigma' + s^2 sigma'^2
*
* whose minimum sits at s = -rho sigma / sigma'. So rho < 0 converges onto a
* waist inside the phantom, rho > 0 has already passed it.
*******************************************************************************/

/* Everything second order about the beam where it enters the phantom. */
struct PhaseSpace {
    double varX, varY;          /* position */
    double varAx, varAy;        /* angle */
    double covXAx, covYAy;      /* position with its own angle */
    double covXAy;              /* position with the other one */
};

static void gather(const struct OmcSource *source, int n,
                   struct PhaseSpace *out) {

    double mx = 0, my = 0, max_ = 0, may = 0;
    double xx = 0, yy = 0, aa = 0, bb = 0, xa = 0, yb = 0, xb = 0;

    for (int i = 0; i < n; i++) {
        struct OmcSourceParticle particle;
        memset(&particle, 0, sizeof(particle));

        setRandomHistory((uint64_t)i);
        source->sample(source, (uint64_t)i, i, &particle);

        double x, y;
        atEntrance(&particle, &x, &y);

        double ax = particle.u/particle.w;
        double ay = particle.v/particle.w;

        mx += x; my += y; max_ += ax; may += ay;
        xx += x*x; yy += y*y; aa += ax*ax; bb += ay*ay;
        xa += x*ax; yb += y*ay; xb += x*ay;
    }

    double d = (double)n;
    mx /= d; my /= d; max_ /= d; may /= d;

    out->varX = xx/d - mx*mx;
    out->varY = yy/d - my*my;
    out->varAx = aa/d - max_*max_;
    out->varAy = bb/d - may*may;
    out->covXAx = xa/d - mx*max_;
    out->covYAy = yb/d - my*may;
    out->covXAy = xb/d - mx*may;
}

static void test_a_correlation_tilts_the_phase_space(void) {

    setUpCylinder();

    double sigma = 0.5, sigmaPrime = 0.06, rho = -0.6;

    struct OmcPencilSource pencil = pencilGaussian(sigma, sigmaPrime);
    pencil.correlation = rho;

    struct OmcSource source;
    omcPencilSourceAsSource(&pencil, &source);

    struct PhaseSpace ps;
    gather(&source, NSAMPLES, &ps);

    /* The widths are the ones asked for, correlation or not */
    CHECK_CLOSE(sqrt(ps.varX), sigma, 0.05*sigma);
    CHECK_CLOSE(sqrt(ps.varAx), sigmaPrime, 0.05*sigmaPrime);

    /* And the correlation is the one asked for, in both planes */
    double rx = ps.covXAx/sqrt(ps.varX*ps.varAx);
    double ry = ps.covYAy/sqrt(ps.varY*ps.varAy);

    CHECK_CLOSE(rx, rho, 0.03);
    CHECK_CLOSE(ry, rho, 0.03);

    /* Each plane is correlated with its OWN angle and no other. A beam that
     mixed them would not be round any more. */
    double cross = ps.covXAy/sqrt(ps.varX*ps.varAy);
    CHECK(fabs(cross) < 0.03);

    CHECK_CLOSE(sqrt(ps.varY), sigma, 0.05*sigma);
    CHECK_CLOSE(sqrt(ps.varAy), sigmaPrime, 0.05*sigmaPrime);
}

/* The point of the correlation: the beam is narrowest somewhere other than
 where it started. */
static void test_a_converging_beam_is_narrowest_at_its_waist(void) {

    setUpCylinder();

    double sigma = 0.5, sigmaPrime = 0.06, rho = -0.6;
    double expectedWaist = -rho*sigma/sigmaPrime;      /* 5 cm */

    struct OmcPencilSource pencil = pencilGaussian(sigma, sigmaPrime);
    pencil.correlation = rho;

    struct OmcSource source;
    omcPencilSourceAsSource(&pencil, &source);

    struct PhaseSpace ps;
    gather(&source, NSAMPLES, &ps);

    /* var(s) = varX + 2 s covXAx + s^2 varAx, minimized where its derivative
     vanishes. Read off the sampled beam rather than assumed. */
    double waist = -ps.covXAx/ps.varAx;
    CHECK_CLOSE(waist, expectedWaist, 0.05*expectedWaist);

    /* And it really is narrower there than at either end */
    double atFace = ps.varX;
    double atWaist = ps.varX + 2.0*waist*ps.covXAx + waist*waist*ps.varAx;
    double beyond = ps.varX + 2.0*(2.0*waist)*ps.covXAx
                    + 4.0*waist*waist*ps.varAx;

    CHECK(atWaist < atFace);
    CHECK(atWaist < beyond);

    /* A waist of zero width is not what this is: the beam still has the
     emittance it started with. */
    CHECK(atWaist > 0.0);
    CHECK_CLOSE(sqrt(atWaist), sigma*sqrt(1.0 - rho*rho), 0.05*sigma);
}

/* A correlation of zero has to leave the uncorrelated beam exactly as it was,
 down to which random number went where. */
static void test_zero_correlation_changes_nothing(void) {

    setUpCylinder();

    struct OmcPencilSource plain = pencilGaussian(0.3, 0.02);
    struct OmcPencilSource zeroed = pencilGaussian(0.3, 0.02);
    zeroed.correlation = 0.0;

    struct OmcSource a, b;
    omcPencilSourceAsSource(&plain, &a);
    omcPencilSourceAsSource(&zeroed, &b);

    for (uint64_t ihist = 0; ihist < 64; ihist++) {
        struct OmcSourceParticle first, again;
        memset(&first, 0, sizeof(first));
        memset(&again, 0, sizeof(again));

        setRandomHistory(ihist);
        a.sample(&a, ihist, 0, &first);

        setRandomHistory(ihist);
        b.sample(&b, ihist, 0, &again);

        CHECK(memcmp(&first, &again, sizeof(first)) == 0);
    }
}

/* Correlating with a width that is not there means nothing, and must not cost
 a random number either. */
static void test_correlation_without_both_widths_is_ignored(void) {

    setUpCylinder();

    struct OmcSourceParticle particle;

    struct {
        double spot, divergence;
        int draws;
    } cases[] = {
        { 0.0, 0.0, 0 },
        { 0.3, 0.0, 2 },
        { 0.0, 0.02, 2 },
    };

    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); i++) {

        struct OmcPencilSource pencil =
            pencilGaussian(cases[i].spot, cases[i].divergence);
        pencil.correlation = -0.9;

        struct OmcSource source;
        omcPencilSourceAsSource(&pencil, &source);

        setRandomHistory(31);
        for (int d = 0; d < cases[i].draws; d++) {
            setRandom();
        }
        double expected = setRandom();

        setRandomHistory(31);
        memset(&particle, 0, sizeof(particle));
        source.sample(&source, 31, 0, &particle);

        CHECK(setRandom() == expected);
    }
}

/* A sigma of zero has to be the delta beam exactly -- same particle, and the
 same number of random draws -- so that adding these knobs changes no result
 that never asked for them. */
static void test_zero_sigma_is_the_delta_beam_exactly(void) {

    setUpCylinder();

    struct OmcPencilSource plain = pencilParallel();
    struct OmcPencilSource zeroed = pencilGaussian(0.0, 0.0);

    struct OmcSource a, b;
    omcPencilSourceAsSource(&plain, &a);
    omcPencilSourceAsSource(&zeroed, &b);

    for (uint64_t ihist = 0; ihist < 32; ihist++) {
        struct OmcSourceParticle first, again;
        memset(&first, 0, sizeof(first));
        memset(&again, 0, sizeof(again));

        setRandomHistory(ihist);
        a.sample(&a, ihist, 0, &first);

        setRandomHistory(ihist);
        b.sample(&b, ihist, 0, &again);

        CHECK(memcmp(&first, &again, sizeof(first)) == 0);
    }

    /* And neither of them touched the generator */
    setRandomHistory(5);
    double straightAway = setRandom();

    struct OmcSourceParticle particle;
    setRandomHistory(5);
    memset(&particle, 0, sizeof(particle));
    b.sample(&b, 5, 0, &particle);
    CHECK(setRandom() == straightAway);
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

    /* Each Gaussian blur is one Box-Muller pair, and only when it is asked
     for. The counts below are the source's contract, not an accident of how
     it happens to be written today. */
    struct {
        struct OmcPencilSource pencil;
        int draws;
    } cases[] = {
        { pencilGaussian(0.0, 0.0), 0 },
        { pencilGaussian(0.5, 0.0), 2 },
        { pencilGaussian(0.0, 0.01), 2 },
        { pencilGaussian(0.5, 0.01), 4 },
    };

    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); i++) {

        struct OmcSource source;
        omcPencilSourceAsSource(&cases[i].pencil, &source);

        setRandomHistory(23);
        for (int d = 0; d < cases[i].draws; d++) {
            setRandom();
        }
        double expected = setRandom();

        setRandomHistory(23);
        memset(&particle, 0, sizeof(particle));
        source.sample(&source, 23, 0, &particle);
        double afterSampling = setRandom();

        CHECK(afterSampling == expected);
    }

    /* And they stack on top of the SSD source's own two */
    {
        struct OmcPencilSource pencil = pencilSsd(90.0, 1.0);
        pencil.spotSigma = 0.1;
        pencil.divergenceSigma = 0.01;

        struct OmcSource source;
        omcPencilSourceAsSource(&pencil, &source);

        setRandomHistory(23);
        for (int d = 0; d < 6; d++) {
            setRandom();
        }
        double expected = setRandom();

        setRandomHistory(23);
        memset(&particle, 0, sizeof(particle));
        source.sample(&source, 23, 0, &particle);

        CHECK(setRandom() == expected);
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

    /* A negative width is a typo, not a beam. Zero is not: it is the delta
     the blur widens from. */
    {
        struct OmcPencilSource pencil = pencilGaussian(-0.1, 0.0);
        omcPencilSourceAsSource(&pencil, &source);
        EXPECT_FAIL("ompMC:pencil:badSpotSigma", source.check(&source));
    }

    {
        struct OmcPencilSource pencil = pencilGaussian(0.0, -0.01);
        omcPencilSourceAsSource(&pencil, &source);
        EXPECT_FAIL("ompMC:pencil:badDivergenceSigma", source.check(&source));
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

    RUN(test_a_parallel_pencil_starts_on_the_axis_at_the_face);
    RUN(test_the_charge_is_the_one_it_was_given);
    RUN(test_an_ssd_source_fills_the_field_it_was_given);
    RUN(test_a_field_radius_of_zero_means_the_whole_face);
    RUN(test_a_spot_sigma_widens_the_beam_where_it_enters);
    RUN(test_a_focal_spot_does_not_move_a_point_source_on_the_face);
    RUN(test_a_divergence_sigma_spreads_the_direction);
    RUN(test_divergence_does_not_displace_the_beam);
    RUN(test_spot_and_divergence_compose);
    RUN(test_a_correlation_tilts_the_phase_space);
    RUN(test_a_converging_beam_is_narrowest_at_its_waist);
    RUN(test_zero_correlation_changes_nothing);
    RUN(test_correlation_without_both_widths_is_ignored);
    RUN(test_zero_sigma_is_the_delta_beam_exactly);
    RUN(test_sampling_depends_only_on_the_history);
    RUN(test_the_draw_count_is_what_it_says);
    RUN(test_prepare_reports_the_dose_per_history);
    RUN(test_check_refuses_what_it_cannot_sample);

    omcSpectrumFree(&mono);
    cleanRandom();

    printf("\n%d tests, %d failures\n", tests_run, tests_failed);

    return tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

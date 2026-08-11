/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations

 Copyright (C) 2018 Edgardo Doerner (edoerner@fis.puc.cl)


 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
*****************************************************************************/

#include "omc_source_phsp.h"

#include "omc_geom.h"
#include "omc_host.h"
#include "omc_phsp.h"
#include "omc_random.h"
#include "omc_score.h"
#include "omc_utilities.h"
#include "ompmc.h"

#include <float.h>
#include <math.h>

#if defined(_MSC_VER)
    //use __declspec(thread) instead of threadprivate to avoid
    //error C3053. More information in:
    // https://stackoverflow.com/questions/12560243/using-threadprivate-directive-in-visual-studio
    __declspec(thread) extern struct Stack stack;
#else
    extern struct Stack stack;
    #pragma omp threadprivate(stack)
#endif

/******************************************************************************/

void omcPhspTransformIdentity(struct OmcPhspTransform *transform) {

    for (int i = 0; i < 9; i++) {
        transform->rotation[i] = (i % 4) == 0 ? 1.0 : 0.0;
    }
    for (int i = 0; i < 3; i++) {
        transform->translation[i] = 0.0;
    }

    return;
}

void omcPhspSourceCheck(const struct OmcPhspSampler *sampler) {

    if (sampler->phsp == NULL || omcPhspCount(sampler->phsp) == 0) {
        omcFail("ompMC:phspSource:noParticles",
            "The phase space source has no particles to draw from.");
    }

    if (sampler->order != OMC_PHSP_REPLAY &&
        sampler->order != OMC_PHSP_RANDOM) {
        omcFail("ompMC:phspSource:badOrder",
            "The phase space source was given order %d, and knows %d for "
            "replaying the file and %d for drawing from it at random.",
            (int)sampler->order, (int)OMC_PHSP_REPLAY, (int)OMC_PHSP_RANDOM);
    }

    if (geometry.xbounds == NULL || geometry.ybounds == NULL ||
        geometry.zbounds == NULL) {
        omcFail("ompMC:phspSource:noGeometry",
            "The phase space source needs the phantom set up before it can "
            "work out where its particles enter one.");
    }

    /* The determinant says whether the rotation is one. A caller who meant
     to turn the phase space by 30 degrees and mistyped a matrix element
     would otherwise find out from the dose distribution. */
    const double *r = sampler->transform.rotation;
    double det = r[0]*(r[4]*r[8] - r[5]*r[7])
               - r[1]*(r[3]*r[8] - r[5]*r[6])
               + r[2]*(r[3]*r[7] - r[4]*r[6]);

    if (fabs(det - 1.0) > 1.0e-6) {
        omcFail("ompMC:phspSource:notARotation",
            "The phase space to phantom rotation has determinant %g, and a "
            "rotation has 1. A matrix that is not one would stretch the "
            "directions it turns, and they have to stay unit vectors.",
            det);
    }

    if (sampler->phsp->newHistories == 0) {
        omcLog(OMC_LOG_WARNING, "The phase space marks no histories, so every "
               "particle is drawn as a history of its own. The dose is right; "
               "the uncertainty this run reports for it will be smaller than "
               "the truth by however much the particles of one original "
               "history are correlated.");
    }

    omcLog(OMC_LOG_INFO, "Phase space source: %llu particles, %s.",
           omcPhspCount(sampler->phsp),
           sampler->order == OMC_PHSP_REPLAY ? "replayed in order" :
           "drawn at random");

    return;
}

/* Turn and move one point of the phase space into the phantom's world. */
static void applyTransform(const struct OmcPhspTransform *transform,
                           double *x, double *y, double *z, int isDirection) {

    const double *r = transform->rotation;
    double px = *x, py = *y, pz = *z;

    *x = r[0]*px + r[1]*py + r[2]*pz;
    *y = r[3]*px + r[4]*py + r[5]*pz;
    *z = r[6]*px + r[7]*py + r[8]*pz;

    /* A direction is turned but not moved. */
    if (!isDirection) {
        *x += transform->translation[0];
        *y += transform->translation[1];
        *z += transform->translation[2];
    }

    return;
}

/* Cut the ray down to where it is inside one slab of the phantom's bounding
 box, narrowing the interval [tenter, texit] it is inside all of them.

 @return 0 as soon as the interval is empty, i.e. the ray misses. */
static int clipSlab(double p, double d, double lo, double hi,
                    double *tenter, double *texit) {

    if (d == 0.0) {
        /* Parallel to the slab: either it starts inside and stays, or it is
         never in it. Written out rather than left to divide by zero, whose
         infinities are fine but whose 0/0 is not. */
        return p >= lo && p <= hi;
    }

    double t1 = (lo - p)/d;
    double t2 = (hi - p)/d;

    if (t1 > t2) {
        double swap = t1;
        t1 = t2;
        t2 = swap;
    }

    if (t1 > *tenter) {
        *tenter = t1;
    }
    if (t2 < *texit) {
        *texit = t2;
    }

    return *tenter <= *texit;
}

/* Carry a particle to where it enters the phantom.

 Unlike a beamlet particle, which is aimed at the phantom by construction, a
 phase space particle was recorded wherever the simulation that made it
 scored one, and may be pointing anywhere at all.

 @return 1 with the position moved to the entry point, 0 if the ray never
 reaches the phantom. */
static int enterPhantom(double *x, double *y, double *z,
                        double u, double v, double w) {

    double tenter = 0.0;        /* already inside enters at once */
    double texit = DBL_MAX;

    if (!clipSlab(*x, u, geometry.xbounds[0],
                  geometry.xbounds[geometry.isize], &tenter, &texit) ||
        !clipSlab(*y, v, geometry.ybounds[0],
                  geometry.ybounds[geometry.jsize], &tenter, &texit) ||
        !clipSlab(*z, w, geometry.zbounds[0],
                  geometry.zbounds[geometry.ksize], &tenter, &texit)) {
        return 0;
    }

    /* Behind the particle rather than in front of it: it is heading away. */
    if (texit < 0.0) {
        return 0;
    }

    *x += tenter*u;
    *y += tenter*v;
    *z += tenter*w;

    /* For numerical stability, make sure the point really is inside the
     phantom. nextafter() moves one representable step towards the opposite
     face, which is the same guard omc_source_beamlet.c puts on the particles
     it starts. */
    if (*x < geometry.xbounds[0]) {
        *x = nextafter(geometry.xbounds[0], geometry.xbounds[geometry.isize]);
    }
    if (*x > geometry.xbounds[geometry.isize]) {
        *x = nextafter(geometry.xbounds[geometry.isize], geometry.xbounds[0]);
    }

    if (*y < geometry.ybounds[0]) {
        *y = nextafter(geometry.ybounds[0], geometry.ybounds[geometry.jsize]);
    }
    if (*y > geometry.ybounds[geometry.jsize]) {
        *y = nextafter(geometry.ybounds[geometry.jsize], geometry.ybounds[0]);
    }

    if (*z < geometry.zbounds[0]) {
        *z = nextafter(geometry.zbounds[0], geometry.zbounds[geometry.ksize]);
    }
    if (*z > geometry.zbounds[geometry.ksize]) {
        *z = nextafter(geometry.zbounds[geometry.ksize], geometry.zbounds[0]);
    }

    return 1;
}

/* Which particle of the file this history gets.

 @warning Depends on ihist and nothing else, which is what keeps a run from
 depending on how its histories were scheduled. Note what it does NOT do:
 read the next particle. That read position belongs to whoever is stepping
 through the file serially, and there is one of it for all the threads. */
static unsigned long long recordFor(const struct OmcPhspSampler *sampler,
                                    uint64_t ihist) {

    unsigned long long count = omcPhspCount(sampler->phsp);

    if (sampler->order == OMC_PHSP_RANDOM) {
        /* setRandom() is in (0,1), so this is in range; the clamp is for the
         rounding at the very top of it rather than for the mathematics. */
        unsigned long long index = (unsigned long long)(setRandom()*count);
        return index < count ? index : count - 1;
    }

    return (sampler->first + (unsigned long long)ihist) % count;
}

int omcPhspSourceSample(const struct OmcPhspSampler *sampler, uint64_t ihist,
                        double weight) {

    struct OmcPhspRecord particle;
    int charge;

    /* omcPhspSourceCheck() turns an empty phase space away before any of
     this runs, so this is only a backstop for a caller who skipped it: the
     wrap in recordFor() would divide by zero, and omcFail() from a worker
     thread would call the host from a place the host cannot expect. */
    if (omcPhspCount(sampler->phsp) == 0) {
        return 0;
    }

    omcPhspGet(sampler->phsp, recordFor(sampler, ihist), &particle);

    switch (particle.type) {
        case OMC_PHSP_PHOTON:
            charge = 0;
            break;
        case OMC_PHSP_ELECTRON:
            charge = -1;
            break;
        case OMC_PHSP_POSITRON:
            charge = 1;
            break;
        default:
            /* A neutron or a proton. ompMC transports neither, and a phase
             space holding them is not wrong for it -- this history simply
             has nothing in it. */
            return 0;
    }

    double x = particle.x, y = particle.y, z = particle.z;
    double u = particle.u, v = particle.v, w = particle.w;

    applyTransform(&sampler->transform, &x, &y, &z, 0);
    applyTransform(&sampler->transform, &u, &v, &w, 1);

    if (!enterPhantom(&x, &y, &z, u, v, w)) {
        return 0;
    }

    double ein = particle.energy;
    double wt = particle.weight*weight;

    stack.np = 0;
    stack.p[stack.np].iq = charge;
    stack.p[stack.np].e = charge != 0 ? ein + RM : ein;

    stack.p[stack.np].x = x;
    stack.p[stack.np].y = y;
    stack.p[stack.np].z = z;

    stack.p[stack.np].u = u;
    stack.p[stack.np].v = v;
    stack.p[stack.np].w = w;

    stack.p[stack.np].wt = wt;
    stack.p[stack.np].dnear = 0.0;

    int ix = omcFindVoxelIndex(geometry.xbounds, geometry.isize, x);
    int iy = omcFindVoxelIndex(geometry.ybounds, geometry.jsize, y);
    int iz = omcFindVoxelIndex(geometry.zbounds, geometry.ksize, z);

    stack.p[stack.np].ir = 1 + ix + iy*geometry.isize
                             + iz*geometry.isize*geometry.jsize;

    /* Only what actually got into the phantom counts as energy put in, so
     that the fraction of it that ends up deposited means what it says. */
    scoreSource(ein*wt);

    return 1;
}

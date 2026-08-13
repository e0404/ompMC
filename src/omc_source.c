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

#include "omc_source.h"

#include "omc_geom.h"
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

/* Cut a ray down to where it is inside one slab of the phantom's bounding
 box, narrowing the interval [tenter, texit] over which it is inside all of
 them.

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

    /* Strictly less: an interval of no length is a ray that touches the
     phantom at a single point and is inside it for no distance at all, which
     is not entering it. A particle sitting exactly on a face and pointing
     out of it is the case that matters -- it would otherwise be placed on
     the boundary, counted among the histories that started, and transported
     just far enough to leave. */
    return *tenter < *texit;
}

int omcSourcePlace(const struct OmcSourceParticle *particle) {

    double x = particle->x;
    double y = particle->y;
    double z = particle->z;

    /* tenter starts at 0 and only ever grows, so a ray that clears every slab
     clears it in front of the particle: one heading away from the phantom is
     turned down by clipSlab() itself, on the slab whose exit is behind it,
     rather than needing a check of its own here. */
    double tenter = 0.0;        /* already inside enters at once */
    double texit = DBL_MAX;

    if (!clipSlab(x, particle->u, geometry.xbounds[0],
                  geometry.xbounds[geometry.isize], &tenter, &texit) ||
        !clipSlab(y, particle->v, geometry.ybounds[0],
                  geometry.ybounds[geometry.jsize], &tenter, &texit) ||
        !clipSlab(z, particle->w, geometry.zbounds[0],
                  geometry.zbounds[geometry.ksize], &tenter, &texit)) {
        return 0;
    }

    x += tenter*particle->u;
    y += tenter*particle->v;
    z += tenter*particle->w;

    /* For numerical stability, make sure the point really is inside the
     phantom. nextafter() moves one representable step towards the opposite
     face; the 2.0*DBL_MIN offset used before was denormal-small and was
     absorbed entirely when added to any normal boundary coordinate, leaving
     the particle exactly on the boundary. */
    if (x < geometry.xbounds[0]) {
        x = nextafter(geometry.xbounds[0], geometry.xbounds[geometry.isize]);
    }
    if (x > geometry.xbounds[geometry.isize]) {
        x = nextafter(geometry.xbounds[geometry.isize], geometry.xbounds[0]);
    }

    if (y < geometry.ybounds[0]) {
        y = nextafter(geometry.ybounds[0], geometry.ybounds[geometry.jsize]);
    }
    if (y > geometry.ybounds[geometry.jsize]) {
        y = nextafter(geometry.ybounds[geometry.jsize], geometry.ybounds[0]);
    }

    if (z < geometry.zbounds[0]) {
        z = nextafter(geometry.zbounds[0], geometry.zbounds[geometry.ksize]);
    }
    if (z > geometry.zbounds[geometry.ksize]) {
        z = nextafter(geometry.zbounds[geometry.ksize], geometry.zbounds[0]);
    }

    /* Initialize first particle of the stack from the source data */
    stack.np = 0;
    stack.p[stack.np].iq = particle->charge;

    /* A charged particle carries its rest mass on top of the kinetic energy
     the source spoke in. */
    stack.p[stack.np].e = particle->charge != 0 ?
        particle->energy + RM : particle->energy;

    stack.p[stack.np].x = x;
    stack.p[stack.np].y = y;
    stack.p[stack.np].z = z;

    stack.p[stack.np].u = particle->u;
    stack.p[stack.np].v = particle->v;
    stack.p[stack.np].w = particle->w;

    stack.p[stack.np].wt = particle->weight;
    stack.p[stack.np].dnear = 0.0;

    /* Determine region index of source particle */
    int ix = omcFindVoxelIndex(geometry.xbounds, geometry.isize, x);
    int iy = omcFindVoxelIndex(geometry.ybounds, geometry.jsize, y);
    int iz = omcFindVoxelIndex(geometry.zbounds, geometry.ksize, z);

    stack.p[stack.np].ir = 1 + ix + iy*geometry.isize
                             + iz*geometry.isize*geometry.jsize;

    return 1;
}

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

#include "omc_geom_cyl.h"

#include "omc_geom.h"
#include "omc_host.h"
#include "omc_utilities.h"
#include "ompmc.h"

#include <float.h>
#include <math.h>
#include <stddef.h>

#if defined(_MSC_VER)
    __declspec(thread) extern struct Stack stack;
#else
    extern struct Stack stack;
    #pragma omp threadprivate(stack)
#endif

/******************************************************************************/
/* Where a ray meets a cylinder.

 Along a ray (x,y) + t*(u,v) the squared distance from the axis is a quadratic
 in t:

     a t^2 + 2 b t + c = 0,   a = u^2 + v^2, b = xu + yv, c = x^2 + y^2 - R^2

 so the roots are (-b +- sqrt(b^2 - ac))/a -- the reduced form, b being half
 the textbook coefficient. Two things about them matter here.

 a is zero for a ray parallel to the axis, which meets no cylinder at all; the
 quadratic degenerates rather than having a large root, so that case is taken
 out before anything is divided by a.

 And one of the two roots is always the difference of two nearly equal numbers
 when |ac| is small against b^2 -- a ray that only just reaches the surface it
 is aimed at. The cancelling form is never the one used below: each root is
 written in whichever of its two algebraically identical forms adds rather
 than subtracts. */

/*! The root where the ray leaves the surface of radius @p radius, given that
 it is inside it (c <= 0), which is the case for the outer surface of the ring
 a particle is in. */
static inline double exitRoot(double a, double b, double c, double disc) {

    double sq = sqrt(disc);

    /* c <= 0 makes the roots opposite in sign, so the positive one is the
     one with +sqrt. Rationalized when b > 0, where -b + sq cancels. */
    return b <= 0.0 ? (-b + sq)/a : -c/(b + sq);
}

/*! The root where the ray first meets the surface of radius @p radius from
 outside it (c >= 0), which is the case for the inner surface of the ring a
 particle is in. Only called with b < 0, i.e. for a particle heading inwards.

 @return The distance, which cannot be negative. */
static inline double entryRoot(double b, double c, double disc) {

    /* The smaller root, (-b - sqrt(disc))/a, cancels for b < 0. This is the
     same number written so that it does not. */
    return c/(-b + sqrt(disc));
}

/*! Narrow [tenter, texit] to where the ray is inside one slab, exactly as
 clipSlab() in omc_source.c does for the rectilinear bounding box. It is a
 sibling of that one rather than a shared function because sharing it would
 mean a third translation unit for fifteen lines.

 @return 0 as soon as the interval is empty. */
static int clipSlab(double p, double d, double lo, double hi,
                    double *tenter, double *texit) {

    if (d == 0.0) {
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

    return *tenter < *texit;
}

/******************************************************************************/

void omcGeomCylInit(void) {

    if (geometry.isize < 1 || geometry.ksize < 1) {
        omcFail("ompMC:geomCyl:badGrid",
            "A cylinder needs at least one ring and one depth slab, not %d "
            "and %d.", geometry.isize, geometry.ksize);
    }

    if (geometry.rbounds == NULL) {
        omcFail("ompMC:geomCyl:badRadialBounds",
            "The cylinder has no ring boundaries.");
    }

    if (geometry.zbounds == NULL) {
        omcFail("ompMC:geomCyl:badDepthBounds",
            "The cylinder has no depth boundaries.");
    }

    /* Region 0 already means "outside the phantom", so there is nothing left
     for a hole in the middle to be. */
    if (geometry.rbounds[0] != 0.0) {
        omcFail("ompMC:geomCyl:badRadialBounds",
            "The innermost ring boundary is at r = %g; it has to be the axis, "
            "r = 0. A cylinder with a hole in it is not something this "
            "geometry can transport.", geometry.rbounds[0]);
    }

    for (int ir = 0; ir < geometry.isize; ir++) {
        if (geometry.rbounds[ir+1] <= geometry.rbounds[ir]) {
            omcFail("ompMC:geomCyl:badRadialBounds",
                "Ring boundaries have to ascend, but boundary %d is at r = %g "
                "and boundary %d at r = %g.",
                ir, geometry.rbounds[ir], ir + 1, geometry.rbounds[ir+1]);
        }
    }

    for (int iz = 0; iz < geometry.ksize; iz++) {
        if (geometry.zbounds[iz+1] <= geometry.zbounds[iz]) {
            omcFail("ompMC:geomCyl:badDepthBounds",
                "Depth boundaries have to ascend, but boundary %d is at "
                "z = %g and boundary %d at z = %g.",
                iz, geometry.zbounds[iz], iz + 1, geometry.zbounds[iz+1]);
        }
    }

    /* There is no azimuthal binning to have an opinion about. Pinning it here
     rather than asking the host for it keeps the rectilinear grid the region
     numbering is borrowed from out of every host's business. */
    geometry.jsize = 1;

    geometry.dri = omcUniformSpacingInv(geometry.rbounds, geometry.isize);
    geometry.dzi = omcUniformSpacingInv(geometry.zbounds, geometry.ksize);

    /* Unused in this mode, and left at 0 so that anything reaching for them
     gets the binary search rather than a stale multiplier. */
    geometry.dxi = 0.0;
    geometry.dyi = 0.0;

    geometry.mode = OMC_GEOM_CYLINDRICAL;

    return;
}

/******************************************************************************/

int omcCylRegionIndex(double x, double y, double z) {

    if (z < geometry.zbounds[0] || z > geometry.zbounds[geometry.ksize]) {
        return 0;
    }

    double radius = geometry.rbounds[geometry.isize];
    double r2 = x*x + y*y;

    if (r2 > radius*radius) {
        return 0;
    }

    /* Points exactly on the outer boundaries count as inside, consistent
     with the clamping of omcVoxelIndexFast() and with the rectilinear
     geometry. */
    int ir = omcVoxelIndexFast(geometry.rbounds, geometry.isize, geometry.dri,
                               sqrt(r2));
    int iz = omcVoxelIndexFast(geometry.zbounds, geometry.ksize, geometry.dzi,
                               z);

    return 1 + ir + iz*geometry.isize;
}

/******************************************************************************/

void omcCylHowfar(int *idisc, int *irnew, double *ustep) {

    int np = stack.np;
    int irl = stack.p[np].ir;

    if (irl == 0) {
        /* The particle is outside the geometry, terminate history */
        *idisc = 1;
        return;
    }

    int nr = geometry.isize;

    /* Same memo the rectilinear geometry uses, and correct here for the same
     reason the region numbering is: jsize is 1, so the y index it decodes is
     always 0 and the z index is the depth slab. */
    int ir, iy, iz;
    omcCachedDecodeRegion(irl, nr, 1, &ir, &iy, &iz);

    double x = stack.p[np].x;
    double y = stack.p[np].y;
    double z = stack.p[np].z;
    double u = stack.p[np].u;
    double v = stack.p[np].v;
    double w = stack.p[np].w;

    double dist;

    /* The depth slabs, which are ordinary planes and behave exactly as they
     do in the rectilinear geometry. The neighbour is a whole ring's worth of
     regions away, since ir runs fastest. */
    if (w > 0.0) {
        dist = (geometry.zbounds[iz+1] - z)/w;
        if (dist < *ustep) {
            *ustep = dist;
            if (iz != geometry.ksize - 1) {
                *irnew = irl + nr;
                omcStageRegion(*irnew, ir, 0, iz + 1);
            }
            else {
                *irnew = 0;         /* leaving geometry */
            }
        }
    }
    else if (w < 0.0) {
        dist = (geometry.zbounds[iz] - z)/w;
        if (dist < *ustep) {
            *ustep = dist;
            if (iz != 0) {
                *irnew = irl - nr;
                omcStageRegion(*irnew, ir, 0, iz - 1);
            }
            else {
                *irnew = 0;         /* leaving geometry */
            }
        }
    }

    /* The ring surfaces. A ray parallel to the axis crosses none of them, and
     is taken out here rather than left to divide by a == 0. */
    double a = u*u + v*v;

    if (a == 0.0) {
        return;
    }

    double r2 = x*x + y*y;
    double b = x*u + y*v;

    /* Outwards, through the surface the ring ends at. The particle is inside
     it, so c <= 0 and there is always a root ahead of it. */
    double router = geometry.rbounds[ir+1];
    double couter = r2 - router*router;
    double disc = b*b - a*couter;

    if (disc > 0.0) {
        dist = exitRoot(a, b, couter, disc);

        if (dist >= 0.0 && dist < *ustep) {
            *ustep = dist;
            if (ir != nr - 1) {
                *irnew = irl + 1;
                omcStageRegion(*irnew, ir + 1, 0, iz);
            }
            else {
                *irnew = 0;         /* leaving geometry */
            }
        }
    }

    /* Inwards, through the surface the ring starts at. Only worth asking
     about for a particle actually heading inwards -- b is the rate of change
     of r^2, so b >= 0 is one moving away from the axis -- and ring 0 has no
     inner surface at all, the axis not being a boundary. */
    if (ir > 0 && b < 0.0) {
        double rinner = geometry.rbounds[ir];
        double cinner = r2 - rinner*rinner;

        disc = b*b - a*cinner;

        /* Negative here is a chord that passes the axis further out than the
         inner surface: the ray never reaches it, and leaves through the outer
         surface found above instead. */
        if (disc >= 0.0) {
            dist = entryRoot(b, cinner, disc);

            if (dist >= 0.0 && dist < *ustep) {
                *ustep = dist;
                *irnew = irl - 1;
                omcStageRegion(*irnew, ir - 1, 0, iz);
            }
        }
    }

    return;
}

/******************************************************************************/

double omcCylHownear(void) {

    int np = stack.np;
    int irl = stack.p[np].ir;

    if (irl == 0) {
        /* Particle exiting geometry */
        return 0.0;
    }

    int ir, iy, iz;
    omcCachedDecodeRegion(irl, geometry.isize, 1, &ir, &iy, &iz);

    double x = stack.p[np].x;
    double y = stack.p[np].y;
    double z = stack.p[np].z;

    double r = sqrt(x*x + y*y);

    /* All four are exact perpendicular distances to a boundary of this
     region, so the smallest of them is the distance the stepping needs. No
     division anywhere, which is what makes a particle on the axis -- r
     exactly 0, and every pencil beam starts there -- an ordinary case rather
     than a singular one. */
    double tperp = geometry.zbounds[iz+1] - z;

    tperp = fmin(tperp, z - geometry.zbounds[iz]);
    tperp = fmin(tperp, geometry.rbounds[ir+1] - r);

    /* Ring 0 reaches the axis, and the axis is not a boundary */
    if (ir > 0) {
        tperp = fmin(tperp, r - geometry.rbounds[ir]);
    }

    return tperp;
}

/******************************************************************************/

int omcCylClipRay(double x, double y, double z, double u, double v, double w,
                  double *tenter) {

    /* tenter starts at 0 and only ever grows, so a ray that clears the
     cylinder clears it in front of the particle; one heading away is turned
     down by the interval going empty, as in omcSourcePlace(). */
    double enter = 0.0;
    double exit = DBL_MAX;

    if (!clipSlab(z, w, geometry.zbounds[0], geometry.zbounds[geometry.ksize],
                  &enter, &exit)) {
        return 0;
    }

    double radius = geometry.rbounds[geometry.isize];
    double a = u*u + v*v;
    double r2 = x*x + y*y;
    double c = r2 - radius*radius;

    if (a == 0.0) {
        /* Parallel to the axis: inside the barrel for the whole ray, or for
         none of it. The same case clipSlab() spells out for a plane. */
        if (c > 0.0) {
            return 0;
        }
    }
    else {
        double b = x*u + y*v;
        double disc = b*b - a*c;

        /* An interval of no length is a ray that touches the barrel at a
         single point and is inside it for no distance at all, which is not
         entering it. */
        if (disc <= 0.0) {
            return 0;
        }

        double sq = sqrt(disc);

        /* Both roots without cancellation: whichever of them the stable form
         gives directly, the other follows from their product being c/a. */
        double q = b >= 0.0 ? -(b + sq) : -(b - sq);
        double t1 = q/a;
        double t2 = c/q;

        if (t1 > t2) {
            double swap = t1;
            t1 = t2;
            t2 = swap;
        }

        if (t1 > enter) {
            enter = t1;
        }
        if (t2 < exit) {
            exit = t2;
        }

        if (!(enter < exit)) {
            return 0;
        }
    }

    *tenter = enter;

    return 1;
}

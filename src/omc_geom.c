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

#include "omc_geom.h"

#include "omc_geom_cyl.h"
#include "omc_host.h"
#include "omc_utilities.h"
#include "ompmc.h"

#include <math.h>
#include <stdlib.h>

#if defined(_MSC_VER)
    //use __declspec(thread) instead of threadprivate to avoid
    //error C3053. More information in:
    // https://stackoverflow.com/questions/12560243/using-threadprivate-directive-in-visual-studio
    __declspec(thread) extern struct Stack stack;
#else
    extern struct Stack stack;
    #pragma omp threadprivate(stack)
#endif
extern struct Media media;
extern struct Pegs pegs_data;
extern struct Region region;

struct Geom geometry;

void omcGeomDetectSpacing(void) {

    /* Detect uniform voxel spacing for the fast point location used by
     Woodcock photon tracking */
    geometry.dxi = omcUniformSpacingInv(geometry.xbounds, geometry.isize);
    geometry.dyi = omcUniformSpacingInv(geometry.ybounds, geometry.jsize);
    geometry.dzi = omcUniformSpacingInv(geometry.zbounds, geometry.ksize);

    /* Every loader of a voxel grid ends up here, which is what makes this the
     one place the mode can be set without a host having to remember. A
     resident host running a cylinder and then a cube would otherwise carry
     the cylinder over into the cube's transport. */
    geometry.mode = OMC_GEOM_CARTESIAN;

    return;
}

/* The three functions below are the geometry side of the contract ompmc.h
 declares, and each begins by asking which shape it is answering for.

 A predictable branch rather than a pointer through a table on purpose: these
 are called once per electron step, and the whole reason this project turns on
 link time optimization (see CMakeLists.txt) is so that they inline into the
 stepping loop. A direct call inlines; an indirect one through a function
 pointer does not, and would cost the rectilinear geometry -- which is every
 existing user code -- to buy the cylinder something it does not need. The
 mode cannot change during a run, so the branch predictor gets it right every
 time after the first. */

void howfar(int *idisc, int *irnew, double *ustep) {

    if (geometry.mode == OMC_GEOM_CYLINDRICAL) {
        omcCylHowfar(idisc, irnew, ustep);
        return;
    }

    int np = stack.np;
    int irl = stack.p[np].ir;
    double dist = 0.0;

    if (stack.p[np].ir == 0) {
        /* The particle is outside the geometry, terminate history */
        *idisc = 1;
        return;
    }

    /* If here, the particle is in the geometry, do transport checks */
    int imax = geometry.isize;
    int jmax = geometry.jsize;
    int ijmax = imax*jmax;

    /* First we need to decode the region number of the particle in terms of
     the region indices in each direction. The memo makes this free whenever
     this region was decoded, or staged as a neighbour, by an earlier call */
    int irx, iry, irz;
    omcCachedDecodeRegion(irl, imax, jmax, &irx, &iry, &irz);

    /* Reciprocal direction cosines, cached so that a photon marching through
     voxels divides only on its first step in a given direction */
    double ui, vi, wi;
    omcInvDir(stack.p[np].u, stack.p[np].v, stack.p[np].w, &ui, &vi, &wi);

    /* Whenever the step is truncated to a voxel face the indices of the
     neighbour behind it are known without any division; stage them so the
     next call, which runs in that neighbour, hits the memo. */

    /* Check in z-direction */
    if (stack.p[np].w > 0.0) {
        /* Going towards outer plane */
        dist = (geometry.zbounds[irz+1] - stack.p[np].z)*wi;
        if (dist < *ustep) {
            *ustep = dist;
            if (irz != (geometry.ksize - 1)) {
                *irnew = irl + ijmax;
                omcStageRegion(*irnew, irx, iry, irz + 1);
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }

    else if (stack.p[np].w < 0.0) {
        /* Going towards inner plane */
        dist = -(stack.p[np].z - geometry.zbounds[irz])*wi;
        if (dist < *ustep) {
            *ustep = dist;
            if (irz != 0) {
                *irnew = irl - ijmax;
                omcStageRegion(*irnew, irx, iry, irz - 1);
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }

    /* Check in x-direction */
    if (stack.p[np].u > 0.0) {
        /* Going towards positive plane */
        dist = (geometry.xbounds[irx+1] - stack.p[np].x)*ui;
        if (dist < *ustep) {
            *ustep = dist;
            if (irx != (geometry.isize - 1)) {
                *irnew = irl + 1;
                omcStageRegion(*irnew, irx + 1, iry, irz);
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }

    else if (stack.p[np].u < 0.0) {
        /* Going towards negative plane */
        dist = -(stack.p[np].x - geometry.xbounds[irx])*ui;
        if (dist < *ustep) {
            *ustep = dist;
            if (irx != 0) {
                *irnew = irl - 1;
                omcStageRegion(*irnew, irx - 1, iry, irz);
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }

    /* Check in y-direction */
    if (stack.p[np].v > 0.0) {
        /* Going towards positive plane */
        dist = (geometry.ybounds[iry+1] - stack.p[np].y)*vi;
        if (dist < *ustep) {
            *ustep = dist;
            if (iry != (geometry.jsize - 1)) {
                *irnew = irl + imax;
                omcStageRegion(*irnew, irx, iry + 1, irz);
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }

    else if (stack.p[np].v < 0.0) {
        /* Going towards negative plane */
        dist = -(stack.p[np].y - geometry.ybounds[iry])*vi;
        if (dist < *ustep) {
            *ustep = dist;
            if (iry != 0) {
                *irnew = irl - imax;
                omcStageRegion(*irnew, irx, iry - 1, irz);
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }

    return;
}

int regionIndex(double x, double y, double z) {

    if (geometry.mode == OMC_GEOM_CYLINDRICAL) {
        return omcCylRegionIndex(x, y, z);
    }

    /* Region containing the point, 0 if outside the phantom. Points exactly
     on the outer boundaries count as inside, consistent with the clamping
     of omcFindVoxelIndex(). */
    if (x < geometry.xbounds[0] || x > geometry.xbounds[geometry.isize] ||
        y < geometry.ybounds[0] || y > geometry.ybounds[geometry.jsize] ||
        z < geometry.zbounds[0] || z > geometry.zbounds[geometry.ksize]) {
        return 0;
    }

    int ix = omcVoxelIndexFast(geometry.xbounds, geometry.isize,
                               geometry.dxi, x);
    int iy = omcVoxelIndexFast(geometry.ybounds, geometry.jsize,
                               geometry.dyi, y);
    int iz = omcVoxelIndexFast(geometry.zbounds, geometry.ksize,
                               geometry.dzi, z);

    return 1 + ix + iy*geometry.isize + iz*geometry.isize*geometry.jsize;
}

double hownear(void) {

    if (geometry.mode == OMC_GEOM_CYLINDRICAL) {
        return omcCylHownear();
    }

    int np = stack.np;
    int irl = stack.p[np].ir;
    double tperp = 1.0E10;  /* perpendicular distance to closest boundary */

    if (irl == 0) {
        /* Particle exiting geometry */
        tperp = 0.0;
    }
    else {
        /* In the geometry, do transport checks */

        /* First we need to decode the region number of the particle in terms
         of the region indices in each direction */
        int irx, iry, irz;
        omcCachedDecodeRegion(irl, geometry.isize, geometry.jsize,
                              &irx, &iry, &irz);

        /* Check in x-direction */
        tperp = fmin(tperp, geometry.xbounds[irx+1] - stack.p[np].x);
        tperp = fmin(tperp, stack.p[np].x - geometry.xbounds[irx]);

        /* Check in y-direction */
        tperp = fmin(tperp, geometry.ybounds[iry+1] - stack.p[np].y);
        tperp = fmin(tperp, stack.p[np].y - geometry.ybounds[iry]);

        /* Check in z-direction */
        tperp = fmin(tperp, geometry.zbounds[irz+1] - stack.p[np].z);
        tperp = fmin(tperp, stack.p[np].z - geometry.zbounds[irz]);
    }

    return tperp;
}

void initRegions(void) {

    /* +1 : consider region surrounding phantom */
    int nreg = geometry.isize*geometry.jsize*geometry.ksize + 1;

    /* Allocate memory for region data. The cut-offs are per medium and live
     inside the struct, so only these two scale with the geometry. */
    region.med = malloc(nreg*sizeof(int));
    region.rhof = malloc(nreg*sizeof(double));

    /* First get global energy cutoff parameters */
    char buffer[BUFFER_SIZE];
    if (getInputValue(buffer, "global ecut") != 1) {
        omcFail("ompMC:geometry:missingInput",
            "Can not find 'global ecut' key on input file.");
    }
    double ecut = atof(buffer);

    if (getInputValue(buffer, "global pcut") != 1) {
        omcFail("ompMC:geometry:missingInput",
            "Can not find 'global pcut' key on input file.");
    }
    double pcut = atof(buffer);

    /* Transport cut-offs, per medium rather than per voxel. Slot 0 stands for
     vacuum, so the table is indexed by medium + 1. Doing this once per medium
     also means the warnings below are printed once each, rather than once per
     voxel of the medium. */
    region.pcut[0] = 0.0;
    region.ecut[0] = 0.0;

    for (int imed = 0; imed < media.nmed; imed++) {
        /* Check if global cut-off values are within PEGS data */
        if (pegs_data.ap[imed] <= pcut) {
            region.pcut[imed + 1] = pcut;
        } else {
            omcLog(OMC_LOG_WARNING,
                   "Warning!, global pcut value is below PEGS's pcut value "
                   "%f for medium %d, using PEGS value.",
                   pegs_data.ap[imed], imed);
            region.pcut[imed + 1] = pegs_data.ap[imed];
        }
        if (pegs_data.ae[imed] <= ecut) {
            region.ecut[imed + 1] = ecut;
        } else {
            omcLog(OMC_LOG_WARNING,
                   "Warning!, global ecut value is below PEGS's ecut value "
                   "%f for medium %d, using PEGS value.",
                   pegs_data.ae[imed], imed);
            region.ecut[imed + 1] = pegs_data.ae[imed];
        }
    }

    /* Initialize transport parameters on each region. Region 0 is outside the
     geometry */
    region.med[0] = VACUUM;
    region.rhof[0] = 0.0;

    /* Largest density ratio per medium, the basis of the Woodcock majorant */
    for (int imed = 0; imed < media.nmed; imed++) {
        region.rhof_max[imed] = 0.0;
    }

    for (int i=1; i<nreg; i++) {

        /* -1 : EGS counts media from 1. Substract 1 to get medium index */
        int imed = geometry.med_indices[i - 1] - 1;

        /* The cut-off tables are indexed by this, so a bad material index in
         the input would read past them rather than merely give odd physics */
        if (imed < VACUUM || imed >= media.nmed) {
            omcFail("ompMC:geometry:badMaterialIndex",
                "Voxel %d has material index %d, outside the %d media given.",
                i - 1, imed + 1, media.nmed);
        }

        region.med[i] = imed;

        if (imed == VACUUM) {
            region.rhof[i] = 0.0F;
        }
        else {
            if (geometry.med_densities[i - 1] == 0.0F) {
                region.rhof[i] = 1.0;
            }
            else {
                region.rhof[i] =
                    geometry.med_densities[i - 1]/pegs_data.rho[imed];
            }

            if (region.rhof[i] > region.rhof_max[imed]) {
                region.rhof_max[imed] = region.rhof[i];
            }
        }
    }

    return;
}

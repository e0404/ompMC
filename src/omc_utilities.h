#ifndef OMC_UTILITIES_H
#define OMC_UTILITIES_H
/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations
 
 Copyright (C) 2020 Edgardo Doerner (edoerner@fis.puc.cl)


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

/******************************************************************************/
/* Timing utilities. If OpenMP is enabled it calculates the wall time through 
 omp_get_wtime() function. Otherwise, it calculates CPU time through the clock() 
 function, available in time.h library. */

double omc_get_time();
/******************************************************************************/

/******************************************************************************/
/* A simple C/C++ class to parse input files and return requested key value 
https://github.com/bmaynard/iniReader */

#define BUFFER_SIZE 256
#define INPUT_PAIRS 80
#define INPUT_EXT ".inp"  // extension of input files

/* Parse a configuration file */
void parseInputFile(char *file_name);

/* Copy the value of the selected input item to the char pointer */
int getInputValue(char *dest, char *key);

/* Returns nonzero if line is a string containing only whitespace or is empty */
int lineBlack(char *line);

/* Remove white spaces from string str_untrimmed and saves the results in
 str_trimmed. Useful for string input values, such as file names */
void removeSpaces(char* str_trimmed, const char* str_untrimmed);

struct inputItems {
    char key[BUFFER_SIZE];
    char value[BUFFER_SIZE];
};


/******************************************************************************/

/******************************************************************************/
/* Voxel geometry helpers shared by the user codes. Both are on the transport
 hot path -- omcDecodeRegion() runs on every howfar() and hownear() call -- so
 they are inline in the header rather than a call into another translation
 unit. Keeping them here also makes them reachable from the unit tests. */

/* Decode a region number into its voxel indices along each axis. Regions are
 numbered 1 + ix + iy*imax + iz*imax*jmax, with 0 reserved for "outside the
 geometry"; irl must be >= 1.

 Written with two integer divisions rather than the three the arithmetic
 suggests: the quotient of the first division is imax*(iy + iz*jmax)/imax,
 i.e. exactly the combined y,z index, so the second division can work on that
 directly. Each division pairs with its own remainder into one instruction. */
static inline void omcDecodeRegion(int irl, int imax, int jmax,
                                   int *ix, int *iy, int *iz) {

    int ir0 = irl - 1;
    int irxy = ir0/imax;

    *ix = ir0 - irxy*imax;
    *iz = irxy/jmax;
    *iy = irxy - (*iz)*jmax;
}

/* Index of the voxel along one axis containing pos, i.e. the smallest i in
 [0, n-1] with bounds[i+1] >= pos. bounds holds n+1 ascending values.
 Positions outside the grid clamp to the first or last voxel rather than
 running off the end of bounds[]. */
static inline int omcFindVoxelIndex(const double *bounds, int n, double pos) {

    int lo = 0;
    int hi = n - 1;

    while (lo < hi) {
        int mid = lo + (hi - lo)/2;
        if (bounds[mid+1] < pos) {
            lo = mid + 1;
        }
        else {
            hi = mid;
        }
    }

    return lo;
}

/* Thread-local memo used by the user codes' howfar()/hownear() to keep the
 integer divisions of omcDecodeRegion() and the floating point divisions by
 the direction cosines off the per-step hot path.

 The region half is a pure irl -> (ix,iy,iz) memo: entries are only ever
 written as consistent pairs, so a staged entry that the transport ends up
 not visiting is harmless -- it is just a memo of a region nobody asks
 about. howfar() stages the indices of the neighbour region whenever it
 truncates the step to a voxel face, which it knows without any division, so
 the next call inside the new voxel hits the memo.

 The direction half holds the reciprocals of the last direction seen.
 Photons keep their direction while marching through voxels, so every
 howfar() call after the first works with multiplications instead of up to
 three divisions. Directions of exactly zero get a reciprocal of zero; the
 sign tests in howfar() ensure such a component is never used.

 Zero initialization leaves the memo empty: region 0 is outside the geometry
 and is rejected by the callers before any lookup, and no transported
 particle has the zero direction. */
struct OmcGeomCache {
    int irl;            /* region the indices below belong to; 0 = empty */
    int ix, iy, iz;

    double u, v, w;     /* direction the reciprocals below belong to */
    double ui, vi, wi;
};

#if defined(_MSC_VER)
    extern __declspec(thread) struct OmcGeomCache omc_geom_cache;
#else
    extern struct OmcGeomCache omc_geom_cache;
    #pragma omp threadprivate(omc_geom_cache)
#endif

static inline void omcCachedDecodeRegion(int irl, int imax, int jmax,
                                         int *ix, int *iy, int *iz) {

    if (omc_geom_cache.irl != irl) {
        omcDecodeRegion(irl, imax, jmax,
                        &omc_geom_cache.ix, &omc_geom_cache.iy,
                        &omc_geom_cache.iz);
        omc_geom_cache.irl = irl;
    }

    *ix = omc_geom_cache.ix;
    *iy = omc_geom_cache.iy;
    *iz = omc_geom_cache.iz;
}

/* Prefill the memo with a region whose indices the caller already knows */
static inline void omcStageRegion(int irl, int ix, int iy, int iz) {

    omc_geom_cache.irl = irl;
    omc_geom_cache.ix = ix;
    omc_geom_cache.iy = iy;
    omc_geom_cache.iz = iz;
}

static inline void omcInvDir(double u, double v, double w,
                             double *ui, double *vi, double *wi) {

    if (u != omc_geom_cache.u || v != omc_geom_cache.v ||
        w != omc_geom_cache.w) {
        omc_geom_cache.u = u;
        omc_geom_cache.v = v;
        omc_geom_cache.w = w;
        omc_geom_cache.ui = (u != 0.0) ? 1.0/u : 0.0;
        omc_geom_cache.vi = (v != 0.0) ? 1.0/v : 0.0;
        omc_geom_cache.wi = (w != 0.0) ? 1.0/w : 0.0;
    }

    *ui = omc_geom_cache.ui;
    *vi = omc_geom_cache.vi;
    *wi = omc_geom_cache.wi;
}
/******************************************************************************/

/* Flag set by '--verbose' argument */
extern int verbose_flag;

#endif
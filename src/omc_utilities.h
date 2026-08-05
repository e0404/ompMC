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

/*!
 @file
 Small utilities shared by the user codes: timing, input-file parsing, and
 voxel geometry helpers.
*****************************************************************************/

/*! @return The wall time in seconds if OpenMP is enabled (via
 `omp_get_wtime()`), otherwise the CPU time (via `clock()`, from `time.h`). */
double omc_get_time();

/*!
 Input-file parsing: a simple C/C++ class to parse input files and return
 requested key value pairs. https://github.com/bmaynard/iniReader
*/

#define BUFFER_SIZE 256      ///< maximum length of one input key or value, including the terminator
#define INPUT_PAIRS 80       ///< maximum number of key/value pairs #input_items holds
#define INPUT_EXT ".inp"     ///< extension of input files

/*! A path assembled by appending one of the data file names to a folder read
 from the input table. The folder is a value, so it is at most
 `BUFFER_SIZE-1` characters, and the names appended to it are short. Sizing
 these buffers by hand is how a Python package installed under a long
 temporary directory used to run off the end of a 128 byte array. */
#define PATH_SIZE (BUFFER_SIZE + 32)

/*! Parse a configuration file into #input_items.

 @param file_name Path to the input file. */
void parseInputFile(char *file_name);

/*! Copy the value of the selected input item to the char pointer.

 @param dest Buffer the value is copied into; must be at least
 `BUFFER_SIZE` bytes.
 @param key The key to look up.
 @return Nonzero if @p key was found. */
int getInputValue(char *dest, char *key);

/*! Set one key/value pair directly, for hosts that get their configuration
 from somewhere other than a file -- a MATLAB struct, a Python dict.
 Replaces the value when the key is already there.

 @param key The key, copied and truncated at `BUFFER_SIZE-1` characters.
 @param value The value, copied and truncated at `BUFFER_SIZE-1` characters. */
void omcSetInputValue(const char *key, const char *value);

/*! Forget every key/value pair. Hosts that stay resident between runs -- a
 MEX file, a Python module -- have to start each run from a clean table
 rather than inheriting the previous one. */
void omcClearInputValues(void);

/*! @param line The string to test.
 @return Nonzero if @p line contains only whitespace or is empty. */
int lineBlack(char *line);

/*! Remove white spaces from a string. Useful for string input values, such
 as file names.

 @param str_trimmed Destination buffer for the result.
 @param str_untrimmed The string to remove whitespace from. */
void removeSpaces(char* str_trimmed, const char* str_untrimmed);

/*! One key/value pair of #input_items. */
struct inputItems {
    char key[BUFFER_SIZE];       ///< the key, NUL terminated
    char value[BUFFER_SIZE];     ///< its value, NUL terminated
};

/*! The key/value table itself. Declared here rather than left for each user
 code to declare extern for itself, because that is how the two halves of the
 invariant below drifted apart in the first place.

 #input_idx is the NUMBER of pairs stored, and they occupy `input_items[0]`
 up to `input_items[input_idx - 1]`. An empty table is `input_idx == 0`,
 with no slot to look at -- which is what makes "is this table empty"
 answerable at all.

 @warning Anything filling the table directly rather than through
 omcSetInputValue() has to leave it that way. */
extern struct inputItems input_items[INPUT_PAIRS];
extern int input_idx;               ///< number of pairs stored in #input_items

/*!
 Voxel geometry helpers, shared by the user codes. Both omcDecodeRegion() and
 omcFindVoxelIndex() are on the transport hot path -- omcDecodeRegion() runs
 on every howfar() and hownear() call -- so they are inline in the header
 rather than a call into another translation unit. Keeping them here also
 makes them reachable from the unit tests.
*/

/*! Decode a region number into its voxel indices along each axis. Regions are
 numbered `1 + ix + iy*imax + iz*imax*jmax`, with 0 reserved for "outside the
 geometry".

 Written with two integer divisions rather than the three the arithmetic
 suggests: the quotient of the first division is `imax*(iy + iz*jmax)/imax`,
 i.e. exactly the combined y,z index, so the second division can work on that
 directly. Each division pairs with its own remainder into one instruction.

 @param irl Region number, must be >= 1.
 @param imax Number of voxels along x.
 @param jmax Number of voxels along y.
 @param ix Set to the voxel index along x.
 @param iy Set to the voxel index along y.
 @param iz Set to the voxel index along z. */
static inline void omcDecodeRegion(int irl, int imax, int jmax,
                                   int *ix, int *iy, int *iz) {

    int ir0 = irl - 1;
    int irxy = ir0/imax;

    *ix = ir0 - irxy*imax;
    *iz = irxy/jmax;
    *iy = irxy - (*iz)*jmax;
}

/*! Index of the voxel along one axis containing pos, i.e. the smallest i in
 `[0, n-1]` with `bounds[i+1] >= pos`. Positions outside the grid clamp to
 the first or last voxel rather than running off the end of @p bounds.

 @param bounds `n+1` ascending values.
 @param n Number of voxels along this axis.
 @param pos Position along this axis.
 @return The voxel index. */
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

/*! Evaluated once at initialization; the tolerance absorbs the
 single-precision noise that phantom files carry in their boundary lists.

 @param bounds `n+1` ascending values.
 @param n Number of voxels along this axis.
 @return The reciprocal of the grid spacing if the values in @p bounds are
 uniformly spaced to within a relative tolerance, 0.0 otherwise. */
static inline double omcUniformSpacingInv(const double *bounds, int n) {

    double dx = (bounds[n] - bounds[0])/(double)n;

    for (int i = 0; i < n; i++) {
        double d = bounds[i+1] - bounds[i];
        if (d < 0.999999*dx || d > 1.000001*dx) {
            return 0.0;
        }
    }

    return 1.0/dx;
}

/*! Voxel index of pos along one axis: a single multiplication on a uniform
 grid (@p invdx from omcUniformSpacingInv()), the binary search otherwise.
 The clamp keeps in-range results for positions on the outer boundaries;
 callers reject positions outside the grid before asking.

 @param bounds `n+1` ascending values.
 @param n Number of voxels along this axis.
 @param invdx Reciprocal grid spacing from omcUniformSpacingInv(), or 0.0 for
 a non-uniform grid.
 @param pos Position along this axis.
 @return The voxel index. */
static inline int omcVoxelIndexFast(const double *bounds, int n,
                                    double invdx, double pos) {

    if (invdx > 0.0) {
        int i = (int)((pos - bounds[0])*invdx);
        if (i < 0) {
            i = 0;
        }
        if (i > n - 1) {
            i = n - 1;
        }
        return i;
    }

    return omcFindVoxelIndex(bounds, n, pos);
}

/*! Thread-local memo used by the user codes' howfar()/hownear() to keep the
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
    int irl;            /**< region the indices below belong to; 0 = empty */
    int ix;             /**< voxel index along x of region #irl */
    int iy;             /**< voxel index along y of region #irl */
    int iz;             /**< voxel index along z of region #irl */

    double u;           /**< x component of the direction the reciprocals below belong to */
    double v;           /**< y component of the direction the reciprocals below belong to */
    double w;           /**< z component of the direction the reciprocals below belong to */
    double ui;          /**< reciprocal of #u */
    double vi;          /**< reciprocal of #v */
    double wi;          /**< reciprocal of #w */
};

/*! Per-thread instance of struct OmcGeomCache. */
#if defined(_MSC_VER)
    extern __declspec(thread) struct OmcGeomCache omc_geom_cache;
#else
    extern struct OmcGeomCache omc_geom_cache;
    #pragma omp threadprivate(omc_geom_cache)
#endif

/*! omcDecodeRegion(), memoized in #omc_geom_cache against the last region
 decoded on this thread. */
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

/*! Prefill #omc_geom_cache with a region whose indices the caller already
 knows. */
static inline void omcStageRegion(int irl, int ix, int iy, int iz) {

    omc_geom_cache.irl = irl;
    omc_geom_cache.ix = ix;
    omc_geom_cache.iy = iy;
    omc_geom_cache.iz = iz;
}

/*! Reciprocals of a direction, memoized in #omc_geom_cache against the last
 direction seen on this thread. */
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

/*! Flag set by the '--verbose' command line argument. */
extern int verbose_flag;

#endif

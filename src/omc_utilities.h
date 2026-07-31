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
/******************************************************************************/

/* Flag set by '--verbose' argument */
extern int verbose_flag;

#endif
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

#include "omc_score.h"
#include "omc_geom.h"
#include "ompmc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Redefine printf() function due to conflicts with mex and OpenMP */
#ifdef _OPENMP
    #include <omp.h>

    #undef printf
    #define printf(...) fprintf(stderr,__VA_ARGS__)
#endif

struct Score score;

/* The particle stack lives in the core library and is thread local. */
#if defined(_MSC_VER)
    __declspec(thread) extern struct Stack stack;
#else
    extern struct Stack stack;
    #pragma omp threadprivate(stack)
#endif

void initScore(int gridsize) {

    size_t n = (size_t)gridsize + 1;    // + 1 for region 0

    score.ensrc = 0.0;
    score.gridsize = gridsize;

    score.endep = malloc(n*sizeof(double));
    score.accum_endep = malloc(n*sizeof(double));
    score.accum_endep2 = malloc(n*sizeof(double));

    score.beam_list = malloc(n*sizeof(int));
    score.beam_flag = malloc(n*sizeof(unsigned char));

    if (!score.endep || !score.accum_endep || !score.accum_endep2 ||
        !score.beam_list || !score.beam_flag) {
        printf("Could not allocate the scoring arrays for %d voxels.\n",
               gridsize);
        exit(EXIT_FAILURE);
    }

    memset(score.endep, 0, n*sizeof(double));
    memset(score.accum_endep, 0, n*sizeof(double));
    memset(score.accum_endep2, 0, n*sizeof(double));
    memset(score.beam_flag, 0, n*sizeof(unsigned char));

    score.beam_count = 0;
    score.beam_sorted = 1;      // an empty list is trivially sorted

    return;
}

void cleanScore(void) {

    free(score.endep);
    free(score.accum_endep);
    free(score.accum_endep2);
    free(score.beam_list);
    free(score.beam_flag);

    return;
}

void scoreSource(double ein) {

    #pragma omp atomic
    score.ensrc += ein;

    return;
}

/* A relaxed atomic load and store of one beam_flag byte.

 The unlocked read in markDirty() below is a deliberate double check and the
 algorithm tolerates a stale answer, but "benign race" is not a category the C
 memory model has: an unsynchronised read concurrent with a write is a data
 race, and therefore undefined, however bounded the consequences look. Spell
 both sides out as atomic so there is no race to reason about.

 Relaxed is the whole ordering this needs. A thread that reads 1 only has to
 learn that the voxel is claimed; it never touches beam_list, and the list is
 read in accumEndep(), getBeamVoxels() and resetBeamScore(), all of which run
 outside the parallel region and so are already ordered after every write by
 the join barrier.

 OpenMP's own atomic read/write would say this in one line, but they are
 OpenMP 3.1 and MSVC compiles /openmp as 2.0, where both are a hard error
 (C3005). Hence the intrinsics. Both sides stay a single instruction on the
 targets this builds for: a relaxed byte load is a plain mov. */
#if defined(__GNUC__) || defined(__clang__)
    #define OMC_FLAG_LOAD(p)      __atomic_load_n((p), __ATOMIC_RELAXED)
    #define OMC_FLAG_STORE(p, v)  __atomic_store_n((p), (v), __ATOMIC_RELAXED)
#elif defined(_MSC_VER)
    /* An aligned byte access is indivisible on every architecture MSVC
     targets, so what is actually needed here is only that the compiler not
     invent, cache or reorder the accesses, which volatile gives. */
    #define OMC_FLAG_LOAD(p)      (*(volatile unsigned char *)(p))
    #define OMC_FLAG_STORE(p, v)  (*(volatile unsigned char *)(p) = (v))
#else
    #define OMC_FLAG_LOAD(p)      (*(volatile unsigned char *)(p))
    #define OMC_FLAG_STORE(p, v)  (*(volatile unsigned char *)(p) = (v))
#endif

/* Record that irl has received energy from this beamlet.

 Within a beamlet the flag only ever goes from 0 to 1 -- it is reset in
 resetBeamScore(), outside any parallel region -- so a read can only be stale
 in the direction of reporting 0 for a voxel another thread has just claimed.
 That costs one needless lock acquisition and nothing else, because the
 decision is retaken under the lock. Reading 1 is always truthful. The list
 therefore holds each voxel exactly once, and cannot overflow its gridsize+1
 entries.

 The locked path runs at most once per voxel per beamlet. After the first
 batch has mapped out the beamlet's footprint it essentially stops firing,
 and the millions of deposits that follow take the unlocked branch. */
/* Kept out of line, and separate from the test below, because a function
 containing an OpenMP construct is outlined by the compiler and so will not
 inline into its caller. Leaving the critical section in here lets the test
 that guards it collapse into a load and a branch at the call site. */
static void claimVoxel(int irl) {

    #pragma omp critical (ompmc_score_dirty)
    {
        /* This read needs no atomic of its own: every write to the flag is
         either in this critical section or outside the parallel region. */
        if (!score.beam_flag[irl]) {
            OMC_FLAG_STORE(&score.beam_flag[irl], 1);
            score.beam_list[score.beam_count] = irl;
            score.beam_count++;
            score.beam_sorted = 0;
        }
    }

    return;
}

static inline void markDirty(int irl) {

    if (!OMC_FLAG_LOAD(&score.beam_flag[irl])) {
        claimVoxel(irl);
    }

    return;
}

void ausgab(double edep) {

    int np = stack.np;
    int irl = stack.p[np].ir;
    double endep = stack.p[np].wt*edep;

    #pragma omp atomic
    score.endep[irl] += endep;

    markDirty(irl);

    return;
}

void accumEndep(double scale) {

    /* Walks the beamlet's footprint rather than the grid. Voxels this
     particular batch did not reach carry endep == 0 and contribute nothing,
     exactly as they did when this swept the whole grid. */
    for (int n = 0; n < score.beam_count; n++) {
        int irl = score.beam_list[n];
        double edep = score.endep[irl]*scale;

        score.accum_endep[irl] += edep;
        score.accum_endep2[irl] += edep*edep;

        /* Clear only what was written */
        score.endep[irl] = 0.0;
    }

    return;
}

static int compareIndices(const void *a, const void *b) {

    int ia = *(const int *)a;
    int ib = *(const int *)b;

    return (ia > ib) - (ia < ib);
}

int scoreBeamVoxels(const int **list) {

    /* Callers feed these straight into a CSC column, which wants ascending
     row indices. The list is built in the order voxels happened to be hit,
     so sort it here -- once; a beamlet asks for it more than once. */
    if (!score.beam_sorted) {
        qsort(score.beam_list, (size_t)score.beam_count, sizeof(int),
              compareIndices);
        score.beam_sorted = 1;
    }

    *list = score.beam_list;

    return score.beam_count;
}

void resetBeamScore(void) {

    for (int n = 0; n < score.beam_count; n++) {
        int irl = score.beam_list[n];

        score.accum_endep[irl] = 0.0;
        score.accum_endep2[irl] = 0.0;
        score.beam_flag[irl] = 0;
    }

    score.beam_count = 0;
    score.beam_sorted = 1;

    return;
}

/******************************************************************************/

/******************************************************************************/
/* Turn what the batches accumulated into dose and its relative uncertainty,
 and copy both out into the caller's arrays. Unlike the per beamlet
 accumulation in omc_engine_dij.c this walks the whole grid: a voxel that
 received nothing still has to come out with the 0.9999999 uncertainty the
 .3ddose format expects.

 Every engine producing a dense cube shares this, so that they cannot drift
 apart on the air threshold, the empty voxel convention or the batch
 variance. incFluence is what the accumulated energy is divided by: the
 histories per batch where they all count the same (omc_engine_cube), or
 1.0 where the caller has already folded the fluence into the weights it
 passed accumEndep() (omc_engine_forward). */

void omcScoreToCube(int nbatch, double incFluence, int outputDose,
                    double *dose, double *uncertainty) {

    int irl;
    int imax = geometry.isize;
    int ijmax = geometry.isize*geometry.jsize;
    double endep, endep2, unc_endep;

    double inc_fluence = incFluence;
    double mass;
    int iz;

    #pragma omp parallel for private(irl,endep,endep2,unc_endep,mass)
    for (iz=0; iz<geometry.ksize; iz++) {
        for (int iy=0; iy<geometry.jsize; iy++) {
            for (int ix=0; ix<geometry.isize; ix++) {
                irl = 1 + ix + iy*imax + iz*ijmax;

                /* Air is always reported as zero dose. Handle it before the
                 Gy conversion so a zero-density voxel cannot raise divide by
                 zero (or 0*inf invalid-operation) flags for a value that
                 would immediately be discarded. */
                if (geometry.med_densities[irl-1] < 0.044) {
                    dose[irl - 1] = 0.0;
                    if (uncertainty) {
                        uncertainty[irl - 1] = 0.9999999;
                    }
                    continue;
                }

                endep = score.accum_endep[irl];
                endep2 = score.accum_endep2[irl];

                /* First calculate mean deposited energy across batches and its
                 uncertainty */
                endep /= (double)nbatch;
                endep2 /= (double)nbatch;

                /* Batch approach uncertainty calculation */
                if (endep != 0.0) {
                    unc_endep = endep2 - endep*endep;
                    unc_endep /= (double)(nbatch - 1);

                    /* Relative uncertainty */
                    unc_endep = sqrt(unc_endep)/endep;
                }
                else {
                    endep = 0.0;
                    unc_endep = 0.9999999;
                }

                /* We separate de calculation of dose, to give the user the
                 option to output mean energy (outputDose=0) or deposited dose
                 (outputDose=1) per incident fluence */

                if (outputDose) {

                    /* Convert deposited energy to dose */
                    mass = (geometry.xbounds[ix+1] - geometry.xbounds[ix])*
                        (geometry.ybounds[iy+1] - geometry.ybounds[iy])*
                        (geometry.zbounds[iz+1] - geometry.zbounds[iz]);

                    /* Transform deposited energy to Gy */
                    mass *= geometry.med_densities[irl-1];
                    endep *= 1.602E-10/(mass*inc_fluence);

                } else {    /* Output mean deposited energy */
                    endep /= inc_fluence;
                }

                /* Store output quantities */
                dose[irl - 1] = endep;
                if (uncertainty) {
                    uncertainty[irl - 1] = unc_endep;
                }
            }
        }
    }

    return;
}

/******************************************************************************/
/* The same for a cylinder. Everything but the mass of a region is identical to
 omcScoreToCube() above -- the same batch variance, the same air threshold, the
 same 0.9999999 for a region nothing reached -- because they are the same
 conventions and a reader comparing an r-z result with a cube one should not
 have to wonder which of them is being used.

 What differs is that a ring is an annulus rather than a box, so its volume
 grows with how far out it is: pi*(r_out^2 - r_in^2)*dz rather than dx*dy*dz.
 That is also why the outer rings of a pencil beam calculation come out quiet
 despite receiving little -- a small energy spread over a large mass. */

void omcScoreToRadial(int nbatch, double incFluence, int outputDose,
                      double *dose, double *uncertainty) {

    int irl;
    int nr = geometry.isize;
    double endep, endep2, unc_endep;

    double inc_fluence = incFluence;
    double mass;
    int iz;

    #pragma omp parallel for private(irl,endep,endep2,unc_endep,mass)
    for (iz = 0; iz < geometry.ksize; iz++) {
        for (int ir = 0; ir < nr; ir++) {
            irl = 1 + ir + iz*nr;

            /* Air is always reported as zero dose. Handled before the Gy
             conversion so a zero-density region cannot raise divide by zero
             for a value that would immediately be discarded. */
            if (geometry.med_densities[irl-1] < 0.044) {
                dose[irl - 1] = 0.0;
                if (uncertainty) {
                    uncertainty[irl - 1] = 0.9999999;
                }
                continue;
            }

            endep = score.accum_endep[irl];
            endep2 = score.accum_endep2[irl];

            /* Mean deposited energy across batches and its uncertainty */
            endep /= (double)nbatch;
            endep2 /= (double)nbatch;

            /* Batch approach uncertainty calculation */
            if (endep != 0.0) {
                unc_endep = endep2 - endep*endep;
                unc_endep /= (double)(nbatch - 1);

                /* Relative uncertainty */
                unc_endep = sqrt(unc_endep)/endep;
            }
            else {
                endep = 0.0;
                unc_endep = 0.9999999;
            }

            if (outputDose) {

                /* The volume of the annulus this region is */
                double rin = geometry.rbounds[ir];
                double rout = geometry.rbounds[ir+1];

                mass = M_PI*(rout*rout - rin*rin)*
                    (geometry.zbounds[iz+1] - geometry.zbounds[iz]);

                /* Transform deposited energy to Gy */
                mass *= geometry.med_densities[irl-1];
                endep *= 1.602E-10/(mass*inc_fluence);

            } else {    /* Output mean deposited energy */
                endep /= inc_fluence;
            }

            /* Store output quantities */
            dose[irl - 1] = endep;
            if (uncertainty) {
                uncertainty[irl - 1] = unc_endep;
            }
        }
    }

    return;
}

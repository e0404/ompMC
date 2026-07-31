#ifndef OMC_SCORE_H
#define OMC_SCORE_H
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

/*******************************************************************************
* Energy scoring, shared by the user codes.
*
* The dose from one beamlet occupies a small fraction of the grid, often well
* under one percent, so anything that sweeps the whole grid once per batch or
* once per beamlet costs far more than the scoring itself -- and a Dij run
* does exactly that, nbeamlets times over. Every voxel that receives energy is
* therefore recorded in a list, and the accumulation, output and reset steps
* walk that list instead of the grid.
*
* ausgab() itself keeps the plain atomic add. Combining deposits per voxel in
* thread local state first was tried and measured slower: an electron does not
* stay in one voxel for long enough runs of steps to pay for touching thread
* local state on every call, and the extra branch and TLS access cost about
* ten percent on omc_dosxyz while saving nothing measurable.
*
* Call order over a beamlet:
*
*   initScore(gridsize)                once
*     ... per batch:
*       parallel { initHistory(); shower(); }
*       accumEndep(scale)
*     scoreBeamVoxels(&list)           to emit the beamlet's results
*     resetBeamScore()                 before the next beamlet
*   cleanScore()                       once
*******************************************************************************/

struct Score {
    double ensrc;               // total energy from source

    int gridsize;               // voxel count; regions run 1..gridsize, with
                                // region 0 reserved for outside the geometry

    double *endep;              // energy deposited during the current batch
    double *accum_endep;        // accumulated across batches
    double *accum_endep2;       // accumulated squares, for the batch variance

    /* Voxels this beamlet has deposited energy in, i.e. since the last
     resetBeamScore(). The flag keeps the list duplicate free; see
     omc_score.c for how it stays correct under concurrent updates.

     Tracking runs per beamlet rather than per batch on purpose. The set a
     single batch touches is barely smaller than the set the whole beamlet
     touches -- same beamlet, same footprint -- so a per batch list would
     save nothing in the accumulation while making every batch re-enter the
     lock once for every voxel it hits. */
    int *beam_list;
    int beam_count;
    int beam_sorted;            // beam_list is already in ascending order
    unsigned char *beam_flag;
};

extern struct Score score;

/* gridsize is the number of voxels, not counting region 0. */
void initScore(int gridsize);
void cleanScore(void);

/* Add to the source energy tally. Safe to call from inside a parallel
 region, unlike a bare score.ensrc += ein. */
void scoreSource(double ein);

/* Fold the current batch into the accumulators, multiplying by scale, then
 clear it. Costs O(voxels this beamlet has touched), not O(gridsize). Call
 outside any parallel region. */
void accumEndep(double scale);

/* The voxels this beamlet has deposited energy in, ascending so that callers
 can emit them straight into a column of a CSC sparse matrix. Returns the
 count and, through list, the indices. */
int scoreBeamVoxels(const int **list);

/* Zero the accumulators over this beamlet's touched set and begin a new
 beamlet. Costs O(voxels touched). */
void resetBeamScore(void);

#endif  // OMC_SCORE_H

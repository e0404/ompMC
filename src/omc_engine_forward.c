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

#include "omc_engine_forward.h"

#include "omc_geom.h"
#include "omc_host.h"
#include "omc_random.h"
#include "omc_score.h"
#include "omc_spectrum.h"
#include "omc_utilities.h"
#include "ompmc.h"

#include <math.h>
#include <stdlib.h>

/******************************************************************************/
/* Handing the histories out to the beamlets.

 Every batch runs the same experiment: beamlet i contributes the same count[i]
 histories to each of them, so the batches stay the independent replicas the
 variance estimate assumes. The counts are shared out by walking the cumulative
 weight, which needs no sort, is deterministic, and leaves each count within
 one history of the exact share.

 A beamlet whose share rounds to zero is dropped rather than given a history it
 has not earned. What that costs is bounded -- its weight is below one
 nperbatch-th of the total -- and it is reported rather than hidden.

 The rounding that is left over rides on the particle weight instead of on the
 counts: beamlet i wants weight[i] of the fluence and got count[i] of the
 nperbatch histories, so each of its particles carries

     wt[i] = (weight[i]/count[i]) / (W/nperbatch)

 which is 1 up to the rounding, and exactly 1 when the share came out whole.
 Keeping it near 1 rather than folding the whole fluence into it leaves the
 transport working with the weights it has always seen; the physical scale
 goes on the batch instead, in the accumEndep() call below. */

struct Allocation {
    int *count;                 // histories per beamlet per batch
    int *offset;                // prefix sum, nbeamlets + 1 entries
    double *weight;             // statistical weight of each beamlet's particles

    int nweighted;              // beamlets asked for with a weight above zero
    int nsampled;               // of those, the ones that got any histories
    double totalWeight;
    double sampledWeight;
};

static void freeAllocation(struct Allocation *a) {

    free(a->count);
    free(a->offset);
    free(a->weight);

    a->count = NULL;
    a->offset = NULL;
    a->weight = NULL;

    return;
}

static void buildAllocation(struct Allocation *a, int nbeamlets,
                            const double *weights, int nperbatch) {

    a->count = (int*) malloc((size_t)nbeamlets*sizeof(int));
    a->offset = (int*) malloc(((size_t)nbeamlets + 1)*sizeof(int));
    a->weight = (double*) malloc((size_t)nbeamlets*sizeof(double));

    if (!a->count || !a->offset || !a->weight) {
        freeAllocation(a);
        omcFail("ompMC:forward:outOfMemory",
            "Could not allocate the history distribution for %d beamlets.",
            nbeamlets);
    }

    /* Total weight, and the heaviest beamlet, which absorbs the rounding of
     the cumulative sum at the end */
    double total = 0.0;
    int heaviest = 0;

    for (int i = 0; i < nbeamlets; i++) {
        /* Written so that a NaN fails it: NaN < 0.0 is false. */
        if (!(weights[i] >= 0.0)) {
            freeAllocation(a);
            omcFail("ompMC:forward:invalidWeight",
                "Beamlet weight %d is %g; weights must be zero or positive.",
                i + 1, weights[i]);
        }
        if (weights[i] > weights[heaviest]) {
            heaviest = i;
        }
        total += weights[i];
    }

    if (!(total > 0.0)) {
        freeAllocation(a);
        omcFail("ompMC:forward:noWeight",
            "Every beamlet weight is zero, so there is nothing to calculate.");
    }

    double cumulative = 0.0;
    int handedOut = 0;

    a->nweighted = 0;
    a->nsampled = 0;
    a->totalWeight = total;
    a->sampledWeight = 0.0;

    a->offset[0] = 0;

    for (int i = 0; i < nbeamlets; i++) {
        cumulative += weights[i];

        int upto = (int) floor((double)nperbatch*(cumulative/total) + 0.5);
        if (upto > nperbatch) {
            upto = nperbatch;
        }
        if (upto < handedOut) {
            upto = handedOut;
        }

        a->count[i] = upto - handedOut;
        handedOut = upto;

        a->offset[i+1] = handedOut;

        if (weights[i] > 0.0) {
            a->nweighted++;
        }
    }

    /* The cumulative sum ends at total, so the last beamlet with a weight
     should have taken the count up to nperbatch exactly. Rounding can leave
     it a history short or over; put the difference on the heaviest beamlet,
     which is the one least disturbed by it -- and never on a beamlet the
     caller asked for zero of. */
    if (handedOut != nperbatch) {
        int fix = nperbatch - handedOut;

        if (a->count[heaviest] + fix < 0) {
            fix = -a->count[heaviest];
        }

        a->count[heaviest] += fix;
        for (int i = heaviest + 1; i <= nbeamlets; i++) {
            a->offset[i] += fix;
        }
    }

    for (int i = 0; i < nbeamlets; i++) {
        if (a->count[i] > 0) {
            a->weight[i] = (weights[i]/(double)a->count[i])
                           *((double)nperbatch/total);
            a->nsampled++;
            a->sampledWeight += weights[i];
        }
        else {
            a->weight[i] = 0.0;
        }
    }

    return;
}

/* The beamlet history h of a batch belongs to: the largest i with
 offset[i] <= h. Beamlets that got no histories have offset[i] == offset[i+1]
 and so can never be the largest, which is what keeps them out. */
static int findBeamlet(const int *offset, int nbeamlets, int h) {

    int lo = 0;
    int hi = nbeamlets;         // the answer is in [lo, hi)

    while (hi - lo > 1) {
        int mid = lo + (hi - lo)/2;

        if (offset[mid] <= h) {
            lo = mid;
        }
        else {
            hi = mid;
        }
    }

    return lo;
}

/******************************************************************************/

int omcCalcForward(const struct OmcForwardOptions *opt,
                   const struct OmcBeamletSource *src,
                   const double *weights,
                   const struct OmcSpectrum *spec,
                   double *dose, double *uncertainty,
                   const struct OmcForwardCallbacks *callbacks,
                   struct OmcForwardSummary *summary) {

    if (opt->sourceGeometry != OMC_SOURCE_POINT &&
        opt->sourceGeometry != OMC_SOURCE_GAUSSIAN) {
        omcFail("ompMC:forward:invalidSourceGeometry",
            "Source geometry %d is not defined.", (int)opt->sourceGeometry);
    }
    if (src->nbeamlets < 1) {
        omcFail("ompMC:forward:noBeamlets",
            "There are no beamlets to calculate.");
    }
    if (opt->nbatch < 2) {
        /* The batch variance divides by nbatch - 1 */
        omcFail("ompMC:forward:tooFewBatches",
            "Number of batches is %d, at least 2 are needed for the "
            "uncertainty estimate.", opt->nbatch);
    }

    struct OmcBeamletSampler sampler;
    sampler.source = src;
    sampler.spectrum = spec;
    sampler.charge = opt->charge;
    sampler.geometry = opt->sourceGeometry;
    sampler.gaussianWidth = opt->sourceGaussianWidth;

    int nhist = opt->nhist;
    int nbatch = opt->nbatch;

    if (nhist/nbatch == 0) {
        nhist = nbatch;
    }

    int nperbatch = nhist/nbatch;
    nhist = nperbatch*nbatch;

    struct Allocation alloc;
    alloc.count = NULL;
    alloc.offset = NULL;
    alloc.weight = NULL;

    buildAllocation(&alloc, src->nbeamlets, weights, nperbatch);

    int gridsize = geometry.isize*geometry.jsize*geometry.ksize;

    omcLog(OMC_LOG_DETAIL, "Total number of particle histories: %d", nhist);
    omcLog(OMC_LOG_DETAIL, "Number of statistical batches: %d", nbatch);
    omcLog(OMC_LOG_DETAIL, "Histories per batch: %d", nperbatch);
    omcLog(OMC_LOG_DETAIL, "Beamlets with weight: %d of %d, %d of them sampled",
           alloc.nweighted, src->nbeamlets, alloc.nsampled);

    double dropped = alloc.totalWeight - alloc.sampledWeight;

    if (dropped > 1.0E-3*alloc.totalWeight) {
        omcLog(OMC_LOG_WARNING,
            "%d of %d weighted beamlets are too weak to be given a history "
            "each batch, which leaves out %.2f%% of the fluence. Raise the "
            "number of histories or lower the number of batches.",
            alloc.nweighted - alloc.nsampled, alloc.nweighted,
            100.0*dropped/alloc.totalWeight);
    }

    /* Preparation of scoring struct */
    initScore(gridsize);

    #pragma omp parallel
    {
      /* Initialize random number generator */
      initRandom();

      /* Initialize particle stack */
      initStack();
    }

    int aborted = 0;

    for (int ibatch = 0; ibatch < nbatch; ibatch++) {
        int ihist;

        #pragma omp parallel for schedule(dynamic)
        for (ihist = 0; ihist < nperbatch; ihist++) {
            /* Point the RNG at this history's stream; the index is unique
             across batches, so results do not depend on the scheduling */
            setRandomHistory((uint64_t)ibatch*(uint64_t)nperbatch
                             + (uint64_t)ihist);

            int ibeamlet = findBeamlet(alloc.offset, src->nbeamlets, ihist);

            /* Initialize particle history */
            omcBeamletSample(&sampler, ibeamlet, alloc.weight[ibeamlet]);

            /* Start electromagnetic shower simulation */
            shower();
        }

        /* Accumulate results of current batch for statistical analysis. The
         particle weights above sum the batch to the fluence of nperbatch
         histories rather than to the fluence the caller asked for, so the
         ratio between the two goes on here -- once per batch, rather than on
         every particle. */
        accumEndep(alloc.totalWeight/(double)nperbatch);

        if (callbacks && callbacks->progress &&
            !callbacks->progress((double)(ibatch+1)/(double)nbatch,
                                 callbacks->user)) {
            aborted = 1;
            break;
        }
    }

    /* The fraction of the incident energy that stayed in the phantom, while
     the scoring arrays still hold energies rather than doses. Both sides are
     weighted -- omcBeamletSample() puts the particle weight through to
     scoreSource() as well -- so the batch scale above is all that separates
     them. */
    if (summary && !aborted) {
        double etot = 0.0;
        for (int irl = 1; irl < gridsize + 1; irl++) {
            etot += score.accum_endep[irl];
        }

        summary->nhist = nhist;
        summary->nperbatch = nperbatch;
        summary->nweighted = alloc.nweighted;
        summary->nsampled = alloc.nsampled;
        summary->totalWeight = alloc.totalWeight;
        summary->sampledWeight = alloc.sampledWeight;
        summary->energyFraction = score.ensrc > 0.0
            ? etot/(score.ensrc*alloc.totalWeight/(double)nperbatch)
            : 0.0;
    }

    /* The fluence is already in the accumulators, so there is nothing left to
     divide the energy by here. */
    if (!aborted) {
        omcScoreToCube(nbatch, 1.0, opt->outputDose, dose, uncertainty);
    }

    cleanScore();

    //Cleaning private random generators and particle stack
    #pragma omp parallel
    {
      cleanRandom();
      cleanStack();
    }

    freeAllocation(&alloc);

    return !aborted;
}

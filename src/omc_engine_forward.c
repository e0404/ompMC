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
#include "omc_phsp.h"
#include "omc_random.h"
#include "omc_score.h"
#include "omc_source_phsp.h"
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
        if (!isfinite(weights[i]) || weights[i] < 0.0) {
            freeAllocation(a);
            omcFail("ompMC:forward:invalidWeight",
                "Beamlet weight %d is %g; weights must be finite and zero "
                "or positive.",
                i + 1, weights[i]);
        }
        if (weights[i] > weights[heaviest]) {
            heaviest = i;
        }
        total += weights[i];
    }

    /* Individually finite values can still overflow when summed. That would
     make every cumulative/total allocation ratio invalid. */
    if (!isfinite(total)) {
        freeAllocation(a);
        omcFail("ompMC:forward:invalidWeight",
            "The beamlet weights sum to a non-finite value; reduce their "
            "scale.");
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
/* Running the batches.

 The two calculations below differ in one line -- where a history's particle
 comes from -- and agree on everything around it: how many batches there are,
 which random stream each history gets, when the batch is accumulated and when
 the caller is asked whether to carry on. That agreement is worth more than it
 looks. The random stream is indexed by a history number that has to be unique
 over the whole run for the answer not to depend on how OpenMP handed the
 histories out, and two copies of that indexing would be two chances to get it
 wrong. So it lives here once, and what differs is passed in. */

struct HistoryStarter {
    /*! Put this history's particle on the stack.

     @param source Whatever the starter needs, unchanged.
     @param ihist Global history index, unique over the run.
     @param ihistInBatch Index within the batch, which is what shares the
     histories out among beamlets.
     @return 1 if there is a particle to shower, 0 if the history is empty. */
    int (*start)(const void *source, uint64_t ihist, int ihistInBatch);

    const void *source;
};

/*! @return 1 if the progress callback stopped the run. */
static int runBatches(const struct HistoryStarter *starter, int nbatch,
                      int nperbatch, double batchScale,
                      const struct OmcForwardCallbacks *callbacks,
                      unsigned long long *started) {

    unsigned long long nstarted = 0;
    int aborted = 0;

    for (int ibatch = 0; ibatch < nbatch; ibatch++) {
        int ihist;
        /* int rather than a wider type because MSVC implements OpenMP 2.0,
         whose reductions are fussier, and a batch cannot start more
         histories than the nperbatch it runs. */
        int batchStarted = 0;

        #pragma omp parallel for schedule(dynamic) reduction(+:batchStarted)
        for (ihist = 0; ihist < nperbatch; ihist++) {
            /* Point the RNG at this history's stream; the index is unique
             across batches, so results do not depend on the scheduling */
            uint64_t global = (uint64_t)ibatch*(uint64_t)nperbatch
                              + (uint64_t)ihist;

            setRandomHistory(global);

            /* Initialize particle history. A history that starts nothing is
             a history all the same -- it happened, it just had nothing in
             it -- so it counts towards the fluence and only skips the
             shower. */
            if (starter->start(starter->source, global, ihist)) {
                batchStarted++;

                /* Start electromagnetic shower simulation */
                shower();
            }
        }

        nstarted += (unsigned long long)batchStarted;

        /* Accumulate results of current batch for statistical analysis. */
        accumEndep(batchScale);

        if (callbacks && callbacks->progress &&
            !callbacks->progress((double)(ibatch+1)/(double)nbatch,
                                 callbacks->user)) {
            aborted = 1;
            break;
        }
    }

    if (started != NULL) {
        *started = nstarted;
    }

    return aborted;
}

/* Rounding the run to whole batches, which both calculations do the same
 way: a run too short for one history per batch is stretched rather than
 refused, and what is left over after the division is dropped. */
static void roundToBatches(int nhistWanted, int nbatch, int *nhist,
                           int *nperbatch) {

    int histories = nhistWanted;

    if (histories/nbatch == 0) {
        histories = nbatch;
    }

    *nperbatch = histories/nbatch;
    *nhist = *nperbatch*nbatch;

    return;
}

/******************************************************************************/

/* What a beamlet history needs: which beamlets there are, and how the
 histories of a batch were shared out among them. */
struct BeamletStarter {
    const struct OmcBeamletSampler *sampler;
    const struct Allocation *alloc;
    int nbeamlets;
};

static int startBeamlet(const void *source, uint64_t ihist, int ihistInBatch) {

    const struct BeamletStarter *s = (const struct BeamletStarter *)source;

    (void)ihist;

    int ibeamlet = findBeamlet(s->alloc->offset, s->nbeamlets, ihistInBatch);

    omcBeamletSample(s->sampler, ibeamlet, s->alloc->weight[ibeamlet]);

    /* A beamlet particle is aimed at the phantom by construction, so there
     is always one to shower. */
    return 1;
}

static int startPhsp(const void *source, uint64_t ihist, int ihistInBatch) {

    (void)ihistInBatch;

    return omcPhspSourceSample((const struct OmcPhspSampler *)source, ihist,
                               1.0);
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

    int nbatch = opt->nbatch;
    int nhist;
    int nperbatch;

    roundToBatches(opt->nhist, nbatch, &nhist, &nperbatch);

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

    struct BeamletStarter beamlets;
    beamlets.sampler = &sampler;
    beamlets.alloc = &alloc;
    beamlets.nbeamlets = src->nbeamlets;

    struct HistoryStarter starter;
    starter.start = startBeamlet;
    starter.source = &beamlets;

    /* The particle weights sum the batch to the fluence of nperbatch
     histories rather than to the fluence the caller asked for, so the ratio
     between the two goes on the batch -- once per batch, rather than on
     every particle. */
    int aborted = runBatches(&starter, nbatch, nperbatch,
                             alloc.totalWeight/(double)nperbatch,
                             callbacks, NULL);

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

/******************************************************************************/

int omcCalcForwardPhsp(const struct OmcForwardPhspOptions *opt,
                       const struct OmcPhsp *phsp,
                       double *dose, double *uncertainty,
                       const struct OmcForwardCallbacks *callbacks,
                       struct OmcForwardPhspSummary *summary) {

    if (opt->nbatch < 2) {
        /* The batch variance divides by nbatch - 1 */
        omcFail("ompMC:forward:tooFewBatches",
            "Number of batches is %d, at least 2 are needed for the "
            "uncertainty estimate.", opt->nbatch);
    }

    struct OmcPhspSampler sampler;
    sampler.phsp = phsp;
    sampler.order = opt->order;
    sampler.first = opt->first;
    sampler.transform = opt->transform;

    /* Everything that could be wrong with the source is settled here, on the
     master thread, rather than from inside the parallel region where a
     failure would call the host from a place it cannot expect. */
    omcPhspSourceCheck(&sampler);

    int nbatch = opt->nbatch;
    int nhist;
    int nperbatch;

    roundToBatches(opt->nhist, nbatch, &nhist, &nperbatch);

    int gridsize = geometry.isize*geometry.jsize*geometry.ksize;

    omcLog(OMC_LOG_DETAIL, "Total number of particle histories: %d", nhist);
    omcLog(OMC_LOG_DETAIL, "Number of statistical batches: %d", nbatch);
    omcLog(OMC_LOG_DETAIL, "Histories per batch: %d", nperbatch);

    if ((unsigned long long)nhist > omcPhspCount(phsp) &&
        opt->order == OMC_PHSP_REPLAY) {
        omcLog(OMC_LOG_WARNING, "Running %d histories through a phase space "
               "of %llu particles replays some of them more than once, which "
               "buys less than the history count suggests: a particle used "
               "twice tells you no more the second time about what the beam "
               "does, only about what this phantom does with it.",
               nhist, omcPhspCount(phsp));
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

    struct HistoryStarter starter;
    starter.start = startPhsp;
    starter.source = &sampler;

    /* Nothing to rescale per batch: the particles carry the weights the file
     gave them, and the dose comes out per history when the accumulated
     energy is divided by the history count below. */
    unsigned long long started = 0;
    int aborted = runBatches(&starter, nbatch, nperbatch, 1.0, callbacks,
                             &started);

    if (!aborted) {
        omcLog(OMC_LOG_DETAIL, "%llu of %d histories put a particle in the "
               "phantom.", started, nhist);

        if (started == 0) {
            omcLog(OMC_LOG_WARNING, "Not one history put a particle in the "
                   "phantom. Check where the phase space sits relative to it: "
                   "the transform that carries one to the other is the usual "
                   "thing to have wrong.");
        }
    }

    /* The fraction of the incident energy that stayed in the phantom, while
     the scoring arrays still hold energies rather than doses. Both sides are
     weighted -- omcPhspSourceSample() puts the particle weight through to
     scoreSource() as well -- and the batch scale is 1, so they are directly
     comparable. */
    if (summary && !aborted) {
        double etot = 0.0;
        for (int irl = 1; irl < gridsize + 1; irl++) {
            etot += score.accum_endep[irl];
        }

        summary->nhist = nhist;
        summary->nperbatch = nperbatch;
        summary->started = started;
        summary->energyFraction = score.ensrc > 0.0 ? etot/score.ensrc : 0.0;
    }

    /* Per history, counting the ones whose particle missed: they are part of
     the fluence the file stands for. */
    if (!aborted) {
        omcScoreToCube(nbatch, (double)nperbatch, opt->outputDose, dose,
                       uncertainty);
    }

    cleanScore();

    //Cleaning private random generators and particle stack
    #pragma omp parallel
    {
      cleanRandom();
      cleanStack();
    }

    return !aborted;
}

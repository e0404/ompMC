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

#include "omc_collimator.h"
#include "omc_engine_batches.h"
#include "omc_geom.h"
#include "omc_host.h"
#include "omc_random.h"
#include "omc_score.h"
#include "omc_source.h"
#include "omc_utilities.h"
#include "ompmc.h"

#include <stdlib.h>

/******************************************************************************/
/* Running the batches.

 Whatever the particles come from, this is the same: how many batches there
 are, which random stream each history gets, when a batch is accumulated and
 when the caller is asked whether to carry on. The random stream is indexed by
 a history number that has to be unique over the whole run for the answer not
 to depend on how OpenMP handed the histories out, so it is worth having in
 one place rather than once per kind of source.

 It lives here, and is declared in omc_engine_batches.h, because the radial
 engine runs the very same loop: the only thing that differs between the two
 is the shape of the phantom underneath and how the result is written out. */
int omcEngineRunBatches(struct OmcSource *source,
                        const struct OmcBeamModifier *modifier,
                        int nbatch, int nperbatch,
                        const struct OmcForwardCallbacks *callbacks,
                        unsigned long long *started,
                        unsigned long long *blocked) {

    unsigned long long nstarted = 0;
    unsigned long long nblocked = 0;
    int aborted = 0;

    for (int ibatch = 0; ibatch < nbatch; ibatch++) {
        int ihist;
        /* int rather than a wider type because MSVC implements OpenMP 2.0,
         whose reductions are fussier, and a batch cannot start more
         histories than the nperbatch it runs. */
        int batchStarted = 0;
        int batchBlocked = 0;

        #pragma omp parallel for schedule(dynamic) \
            reduction(+:batchStarted) reduction(+:batchBlocked)
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
            struct OmcSourceParticle particle;

            if (source->sample(source, global, ihist, &particle)) {

                /* What the collimator does with it, asked before the particle
                 is carried anywhere: one that is stopped is stopped, and need
                 not be carried first. */
                if (!omcBeamModifierApply(modifier, &particle)) {
                    batchBlocked++;
                }
                else if (omcSourcePlace(&particle)) {
                    /* Only what got into the phantom counts as energy put in,
                     so that the fraction of it that ends up deposited means
                     what it says. */
                    scoreSource(particle.energy*particle.weight);

                    batchStarted++;

                    /* Start electromagnetic shower simulation */
                    shower();
                }
            }
        }

        nstarted += (unsigned long long)batchStarted;
        nblocked += (unsigned long long)batchBlocked;

        /* Accumulate results of current batch for statistical analysis. */
        accumEndep(source->batchScale);

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
    if (blocked != NULL) {
        *blocked = nblocked;
    }

    return aborted;
}

/******************************************************************************/

int omcCalcForward(const struct OmcForwardOptions *opt,
                   struct OmcSource *source,
                   const struct OmcBeamModifier *modifier,
                   double *dose, double *uncertainty,
                   const struct OmcForwardCallbacks *callbacks,
                   struct OmcForwardSummary *summary) {

    if (opt->nbatch < 2) {
        /* The batch variance divides by nbatch - 1 */
        omcFail("ompMC:forward:tooFewBatches",
            "Number of batches is %d, at least 2 are needed for the "
            "uncertainty estimate.", opt->nbatch);
    }

    if (source->sample == NULL) {
        omcFail("ompMC:forward:noSource",
            "The source has no way of making a particle.");
    }

    /* Everything that could be wrong with the source is settled here, on the
     master thread, rather than from inside the parallel region where a
     failure would call the host from a place it cannot expect. */
    if (source->check != NULL) {
        source->check(source);
    }
    if (modifier != NULL && modifier->check != NULL) {
        modifier->check(modifier);
    }

    /* A run too short for one history per batch is stretched rather than
     refused, and what is left over after the division is dropped. */
    int nbatch = opt->nbatch;
    int nhist = opt->nhist;

    if (nhist/nbatch == 0) {
        nhist = nbatch;
    }

    int nperbatch = nhist/nbatch;
    nhist = nperbatch*nbatch;

    /* What one history is worth is the source's to say. */
    source->batchScale = 1.0;
    source->incidentFluence = 1.0;

    if (source->prepare != NULL) {
        source->prepare(source, nperbatch);
    }

    int gridsize = geometry.isize*geometry.jsize*geometry.ksize;

    omcLog(OMC_LOG_DETAIL, "Total number of particle histories: %d", nhist);
    omcLog(OMC_LOG_DETAIL, "Number of statistical batches: %d", nbatch);
    omcLog(OMC_LOG_DETAIL, "Histories per batch: %d", nperbatch);

    /* Preparation of scoring struct */
    initScore(gridsize);

    #pragma omp parallel
    {
      /* Initialize random number generator */
      initRandom();

      /* Initialize particle stack */
      initStack();
    }

    unsigned long long started = 0;
    unsigned long long blocked = 0;
    int aborted = omcEngineRunBatches(source, modifier, nbatch, nperbatch,
                                      callbacks, &started, &blocked);

    if (!aborted && blocked > 0) {
        omcLog(OMC_LOG_DETAIL, "%llu of %d histories were stopped by the "
               "collimator.", blocked, nhist);
    }

    if (!aborted && started == 0) {
        omcLog(OMC_LOG_WARNING, "Not one history put a particle in the "
               "phantom. Check where the source sits relative to it%s.",
               blocked > 0 ? ", and whether the collimator is open at all"
                           : "");
    }
    else if (!aborted && started < (unsigned long long)nhist) {
        omcLog(OMC_LOG_DETAIL, "%llu of %d histories put a particle in the "
               "phantom.", started, nhist);
    }

    /* The fraction of the incident energy that stayed in the phantom, while
     the scoring arrays still hold energies rather than doses. Both sides are
     weighted -- the particle weight goes through to scoreSource() as well --
     so the batch scale above is all that separates them. */
    if (summary && !aborted) {
        double etot = 0.0;
        for (int irl = 1; irl < gridsize + 1; irl++) {
            etot += score.accum_endep[irl];
        }

        summary->nhist = nhist;
        summary->nperbatch = nperbatch;
        summary->started = started;
        summary->blocked = blocked;
        summary->energyFraction = score.ensrc > 0.0
            ? etot/(score.ensrc*source->batchScale)
            : 0.0;
    }

    if (!aborted) {
        omcScoreToCube(nbatch, source->incidentFluence, opt->outputDose, dose,
                       uncertainty);
    }

    cleanScore();

    //Cleaning private random generators and particle stack
    #pragma omp parallel
    {
      cleanRandom();
      cleanStack();
    }

    if (source->release != NULL) {
        source->release(source);
    }

    return !aborted;
}

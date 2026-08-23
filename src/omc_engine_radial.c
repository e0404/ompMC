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

#include "omc_engine_radial.h"

#include "omc_collimator.h"
#include "omc_engine_batches.h"
#include "omc_geom.h"
#include "omc_host.h"
#include "omc_random.h"
#include "omc_score.h"
#include "omc_source.h"
#include "ompmc.h"

#include <stddef.h>

int omcCalcRadial(const struct OmcRadialOptions *opt,
                  struct OmcSource *source,
                  const struct OmcBeamModifier *modifier,
                  double *dose, double *uncertainty,
                  const struct OmcForwardCallbacks *callbacks,
                  struct OmcForwardSummary *summary) {

    /* Asked first, because everything after it -- the region count, the ring
     masses, where a source particle enters -- reads a geometry that has to be
     the right shape for any of it to mean anything. */
    if (geometry.mode != OMC_GEOM_CYLINDRICAL) {
        omcFail("ompMC:radial:notCylindrical",
            "The radial engine scores in rings, and the phantom is a "
            "rectilinear voxel grid. Set the geometry up with "
            "omcGeomCylInit(), or use omcCalcForward() for a cube.");
    }

    if (opt->nbatch < 2) {
        /* The batch variance divides by nbatch - 1 */
        omcFail("ompMC:radial:tooFewBatches",
            "Number of batches is %d, at least 2 are needed for the "
            "uncertainty estimate.", opt->nbatch);
    }

    if (source->sample == NULL) {
        omcFail("ompMC:radial:noSource",
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

    /* jsize is 1 in this geometry, so this is the rings times the slabs --
     written the same way the cube engine writes it because it is the same
     region count, arrived at the same way. */
    int gridsize = geometry.isize*geometry.jsize*geometry.ksize;

    omcLog(OMC_LOG_DETAIL, "Total number of particle histories: %d", nhist);
    omcLog(OMC_LOG_DETAIL, "Number of statistical batches: %d", nbatch);
    omcLog(OMC_LOG_DETAIL, "Histories per batch: %d", nperbatch);
    omcLog(OMC_LOG_DETAIL, "Scoring %d rings by %d depth slabs",
           geometry.isize, geometry.ksize);

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
               "cylinder. Check where the source sits relative to it%s.",
               blocked > 0 ? ", and whether the collimator is open at all"
                           : "");
    }
    else if (!aborted && started < (unsigned long long)nhist) {
        omcLog(OMC_LOG_DETAIL, "%llu of %d histories put a particle in the "
               "cylinder.", started, nhist);
    }

    /* The fraction of the incident energy that stayed in the phantom, while
     the scoring arrays still hold energies rather than doses. */
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
        omcScoreToRadial(nbatch, source->incidentFluence, opt->outputDose,
                         dose, uncertainty);
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

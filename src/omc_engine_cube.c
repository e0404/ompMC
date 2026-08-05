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

#include "omc_engine_cube.h"

#include "omc_geom.h"
#include "omc_host.h"
#include "omc_random.h"
#include "omc_score.h"
#include "omc_spectrum.h"
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

/* What the current call is working on. Read-only once the histories start. */
static const struct OmcCubeOptions *options;
static const struct OmcSsdSource *source;
static const struct OmcSpectrum *spectrum;

/******************************************************************************/

void omcSsdSourceInit(struct OmcSsdSource *src) {

    /* Calculate x-direction input zones */
    if (src->xinl < geometry.xbounds[0]) {
        src->xinl = geometry.xbounds[0];
    }
    if (src->xinu <= src->xinl) {
        src->xinu = src->xinl;  /* default a pencil beam */
    }

    /* Check radiation field is not too big against the phantom */
    if (src->xinu > geometry.xbounds[geometry.isize]) {
        src->xinu = geometry.xbounds[geometry.isize];
    }
    if (src->xinl > geometry.xbounds[geometry.isize]) {
        src->xinl = geometry.xbounds[geometry.isize];
    }

    /* Now search for initial region x index range */
    omcLog(OMC_LOG_INFO, "Index ranges for radiation field:");
    src->ixinl = 0;
    while ((geometry.xbounds[src->ixinl] <= src->xinl) &&
           (geometry.xbounds[src->ixinl + 1] < src->xinl)) {
        src->ixinl++;
    }

    /* The upper index is never below the lower one, so start the search at
     ixinl. Starting one below it, as this used to, reads xbounds[-1] whenever
     the field begins at or before the first boundary -- which is the ordinary
     case, since xinl was just clamped up to xbounds[0]. For ixinl > 0 the two
     are equivalent: at ixinl - 1 both loop conditions hold by construction
     (xbounds[ixinl - 1] <= xinl <= xinu and xbounds[ixinl] < xinl <= xinu),
     so the first iteration only ever steps back up to ixinl. */
    src->ixinu = src->ixinl;
    while ((geometry.xbounds[src->ixinu] <= src->xinu) &&
           (geometry.xbounds[src->ixinu + 1] < src->xinu)) {
        src->ixinu++;
    }
    omcLog(OMC_LOG_INFO, "i index ranges over i = %d to %d",
           src->ixinl, src->ixinu);

    /* Calculate y-direction input zones */
    if (src->yinl < geometry.ybounds[0]) {
        src->yinl = geometry.ybounds[0];
    }
    if (src->yinu <= src->yinl) {
        src->yinu = src->yinl;  /* default a pencil beam */
    }

    /* Check radiation field is not too big against the phantom */
    if (src->yinu > geometry.ybounds[geometry.jsize]) {
        src->yinu = geometry.ybounds[geometry.jsize];
    }
    if (src->yinl > geometry.ybounds[geometry.jsize]) {
        src->yinl = geometry.ybounds[geometry.jsize];
    }

    /* Now search for initial region y index range */
    src->iyinl = 0;
    while ((geometry.ybounds[src->iyinl] <= src->yinl) &&
           (geometry.ybounds[src->iyinl + 1] < src->yinl)) {
        src->iyinl++;
    }
    /* Starts at iyinl for the same reason the x search above does */
    src->iyinu = src->iyinl;
    while ((geometry.ybounds[src->iyinu] <= src->yinu) &&
           (geometry.ybounds[src->iyinu + 1] < src->yinu)) {
        src->iyinu++;
    }
    omcLog(OMC_LOG_INFO, "j index ranges over j = %d to %d",
           src->iyinl, src->iyinu);

    /* Calculate collimator sizes */
    src->xsize = src->xinu - src->xinl;
    src->ysize = src->yinu - src->yinl;

    return;
}

/******************************************************************************/

static void initHistory(void) {

    /* Initialize first particle of the stack from source data */
    stack.np = 0;
    stack.p[stack.np].iq = options->charge;

    /* Get primary particle energy */
    double ein = omcSpectrumSample(spectrum);

    /* Check if the particle is an electron, in such a case add electron
     rest mass energy */
    if (stack.p[stack.np].iq != 0) {
        /* Electron or positron */
        stack.p[stack.np].e = ein + RM;
    }
    else {
        /* Photon */
        stack.p[stack.np].e = ein;
    }

    /* Accumulate sampled kinetic energy for fraction of deposited energy
     calculations. This runs inside the parallel history loop, so it has to
     go through scoreSource() rather than a bare += . */
    scoreSource(ein);

    /* Set particle position. First obtain a random position in the rectangle
     defined by the collimator */
    double rxyz = 0.0;
    if (source->xsize == 0.0 || source->ysize == 0.0) {
        stack.p[stack.np].x = source->xinl;
        stack.p[stack.np].y = source->yinl;

        rxyz = sqrt(pow(source->ssd, 2.0) + pow(stack.p[stack.np].x, 2.0) +
                    pow(stack.p[stack.np].y, 2.0));

        /* Get direction along z-axis */
        stack.p[stack.np].w = source->ssd/rxyz;

    } else {
        double fw;
        double rnno3;
        do { /* rejection sampling of the initial position */
            rnno3 = setRandom();
            stack.p[stack.np].x = rnno3*source->xsize + source->xinl;
            rnno3 = setRandom();
            stack.p[stack.np].y = rnno3*source->ysize + source->yinl;
            rnno3 = setRandom();
            rxyz = sqrt(source->ssd*source->ssd +
                stack.p[stack.np].x*stack.p[stack.np].x +
                stack.p[stack.np].y*stack.p[stack.np].y);

            /* Get direction along z-axis */
            stack.p[stack.np].w = source->ssd/rxyz;
            fw = stack.p[stack.np].w*stack.p[stack.np].w*stack.p[stack.np].w;
        } while(rnno3 >= fw);
    }
    /* Set position of the particle in front of the geometry */
    stack.p[stack.np].z = geometry.zbounds[0];

    /* At this point the position has been found, calculate particle
     direction */
    stack.p[stack.np].u = stack.p[stack.np].x/rxyz;
    stack.p[stack.np].v = stack.p[stack.np].y/rxyz;

    /* Determine region index of source particle */
    int ix, iy;
    if (source->xsize == 0.0) {
        ix = source->ixinl;
    } else {
        ix = source->ixinl - 1;
        while ((geometry.xbounds[ix+1] < stack.p[stack.np].x) && ix < geometry.isize-1) {
            ix++;
        }
    }
    if (source->ysize == 0.0) {
        iy = source->iyinl;
    } else {
        iy = source->iyinl - 1;
        while ((geometry.ybounds[iy+1] < stack.p[stack.np].y) && iy < geometry.jsize-1) {
            iy++;
        }
    }
    stack.p[stack.np].ir = 1 + ix + iy*geometry.isize;

    /* Set statistical weight and distance to closest boundary*/
    stack.p[stack.np].wt = 1.0;
    stack.p[stack.np].dnear = 0.0;

    return;
}

/******************************************************************************/

int omcCalcCube(const struct OmcCubeOptions *opt,
                const struct OmcSsdSource *src,
                const struct OmcSpectrum *spec,
                double *dose, double *uncertainty,
                const struct OmcCubeCallbacks *callbacks,
                struct OmcCubeSummary *summary) {

    if (opt->nbatch < 2) {
        /* The batch variance below divides by nbatch - 1 */
        omcFail("ompMC:cube:tooFewBatches",
            "Number of batches is %d, at least 2 are needed for the "
            "uncertainty estimate.", opt->nbatch);
    }

    options = opt;
    source = src;
    spectrum = spec;

    int nhist = opt->nhist;
    int nbatch = opt->nbatch;

    if (nhist/nbatch == 0) {
        nhist = nbatch;
    }

    int nperbatch = nhist/nbatch;
    nhist = nperbatch*nbatch;

    int gridsize = geometry.isize*geometry.jsize*geometry.ksize;

    omcLog(OMC_LOG_INFO, "Total number of particle histories: %d", nhist);
    omcLog(OMC_LOG_INFO, "Number of statistical batches: %d", nbatch);
    omcLog(OMC_LOG_INFO, "Histories per batch: %d", nperbatch);

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

    for (int ibatch=0; ibatch<nbatch; ibatch++) {
        if (callbacks && callbacks->batch &&
            !callbacks->batch(ibatch, nbatch,
                              (uint64_t)ibatch*(uint64_t)nperbatch,
                              callbacks->user)) {
            aborted = 1;
            break;
        }

        int ihist;
        #pragma omp parallel for schedule(dynamic)
        for (ihist=0; ihist<nperbatch; ihist++) {
            /* Point the RNG at this history's stream; the index is unique
             across batches, so results do not depend on the scheduling */
            setRandomHistory((uint64_t)ibatch*(uint64_t)nperbatch
                             + (uint64_t)ihist);

            /* Initialize particle history */
            initHistory();

            /* Start electromagnetic shower simulation */
            shower();
        }

        /* Accumulate results of current batch for statistical analysis */
        accumEndep(1.0);
    }

    /* The fraction of the incident energy that stayed in the phantom, while
     the scoring arrays still hold energies rather than doses */
    if (summary && !aborted) {
        double etot = 0.0;
        for (int irl=1; irl<gridsize+1; irl++) {
            etot += score.accum_endep[irl];
        }

        summary->nhist = nhist;
        summary->nperbatch = nperbatch;
        summary->energyFraction = etot/score.ensrc;
    }

    /* The normalization is per batch, not per run: each batch contributed
     nperbatch histories and the batches are averaged afterwards. */
    if (!aborted) {
        omcScoreToCube(nbatch, (double)nperbatch, options->outputDose,
                       dose, uncertainty);
    }

    cleanScore();

    //Cleaning private random generators and particle stack
    #pragma omp parallel
    {
      cleanRandom();
      cleanStack();
    }

    options = NULL;
    source = NULL;
    spectrum = NULL;

    return !aborted;
}

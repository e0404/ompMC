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

#include "omc_engine_dij.h"

#include "omc_geom.h"
#include "omc_host.h"
#include "omc_random.h"
#include "omc_score.h"
#include "omc_spectrum.h"
#include "omc_utilities.h"
#include "ompmc.h"

#include <float.h>
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

/* What the current call is working on. initHistory() runs once per history on
 every thread, so this is read-only for the duration of the call and set up
 before any parallel region starts. */
static const struct OmcDijOptions *options;
static const struct OmcBeamletSource *source;
static const struct OmcSpectrum *spectrum;

/******************************************************************************/

static void initHistory(int ibeamlet) {

    double rnno1;
    double rnno2;

    int ijmax = geometry.isize*geometry.jsize;
    int imax = geometry.isize;

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
     calculations */
    scoreSource(ein);

    /* Set particle position. First obtain a random position in the rectangle
     defined by the bixel at isocenter*/
    double xiso = 0.0;
    double yiso = 0.0;
    double ziso = 0.0;

    rnno1 = setRandom();
    rnno2 = setRandom();

    xiso = rnno1*source->xside1[ibeamlet] + rnno2*source->xside2[ibeamlet] +
            source->xcorner[ibeamlet];
    yiso = rnno1*source->yside1[ibeamlet] + rnno2*source->yside2[ibeamlet] +
            source->ycorner[ibeamlet];
    ziso = rnno1*source->zside1[ibeamlet] + rnno2*source->zside2[ibeamlet] +
            source->zcorner[ibeamlet];


    /* Norm of the resulting vector from the source of current beam to the
     position of the particle on bixel */
    int ibeam = source->ibeam[ibeamlet];

    double sourcePos[3];

    //Gaussian Source

    switch (options->sourceGeometry)
    {
        case OMC_SOURCE_POINT: ;
            sourcePos[0] = source->xsource[ibeam];
            sourcePos[1] = source->ysource[ibeam];
            sourcePos[2] = source->zsource[ibeam];
            break;
        case OMC_SOURCE_GAUSSIAN: ;
            //Get the normalized collimator plane vectors
            double planeVec1_norm;
            double planeVec2_norm;
            planeVec1_norm = sqrt(
                                            source->xside1[ibeamlet]*source->xside1[ibeamlet] +
                                            source->yside1[ibeamlet]*source->yside1[ibeamlet] +
                                            source->zside1[ibeamlet]*source->zside1[ibeamlet]
                                        );
            planeVec2_norm = sqrt(
                                            source->xside2[ibeamlet]*source->xside2[ibeamlet] +
                                            source->yside2[ibeamlet]*source->yside2[ibeamlet] +
                                            source->zside2[ibeamlet]*source->zside2[ibeamlet]
                                        );
            double planeVec1[3];
            planeVec1[0] = source->xside1[ibeamlet] / planeVec1_norm;
            planeVec1[1] = source->yside1[ibeamlet] / planeVec1_norm;
            planeVec1[2] = source->zside1[ibeamlet] / planeVec1_norm;

            double planeVec2[3];
            planeVec2[0] = source->xside2[ibeamlet] / planeVec2_norm;
            planeVec2[1] = source->yside2[ibeamlet] / planeVec2_norm;
            planeVec2[2] = source->zside2[ibeamlet] / planeVec2_norm;

            //Create two normally distributed random veriables with box-muller transform
            double rnSource[2];
            boxMuller(rnSource);

            //Scale with source width
            rnSource[0] *= options->sourceGaussianWidth;
            rnSource[1] *= options->sourceGaussianWidth;

            //Now use the plane vectors to add the random 2D offset to the source
            sourcePos[0] = source->xsource[ibeam] + rnSource[0]*planeVec1[0] + rnSource[1]*planeVec2[0];
            sourcePos[1] = source->ysource[ibeam] + rnSource[0]*planeVec1[1] + rnSource[1]*planeVec2[1];
            sourcePos[2] = source->zsource[ibeam] + rnSource[0]*planeVec1[2] + rnSource[1]*planeVec2[2];


            break;
        default: ;
            /* Checked before the parallel region starts, so this is only a
             backstop; omcFail() from a worker thread would call the host from
             a place the host cannot expect. */
            sourcePos[0] = source->xsource[ibeam];
            sourcePos[1] = source->ysource[ibeam];
            sourcePos[2] = source->zsource[ibeam];
    }


    //Point source
    double xd = xiso - sourcePos[0];
    double yd = yiso - sourcePos[1];
    double zd = ziso - sourcePos[2];


    double vnorm = sqrt(xd*xd + yd*yd + zd*zd);

    /* Direction of the particle from position on bixel to beam source*/
    double u = -(xd)/vnorm;
    double v = -(yd)/vnorm;
    double w = -(zd)/vnorm;

    /* Calculate the minimum distance from particle position on bixel to
     phantom boundaries */
    double ustep = DBL_MAX; //1.0E5;
    double dist;

    if(u > 0.0) {
        dist = (geometry.xbounds[geometry.isize]-xiso)/u;
        if(dist < ustep) {
            ustep = dist;
        }
    }
    if(u < 0.0) {
        dist = -(xiso-geometry.xbounds[0])/u;
        if(dist < ustep) {
            ustep = dist;
        }
    }

    if(v > 0.0) {
        dist = (geometry.ybounds[geometry.jsize]-yiso)/v;
        if(dist < ustep) {
            ustep = dist;
        }
    }
    if(v < 0.0) {
        dist = -(yiso-geometry.ybounds[0])/v;
        if(dist < ustep) {
            ustep = dist;
        }
    }

    if(w > 0.0) {
        dist = (geometry.zbounds[geometry.ksize]-ziso)/w;
        if(dist < ustep) {
            ustep = dist;
        }
    }
    if(w < 0.0) {
        dist = -(ziso-geometry.zbounds[0])/w;
        if(dist < ustep) {
            ustep = dist;
        }
    }

    /* Transport particle from bixel to surface. Adjust particle direction
     to be incident to phantom surface */
    stack.p[stack.np].x = xiso + ustep*u;
    stack.p[stack.np].y = yiso + ustep*v;
    stack.p[stack.np].z = ziso + ustep*w;

    stack.p[stack.np].u = -u;
    stack.p[stack.np].v = -v;
    stack.p[stack.np].w = -w;

    /* For numerical stability, make sure that points are really inside the
     phantom. nextafter() moves one representable step towards the opposite
     face; the 2.0*DBL_MIN offset used before is denormal-small and was
     absorbed entirely when added to any normal boundary coordinate, leaving
     the particle exactly on the boundary. */
    if(stack.p[stack.np].x < geometry.xbounds[0]) {
        stack.p[stack.np].x = nextafter(geometry.xbounds[0],
                                        geometry.xbounds[geometry.isize]);
    }
    if(stack.p[stack.np].x > geometry.xbounds[geometry.isize]) {
        stack.p[stack.np].x = nextafter(geometry.xbounds[geometry.isize],
                                        geometry.xbounds[0]);
    }

    if(stack.p[stack.np].y < geometry.ybounds[0]) {
        stack.p[stack.np].y = nextafter(geometry.ybounds[0],
                                        geometry.ybounds[geometry.jsize]);
    }
    if(stack.p[stack.np].y > geometry.ybounds[geometry.jsize]) {
        stack.p[stack.np].y = nextafter(geometry.ybounds[geometry.jsize],
                                        geometry.ybounds[0]);
    }

    if(stack.p[stack.np].z < geometry.zbounds[0]) {
        stack.p[stack.np].z = nextafter(geometry.zbounds[0],
                                        geometry.zbounds[geometry.ksize]);
    }
    if(stack.p[stack.np].z > geometry.zbounds[geometry.ksize]) {
        stack.p[stack.np].z = nextafter(geometry.zbounds[geometry.ksize],
                                        geometry.zbounds[0]);
    }

    /* Determine region index of source particle */
    int ix = omcFindVoxelIndex(geometry.xbounds, geometry.isize,
                               stack.p[stack.np].x);
    int iy = omcFindVoxelIndex(geometry.ybounds, geometry.jsize,
                               stack.p[stack.np].y);
    int iz = omcFindVoxelIndex(geometry.zbounds, geometry.ksize,
                               stack.p[stack.np].z);

    stack.p[stack.np].ir = 1 + ix + iy*imax + iz*ijmax;

    /* Set statistical weight and distance to closest boundary*/
    stack.p[stack.np].wt = 1.0;
    stack.p[stack.np].dnear = 0.0;

    return;
}

/******************************************************************************/
/* Turn what the batches accumulated into dose and its uncertainty, in place. */

static void accumulateResults(int nbatch) {

    /* Only voxels this beamlet actually deposited in can be nonzero, and for
     an untouched voxel the arithmetic below reduces to writing back the zeros
     that are already there: accum_endep is 0, so endep and endep2 come out 0,
     the endep != 0 branch is not taken, and both outputs are set to 0. Zeroing
     the dose in air is likewise a no-op on a voxel that never received any.
     So walking the touched set is equivalent to walking the grid, at a
     fraction of the cost for a single beamlet. */
    const int *touched;
    int ntouched = scoreBeamVoxels(&touched);

    /* MSVC only implements OpenMP 2.0, which in C does not allow declaring
     the loop variable inside the for statement */
    int n;
    #pragma omp parallel for
    for (n = 0; n < ntouched; n++) {
        int irl = touched[n];

        /* Region 0 is outside the geometry. ausgab() does reach it, through
         the discard path in electron(), but it has no voxel and is not part
         of the output. */
        if (irl == 0) {
            continue;
        }

        int ix, iy, iz;
        omcDecodeRegion(irl, geometry.isize, geometry.jsize, &ix, &iy, &iz);

        double endep = score.accum_endep[irl];
        double endep2 = score.accum_endep2[irl];

        /* Convert deposited energy to dose */
        double mass = (geometry.xbounds[ix+1] - geometry.xbounds[ix])*
            (geometry.ybounds[iy+1] - geometry.ybounds[iy])*
            (geometry.zbounds[iz+1] - geometry.zbounds[iz]);

        /* Transform deposited energy to Gy */
        mass *= geometry.med_densities[irl-1];

        double factor = 1.602E-10/(mass);

        endep *= factor;
        endep2 *= factor*factor;

        /* First calculate mean deposited energy across batches and its
         uncertainty */
        endep /= (double) nbatch;
        endep2 /= (double) nbatch;

        double unc_endep;

        /* Batch approach uncertainty calculation: sample variance of the
         batch means over (nbatch - 1) gives the variance of the mean. The
         divisors here must not be swapped -- dividing endep2 by (nbatch - 1)
         instead leaves a spurious mean^2/(nbatch*(nbatch - 1)) term that puts
         a floor of ~10% relative uncertainty under every voxel regardless of
         the statistics. */
        if (endep != 0.0) {
            unc_endep = endep2 - endep*endep;

            //Variance of the mean
            unc_endep /= (double) (nbatch - 1);
        }
        else {
            endep = 0.0;
            unc_endep = 0.0;
        }

        /* Zero dose in air */
        if (geometry.med_densities[irl-1] < 0.044) {
            endep = 0.0;
            unc_endep = 0.0;
        }

        /* Store output quantities */
        score.accum_endep[irl] = endep;
        score.accum_endep2[irl] = unc_endep;
    }

    return;
}

/******************************************************************************/

void omcCalcDij(const struct OmcDijOptions *opt,
                const struct OmcBeamletSource *src,
                const struct OmcSpectrum *spec,
                const struct OmcDijCallbacks *callbacks) {

    if (opt->sourceGeometry != OMC_SOURCE_POINT &&
        opt->sourceGeometry != OMC_SOURCE_GAUSSIAN) {
        omcFail("ompMC:dij:invalidSourceGeometry",
            "Source geometry %d is not defined.", (int)opt->sourceGeometry);
    }
    if (src->nbeamlets < 1) {
        omcFail("ompMC:dij:noBeamlets",
            "There are no beamlets to calculate.");
    }
    if (opt->nbatch < 2) {
        /* The batch variance below divides by nbatch - 1 */
        omcFail("ompMC:dij:tooFewBatches",
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

    omcLog(OMC_LOG_DETAIL, "Total number of particle histories: %d", nhist);
    omcLog(OMC_LOG_DETAIL, "Number of statistical batches: %d", nbatch);
    omcLog(OMC_LOG_DETAIL, "Histories per batch: %d", nperbatch);
    omcLog(OMC_LOG_DEBUG, "Using a relative dose cut-off of %f",
           opt->relDoseThreshold);

    /* Preparation of scoring struct */
    initScore(geometry.isize*geometry.jsize*geometry.ksize);

    #pragma omp parallel
    {
      /* Initialize random number generator */
      initRandom();

      /* Initialize particle stack */
      initStack();
    }

    /* Compacted results of one beamlet, handed to the callback. Grown as
     needed rather than sized to the grid, which would be a large allocation
     for the sake of a few thousand voxels. */
    int nscratch = 0;
    int *voxels = NULL;
    double *dose = NULL;
    double *variance = NULL;

    for(int ibeamlet=0; ibeamlet<src->nbeamlets; ibeamlet++) {
        for (int ibatch=0; ibatch<nbatch; ibatch++) {
            int ihist;

            #pragma omp parallel for schedule(guided)
            for (ihist=0; ihist<nperbatch; ihist++) {
                /* Point the RNG at this history's stream; the index is
                 unique across batches and beamlets, so results do not
                 depend on the scheduling */
                setRandomHistory(((uint64_t)ibeamlet*(uint64_t)nbatch
                                  + (uint64_t)ibatch)*(uint64_t)nperbatch
                                 + (uint64_t)ihist);

                /* Initialize particle history */
                initHistory(ibeamlet);

                /* Start electromagnetic shower simulation */
                shower();
            }

            /* Accumulate results of current batch for statistical analysis */
            accumEndep(1.0/(double)nperbatch);

            if (callbacks->progress) {
                callbacks->progress(((double)ibeamlet
                                     + (double)(ibatch+1)/nbatch)
                                    /src->nbeamlets, callbacks->user);
            }
        }

        /* Output of results for current beamlet */
        accumulateResults(nbatch);

        /* Everything from here to the end of the beamlet only ever looks at
         voxels this beamlet deposited in; the rest of the grid is zero and
         below any positive threshold. The list comes back ascending, which is
         what a sparse column needs. */
        const int *touched;
        int ntouched = scoreBeamVoxels(&touched);

        /* Get maximum value to apply threshold */
        double doseMax = 0.0;
        for (int n = 0; n < ntouched; n++) {
            int irl = touched[n];
            if (irl != 0 && score.accum_endep[irl] > doseMax) {
                doseMax = score.accum_endep[irl];
            }
        }
        double thresh = doseMax*opt->relDoseThreshold;

        if (ntouched > nscratch) {
            nscratch = ntouched;
            voxels = realloc(voxels, nscratch*sizeof(int));
            dose = realloc(dose, nscratch*sizeof(double));
            variance = realloc(variance, nscratch*sizeof(double));
        }

        int nvoxels = 0;
        for (int n = 0; n < ntouched; n++) {
            int irl = touched[n];
            if (irl != 0 && score.accum_endep[irl] > thresh) {
                /* -1 : the caller counts voxels from 0, the transport counts
                 regions from 1 with 0 for the world around the phantom */
                voxels[nvoxels] = irl - 1;
                dose[nvoxels] = score.accum_endep[irl];
                variance[nvoxels] = score.accum_endep2[irl];
                nvoxels++;
            }
        }

        callbacks->beamlet(ibeamlet, nvoxels, voxels, dose,
                           opt->wantVariance ? variance : NULL,
                           callbacks->user);

        /* Reset the accumulators for the following beamlet. This clears
         accum_endep2 as well, which the memset it replaces did not, so the
         variance of one beamlet no longer leaks into the next. */
        resetBeamScore();

        if (callbacks->progress) {
            callbacks->progress((double)(ibeamlet+1)/(double)src->nbeamlets,
                                callbacks->user);
        }
    }

    free(voxels);
    free(dose);
    free(variance);

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

    return;
}

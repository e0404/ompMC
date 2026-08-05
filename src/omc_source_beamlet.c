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

#include "omc_source_beamlet.h"

#include "omc_geom.h"
#include "omc_random.h"
#include "omc_score.h"
#include "omc_spectrum.h"
#include "omc_utilities.h"
#include "ompmc.h"

#include <float.h>
#include <math.h>

#if defined(_MSC_VER)
    //use __declspec(thread) instead of threadprivate to avoid
    //error C3053. More information in:
    // https://stackoverflow.com/questions/12560243/using-threadprivate-directive-in-visual-studio
    __declspec(thread) extern struct Stack stack;
#else
    extern struct Stack stack;
    #pragma omp threadprivate(stack)
#endif

/******************************************************************************/

void omcBeamletSample(const struct OmcBeamletSampler *sampler,
                      int ibeamlet, double weight) {

    double rnno1;
    double rnno2;

    int ijmax = geometry.isize*geometry.jsize;
    int imax = geometry.isize;

    /* Initialize first particle of the stack from source data */
    stack.np = 0;
    stack.p[stack.np].iq = sampler->charge;

    /* Get primary particle energy */
    double ein = omcSpectrumSample(sampler->spectrum);

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
    scoreSource(ein*weight);

    /* Set particle position. First obtain a random position in the rectangle
     defined by the bixel at isocenter*/
    double xiso = 0.0;
    double yiso = 0.0;
    double ziso = 0.0;

    rnno1 = setRandom();
    rnno2 = setRandom();

    xiso = rnno1*sampler->source->xside1[ibeamlet] + rnno2*sampler->source->xside2[ibeamlet] +
            sampler->source->xcorner[ibeamlet];
    yiso = rnno1*sampler->source->yside1[ibeamlet] + rnno2*sampler->source->yside2[ibeamlet] +
            sampler->source->ycorner[ibeamlet];
    ziso = rnno1*sampler->source->zside1[ibeamlet] + rnno2*sampler->source->zside2[ibeamlet] +
            sampler->source->zcorner[ibeamlet];


    /* Norm of the resulting vector from the source of current beam to the
     position of the particle on bixel */
    int ibeam = sampler->source->ibeam[ibeamlet];

    double sourcePos[3];

    //Gaussian Source

    switch (sampler->geometry)
    {
        case OMC_SOURCE_POINT: ;
            sourcePos[0] = sampler->source->xsource[ibeam];
            sourcePos[1] = sampler->source->ysource[ibeam];
            sourcePos[2] = sampler->source->zsource[ibeam];
            break;
        case OMC_SOURCE_GAUSSIAN: ;
            //Get the normalized collimator plane vectors
            double planeVec1_norm;
            double planeVec2_norm;
            planeVec1_norm = sqrt(
                                            sampler->source->xside1[ibeamlet]*sampler->source->xside1[ibeamlet] +
                                            sampler->source->yside1[ibeamlet]*sampler->source->yside1[ibeamlet] +
                                            sampler->source->zside1[ibeamlet]*sampler->source->zside1[ibeamlet]
                                        );
            planeVec2_norm = sqrt(
                                            sampler->source->xside2[ibeamlet]*sampler->source->xside2[ibeamlet] +
                                            sampler->source->yside2[ibeamlet]*sampler->source->yside2[ibeamlet] +
                                            sampler->source->zside2[ibeamlet]*sampler->source->zside2[ibeamlet]
                                        );
            double planeVec1[3];
            planeVec1[0] = sampler->source->xside1[ibeamlet] / planeVec1_norm;
            planeVec1[1] = sampler->source->yside1[ibeamlet] / planeVec1_norm;
            planeVec1[2] = sampler->source->zside1[ibeamlet] / planeVec1_norm;

            double planeVec2[3];
            planeVec2[0] = sampler->source->xside2[ibeamlet] / planeVec2_norm;
            planeVec2[1] = sampler->source->yside2[ibeamlet] / planeVec2_norm;
            planeVec2[2] = sampler->source->zside2[ibeamlet] / planeVec2_norm;

            //Create two normally distributed random veriables with box-muller transform
            double rnSource[2];
            boxMuller(rnSource);

            //Scale with source width
            rnSource[0] *= sampler->gaussianWidth;
            rnSource[1] *= sampler->gaussianWidth;

            //Now use the plane vectors to add the random 2D offset to the source
            sourcePos[0] = sampler->source->xsource[ibeam] + rnSource[0]*planeVec1[0] + rnSource[1]*planeVec2[0];
            sourcePos[1] = sampler->source->ysource[ibeam] + rnSource[0]*planeVec1[1] + rnSource[1]*planeVec2[1];
            sourcePos[2] = sampler->source->zsource[ibeam] + rnSource[0]*planeVec1[2] + rnSource[1]*planeVec2[2];


            break;
        default: ;
            /* Checked before the parallel region starts, so this is only a
             backstop; omcFail() from a worker thread would call the host from
             a place the host cannot expect. */
            sourcePos[0] = sampler->source->xsource[ibeam];
            sourcePos[1] = sampler->source->ysource[ibeam];
            sourcePos[2] = sampler->source->zsource[ibeam];
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
    stack.p[stack.np].wt = weight;
    stack.p[stack.np].dnear = 0.0;

    return;
}

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

/*!
 @file
 omc_engine_cube - Dose in every voxel from one collimated beam.

 This is what omc_dosxyz calculates: a point source at a given distance from
 the phantom surface, collimated to a rectangle on that surface, and the dose
 it deposits everywhere in the phantom. Where omc_engine_dij.h produces one
 sparse column per beamlet, this produces one dense cube.

 The two engines share everything below the source: the same phantom
 (omc_geom.h), the same spectrum (omc_spectrum.h), the same transport. They
 differ in where particles start and in how the result comes back.

 Before calling omcCalcCube() the host must have

   1. filled struct Geom and called initRegions()   (omc_geom.h)
   2. called initMediaData() and initVrt()          (ompmc.h)
   3. built the source spectrum                     (omc_spectrum.h)
   4. called omcSsdSourceInit() on the source below

 and afterwards it owns the cleanup of those.

 @warning Like the rest of ompMC this is a singleton: one calculation at a
 time per process.
*****************************************************************************/

#ifndef OMC_ENGINE_CUBE_H
#define OMC_ENGINE_CUBE_H

#include <stdint.h>

struct OmcSpectrum;

/*! A point source at distance ssd in front of the phantom, shining through a
 rectangular collimator opening on the phantom surface. The host fills the
 first five members; omcSsdSourceInit() clamps the rectangle to the phantom
 and works out the rest. */
struct OmcSsdSource {
    double ssd;                 ///< distance of point source to phantom surface

    double xinl;                 ///< lower x-bound of the field on the phantom surface
    double xinu;                 ///< upper x-bound of the field on the phantom surface
    double yinl;                 ///< lower y-bound of the field on the phantom surface
    double yinu;                 ///< upper y-bound of the field on the phantom surface

    /* Derived by omcSsdSourceInit() */
    double xsize;                ///< x-width of the collimated field
    double ysize;                ///< y-width of the collimated field
    int ixinl;                    ///< voxel index of the field's lower x-bound
    int ixinu;                    ///< voxel index of the field's upper x-bound
    int iyinl;                    ///< voxel index of the field's lower y-bound
    int iyinu;                    ///< voxel index of the field's upper y-bound
};

/*! Clamp the collimator rectangle to the phantom and find the voxel indices it
 covers. A rectangle of zero width in a direction is a pencil beam there.

 @param source The x/y bounds and ssd must already be filled in; the derived
 fields are written by this call. */
void omcSsdSourceInit(struct OmcSsdSource *source);

/*! Run parameters for one cube calculation. */
struct OmcCubeOptions {
    int nhist;                  ///< total histories
    int nbatch;                 ///< statistical batches to split them into
    int charge;                 ///< 0 : photons, -1 : electrons, +1 : positrons

    int outputDose;             ///< 1 : dose in Gy per incident fluence, 0 : mean deposited energy
};

/*! Callbacks omcCalcCube() reports progress through. */
struct OmcCubeCallbacks {
    /*! About to start a batch. Optional. Called on the master thread.

     @param ibatch Batch index, counting from 0.
     @param nbatch Total number of batches.
     @param firstHistory Global history index of the batch's first history.
     @param user The pointer from struct OmcCubeCallbacks::user, untouched.
     @return 0 to abandon the calculation: it stops before that batch, tears
     its state down and returns 0 without touching dose[] or uncertainty[].
     There is no partial result to keep -- the batches are averaged, so a run
     that stopped halfway would be a dose with no meaning. Return nonzero to
     carry on. */
    int (*batch)(int ibatch, int nbatch, uint64_t firstHistory, void *user);

    void *user;                 ///< passed back to the callback, untouched
};

/*! What the run did, for hosts that want to report it. Optional. */
struct OmcCubeSummary {
    int nhist;                  ///< histories actually run, rounded to whole batches
    int nperbatch;               ///< histories per batch
    double energyFraction;      ///< deposited energy over incident kinetic energy
};

/*! Transport the histories and write the results into dose[] and, unless it
 is NULL, uncertainty[]. Both are supplied by the caller and hold one entry
 per voxel, indexed like the phantom: `ix + iy*isize + iz*isize*jsize`.

 @param options Run parameters.
 @param source The collimated point source, already passed through
 omcSsdSourceInit().
 @param spectrum Source energy spectrum.
 @param dose Caller-supplied array of `isize*jsize*ksize` entries.
 @param uncertainty Caller-supplied array of the same size, or `NULL`. Holds
 the RELATIVE uncertainty of the dose in that voxel, and is 0.9999999
 wherever nothing was deposited -- the convention the .3ddose format
 expects.
 @param callbacks Progress reporting; see struct OmcCubeCallbacks.
 @param summary Optional; filled in with what the run did.
 @return Nonzero when the run finished, 0 when the batch callback stopped
 it. */
int omcCalcCube(const struct OmcCubeOptions *options,
                const struct OmcSsdSource *source,
                const struct OmcSpectrum *spectrum,
                double *dose, double *uncertainty,
                const struct OmcCubeCallbacks *callbacks,
                struct OmcCubeSummary *summary);

#endif

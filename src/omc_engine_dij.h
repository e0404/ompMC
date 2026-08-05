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
 omc_engine_dij - Dose influence matrix for a set of beamlets.

 This is what the matRad interface calculates: each beamlet is an aperture
 rectangle at isocentre, particles start on it and fly away from the source
 point of the beam it belongs to, and the dose they deposit becomes one column
 of a sparse matrix.

 The engine takes plain C arrays and reports its results through callbacks, so
 that the host -- a MEX file, a Python extension -- decides how the columns are
 stored without the engine knowing anything about mxArray or numpy. Before
 calling omcCalcDij() the host must have

   1. filled struct Geom and called initRegions()   (omc_geom.h)
   2. called initMediaData() and initVrt()          (ompmc.h)
   3. built the source spectrum                     (omc_spectrum.h)

 and afterwards it owns the cleanup of all three. The engine itself owns only
 the scoring arrays, the random number generators and the particle stacks,
 which it sets up and tears down per call.

 @warning Like the rest of ompMC this is a singleton: one calculation at a
 time per process, since the transport state lives in globals.
*****************************************************************************/

#ifndef OMC_ENGINE_DIJ_H
#define OMC_ENGINE_DIJ_H

/* enum OmcSourceGeometry and struct OmcBeamletSource live here: they describe
 the source omc_engine_forward.h shares with this one. Included rather than
 forward declared so that a host which only knows about the Dij engine keeps
 compiling unchanged. */
#include "omc_source_beamlet.h"

struct OmcSpectrum;

/*! Run parameters for one Dij calculation. */
struct OmcDijOptions {
    int nhist;                  ///< total histories per beamlet
    int nbatch;                 ///< statistical batches to split them into
    int charge;                 ///< 0 : photons, -1 : electrons, +1 : positrons

    /*! Voxels below this fraction of the beamlet's maximum dose are dropped
     from the column rather than reported. */
    double relDoseThreshold;

    enum OmcSourceGeometry sourceGeometry;   ///< POINT or GAUSSIAN, see omc_source_beamlet.h
    double sourceGaussianWidth; ///< standard deviation in cm, GAUSSIAN only

    int wantVariance;           ///< also report the variance of the mean
};

/*! Callbacks omcCalcDij() reports results and progress through. */
struct OmcDijCallbacks {
    /*! One finished beamlet.

     @param ibeamlet Index of the beamlet just finished.
     @param nvoxels Number of entries in @p voxels, @p dose and @p variance.
     @param voxels @p nvoxels grid indices in ascending order, 0 based.
     @param dose Dose in Gy, one entry per voxel in @p voxels.
     @param variance Variance of the mean, one entry per voxel in @p voxels,
     or `NULL` if it was not asked for.
     @param user The pointer from struct OmcDijCallbacks::user, untouched.

     All three arrays belong to the engine and are only valid for the
     duration of the call. Called on the master thread, outside any parallel
     region. */
    void (*beamlet)(int ibeamlet, int nvoxels, const int *voxels,
                    const double *dose, const double *variance, void *user);

    /*! Progress report, called once per batch and once per beamlet, also on
     the master thread.

     @param fraction Fraction of the whole calculation finished, in [0,1].
     @param user The pointer from struct OmcDijCallbacks::user, untouched.
     @return 0 to abandon the calculation. It stops after the current batch,
     tears its state down and returns normally, having reported fewer
     beamlets than were asked for -- omcCalcDij() tells the caller how many
     through its return value, and anything already handed to beamlet() stays
     valid. Return nonzero to carry on.

     Optional: pass `NULL` to skip progress reporting. */
    int (*progress)(double fraction, void *user);

    void *user;                 ///< passed back to both callbacks, untouched
};

/*! Run a Dij calculation.

 @param options Run parameters.
 @param source The beamlets to calculate a column for.
 @param spectrum Source energy spectrum.
 @param callbacks Where the results and progress go; see struct
 OmcDijCallbacks.
 @return The number of beamlets reported through the beamlet callback, which
 is `source->nbeamlets` unless the progress callback asked to stop early. */
int omcCalcDij(const struct OmcDijOptions *options,
               const struct OmcBeamletSource *source,
               const struct OmcSpectrum *spectrum,
               const struct OmcDijCallbacks *callbacks);

#endif

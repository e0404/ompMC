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
 omc_engine_forward - Dose from a whole weighted set of beamlets, in one cube.

 Forward dose for a fluence map. The beamlets and their geometry are the same
 ones omc_engine_dij.h takes; what is added is one weight per beamlet, which
 is where the collimation comes in -- a closed leaf is a beamlet of weight
 zero, a partly transmitting one a beamlet of reduced weight. The result is
 what dij*weights would have been, computed directly:

     omcCalcDij     nhist histories per beamlet, a sparse column each,
                    weights applied afterwards by the caller
     omcCalcForward nhist histories in total, spread over the beamlets in
                    proportion to their weight, one dense cube

 so the cost no longer grows with the number of beamlets, and a beamlet that
 is closed costs nothing at all.

 @warning The weights modulate FLUENCE, not spectrum: a weight of 0.02 for a
 leaf that transmits 2% starts 2% of the particles it would otherwise have
 started, with the unhardened spectrum. Attenuation through the collimator,
 its scatter and the beam hardening that comes with it are not modelled.

 Before calling omcCalcForward() the host must have

   1. filled struct Geom and called initRegions()   (omc_geom.h)
   2. called initMediaData() and initVrt()          (ompmc.h)
   3. built the source spectrum                     (omc_spectrum.h)

 and afterwards it owns the cleanup of those.

 @warning Like the rest of ompMC this is a singleton: one calculation at a
 time per process.
*****************************************************************************/

#ifndef OMC_ENGINE_FORWARD_H
#define OMC_ENGINE_FORWARD_H

#include "omc_source_beamlet.h"

struct OmcSpectrum;

/*! Run parameters for one forward calculation. */
struct OmcForwardOptions {
    int nhist;                  ///< total histories, over all beamlets together
    int nbatch;                 ///< statistical batches to split them into
    int charge;                 ///< 0 : photons, -1 : electrons, +1 : positrons

    enum OmcSourceGeometry sourceGeometry;   ///< POINT or GAUSSIAN, see omc_source_beamlet.h
    double sourceGaussianWidth; ///< standard deviation in cm, GAUSSIAN only

    int outputDose;              ///< 1 : dose in Gy for the weights given, 0 : mean deposited energy
};

/*! Callbacks omcCalcForward() reports progress through. */
struct OmcForwardCallbacks {
    /*! Progress report, called once per batch on the master thread, outside
     any parallel region.

     @param fraction Fraction of the calculation finished, in [0,1].
     @param user The pointer from struct OmcForwardCallbacks::user, untouched.
     @return 0 to abandon the calculation. It stops after the current batch
     and returns 0 without touching dose[] or uncertainty[] -- the batches
     are averaged, so a run that stopped halfway would be a dose with no
     meaning. Return nonzero to carry on.

     Optional: pass `NULL` to skip progress reporting. */
    int (*progress)(double fraction, void *user);

    void *user;                 ///< passed back to the callback, untouched
};

/*! What the run did, for hosts that want to report it. Optional. */
struct OmcForwardSummary {
    int nhist;                  ///< histories actually run, rounded to whole batches
    int nperbatch;              ///< histories per batch

    int nweighted;              ///< beamlets with a weight above zero
    int nsampled;                ///< of those, the ones that got any histories

    double totalWeight;         ///< sum of the weights asked for
    double sampledWeight;       ///< sum over the beamlets that got histories

    double energyFraction;      ///< deposited energy over incident kinetic energy
};

/*! Transport the histories and write the results into dose[] and, unless it
 is NULL, uncertainty[]. Both are supplied by the caller and hold one entry
 per voxel, indexed like the phantom: `ix + iy*isize + iz*isize*jsize`.

 @param options Run parameters.
 @param source The beamlets, with the geometry of struct OmcBeamletSource.
 @param weights One finite, non-negative value per beamlet, and their sum
 must also be finite. Its scale carries through to the result: doubling
 every weight doubles the dose, and the dose returned is the dose for
 exactly these weights, so that it can be held against `dij*weights`.
 Beamlets are given histories in proportion to their weight, and one whose
 share rounds to zero histories contributes nothing -- the summary reports
 how much weight that was, and a warning goes to the host when it is more
 than a thousandth of the total.
 @param spectrum Source energy spectrum.
 @param dose Caller-supplied array of `isize*jsize*ksize` entries.
 @param uncertainty Caller-supplied array of the same size, or `NULL`. Holds
 the RELATIVE uncertainty of the dose in that voxel, and is 0.9999999
 wherever nothing was deposited, the same convention omcCalcCube() follows.
 @param callbacks Progress reporting; see struct OmcForwardCallbacks.
 @param summary Optional; filled in with what the run did.
 @return Nonzero when the run finished, 0 when the progress callback stopped
 it. */
int omcCalcForward(const struct OmcForwardOptions *options,
                   const struct OmcBeamletSource *source,
                   const double *weights,
                   const struct OmcSpectrum *spectrum,
                   double *dose, double *uncertainty,
                   const struct OmcForwardCallbacks *callbacks,
                   struct OmcForwardSummary *summary);

#endif

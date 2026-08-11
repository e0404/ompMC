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
 omc_engine_forward - Dose from a whole source at once, in one cube.

 One run, one dense dose cube, whatever the particles come from: weighted
 beamlets making up a fluence map (omc_source_beamlet.h), a phase space
 recorded by an earlier simulation (omc_source_phsp.h), or anything else that
 fills in a struct OmcSource. Held against the other engine that takes
 beamlets:

     omcCalcDij     nhist histories per beamlet, a sparse column each,
                    weights applied afterwards by the caller
     omcCalcForward nhist histories in total over the whole source, one dense
                    cube

 so with beamlets the cost no longer grows with how many there are, and one
 that is closed costs nothing at all.

 What the result MEANS is the source's business too, and it says so through
 struct OmcSource::batchScale and struct OmcSource::incidentFluence. Weighted
 beamlets give the dose for exactly the weights they were handed, which is
 what makes it comparable with `dij*weights`. A phase space gives the dose per
 history, since a file does not come with a fluence.

 @warning With beamlets, the weights modulate FLUENCE, not spectrum: a weight
 of 0.02 for a leaf that transmits 2% starts 2% of the particles it would
 otherwise have started, with the unhardened spectrum. Attenuation through the
 collimator, its scatter and the beam hardening that comes with it are not
 modelled.

 Before calling omcCalcForward() the host must have

   1. filled struct Geom and called initRegions()   (omc_geom.h)
   2. called initMediaData() and initVrt()          (ompmc.h)
   3. set up the source, including its spectrum if it has one

 and afterwards it owns the cleanup of those.

 @warning Like the rest of ompMC this is a singleton: one calculation at a
 time per process.
*****************************************************************************/

#ifndef OMC_ENGINE_FORWARD_H
#define OMC_ENGINE_FORWARD_H

#include "omc_source.h"

struct OmcSpectrum;

/*! Run parameters for one forward calculation. What the particles ARE is the
 source's business, not this struct's. */
struct OmcForwardOptions {
    int nhist;                  ///< total histories, over the whole calculation
    int nbatch;                 ///< statistical batches to split them into

    int outputDose;              ///< 1 : dose in Gy, 0 : mean deposited energy
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

/*! What the run did, for hosts that want to report it. Optional.

 Anything specific to a kind of source -- how the histories were shared out
 among beamlets, say -- belongs to that source and is asked of it, e.g.
 through omcBeamletHistoriesStats(). */
struct OmcForwardSummary {
    int nhist;                  ///< histories actually run, rounded to whole batches
    int nperbatch;              ///< histories per batch

    /*! Histories that put a particle in the phantom. The rest drew one
     pointing somewhere else, or one ompMC does not transport. */
    unsigned long long started;

    double energyFraction;      ///< deposited energy over incident kinetic energy
};

/*! Transport the histories and write the results into dose[] and, unless it
 is NULL, uncertainty[]. Both are supplied by the caller and hold one entry
 per voxel, indexed like the phantom: `ix + iy*isize + iz*isize*jsize`.

 @param options Run parameters.
 @param source Where the particles come from. Checked over before the
 histories start, prepared with the batch size, and released afterwards, so
 the same source struct can be run again but must outlive the call.
 @param dose Caller-supplied array of `isize*jsize*ksize` entries.
 @param uncertainty Caller-supplied array of the same size, or `NULL`. Holds
 the RELATIVE uncertainty of the dose in that voxel, and is 0.9999999
 wherever nothing was deposited, the same convention omcCalcCube() follows.
 @param callbacks Progress reporting; see struct OmcForwardCallbacks.
 @param summary Optional; filled in with what the run did.
 @return Nonzero when the run finished, 0 when the progress callback stopped
 it.

 @warning Histories whose particle never reaches the phantom still count
 among the ones the result is divided by. They are fluence the source stands
 for that happened to miss, and leaving them out would scale the answer up by
 however much of the beam does. struct OmcForwardSummary::started says how
 many of them there were. */
int omcCalcForward(const struct OmcForwardOptions *options,
                   struct OmcSource *source,
                   double *dose, double *uncertainty,
                   const struct OmcForwardCallbacks *callbacks,
                   struct OmcForwardSummary *summary);

#endif

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
 omc_source - What every source of primary particles has in common.

 A source's job is to answer one question per history: which particle starts
 it, and where is it going. Everything after that -- carrying the particle to
 the phantom, working out which voxel it lands in, putting it on the stack,
 counting the energy it brought -- is the same whatever the source is, and is
 omcSourcePlace()'s job rather than each source's.

 That split is what lets an engine take any source at all. A source fills in
 a struct OmcSourceParticle and says nothing about phantoms; an engine calls
 sample() and then omcSourcePlace() and knows nothing about beamlets or phase
 spaces.

     struct OmcSourceParticle particle;

     if (source->sample(source, ihist, ihistInBatch, &particle) &&
         omcSourcePlace(&particle)) {
         scoreSource(particle.energy*particle.weight);
         shower();
     }

 @warning A source emits from where it IS, upstream of the phantom, and
 omcSourcePlace() flies the particle forward from there. Do not hand it a
 particle already sitting on the phantom surface unless that is genuinely
 where the source put it: one that starts inside the phantom starts inside
 it, and skips the build up region it should have travelled through.
*****************************************************************************/

#ifndef OMC_SOURCE_H
#define OMC_SOURCE_H

#include <stdint.h>

/*! A primary particle, as the source made it: in the phantom's coordinate
 system, but not yet anywhere near the phantom. */
struct OmcSourceParticle {
    int charge;                 ///< 0 : photon, -1 : electron, +1 : positron
    double energy;              ///< KINETIC energy in MeV, without the rest mass

    double x, y, z;             ///< where it starts, in cm
    double u, v, w;             ///< where it is going, a unit vector

    double weight;              ///< statistical weight
};

struct OmcSource;

/*! A source of primary particles, as an engine sees it.

 Only #sample is required. The rest may be left NULL by a source that has
 nothing to check, nothing to set up per run and nothing to release. */
struct OmcSource {

    /*! Look the source over and report anything wrong with it through
     omcFail(). Called once, on the master thread, before the histories
     start -- which is the only place a source may fail from, since
     omcFail() from a worker thread would call the host somewhere it cannot
     expect to be called.

     @param self This source. */
    void (*check)(const struct OmcSource *self);

    /*! Get ready for a run of @p nperbatch histories per batch, and say what
     one history is worth by filling in #batchScale and #incidentFluence.

     @param self This source.
     @param nperbatch Histories in each batch. */
    void (*prepare)(struct OmcSource *self, int nperbatch);

    /*! Make the particle that starts one history.

     @param self This source.
     @param ihist Global history index, unique over the whole run. What a
     source draws must depend on this and on nothing else, or the answer
     starts depending on how the histories were scheduled.
     @param ihistInBatch Index within the batch, for sources that share their
     histories out among sub-sources.
     @param particle Filled in with the particle.
     @return 1 if there is a particle, 0 if this history is empty -- which is
     a thing that happens, and is not an error.

     @warning Runs inside the parallel history loop, so it may touch nothing
     but the thread's own random number generator and this read-only
     source. */
    int (*sample)(const struct OmcSource *self, uint64_t ihist,
                  int ihistInBatch, struct OmcSourceParticle *particle);

    /*! Give back whatever prepare() took.

     @param self This source. */
    void (*release)(struct OmcSource *self);

    void *impl;                 ///< the concrete source, for its own methods

    /*! What a batch's deposits are multiplied by before being accumulated.
     Set by prepare(). A source whose particles already carry the fluence the
     caller asked for puts the batch's share of it here; one that reports per
     history leaves it at 1. */
    double batchScale;

    /*! What the accumulated energy is divided by at the end. Set by
     prepare(). A source reporting per history puts the history count here;
     one whose weights already carry the fluence leaves it at 1. */
    double incidentFluence;
};

/*! Carry a particle to the phantom and put it on the (thread local) stack.

 The particle flies from where the source put it until it meets the phantom.
 One that starts inside it starts where it is.

 @param particle The particle, as the source made it.
 @return 1 if it is on the stack, 0 if its ray never reaches the phantom at
 all, in which case the stack is untouched and the history is over.

 @warning Runs inside the parallel history loop and writes the thread's own
 stack. */
int omcSourcePlace(const struct OmcSourceParticle *particle);

#endif

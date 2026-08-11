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
 omc_source_phsp - Starting histories from a phase space file.

 omc_phsp reads the particles; this makes one of them the primary of a
 history, moved into the phantom's coordinate system. Where it meets the
 phantom is omc_source.h's business, as it is for every source, so what an
 engine needs is the struct OmcSource that omcPhspSamplerAsSource() fills in:

     struct OmcPhspSampler sampler = {0};
     sampler.phsp = &phsp;
     sampler.order = OMC_PHSP_REPLAY;
     omcPhspTransformIdentity(&sampler.transform);

     struct OmcSource source;
     omcPhspSamplerAsSource(&sampler, &source);

     omcCalcForward(&options, &source, dose, uncertainty, NULL, &summary);

 A phase space particle differs from a beamlet one in two ways worth knowing.
 It is recorded wherever the original simulation scored it, which need not be
 aimed at the phantom at all, so plenty of histories produce nothing. And it
 can be a neutron or a proton, which ompMC does not transport; those produce
 nothing either.

 @warning ONE PARTICLE PER HISTORY. A phase space records which particles a
 single original history left behind, and those are correlated -- a
 bremsstrahlung photon and the electron that made it land in the same place
 more often than two unrelated particles do. Drawing them one at a time, as
 this does, still gets the dose right on average, but a run's own estimate of
 its uncertainty comes out too small by however much they are correlated.
 Grouping them is the obvious next step and needs a file that says where its
 histories begin, which struct OmcPhsp::newHistories reports and the phase
 spaces published by the IAEA do not all do.
*****************************************************************************/

#ifndef OMC_SOURCE_PHSP_H
#define OMC_SOURCE_PHSP_H

#include "omc_source.h"

#include <stdint.h>

struct OmcPhsp;

/*! Which particle of the file a history gets. */
enum OmcPhspOrder {
    /*! Particle `first + ihist`, wrapping at the end of the file. Draws no
     random numbers at all, and every particle gets used as often as every
     other one. */
    OMC_PHSP_REPLAY = 0,

    /*! A particle picked at random, which costs one random number per
     history. Worth it when a run is much shorter than the file and taking a
     contiguous run of it would sample one part of the beam. */
    OMC_PHSP_RANDOM = 1
};

/*! Where the phase space sits in the phantom's world: the particles are
 turned by #rotation and then moved by #translation, positions and directions
 alike.

 A phase space is recorded in the coordinate system of the machine that made
 it -- for the IAEA sets, with z along the beam and the origin in the target
 -- and the phantom has its own. This is what carries one to the other, and
 omcPhspTransformIdentity() is the do-nothing case for a phase space already
 in the right place. */
struct OmcPhspTransform {
    double rotation[9];         ///< row major 3x3, applied first
    double translation[3];      ///< added afterwards, in cm
};

/*! Everything the sampling needs that does not change from history to
 history. An engine fills this once, before its first parallel region, and
 hands it to omcPhspProduce() unchanged from then on. */
struct OmcPhspSampler {
    const struct OmcPhsp *phsp;         ///< the particles, already read
    enum OmcPhspOrder order;            ///< REPLAY or RANDOM
    unsigned long long first;           ///< first particle to replay, REPLAY only
    struct OmcPhspTransform transform;  ///< phase space to phantom
};

/*! Fill in the transform that changes nothing.

 @param transform Set to the identity rotation and no translation. */
void omcPhspTransformIdentity(struct OmcPhspTransform *transform);

/*! Check a sampler over before the histories start.

 Everything omcPhspProduce() would have to complain about is settled
 here instead, because it runs on worker threads, where omcFail() would call
 back into a host that has no business being entered from one.

 @param sampler The sampler to check.

 @pre The geometry (omc_geom.h) is set up.
 @warning Call from the master thread, before the parallel region. */
void omcPhspSourceCheck(const struct OmcPhspSampler *sampler);

/*! Make the primary particle history #ihist draws, moved into the phantom's
 coordinate system.

 @param sampler Sampling parameters, unchanged since the caller filled them.
 @param ihist Global history index, the same one setRandomHistory() was
 given. Which particle it draws depends only on this, never on which thread
 ran it or on what ran before it, so a run gives the same answer however the
 histories are scheduled.
 @param weight Scales the weight the particle carries in the file. Pass 1.0
 to use the file's own weights.
 @param particle Filled in with the particle.

 @return 1 if there is a particle, 0 if this history produced nothing --
 which here means the file offered a neutron or a proton, and ompMC
 transports neither. A history that produced nothing still happened, and
 still counts towards the fluence a run represents.

 @warning Runs inside the parallel history loop, so it touches nothing but
 its own random number generator and the read-only sampler. In particular it
 leaves the read position omcPhspNext() uses alone: that is one cursor shared
 by every thread, and a source that moved it would make the answer depend on
 the scheduling. */
int omcPhspProduce(const struct OmcPhspSampler *sampler, uint64_t ihist,
                   double weight, struct OmcSourceParticle *particle);

/*! Present a phase space to an engine as a source.

 @param sampler The phase space and how to draw from it. Must outlive
 @p source.
 @param source Filled in with the source interface. */
void omcPhspSamplerAsSource(struct OmcPhspSampler *sampler,
                            struct OmcSource *source);

#endif

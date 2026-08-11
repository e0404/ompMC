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
 omc_collimator - Shaping a beam after the source has made it.

 A source says what particles there are; a modifier says what happens to them
 on the way to the phantom. The two are separate because they compose: any
 source can be run through any modifier, which is what makes it possible to
 cut a field out of a phase space recorded above the jaws -- and the phase
 spaces published by the IAEA are recorded exactly there, field independent
 on purpose.

 It works by back projection. A collimator sits on a plane the particle
 crossed in a straight line, so where it crossed follows from where the
 particle is and where it is going, whichever side of the plane the source
 happened to put it on. The modifier therefore needs nothing from the source
 and nothing from the phantom.

 What comes back is the fraction of the particle's weight that gets through:
 1 for an open aperture, 0 for a blocked one, and something between for a
 leaf that transmits.

 @warning This is a MASK, not a collimator. It attenuates and it blocks; it
 does not scatter, and it does not harden the spectrum of what it lets
 through. A particle that would have scattered off a leaf edge into the field
 is simply gone. That is the same simplification the beamlet weights make in
 omc_engine_forward.h, and it is a good one for the fluence and a poor one
 for the penumbra.

 @warning Weight, not roulette: a leaf transmitting 2% gives a particle 2% of
 its weight and then transports it in full, so the cost of a nearly closed
 leaf is the same as an open one for a fiftieth of the dose. Playing Russian
 roulette instead would be cheaper and is the obvious next step -- at the
 price of a random number per history, which would shift every stream and
 stop a collimated run being comparable, history by history, with the open
 one it came from. Nothing here draws a random number.
*****************************************************************************/

#ifndef OMC_COLLIMATOR_H
#define OMC_COLLIMATOR_H

struct OmcSourceParticle;
struct OmcBeamModifier;

/*! Something in the beam's way. */
struct OmcBeamModifier {

    /*! Look the modifier over and report anything wrong with it through
     omcFail(). Called once, on the master thread, before the histories
     start. May be NULL.

     @param self This modifier. */
    void (*check)(const struct OmcBeamModifier *self);

    /*! How much of this particle gets through.

     @param self This modifier.
     @param particle The particle, as the source made it, before it has been
     carried to the phantom -- which is cheaper, since one that is blocked
     never needs carrying.
     @return The fraction of the weight that survives: 1 to let it through
     untouched, 0 to stop it, anything between to attenuate it. Values
     outside [0,1] are the modifier's own bug.

     @warning Runs inside the parallel history loop. It must draw no random
     numbers -- see the file comment -- and touch nothing but itself. */
    double (*transmission)(const struct OmcBeamModifier *self,
                           const struct OmcSourceParticle *particle);

    void *impl;                 ///< the concrete modifier, for its own methods
};

/*! A transmission mask on a plane across the beam.

 The mask is a regular grid of transmission values on the plane z = #z of the
 phantom's coordinate system, which is where a jaw or a leaf bank sits. Cell
 (i,j) covers x from `x0 + i*dx` to `x0 + (i+1)*dx` and y likewise, and holds
 what gets through it.

 The simplest useful case is a single cell: one open cell of the size of the
 field, with #outside left at 0, is a rectangular aperture.

 @warning #transmission belongs to the caller and must outlive the run. */
struct OmcApertureMask {
    double z;                   ///< the plane the mask sits on, in cm

    double x0, y0;              ///< lower corner of the grid, in cm
    double dx, dy;              ///< cell size, in cm, above zero
    int nx, ny;                 ///< cells along x and y, at least one

    /*! What gets through each cell, from 0 to 1. #nx times #ny values, x
     running fastest: cell (i,j) is `transmission[i + j*nx]`. */
    const double *transmission;

    /*! What gets through beside the grid, from 0 to 1. Zero -- nothing --
     is what a field stop does, and is what a mask covering only the field
     wants. */
    double outside;
};

/*! Present an aperture mask to an engine as a modifier.

 @param mask The mask. Must outlive @p modifier.
 @param modifier Filled in with the modifier interface. */
void omcApertureMaskAsModifier(struct OmcApertureMask *mask,
                               struct OmcBeamModifier *modifier);

/*! What a modifier lets through, with the do-nothing case folded in.

 @param modifier The modifier, or `NULL` for nothing in the way.
 @param particle The particle.
 @return The fraction of the weight that survives, 1 when @p modifier is
 `NULL`. */
double omcBeamModifierTransmission(const struct OmcBeamModifier *modifier,
                                   const struct OmcSourceParticle *particle);

#endif

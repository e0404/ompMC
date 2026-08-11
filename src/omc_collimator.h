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

 How a fraction is spent is a separate choice, struct
 OmcBeamModifier::apply's -- see enum OmcModifierApply. What is never in
 question is the transmission() method itself: it draws no random numbers
 whatever the mode, so anything it costs is arithmetic.
*****************************************************************************/

#ifndef OMC_COLLIMATOR_H
#define OMC_COLLIMATOR_H

struct OmcSourceParticle;
struct OmcBeamModifier;

/*! What to do with the fraction a modifier lets through.

 Both give the same dose in the mean and differ in how they spend the
 computer's time on it. Neither is right in general: it depends on whether a
 run is dominated by the particles that get through a leaf or by the ones
 that do not. */
enum OmcModifierApply {

    /*! Multiply the particle's weight by the fraction and transport it. A
     leaf transmitting 2% gives a particle 2% of the weight and then costs a
     full shower for a fiftieth of the dose, so a run behind thick leaves
     spends nearly all of itself on particles that hardly matter. In return
     it draws no random numbers at all, and every history's random stream is
     exactly where it would have been with the beam open -- which is what
     makes a collimated run comparable, history by history, with the open one
     it came from. */
    OMC_MODIFIER_WEIGHT = 0,

    /*! Let the particle through with probability equal to the fraction, at
     the weight it already had, and stop it otherwise. The 2% leaf now costs
     a shower one time in fifty and nothing the other forty nine, which is
     where the time goes instead of into weight that will not be seen.

     The price is one random number per history -- but only for the
     particles it is actually asked about: a fraction of 0 or 1 is decided
     without drawing, so a fully open mask leaves every stream untouched and
     an all-or-nothing aperture like a jaw behaves exactly as it does under
     OMC_MODIFIER_WEIGHT.

     @warning Noisier per history where it is cheaper per history. A leaf
     that half the particles survive at full weight scatters the dose behind
     it more than half of them at half weight would. It pays when the leaves
     are thick, and costs when they are nearly open. */
    OMC_MODIFIER_ROULETTE = 1
};

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
     numbers -- whether one gets drawn is #apply's decision, made once in
     omcBeamModifierApply() where its effect on the random streams is
     visible -- and it must touch nothing but itself. */
    double (*transmission)(const struct OmcBeamModifier *self,
                           const struct OmcSourceParticle *particle);

    void *impl;                 ///< the concrete modifier, for its own methods

    /*! How omcBeamModifierApply() spends what #transmission returns. Zero,
     the value a zeroed struct has, is OMC_MODIFIER_WEIGHT. */
    enum OmcModifierApply apply;
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

 Sets struct OmcBeamModifier::apply to OMC_MODIFIER_WEIGHT; a caller wanting
 roulette overwrites it afterwards.

 @param mask The mask. Must outlive @p modifier.
 @param modifier Filled in with the modifier interface. */
void omcApertureMaskAsModifier(struct OmcApertureMask *mask,
                               struct OmcBeamModifier *modifier);

/*! What a modifier lets through, with the do-nothing case folded in.

 This is the fraction itself, before struct OmcBeamModifier::apply has had
 anything to say about it. Engines want omcBeamModifierApply() instead.

 @param modifier The modifier, or `NULL` for nothing in the way.
 @param particle The particle.
 @return The fraction of the weight that survives, 1 when @p modifier is
 `NULL`. */
double omcBeamModifierTransmission(const struct OmcBeamModifier *modifier,
                                   const struct OmcSourceParticle *particle);

/*! Put a particle through a modifier: ask what gets through and spend it the
 way struct OmcBeamModifier::apply asks for.

 @param modifier The modifier, or `NULL` for nothing in the way, in which
 case every particle survives untouched and nothing is drawn.
 @param particle The particle as the source made it, before it has been
 carried to the phantom. Its weight is scaled in place under
 OMC_MODIFIER_WEIGHT and left alone under OMC_MODIFIER_ROULETTE.
 @return 1 if the particle carries on, 0 if the modifier stopped it.

 @warning Runs inside the parallel history loop, and under
 OMC_MODIFIER_ROULETTE draws from the current history's random stream. Call
 it after setRandomHistory(), once per history, so that what a history gets
 stays a function of its index alone. */
int omcBeamModifierApply(const struct OmcBeamModifier *modifier,
                         struct OmcSourceParticle *particle);

#endif

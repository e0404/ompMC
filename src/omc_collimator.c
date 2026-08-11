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

#include "omc_collimator.h"

#include "omc_host.h"
#include "omc_random.h"
#include "omc_source.h"

#include <math.h>

/******************************************************************************/

double omcBeamModifierTransmission(const struct OmcBeamModifier *modifier,
                                   const struct OmcSourceParticle *particle) {

    if (modifier == NULL || modifier->transmission == NULL) {
        return 1.0;
    }

    return modifier->transmission(modifier, particle);
}

int omcBeamModifierApply(const struct OmcBeamModifier *modifier,
                         struct OmcSourceParticle *particle) {

    double through = omcBeamModifierTransmission(modifier, particle);

    /* Nothing gets through, and nothing more needs deciding. */
    if (!(through > 0.0)) {
        return 0;
    }

    /* Everything gets through, and nothing more needs deciding either -- in
     particular no random number, which is what keeps an open mask, and the
     open parts of any mask, on the streams they would have had with no
     collimator at all. A fraction above one is a modifier's own bug that
     check() is there to catch; passing the particle on unchanged is the
     harmless reading of it. */
    if (through >= 1.0) {
        return 1;
    }

    if (modifier->apply == OMC_MODIFIER_ROULETTE) {
        /* Survive with probability `through`, at the weight already carried:
         the mean weight through the leaf is the same as multiplying by it,
         and the particles that do get through are worth simulating. */
        return setRandom() < through;
    }

    particle->weight *= through;

    return 1;
}

/******************************************************************************/
/* The aperture mask */

static void maskCheck(const struct OmcBeamModifier *self) {

    const struct OmcApertureMask *mask =
        (const struct OmcApertureMask *)self->impl;

    if (mask->nx < 1 || mask->ny < 1) {
        omcFail("ompMC:collimator:emptyMask",
            "The aperture mask is %d by %d cells, and needs at least one of "
            "each.", mask->nx, mask->ny);
    }

    if (!(mask->dx > 0.0) || !(mask->dy > 0.0) ||
        !isfinite(mask->dx) || !isfinite(mask->dy)) {
        omcFail("ompMC:collimator:badCellSize",
            "The aperture mask has cells of %g by %g cm, and they have to be "
            "finite and above zero.", mask->dx, mask->dy);
    }

    if (!isfinite(mask->z) || !isfinite(mask->x0) || !isfinite(mask->y0)) {
        omcFail("ompMC:collimator:badPlane",
            "The aperture mask sits at a plane or corner that is not a "
            "number.");
    }

    if (mask->transmission == NULL) {
        omcFail("ompMC:collimator:noTransmission",
            "The aperture mask has no transmission values.");
    }

    if (!(mask->outside >= 0.0) || !(mask->outside <= 1.0)) {
        omcFail("ompMC:collimator:badTransmission",
            "The aperture mask lets %g through beside the grid, and a "
            "fraction has to be between 0 and 1.", mask->outside);
    }

    /* Checked here rather than trusted, because a value above one would
     quietly multiply the dose rather than fail. */
    for (int j = 0; j < mask->ny; j++) {
        for (int i = 0; i < mask->nx; i++) {
            double t = mask->transmission[i + j*mask->nx];

            if (!(t >= 0.0) || !(t <= 1.0)) {
                omcFail("ompMC:collimator:badTransmission",
                    "Cell %d,%d of the aperture mask lets %g through, and a "
                    "fraction has to be between 0 and 1.", i, j, t);
            }
        }
    }

    return;
}

static double maskTransmission(const struct OmcBeamModifier *self,
                               const struct OmcSourceParticle *particle) {

    const struct OmcApertureMask *mask =
        (const struct OmcApertureMask *)self->impl;

    /* Where the particle crossed the plane the mask sits on. A particle
     travelling along the plane never crosses it, and is stopped: it has no
     business in a beam pointed at the phantom, and there is no cell to ask
     about it. */
    if (particle->w == 0.0) {
        return 0.0;
    }

    /* The sign of t is not looked at. A negative one means the source put
     the particle downstream of the mask -- a phase space recorded below the
     jaws, say -- and the crossing point is still the one its straight line
     went through. */
    double t = (mask->z - particle->z)/particle->w;

    double x = particle->x + t*particle->u;
    double y = particle->y + t*particle->v;

    double fi = (x - mask->x0)/mask->dx;
    double fj = (y - mask->y0)/mask->dy;

    /* floor() rather than a cast: a cast truncates towards zero, which folds
     the first cell below the corner onto the first cell above it. */
    double di = floor(fi);
    double dj = floor(fj);

    if (di < 0.0 || dj < 0.0 ||
        di >= (double)mask->nx || dj >= (double)mask->ny) {
        return mask->outside;
    }

    return mask->transmission[(int)di + (int)dj*mask->nx];
}

void omcApertureMaskAsModifier(struct OmcApertureMask *mask,
                               struct OmcBeamModifier *modifier) {

    modifier->check = maskCheck;
    modifier->transmission = maskTransmission;
    modifier->impl = mask;
    modifier->apply = OMC_MODIFIER_WEIGHT;

    return;
}

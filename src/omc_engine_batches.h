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
 omc_engine_batches - The history loop the forward-style engines share.

 Internal to the core: an engine includes this, a host never does. It is a
 header of its own rather than part of omc_engine_forward.h because what it
 declares is not part of any engine's interface -- it is the one copy of the
 loop that omcCalcForward() and omcCalcRadial() both run, kept in one place
 so that the two cannot drift on the thing that must not drift.

 Which random stream a history gets is the whole reason this is shared. The
 index has to be unique over the run and depend on nothing else, or the answer
 starts depending on how OpenMP handed the histories out; getting that right
 once is better than getting it right twice.
*****************************************************************************/

#ifndef OMC_ENGINE_BATCHES_H
#define OMC_ENGINE_BATCHES_H

#include "omc_collimator.h"
#include "omc_engine_forward.h"
#include "omc_source.h"

/*! Run @p nbatch batches of @p nperbatch histories, accumulating each batch as
 it finishes and asking the caller whether to carry on.

 @param source Where the particles come from; already checked and prepared.
 @param modifier What is in the beam's way, or `NULL`.
 @param nbatch Statistical batches.
 @param nperbatch Histories in each.
 @param callbacks Progress reporting, or `NULL`.
 @param started Set to the number of histories that put a particle in the
 phantom, or `NULL`.
 @param blocked Set to the number the modifier stopped, or `NULL`.
 @return 1 if the progress callback stopped the run, 0 if it ran to the end.

 @pre initScore(), initRandom() and initStack() have been called. */
int omcEngineRunBatches(struct OmcSource *source,
                        const struct OmcBeamModifier *modifier,
                        int nbatch, int nperbatch,
                        const struct OmcForwardCallbacks *callbacks,
                        unsigned long long *started,
                        unsigned long long *blocked);

#endif

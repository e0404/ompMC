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
 omc_engine_radial - Dose in a cylinder, by ring and depth.

 What omcCalcForward() is to a voxel cube, this is to the r-z geometry of
 omc_geom_cyl.h: one run over a whole source, one dense result. The result is
 `nr*nz` entries rather than a cube, indexed `ir + iz*nr` with the ring
 running fastest.

 It exists because of what a pencil beam does to a rectilinear grid. Dose
 around a narrow beam falls by orders of magnitude over the first few
 millimetres off the axis, so following it needs voxels far finer than the
 rest of the phantom will ever need -- and then the answer still has to be
 rebinned into rings afterwards, at which point the uncertainty of a ring is
 no longer something the batch statistics can tell you, the voxels summed into
 it being correlated. Scoring the rings directly avoids both: they are the
 transport's own regions, so each one gets its batch variance from exactly the
 machinery every other ompMC geometry uses.

 Any source will do -- omc_source_pencil.h for the beams that shine down the
 axis, omc_source_phsp.h for a phase space, anything else that fills in a
 struct OmcSource -- because carrying a particle into the phantom is
 omcSourcePlace()'s job and it knows about cylinders too.

 Before calling omcCalcRadial() the host must have

   1. filled struct Geom and called omcGeomCylInit() and initRegions()
   2. called initMediaData() and initVrt()                     (ompmc.h)
   3. set up the source, including its spectrum if it has one

 and afterwards it owns the cleanup of those.

 @warning Like the rest of ompMC this is a singleton: one calculation at a
 time per process.
*****************************************************************************/

#ifndef OMC_ENGINE_RADIAL_H
#define OMC_ENGINE_RADIAL_H

#include "omc_collimator.h"
#include "omc_engine_forward.h"
#include "omc_source.h"

/*! Run parameters for one radial calculation. What the particles ARE is the
 source's business, not this struct's. */
struct OmcRadialOptions {
    int nhist;                  ///< total histories, over the whole calculation
    int nbatch;                 ///< statistical batches to split them into

    int outputDose;             ///< 1 : dose in Gy, 0 : mean deposited energy
};

/*! Transport the histories and write the results into dose[] and, unless it
 is NULL, uncertainty[]. Both are supplied by the caller and hold one entry
 per region, indexed `ir + iz*nr`.

 What the result MEANS is the source's business, through
 struct OmcSource::batchScale and struct OmcSource::incidentFluence, exactly
 as it is for omcCalcForward(). The beams in omc_source_pencil.h and a phase
 space both report per incident history.

 @param options Run parameters.
 @param source Where the particles come from. Checked over before the
 histories start, prepared with the batch size, and released afterwards, so
 the same source struct can be run again but must outlive the call.
 @param modifier What is in the beam's way -- a collimator, typically -- or
 `NULL` for an open beam.
 @param dose Caller-supplied array of `isize*ksize` entries.
 @param uncertainty Caller-supplied array of the same size, or `NULL`. Holds
 the RELATIVE uncertainty, and is 0.9999999 wherever nothing was deposited,
 the same convention omcCalcCube() and omcCalcForward() follow.
 @param callbacks Progress reporting; see struct OmcForwardCallbacks.
 @param summary Optional; filled in with what the run did.
 @return Nonzero when the run finished, 0 when the progress callback stopped
 it.

 Reports through omcFail(): `ompMC:radial:notCylindrical` when the phantom is
 a rectilinear grid, `ompMC:radial:tooFewBatches` for fewer than the two
 batches the variance needs, and `ompMC:radial:noSource` for a source that
 cannot make a particle.

 @warning Histories whose particle never reaches the phantom still count among
 the ones the result is divided by; see the warning on omcCalcForward().

 @note A MATLAB host could reach this the same way the Python one does: the
 geometry is pure data -- ring bounds, depth bounds, one medium -- so a fourth
 `mcOpt.mode` would need no more of the core than is already here. */
int omcCalcRadial(const struct OmcRadialOptions *options,
                  struct OmcSource *source,
                  const struct OmcBeamModifier *modifier,
                  double *dose, double *uncertainty,
                  const struct OmcForwardCallbacks *callbacks,
                  struct OmcForwardSummary *summary);

#endif

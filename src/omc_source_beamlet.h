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
 omc_source_beamlet - Particles starting on a beamlet aperture.

 The source model the matRad interface uses: a beam has a source point, a
 beamlet is a rectangle somewhere in front of it -- at isocentre, in matRad's
 case -- and a primary particle starts at the source point, flying through a
 uniformly sampled point of that rectangle.

 The aperture is sampled rather than tested against, which is what makes this
 cheap: every particle it makes goes through the beamlet, so none is wasted.
 That is also why it cannot be built out of a general beam and a collimator --
 a collimator can only throw particles away.

 Two engines start their histories this way and must keep agreeing on what
 they mean, so the sampling lives here rather than in either of them:

   omc_engine_dij      one beamlet at a time, one sparse column each
   omc_engine_forward  every beamlet at once, weighted, one dense cube

 The geometry (omc_geom.h), the media and the spectrum have to be set up
 before any of this is called; see the engine headers.
*****************************************************************************/

#ifndef OMC_SOURCE_BEAMLET_H
#define OMC_SOURCE_BEAMLET_H

#include "omc_source.h"

struct OmcSpectrum;

/*! Where on the source the particles start. */
enum OmcSourceGeometry {
    OMC_SOURCE_POINT = 0,     /**< the classic point source */
    OMC_SOURCE_GAUSSIAN       /**< spreads the starting point over the
                               collimator plane, which softens the penumbra */
};

/*! The beamlets. Per beam: the source position. Per beamlet: which beam it
 belongs to, and the corner plus two edge vectors of its aperture rectangle at
 isocentre.

 @warning All arrays belong to the caller and must outlive the call. */
struct OmcBeamletSource {
    int nbeamlets;               ///< number of beamlets
    const int *ibeam;           ///< index of the beam of each beamlet, 0 based

    const double *xsource;      ///< x coordinate of the source of each beam
    const double *ysource;      ///< y coordinate of the source of each beam
    const double *zsource;      ///< z coordinate of the source of each beam

    const double *xcorner;      ///< x coordinate of the beamlet corner
    const double *ycorner;      ///< y coordinate of the beamlet corner
    const double *zcorner;      ///< z coordinate of the beamlet corner

    const double *xside1;       ///< x component of the first edge vector of the beamlet
    const double *yside1;       ///< y component of the first edge vector of the beamlet
    const double *zside1;       ///< z component of the first edge vector of the beamlet

    const double *xside2;       ///< x component of the second edge vector of the beamlet
    const double *yside2;       ///< y component of the second edge vector of the beamlet
    const double *zside2;       ///< z component of the second edge vector of the beamlet
};

/*! Everything the sampling needs that does not change from history to
 history. */
struct OmcBeamletSampler {
    const struct OmcBeamletSource *source;   ///< the beamlets
    const struct OmcSpectrum *spectrum;      ///< source energy spectrum

    int charge;                 ///< 0 : photons, -1 : electrons, +1 : positrons
    enum OmcSourceGeometry geometry;   ///< POINT or GAUSSIAN
    double gaussianWidth;       ///< standard deviation in cm, GAUSSIAN only
};

/*! Make the primary particle of one history of beamlet @p ibeamlet.

 @param sampler Sampling parameters, unchanged since the caller filled them.
 @param ibeamlet Index of the beamlet to sample from.
 @param weight Becomes the particle's statistical weight. Pass 1.0 for an
 unweighted history; omc_engine_forward passes the beamlet's share of the
 fluence.
 @param particle Filled in with the particle.
 @return 1 always: a beamlet particle is aimed through its aperture by
 construction, so there is always one.

 @warning Runs inside the parallel history loop, so it touches nothing but
 its own random number generator and the read-only sampler. */
int omcBeamletProduce(const struct OmcBeamletSampler *sampler, int ibeamlet,
                      double weight, struct OmcSourceParticle *particle);

/*! Beamlets with a weight each, as a source an engine can run.

 The histories of a batch are shared out among the beamlets in proportion to
 their weights, which is what makes one run cover a whole fluence map instead
 of one beamlet. Fill in #sampler and #weights; the rest is private and set up
 by struct OmcSource::prepare(). */
struct OmcBeamletHistories {
    struct OmcBeamletSampler sampler;   ///< which beamlets, which spectrum
    const double *weights;              ///< one per beamlet, finite and >= 0

    /*! @cond OMC_INTERNAL */
    void *allocation;
    /*! @endcond */
};

/*! What sharing the histories out among the beamlets came to. */
struct OmcBeamletStats {
    int nweighted;              ///< beamlets asked for with a weight above zero
    int nsampled;               ///< of those, the ones that got any histories
    double totalWeight;         ///< sum of the weights asked for
    double sampledWeight;       ///< sum over the beamlets that got histories
};

/*! Present weighted beamlets to an engine as a source.

 @param histories The beamlets and their weights. Must outlive @p source, and
 struct OmcSource::release() gives back what prepare() took.
 @param source Filled in with the source interface. */
void omcBeamletHistoriesAsSource(struct OmcBeamletHistories *histories,
                                 struct OmcSource *source);

/*! @param histories The beamlets, after a run has prepared them.
 @param stats Filled in with how the histories were shared out. Zeroed if the
 source has not been prepared. */
void omcBeamletHistoriesStats(const struct OmcBeamletHistories *histories,
                              struct OmcBeamletStats *stats);

#endif

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

/******************************************************************************
 omc_source_beamlet - Particles starting on a beamlet aperture.

 The source model the matRad interface uses: a beam has a source point, a
 beamlet is a rectangle somewhere in front of it -- at isocentre, in matRad's
 case -- and a primary particle starts at a uniformly sampled point of that
 rectangle, flying away from the source point.

 Two engines start their histories this way and must keep agreeing on what
 they mean, so the sampling lives here rather than in either of them:

   omc_engine_dij      one beamlet at a time, one sparse column each
   omc_engine_forward  every beamlet at once, weighted, one dense cube

 The geometry (omc_geom.h), the media and the spectrum have to be set up
 before any of this is called; see the engine headers.
*****************************************************************************/

#ifndef OMC_SOURCE_BEAMLET_H
#define OMC_SOURCE_BEAMLET_H

struct OmcSpectrum;

/* Where on the source the particles start. POINT is the classic point source;
 GAUSSIAN spreads the starting point over the collimator plane, which softens
 the penumbra. */
enum OmcSourceGeometry {
    OMC_SOURCE_POINT = 0,
    OMC_SOURCE_GAUSSIAN
};

/* The beamlets. Per beam: the source position. Per beamlet: which beam it
 belongs to, and the corner plus two edge vectors of its aperture rectangle at
 isocentre. All arrays belong to the caller and must outlive the call. */
struct OmcBeamletSource {
    int nbeamlets;
    const int *ibeam;           // index of the beam of each beamlet, 0 based

    const double *xsource;      // coordinates of the source of each beam
    const double *ysource;
    const double *zsource;

    const double *xcorner;      // coordinates of the beamlet corner
    const double *ycorner;
    const double *zcorner;

    const double *xside1;       // first edge vector of the beamlet
    const double *yside1;
    const double *zside1;

    const double *xside2;       // second edge vector of the beamlet
    const double *yside2;
    const double *zside2;
};

/* Everything the sampling needs that does not change from history to history.
 An engine fills this once, before its first parallel region, and hands it to
 omcBeamletSample() unchanged from then on. */
struct OmcBeamletSampler {
    const struct OmcBeamletSource *source;
    const struct OmcSpectrum *spectrum;

    int charge;                 // 0 : photons, -1 : electrons, +1 : positrons
    enum OmcSourceGeometry geometry;
    double gaussianWidth;       // standard deviation in cm, GAUSSIAN only
};

/* Put one primary particle of beamlet ibeamlet on the (thread local) stack,
 already transported to the phantom surface and with its region index found.

 weight becomes the particle's statistical weight, and also scales what the
 history contributes to the incident energy tally. Pass 1.0 for an unweighted
 history; omc_engine_forward passes the beamlet's share of the fluence.

 Runs inside the parallel history loop, so it touches nothing but the thread's
 own stack, its own random number generator, and the read-only sampler. */
void omcBeamletSample(const struct OmcBeamletSampler *sampler, int ibeamlet,
                      double weight);

#endif

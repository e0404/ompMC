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
 omc_engine_dij - Dose influence matrix for a set of beamlets.

 This is what the matRad interface calculates: each beamlet is an aperture
 rectangle at isocentre, particles start on it and fly away from the source
 point of the beam it belongs to, and the dose they deposit becomes one column
 of a sparse matrix.

 The engine takes plain C arrays and reports its results through callbacks, so
 that the host -- a MEX file, a Python extension -- decides how the columns are
 stored without the engine knowing anything about mxArray or numpy. Before
 calling omcCalcDij() the host must have

   1. filled struct Geom and called initRegions()   (omc_geom.h)
   2. called initMediaData() and initVrt()          (ompmc.h)
   3. built the source spectrum                     (omc_spectrum.h)

 and afterwards it owns the cleanup of all three. The engine itself owns only
 the scoring arrays, the random number generators and the particle stacks,
 which it sets up and tears down per call.

 Like the rest of ompMC this is a singleton: one calculation at a time per
 process, since the transport state lives in globals.
*****************************************************************************/

#ifndef OMC_ENGINE_DIJ_H
#define OMC_ENGINE_DIJ_H

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

struct OmcDijOptions {
    int nhist;                  // total histories per beamlet
    int nbatch;                 // statistical batches to split them into
    int charge;                 // 0 : photons, -1 : electrons, +1 : positrons

    /* Voxels below this fraction of the beamlet's maximum dose are dropped
     from the column rather than reported. */
    double relDoseThreshold;

    enum OmcSourceGeometry sourceGeometry;
    double sourceGaussianWidth; // standard deviation in cm, GAUSSIAN only

    int wantVariance;           // also report the variance of the mean
};

struct OmcDijCallbacks {
    /* One finished beamlet. voxels holds nvoxels grid indices in ascending
     order, 0 based, and dose the dose in Gy in each of them; variance holds
     the variance of the mean where it was asked for and is NULL otherwise.
     All three arrays belong to the engine and are only valid for the duration
     of the call.

     Called on the master thread, outside any parallel region. */
    void (*beamlet)(int ibeamlet, int nvoxels, const int *voxels,
                    const double *dose, const double *variance, void *user);

    /* Fraction of the whole calculation finished, in [0,1]. Optional; called
     once per batch and once per beamlet, also on the master thread. */
    void (*progress)(double fraction, void *user);

    void *user;                 // passed back to both, untouched
};

void omcCalcDij(const struct OmcDijOptions *options,
                const struct OmcBeamletSource *source,
                const struct OmcSpectrum *spectrum,
                const struct OmcDijCallbacks *callbacks);

#endif

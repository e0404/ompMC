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
 omc_spectrum - The energy distribution of the source particles.

 A spectrum is a histogram: nbins bins, bin i running from the upper energy of
 bin i-1 (or emin, for the first) to upperEnergy[i], holding counts[i] of the
 particles. It reaches ompMC either from a .spectrum file or straight from the
 host as arrays, and both end up in the same sampling tables, so a spectrum
 handed over from MATLAB or Python is sampled exactly like one read from disk.

 The tables are the inverse of the cumulative distribution, evaluated on a
 fixed grid, which turns sampling an energy into two random numbers and an
 array lookup.
*****************************************************************************/

#ifndef OMC_SPECTRUM_H
#define OMC_SPECTRUM_H

#include "omc_random.h"

#include <math.h>

/* What counts[] means. Counts per MeV are converted to counts per bin on the
 way in, by scaling with the bin widths. */
enum OmcSpectrumMode {
    OMC_SPECTRUM_COUNTS_PER_BIN = 0,
    OMC_SPECTRUM_COUNTS_PER_MEV = 1
};

struct OmcSpectrum {
    int monoenergetic;          /* 1 : every particle starts at energy */
    double energy;              /* the energy, when monoenergetic */

    double deltak;              /* number of elements in the inverse CDF */
    double *cdfinv1;            /* lower energy of the bin an element falls in */
    double *cdfinv2;            /* width of that bin */
};

/* All of these leave the spectrum ready to sample from, and take ownership of
 nothing: the arrays passed in may be freed by the caller afterwards. */

void omcSpectrumMonoenergetic(struct OmcSpectrum *spectrum, double energy);

/* nbins bins with the given upper energies, ascending and all above emin, and
 non-negative counts that sum to something positive. A caller that cannot
 guarantee that should check first -- the failures here are reported through
 omcFail(), which does not return. */
void omcSpectrumFromHistogram(struct OmcSpectrum *spectrum,
                              const double *upperEnergy, const double *counts,
                              int nbins, double emin, int mode);

/* Read an EGSnrc style .spectrum file: a title line, then "nbins emin mode",
 then one "upperEnergy count" pair per line. */
void omcSpectrumFromFile(struct OmcSpectrum *spectrum, const char *path);

void omcSpectrumFree(struct OmcSpectrum *spectrum);

/* Sample a kinetic energy in MeV. Called once per history, so it is inline
 rather than a call into another object file; the arithmetic is unchanged from
 when it sat in the user codes.

 It draws its own random numbers, and deliberately draws NONE for a
 monoenergetic source. That is not just an optimization: the random stream is
 indexed per history, so drawing two numbers that are then thrown away would
 shift every later draw in the history and change the result of an otherwise
 identical run. */
static inline double omcSpectrumSample(const struct OmcSpectrum *spectrum) {

    if (spectrum->monoenergetic) {
        return spectrum->energy;
    }

    /* Sample initial energy from spectrum data */
    double rnno1 = setRandom();
    double rnno2 = setRandom();

    /* Sample bin number in order to select particle energy */
    int k = (int)fmin(spectrum->deltak*rnno1, spectrum->deltak - 1.0);

    return spectrum->cdfinv1[k] + rnno2*spectrum->cdfinv2[k];
}

#endif

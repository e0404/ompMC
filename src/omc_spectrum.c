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

#include "omc_spectrum.h"

#include "omc_host.h"
#include "omc_utilities.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* static, unlike the copies in the user codes these replace: in C a file
 scope const has external linkage, so nothing keeps a user code from defining
 its own MXEBIN again without the linker complaining about this one. */
static const int MXEBIN = 200;     // number of energy bins of spectrum
static const int INVDIM = 1000;    // number of bins in inverse CDF

void omcSpectrumMonoenergetic(struct OmcSpectrum *spectrum, double energy) {

    spectrum->monoenergetic = 1;
    spectrum->energy = energy;
    spectrum->deltak = 0.0;
    spectrum->cdfinv1 = NULL;
    spectrum->cdfinv2 = NULL;

    omcLog(OMC_LOG_INFO, "%f monoenergetic source", energy);

    return;
}

void omcSpectrumFromHistogram(struct OmcSpectrum *spectrum,
                              const double *upperEnergy, const double *counts,
                              int nbins, double emin, int mode) {

    spectrum->monoenergetic = 0;
    spectrum->energy = 0.0;

    /* Counts per MeV are turned into counts per bin here, which is why this
     needs a copy of the caller's array rather than reading it in place. */
    double *srcpdf = malloc(nbins*sizeof(double));
    for (int i = 0; i < nbins; i++) {
        srcpdf[i] = counts[i];
    }

    if (mode == OMC_SPECTRUM_COUNTS_PER_MEV) {
        omcLog(OMC_LOG_DEBUG, "Counts/MeV assumed.");
        srcpdf[0] *= (upperEnergy[0] - emin);
        for (int i = 1; i < nbins; i++) {
            srcpdf[i] *= (upperEnergy[i] - upperEnergy[i - 1]);
        }
    }
    else if (mode == OMC_SPECTRUM_COUNTS_PER_BIN) {
        omcLog(OMC_LOG_DEBUG, "Counts/bin assumed.");
    }
    else {
        free(srcpdf);
        omcFail("ompMC:spectrum:invalidMode",
            "Invalid spectrum mode %d, expected 0 for counts per bin or 1 for "
            "counts per MeV.", mode);
    }

    omcLog(OMC_LOG_DETAIL, "Energy ranges from %f to %f MeV",
           emin, upperEnergy[nbins - 1]);

    /* Initialization routine to calculate the inverse of the
     cumulative probability distribution that is used during execution to
     sample the incident particle energy. */
    double *srccdf = malloc(nbins*sizeof(double));

    srccdf[0] = srcpdf[0];
    for (int i=1; i<nbins; i++) {
        srccdf[i] = srccdf[i-1] + srcpdf[i];
    }

    double fnorm = 1.0/srccdf[nbins - 1];
    double binsok = 0.0;
    spectrum->deltak = INVDIM; /* number of elements in inverse CDF */
    double gridsz = 1.0f/spectrum->deltak;

    for (int i=0; i<nbins; i++) {
        srccdf[i] *= fnorm;
        if (i == 0) {
            if (srccdf[0] <= 3.0*gridsz) {
                binsok = 1.0;
            }
        }
        else {
            if ((srccdf[i] - srccdf[i - 1]) < 3.0*gridsz) {
                binsok = 1.0;
            }
        }
    }

    if (binsok != 0.0) {
        omcLog(OMC_LOG_DETAIL, "Warning! Some of normalized bin probabilities "
               "are so small that bins may be missed.");
    }

    /* Calculate cdfinv. This array allows the rapid sampling for the
     energy by precomputing the results for a fine grid. */
    spectrum->cdfinv1 = malloc(spectrum->deltak*sizeof(double));
    spectrum->cdfinv2 = malloc(spectrum->deltak*sizeof(double));
    double ak;

    for (int k=0; k<spectrum->deltak; k++) {
        ak = (double)k*gridsz;
        int i;

        for (i=0; i<nbins; i++) {
            if (ak <= srccdf[i]) {
                break;
            }
        }

        /* We should fall here only through the above break sentence. */
        if (i != 0) {
            spectrum->cdfinv1[k] = upperEnergy[i - 1];
        }
        else {
            spectrum->cdfinv1[k] = emin;
        }
        spectrum->cdfinv2[k] = upperEnergy[i] - spectrum->cdfinv1[k];

    }

    free(srcpdf);
    free(srccdf);

    return;
}

void omcSpectrumFromFile(struct OmcSpectrum *spectrum, const char *path) {

    char buffer[BUFFER_SIZE];
    char *fstatus;

    FILE *fp;

    if ((fp = fopen(path, "r")) == NULL) {
        omcFail("ompMC:spectrum:openFailed",
            "Unable to open spectrum file: %s", path);
    }

    omcLog(OMC_LOG_DEBUG, "Path to spectrum file : %s", path);

    /* Read spectrum file title */
    fstatus = fgets(buffer, BUFFER_SIZE, fp);
    if (fstatus == NULL) {
        fclose(fp);
        omcFail("ompMC:spectrum:parseFailed",
            "Could not parse spectrum file %s: it is empty.", path);
    }

    /* The title carries a trailing newline the log sink would double up */
    buffer[strcspn(buffer, "\r\n")] = '\0';
    omcLog(OMC_LOG_DETAIL, "Spectrum file title: %s", buffer);

    /* Read number of bins and spectrum type */
    double enmin;   /* lower energy of first bin */
    int nensrc;     /* number of energy bins in spectrum histogram */
    int imode;      /* 0 : histogram counts/bin, 1 : counts/MeV*/

    fstatus = fgets(buffer, BUFFER_SIZE, fp);
    if (fstatus == NULL || sscanf(buffer, "%d %lf %d",
                                  &nensrc, &enmin, &imode) != 3) {
        fclose(fp);
        omcFail("ompMC:spectrum:parseFailed",
            "Could not read the bin count, lower energy and mode from "
            "spectrum file %s.", path);
    }

    if (nensrc < 1 || nensrc > MXEBIN) {
        fclose(fp);
        omcFail("ompMC:spectrum:tooManyBins",
            "Number of energy bins = %d in %s is outside 1 to the maximum "
            "allowed %d. Increase MXEBIN macro!", nensrc, path, MXEBIN);
    }

    /* upper energy of bin i in MeV */
    double *ensrcd = malloc(nensrc*sizeof(double));
    /* prob. of finding a particle in bin i */
    double *srcpdf = malloc(nensrc*sizeof(double));

    /* Read spectrum information */
    for (int i=0; i<nensrc; i++) {
        fstatus = fgets(buffer, BUFFER_SIZE, fp);
        if (fstatus == NULL || sscanf(buffer, "%lf %lf",
                                      &ensrcd[i], &srcpdf[i]) != 2) {
            fclose(fp);
            free(ensrcd);
            free(srcpdf);
            omcFail("ompMC:spectrum:parseFailed",
                "Could not read bin %d of the %d announced by spectrum file "
                "%s.", i + 1, nensrc, path);
        }
    }

    fclose(fp);

    omcLog(OMC_LOG_DEBUG, "Have read %d input energy bins from spectrum file.",
           nensrc);

    omcSpectrumFromHistogram(spectrum, ensrcd, srcpdf, nensrc, enmin, imode);

    free(ensrcd);
    free(srcpdf);

    return;
}

void omcSpectrumFree(struct OmcSpectrum *spectrum) {

    free(spectrum->cdfinv1);
    free(spectrum->cdfinv2);

    spectrum->cdfinv1 = NULL;
    spectrum->cdfinv2 = NULL;

    return;
}

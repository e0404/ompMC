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
 omc_matrad - An ompMC user code to calculate deposited dose on voxelized 
 geometries to be used with the matRad treatment planning system.  
*****************************************************************************/

/******************************************************************************
 Definitions needed if source file compiled with mex. This macro must be 
 enabled during compilation time.
*****************************************************************************/
#include <mex.h>


/* Redefine printf() function due to conflicts with mex and OpenMP */
#include <stdio.h>
#ifdef _OPENMP
    #include <omp.h>

    #undef printf
    #define printf(...) fprintf(stdout,__VA_ARGS__)
#endif

/* Shared ompMC code reports through omcLog()/omcFail(); the sinks installed in
 initHost() below turn those into mexPrintf() and mexErrMsgIdAndTxt(). What
 remains of the old "#define exit(EXIT_FAILURE) mexErrMsgIdAndTxt(...)" trick
 is gone with them: it hid the real message behind a generic one, and quietly
 turned every exit() in scope into something that unwinds instead. */

#include "omc_engine_dij.h"
#include "omc_engine_forward.h"
#include "omc_geom.h"
#include "omc_host.h"
#include "omc_spectrum.h"
#include "omc_utilities.h"
#include "ompmc.h"
#include "omc_version.h"

#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Variables needed to parse inputs from matRad */
const mxArray *cubeRho;
const mxArray *cubeMatIx;
const mxArray *mcGeo;
const mxArray *mcSrc;
const mxArray *mcOpt;

//verbose flag
int verbose_flag;

/* Optional progress callback: a MATLAB function handle taking a single
 scalar progress argument in [0,1]. Points into mcOpt (an input array), so
 it must not be destroyed. NULL when the caller did not supply one, in
 which case progress falls back to the built-in waitbar. */
mxArray *progressCallback;

/* The particle stack, the regions and the PEGS data are the engine's business
 now; all this file still touches of the core's state is the media table it
 fills from mcGeo, and the input items it fills from mcOpt. */
extern struct Media media;

/* Everything parsed out of the MC options struct that the engine does not
 take through struct OmcDijOptions: where the spectrum comes from, and the
 file paths. */
struct OmcConfig {
    //Source Parameters
    double monoEnergy;
    int useMonoEnergy;              // mcOpt.monoEnergy was given and wins
    char * spectrumFile;            // NULL when the energies come from elsewhere

    /* Spectrum handed over directly from MATLAB in mcOpt.spectrum, used
     instead of reading spectrumFile when given. The two pointers alias the
     caller's mxArrays, which stay alive for the whole call, and are NULL
     when no spectrum was passed. */
    const double *spectrumEnergy;   // upper energy of each bin, MeV, ascending
    const double *spectrumFluence;  // relative number of particles per bin
    int spectrumNbins;
    double spectrumEnMin;           // lower energy of the first bin, MeV
    int spectrumMode;               // 0 : counts/bin, 1 : counts/MeV
};

struct OmcConfig omcConfig;

/* Which calculation the call is asking for. 'dij' is what this interface has
 always done and stays the default, so an mcOpt struct written before the
 forward mode existed keeps working untouched.

 The names carry the source model rather than the output, because the output
 is the same dense cube for all of them: forward_beamlet collimates by giving
 each of matRad's beamlets a weight, and a mode that takes real collimator
 geometry would join it here rather than replace it. */
enum OmcMode {
    OMC_MODE_DIJ = 0,
    OMC_MODE_FORWARD_BEAMLET
};

static const char *const modeNames[] = { "dij", "forward_beamlet" };

enum OmcMode omcMode;

/* What the engine is asked to calculate. Filled by parseInput(); only the one
 belonging to omcMode is used. */
struct OmcDijOptions dijOptions;
struct OmcForwardOptions forwardOptions;

/* One weight per beamlet, mcSrc.bixelWeights, used in place. NULL outside the
 forward mode. */
static const double *bixelWeights;

/******************************************************************************/
/* Host sinks. Shared ompMC code calls omcLog()/omcFail() and these turn them
 into the MATLAB equivalents. Both are only ever called on the master thread,
 outside any parallel region, which is what makes calling into the MEX API
 from here safe -- transport code that prints from inside a parallel region
 keeps using printf(), redefined above. */

static void mexLogSink(int level, const char *message, void *user) {

    (void)user;

    /* Warnings were unconditional before the sinks existed, and stay so.
     Everything else follows the verbosity the caller asked for. */
    if (level > OMC_LOG_WARNING && verbose_flag < level) {
        return;
    }

    mexPrintf("%s\n", message);

    return;
}

static void mexFailSink(const char *id, const char *message, void *user) {

    (void)user;

    /* Does not return: mexErrMsgIdAndTxt() unwinds back to MATLAB. The "%s"
     is deliberate -- message is already formatted and may well contain a
     stray percent sign from a file path. */
    mexErrMsgIdAndTxt(id, "%s", message);
}

static void initHost(void) {

    struct OmcHost host;
    host.log = mexLogSink;
    host.fail = mexFailSink;
    host.user = NULL;

    omcSetHost(&host);

    return;
}

/* Fetch a required field of the MC options struct, failing with a clear
 error message instead of handing a NULL pointer to the MATLAB API, which
 would take the whole MATLAB session down. */
static mxArray *getRequiredField(const mxArray *opts, const char *name) {

    mxArray *field = mxGetField(opts, 0, name);
    if (field == NULL) {
        mexErrMsgIdAndTxt("matRad:omc_matrad:missingField",
            "Required field '%s' is missing from the MC options struct.",
            name);
    }

    return field;
}

/* Fetch a numeric vector field of the spectrum struct. Returns the data
 pointer and, through n, its length. Optional fields come back NULL when
 absent; a required one that is missing, or any field that is not a real
 double vector, is an error. */
static const double *getSpectrumVector(const mxArray *spec, const char *name,
                                       int required, size_t *n) {

    mxArray *field = mxGetField(spec, 0, name);
    if (field == NULL) {
        if (required) {
            mexErrMsgIdAndTxt("matRad:omc_matrad:invalidSpectrum",
                "Field '%s' is missing from the spectrum struct.", name);
        }
        *n = 0;
        return NULL;
    }

    if (!mxIsDouble(field) || mxIsComplex(field) || mxIsSparse(field)) {
        mexErrMsgIdAndTxt("matRad:omc_matrad:invalidSpectrum",
            "Field 'spectrum.%s' must be a real double vector.", name);
    }

    const mwSize *dims = mxGetDimensions(field);
    if (mxGetNumberOfDimensions(field) != 2 || (dims[0] != 1 && dims[1] != 1)) {
        mexErrMsgIdAndTxt("matRad:omc_matrad:invalidSpectrum",
            "Field 'spectrum.%s' must be a vector.", name);
    }

    *n = mxGetNumberOfElements(field);
    return mxGetPr(field);
}

/* A spectrum passed in through mcOpt.spectrum, as an alternative to reading
 one from disk. It mirrors the contents of a .spectrum file and is given as a
 struct with the fields

    energy   : upper energy of each bin in MeV, strictly ascending
    fluence  : relative number of particles in each bin, same length
    eMin     : lower energy of the first bin in MeV, optional, default 0
    mode     : 0 for counts/bin (default), 1 for counts/MeV

 The arrays are only validated here; omcSpectrumFromHistogram() turns them
 into the sampling tables, the same ones a spectrum read from file ends up
 in. */
static void parseSpectrum(const mxArray *spec) {

    if (!mxIsStruct(spec) || mxGetNumberOfElements(spec) != 1) {
        mexErrMsgIdAndTxt("matRad:omc_matrad:invalidSpectrum",
            "Option 'spectrum' must be a 1x1 struct with fields 'energy' and "
            "'fluence'.");
    }

    size_t nEnergy, nFluence;
    const double *energy = getSpectrumVector(spec, "energy", 1, &nEnergy);
    const double *fluence = getSpectrumVector(spec, "fluence", 1, &nFluence);

    if (nEnergy != nFluence) {
        mexErrMsgIdAndTxt("matRad:omc_matrad:invalidSpectrum",
            "'spectrum.energy' has %d elements but 'spectrum.fluence' has %d.",
            (int) nEnergy, (int) nFluence);
    }
    if (nEnergy == 0) {
        mexErrMsgIdAndTxt("matRad:omc_matrad:invalidSpectrum",
            "The spectrum is empty.");
    }

    /* Lower edge of the first bin. Everything below it is never sampled, so
     0 is the safe default: it makes the first bin span [0, energy(1)]. */
    double enmin = 0.0;
    mxArray *field = mxGetField(spec, 0, "eMin");
    if (field != NULL) {
        if (!mxIsDouble(field) || mxIsComplex(field) ||
            mxGetNumberOfElements(field) != 1) {
            mexErrMsgIdAndTxt("matRad:omc_matrad:invalidSpectrum",
                "Field 'spectrum.eMin' must be a real scalar.");
        }
        enmin = mxGetScalar(field);
    }

    int imode = 0;
    field = mxGetField(spec, 0, "mode");
    if (field != NULL) {
        if (!mxIsDouble(field) || mxIsComplex(field) ||
            mxGetNumberOfElements(field) != 1) {
            mexErrMsgIdAndTxt("matRad:omc_matrad:invalidSpectrum",
                "Field 'spectrum.mode' must be a real scalar, 0 for counts "
                "per bin or 1 for counts per MeV.");
        }
        imode = (int) mxGetScalar(field);
        if (imode != 0 && imode != 1) {
            mexErrMsgIdAndTxt("matRad:omc_matrad:invalidSpectrum",
                "Field 'spectrum.mode' is %d, expected 0 for counts per bin "
                "or 1 for counts per MeV.", imode);
        }
    }

    /* The bin edges have to be usable as such: ascending, and above the lower
     edge of the first bin. A descending or repeated entry would come out of
     the sampling below as a negative or zero-width bin rather than as an
     obvious failure. */
    if (enmin < 0.0) {
        mexErrMsgIdAndTxt("matRad:omc_matrad:invalidSpectrum",
            "'spectrum.eMin' is %f MeV, it cannot be negative.", enmin);
    }
    if (energy[0] <= enmin) {
        mexErrMsgIdAndTxt("matRad:omc_matrad:invalidSpectrum",
            "The first bin of the spectrum ends at %f MeV, which is not above "
            "its lower edge 'eMin' = %f MeV.", energy[0], enmin);
    }

    double fluenceSum = 0.0;
    for (size_t i = 0; i < nEnergy; i++) {
        if (!mxIsFinite(energy[i])) {
            mexErrMsgIdAndTxt("matRad:omc_matrad:invalidSpectrum",
                "'spectrum.energy' entry %d is not finite.", (int) i + 1);
        }
        if (i > 0 && energy[i] <= energy[i - 1]) {
            mexErrMsgIdAndTxt("matRad:omc_matrad:invalidSpectrum",
                "'spectrum.energy' must be strictly ascending, but entry %d "
                "(%f MeV) does not exceed entry %d (%f MeV).",
                (int) i + 1, energy[i], (int) i, energy[i - 1]);
        }
        if (!(fluence[i] >= 0.0)) {   /* also catches NaN */
            mexErrMsgIdAndTxt("matRad:omc_matrad:invalidSpectrum",
                "'spectrum.fluence' entry %d is %f, it must be non-negative.",
                (int) i + 1, fluence[i]);
        }
        fluenceSum += fluence[i];
    }
    if (!(fluenceSum > 0.0) || !mxIsFinite(fluenceSum)) {
        mexErrMsgIdAndTxt("matRad:omc_matrad:invalidSpectrum",
            "'spectrum.fluence' does not sum to a positive, finite value.");
    }

    omcConfig.spectrumEnergy = energy;
    omcConfig.spectrumFluence = fluence;
    omcConfig.spectrumNbins = (int) nEnergy;
    omcConfig.spectrumEnMin = enmin;
    omcConfig.spectrumMode = imode;

    return;
}

/* Function used to parse input from matRad */
void parseInput(int nrhs, const mxArray *prhs[]) {
    //Default values
    dijOptions.nhist = 1e4;
    dijOptions.nbatch = 10;
    dijOptions.relDoseThreshold = 0.01;
    dijOptions.charge = 0;
    dijOptions.sourceGeometry = OMC_SOURCE_POINT;
    dijOptions.sourceGaussianWidth = 0.2123; //Assuming 5mm FWHM penumbra if the source is gaussian
    dijOptions.wantVariance = 0;    // set from nlhs by mexFunction()

    omcConfig.monoEnergy = 0.1;
    omcConfig.useMonoEnergy = 0;
    /* Reset explicitly rather than relying on the zero initialization of the
     global: the MEX file stays locked in memory, so a spectrum passed in one
     call would otherwise still be pointed at by the next one, which no longer
     owns those arrays. */
    omcConfig.spectrumEnergy = NULL;
    omcConfig.spectrumFluence = NULL;
    omcConfig.spectrumNbins = 0;
    omcConfig.spectrumEnMin = 0.0;
    omcConfig.spectrumMode = 0;

    
    mxArray *tmp_fieldpointer;
    char *tmp;

    cubeRho = prhs[0];
    cubeMatIx = prhs[1];
    mcGeo = prhs[2];
    mcSrc = prhs[3];
    mcOpt = prhs[4];

    /* Check data type of input arguments */
    if (!(mxIsDouble(cubeRho))){
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:inputNotDouble",
                "Input argument must be of type double.");
    }    
    if (mxGetNumberOfDimensions(cubeRho) != 3){
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:inputNot3D",
                "Input argument 1 must be a three-dimensional cube\n");
    }
    if (!mxIsInt32(cubeMatIx)) {
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:inputNotInt32","The density cube must be a 32 bit integer array!");
    }
    if(!mxIsStruct(mcGeo)) {
        mexErrMsgIdAndTxt( "MATLAB:phonebook:inputNotStruct",
                "Input 3 must be a mcGeo Structure.");
    }
    if(!mxIsStruct(mcSrc)) {
        mexErrMsgIdAndTxt( "MATLAB:phonebook:inputNotStruct",
                "Input 4 must be a mcSrc Structure.");
    }

    /* Parse Monte Carlo options and create input items structure */
    tmp_fieldpointer = mxGetField(mcOpt,0,"verbose");
    if (tmp_fieldpointer)
        verbose_flag = (int) mxGetScalar(tmp_fieldpointer);
    else
        verbose_flag = 0;

    if (verbose_flag)
        mexPrintf("ompMC output Option: Verbose flag is set to %d!\n",verbose_flag);
    else
        mexPrintf("ompMC logging disabled.\n");

    /* Which calculation to run. Absent means 'dij', which is what every
     caller written before the forward mode existed is asking for. */
    omcMode = OMC_MODE_DIJ;
    tmp_fieldpointer = mxGetField(mcOpt,0,"mode");
    if (tmp_fieldpointer) {
        if (!mxIsChar(tmp_fieldpointer)) {
            mexErrMsgIdAndTxt("matRad:omc_matrad:invalidMode",
                "Field 'mcOpt.mode' must be a string.");
        }

        char *modeName = mxArrayToString(tmp_fieldpointer);
        int nmodes = (int)(sizeof(modeNames)/sizeof(modeNames[0]));
        int known = 0;

        for (int imode = 0; imode < nmodes; imode++) {
            if (modeName != NULL && strcmp(modeName, modeNames[imode]) == 0) {
                omcMode = (enum OmcMode) imode;
                known = 1;
                break;
            }
        }

        if (!known) {
            /* mexErrMsgIdAndTxt() does not return, so the message is built
             while modeName is still around and freed before it is raised. */
            char message[BUFFER_SIZE];
            snprintf(message, sizeof(message),
                "Unknown mcOpt.mode '%s'. The modes are '%s' and '%s'.",
                modeName ? modeName : "", modeNames[0], modeNames[1]);
            mxFree(modeName);

            mexErrMsgIdAndTxt("matRad:omc_matrad:invalidMode", "%s", message);
        }

        mxFree(modeName);
    }

    /* Optional caller-supplied progress callback, e.g.
     options.progressCallback = @(p) waitbar(p, h, msg); replaces the
     built-in waitbar when given. */
    progressCallback = mxGetField(mcOpt,0,"progressCallback");
    if (progressCallback && !mxIsClass(progressCallback,"function_handle")) {
        mexPrintf("ompMC option 'progressCallback' is not a function handle, ignoring it.\n");
        progressCallback = NULL;
    }

    mxArray* tmp2;
    int status;

    /* Every block below raises nInput before it writes, so starting one below
     zero is what puts the first pair in slot 0. It used to start at 0 and
     leave that slot empty, which only worked because the lookup scanned one
     past the last pair; input_idx is a count now (see omc_utilities.h). */
    int nInput = -1;
        
    tmp_fieldpointer = mxGetField(mcOpt,0,"nHistories");

    //size_t nHistLength = mxGetNumberOfElements(tmp_fieldpointer);
    if (tmp_fieldpointer)
        dijOptions.nhist = mxGetScalar(tmp_fieldpointer);

    tmp_fieldpointer = mxGetField(mcOpt,0,"nBatches");
    if (tmp_fieldpointer)
        dijOptions.nbatch = mxGetScalar(tmp_fieldpointer);

    tmp_fieldpointer = mxGetField(mcOpt,0,"sourceGaussianWidth");
    if (tmp_fieldpointer) {
        dijOptions.sourceGaussianWidth = mxGetScalar(tmp_fieldpointer);
    }

    tmp_fieldpointer = mxGetField(mcOpt,0,"sourceGeometry");
    if (tmp_fieldpointer) {
        size_t buflen = mxGetNumberOfElements(tmp_fieldpointer) + 1;
        char* sourceGeoTmpStr = (char*) mxCalloc(buflen + 1,sizeof(char));
        if (mxGetString(tmp_fieldpointer, sourceGeoTmpStr, buflen) != 0) 
            mexErrMsgIdAndTxt("MATLAB:explore:invalidStringArray","Invalid string for source Geometry");

        //Parse source definition
        if (strcmp(sourceGeoTmpStr,"gaussian") == 0)
        {
            dijOptions.sourceGeometry = OMC_SOURCE_GAUSSIAN;
            mexPrintf("Using 'gaussian' source geometry with %f mm width...\n",dijOptions.sourceGaussianWidth);
        }
        else if (strcmp(sourceGeoTmpStr,"point") == 0)
        {
            dijOptions.sourceGeometry = OMC_SOURCE_POINT;
            mexPrintf("Using 'point' source geometry...\n");
        }
        else
        {            
            mexPrintf("Source geometry '%s' unkwnown, using 'point'\n",sourceGeoTmpStr);            
        }
    }

    
    
    /* Get splitting factor */
    /*  
    tmp_fieldpointer = mxGetField(mcOpt,0,"nSplit");
    if (tmp_fieldpointer)
        omcConfig.nSplit = mxGetScalar(tmp_fieldpointer);
    */

    nInput++;
    sprintf(input_items[nInput].key,"nsplit");
    tmp_fieldpointer = getRequiredField(mcOpt,"nSplit");
    status = mexCallMATLAB(1, &tmp2, 1,  &tmp_fieldpointer, "num2str");    
    if (status != 0)
        mexErrMsgIdAndTxt( "matRad:omc_matrad:Error","Call to num2str not successful");
    else
    {
        tmp = mxArrayToString(tmp2);        
        strcpy(input_items[nInput].value,tmp);
    }

    /* Where the source energies come from. Three ways of saying it, in
     descending precedence: a spectrum passed in as arrays, a spectrum file,
     or a single energy. Whatever is left over is announced rather than
     silently dropped, since a caller who sets two of them has a wrong idea of
     what the run is doing. Nothing at all still means the default file. */
    tmp_fieldpointer = mxGetField(mcOpt,0,"monoEnergy");
    if (tmp_fieldpointer && !mxIsEmpty(tmp_fieldpointer)) {
        if (!mxIsNumeric(tmp_fieldpointer) || mxIsComplex(tmp_fieldpointer) ||
            mxGetNumberOfElements(tmp_fieldpointer) != 1) {
            mexErrMsgIdAndTxt("matRad:omc_matrad:invalidMonoEnergy",
                "Option 'monoEnergy' must be a real scalar.");
        }

        omcConfig.monoEnergy = mxGetScalar(tmp_fieldpointer);

        if (!(omcConfig.monoEnergy > 0.0) || !mxIsFinite(omcConfig.monoEnergy)) {
            mexErrMsgIdAndTxt("matRad:omc_matrad:invalidMonoEnergy",
                "Option 'monoEnergy' is %g MeV, it must be positive and "
                "finite.", omcConfig.monoEnergy);
        }

        omcConfig.useMonoEnergy = 1;
    }

    tmp_fieldpointer = mxGetField(mcOpt,0,"spectrum");
    if (tmp_fieldpointer && !mxIsEmpty(tmp_fieldpointer)) {
        parseSpectrum(tmp_fieldpointer);
    }

    tmp_fieldpointer = mxGetField(mcOpt,0,"spectrumFile");
    if (omcConfig.spectrumEnergy != NULL) {
        /* Passed spectrum wins */
        omcConfig.spectrumFile = NULL;

        if (tmp_fieldpointer)
            mexPrintf("Both 'spectrum' and 'spectrumFile' were given, using "
                "the passed spectrum and ignoring the file.\n");
        if (omcConfig.useMonoEnergy)
            mexPrintf("Both 'spectrum' and 'monoEnergy' were given, using the "
                "passed spectrum and ignoring the single energy.\n");

        omcConfig.useMonoEnergy = 0;
    }
    else if (tmp_fieldpointer) {
        size_t buflen = mxGetNumberOfElements(tmp_fieldpointer) + 1;
        omcConfig.spectrumFile = (char*) mxCalloc(buflen + 1,sizeof(char));
        if (mxGetString(tmp_fieldpointer, omcConfig.spectrumFile, buflen) != 0)
            mexErrMsgIdAndTxt("MATLAB:explore:invalidStringArray","Invalid string for path to spectrum file!");

        if (omcConfig.useMonoEnergy)
            mexPrintf("Both 'spectrumFile' and 'monoEnergy' were given, using "
                "the file and ignoring the single energy.\n");

        omcConfig.useMonoEnergy = 0;
    }
    else if (omcConfig.useMonoEnergy) {
        omcConfig.spectrumFile = NULL;
    }
    else
    {
        size_t buflen = 255;
        omcConfig.spectrumFile = (char*) mxCalloc(buflen + 1,sizeof(char));
        strcpy(omcConfig.spectrumFile, "./spectra/mohan6.spectrum");
    }

    nInput++;
    sprintf(input_items[nInput].key,"charge");
    tmp_fieldpointer = getRequiredField(mcOpt,"charge");
    status = mexCallMATLAB(1, &tmp2, 1,  &tmp_fieldpointer, "num2str");
    if (status != 0)
        mexErrMsgIdAndTxt( "matRad:omc_matrad:Error","Call to num2str not successful");
    else
    {
        tmp = mxArrayToString(tmp2);
        strcpy(input_items[nInput].value,tmp);
    }

    /* The engine needs the charge itself, not just the input item: it decides
     what the source particles are. Before this was read back the field was
     accepted and then ignored, so an electron source silently ran as photons. */
    if (!mxIsNumeric(tmp_fieldpointer) || mxIsComplex(tmp_fieldpointer) ||
        mxGetNumberOfElements(tmp_fieldpointer) != 1) {
        mexErrMsgIdAndTxt("matRad:omc_matrad:invalidCharge",
            "Option 'charge' must be a real scalar: -1, 0 or +1.");
    }
    double chargeValue = mxGetScalar(tmp_fieldpointer);
    if (chargeValue != -1.0 && chargeValue != 0.0 && chargeValue != 1.0) {
        mexErrMsgIdAndTxt("matRad:omc_matrad:invalidCharge",
            "Option 'charge' is %g, expected -1 for electrons, 0 for photons "
            "or +1 for positrons.", chargeValue);
    }
    dijOptions.charge = (int)chargeValue;

    if (verbose_flag > 1) {
        const char *particle = dijOptions.charge == 0 ? "photons" :
            (dijOptions.charge < 0 ? "electrons" : "positrons");
        mexPrintf("Source charge : %d (%s)\n", dijOptions.charge, particle);
    }

    nInput++;
    sprintf(input_items[nInput].key,"global ecut");
    tmp_fieldpointer = getRequiredField(mcOpt,"global_ecut");
    status = mexCallMATLAB(1, &tmp2, 1,  &tmp_fieldpointer, "num2str");    
    if (status != 0)
        mexErrMsgIdAndTxt( "matRad:omc_matrad:Error","Call to num2str not successful");
    else
    {
        tmp = mxArrayToString(tmp2);        
        strcpy(input_items[nInput].value,tmp);
    }

    nInput++;
    sprintf(input_items[nInput].key,"global pcut");
    tmp_fieldpointer = getRequiredField(mcOpt,"global_pcut");
    status = mexCallMATLAB(1, &tmp2, 1,  &tmp_fieldpointer, "num2str");
    if (status != 0)
        mexErrMsgIdAndTxt( "matRad:omc_matrad:Error","Call to num2str not successful");
    else
    {
        tmp = mxArrayToString(tmp2);
        strcpy(input_items[nInput].value,tmp);
    }

    /* Optional VRT parameters: electron range rejection threshold "esave"
     and electron Russian roulette threshold/factor "e_rr"/"f_rr" (energies
     as total MeV); an absent field leaves the technique disabled */
    const char *vrtFields[] = {"esave", "e_rr", "f_rr"};
    for (int ivrt = 0; ivrt < 3; ivrt++) {
        tmp_fieldpointer = mxGetField(mcOpt,0,vrtFields[ivrt]);
        if (tmp_fieldpointer) {
            nInput++;
            sprintf(input_items[nInput].key,"%s",vrtFields[ivrt]);
            status = mexCallMATLAB(1, &tmp2, 1,  &tmp_fieldpointer, "num2str");
            if (status != 0)
                mexErrMsgIdAndTxt( "matRad:omc_matrad:Error","Call to num2str not successful");
            else
            {
                tmp = mxArrayToString(tmp2);
                strcpy(input_items[nInput].value,tmp);
            }
        }
    }

    nInput++;
    sprintf(input_items[nInput].key,"rng seeds");
    tmp_fieldpointer = getRequiredField(mcOpt,"randomSeeds");
    status = mexCallMATLAB(1, &tmp2, 1,  &tmp_fieldpointer, "num2str");    
    if (status != 0)
        mexErrMsgIdAndTxt( "matRad:omc_matrad:Error","Call to num2str not successful");
    else
    {
        tmp = mxArrayToString(tmp2);        
        strcpy(input_items[nInput].value,tmp);
    }
    
    nInput++;
    sprintf(input_items[nInput].key,"pegs file");
    tmp_fieldpointer = getRequiredField(mcOpt,"pegsFile");
    tmp = mxArrayToString(tmp_fieldpointer);
    strcpy(input_items[nInput].value,tmp);
    
    nInput++;
    sprintf(input_items[nInput].key,"pgs4form file");
    tmp_fieldpointer = getRequiredField(mcOpt,"pgs4formFile");
    tmp = mxArrayToString(tmp_fieldpointer);
    strcpy(input_items[nInput].value,tmp);
    
    nInput++;
    sprintf(input_items[nInput].key,"data folder");
    tmp_fieldpointer = getRequiredField(mcOpt,"dataFolder");
    tmp = mxArrayToString(tmp_fieldpointer);
    strcpy(input_items[nInput].value,tmp);
    
    nInput++;
    sprintf(input_items[nInput].key,"output folder");
    tmp_fieldpointer = getRequiredField(mcOpt,"outputFolder");    
    tmp = mxArrayToString(tmp_fieldpointer);
    strcpy(input_items[nInput].value,tmp);

    tmp_fieldpointer = mxGetField(mcOpt,0,"relDoseThreshold");
    if (tmp_fieldpointer)
        dijOptions.relDoseThreshold = mxGetScalar(tmp_fieldpointer);
    /* The forward mode is the same source and the same physics, so it reads
     its options out of the same fields rather than out of a second set of
     names. Only two things differ, and both are the point of the mode:
     nHistories counts the whole calculation instead of one beamlet, and the
     dose threshold has nothing to prune. */
    forwardOptions.nhist = dijOptions.nhist;
    forwardOptions.nbatch = dijOptions.nbatch;
    forwardOptions.charge = dijOptions.charge;
    forwardOptions.sourceGeometry = dijOptions.sourceGeometry;
    forwardOptions.sourceGaussianWidth = dijOptions.sourceGaussianWidth;

    /* Dose in Gy for the weights given. Setting this to 0 asks for the mean
     deposited energy instead, the way omc_dosxyz's 'iout' does. */
    forwardOptions.outputDose = 1;
    tmp_fieldpointer = mxGetField(mcOpt,0,"outputDose");
    if (tmp_fieldpointer)
        forwardOptions.outputDose = (int) mxGetScalar(tmp_fieldpointer) != 0;

    if (verbose_flag > 0 && omcMode == OMC_MODE_FORWARD_BEAMLET)
        mexPrintf("ompMC mode '%s': %d histories over all beamlets together, "
                  "not per beamlet.\n",
                  modeNames[omcMode], forwardOptions.nhist);

    /* nInput is the index the last block wrote, so the count is one more */
    input_idx = nInput + 1;

    if (verbose_flag > 1)
    {
        mexPrintf("Input Options:\n");
        for (int iInput = 0; iInput < input_idx; iInput++)
            mexPrintf("%s: %s\n",input_items[iInput].key,input_items[iInput].value);
    }
          
    return;
}


/******************************************************************************/
/* Geometry definitions */
void initPhantom() {
    
    /* Get phantom information from matRad */
    //int ngeostructfields;
    mwSize nmaterials;
    const mwSize *materialdim;
    mxArray *tmp_fieldpointer;

    //ngeostructfields = mxGetNumberOfFields(mcGeo);

    /* Get number of media and media names. This info is saved in media struct */
    tmp_fieldpointer = mxGetField(mcGeo,0,"material");

    if (tmp_fieldpointer == NULL)
        mexErrMsgIdAndTxt( "matRad:omc_matrad:Error","No materials specified!");
    
    
    materialdim = mxGetDimensions(tmp_fieldpointer);
    nmaterials = materialdim[0];    
    media.nmed = nmaterials;
    
    mwIndex tmpSubs[2];
    mwSize iMat;
    mwIndex linIx;
    mxArray* tmpCellPointer;
    
    for (iMat = 0; iMat < nmaterials; ++iMat) 
    {
        tmpSubs[0] = (mwIndex) iMat;
        tmpSubs[1] = 0;                        

        linIx = mxCalcSingleSubscript(tmp_fieldpointer,2,tmpSubs);

        tmpCellPointer = mxGetCell(tmp_fieldpointer,linIx);
        
        if (tmpCellPointer == NULL)
            mexErrMsgIdAndTxt("matRad:omc_matrad:Error","Material could not be read!");
        
        char *tmp;
        tmp = mxArrayToString(tmpCellPointer);
        
        if (tmp == NULL)
            mexErrMsgIdAndTxt( "matRad:omc_matrad:Error","Material string could not be read!");
        
        tmp = strcpy(media.med_names[iMat],tmp);        
    }

    /* Get boundaries, density and material index for each voxel */
    const mwSize *cubeDim = mxGetDimensions(cubeRho);        
    
    geometry.isize = cubeDim[0];
    geometry.jsize = cubeDim[1];
    geometry.ksize = cubeDim[2];
    
    tmp_fieldpointer = mxGetField(mcGeo,0,"xBounds");    
    geometry.xbounds = mxGetPr(tmp_fieldpointer);
    tmp_fieldpointer = mxGetField(mcGeo,0,"yBounds");    
    geometry.ybounds = mxGetPr(tmp_fieldpointer);
    tmp_fieldpointer = mxGetField(mcGeo,0,"zBounds");    
    geometry.zbounds = mxGetPr(tmp_fieldpointer);
    
    geometry.med_densities = mxGetPr(cubeRho);

    geometry.med_indices = (int*)mxGetPr(cubeMatIx);

    omcGeomDetectSpacing();

    /* Summary with geometry information */
    if (verbose_flag > 1)
        mexPrintf("Number of media in phantom : %d\n", media.nmed);
    if (verbose_flag > 2)
    {
        mexPrintf("Media names: ");
        for (int i=0; i<media.nmed; i++) {
            mexPrintf("%s, ", media.med_names[i]);
        }
        mexPrintf("\n");
    }
    if (verbose_flag > 1) 
        mexPrintf("Number of voxels on each direction (X,Y,Z) : (%d, %d, %d)\n",geometry.isize, geometry.jsize, geometry.ksize);
    
    if (verbose_flag > 2) {
        mexPrintf("Minimum and maximum boundaries on each direction : \n");
        mexPrintf("\tX (cm) : %lf, %lf\n",
            geometry.xbounds[0], geometry.xbounds[geometry.isize]);
        mexPrintf("\tY (cm) : %lf, %lf\n",
            geometry.ybounds[0], geometry.ybounds[geometry.jsize]);
        mexPrintf("\tZ (cm) : %lf, %lf\n",
            geometry.zbounds[0], geometry.zbounds[geometry.ksize]);
    }
    return;
}

void cleanPhantom() {
    
    /* The memory inside geometry structure is shared with Matlab, therefore 
    it is not freed here */
    
    return;
}


/******************************************************************************/
/* Source definitions */

/* The beamlet source, taken straight from the mcSrc struct. Every array but
 the beam index is used in place, so mcSrc has to stay alive for the whole
 call -- it does, it is one of the inputs. */
static struct OmcBeamletSource beamletSource;
static int *beamIndex = NULL;   // 0 based, converted from mcSrc.iBeam

static const double *getSourceArray(const char *name) {

    mxArray *field = mxGetField(mcSrc, 0, name);

    if (field == NULL) {
        mexErrMsgIdAndTxt("matRad:omc_matrad:missingField",
            "Required field '%s' is missing from the mcSrc struct.", name);
    }
    if (!mxIsDouble(field) || mxIsComplex(field) || mxIsSparse(field)) {
        mexErrMsgIdAndTxt("matRad:omc_matrad:invalidField",
            "Field 'mcSrc.%s' must be a real double array.", name);
    }

    return mxGetPr(field);
}

static void initSource(void) {

    mxArray *field = mxGetField(mcSrc, 0, "nBixels");
    if (field == NULL) {
        mexErrMsgIdAndTxt("matRad:omc_matrad:missingField",
            "Required field 'nBixels' is missing from the mcSrc struct.");
    }

    beamletSource.nbeamlets = (int) mxGetScalar(field);

    if (verbose_flag > 1)
        mexPrintf("%s%d\n", "Total Number of Beamlets:",
                  beamletSource.nbeamlets);

    /* The beam index is the one field that cannot be shared: matRad counts
     beams from 1 and hands them over as doubles, the engine wants 0 based
     ints. */
    const double *iBeamPerBeamlet = getSourceArray("iBeam");

    beamIndex = (int*) malloc(beamletSource.nbeamlets*sizeof(int));
    for(int i=0; i<beamletSource.nbeamlets; i++) {
        beamIndex[i] = (int) iBeamPerBeamlet[i] - 1; // C indexing style
    }
    beamletSource.ibeam = beamIndex;

    beamletSource.xsource = getSourceArray("xSource");
    beamletSource.ysource = getSourceArray("ySource");
    beamletSource.zsource = getSourceArray("zSource");

    beamletSource.xcorner = getSourceArray("xCorner");
    beamletSource.ycorner = getSourceArray("yCorner");
    beamletSource.zcorner = getSourceArray("zCorner");

    beamletSource.xside1 = getSourceArray("xSide1");
    beamletSource.yside1 = getSourceArray("ySide1");
    beamletSource.zside1 = getSourceArray("zSide1");

    beamletSource.xside2 = getSourceArray("xSide2");
    beamletSource.yside2 = getSourceArray("ySide2");
    beamletSource.zside2 = getSourceArray("zSide2");

    /* The collimation of the forward mode: one weight per beamlet, which the
     engine turns into that beamlet's share of the histories. Only the shape
     is checked here; the engine rejects negative and NaN weights, and a set
     that is zero everywhere. */
    bixelWeights = NULL;

    if (omcMode == OMC_MODE_FORWARD_BEAMLET) {
        mxArray *weights = mxGetField(mcSrc, 0, "bixelWeights");

        if (weights == NULL) {
            mexErrMsgIdAndTxt("matRad:omc_matrad:missingField",
                "Mode '%s' needs one weight per beamlet in "
                "'mcSrc.bixelWeights'.", modeNames[OMC_MODE_FORWARD_BEAMLET]);
        }
        if (!mxIsDouble(weights) || mxIsComplex(weights) ||
            mxIsSparse(weights)) {
            mexErrMsgIdAndTxt("matRad:omc_matrad:invalidField",
                "Field 'mcSrc.bixelWeights' must be a real double array.");
        }
        if ((int) mxGetNumberOfElements(weights) != beamletSource.nbeamlets) {
            mexErrMsgIdAndTxt("matRad:omc_matrad:invalidField",
                "'mcSrc.bixelWeights' has %d entries, but there are %d "
                "beamlets.", (int) mxGetNumberOfElements(weights),
                beamletSource.nbeamlets);
        }

        bixelWeights = mxGetPr(weights);
    }

    return;
}

static void cleanSource(void) {

    /* Everything else is shared with MATLAB and freed there */
    free(beamIndex);
    beamIndex = NULL;

    return;
}

/******************************************************************************/
/* Collecting the results. The engine hands over one finished beamlet at a
 time and this grows the two sparse matrices MATLAB gets back, which is what
 the beamlet loop used to do inline. */

struct SparseDij {
    mxArray *dose;              // the matrices being filled
    mxArray *variance;          // NULL unless a second output was asked for

    mwSize nCubeElements;
    mwSize nbeamlets;

    double *sr;                 // dose: values, row indices, column starts
    mwIndex *irs;
    mwIndex *jcs;

    double *sr_var;             // the same for the variance
    mwIndex *irs_var;
    mwIndex *jcs_var;

    mwIndex linIx;              // values written so far
    mwSize nzmax;               // values the arrays have room for
    double percentage_steps;    // steps in which the sparse matrix is allocated
    double percent_sparse;      // fraction currently allocated for
    int reallocations;
};

static void appendBeamlet(int ibeamlet, int nvoxels, const int *voxels,
                          const double *dose, const double *variance,
                          void *user) {

    struct SparseDij *dij = (struct SparseDij*) user;

    /* The number of new non-zero values is the current linear index plus the
     entries coming from this beamlet */
    mwSize newnnz = (mwSize)nvoxels + (mwSize) dij->linIx;

    /* Check if we need to reallocate for sparse matrix */
    if (newnnz > dij->nzmax) {
        mwSize oldnzmax = dij->nzmax;
        dij->percent_sparse += dij->percentage_steps;
        dij->nzmax = (mwSize) ceil((double)dij->nCubeElements
                                   *(double)dij->nbeamlets
                                   *dij->percent_sparse);

        /* Make sure nzmax increases at least by 1. */
        if (oldnzmax == dij->nzmax) {
            dij->nzmax++;
        }

        /* Check that the new nmax is large enough and if not, also adjust
        the percentage_steps since we seem to have set it too small for this
        particular use case */
        if (dij->nzmax < newnnz) {
            dij->nzmax = newnnz;
            dij->percent_sparse = (double)dij->nzmax/dij->nCubeElements;
            dij->percentage_steps = dij->percent_sparse;
        }

        if (verbose_flag > 2) {
            mexPrintf("Reallocating Sparse Matrix from nzmax=%d to nzmax=%d\n",
                      (int)oldnzmax, (int)dij->nzmax);
        }

        /* Set new nzmax and reallocate more memory */
        mxSetNzmax(dij->dose, dij->nzmax);
        mxSetPr(dij->dose, (double *) mxRealloc(dij->sr,
            dij->nzmax*sizeof(double)));
        mxSetIr(dij->dose, (mwIndex *) mxRealloc(dij->irs,
            dij->nzmax*sizeof(mwIndex)));

        /* Use the new pointers */
        dij->sr  = mxGetPr(dij->dose);
        dij->irs = mxGetIr(dij->dose);

        if (dij->variance) {
            /* Set new nzmax and reallocate more memory */
            mxSetNzmax(dij->variance, dij->nzmax);
            mxSetPr(dij->variance, (double *) mxRealloc(dij->sr_var,
                dij->nzmax*sizeof(double)));
            mxSetIr(dij->variance, (mwIndex *) mxRealloc(dij->irs_var,
                dij->nzmax*sizeof(mwIndex)));

            /* Use the new pointers */
            dij->sr_var  = mxGetPr(dij->variance);
            dij->irs_var = mxGetIr(dij->variance);
        }

        dij->reallocations++;
    }

    /* Writing past the arrays would corrupt the heap rather than produce a
     wrong number, so make sure of the arithmetic above before trusting it */
    if (newnnz > dij->nzmax) {
        mexErrMsgIdAndTxt("matRad:omc_matrad:sparseOverflow",
            "Beamlet %d needs room for %d values but only %d are allocated.",
            ibeamlet, (int)newnnz, (int)dij->nzmax);
    }

    //Populate sparse matrix arrays
    for (int n = 0; n < nvoxels; n++) {
        dij->sr[dij->linIx] = dose[n];
        dij->irs[dij->linIx] = voxels[n];

        if (dij->variance) {
            dij->sr_var[dij->linIx] = variance[n];
            dij->irs_var[dij->linIx] = voxels[n];
        }
        dij->linIx++;
    }

    dij->jcs[ibeamlet+1] = dij->linIx;
    if (dij->variance) {
        dij->jcs_var[ibeamlet+1] = dij->linIx;
    }

    return;
}

/******************************************************************************/
/* Progress reporting for the main simulation loop. If the caller supplied
 options.progressCallback, report through it (progress in [0,1]) and let
 the MATLAB side own any waitbar/handle lifecycle. Otherwise fall back to
 the built-in waitbar, lazily opened on first use, when verbose_flag > 1. */
static mxArray *builtinWaitbarHandle = NULL;

static const char *progressMessage =
    "calculate dose influence matrix for photons (ompMC) ...";

static int reportProgress(double progress, void *user) {

    (void)user;

    /* Always 1: the MATLAB interface has no way of asking to stop, and a
     partial dose influence matrix is not something it could hand back. */
    if (progressCallback != NULL) {
        mxArray *progressArg = mxCreateDoubleScalar(progress);
        mxArray *cbArgs[2] = { progressCallback, progressArg };
        mexCallMATLAB(0, NULL, 2, cbArgs, "feval");
        mxDestroyArray(progressArg);
        return 1;
    }

    if (verbose_flag <= 1)
        return 1;

    mxArray *progressArg = mxCreateDoubleScalar(progress);
    mxArray *messageArg = mxCreateString(progressMessage);
    mxArray *waitbarOutput[1];

    if (builtinWaitbarHandle == NULL) {
        mxArray *waitbarInputs[2] = { progressArg, messageArg };
        mexCallMATLAB(1, waitbarOutput, 2, waitbarInputs, "waitbar");
        builtinWaitbarHandle = waitbarOutput[0];
    } else {
        mxArray *waitbarInputs[3] = { progressArg, builtinWaitbarHandle, messageArg };
        mexCallMATLAB(0, waitbarOutput, 3, waitbarInputs, "waitbar");
    }

    mxDestroyArray(progressArg);
    mxDestroyArray(messageArg);

    return 1;
}

static void closeProgress(void) {
    if (builtinWaitbarHandle != NULL) {
        mxArray *waitbarOutput[1];
        mxArray *waitbarInputs[1] = { builtinWaitbarHandle };
        mexCallMATLAB(0, waitbarOutput, 1, waitbarInputs, "close");
        mxDestroyArray(builtinWaitbarHandle);
        builtinWaitbarHandle = NULL;
    }
}

/******************************************************************************/
/* mode 'forward_beamlet': every beamlet at once, weighted by mcSrc.bixelWeights,
 into one dense dose cube.

 The cube comes back with the same [isize jsize ksize] shape matRad hands the
 density cube over in. That needs no transposition: the engine indexes a voxel
 as ix + iy*isize + iz*isize*jsize, which is exactly how MATLAB lays out an
 array of that size. */

static void runForward(int nlhs, mxArray *plhs[], double tbegin,
                       struct OmcSpectrum *spectrum) {

    mwSize dims[3];
    dims[0] = (mwSize) geometry.isize;
    dims[1] = (mwSize) geometry.jsize;
    dims[2] = (mwSize) geometry.ksize;

    plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
    double *dose = mxGetPr(plhs[0]);

    /* The relative uncertainty is only computed when it is asked for, the
     same way the Dij mode decides about the variance. */
    double *uncertainty = NULL;

    if (nlhs >= 2) {
        plhs[1] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
        uncertainty = mxGetPr(plhs[1]);
    }

    if (verbose_flag > 0)
        mexPrintf("done!\n");

    if (verbose_flag > 2)
        mexPrintf("Execution time up to this point : %8.2f seconds\n",
                  (omc_get_time() - tbegin));

    if (verbose_flag > 0)
        mexPrintf("Running ompMC simulation...\n");

    struct OmcForwardCallbacks callbacks;
    callbacks.progress = reportProgress;
    callbacks.user = NULL;

    struct OmcForwardSummary summary;

    int finished = omcCalcForward(&forwardOptions, &beamletSource,
                                  bixelWeights, spectrum, dose, uncertainty,
                                  &callbacks, &summary);

    if (verbose_flag > 0)
        mexPrintf("Simulation finished!\nFinalizing output...\n");

    closeProgress();

    /* reportProgress() never asks to stop, so this cannot happen -- but the
     engine is allowed to return a cube it has not filled, and handing that
     back as a dose would be worse than saying so. */
    if (!finished) {
        mexErrMsgIdAndTxt("matRad:omc_matrad:aborted",
            "The forward calculation was stopped before any result was "
            "available.");
    }

    if (verbose_flag >= 3) {
        mexPrintf("Ran %d histories over %d of %d weighted beamlets.\n",
                  summary.nhist, summary.nsampled, summary.nweighted);
        mexPrintf("Deposited %.2f%% of the incident energy.\n",
                  100.0*summary.energyFraction);
    }

    return;
}

/******************************************************************************/
/* mode 'dij': one sparse column per beamlet, weights applied by the caller. */

static void runDij(int nlhs, mxArray *plhs[], int gridsize, double tbegin,
                   struct OmcSpectrum *spectrum) {

        /* Create output matrix */
        struct SparseDij dij;
        dij.nCubeElements = (mwSize) gridsize;
        dij.nbeamlets = (mwSize) beamletSource.nbeamlets;
        dij.percentage_steps = 0.01;            // steps in which it is allocated
        dij.percent_sparse = dij.percentage_steps;
        dij.nzmax = (mwSize) ceil((double)dij.nCubeElements*(double)dij.nbeamlets
                                  *dij.percent_sparse);
        dij.linIx = 0;
        dij.reallocations = 0;

        plhs[0] = mxCreateSparse(dij.nCubeElements, dij.nbeamlets, dij.nzmax, mxREAL);
        dij.dose = plhs[0];
        dij.sr  = mxGetPr(plhs[0]);
        dij.irs = mxGetIr(plhs[0]);
        dij.jcs = mxGetJc(plhs[0]);
        dij.jcs[0] = 0;

        dij.variance = NULL;
        dij.sr_var = NULL;
        dij.irs_var = NULL;
        dij.jcs_var = NULL;

        if (dijOptions.wantVariance)
        {
            plhs[1] = mxCreateSparse(dij.nCubeElements, dij.nbeamlets, dij.nzmax, mxREAL);
            dij.variance = plhs[1];
            dij.sr_var  = mxGetPr(plhs[1]);
            dij.irs_var = mxGetIr(plhs[1]);
            dij.jcs_var = mxGetJc(plhs[1]);
            dij.jcs_var[0] = 0;
        }

        if (verbose_flag > 0)
            mexPrintf("done!\n");

        /* Execution time up to this point */
        if (verbose_flag > 2)
            mexPrintf("Execution time up to this point : %8.2f seconds\n",(omc_get_time() - tbegin));

        if (verbose_flag > 0)
            mexPrintf("Running ompMC simulation...\n");

        struct OmcDijCallbacks callbacks;
        callbacks.beamlet = appendBeamlet;
        callbacks.progress = reportProgress;
        callbacks.user = &dij;

        omcCalcDij(&dijOptions, &beamletSource, spectrum, &callbacks);

        /* Print some output and execution time up to this point */
        if (verbose_flag > 0)
            mexPrintf("Simulation finished!\nFinalizing output...\n");

        closeProgress();

        if (verbose_flag >= 3)
            mexPrintf("Sparse MC Dij has %d (%f percent) elements!\n", (int)dij.linIx,
                (double)dij.linIx/((double)dij.nCubeElements*(double)dij.nbeamlets));

        if (verbose_flag >= 3)
            mexPrintf("Needed %d sparse matrix reallocations.\n", dij.reallocations);

        /* Truncate the matrix to the exact size by reallocation */
        mxSetNzmax(plhs[0], dij.linIx);
        mxSetPr(plhs[0], mxRealloc(dij.sr, dij.linIx*sizeof(double)));
        mxSetIr(plhs[0], mxRealloc(dij.irs, dij.linIx*sizeof(mwIndex)));

        mwIndex *irs = mxGetIr(plhs[0]);

        //Check output
        if (verbose_flag >= 3)
            mexPrintf("Verifying sparse Matrix... ");
        for (mwIndex ix = 0; ix < dij.linIx; ix++)
        {
            mwIndex currIx = irs[ix];

            if (currIx > (mwIndex)gridsize)
                mexPrintf("Invalid dose-cube index %d at linear index %d in sparse matrix check!",(int)currIx,(int)dij.linIx);
        }
        if (verbose_flag >= 3)
            mexPrintf("done!\n");

        if (dijOptions.wantVariance) {
            /* Truncate the matrix to the exact size by reallocation */
            mxSetNzmax(plhs[1], dij.linIx);
            mxSetPr(plhs[1], mxRealloc(dij.sr_var, dij.linIx*sizeof(double)));
            mxSetIr(plhs[1], mxRealloc(dij.irs_var, dij.linIx*sizeof(mwIndex)));
        }
    return;
}

/******************************************************************************/
/* omc_matrad main function */
void mexFunction (int nlhs, mxArray *plhs[],    // output of the function
    int nrhs, const mxArray *prhs[])            // input of the function
{
    /* A single "version"/"-v"/"--version" string argument is a version query,
     answered without locking the MEX file or touching any of the dose
     calculation machinery below. */
    if (nrhs == 1 && mxIsChar(prhs[0])) {
        char *arg = mxArrayToString(prhs[0]);
        int isVersionQuery = arg != NULL &&
            (strcmp(arg, "version") == 0 ||
             strcmp(arg, "-v") == 0 ||
             strcmp(arg, "--version") == 0);
        mxFree(arg);

        if (isVersionQuery) {
            if (nlhs > 1) {
                mexErrMsgIdAndTxt("matRad:omc_matrad:invalidNumOutputs",
                    "Too many output arguments.");
            }
            if (nlhs == 1) {
                plhs[0] = mxCreateString(OMPMC_VERSION_STRING);
            } else {
                mexPrintf("ompMC version %s\n", OMPMC_VERSION_STRING);
            }
            return;
        }
    }

    /* Execution time measurement */
    double tbegin;
    tbegin = omc_get_time();

    /* Keep this MEX file resident for the rest of the MATLAB session. Once an
     OpenMP parallel region has run, the worker threads of the OpenMP runtime
     outlive the MEX file, and unloading it -- through "clear mex" or when
     MATLAB exits -- takes the runtime down with it while those threads are
     still alive. With the Microsoft runtime (vcomp140, used by MSVC builds)
     that reliably crashes MATLAB with an access violation. Locking is the
     supported way out; the price is that a rebuilt MEX file is only picked up
     after restarting MATLAB. */
    if (!mexIsLocked()) {
        mexLock();
    }

    /* Parsing program options */

    /* Check for proper number of input and output arguments */
    if (nrhs != 5) {
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:invalidNumInputs","Two or three input arguments required.");
    }
    if(nlhs > 2){
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:invalidNumOutputs","Too many output arguments.");
    }

    mexPrintf("Running ompMC version %s...\n", OMPMC_VERSION_STRING);

    /* Route the shared code's diagnostics into MATLAB before anything that
     might have something to report runs */
    initHost();

    parseInput(nrhs, prhs);
    dijOptions.wantVariance = (nlhs >= 2);

    if (verbose_flag > 0)
    {
        mexPrintf("Input successfully parsed!\n");
        mexPrintf("Initalizing ompMC...\n");
    }

    /* Get information of OpenMP environment */
#ifdef _OPENMP
    int omp_size = omp_get_num_procs();
    if (verbose_flag > 1)
        mexPrintf("Number of OpenMP threads: %d\n", omp_size);
    omp_set_num_threads(omp_size);
#else
    if (verbose_flag > 1)
        mexPrintf("ompMC compiled without OpenMP support. Serial execution.\n");
#endif

    /* Read geometry information from matRad and initialize geometry */
    initPhantom();

    /* With number of media and media names initialize the medium data */
    initMediaData();

    /* Initialize the source: first the energy spectrum, either passed in
     directly or read from file, then the beamlet apertures */
    struct OmcSpectrum spectrum;
    if (omcConfig.spectrumEnergy != NULL) {
        omcSpectrumFromHistogram(&spectrum, omcConfig.spectrumEnergy,
            omcConfig.spectrumFluence, omcConfig.spectrumNbins,
            omcConfig.spectrumEnMin, omcConfig.spectrumMode);
    }
    else if (omcConfig.useMonoEnergy) {
        omcSpectrumMonoenergetic(&spectrum, omcConfig.monoEnergy);
    }
    else {
        omcSpectrumFromFile(&spectrum, omcConfig.spectrumFile);
    }
    initSource();

    /* Initialize data on a region-by-region basis */
    initRegions();

    /* Initialize VRT data */
    initVrt();

    int gridsize = geometry.isize*geometry.jsize*geometry.ksize;

    if (omcMode == OMC_MODE_FORWARD_BEAMLET) {
        runForward(nlhs, plhs, tbegin, &spectrum);
    }
    else {
        runDij(nlhs, plhs, gridsize, tbegin, &spectrum);
    }

    /* Cleaning */
    cleanPhantom();
    cleanPhoton();
    cleanRayleigh();
    cleanPair();
    cleanElectron();
    cleanMscat();
    cleanSpin();
    cleanRegions();
    omcSpectrumFree(&spectrum);
    cleanSource();

    /* Get total execution time */
    if (verbose_flag > 0)
        mexPrintf("Finished! Total execution time : %8.5f seconds\n", (omc_get_time() - tbegin));

}

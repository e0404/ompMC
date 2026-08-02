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

#include "omc_geom.h"
#include "omc_host.h"
#include "omc_utilities.h"
#include "omc_random.h"
#include "ompmc.h"
#include "omc_score.h"
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

#if defined(_MSC_VER)
	//use __declspec(thread) instead of threadprivate to avoid 
	//error C3053. More information in:
	// https://stackoverflow.com/questions/12560243/using-threadprivate-directive-in-visual-studio 
	__declspec(thread) extern struct Stack stack;
#else
	extern struct Stack stack;
    #pragma omp threadprivate(stack)
#endif
extern struct Media media;
extern struct Pegs pegs_data;
extern struct Region region;

extern struct inputItems input_items[];     // key,value pairs
extern int input_idx;                       // number of key,value pair

//Data Types and Structs

struct Source {
    int nmed;                   // number of media in phantom file
    int spectrum;               // 0 : monoenergetic, 1 : spectrum
    int charge;                 // 0 : photons, -1 : electron, +1 : positron
    
    /* For monoenergetic source */
    double energy;
    
    /* For spectrum */
    double deltak;              // number of elements in inverse CDF
    double *cdfinv1;            // energy value of bin
    double *cdfinv2;            // prob. that particle has energy xi
    
    /* Beamlets shape information */
    int nbeamlets;               // number of beamlets per beam
    int *ibeam;                  // index of beam per beamlet
    
    double *xsource;           // coordinates of the source of each beam
    double *ysource;          
    double *zsource;          
        
    double *xcorner;           // coordinates of the bixel corner
    double *ycorner;           
    double *zcorner;  
    
    double *xside1;           // coordinates of the first side of bixel
    double *yside1;           
    double *zside1;
    
    double *xside2;           // coordinates of the second side of bixel
    double *yside2;           
    double *zside2;
        
};
struct Source source;



enum sourceGeometryType {POINT, GAUSSIAN};
struct OmcConfig {
    //Simulation parameters
    int nHist;
    int nBatch;
    double doseThreshold;

    //Source Parameters
    double monoEnergy;
    char * spectrumFile;

    /* Spectrum handed over directly from MATLAB in mcOpt.spectrum, used
     instead of reading spectrumFile when given. The two pointers alias the
     caller's mxArrays, which stay alive for the whole call, and are NULL
     when no spectrum was passed. */
    const double *spectrumEnergy;   // upper energy of each bin, MeV, ascending
    const double *spectrumFluence;  // relative number of particles per bin
    int spectrumNbins;
    double spectrumEnMin;           // lower energy of the first bin, MeV
    int spectrumMode;               // 0 : counts/bin, 1 : counts/MeV

    enum sourceGeometryType sourceGeometry;
    double sourceGaussianWidth; //Assuming 5mm FWHM penumbra if the source is gaussian
};

struct OmcConfig omcConfig;

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

 The arrays are only validated here; they are turned into the sampling tables
 in initSource(), together with the ones read from file. */
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
    omcConfig.nHist = 1e4;
    omcConfig.nBatch = 10;
    omcConfig.doseThreshold = 0.01;
    omcConfig.monoEnergy = 0.1;
    /* Reset explicitly rather than relying on the zero initialization of the
     global: the MEX file stays locked in memory, so a spectrum passed in one
     call would otherwise still be pointed at by the next one, which no longer
     owns those arrays. */
    omcConfig.spectrumEnergy = NULL;
    omcConfig.spectrumFluence = NULL;
    omcConfig.spectrumNbins = 0;
    omcConfig.spectrumEnMin = 0.0;
    omcConfig.spectrumMode = 0;
    omcConfig.sourceGeometry = POINT;
    omcConfig.sourceGaussianWidth = 0.2123; //Assuming 5mm FWHM penumbra if the source is gaussian
    
    
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
    int nInput = 0;
        
    tmp_fieldpointer = mxGetField(mcOpt,0,"nHistories");
    
    //size_t nHistLength = mxGetNumberOfElements(tmp_fieldpointer);
    if (tmp_fieldpointer)    
        omcConfig.nHist = mxGetScalar(tmp_fieldpointer);
    
    tmp_fieldpointer = mxGetField(mcOpt,0,"nBatches");
    if (tmp_fieldpointer)
        omcConfig.nBatch = mxGetScalar(tmp_fieldpointer);

    tmp_fieldpointer = mxGetField(mcOpt,0,"sourceGaussianWidth");
    if (tmp_fieldpointer) {
        omcConfig.sourceGaussianWidth = mxGetScalar(tmp_fieldpointer);
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
            omcConfig.sourceGeometry = GAUSSIAN;
            mexPrintf("Using 'gaussian' source geometry with %f mm width...\n",omcConfig.sourceGaussianWidth);
        }
        else if (strcmp(sourceGeoTmpStr,"point") == 0)
        {
            omcConfig.sourceGeometry = POINT;
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

    /* The spectrum can either be passed in directly or read from file, with
     the passed one taking precedence when both are given. */
    tmp_fieldpointer = mxGetField(mcOpt,0,"spectrum");
    if (tmp_fieldpointer && !mxIsEmpty(tmp_fieldpointer)) {
        parseSpectrum(tmp_fieldpointer);

        if (mxGetField(mcOpt,0,"spectrumFile"))
            mexPrintf("Both 'spectrum' and 'spectrumFile' were given, using "
                "the passed spectrum and ignoring the file.\n");
    }

    tmp_fieldpointer = mxGetField(mcOpt,0,"spectrumFile");
    if (omcConfig.spectrumEnergy != NULL) {
        omcConfig.spectrumFile = NULL;
    }
    else if (tmp_fieldpointer) {
        size_t buflen = mxGetNumberOfElements(tmp_fieldpointer) + 1;
        omcConfig.spectrumFile = (char*) mxCalloc(buflen + 1,sizeof(char));
        if (mxGetString(tmp_fieldpointer, omcConfig.spectrumFile, buflen) != 0)
            mexErrMsgIdAndTxt("MATLAB:explore:invalidStringArray","Invalid string for path to spectrum file!");

    }
    else
    {
        size_t buflen = 255;
        omcConfig.spectrumFile = (char*) mxCalloc(buflen + 1,sizeof(char));
        strcpy(omcConfig.spectrumFile, "./spectra/mohan6.spectrum");
    }

    tmp_fieldpointer = mxGetField(mcOpt,0,"monoEnergy");
    if (tmp_fieldpointer)
        omcConfig.monoEnergy = mxGetScalar(tmp_fieldpointer);    
    
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
        omcConfig.doseThreshold = mxGetScalar(tmp_fieldpointer);
    
    
    input_idx = nInput;
    
    if (verbose_flag > 1)
    {
        mexPrintf("Input Options:\n");
        for (int iInput = 0; iInput < nInput; iInput++)
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

/******************************************************************************/
/* Source definitions */
const int MXEBIN = 200;     // number of energy bins of spectrum
const int INVDIM = 1000;    // number of bins in inverse CDF

/* Build the tables used to sample the incident energy, from a histogram of
 nensrc bins with upper energies ensrcd[], lower edge enmin of the first bin
 and per-bin probabilities srcpdf[] in counts/bin. Shared by the spectrum
 read from file and the one passed in from MATLAB, so that both are sampled
 in exactly the same way. */
static void initSpectrumCdf(const double *ensrcd, const double *srcpdf,
                            int nensrc, double enmin) {

    if (verbose_flag > 1)
        mexPrintf("Energy ranges from %f to %f MeV\n", enmin, ensrcd[nensrc - 1]);

    /* Initialization routine to calculate the inverse of the
     cumulative probability distribution that is used during execution to
     sample the incident particle energy. */
    double *srccdf = malloc(nensrc*sizeof(double));

    srccdf[0] = srcpdf[0];
    for (int i=1; i<nensrc; i++) {
        srccdf[i] = srccdf[i-1] + srcpdf[i];
    }

    double fnorm = 1.0/srccdf[nensrc - 1];
    double binsok = 0.0;
    source.deltak = INVDIM; /* number of elements in inverse CDF */
    double gridsz = 1.0f/source.deltak;

    for (int i=0; i<nensrc; i++) {
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

    if (verbose_flag > 1 && binsok != 0.0) {
        mexPrintf("Warning! Some of normalized bin probabilities are so small that bins may be missed.\n");
    }

    /* Calculate cdfinv. This array allows the rapid sampling for the
     energy by precomputing the results for a fine grid. */
    source.cdfinv1 = malloc(source.deltak*sizeof(double));
    source.cdfinv2 = malloc(source.deltak*sizeof(double));
    double ak;

    for (int k=0; k<source.deltak; k++) {
        ak = (double)k*gridsz;
        int i;

        for (i=0; i<nensrc; i++) {
            if (ak <= srccdf[i]) {
                break;
            }
        }

        /* We should fall here only through the above break sentence. */
        if (i != 0) {
            source.cdfinv1[k] = ensrcd[i - 1];
        }
        else {
            source.cdfinv1[k] = enmin;
        }
        source.cdfinv2[k] = ensrcd[i] - source.cdfinv1[k];

    }

    free(srccdf);

    return;
}

/* Read the spectrum from the .spectrum file given in mcOpt.spectrumFile */
static void initSpectrumFromFile(void) {

    char buffer[BUFFER_SIZE];
    char* fstatus;

    //removeSpaces(omcConfig.spectrumFile, buffer);

    /* Open .source file */
    FILE *fp;

    if ((fp = fopen(omcConfig.spectrumFile, "r")) == NULL) {
        omcFail("matRad:omc_matrad:spectrumFile",
            "Unable to open spectrum file: %s", omcConfig.spectrumFile);
    }

    if (verbose_flag > 2)
        mexPrintf("Path to spectrum file : %s\n", omcConfig.spectrumFile);

    /* Read spectrum file title */
    fstatus = fgets(buffer, BUFFER_SIZE, fp);
    if (fstatus == NULL)
        mexErrMsgIdAndTxt("matRad:omc_matrad:Error","Could not parse spectrum file.\n");

    if (verbose_flag > 1)
        mexPrintf("Spectrum file title: %s", buffer);


    /* Read number of bins and spectrum type */
    double enmin;   /* lower energy of first bin */
    int nensrc;     /* number of energy bins in spectrum histogram */
    int imode;      /* 0 : histogram counts/bin, 1 : counts/MeV*/

    fstatus = fgets(buffer, BUFFER_SIZE, fp);
    if (fstatus == NULL)
        mexErrMsgIdAndTxt("matRad:omc_matrad:Error","Could not parse spectrum file.\n");

    sscanf(buffer, "%d %lf %d", &nensrc, &enmin, &imode);

    if (nensrc > MXEBIN) {
        omcFail("matRad:omc_matrad:spectrumFile",
            "Number of energy bins = %d is greater than max allowed = %d. "
            "Increase MXEBIN macro!", nensrc, MXEBIN);
    }

    /* upper energy of bin i in MeV */
    double *ensrcd = malloc(nensrc*sizeof(double));
    /* prob. of finding a particle in bin i */
    double *srcpdf = malloc(nensrc*sizeof(double));

    /* Read spectrum information */
    for (int i=0; i<nensrc; i++) {
        fstatus = fgets(buffer, BUFFER_SIZE, fp);
        if (fstatus == NULL)
            mexErrMsgIdAndTxt("matRad:omc_matrad:Error","Could not parse spectrum file.\n");

        sscanf(buffer, "%lf %lf", &ensrcd[i], &srcpdf[i]);
    }
    if (verbose_flag > 2)
        mexPrintf("Have read %d input energy bins from spectrum file.\n", nensrc);

    if (imode == 0) {
        if (verbose_flag > 2)
            mexPrintf("Counts/bin assumed.\n");
    }
    else if (imode == 1) {
        if (verbose_flag > 2)
            mexPrintf("Counts/MeV assumed.\n");
        srcpdf[0] *= (ensrcd[0] - enmin);
        for(int i=1; i<nensrc; i++) {
            srcpdf[i] *= (ensrcd[i] - ensrcd[i - 1]);
        }
    }
    else {
        omcFail("matRad:omc_matrad:spectrumFile",
            "Invalid mode number in spectrum file.");
    }

    initSpectrumCdf(ensrcd, srcpdf, nensrc, enmin);

    /* Cleaning */
    fclose(fp);
    free(ensrcd);
    free(srcpdf);

    return;
}

/* Use the spectrum passed in through mcOpt.spectrum, already validated in
 parseSpectrum(). The bin probabilities are copied because counts/MeV input
 has to be scaled by the bin widths, and the array belongs to MATLAB. */
static void initSpectrumFromArrays(void) {

    int nensrc = omcConfig.spectrumNbins;
    double enmin = omcConfig.spectrumEnMin;
    const double *ensrcd = omcConfig.spectrumEnergy;

    double *srcpdf = malloc(nensrc*sizeof(double));
    for (int i=0; i<nensrc; i++) {
        srcpdf[i] = omcConfig.spectrumFluence[i];
    }

    if (omcConfig.spectrumMode == 1) {
        if (verbose_flag > 2)
            mexPrintf("Counts/MeV assumed.\n");
        srcpdf[0] *= (ensrcd[0] - enmin);
        for (int i=1; i<nensrc; i++) {
            srcpdf[i] *= (ensrcd[i] - ensrcd[i - 1]);
        }
    }
    else if (verbose_flag > 2) {
        mexPrintf("Counts/bin assumed.\n");
    }

    if (verbose_flag > 2)
        mexPrintf("Have taken %d energy bins from the passed spectrum.\n", nensrc);

    initSpectrumCdf(ensrcd, srcpdf, nensrc, enmin);

    free(srcpdf);

    return;
}

void initSource() {

    /* Read the charge of the source particles from mcOpt.charge, which
     parseInput() has put into the input items. Without this the field was
     accepted and then ignored, leaving source.charge at its static zero, so
     an electron or positron source silently ran as photons. */
    char buffer[BUFFER_SIZE];
    if (getInputValue(buffer, "charge") != 1) {
        omcFail("matRad:omc_matrad:missingInput",
            "Can not find 'charge' key on input file.");
    }

    char *endptr;
    long charge = strtol(buffer, &endptr, 10);

    /* strtol() stops at the first character it cannot use and returns 0 for a
     string it could not read at all, which would quietly turn a bad value
     into a photon source */
    while (isspace((unsigned char)*endptr)) {
        endptr++;
    }
    if (endptr == buffer || *endptr != '\0') {
        mexErrMsgIdAndTxt("matRad:omc_matrad:invalidCharge",
            "Option 'charge' is '%s', expected -1, 0 or +1.", buffer);
    }
    if (charge < -1 || charge > 1) {
        mexErrMsgIdAndTxt("matRad:omc_matrad:invalidCharge",
            "Option 'charge' is %ld, expected -1 for electrons, 0 for photons "
            "or +1 for positrons.", charge);
    }
    source.charge = (int)charge;

    if (verbose_flag > 1) {
        const char *particle = source.charge == 0 ? "photons" :
            (source.charge < 0 ? "electrons" : "positrons");
        mexPrintf("Source charge : %d (%s)\n", source.charge, particle);
    }

    source.spectrum = 1;    /* energy spectrum as default case */

    if (source.spectrum) {
        if (omcConfig.spectrumEnergy != NULL) {
            initSpectrumFromArrays();
        }
        else {
            initSpectrumFromFile();
        }
    }
    else {  /* monoenergetic source */
        source.energy = omcConfig.monoEnergy;
        mexPrintf("%f monoenergetic source\n", source.energy);

    }

    /* Parse data of the beamlets */
    unsigned int nfields;
    mxArray *tmp_fieldpointer;

    tmp_fieldpointer = mxGetField(mcSrc,0,"nBixels");
    nfields = mxGetScalar(tmp_fieldpointer);
    source.nbeamlets = nfields;
    
    if (verbose_flag > 1)
        mexPrintf("%s%d\n", "Total Number of Beamlets:", source.nbeamlets);
    
    tmp_fieldpointer = mxGetField(mcSrc,0,"iBeam");
    const double* iBeamPerBeamlet = mxGetPr(tmp_fieldpointer);
    
    source.ibeam = (int*) malloc(source.nbeamlets*sizeof(int));
    for(int i=0; i<source.nbeamlets; i++) {
        source.ibeam[i] = (int) iBeamPerBeamlet[i] - 1; // C indexing style
    }
        
    tmp_fieldpointer = mxGetField(mcSrc,0,"xSource");
    source.xsource = mxGetPr(tmp_fieldpointer);
    
    tmp_fieldpointer = mxGetField(mcSrc,0,"ySource");
    source.ysource = mxGetPr(tmp_fieldpointer);
    
    tmp_fieldpointer = mxGetField(mcSrc,0,"zSource");
    source.zsource = mxGetPr(tmp_fieldpointer);
            
    tmp_fieldpointer = mxGetField(mcSrc,0,"xCorner");
    source.xcorner = mxGetPr(tmp_fieldpointer);
    
    tmp_fieldpointer = mxGetField(mcSrc,0,"yCorner");
    source.ycorner = mxGetPr(tmp_fieldpointer);
    
    tmp_fieldpointer = mxGetField(mcSrc,0,"zCorner");
    source.zcorner = mxGetPr(tmp_fieldpointer);
    
    tmp_fieldpointer = mxGetField(mcSrc,0,"xSide1");
    source.xside1 = mxGetPr(tmp_fieldpointer);
    
    tmp_fieldpointer = mxGetField(mcSrc,0,"ySide1");
    source.yside1 = mxGetPr(tmp_fieldpointer);
    
    tmp_fieldpointer = mxGetField(mcSrc,0,"zSide1");
    source.zside1 = mxGetPr(tmp_fieldpointer);
    
    tmp_fieldpointer = mxGetField(mcSrc,0,"xSide2");
    source.xside2 = mxGetPr(tmp_fieldpointer);
    
    tmp_fieldpointer = mxGetField(mcSrc,0,"ySide2");
    source.yside2 = mxGetPr(tmp_fieldpointer);
    
    tmp_fieldpointer = mxGetField(mcSrc,0,"zSide2");
    source.zside2 = mxGetPr(tmp_fieldpointer);    
    
    return;
}

void cleanSource() {
    
    /* Memory related to the beamlets is freed within Matlab */
    free(source.cdfinv1);
    free(source.cdfinv2);
    
    return;
}

/******************************************************************************/
/* Scoring definitions. The scoring arrays, ausgab() and accumEndep() live in
 the core library, in omc_score.c, so that both user codes share the touched
 voxel bookkeeping. */

void accumulateResults(int iout, int nhist, int nbatch)
{
    /* Only voxels this beamlet actually deposited in can be nonzero, and for
     an untouched voxel the arithmetic below reduces to writing back the zeros
     that are already there: accum_endep is 0, so endep and endep2 come out 0,
     the endep != 0 branch is not taken, and both outputs are set to 0. Zeroing
     the dose in air is likewise a no-op on a voxel that never received any.
     So walking the touched set is equivalent to walking the grid, at a
     fraction of the cost for a single beamlet. */
    const int *touched;
    int ntouched = scoreBeamVoxels(&touched);

    /* MSVC only implements OpenMP 2.0, which in C does not allow declaring
     the loop variable inside the for statement */
    int n;
    #pragma omp parallel for
    for (n = 0; n < ntouched; n++) {
        int irl = touched[n];

        /* Region 0 is outside the geometry. ausgab() does reach it, through
         the discard path in electron(), but it has no voxel and is not part
         of the output. */
        if (irl == 0) {
            continue;
        }

        int ix, iy, iz;
        omcDecodeRegion(irl, geometry.isize, geometry.jsize, &ix, &iy, &iz);

        double endep = score.accum_endep[irl];
        double endep2 = score.accum_endep2[irl];
        double factor;

        if (iout) {
            /* Convert deposited energy to dose */
            double mass = (geometry.xbounds[ix+1] - geometry.xbounds[ix])*
                (geometry.ybounds[iy+1] - geometry.ybounds[iy])*
                (geometry.zbounds[iz+1] - geometry.zbounds[iz]);

            /* Transform deposited energy to Gy */
            mass *= geometry.med_densities[irl-1];

            factor = 1.602E-10/(mass);
        }
        else {  /* Output mean deposited energy */
            factor = 1.0;
        }

        endep *= factor;
        endep2 *= factor*factor;

        /* First calculate mean deposited energy across batches and its
         uncertainty */
        endep /= (double) nbatch;
        endep2 /= (double) nbatch;

        double unc_endep;

        /* Batch approach uncertainty calculation: sample variance of the
         batch means over (nbatch - 1) gives the variance of the mean. The
         divisors here must not be swapped -- dividing endep2 by (nbatch - 1)
         instead leaves a spurious mean^2/(nbatch*(nbatch - 1)) term that puts
         a floor of ~10% relative uncertainty under every voxel regardless of
         the statistics. */
        if (endep != 0.0) {
            unc_endep = endep2 - endep*endep;

            //Variance of the mean
            unc_endep /= (double) (nbatch - 1);
        }
        else {
            endep = 0.0;
            unc_endep = 0.0;
        }

        /* Zero dose in air */
        if (geometry.med_densities[irl-1] < 0.044) {
            endep = 0.0;
            unc_endep = 0.0;
        }

        /* Store output quantities */
        score.accum_endep[irl] = endep;
        score.accum_endep2[irl] = unc_endep;
    }

    return;
}

void outputResults(char *output_file, int iout, int nhist, int nbatch) {
    
    /* Accumulate the results */
    accumulateResults(iout, nhist,nbatch);
    
    int irl;
    int imax = geometry.isize;
    int ijmax = geometry.isize*geometry.jsize;
    
    /* Output to file */
    char extension[15];
    if (iout) {
        strcpy(extension, ".3ddose");
    } else {
        strcpy(extension, ".3denergy");
    }
    
    /* Get file path from input data */
    char output_folder[128];
    char buffer[BUFFER_SIZE];
    
    if (getInputValue(buffer, "output folder") != 1) {
        omcFail("matRad:omc_matrad:missingInput",
            "Can not find 'output folder' key on input file.");
    }
    removeSpaces(output_folder, buffer);
    
    /* Make space for the new string */
    char* file_name = malloc(strlen(output_folder) + strlen(output_file) + 
        strlen(extension) + 1);
    strcpy(file_name, output_folder);
    strcat(file_name, output_file); /* add the file name */
    strcat(file_name, extension); /* add the extension */
    
    FILE *fp;
    if ((fp = fopen(file_name, "w")) == NULL) {
        omcFail("matRad:omc_matrad:outputFile",
            "Unable to open file: %s", file_name);
    }
    
    /* Grid dimensions */
    fprintf(fp, "%5d%5d%5d\n",
            geometry.isize, geometry.jsize, geometry.ksize);
    
    /* Boundaries in x-, y- and z-directions */
    for (int ix = 0; ix<=geometry.isize; ix++) {
        fprintf(fp, "%f ", geometry.xbounds[ix]);
    }
    fprintf(fp, "\n");
    for (int iy = 0; iy<=geometry.jsize; iy++) {
        fprintf(fp, "%f ", geometry.ybounds[iy]);
    }
    fprintf(fp, "\n");
    for (int iz = 0; iz<=geometry.ksize; iz++) {
        fprintf(fp, "%f ", geometry.zbounds[iz]);
    }
    fprintf(fp, "\n");
    
    /* Dose or energy array */
    for (int iz=0; iz<geometry.ksize; iz++) {
        for (int iy=0; iy<geometry.jsize; iy++) {
            for (int ix=0; ix<geometry.isize; ix++) {
                irl = 1 + ix + iy*imax + iz*ijmax;
                fprintf(fp, "%e ", score.accum_endep[irl]);
            }
        }
    }
    fprintf(fp, "\n");
    
    /* Uncertainty array */
    for (int iz=0; iz<geometry.ksize; iz++) {
        for (int iy=0; iy<geometry.jsize; iy++) {
            for (int ix=0; ix<geometry.isize; ix++) {
                irl = 1 + ix + iy*imax + iz*ijmax;
                fprintf(fp, "%f ", score.accum_endep2[irl]);
            }
        }
    }
    fprintf(fp, "\n");
    
    /* Cleaning */
    fclose(fp);
    free(file_name);

    return;
}


void initHistory(int ibeamlet) {

    double rnno1;
    double rnno2;
    
    int ijmax = geometry.isize*geometry.jsize;
    int imax = geometry.isize;
    
    /* Initialize first particle of the stack from source data */
    stack.np = 0;
    stack.p[stack.np].iq = source.charge;
    
    /* Get primary particle energy */
    double ein = 0.0;
    if (source.spectrum) {
        /* Sample initial energy from spectrum data */
        rnno1 = setRandom();
        rnno2 = setRandom();
        
        /* Sample bin number in order to select particle energy */
        int k = (int)fmin(source.deltak*rnno1, source.deltak - 1.0);
        ein = source.cdfinv1[k] + rnno2*source.cdfinv2[k];
    }
    else {
        /* Monoenergetic source */
        ein = source.energy;
    }
    
    /* Check if the particle is an electron, in such a case add electron
     rest mass energy */
    if (stack.p[stack.np].iq != 0) {
        /* Electron or positron */
        stack.p[stack.np].e = ein + RM;
    }
    else {
        /* Photon */
        stack.p[stack.np].e = ein;
    }
    
    /* Accumulate sampled kinetic energy for fraction of deposited energy
     calculations */
    scoreSource(ein);
    
    /* Set particle position. First obtain a random position in the rectangle
     defined by the bixel at isocenter*/    
    double xiso = 0.0; 
    double yiso = 0.0;
    double ziso = 0.0;
    
    rnno1 = setRandom();
    rnno2 = setRandom();

    xiso = rnno1*source.xside1[ibeamlet] + rnno2*source.xside2[ibeamlet] + 
            source.xcorner[ibeamlet];
    yiso = rnno1*source.yside1[ibeamlet] + rnno2*source.yside2[ibeamlet] + 
            source.ycorner[ibeamlet];
    ziso = rnno1*source.zside1[ibeamlet] + rnno2*source.zside2[ibeamlet] + 
            source.zcorner[ibeamlet];
    
    
    /* Norm of the resulting vector from the source of current beam to the 
     position of the particle on bixel */
    int ibeam = source.ibeam[ibeamlet];

    double sourcePos[3];

    //Gaussian Source

    switch (omcConfig.sourceGeometry)
    {
        case POINT: ;
            sourcePos[0] = source.xsource[ibeam];
            sourcePos[1] = source.ysource[ibeam];
            sourcePos[2] = source.zsource[ibeam];
            break;
        case GAUSSIAN: ;        
            //double stdSource[3] = {omcConfig.sourceGaussianWidth, omcConfig.sourceGaussianWidth, omcConfig.sourceGaussianWidth};
            //sourcePos[0] = setStandardNormalRandom(source.xsource[ibeam],stdSource[0]);
            //sourcePos[1] = setStandardNormalRandom(source.ysource[ibeam],stdSource[1]);
            //sourcePos[2] = setStandardNormalRandom(source.zsource[ibeam],stdSource[2]);
                        
            //Get the normalized collimator plane vectors
            double planeVec1_norm;
            double planeVec2_norm;
            planeVec1_norm = sqrt(   
                                            source.xside1[ibeamlet]*source.xside1[ibeamlet] + 
                                            source.yside1[ibeamlet]*source.yside1[ibeamlet] + 
                                            source.zside1[ibeamlet]*source.zside1[ibeamlet]
                                        );
            planeVec2_norm = sqrt(   
                                            source.xside2[ibeamlet]*source.xside2[ibeamlet] + 
                                            source.yside2[ibeamlet]*source.yside2[ibeamlet] + 
                                            source.zside2[ibeamlet]*source.zside2[ibeamlet]
                                        );
            double planeVec1[3];
            planeVec1[0] = source.xside1[ibeamlet] / planeVec1_norm;
            planeVec1[1] = source.yside1[ibeamlet] / planeVec1_norm;
            planeVec1[2] = source.zside1[ibeamlet] / planeVec1_norm;    

            double planeVec2[3];
            planeVec2[0] = source.xside2[ibeamlet] / planeVec2_norm;
            planeVec2[1] = source.yside2[ibeamlet] / planeVec2_norm;
            planeVec2[2] = source.zside2[ibeamlet] / planeVec2_norm;            

            //Create two normally distributed random veriables with box-muller transform
            double rnSource[2]; 
            boxMuller(rnSource);

            //Scale with source width
            rnSource[0] *= omcConfig.sourceGaussianWidth;
            rnSource[1] *= omcConfig.sourceGaussianWidth;

            //Now use the plane vectors to add the random 2D offset to the source
            sourcePos[0] = source.xsource[ibeam] + rnSource[0]*planeVec1[0] + rnSource[1]*planeVec2[0];
            sourcePos[1] = source.ysource[ibeam] + rnSource[0]*planeVec1[1] + rnSource[1]*planeVec2[1];
            sourcePos[2] = source.zsource[ibeam] + rnSource[0]*planeVec1[2] + rnSource[1]*planeVec2[2];

            
            break;
        default: ;
            mexErrMsgIdAndTxt("matRad:matRad_ompInterface:invalidSourceGeometry","Source type not defined!");
    }
        

    //Point source
    double xd = xiso - sourcePos[0];
    double yd = yiso - sourcePos[1];
    double zd = ziso - sourcePos[2];


    double vnorm = sqrt(xd*xd + yd*yd + zd*zd);            
        
    /* Direction of the particle from position on bixel to beam source*/
    double u = -(xd)/vnorm;
    double v = -(yd)/vnorm;
    double w = -(zd)/vnorm;
    
    /* Calculate the minimum distance from particle position on bixel to 
     phantom boundaries */
    double ustep = DBL_MAX; //1.0E5; 
    double dist;
    
    if(u > 0.0) {
        dist = (geometry.xbounds[geometry.isize]-xiso)/u;
        if(dist < ustep) {
            ustep = dist;
        }        
    }
    if(u < 0.0) {
        dist = -(xiso-geometry.xbounds[0])/u;
        if(dist < ustep) {
            ustep = dist;
        }        
    }
    
    if(v > 0.0) {
        dist = (geometry.ybounds[geometry.jsize]-yiso)/v;
        if(dist < ustep) {
            ustep = dist;
        }        
    }
    if(v < 0.0) {
        dist = -(yiso-geometry.ybounds[0])/v;
        if(dist < ustep) {
            ustep = dist;
        }        
    }
    
    if(w > 0.0) {
        dist = (geometry.zbounds[geometry.ksize]-ziso)/w;
        if(dist < ustep) {
            ustep = dist;
        }        
    }
    if(w < 0.0) {
        dist = -(ziso-geometry.zbounds[0])/w;
        if(dist < ustep) {
            ustep = dist;
        }        
    }
    
    /* Transport particle from bixel to surface. Adjust particle direction 
     to be incident to phantom surface */
    stack.p[stack.np].x = xiso + ustep*u;
    stack.p[stack.np].y = yiso + ustep*v;
    stack.p[stack.np].z = ziso + ustep*w;
    
    stack.p[stack.np].u = -u;
    stack.p[stack.np].v = -v;
    stack.p[stack.np].w = -w;

    /* For numerical stability, make sure that points are really inside the
     phantom. nextafter() moves one representable step towards the opposite
     face; the 2.0*DBL_MIN offset used before is denormal-small and was
     absorbed entirely when added to any normal boundary coordinate, leaving
     the particle exactly on the boundary. */
    if(stack.p[stack.np].x < geometry.xbounds[0]) {
        stack.p[stack.np].x = nextafter(geometry.xbounds[0],
                                        geometry.xbounds[geometry.isize]);
    }
    if(stack.p[stack.np].x > geometry.xbounds[geometry.isize]) {
        stack.p[stack.np].x = nextafter(geometry.xbounds[geometry.isize],
                                        geometry.xbounds[0]);
    }

    if(stack.p[stack.np].y < geometry.ybounds[0]) {
        stack.p[stack.np].y = nextafter(geometry.ybounds[0],
                                        geometry.ybounds[geometry.jsize]);
    }
    if(stack.p[stack.np].y > geometry.ybounds[geometry.jsize]) {
        stack.p[stack.np].y = nextafter(geometry.ybounds[geometry.jsize],
                                        geometry.ybounds[0]);
    }

    if(stack.p[stack.np].z < geometry.zbounds[0]) {
        stack.p[stack.np].z = nextafter(geometry.zbounds[0],
                                        geometry.zbounds[geometry.ksize]);
    }
    if(stack.p[stack.np].z > geometry.zbounds[geometry.ksize]) {
        stack.p[stack.np].z = nextafter(geometry.zbounds[geometry.ksize],
                                        geometry.zbounds[0]);
    }
    
    /* Determine region index of source particle */
    int ix = omcFindVoxelIndex(geometry.xbounds, geometry.isize,
                               stack.p[stack.np].x);
    int iy = omcFindVoxelIndex(geometry.ybounds, geometry.jsize,
                               stack.p[stack.np].y);
    int iz = omcFindVoxelIndex(geometry.zbounds, geometry.ksize,
                               stack.p[stack.np].z);

    stack.p[stack.np].ir = 1 + ix + iy*imax + iz*ijmax;
          
    /* Set statistical weight and distance to closest boundary*/
    stack.p[stack.np].wt = 1.0;
    stack.p[stack.np].dnear = 0.0;
    
    return;
}

/* Progress reporting for the main simulation loop. If the caller supplied
 options.progressCallback, report through it (progress in [0,1]) and let
 the MATLAB side own any waitbar/handle lifecycle. Otherwise fall back to
 the built-in waitbar, lazily opened on first use, when verbose_flag > 1. */
static mxArray *builtinWaitbarHandle = NULL;

static void reportProgress(double progress, const char *message) {
    if (progressCallback != NULL) {
        mxArray *progressArg = mxCreateDoubleScalar(progress);
        mxArray *cbArgs[2] = { progressCallback, progressArg };
        mexCallMATLAB(0, NULL, 2, cbArgs, "feval");
        mxDestroyArray(progressArg);
        return;
    }

    if (verbose_flag <= 1)
        return;

    mxArray *progressArg = mxCreateDoubleScalar(progress);
    mxArray *messageArg = mxCreateString(message);
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
    
    /* Initialize radiation source */
    initSource();
    
    /* Initialize data on a region-by-region basis */
    initRegions();
    
    /* Initialize VRT data */
    initVrt();
    
    /* Preparation of scoring struct */
    initScore(geometry.isize*geometry.jsize*geometry.ksize);

    #pragma omp parallel
    {
      /* Initialize random number generator */
      initRandom();

      /* Initialize particle stack */
      initStack();
    }

    /* Shower call */
    
    /* Get number of histories, statistical batches and splitting factor */
    char buffer[BUFFER_SIZE];
    
    int nhist = omcConfig.nHist;
    int nbatch = omcConfig.nBatch; 
    
    if (nhist/nbatch == 0) {
        nhist = nbatch;
    }
    
    int nperbatch = nhist/nbatch;
    nhist = nperbatch*nbatch;
    
    int gridsize = geometry.isize*geometry.jsize*geometry.ksize;
    
    if (verbose_flag > 1) 
    {
        mexPrintf("Total number of particle histories: %d\n", nhist);
        mexPrintf("Number of statistical batches: %d\n", nbatch);
        mexPrintf("Histories per batch: %d\n", nperbatch);
    }

    double relDoseThreshold = omcConfig.doseThreshold;

    if (verbose_flag > 2)
        mexPrintf("Using a relative dose cut-off of %f\n",relDoseThreshold);
    
    const char *progressMessage = "calculate dose influence matrix for photons (ompMC) ...";


    /* Create output matrix */
    mwSize nCubeElements = geometry.isize*geometry.jsize*geometry.ksize;
    double percentage_steps = 0.01;             // steps in which the sparse matrix is allocated
    double percent_sparse = percentage_steps;   // initial percentage to allocate memory for

    mwSize nzmax = (mwSize) ceil((double)nCubeElements*(double)source.nbeamlets*percent_sparse);
    plhs[0] = mxCreateSparse(nCubeElements,(mwSize) source.nbeamlets,nzmax,mxREAL);

    double *sr  = mxGetPr(plhs[0]);
    mwIndex *irs = mxGetIr(plhs[0]);
    mwIndex *jcs = mxGetJc(plhs[0]);
    mwIndex linIx = 0;
    jcs[0] = 0;

    int outputVariance = (nlhs >= 2);

    double *sr_var = NULL;
    mwIndex *irs_var = NULL;
    mwIndex *jcs_var = NULL;
    if (outputVariance)
    {
        plhs[1] = mxCreateSparse(nCubeElements,(mwSize) source.nbeamlets,nzmax,mxREAL);
        sr_var  = mxGetPr(plhs[1]);
        irs_var = mxGetIr(plhs[1]);
        jcs_var = mxGetJc(plhs[1]);
        jcs_var[0] = 0;
    }    
    
    double progress = 0.0;

    if (verbose_flag > 0)
        mexPrintf("done!\n");        

    /* Execution time up to this point */
    if (verbose_flag > 2)
        mexPrintf("Execution time up to this point : %8.2f seconds\n",(omc_get_time() - tbegin));
    
    if (verbose_flag > 0)
        mexPrintf("Running ompMC simulation...\n");

    int sparse_reallocations;
    sparse_reallocations = 0;
    
    for(int ibeamlet=0; ibeamlet<source.nbeamlets; ibeamlet++) {
        for (int ibatch=0; ibatch<nbatch; ibatch++) {            
            int ihist;

            #pragma omp parallel for schedule(guided)
            for (ihist=0; ihist<nperbatch; ihist++) {
                /* Point the RNG at this history's stream; the index is
                 unique across batches and beamlets, so results do not
                 depend on the scheduling */
                setRandomHistory(((uint64_t)ibeamlet*(uint64_t)nbatch
                                  + (uint64_t)ibatch)*(uint64_t)nperbatch
                                 + (uint64_t)ihist);

                /* Initialize particle history */
                initHistory(ibeamlet);

                /* Start electromagnetic shower simulation */
                shower();
            }

            /* Accumulate results of current batch for statistical analysis */
            accumEndep(1.0/(double)nperbatch);

            progress = ((double)ibeamlet + (double)(ibatch+1)/nbatch)/source.nbeamlets;
            reportProgress(progress, progressMessage);
        }

        /* Output of results for current beamlet */
        int iout = 1;   /* i.e. deposit mean dose per particle fluence */
        accumulateResults(iout, nhist, nbatch);

        /* Everything from here to the end of the beamlet only ever looks at
         voxels this beamlet deposited in; the rest of the grid is zero and
         below any positive threshold. The list comes back ascending, which is
         what the sparse column below needs. */
        const int *touched;
        int ntouched = scoreBeamVoxels(&touched);

        /* Get maximum value to apply threshold */
        double doseMax = 0.0;
        for (int n = 0; n < ntouched; n++) {
            int irl = touched[n];
            if (irl != 0 && score.accum_endep[irl] > doseMax) {
                doseMax = score.accum_endep[irl];
            }
        }
        double thresh = doseMax*relDoseThreshold;

        /* Count values above threshold */
        mwSize j_nnz = 0; //Number of nonzeros in the dose cube for the current beamlet
        for (int n = 0; n < ntouched; n++) {
            int irl = touched[n];
            if (irl != 0 && score.accum_endep[irl] > thresh) {
                j_nnz++;
            }
        }

        //The number of new non-zero values is the current linear index + new upcoming entries from current beamlet + 1
        mwSize newnnz = j_nnz + (mwSize) linIx;

        /* Check if we need to reallocate for sparse matrix */
        if (newnnz > nzmax) {
            mwSize oldnzmax = nzmax;
            percent_sparse += percentage_steps;
            nzmax = (mwSize) ceil((double)nCubeElements*(double)source.nbeamlets*percent_sparse);
            
            /* Make sure nzmax increases at least by 1. */
            if (oldnzmax == nzmax) {
                nzmax++;
            }                

            /* Check that the new nmax is large enough and if not, also adjust 
            the percentage_steps since we seem to have set it too small for this 
            particular use case */
            if (nzmax < newnnz) {
                nzmax = newnnz;
                percent_sparse = (double)nzmax/nCubeElements;
                percentage_steps = percent_sparse;
            }

            if (verbose_flag > 2) {
                mexPrintf("Reallocating Sparse Matrix from nzmax=%d to nzmax=%d\n", oldnzmax, nzmax);
            }                
            
            /* Set new nzmax and reallocate more memory */
            mxSetNzmax(plhs[0], nzmax);
            mxSetPr(plhs[0], (double *) mxRealloc(sr, nzmax*sizeof(double)));
            mxSetIr(plhs[0], (mwIndex *) mxRealloc(irs, nzmax*sizeof(mwIndex)));
            
            /* Use the new pointers */
            sr  = mxGetPr(plhs[0]);
            irs = mxGetIr(plhs[0]);

            if (outputVariance) {
                /* Set new nzmax and reallocate more memory */
                mxSetNzmax(plhs[1], nzmax);
                mxSetPr(plhs[1], (double *) mxRealloc(sr_var, nzmax*sizeof(double)));
                mxSetIr(plhs[1], (mwIndex *)  mxRealloc(irs_var, nzmax*sizeof(mwIndex)));
            
                /* Use the new pointers */
                sr_var  = mxGetPr(plhs[1]);
                irs_var = mxGetIr(plhs[1]);
            }

            sparse_reallocations++;
        }


        //Populate sparse matrix arrays
        for (int n = 0; n < ntouched; n++) {
            int irl = touched[n];
            if (irl != 0 && score.accum_endep[irl] > thresh) {
                sr[linIx] = score.accum_endep[irl];
                irs[linIx] = irl-1;

                if (outputVariance) {
                    sr_var[linIx] = score.accum_endep2[irl];
                    irs_var[linIx] = irl-1;
                }
                linIx++;
            }
        }
        
        if (verbose_flag > 1 && linIx != newnnz)
            mexPrintf("Warning: Discrepancy between linear index %d and maximum number of computed nonzeros %d at beamlet %d finalization!\n",linIx,newnnz,ibeamlet);

        if (verbose_flag > 1 && linIx > nzmax)
            mexPrintf("Warning: Discrepancy between linear index %d and maximum number of allowed nonzeros %d at beamlet %d finalization!\n",linIx,newnnz,ibeamlet);

        jcs[ibeamlet+1] = linIx;
        if (outputVariance) {
            jcs_var[ibeamlet+1] = linIx;
        }
        
        /* Reset the accumulators for the following beamlet. This clears
         accum_endep2 as well, which the memset it replaces did not, so the
         variance of one beamlet no longer leaks into the next. */
        resetBeamScore();
		progress = (double) (ibeamlet+1) / (double) source.nbeamlets;
        reportProgress(progress, progressMessage);
    }

    /* Print some output and execution time up to this point */
    if (verbose_flag > 0)
        mexPrintf("Simulation finished!\nFinalizing output...\n");

    closeProgress();

    if (verbose_flag >= 3)
        mexPrintf("Sparse MC Dij has %d (%f percent) elements!\n", linIx, (double)linIx/((double)nCubeElements*(double)source.nbeamlets));

    if (verbose_flag >= 3)
        mexPrintf("Needed %d sparse matrix reallocations.\n",sparse_reallocations);

    
    /* Truncate the matrix to the exact size by reallocation */
    mxSetNzmax(plhs[0], linIx);
    mxSetPr(plhs[0], mxRealloc(sr, linIx*sizeof(double)));
    mxSetIr(plhs[0], mxRealloc(irs, linIx*sizeof(mwIndex)));
    
    sr  = mxGetPr(plhs[0]);
    irs = mxGetIr(plhs[0]);

    //Check output
    if (verbose_flag >= 3)
        mexPrintf("Verifying sparse Matrix... ");
    for (int ix = 0; ix < linIx; ix++)
    {
        mwIndex currIx = irs[ix];
        
        if (currIx > gridsize)
            mexPrintf("Invalid dose-cube index %d at linear index %d in sparse matrix check!",currIx,linIx);
    }
    if (verbose_flag >= 3)
        mexPrintf("done!\n");

    if (outputVariance) {
        /* Truncate the matrix to the exact size by reallocation */
        mxSetNzmax(plhs[1], linIx);
        mxSetPr(plhs[1], mxRealloc(sr_var, linIx*sizeof(double)));
        mxSetIr(plhs[1], mxRealloc(irs_var, linIx*sizeof(mwIndex)));
        sr_var  = mxGetPr(plhs[1]);
        irs_var = mxGetIr(plhs[1]);           
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
    cleanScore();
    cleanSource();

    //Cleaning private random generators and particle stack
    #pragma omp parallel
    {
      cleanRandom();
      cleanStack();
    }
    /* Get total execution time */
    if (verbose_flag > 0)        
        mexPrintf("Finished! Total execution time : %8.5f seconds\n", (omc_get_time() - tbegin));
    
}

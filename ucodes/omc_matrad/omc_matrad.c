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

#define exit(EXIT_FAILURE) mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:invalid","Error in ompMC mex file. Abort!");

#include "omc_utilities.h"
#include "omc_random.h"
#include "ompmc.h"

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

//Data Types and Structs
struct Geom {
    int *med_indices;           // index of the media in each voxel
    double *med_densities;      // density of the medium in each voxel
    
    int isize;                  // number of voxels on each direction
    int jsize;
    int ksize;
    
    double *xbounds;            // boundaries of voxels on each direction
    double *ybounds;
    double *zbounds;
};
struct Geom geometry;

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
    //TODO: passable spectrum

    enum sourceGeometryType sourceGeometry;
    double sourceGaussianWidth; //Assuming 5mm FWHM penumbra if the source is gaussian
};

struct OmcConfig omcConfig;

/* Function used to parse input from matRad */
void parseInput(int nrhs, const mxArray *prhs[]) {
    //Default values
    omcConfig.nHist = 1e4;
    omcConfig.nBatch = 10;
    omcConfig.doseThreshold = 0.01;
    omcConfig.monoEnergy = 0.1;
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
    tmp_fieldpointer = mxGetField(mcOpt,0,"nSplit");    
    status = mexCallMATLAB(1, &tmp2, 1,  &tmp_fieldpointer, "num2str");    
    if (status != 0)
        mexErrMsgIdAndTxt( "matRad:omc_matrad:Error","Call to num2str not successful");
    else
    {
        tmp = mxArrayToString(tmp2);        
        strcpy(input_items[nInput].value,tmp);
    }

    tmp_fieldpointer = mxGetField(mcOpt,0,"spectrumFile");
    if (tmp_fieldpointer) {
        size_t buflen = mxGetNumberOfElements(tmp_fieldpointer) + 1;
        omcConfig.spectrumFile = (char*) mxCalloc(buflen + 1,sizeof(char));
        if (mxGetString(tmp_fieldpointer, omcConfig.spectrumFile, buflen) != 0) 
            mexErrMsgIdAndTxt("MATLAB:explore:invalidStringArray","Invalid string for path to spectrum file!");
        
    }
    else
    {
        size_t buflen = 255;
        omcConfig.spectrumFile = (char*) mxCalloc(buflen + 1,sizeof(char));
        omcConfig.spectrumFile = "./spectra/mohan6.spectrum";
    }
   
    tmp_fieldpointer = mxGetField(mcOpt,0,"monoEnergy");
    if (tmp_fieldpointer)
        omcConfig.monoEnergy = mxGetScalar(tmp_fieldpointer);    
    
    nInput++;
    sprintf(input_items[nInput].key,"charge");
    tmp_fieldpointer = mxGetField(mcOpt,0,"charge");    
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
    tmp_fieldpointer = mxGetField(mcOpt,0,"global_ecut");    
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
    tmp_fieldpointer = mxGetField(mcOpt,0,"global_pcut");    
    status = mexCallMATLAB(1, &tmp2, 1,  &tmp_fieldpointer, "num2str");    
    if (status != 0)
        mexErrMsgIdAndTxt( "matRad:omc_matrad:Error","Call to num2str not successful");
    else
    {
        tmp = mxArrayToString(tmp2);        
        strcpy(input_items[nInput].value,tmp);
    }
    
    nInput++;
    sprintf(input_items[nInput].key,"rng seeds");
    tmp_fieldpointer = mxGetField(mcOpt,0,"randomSeeds");    
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
    tmp_fieldpointer = mxGetField(mcOpt,0,"pegsFile");    
    tmp = mxArrayToString(tmp_fieldpointer);
    strcpy(input_items[nInput].value,tmp);
    
    nInput++;
    sprintf(input_items[nInput].key,"pgs4form file");
    tmp_fieldpointer = mxGetField(mcOpt,0,"pgs4formFile");    
    tmp = mxArrayToString(tmp_fieldpointer);
    strcpy(input_items[nInput].value,tmp);
    
    nInput++;
    sprintf(input_items[nInput].key,"data folder");
    tmp_fieldpointer = mxGetField(mcOpt,0,"dataFolder");    
    tmp = mxArrayToString(tmp_fieldpointer);
    strcpy(input_items[nInput].value,tmp);
    
    nInput++;
    sprintf(input_items[nInput].key,"output folder");
    tmp_fieldpointer = mxGetField(mcOpt,0,"outputFolder");    
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

void howfar(int *idisc, int *irnew, double *ustep) {
    
    int np = stack.np;
    int irl = stack.ir[np];
    double dist = 0.0;
    
    if (stack.ir[np] == 0) {
        /* The particle is outside the geometry, terminate history */
        *idisc = 1;
        return;
    }
    
    /* If here, the particle is in the geometry, do transport checks */
    int ijmax = geometry.isize*geometry.jsize;
    int imax = geometry.isize;
    
    /* First we need to decode the region number of the particle in terms of
     the region indices in each direction */
    int irx = (irl - 1)%imax;
    int irz = (irl - 1 - irx)/ijmax;
    int iry = ((irl - 1 - irx) - irz*ijmax)/imax;
    
    /* Check in z-direction */
    if (stack.w[np] > 0.0) {
        /* Going towards outer plane */
        dist = (geometry.zbounds[irz+1] - stack.z[np])/stack.w[np];
        if (dist < *ustep) {
            *ustep = dist;
            if (irz != (geometry.ksize - 1)) {
                *irnew = irl + ijmax;
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }
    
    else if (stack.w[np] < 0.0) {
        /* Going towards inner plane */
        dist = -(stack.z[np] - geometry.zbounds[irz])/stack.w[np];
        if (dist < *ustep) {
            *ustep = dist;
            if (irz != 0) {
                *irnew = irl - ijmax;
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }

    /* Check in x-direction */
    if (stack.u[np] > 0.0) {
        /* Going towards positive plane */
        dist = (geometry.xbounds[irx+1] - stack.x[np])/stack.u[np];
        if (dist < *ustep) {
            *ustep = dist;
            if (irx != (geometry.isize - 1)) {
                *irnew = irl + 1;
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }
    
    else if (stack.u[np] < 0.0) {
        /* Going towards negative plane */
        dist = -(stack.x[np] - geometry.xbounds[irx])/stack.u[np];
        if (dist < *ustep) {
            *ustep = dist;
            if (irx != 0) {
                *irnew = irl - 1;
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }
    
    /* Check in y-direction */
    if (stack.v[np] > 0.0) {
        /* Going towards positive plane */
        dist = (geometry.ybounds[iry+1] - stack.y[np])/stack.v[np];
        if (dist < *ustep) {
            *ustep = dist;
            if (iry != (geometry.jsize - 1)) {
                *irnew = irl + imax;
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }
    
    else if (stack.v[np] < 0.0) {
        /* Going towards negative plane */
        dist = -(stack.y[np] - geometry.ybounds[iry])/stack.v[np];
        if (dist < *ustep) {
            *ustep = dist;
            if (iry != 0) {
                *irnew = irl - imax;
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }
    
    return;
}

double hownear(void) {
    
    int np = stack.np;
    int irl = stack.ir[np];
    double tperp = 1.0E10;  /* perpendicular distance to closest boundary */
    
    if (irl == 0) {
        /* Particle exiting geometry */
        tperp = 0.0;
    }
    else {
        /* In the geometry, do transport checks */
        int ijmax = geometry.isize*geometry.jsize;
        int imax = geometry.isize;
        
        /* First we need to decode the region number of the particle in terms
         of the region indices in each direction */
        int irx = (irl - 1)%imax;
        int irz = (irl - 1 - irx)/ijmax;
        int iry = ((irl - 1 - irx) - irz*ijmax)/imax;
        
        /* Check in x-direction */
        tperp = fmin(tperp, geometry.xbounds[irx+1] - stack.x[np]);
        tperp = fmin(tperp, stack.x[np] - geometry.xbounds[irx]);
        
        /* Check in y-direction */
        tperp = fmin(tperp, geometry.ybounds[iry+1] - stack.y[np]);
        tperp = fmin(tperp, stack.y[np] - geometry.ybounds[iry]);
        
        /* Check in z-direction */
        tperp = fmin(tperp, geometry.zbounds[irz+1] - stack.z[np]);
        tperp = fmin(tperp, stack.z[np] - geometry.zbounds[irz]);
    }
    
    return tperp;
}
/******************************************************************************/

/******************************************************************************/
/* Source definitions */
const int MXEBIN = 200;     // number of energy bins of spectrum
const int INVDIM = 1000;    // number of bins in inverse CDF

void initSource() {
    
    /* Get spectrum file path from input data */
    char buffer[BUFFER_SIZE];

    char* fstatus;
    
    source.spectrum = 1;    /* energy spectrum as default case */    
    
    if (source.spectrum) {
        //removeSpaces(omcConfig.spectrumFile, buffer);
        
        /* Open .source file */
        FILE *fp;
        
        if ((fp = fopen(omcConfig.spectrumFile, "r")) == NULL) {
            mexPrintf("Unable to open file: %s\n", omcConfig.spectrumFile);
            exit(EXIT_FAILURE);
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
            mexPrintf("Number of energy bins = %d is greater than max allowed = "
                   "%d. Increase MXEBIN macro!\n", nensrc, MXEBIN);
            exit(EXIT_FAILURE);
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
            mexPrintf("Invalid mode number in spectrum file.");
            exit(EXIT_FAILURE);
        }
        
        double ein = ensrcd[nensrc - 1];
        if (verbose_flag > 1)
            mexPrintf("Energy ranges from %f to %f MeV\n", enmin, ein);
        
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
        
        /* Cleaning */
        fclose(fp);
        free(ensrcd);
        free(srcpdf);
        free(srccdf);
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
/* Scoring definitions */
struct Score {
    double ensrc;               // total energy from source
    double *endep;              // 3D dep. energy matrix per batch
    
    /* The following variables are needed for statistical analysis. Their
     values are accumulated across the simulation */
    double *accum_endep;        // 3D deposited energy matrix
    double *accum_endep2;       // 3D square deposited energy
};
struct Score score;

void initScore() {
    
    int gridsize = geometry.isize*geometry.jsize*geometry.ksize;
    
    score.ensrc = 0.0;
    
    /* Region with index 0 corresponds to region outside phantom */
    score.endep = malloc((gridsize + 1)*sizeof(double));
    score.accum_endep = malloc((gridsize + 1)*sizeof(double));
    score.accum_endep2 = malloc((gridsize + 1)*sizeof(double));
    
    /* Initialize all arrays to zero */
    memset(score.endep, 0.0, (gridsize + 1)*sizeof(double));
    memset(score.accum_endep, 0.0, (gridsize + 1)*sizeof(double));
    memset(score.accum_endep2, 0.0, (gridsize + 1)*sizeof(double));
    
    return;
}

void cleanScore() {
    
    free(score.endep);
    free(score.accum_endep);
    free(score.accum_endep2);
    
    return;
}

void ausgab(double edep) {
    
    int np = stack.np;
    int irl = stack.ir[np];
    double endep = stack.wt[np]*edep;
        
    /* Deposit particle energy on spot */
    #pragma omp atomic
    score.endep[irl] += endep;
    
    return;
}

void accumEndep(int nperbatch) {
    
    int gridsize = geometry.isize*geometry.jsize*geometry.ksize;
    
    /* Accumulate endep and endep squared for statistical analysis */
    double edep = 0.0;
    
    int irl = 0;
    
    #pragma omp parallel for firstprivate(edep)
    for (irl=0; irl<gridsize + 1; irl++) {
        edep = score.endep[irl];
        edep /= (double) nperbatch;
        score.accum_endep[irl] += edep;
        score.accum_endep2[irl] += edep*edep;
    }
    
    /* Clean scoring array */
    memset(score.endep, 0.0, (gridsize + 1)*sizeof(double));
    
    return;
}

void accumulateResults(int iout, int nhist, int nbatch)
{
    int irl;
    int imax = geometry.isize;
    int ijmax = geometry.isize*geometry.jsize;
    double endep, endep2, unc_endep;

    /* Calculate incident fluence */
    //double inc_fluence = ;    
    double mass;
    int iz;


    #pragma omp parallel for private(irl,endep,endep2,unc_endep,mass)
    for (iz=0; iz<geometry.ksize; iz++) {
        for (int iy=0; iy<geometry.jsize; iy++) {
            for (int ix=0; ix<geometry.isize; ix++) {
                irl = 1 + ix + iy*imax + iz*ijmax;
                endep = score.accum_endep[irl];
                endep2 = score.accum_endep2[irl];
                

                double factor;
                if (iout) {
                    
                    /* Convert deposited energy to dose */
                    mass = (geometry.xbounds[ix+1] - geometry.xbounds[ix])*
                        (geometry.ybounds[iy+1] - geometry.ybounds[iy])*
                        (geometry.zbounds[iz+1] - geometry.zbounds[iz]);
                    
                    /* Transform deposited energy to Gy */
                    mass *= geometry.med_densities[irl-1];
                    
                    factor = 1.602E-10/(mass);                                      
                    
                } else {    /* Output mean deposited energy */
                    factor = 1.0;
                }

                endep *= factor;
                endep2 *= factor*factor;


                /* First calculate mean deposited energy across batches and its
                 uncertainty */
                endep /= (double) nbatch;
                endep2 /= (double) (nbatch - 1);
                
                /* Batch approach uncertainty calculation */
                if (endep != 0.0) {
                    unc_endep = endep2 - endep * endep;
                    //unc_endep /= (double)(nbatch - 1);
                    
                    //Variance of the mean
                    unc_endep /= nbatch;
                    
                    /* Relative uncertainty */
                    //unc_endep = sqrt(unc_endep)/endep;
                }
                else {
                    endep = 0.0;
                    unc_endep = 0.0;
                }


                /* We separate de calculation of dose, to give the user the
                 option to output mean energy (iout=0) or deposited dose
                 (iout=1) per incident fluence */
                
                /* Store output quantities */
                score.accum_endep[irl] = endep;
                score.accum_endep2[irl] = unc_endep;
            }
        }
    }
    
    /* Zero dose in air */
    #pragma omp parallel for private(irl)
    for (iz=0; iz<geometry.ksize; iz++) {
        for (int iy=0; iy<geometry.jsize; iy++) {
            for (int ix=0; ix<geometry.isize; ix++) {
                irl = 1 + ix + iy*imax + iz*ijmax;
                
                if(geometry.med_densities[irl-1] < 0.044) {
                    score.accum_endep[irl] = 0.0;
                    score.accum_endep2[irl] = 0.0;
                }
            }
        }
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
        mexPrintf("Can not find 'output folder' key on input file.\n");
        exit(EXIT_FAILURE);
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
        mexPrintf("Unable to open file: %s\n", file_name);
        exit(EXIT_FAILURE);
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

/******************************************************************************/
/* Region-by-region definitions */
void initRegions() {
    
    /* +1 : consider region surrounding phantom */
    int nreg = geometry.isize*geometry.jsize*geometry.ksize + 1;
    
    /* Allocate memory for region data */
    region.med = malloc(nreg*sizeof(int));
    region.rhof = malloc(nreg*sizeof(double));
    region.pcut = malloc(nreg*sizeof(double));
    region.ecut = malloc(nreg*sizeof(double));
    
    /* First get global energy cutoff parameters */
    char buffer[BUFFER_SIZE];
    if (getInputValue(buffer, "global ecut") != 1) {
        mexPrintf("Can not find 'global ecut' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    double ecut = atof(buffer);
    
    if (getInputValue(buffer, "global pcut") != 1) {
        mexPrintf("Can not find 'global pcut' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    double pcut = atof(buffer);
    
    /* Initialize transport parameters on each region. Region 0 is outside the
     geometry */
    region.med[0] = VACUUM;
    region.rhof[0] = 0.0;
    region.pcut[0] = 0.0;
    region.ecut[0] = 0.0;
    
    for (int i=1; i<nreg; i++) {
        
        /* -1 : EGS counts media from 1. Substract 1 to get medium index */
        int imed = geometry.med_indices[i - 1] - 1;
        region.med[i] = imed;
        
        if (imed == VACUUM) {
            region.rhof[0] = 0.0F;
            region.pcut[0] = 0.0F;
            region.ecut[0] = 0.0F;
        }
        else {
            if (geometry.med_densities[i - 1] == 0.0F) {
                region.rhof[i] = 1.0;
            }
            else {
                region.rhof[i] =
                    geometry.med_densities[i - 1]/pegs_data.rho[imed];
            }
            
            /* Check if global cut-off values are within PEGS data */
            if (pegs_data.ap[imed] <= pcut) {
                region.pcut[i] = pcut;
            } else {
                mexPrintf("Warning!, global pcut value is below PEGS's pcut value "
                       "%f for medium %d, using PEGS value.\n",
                       pegs_data.ap[imed], imed);
                region.pcut[i] = pegs_data.ap[imed];
            }
            if (pegs_data.ae[imed] <= ecut) {
                region.ecut[i] = ecut;
            } else {
                mexPrintf("Warning!, global pcut value is below PEGS's ecut value "
                       "%f for medium %d, using PEGS value.\n",
                       pegs_data.ae[imed], imed);
            }
        }
    }

    return;
}

void initHistory(int ibeamlet) {
    
    double rnno1;
    double rnno2;
    
    int ijmax = geometry.isize*geometry.jsize;
    int imax = geometry.isize;
    
    /* Initialize first particle of the stack from source data */
    stack.np = 0;
    stack.iq[stack.np] = source.charge;
    
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
    if (stack.iq[stack.np] != 0) {
        /* Electron or positron */
        stack.e[stack.np] = ein + RM;
    }
    else {
        /* Photon */
        stack.e[stack.np] = ein;
    }
    
    /* Accumulate sampled kinetic energy for fraction of deposited energy
     calculations */
    score.ensrc += ein;
    
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
    stack.x[stack.np] = xiso + ustep*u;
    stack.y[stack.np] = yiso + ustep*v;
    stack.z[stack.np] = ziso + ustep*w;
    
    stack.u[stack.np] = -u;
    stack.v[stack.np] = -v;
    stack.w[stack.np] = -w;

    /* For numerical stability, make sure that points are really inside the phantom */
    if(stack.x[stack.np] < geometry.xbounds[0]) {
        stack.x[stack.np] = geometry.xbounds[0] + 2.0*DBL_MIN;
    }
    if(stack.x[stack.np] > geometry.xbounds[geometry.isize]) {
        stack.x[stack.np] = geometry.xbounds[geometry.isize] - 2.0*DBL_MIN;
    }

    if(stack.y[stack.np] < geometry.ybounds[0]) {
        stack.y[stack.np] = geometry.ybounds[0] + 2.0*DBL_MIN;
    }
    if(stack.y[stack.np] > geometry.ybounds[geometry.jsize]) {
        stack.y[stack.np] = geometry.ybounds[geometry.jsize] - 2.0*DBL_MIN;
    }

    if(stack.z[stack.np] < geometry.zbounds[0]) {
        stack.z[stack.np] = geometry.ybounds[0] + 2.0*DBL_MIN;
    }
    if(stack.z[stack.np] > geometry.zbounds[geometry.ksize]) {
      stack.z[stack.np] = geometry.zbounds[geometry.ksize] - 2.0*DBL_MIN;
    }
    
    /* Determine region index of source particle */
    int ix = 0;
    while (geometry.xbounds[ix+1] < stack.x[stack.np]) {
        ix++;
    }
    
    int iy = 0;
    while (geometry.ybounds[iy+1] < stack.y[stack.np]) {
        iy++;
    }
    
    int iz = 0;
    while (geometry.zbounds[iz+1] < stack.z[stack.np]) {
        iz++;
    }
    
    stack.ir[stack.np] = 1 + ix + iy*imax + iz*ijmax;
          
    /* Set statistical weight and distance to closest boundary*/
    stack.wt[stack.np] = 1.0;
    stack.dnear[stack.np] = 0.0;
    
    return;
}

/******************************************************************************/
/* omc_matrad main function */
void mexFunction (int nlhs, mxArray *plhs[],    // output of the function
    int nrhs, const mxArray *prhs[])            // input of the function
{
    
    /* Execution time measurement */
    double tbegin;
    tbegin = omc_get_time();
    
    
    /* Parsing program options */

    /* Check for proper number of input and output arguments */
    if (nrhs != 5) {
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:invalidNumInputs","Two or three input arguments required.");
    }
    if(nlhs > 2){
        mexErrMsgIdAndTxt( "matRad:matRad_ompInterface:invalidNumOutputs","Too many output arguments.");
    }

    mexPrintf("Running ompMC...\n");


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
    initScore();

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
    
    /* Use Matlab waitbar to show execution progress */
    mxArray* waitbarHandle = NULL;                             // the waitbar handle does not exist yet
	mxArray* waitbarProgress = mxCreateDoubleScalar(0.0);   // allocate a double scalar for the progress
	mxArray* waitbarMessage = mxCreateString("calculate dose influence matrix for photons (ompMC) ...");    // allocate a string for the message
	
	mxArray* waitbarInputs[3];  // array of waitbar inputs
    mxArray* waitbarOutput[1];  // pointer to waitbar output

	waitbarInputs[0] = waitbarProgress; 
	waitbarInputs[1] = waitbarMessage;	
	
    /* Create the waitbar with h = waitbar(progress,message); */
    int matlabCallStatus = 0;
    if (verbose_flag > 1) {
        matlabCallStatus = mexCallMATLAB(1, waitbarOutput, 2, waitbarInputs, "waitbar");
        waitbarHandle = waitbarOutput[0];
    }


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
                /* Initialize particle history */
                initHistory(ibeamlet);
                
                /* Start electromagnetic shower simulation */
                shower();
            }
            
            /* Accumulate results of current batch for statistical analysis */
            accumEndep(nperbatch);

            progress = ((double)ibeamlet + (double)(ibatch+1)/nbatch)/source.nbeamlets;
            (*mxGetPr(waitbarProgress)) = progress;

            if (waitbarHandle != NULL && waitbarOutput != NULL) {              
                waitbarInputs[0] = waitbarProgress;
                waitbarInputs[1] = waitbarHandle;
                waitbarInputs[2] = waitbarMessage;
                matlabCallStatus = mexCallMATLAB(0, waitbarOutput, 2, waitbarInputs, "waitbar");
            }
        }

        /* Output of results for current beamlet */
        int iout = 1;   /* i.e. deposit mean dose per particle fluence */
        accumulateResults(iout, nhist, nbatch);

        /* Get maximum value to apply threshold */
        double doseMax = 0.0;
        for (int irl=1; irl < gridsize+1; irl++) {
            if (score.accum_endep[irl] > doseMax) {
                doseMax = score.accum_endep[irl];
            }
        }
        double thresh = doseMax*relDoseThreshold;
        /* Count values above threshold */
        mwSize j_nnz = 0; //Number of nonzeros in the dose cube for the current beamlet
        int irl = 1;
        #pragma omp parallel for reduction(+:j_nnz)
        for (irl=1; irl < gridsize+1; irl++) {        
            if (score.accum_endep[irl] > thresh) {
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
        for (int irl=1; irl < gridsize+1; irl++) {        
            if (score.accum_endep[irl] > thresh) {            
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
        
        /* Reset accum_endep for following beamlet */
        memset(score.accum_endep, 0.0, (gridsize + 1)*sizeof(double));                
		progress = (double) (ibeamlet+1) / (double) source.nbeamlets;		
        (*mxGetPr(waitbarProgress)) = progress;

		/* Update the waitbar with waitbar(hWaitbar,progress); */
        if (waitbarHandle != NULL && waitbarOutput != NULL) {
            waitbarInputs[0] = waitbarProgress;
            waitbarInputs[1] = waitbarHandle;		
            waitbarInputs[2] = waitbarMessage;
            matlabCallStatus = mexCallMATLAB(0, waitbarOutput, 2, waitbarInputs, "waitbar");
        }
    }

    /* Print some output and execution time up to this point */
    if (verbose_flag > 0)
        mexPrintf("Simulation finished!\nFinalizing output...\n"); 
    
    mxDestroyArray (waitbarProgress);
    mxDestroyArray (waitbarMessage);
    if (waitbarHandle != NULL && waitbarOutput != NULL) {
        waitbarInputs[0] = waitbarHandle;		
        matlabCallStatus = mexCallMATLAB(0,waitbarOutput,1, waitbarInputs,"close") ;
        mxDestroyArray(waitbarHandle);
    }

    
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

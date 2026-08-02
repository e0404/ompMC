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
 omc_dosxyz - An ompMC user code to calculate deposited dose on voxelized 
 geometries.  
*****************************************************************************/

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "omc_engine_cube.h"
#include "omc_geom.h"
#include "omc_host.h"
#include "omc_spectrum.h"
#include "omc_utilities.h"
#include "ompmc.h"
#include "omc_version.h"

/******************************************************************************/
/* Parsing program options with getopt long
 http://www.gnu.org/software/libc/manual/html_node/Getopt.html#Getopt */
#include <getopt.h>

/******************************************************************************/
/* Verbosity of the simulation, set through the --verbose/--brief options and
 read back by the ompMC core library */
int verbose_flag = 0;

/* The particle stack, the regions and the PEGS data are the engine's business
 now; all this file still touches of the core's state is the media table it
 fills from the phantom file. */
extern struct Media media;

/******************************************************************************/
/* Geometry definitions */

void initPhantom() {
    
    /* Get phantom file path from input data */
    char phantom_file[128];
    char buffer[BUFFER_SIZE];
    
    if (getInputValue(buffer, "phantom file") != 1) {
        printf("Can not find 'phantom file' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    removeSpaces(phantom_file, buffer);
    
    /* Open .egsphant file */
    FILE *fp;
    
    if ((fp = fopen(phantom_file, "r")) == NULL) {
        printf("Unable to open file: %s\n", phantom_file);
        exit(EXIT_FAILURE);
    }
    
    printf("Path to phantom file : %s\n", phantom_file);
    
    /* Get number of media in the phantom */
    fgets(buffer, BUFFER_SIZE, fp);
    media.nmed = atoi(buffer);
    
    /* Get media names on phantom file */
    for (int i=0; i<media.nmed; i++) {
        fgets(buffer, BUFFER_SIZE, fp);
        removeSpaces(media.med_names[i], buffer);
    }
    
    /* Skip next line, it contains dummy input */
    fgets(buffer, BUFFER_SIZE, fp);
    
    /* Read voxel numbers on each direction */
    fgets(buffer, BUFFER_SIZE, fp);
    sscanf(buffer, "%d %d %d", &geometry.isize,
           &geometry.jsize, &geometry.ksize);
    
    /* Read voxel boundaries on each direction */
    geometry.xbounds = malloc((geometry.isize + 1)*sizeof(double));
    geometry.ybounds = malloc((geometry.jsize + 1)*sizeof(double));
    geometry.zbounds = malloc((geometry.ksize + 1)*sizeof(double));
    
    for (int i=0; i<=geometry.isize; i++) {
        fscanf(fp, "%lf", &geometry.xbounds[i]);
    }
    for (int i=0; i<=geometry.jsize; i++) {
        fscanf(fp, "%lf", &geometry.ybounds[i]);
     }
    for (int i=0; i<=geometry.ksize; i++) {
        fscanf(fp, "%lf", &geometry.zbounds[i]);
    }
    
    /* Skip the rest of the last line read before */
    fgets(buffer, BUFFER_SIZE, fp);
    
    /* Read media indices */
    int irl = 0;    // region index
    char idx;
    geometry.med_indices =
        malloc(geometry.isize*geometry.jsize*geometry.ksize*sizeof(int));
    for (int k=0; k<geometry.ksize; k++) {
        for (int j=0; j<geometry.jsize; j++) {
            for (int i=0; i<geometry.isize; i++) {
                irl = i + j*geometry.isize + k*geometry.jsize*geometry.isize;
                idx = fgetc(fp);
                /* Convert digit stored as char to int */
                geometry.med_indices[irl] = idx - '0';
            }
            /* Jump to next line */
            fgets(buffer, BUFFER_SIZE, fp);
        }
        /* Skip blank line */
        fgets(buffer, BUFFER_SIZE, fp);
    }
    
    /* Read media densities */
    geometry.med_densities =
        malloc(geometry.isize*geometry.jsize*geometry.ksize*sizeof(double));
    for (int k=0; k<geometry.ksize; k++) {
        for (int j=0; j<geometry.jsize; j++) {
            for (int i=0; i<geometry.isize; i++) {
                irl = i + j*geometry.isize + k*geometry.jsize*geometry.isize;
                fscanf(fp, "%lf", &geometry.med_densities[irl]);
            }
        }
        /* Skip blank line */
        fgets(buffer, BUFFER_SIZE, fp);
    }
    
    /* Summary with geometry information */
    printf("Number of media in phantom : %d\n", media.nmed);
    printf("Media names: ");
    for (int i=0; i<media.nmed; i++) {
        printf("%s, ", media.med_names[i]);
    }
    printf("\n");
    printf("Number of voxels on each direction (X,Y,Z) : (%d, %d, %d)\n",
           geometry.isize, geometry.jsize, geometry.ksize);
    printf("Minimum and maximum boundaries on each direction : \n");
    printf("\tX (cm) : %lf, %lf\n",
           geometry.xbounds[0], geometry.xbounds[geometry.isize]);
    printf("\tY (cm) : %lf, %lf\n",
           geometry.ybounds[0], geometry.ybounds[geometry.jsize]);
    printf("\tZ (cm) : %lf, %lf\n",
           geometry.zbounds[0], geometry.zbounds[geometry.ksize]);
    
    omcGeomDetectSpacing();

    /* Close phantom file */
    fclose(fp);

    return;
}

void cleanPhantom() {
    
    free(geometry.xbounds);
    free(geometry.ybounds);
    free(geometry.zbounds);
    free(geometry.med_indices);
    free(geometry.med_densities);
    return;
}

/******************************************************************************/


/******************************************************************************/
/* Source definitions. The transport side of the source lives in the core, in
 omc_engine_cube.c; what is left here is reading the input file. */

static struct OmcSpectrum spectrum;
static struct OmcSsdSource ssdSource;
static struct OmcCubeOptions cubeOptions;

static void initSource(void) {

    char buffer[BUFFER_SIZE];

    /* Get spectrum file path from input data. Without one the source is
     monoenergetic and the energy has to be given instead. */
    if (getInputValue(buffer, "spectrum file") == 1) {
        char spectrum_file[128];
        removeSpaces(spectrum_file, buffer);

        omcSpectrumFromFile(&spectrum, spectrum_file);
    }
    else {
        printf("Can not find 'spectrum file' key on input file.\n");
        printf("Switch to monoenergetic case.\n");

        if (getInputValue(buffer, "mono energy") != 1) {
            printf("Can not find 'mono energy' key on input file.\n");
            exit(EXIT_FAILURE);
        }

        omcSpectrumMonoenergetic(&spectrum, atof(buffer));
    }

    /* Initialize geometrical data of the source */

    /* Read collimator rectangle */
    if (getInputValue(buffer, "collimator bounds") != 1) {
        printf("Can not find 'collimator bounds' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    sscanf(buffer, "%lf %lf %lf %lf", &ssdSource.xinl,
           &ssdSource.xinu, &ssdSource.yinl, &ssdSource.yinu);

    /* Read source charge */
    if (getInputValue(buffer, "charge") != 1) {
        printf("Can not find 'charge' key on input file.\n");
        exit(EXIT_FAILURE);
    }

    cubeOptions.charge = atoi(buffer);
    if (cubeOptions.charge < -1 || cubeOptions.charge > 1) {
        printf("Particle kind not recognized.\n");
        exit(EXIT_FAILURE);
    }

    /* Read source SSD */
    if (getInputValue(buffer, "ssd") != 1) {
        printf("Can not find 'ssd' key on input file.\n");
        exit(EXIT_FAILURE);
    }

    ssdSource.ssd = atof(buffer);
    if (ssdSource.ssd < 0) {
        printf("SSD must be greater than zero.\n");
        exit(EXIT_FAILURE);
    }

    /* Clamp the collimator to the phantom and find the voxels it covers */
    omcSsdSourceInit(&ssdSource);

    /* Print some information for debugging purposes */
    if (verbose_flag) {
        printf("Source information :\n");
        printf("\t Charge = %d\n", cubeOptions.charge);
        printf("\t SSD (cm) = %f\n", ssdSource.ssd);
        printf("Collimator :\n");
        printf("\t x (cm) : min = %f, max = %f\n", ssdSource.xinl, ssdSource.xinu);
        printf("\t y (cm) : min = %f, max = %f\n", ssdSource.yinl, ssdSource.yinu);
        printf("Sizes :\n");
        printf("\t x (cm) = %f, y (cm) = %f\n", ssdSource.xsize, ssdSource.ysize);
    }

    return;
}

static void cleanSource(void) {

    omcSpectrumFree(&spectrum);

    return;
}

/******************************************************************************/
/* Writing the results out in the EGSnrc .3ddose format */

void outputResults(char *output_file, int iout,
                   const double *dose, const double *uncertainty) {

    int ivox;
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
        printf("Can not find 'output folder' key on input file.\n");
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
        printf("Unable to open file: %s\n", file_name);
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
                ivox = ix + iy*imax + iz*ijmax;
                fprintf(fp, "%e ", dose[ivox]);
            }
        }
    }
    fprintf(fp, "\n");

    /* Uncertainty array */
    for (int iz=0; iz<geometry.ksize; iz++) {
        for (int iy=0; iy<geometry.jsize; iy++) {
            for (int ix=0; ix<geometry.isize; ix++) {
                ivox = ix + iy*imax + iz*ijmax;
                fprintf(fp, "%f ", uncertainty[ivox]);
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
/* Progress: one line per batch, with the elapsed time only this file knows */

static double tbegin;

static int reportBatch(int ibatch, int nbatch, uint64_t firstHistory,
                       void *user) {

    (void)nbatch;
    (void)user;

    if (ibatch == 0) {
        /* Print header for information during simulation */
        printf("%-10s\t%-15s\t%-15s\n", "Batch #", "Elapsed time",
               "First history");
    }
    printf("%-10d\t%-15.2f\t%-15llu\n", ibatch,
           (omc_get_time() - tbegin),
           (unsigned long long)firstHistory);

    /* Nothing to stop the run for: it is the whole point of the program */
    return 1;
}

/******************************************************************************/
/* omc_dosxyz main function */
int main (int argc, char **argv) {

    /* Execution time measurement */
    tbegin = omc_get_time();

    printf("ompMC version %s\n", OMPMC_VERSION_STRING);

    /* Parsing program options */

    int c;
    char *input_file = NULL;
    char *output_file = NULL;

    while (1) {
        static struct option long_options[] =
        {
            /* These options set a flag. */
            {"verbose", no_argument, &verbose_flag, 1},
            {"brief",   no_argument, &verbose_flag, 0},
            /* These options don't set a flag.
             We distinguish them by their indices. */
            {"input",  required_argument, 0, 'i'},
            {"output",    required_argument, 0, 'o'},
            {0, 0, 0, 0}
        };

        /* getopt_long stores the option index here. */
        int option_index = 0;

        c = getopt_long(argc, argv, "i:o:",
                         long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
            break;

        switch (c) {
            case 0:
                /* If this option set a flag, do nothing else now. */
                if (long_options[option_index].flag != 0)
                    break;
                printf ("option %s", long_options[option_index].name);
                if (optarg)
                    printf (" with arg %s", optarg);
                printf ("\n");
                break;

            case 'i':
                input_file = malloc(strlen(optarg) + 1);
                strcpy(input_file, optarg);
                printf ("option -i with value `%s'\n", input_file);
                break;

            case 'o':
                output_file = malloc(strlen(optarg) + 1);
                strcpy(output_file, optarg);
                printf ("option -o with value `%s'\n", output_file);
                break;

            case '?':
                /* getopt_long already printed an error message. */
                break;

            default:
                exit(EXIT_FAILURE);
        }
    }

    /* Instead of reporting '--verbose'
     and '--brief' as they are encountered,
     we report the final status resulting from them. */
    if (verbose_flag)
        puts ("verbose flag is set");

    /* Print any remaining command line arguments (not options). */
    if (optind < argc)
    {
        printf ("non-option ARGV-elements: ");
        while (optind < argc)
            printf ("%s ", argv[optind++]);
        putchar ('\n');
    }

    /* Parse input file and print key,value pairs (test) */
    parseInputFile(input_file);

    /* Get information of OpenMP environment */
#ifdef _OPENMP
    int omp_size = omp_get_num_procs();
    printf("Number of OpenMP threads: %d\n", omp_size);
    omp_set_num_threads(omp_size);
#else
    printf("ompMC compiled without OpenMP support. Serial execution.\n");
#endif

    /* Read geometry information from phantom file and initialize geometry */
    initPhantom();

    /* With number of media and media names initialize the medium data */
    initMediaData();

    /* Initialize radiation source */
    initSource();

    /* Initialize data on a region-by-region basis */
    initRegions();

    /* Initialize VRT data */
    initVrt();

    /* In verbose mode, list interaction data to output folder */
    if (verbose_flag) {
        listRayleigh();
        listPair();
        listPhoton();
        listElectron();
        listMscat();
        listSpin();
    }

    /* Get number of histories and statistical batches */
    char buffer[BUFFER_SIZE];
    if (getInputValue(buffer, "ncase") != 1) {
        printf("Can not find 'ncase' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    cubeOptions.nhist = atoi(buffer);

    if (getInputValue(buffer, "nbatch") != 1) {
        printf("Can not find 'nbatch' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    cubeOptions.nbatch = atoi(buffer);

    cubeOptions.outputDose = 1;  /* i.e. deposit mean dose per particle fluence */

    int gridsize = geometry.isize*geometry.jsize*geometry.ksize;
    double *dose = malloc(gridsize*sizeof(double));
    double *uncertainty = malloc(gridsize*sizeof(double));

    /* Execution time up to this point */
    printf("Execution time up to this point : %8.2f seconds\n",
           (omc_get_time() - tbegin));

    struct OmcCubeCallbacks callbacks;
    callbacks.batch = reportBatch;
    callbacks.user = NULL;

    struct OmcCubeSummary summary;

    omcCalcCube(&cubeOptions, &ssdSource, &spectrum, dose, uncertainty,
                &callbacks, &summary);

    /* Print some output and execution time up to this point */
    printf("Simulation finished\n");
    printf("Execution time up to this point : %8.2f seconds\n",
           (omc_get_time() - tbegin));

    /* Analysis and output of results */
    if (verbose_flag) {
        printf("Fraction of incident energy deposited in the phantom: %5.4f\n",
               summary.energyFraction);
    }

    outputResults(output_file, cubeOptions.outputDose, dose, uncertainty);

    /* Cleaning */
    free(dose);
    free(uncertainty);
    cleanPhantom();
    cleanPhoton();
    cleanRayleigh();
    cleanPair();
    cleanElectron();
    cleanMscat();
    cleanSpin();
    cleanRegions();
    cleanSource();

    free(input_file);
    free(output_file);
    /* Get total execution time */
    printf("Total execution time : %8.5f seconds\n",
           (omc_get_time() - tbegin));

    exit (EXIT_SUCCESS);
}

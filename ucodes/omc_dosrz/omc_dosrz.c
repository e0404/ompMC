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
 omc_dosrz - An ompMC user code to calculate dose in a cylinder, scored by
 radial ring and depth.

 The r-z counterpart of omc_dosxyz, and named after DOSRZnrc for the same
 reason it exists: dose around a narrow beam is the thing a rectilinear grid
 is worst at, and rings are what it is naturally binned in.

 The phantom is one homogeneous cylinder about the beam axis, described in
 the input file rather than read from an .egsphant -- there is no file format
 for a cylinder, and a handful of numbers is the whole geometry.

   ./build/bin/omc_dosrz -i ucodes/omc_dosrz/smoke_test -o smoke_rz

 writes output/smoke_rz.rzdose. See input_file.inp for the keys.
*****************************************************************************/

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "omc_engine_radial.h"
#include "omc_geom.h"
#include "omc_geom_cyl.h"
#include "omc_host.h"
#include "omc_phsp.h"
#include "omc_source.h"
#include "omc_source_pencil.h"
#include "omc_source_phsp.h"
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

/* All this file touches of the core's state is the media table, which it
 fills with the one medium the cylinder is made of. */
extern struct Media media;

/******************************************************************************/
/* Reading the input file.

 The keys follow omc_dosxyz's naming, and the ones it shares with it -- ncase,
 nbatch, the cut-offs, the data paths -- mean exactly what they do there. */

static void requireValue(char *dest, const char *key) {

    if (getInputValue(dest, (char *)key) != 1) {
        printf("Can not find '%s' key on input file.\n", key);
        exit(EXIT_FAILURE);
    }
}

/* An ascending list of boundaries written out in full, as an alternative to
 the uniform "radius + count" form. The value of an input item is at most
 BUFFER_SIZE-1 characters, which is about twenty five boundaries; a geometry
 wanting more than that wants the uniform form or a host that is not a text
 file.

 @return The number of bins, i.e. one less than the boundaries read. */
static int parseBounds(const char *value, double **bounds) {

    /* At most this many, since each takes at least two characters */
    double *parsed = malloc((BUFFER_SIZE/2 + 2)*sizeof(double));
    int n = 0;

    const char *at = value;
    while (*at != '\0') {
        char *end;
        double x = strtod(at, &end);

        if (end == at) {
            break;
        }

        parsed[n++] = x;
        at = end;
    }

    if (n < 2) {
        printf("A boundary list needs at least two values, got %d.\n", n);
        exit(EXIT_FAILURE);
    }

    *bounds = parsed;

    return n - 1;
}

/* Uniform bins from 0 to extent. */
static int uniformBounds(double extent, int nbins, double **bounds) {

    if (!(extent > 0.0)) {
        printf("The cylinder has extent %f; it has to be positive.\n", extent);
        exit(EXIT_FAILURE);
    }
    if (nbins < 1) {
        printf("The cylinder has %d bins; it needs at least one.\n", nbins);
        exit(EXIT_FAILURE);
    }

    double *made = malloc((nbins + 1)*sizeof(double));
    for (int i = 0; i <= nbins; i++) {
        made[i] = extent*(double)i/(double)nbins;
    }

    *bounds = made;

    return nbins;
}

static void initCylinder(void) {

    char buffer[BUFFER_SIZE];

    /* The rings. An explicit list wins over the uniform form, so that a run
     wanting fine bins on the axis and coarse ones outside can say so. */
    if (getInputValue(buffer, "radial bin edges") == 1) {
        geometry.isize = parseBounds(buffer, &geometry.rbounds);
    }
    else {
        requireValue(buffer, "cylinder radius");
        double radius = atof(buffer);

        requireValue(buffer, "radial bins");
        geometry.isize = uniformBounds(radius, atoi(buffer),
                                       &geometry.rbounds);
    }

    /* The depth slabs */
    if (getInputValue(buffer, "depth bin edges") == 1) {
        geometry.ksize = parseBounds(buffer, &geometry.zbounds);
    }
    else {
        requireValue(buffer, "cylinder depth");
        double depth = atof(buffer);

        requireValue(buffer, "depth bins");
        geometry.ksize = uniformBounds(depth, atoi(buffer),
                                       &geometry.zbounds);
    }

    /* One medium throughout. Region 0 is outside, so the arrays run over the
     rings times the slabs, exactly as a voxel phantom's do. */
    requireValue(buffer, "medium");
    media.nmed = 1;
    removeSpaces(media.med_names[0], buffer);

    double density = 0.0;       /* 0 : whatever PEGS says the medium weighs */
    if (getInputValue(buffer, "medium density") == 1) {
        density = atof(buffer);
        if (!(density > 0.0)) {
            printf("The medium density is %f; it has to be positive.\n",
                   density);
            exit(EXIT_FAILURE);
        }
    }

    int nregions = geometry.isize*geometry.ksize;

    geometry.med_indices = malloc(nregions*sizeof(int));
    geometry.med_densities = malloc(nregions*sizeof(double));

    for (int i = 0; i < nregions; i++) {
        geometry.med_indices[i] = 1;    /* EGS counts media from 1 */
        geometry.med_densities[i] = density;
    }

    /* Checks the bounds over, pins the unused azimuthal index and declares
     the phantom cylindrical */
    omcGeomCylInit();

    printf("Cylinder of %s : radius %f cm in %d rings, depth %f cm in %d "
           "slabs\n", media.med_names[0],
           geometry.rbounds[geometry.isize], geometry.isize,
           geometry.zbounds[geometry.ksize] - geometry.zbounds[0],
           geometry.ksize);

    return;
}

static void cleanCylinder(void) {

    free(geometry.rbounds);
    free(geometry.zbounds);
    free(geometry.med_indices);
    free(geometry.med_densities);

    return;
}

/******************************************************************************/
/* The source. Three of them: the two beams down the axis in
 omc_source_pencil.h, and a phase space file. */

static struct OmcSpectrum spectrum;
static int spectrumBuilt = 0;

static struct OmcPencilSource pencil;
static struct OmcPhsp phsp;
static struct OmcPhspSampler phspSampler;
static int phspRead = 0;

static struct OmcSource source;
static struct OmcRadialOptions radialOptions;

static void initSpectrum(void) {

    char buffer[BUFFER_SIZE];

    /* Same fallback omc_dosxyz uses: a file if there is one, otherwise a
     single energy. */
    if (getInputValue(buffer, "spectrum file") == 1) {
        char spectrum_file[BUFFER_SIZE];
        removeSpaces(spectrum_file, buffer);

        omcSpectrumFromFile(&spectrum, spectrum_file);
    }
    else {
        printf("Can not find 'spectrum file' key on input file.\n");
        printf("Switch to monoenergetic case.\n");

        requireValue(buffer, "mono energy");
        omcSpectrumMonoenergetic(&spectrum, atof(buffer));
    }

    spectrumBuilt = 1;

    return;
}

/* Read n doubles out of one input value, for the phase space transform. */
static int readDoubles(const char *value, double *into, int n) {

    const char *at = value;

    for (int i = 0; i < n; i++) {
        char *end;
        into[i] = strtod(at, &end);
        if (end == at) {
            return 0;
        }
        at = end;
    }

    return 1;
}

static void initPhaseSpace(void) {

    char buffer[BUFFER_SIZE];
    char phsp_file[BUFFER_SIZE];

    requireValue(buffer, "phsp file");
    removeSpaces(phsp_file, buffer);

    omcPhspFromFile(&phsp, phsp_file);
    phspRead = 1;

    memset(&phspSampler, 0, sizeof(phspSampler));
    phspSampler.phsp = &phsp;
    phspSampler.order = OMC_PHSP_REPLAY;
    phspSampler.first = 0;
    omcPhspTransformIdentity(&phspSampler.transform);

    if (getInputValue(buffer, "phsp order") == 1) {
        char order[BUFFER_SIZE];
        removeSpaces(order, buffer);

        if (strcmp(order, "random") == 0) {
            phspSampler.order = OMC_PHSP_RANDOM;
        }
        else if (strcmp(order, "replay") != 0) {
            printf("Unknown 'phsp order' value '%s'; it is 'replay' or "
                   "'random'.\n", order);
            exit(EXIT_FAILURE);
        }
    }

    if (getInputValue(buffer, "phsp first") == 1) {
        phspSampler.first = strtoull(buffer, NULL, 10);
    }

    if (getInputValue(buffer, "phsp rotation") == 1) {
        if (!readDoubles(buffer, phspSampler.transform.rotation, 9)) {
            printf("'phsp rotation' needs nine numbers, row by row.\n");
            exit(EXIT_FAILURE);
        }
    }

    if (getInputValue(buffer, "phsp translation") == 1) {
        if (!readDoubles(buffer, phspSampler.transform.translation, 3)) {
            printf("'phsp translation' needs three numbers.\n");
            exit(EXIT_FAILURE);
        }
    }

    omcPhspSamplerAsSource(&phspSampler, &source);

    printf("Phase space : %s, %llu particles\n", phsp_file,
           omcPhspCount(&phsp));

    return;
}

static void initBeam(void) {

    char buffer[BUFFER_SIZE];

    memset(&pencil, 0, sizeof(pencil));

    initSpectrum();
    pencil.spectrum = &spectrum;

    requireValue(buffer, "charge");
    pencil.charge = atoi(buffer);

    if (getInputValue(buffer, "ssd") == 1) {
        pencil.kind = OMC_PENCIL_SSD;
        pencil.ssd = atof(buffer);

        if (getInputValue(buffer, "field radius") == 1) {
            pencil.fieldRadius = atof(buffer);
        }
    }
    else {
        pencil.kind = OMC_PENCIL_PARALLEL;
    }

    /* Either delta the beam does not really have can be widened. Left out,
     both stay 0, which is the delta itself and draws no random numbers. */
    if (getInputValue(buffer, "spot sigma") == 1) {
        pencil.spotSigma = atof(buffer);
    }
    if (getInputValue(buffer, "divergence sigma") == 1) {
        pencil.divergenceSigma = atof(buffer);
    }
    if (getInputValue(buffer, "correlation") == 1) {
        pencil.correlation = atof(buffer);
    }

    /* The same beam said the other way round, which is how beam data is
     usually quoted: a waist of some size, some depth in. It sets both the
     width on the face and the correlation, so having it and either of those
     is a deck that contradicts itself rather than one to reconcile. */
    if (getInputValue(buffer, "waist sigma") == 1) {
        double waistSigma = atof(buffer);
        double waistDepth = 0.0;

        if (pencil.spotSigma > 0.0 || pencil.correlation != 0.0) {
            printf("'waist sigma' already says what 'spot sigma' and "
                   "'correlation' say. Give the beam one way or the other.\n");
            exit(EXIT_FAILURE);
        }

        /* The waist is measured against the width of the beam on the front
         face, and a point source's spot is not that width: every particle
         arrives at the point on the field it was aimed at whatever the spot
         did to where it set off, so the spot cancels over the SSD and what
         sets the width there is 'field radius'. */
        if (pencil.kind == OMC_PENCIL_SSD) {
            printf("'waist sigma' describes a parallel pencil. A point "
                   "source's spot is its focal spot and does not set where "
                   "its beam is, so there is no waist to place; give 'spot "
                   "sigma' and 'correlation' directly.\n");
            exit(EXIT_FAILURE);
        }

        if (!(pencil.divergenceSigma > 0.0)) {
            printf("'waist sigma' needs a 'divergence sigma' as well: a beam "
                   "that does not diverge has the same width everywhere.\n");
            exit(EXIT_FAILURE);
        }

        if (getInputValue(buffer, "waist depth") == 1) {
            waistDepth = atof(buffer);
        }

        omcPencilWaist(waistSigma, pencil.divergenceSigma, waistDepth,
                       &pencil.spotSigma, &pencil.correlation);
    }

    omcPencilSourceAsSource(&pencil, &source);

    if (pencil.kind == OMC_PENCIL_PARALLEL) {
        printf("Source : parallel pencil beam on the axis, charge %d\n",
               pencil.charge);
    }
    else {
        printf("Source : point at SSD %f cm, field radius %f cm, charge %d\n",
               pencil.ssd,
               pencil.fieldRadius > 0.0 ? pencil.fieldRadius
                                        : geometry.rbounds[geometry.isize],
               pencil.charge);
    }

    if (pencil.spotSigma > 0.0 || pencil.divergenceSigma > 0.0) {
        printf("\t spot sigma (cm) = %f, divergence sigma (rad) = %f\n",
               pencil.spotSigma, pencil.divergenceSigma);
    }

    if (pencil.correlation != 0.0 && pencil.spotSigma > 0.0 &&
        pencil.divergenceSigma > 0.0) {
        /* Where that correlation puts the waist, since it is what the deck
         was after either way and is easier to recognize as wrong. */
        printf("\t correlation = %f, waist %f cm wide %f cm past the front "
               "face\n", pencil.correlation,
               pencil.spotSigma*sqrt(1.0 - pencil.correlation
                                          *pencil.correlation),
               -pencil.correlation*pencil.spotSigma/pencil.divergenceSigma);
    }

    return;
}

static void initSource(void) {

    char buffer[BUFFER_SIZE];
    char kind[BUFFER_SIZE];

    requireValue(buffer, "source type");
    removeSpaces(kind, buffer);

    if (strcmp(kind, "phsp") == 0) {
        initPhaseSpace();
    }
    else if (strcmp(kind, "pencil") == 0 || strcmp(kind, "point") == 0) {
        initBeam();

        /* 'pencil' and 'point' are the same source with and without an SSD,
         so say so rather than letting a stray key decide. */
        if (strcmp(kind, "pencil") == 0 &&
            pencil.kind != OMC_PENCIL_PARALLEL) {
            printf("'source type = pencil' is a parallel beam and takes no "
                   "'ssd'; use 'source type = point' for a point source.\n");
            exit(EXIT_FAILURE);
        }
        if (strcmp(kind, "point") == 0 && pencil.kind != OMC_PENCIL_SSD) {
            printf("'source type = point' needs an 'ssd'.\n");
            exit(EXIT_FAILURE);
        }
    }
    else {
        printf("Unknown 'source type' value '%s'; it is 'pencil', 'point' or "
               "'phsp'.\n", kind);
        exit(EXIT_FAILURE);
    }

    return;
}

static void cleanSource(void) {

    if (spectrumBuilt) {
        omcSpectrumFree(&spectrum);
    }
    if (phspRead) {
        omcPhspFree(&phsp);
    }

    return;
}

/******************************************************************************/
/* Writing the results out.

 The .3ddose format with the axis it does not have taken out: the same idea,
 the same order, one fewer boundary list. */

static void outputResults(char *output_file, int iout,
                          const double *dose, const double *uncertainty) {

    int nr = geometry.isize;
    int nz = geometry.ksize;

    char extension[15];
    if (iout) {
        strcpy(extension, ".rzdose");
    } else {
        strcpy(extension, ".rzenergy");
    }

    char output_folder[BUFFER_SIZE];
    char buffer[BUFFER_SIZE];

    requireValue(buffer, "output folder");
    removeSpaces(output_folder, buffer);

    char* file_name = malloc(strlen(output_folder) + strlen(output_file) +
        strlen(extension) + 1);
    strcpy(file_name, output_folder);
    strcat(file_name, output_file);
    strcat(file_name, extension);

    FILE *fp;
    if ((fp = fopen(file_name, "w")) == NULL) {
        printf("Unable to open file: %s\n", file_name);
        exit(EXIT_FAILURE);
    }

    /* Grid dimensions: rings, then depth slabs */
    fprintf(fp, "%5d%5d\n", nr, nz);

    /* Ring boundaries, starting at the axis */
    for (int ir = 0; ir <= nr; ir++) {
        fprintf(fp, "%f ", geometry.rbounds[ir]);
    }
    fprintf(fp, "\n");

    /* Depth boundaries */
    for (int iz = 0; iz <= nz; iz++) {
        fprintf(fp, "%f ", geometry.zbounds[iz]);
    }
    fprintf(fp, "\n");

    /* Dose or energy, the ring running fastest */
    for (int iz = 0; iz < nz; iz++) {
        for (int ir = 0; ir < nr; ir++) {
            fprintf(fp, "%e ", dose[ir + iz*nr]);
        }
    }
    fprintf(fp, "\n");

    /* Relative uncertainty, in the same order */
    for (int iz = 0; iz < nz; iz++) {
        for (int ir = 0; ir < nr; ir++) {
            fprintf(fp, "%f ", uncertainty[ir + iz*nr]);
        }
    }
    fprintf(fp, "\n");

    printf("Results written to %s\n", file_name);

    fclose(fp);
    free(file_name);

    return;
}

/******************************************************************************/
/* Progress: one line per batch, with the elapsed time only this file knows */

static double tbegin;
static int batchesReported = 0;

static int reportProgress(double fraction, void *user) {

    (void)user;

    if (batchesReported == 0) {
        printf("%-10s\t%-15s\n", "Progress", "Elapsed time");
    }
    batchesReported++;

    printf("%-10.1f\t%-15.2f\n", 100.0*fraction, (omc_get_time() - tbegin));

    /* Nothing to stop the run for: it is the whole point of the program */
    return 1;
}

/******************************************************************************/
/* omc_dosrz main function */
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
            {"verbose", no_argument, &verbose_flag, 1},
            {"brief",   no_argument, &verbose_flag, 0},
            {"input",  required_argument, 0, 'i'},
            {"output",    required_argument, 0, 'o'},
            {0, 0, 0, 0}
        };

        int option_index = 0;

        c = getopt_long(argc, argv, "i:o:", long_options, &option_index);

        if (c == -1)
            break;

        switch (c) {
            case 0:
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
                break;

            default:
                exit(EXIT_FAILURE);
        }
    }

    if (input_file == NULL || output_file == NULL) {
        printf("Usage: omc_dosrz -i <input file, without .inp> "
               "-o <output name>\n");
        exit(EXIT_FAILURE);
    }

    if (verbose_flag)
        puts ("verbose flag is set");

    if (optind < argc)
    {
        printf ("non-option ARGV-elements: ");
        while (optind < argc)
            printf ("%s ", argv[optind++]);
        putchar ('\n');
    }

    parseInputFile(input_file);

#ifdef _OPENMP
    int omp_size = omp_get_num_procs();
    printf("Number of OpenMP threads: %d\n", omp_size);
    omp_set_num_threads(omp_size);
#else
    printf("ompMC compiled without OpenMP support. Serial execution.\n");
#endif

    /* Build the cylinder, which also names the one medium in it */
    initCylinder();

    /* With number of media and media names initialize the medium data */
    initMediaData();

    /* Initialize radiation source */
    initSource();

    /* Initialize data on a region-by-region basis */
    initRegions();

    /* Initialize VRT data */
    initVrt();

    if (verbose_flag) {
        listRayleigh();
        listPair();
        listPhoton();
        listElectron();
        listMscat();
        listSpin();
    }

    char buffer[BUFFER_SIZE];
    requireValue(buffer, "ncase");
    radialOptions.nhist = atoi(buffer);

    requireValue(buffer, "nbatch");
    radialOptions.nbatch = atoi(buffer);

    radialOptions.outputDose = 1;   /* Gy per incident history */

    int nregions = geometry.isize*geometry.ksize;
    double *dose = malloc(nregions*sizeof(double));
    double *uncertainty = malloc(nregions*sizeof(double));

    printf("Execution time up to this point : %8.2f seconds\n",
           (omc_get_time() - tbegin));

    struct OmcForwardCallbacks callbacks;
    callbacks.progress = reportProgress;
    callbacks.user = NULL;

    struct OmcForwardSummary summary;

    omcCalcRadial(&radialOptions, &source, NULL, dose, uncertainty,
                  &callbacks, &summary);

    printf("Simulation finished\n");
    printf("Execution time up to this point : %8.2f seconds\n",
           (omc_get_time() - tbegin));

    printf("Histories: %d run, %llu of them started a particle in the "
           "cylinder\n", summary.nhist, summary.started);
    printf("Fraction of incident energy deposited in the cylinder: %5.4f\n",
           summary.energyFraction);

    outputResults(output_file, radialOptions.outputDose, dose, uncertainty);

    /* Cleaning */
    free(dose);
    free(uncertainty);
    cleanCylinder();
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

    printf("Total execution time : %8.5f seconds\n",
           (omc_get_time() - tbegin));

    exit (EXIT_SUCCESS);
}

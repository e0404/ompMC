#ifndef OMPMC_H
#define OMPMC_H
/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations
 
 Copyright (C) 2020 Edgardo Doerner (edoerner@fis.puc.cl)


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

#if defined(_WIN32) || defined(_WIN64)
    /* We are on Windows */
    #define strtok_r strtok_s
#else
    extern char *strtok_r(char *, const char *, char **);
#endif

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

/*******************************************************************************
* User code definitions. The following functions must be provided by the user
* in its code.
*******************************************************************************/
void ausgab(double edep);    // scoring function
void howfar(int *idisc, int *irnew, double *ustep); // geometry functions
double hownear(void);

/*******************************************************************************
* Definitions for Monte Carlo simulation of particle transport 
*******************************************************************************/

/* Physical constants */
#define RM 0.5109989461     // MeV * c^(-2)

/* Common functions and definitions */
#define MXSTACK 10000 // maximum number of particles in stack

struct Stack {
    int np;         // stack pointer
    int npold;       // stack pointer before interactions
    
    int *iq;        // particle charge
    int *ir;        // current region
    
    double *e;      // total particle energy
    
    double *x;      // particle coordinates
    double *y;
    double *z;
    
    double *u;      // particle direction cosines
    double *v;
    double *w;
    
    double *dnear;  // perpendicular distance to nearest boundary
    double *wt;     // particle weight
};

#if defined(_MSC_VER)
	/* use __declspec(thread) instead of threadprivate to avoid
	error C3053. More information in:
	https://stackoverflow.com/questions/12560243/using-threadprivate-directive-in-visual-studio */
	extern __declspec(thread) struct Stack stack;
#else
	extern struct Stack stack;
	#pragma omp threadprivate(stack)
#endif

void initStack(void);
void cleanStack(void);

struct Uphi {
    /* This structure holds data saved between uphi() calls */
    double A, B, C;
    double cosphi, sinphi;
};

void transferProperties(int npnew, int npold);
void selectAzimuthalAngle(double *costhe, double *sinthe);
void uphi21(struct Uphi *uphi, double costhe, double sinthe);
void uphi32(struct Uphi *uphi, double costhe, double sinthe);
int pwlfInterval(int idx, double lvar, double *coef1, double *coef0);
double pwlfEval(int idx, double lvar, double *coef1, double *coef0);

/*******************************************************************************
* Photon physical processes definitions
*******************************************************************************/

#define MXGE 2000       // gamma mapped energy intervals
#define SGMFP 1.0E-05   // smallest gamma mean free path

struct Photon {
    double *ge0, *ge1;
    double *gmfp0, *gmfp1;
    double *gbr10, *gbr11;
    double *gbr20, *gbr21;
    double *cohe0, *cohe1;
};
struct Photon photon_data;

void readXsecData(char *file, int *ndat,
                  double **xsec_data0,
                  double **xsec_data1);

void heap_sort(int n, double *values, int *indices);
double *get_data(int flag, int ne, int *ndat,
                 double **data0, double **data1,
                 double *z_sorted, double *pz_sorted,
                 double ge0, double ge1);
double kn_sigma0(double e);

void initPhotonData(void);
void cleanPhoton(void);
void listPhoton(void);

/* Rayleigh scattering definitions */
#define MXRAYFF 100         // Rayleigh atomic form factor
#define RAYCDFSIZE 100      // CDF from Rayleigh form factors squared
#define HC_INVERSE 80.65506856998
#define TWICE_HC2 0.000307444456

struct Rayleigh {
    double *xgrid;
    double *fcum;
    double *b_array;
    double *c_array;
    double *pmax0;
    double *pmax1;
    int *i_array;
};
struct Rayleigh rayleigh_data;

void readFfData(double *xval, double **aff);
void initRayleighData(void);
void cleanRayleigh(void);
void listRayleigh(void);
void rayleigh(int imed, double eig, double gle, int lgle);

/* Pair production definitions */
#define FSC 0.00729735255664    // fine structure constant

struct Pair {
    double *dl1;
    double *dl2;
    double *dl3;
    double *dl4;
    double *dl5;
    double *dl6;
    
    double *bpar0;
    double *bpar1;
    double *delcm;
    double *zbrang;
};
struct Pair pair_data;

double fcoulc(double zi);
double xsif(double zi, double fc);
void initPairData(void);
void cleanPair(void);
void listPair(void);
double setPairRejectionFunction(int imed, double xi, double esedei,
                                double eseder, double tteig);
void pair(int imed);

/* Compton scattering definitions */
void compton(void);

/* Photo electric effect definitions */
void photo(void);

/* Simulation of photon step */
void photon(void);

/*******************************************************************************
* Electron physical processes definitions
*******************************************************************************/
#define XIMAX 0.5
#define ESTEPE 0.25
#define EPSEMFP 1.0E-5      // smallest electron mean free path
#define SKIN_DEPTH_FOR_BCA 3

struct Electron {
    double *esig0;
    double *esig1;
    double *psig0;
    double *psig1;
    
    double *ededx0;
    double *ededx1;
    double *pdedx0;
    double *pdedx1;
    
    double *ebr10;
    double *ebr11;
    double *pbr10;
    double *pbr11;
    
    double *pbr20;
    double *pbr21;
    
    double *tmxs0;
    double *tmxs1;
    
    double *blcce0;
    double *blcce1;
    
    double *etae_ms0;
    double *etae_ms1;
    double *etap_ms0;
    double *etap_ms1;
    
    double *q1ce_ms0;
    double *q1ce_ms1;
    double *q1cp_ms0;
    double *q1cp_ms1;
    
    double *q2ce_ms0;
    double *q2ce_ms1;
    double *q2cp_ms0;
    double *q2cp_ms1;
    
    double *range_ep;
    double *e_array;
    
    double *eke0;
    double *eke1;
    
    int *sig_ismonotone;
    
    double *esig_e;
    double *psig_e;
    double *xcc;
    double *blcc;
    double *expeke1;
    
};
struct Electron electron_data;

void cleanElectron(void);
void listElectron(void);

/* Spin data */
#define MXE_SPIN 15
#define MXE_SPIN1 2*MXE_SPIN+1
#define MXQ_SPIN 15
#define MXU_SPIN 31

struct Spin {
    double b2spin_min;
    double dbeta2i;
    double espml;
    double dleneri;
    double dqq1i;
    double *spin_rej;
};
struct Spin spin_data;

struct Spinr {
    /* This structure holds data saved between spinRejection calls */
    int i;
    int j;
};

void initSpinData(int nmed);
void cleanSpin(void);
void listSpin(void);
void setSpline(double *x, double *f, double *a, double *b, double *c,
                double *d,int n);
double spline(double s, double *x, double *a, double *b, double *c,
              double *d, int n);
double spinRejection(int imed, int qel,	double elke, double beta2, 
    double q1, double cost, int *spin_index, int is_single, 
    struct Spinr *spin_r);
void sscat(int imed, int qel, double chia2, double elke, double beta2,
	double *cost, double *sint);

/* Screened Rutherford MS data */
#define MXL_MS 63
#define MXQ_MS 7
#define MXU_MS 31
#define LAMBMIN_MS 1.0
#define LAMBMAX_MS 1.0E5
#define QMIN_MS 1.0E-3
#define QMAX_MS 0.5

struct Mscat {

    double *ums_array;
    double *fms_array;
    double *wms_array;
    int *ims_array;
    
    double dllambi;
    double dqmsi;
};
struct Mscat mscat_data;

struct Mscats {
    /* This structure holds data saved between mscat calls */
    int i;
    int j;
    double omega2;
};

void readRutherfordMscat(int nmed);
void initMscatData();
void cleanMscat(void);
void listMscat(void);
void mscat(int imed, int qel, int *spin_index, int *find_index, 
    double elke, double beta2, double q1,  double lambda, double chia2, 
    double *cost, double *sint, struct Mscats *m_scat, struct Spinr *spin_r);
double msdist(int imed, int iq, double rhof, double de, double tustep, 
    double eke, double *x_final, double *y_final, double *z_final, 
    double *u_final, double *v_final, double *w_final);

/* CSDA related definitions */
double computeDrange(int imed, int iq, int lelke, double ekei,  
    double ekef, double elkei, double elkef);
double computeEloss(int imed, int iq, int irl, double rhof, 
    double tustep, double range, double eke, double elke, int lelke);

/* Annihilation in rest */
void rannih(void);

/* Bremsstrahlung */
void brems(void);

/* Moller scattering */
void moller(void);

/* Bhabha scattering */
void bhabha(void);

/* Annihilation in flight */
void annih(void);

/* Simulation of electron step */
void electron(void);

/* Simulation of the particle history */
void shower(void);

/* Media definitions */
#define MXMED 9         // maximum number of media supported by the platform
#define MXELEMENT 50    // maximum number of elements in a single medium
#define MXEKE 500       // electron mapped energy intervals

struct Media {
    int nmed;                   // number of media in the problem
    char med_names[MXMED][60];  // media names
};
struct Media media;

struct Element {
    /* Attributes of an element in a medium */
    char symbol[3];
    double z;
    double wa;
    double pz;
    double rhoz;
    
};

struct Pegs {
    /* Data extracted from pegs file */
    char names[MXMED][60];          // media names (as found in .pegs4dat file)
    int ne[MXMED];                  // number of elements in medium
    int iunrst[MXMED];              // flag for type of stopping power
    int epstfl[MXMED];              // flag for ICRU37 collision stopping powers
    int iaprim[MXMED];              // flag for ICRU37 radiative stopping powers
    
    int msge[MXMED];
    int mge[MXMED];
    int mseke[MXMED];
    int meke[MXMED];
    int mleke[MXMED];
    int mcmfp[MXMED];
    int mrange[MXMED];
    
    double rho[MXMED];              // mass density of medium
    double rlc[MXMED];              // radiation length for the medium (in cm)
    double ae[MXMED], ap[MXMED];    // electron and photon creation threshold E
    double ue[MXMED], up[MXMED];    // upper electron and photon energy
    double te[MXMED];
    double thmoll[MXMED];
    double delcm[MXMED];
    
    struct Element elements[MXMED][MXELEMENT];  // element properties
};
struct Pegs pegs_data;

void initMediaData(void);
int readPegsFile(int *media_found);

/* Region-by-region definition */
#define VACUUM -1

struct Region {
    int *med;
    double *rhof;
    double *pcut;
    double *ecut;
};
struct Region region;

void initRegions(void);  // this function must be defined in user code
void cleanRegions(void);

/******************************************************************************/

/*******************************************************************************
* Variance reduction techniques definitions
*******************************************************************************/

struct Vrt {
    /* photon splitting */
    int nsplit; // number of times the photon is divided
};
struct Vrt vrt;

void initVrt(void);

/******************************************************************************/

#endif  // OMPMC_H

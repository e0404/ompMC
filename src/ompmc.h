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

/* Region containing the point (x,y,z), 0 if the point lies outside the
 geometry. Photon transport uses Woodcock (delta) tracking, which jumps to
 arbitrary points instead of marching from voxel face to voxel face, so it
 needs point location rather than howfar()'s directed distances. Electron
 transport still uses howfar()/hownear(). */
int regionIndex(double x, double y, double z);

/*******************************************************************************
* Definitions for Monte Carlo simulation of particle transport 
*******************************************************************************/

/* Physical constants */
#define RM 0.5109989461     // MeV * c^(-2)

/* Common functions and definitions */
#define MXSTACK 10000 // maximum number of particles in stack

//typedef struct Stack Stack;

/* One entry of the particle stack. The transport code works on a single
 particle at a time, indexed by stack.np, and never sweeps a field across
 particles, so these live together rather than in eleven parallel arrays: a
 particle is then two cache lines and one page instead of eleven of each. */
struct Particle {
    double x;       // particle coordinates
    double y;
    double z;

    double u;       // particle direction cosines
    double v;
    double w;

    double e;       // total particle energy
    double wt;      // particle weight
    double dnear;   // perpendicular distance to nearest boundary

    int iq;         // particle charge
    int ir;         // current region
};

struct Stack {
    int np;         // stack pointer
    int npold;      // stack pointer before interactions

    struct Particle *p;
};

/*
#if defined(_MSC_VER)
	//use __declspec(thread) instead of threadprivate to avoid 
	//error C3053. More information in:
	// https://stackoverflow.com/questions/12560243/using-threadprivate-directive-in-visual-studio 
	__declspec(thread) struct Stack stack;
#else
	struct Stack stack;
    #pragma omp threadprivate(stack)
#endif
*/

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
/* coef holds interleaved {slope, intercept} pairs: entry idx at coef[2*idx]
 and coef[2*idx + 1] */
double pwlfEval(int idx, double lvar, const double *coef);

/*******************************************************************************
* Photon physical processes definitions
*******************************************************************************/

#define MXGE 2000       // gamma mapped energy intervals
#define SGMFP 1.0E-05   // smallest gamma mean free path

/* The per-energy pwlf tables hold their {slope, intercept} coefficient
 pairs interleaved -- entry i lives at [2*i] and [2*i + 1] -- so one lookup
 touches one cache line instead of two. The per-medium mapping pairs
 (ge0/ge1, eke0/eke1) stay separate: they are indexed by medium only and
 always cache-resident. */
struct Photon {
    double *ge0, *ge1;
    double *gmfp;
    double *gbr1;
    double *gbr2;
    double *cohe;
};

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
    double *pmax;       /* interleaved pwlf pairs, see struct Photon */
    int *i_array;
};


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

/* All per-energy tables hold interleaved {slope, intercept} pwlf pairs,
 see struct Photon. eke0/eke1 are the per-medium mapping coefficients and
 stay separate. */
struct Electron {
    double *esig;
    double *psig;

    double *ededx;
    double *pdedx;

    double *ebr1;
    double *pbr1;

    double *pbr2;

    double *tmxs;

    double *blcce;

    double *etae_ms;
    double *etap_ms;

    double *q1ce_ms;
    double *q1cp_ms;

    double *q2ce_ms;
    double *q2cp_ms;

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

void initMediaData(void);
int readPegsFile(int *media_found);

/* Region-by-region definition */
#define VACUUM -1

struct Region {
    int *med;       // medium index, per region
    double *rhof;   // mass density ratio, per region

    /* Photon and electron transport cut-offs. These are a property of the
     medium, not of the individual voxel: initRegions() sets them to
     max(global cut, the medium's PEGS threshold), so every voxel of a given
     medium held an identical copy. Storing them per medium instead keeps two
     arrays the size of the whole geometry out of the transport loop's working
     set -- on a 13.8M voxel dose grid that is 220 MB no longer being read at
     random -- and leaves them permanently in L1.

     Indexed by medium + 1, so that VACUUM (-1) lands on slot 0, which holds
     zero for both. Use regionPcut()/regionEcut() rather than indexing this
     directly. */
    double pcut[MXMED + 1];
    double ecut[MXMED + 1];

    /* Largest mass density ratio among the voxels of each medium, filled by
     the user code's initRegions(). Woodcock photon tracking builds its
     majorant cross-section from these: for a given energy no voxel of
     medium m can attenuate more strongly than rhof_max[m] times the
     medium's tabulated inverse mean free path. Media without any voxel keep
     zero and simply never bound the majorant. */
    double rhof_max[MXMED];
};

extern struct Region region;

/* Transport cut-offs for the region irl. region.med[irl] is on the same cache
 line as the medium lookup the caller has almost always just done, so this
 costs an L1 hit and an index into a table that never leaves L1. */
static inline double regionPcut(int irl) {
    return region.pcut[region.med[irl] + 1];
}

static inline double regionEcut(int irl) {
    return region.ecut[region.med[irl] + 1];
}

void initRegions(void);  // this function must be defined in user code
void cleanRegions(void);

/******************************************************************************/

/*******************************************************************************
* Variance reduction techniques definitions
*******************************************************************************/

struct Vrt {
    /* photon splitting */
    int nsplit; // number of times the photon is divided

    /* Electron range rejection: an electron of total energy below esave
     (MeV) whose residual CSDA range is shorter than the perpendicular
     distance to the closest region boundary cannot leave its voxel, so its
     remaining energy is deposited on the spot (positrons still emit their
     annihilation photons). The approximation is that bremsstrahlung the
     electron would have radiated below esave is absorbed locally; keep
     esave modest (~2 MeV) so that loss stays negligible. 0 disables. */
    double esave;

    /* Unbiased Russian roulette of electrons at their creation point: a new
     electron of total energy below e_rr (MeV) survives with probability
     1/f_rr and carries f_rr times its weight; otherwise it is removed
     without depositing. Unlike range rejection this also kills electrons in
     the boundary-crossing zone, at the price of lumpier dose from the
     amplified survivors. Enabled when e_rr > 0 and f_rr > 1. */
    double e_rr;
    double f_rr;
};

void initVrt(void);

/******************************************************************************/

#endif  // OMPMC_H

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

/*******************************************************************************
* Counter-based random number generator built on Philox4x32-10. See
* omc_random.h for the stream layout. The implementation follows the
* reference in Salmon et al., SC'11 (the Random123 library) and reproduces
* its published test vectors, which the unit tests check.
*******************************************************************************/

#include "omc_random.h"
#include "omc_utilities.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Redefine printf() function due to conflicts with mex and OpenMP */
#ifdef _OPENMP
    #include <omp.h>

    #undef printf
    #define printf(...) fprintf(stderr,__VA_ARGS__)
#endif

/* Common functions and definitions */
#if defined(_MSC_VER)
	/* use __declspec(thread) instead of threadprivate to avoid
	error C3053. More information in:
	https://stackoverflow.com/questions/12560243/using-threadprivate-directive-in-visual-studio */
	__declspec(thread) struct Random rng;
#else
	#pragma omp threadprivate(rng)
	struct Random rng;
#endif

/* Philox4x32 round multipliers and key schedule constants */
#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u
#define PHILOX_W1 0xBB67AE85u

void philox4x32(const uint32_t ctr[4], const uint32_t key[2],
                uint32_t out[4]) {

    uint32_t c0 = ctr[0], c1 = ctr[1], c2 = ctr[2], c3 = ctr[3];
    uint32_t k0 = key[0], k1 = key[1];

    for (int round = 0; round < 10; round++) {
        uint64_t p0 = (uint64_t)PHILOX_M0*c0;
        uint64_t p1 = (uint64_t)PHILOX_M1*c2;

        c0 = (uint32_t)(p1 >> 32) ^ c1 ^ k0;
        c1 = (uint32_t)p1;
        c2 = (uint32_t)(p0 >> 32) ^ c3 ^ k1;
        c3 = (uint32_t)p0;

        k0 += PHILOX_W0;
        k1 += PHILOX_W1;
    }

    out[0] = c0;
    out[1] = c1;
    out[2] = c2;
    out[3] = c3;

    return;
}

/* Read the seeds and initialize the thread-local generator state. Unlike the
 RANMAR version there is no thread-dependent seeding: streams are separated
 by history index, so every thread carries the same key. */
void initRandom() {

    int ixx, jxx;

    /* Get initial seeds from input */
    char buffer[BUFF_SIZE];
    if (getInputValue(buffer, "rng seeds") != 1) {
        printf("Can not find 'rng seeds' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    if (sscanf(buffer, "%d %d", &ixx, &jxx) != 2) {
        printf("Could not parse two integers from 'rng seeds'.\n");
        exit(EXIT_FAILURE);
    }

    rng.key[0] = (uint32_t)ixx;
    rng.key[1] = (uint32_t)jxx;

    /* Park the generator on the all-ones history index, which no real
     history can own, so that any draw made before the first
     setRandomHistory() call still comes from a well defined stream */
    rng.ctr[0] = 0;
    rng.ctr[1] = 0;
    rng.ctr[2] = 0xFFFFFFFFu;
    rng.ctr[3] = 0xFFFFFFFFu;
    rng.buf_pos = 4;

#ifdef _OPENMP
    if (omp_get_thread_num() == 0)
#endif
    printf("RNG : Philox4x32-10, key = %d %d\n", ixx, jxx);

    return;
}

void setRandomHistory(uint64_t ihist) {

    rng.ctr[0] = 0;
    rng.ctr[1] = 0;
    rng.ctr[2] = (uint32_t)ihist;
    rng.ctr[3] = (uint32_t)(ihist >> 32);
    rng.buf_pos = 4;

    return;
}

/* Get a single floating random number in (0,1) using the Philox RNG */
double setRandom() {

    if (rng.buf_pos > 3) {
        uint32_t out[4];
        philox4x32(rng.ctr, rng.key, out);

        /* 2^64 draws per history; the carry never reaches the history words */
        if (++rng.ctr[0] == 0) {
            ++rng.ctr[1];
        }

        /* The half offset centers each value in its 2^-32 cell, keeping the
         mean at 1/2 and excluding the exact endpoints, so callers may take
         log(r) or log(1-r) without a zero guard */
        rng.buf[0] = ((double)out[0] + 0.5)*TWOM32;
        rng.buf[1] = ((double)out[1] + 0.5)*TWOM32;
        rng.buf[2] = ((double)out[2] + 0.5)*TWOM32;
        rng.buf[3] = ((double)out[3] + 0.5)*TWOM32;
        rng.buf_pos = 0;
    }

    return rng.buf[rng.buf_pos++];
}

double erfinv_approx(double x) {
   double tt1, tt2, lnx, sgn;
   sgn = (x < 0) ? -1.0 : 1.0;

   x = (1.0 - x)*(1.0 + x);        // x = 1 - x*x;
   lnx = log(x);

   tt1 = 2.0/(M_PI*0.147) + 0.5 * lnx;
   tt2 = 1.0/(0.147) * lnx;

   return(sgn*sqrt(-tt1 + sqrt(tt1*tt1 - tt2)));
}

double setStandardNormalRandom(const double mu, const double sigma) {
    double rnno = setRandom();
    rnno = sqrt(2.0) * erfinv_approx(2.0*rnno - 1.0);
    rnno = mu + sigma * rnno;
    return rnno;
}

void boxMuller(double rndnormal[2])
{
    rndnormal[0] = setRandom();
    rndnormal[1] = setRandom();

    double R = sqrt(-2.0*log(rndnormal[0]));
    double Theta = 2.0*M_PI*rndnormal[1];

    rndnormal[0] = R*cos(Theta);
    rndnormal[1] = R*sin(Theta);
}

void cleanRandom() {

    /* The counter-based generator holds no allocations; kept so user codes
     need not special-case the teardown */
    return;
}

/******************************************************************************/

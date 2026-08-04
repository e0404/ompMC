#ifndef OMC_RANDOM_H
#define OMC_RANDOM_H
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
* Counter-based random number generator built on Philox4x32-10 (Salmon,
* Moraes, Dror and Shaw, "Parallel random numbers: as easy as 1, 2, 3",
* SC'11). It replaces the RANMAR port used previously.
*
* The generator is a pure function of a 64 bit key and a 128 bit counter.
* The key comes from the 'rng seeds' input. The high 64 bits of the counter
* hold the global history index, set through setRandomHistory() at the start
* of every particle history; the low 64 bits count the draws within the
* history. Every history therefore owns its own stream of 2^64 numbers,
* determined only by the seeds and the history index -- never by the thread
* that happens to simulate it or by how histories are scheduled.
*
* Before using the RNG, it is needed to initialize the RNG by a call to
* initRandom().
*******************************************************************************/

#include <stdint.h>

#define BUFF_SIZE 256

/* Scale factor turning 32 bit words into reals. Exact in binary floating
 point. */
#define TWOM32 (1.0/4294967296.0)

struct Random {
    uint32_t key[2];    /* base key, taken from the 'rng seeds' input */
    uint32_t ctr[4];    /* ctr[2],ctr[3] hold the history index; ctr[0],
                         ctr[1] count the blocks drawn within the history */
    int buf_pos;        /* next unread entry of buf; 4 means empty */
    double buf[4];      /* one Philox block converted to reals in (0,1) */
};

#if defined(_MSC_VER)
	/* use __declspec(thread) instead of threadprivate to avoid
	error C3053. More information in:
	https://stackoverflow.com/questions/12560243/using-threadprivate-directive-in-visual-studio */
	extern __declspec(thread) struct Random rng;
#else
	extern struct Random rng;
	#pragma omp threadprivate(rng)
#endif
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

/* Read the 'rng seeds' input into the thread-local key and leave the
 generator on a sentinel stream no real history uses. Call once per thread
 before any setRandom(). */
void initRandom(void);

/* Point the generator at the stream owned by global history index ihist.
 Call at the start of every particle history; the index must be unique over
 the whole run (across batches, and beamlets where applicable). */
void setRandomHistory(uint64_t ihist);

/* Get a single floating random number in (0,1) from the current stream */
double setRandom(void);

/* One Philox4x32-10 block: 128 bit counter and 64 bit key in, four 32 bit
 words out. Exposed for verification against the published test vectors. */
void philox4x32(const uint32_t ctr[4], const uint32_t key[2],
                uint32_t out[4]);

void cleanRandom(void);

double setStandardNormalRandom(const double mu, const double sigma);

void boxMuller(double rndnormal[2]);

/******************************************************************************/

#endif

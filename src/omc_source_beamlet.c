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

#include "omc_source_beamlet.h"

#include "omc_host.h"
#include "omc_random.h"
#include "omc_spectrum.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/******************************************************************************/

int omcBeamletProduce(const struct OmcBeamletSampler *sampler, int ibeamlet,
                      double weight, struct OmcSourceParticle *particle) {

    const struct OmcBeamletSource *source = sampler->source;

    particle->charge = sampler->charge;

    /* WARNING: the random numbers below are drawn in an order the results
     depend on. The generator is indexed per history, so a draw added,
     removed or moved here shifts every later draw of the same history and
     changes the dose. omcSpectrumSample() deliberately draws NOTHING for a
     monoenergetic source for the same reason. */
    particle->energy = omcSpectrumSample(sampler->spectrum);

    /* A point of the beamlet aperture, uniformly. */
    double rnno1 = setRandom();
    double rnno2 = setRandom();

    double xiso = rnno1*source->xside1[ibeamlet]
                + rnno2*source->xside2[ibeamlet] + source->xcorner[ibeamlet];
    double yiso = rnno1*source->yside1[ibeamlet]
                + rnno2*source->yside2[ibeamlet] + source->ycorner[ibeamlet];
    double ziso = rnno1*source->zside1[ibeamlet]
                + rnno2*source->zside2[ibeamlet] + source->zcorner[ibeamlet];

    int ibeam = source->ibeam[ibeamlet];
    double sourcePos[3];

    switch (sampler->geometry)
    {
        case OMC_SOURCE_POINT: ;
            sourcePos[0] = source->xsource[ibeam];
            sourcePos[1] = source->ysource[ibeam];
            sourcePos[2] = source->zsource[ibeam];
            break;
        case OMC_SOURCE_GAUSSIAN: ;
            /* Get the normalized collimator plane vectors */
            double planeVec1_norm = sqrt(
                source->xside1[ibeamlet]*source->xside1[ibeamlet] +
                source->yside1[ibeamlet]*source->yside1[ibeamlet] +
                source->zside1[ibeamlet]*source->zside1[ibeamlet]);
            double planeVec2_norm = sqrt(
                source->xside2[ibeamlet]*source->xside2[ibeamlet] +
                source->yside2[ibeamlet]*source->yside2[ibeamlet] +
                source->zside2[ibeamlet]*source->zside2[ibeamlet]);

            double planeVec1[3];
            planeVec1[0] = source->xside1[ibeamlet]/planeVec1_norm;
            planeVec1[1] = source->yside1[ibeamlet]/planeVec1_norm;
            planeVec1[2] = source->zside1[ibeamlet]/planeVec1_norm;

            double planeVec2[3];
            planeVec2[0] = source->xside2[ibeamlet]/planeVec2_norm;
            planeVec2[1] = source->yside2[ibeamlet]/planeVec2_norm;
            planeVec2[2] = source->zside2[ibeamlet]/planeVec2_norm;

            /* Create two normally distributed random variables with the
             box-muller transform */
            double rnSource[2];
            boxMuller(rnSource);

            /* Scale with source width */
            rnSource[0] *= sampler->gaussianWidth;
            rnSource[1] *= sampler->gaussianWidth;

            /* Now use the plane vectors to add the random 2D offset to the
             source */
            sourcePos[0] = source->xsource[ibeam]
                + rnSource[0]*planeVec1[0] + rnSource[1]*planeVec2[0];
            sourcePos[1] = source->ysource[ibeam]
                + rnSource[0]*planeVec1[1] + rnSource[1]*planeVec2[1];
            sourcePos[2] = source->zsource[ibeam]
                + rnSource[0]*planeVec1[2] + rnSource[1]*planeVec2[2];
            break;
        default: ;
            /* Checked before the parallel region starts, so this is only a
             backstop; omcFail() from a worker thread would call the host from
             a place the host cannot expect. */
            sourcePos[0] = source->xsource[ibeam];
            sourcePos[1] = source->ysource[ibeam];
            sourcePos[2] = source->zsource[ibeam];
    }

    /* The particle starts at the source and flies through the point sampled
     on the aperture. Where it meets the phantom is omcSourcePlace()'s
     business, not this one's. */
    double xd = xiso - sourcePos[0];
    double yd = yiso - sourcePos[1];
    double zd = ziso - sourcePos[2];

    double vnorm = sqrt(xd*xd + yd*yd + zd*zd);

    particle->x = sourcePos[0];
    particle->y = sourcePos[1];
    particle->z = sourcePos[2];

    particle->u = xd/vnorm;
    particle->v = yd/vnorm;
    particle->w = zd/vnorm;

    particle->weight = weight;

    return 1;
}

/******************************************************************************/
/* Handing the histories out to the beamlets.

 Every batch runs the same experiment: beamlet i contributes the same count[i]
 histories to each of them, so the batches stay the independent replicas the
 variance estimate assumes. The counts are shared out by walking the cumulative
 weight, which needs no sort, is deterministic, and leaves each count within
 one history of the exact share.

 A beamlet whose share rounds to zero is dropped rather than given a history it
 has not earned. What that costs is bounded -- its weight is below one
 nperbatch-th of the total -- and it is reported rather than hidden.

 The rounding that is left over rides on the particle weight instead of on the
 counts: beamlet i wants weight[i] of the fluence and got count[i] of the
 nperbatch histories, so each of its particles carries

     wt[i] = (weight[i]/count[i]) / (W/nperbatch)

 which is 1 up to the rounding, and exactly 1 when the share came out whole.
 Keeping it near 1 rather than folding the whole fluence into it leaves the
 transport working with the weights it has always seen; the physical scale
 goes on the batch instead, through struct OmcSource::batchScale. */

struct Allocation {
    int *count;                 // histories per beamlet per batch
    int *offset;                // prefix sum, nbeamlets + 1 entries
    double *weight;             // statistical weight of each beamlet's particles

    int nweighted;              // beamlets asked for with a weight above zero
    int nsampled;               // of those, the ones that got any histories
    double totalWeight;
    double sampledWeight;
};

static void freeAllocation(struct Allocation *a) {

    free(a->count);
    free(a->offset);
    free(a->weight);

    a->count = NULL;
    a->offset = NULL;
    a->weight = NULL;

    return;
}

static void buildAllocation(struct Allocation *a, int nbeamlets,
                            const double *weights, int nperbatch) {

    a->count = (int*) malloc((size_t)nbeamlets*sizeof(int));
    a->offset = (int*) malloc(((size_t)nbeamlets + 1)*sizeof(int));
    a->weight = (double*) malloc((size_t)nbeamlets*sizeof(double));

    if (!a->count || !a->offset || !a->weight) {
        freeAllocation(a);
        omcFail("ompMC:forward:outOfMemory",
            "Could not allocate the history distribution for %d beamlets.",
            nbeamlets);
    }

    /* Total weight, and the heaviest beamlet, which absorbs the rounding of
     the cumulative sum at the end */
    double total = 0.0;
    int heaviest = 0;

    for (int i = 0; i < nbeamlets; i++) {
        if (!isfinite(weights[i]) || weights[i] < 0.0) {
            freeAllocation(a);
            omcFail("ompMC:forward:invalidWeight",
                "Beamlet weight %d is %g; weights must be finite and zero "
                "or positive.",
                i + 1, weights[i]);
        }
        if (weights[i] > weights[heaviest]) {
            heaviest = i;
        }
        total += weights[i];
    }

    /* Individually finite values can still overflow when summed. That would
     make every cumulative/total allocation ratio invalid. */
    if (!isfinite(total)) {
        freeAllocation(a);
        omcFail("ompMC:forward:invalidWeight",
            "The beamlet weights sum to infinity; their scale has to leave "
            "the total finite.");
    }

    if (total <= 0.0) {
        freeAllocation(a);
        omcFail("ompMC:forward:noWeight",
            "Every beamlet weight is zero, so there is nothing to calculate.");
    }

    a->totalWeight = total;
    a->nweighted = 0;
    a->nsampled = 0;
    a->sampledWeight = 0.0;

    /* Walk the cumulative weight: beamlet i gets the histories between the
     rounded cumulative share before it and the one after it. */
    double cumulative = 0.0;
    int placed = 0;

    a->offset[0] = 0;

    for (int i = 0; i < nbeamlets; i++) {
        cumulative += weights[i];

        int upto = (int)((cumulative/total)*(double)nperbatch + 0.5);
        if (i == nbeamlets - 1) {
            upto = nperbatch;   /* the last one closes the account exactly */
        }
        if (upto < placed) {
            upto = placed;
        }
        if (upto > nperbatch) {
            upto = nperbatch;
        }

        a->count[i] = upto - placed;
        placed = upto;
        a->offset[i + 1] = placed;

        if (weights[i] > 0.0) {
            a->nweighted++;
        }

        if (a->count[i] > 0) {
            /* wt = (weight/count) / (total/nperbatch), which is 1 when the
             share came out whole. */
            a->weight[i] = (weights[i]/(double)a->count[i])
                           /(total/(double)nperbatch);
            if (weights[i] > 0.0) {
                a->nsampled++;
                a->sampledWeight += weights[i];
            }
        }
        else {
            a->weight[i] = 0.0;
        }
    }

    return;
}

/* Which beamlet history h of a batch belongs to: the last one whose offset is
 at or below it. A binary search rather than a walk, because this runs once
 per history. */
static int findBeamlet(const int *offset, int nbeamlets, int h) {

    int lo = 0;
    int hi = nbeamlets;         // the answer is in [lo, hi)

    while (hi - lo > 1) {
        int mid = lo + (hi - lo)/2;

        if (offset[mid] <= h) {
            lo = mid;
        }
        else {
            hi = mid;
        }
    }

    return lo;
}

/******************************************************************************/
/* The source interface */

static void beamletCheck(const struct OmcSource *self) {

    const struct OmcBeamletHistories *h =
        (const struct OmcBeamletHistories *)self->impl;

    if (h->sampler.geometry != OMC_SOURCE_POINT &&
        h->sampler.geometry != OMC_SOURCE_GAUSSIAN) {
        omcFail("ompMC:forward:invalidSourceGeometry",
            "Source geometry %d is not defined.", (int)h->sampler.geometry);
    }
    if (h->sampler.source == NULL || h->sampler.source->nbeamlets < 1) {
        omcFail("ompMC:forward:noBeamlets",
            "There are no beamlets to calculate.");
    }
    if (h->weights == NULL) {
        omcFail("ompMC:forward:invalidWeight",
            "The beamlets have no weights.");
    }

    return;
}

static void beamletPrepare(struct OmcSource *self, int nperbatch) {

    struct OmcBeamletHistories *h = (struct OmcBeamletHistories *)self->impl;

    struct Allocation *a = (struct Allocation *) malloc(sizeof(*a));
    if (a == NULL) {
        omcFail("ompMC:forward:outOfMemory",
            "Could not allocate the history distribution.");
    }

    a->count = NULL;
    a->offset = NULL;
    a->weight = NULL;

    buildAllocation(a, h->sampler.source->nbeamlets, h->weights, nperbatch);

    h->allocation = a;

    /* Copied out of the working memory now, because the engine gives that
     back through release() before it returns and the caller only gets to
     ask afterwards. None of these four change once the histories have been
     shared out, so the copy stays true for the whole run. */
    h->stats.nweighted = a->nweighted;
    h->stats.nsampled = a->nsampled;
    h->stats.totalWeight = a->totalWeight;
    h->stats.sampledWeight = a->sampledWeight;

    omcLog(OMC_LOG_DETAIL, "Beamlets with weight: %d of %d, %d of them sampled",
           a->nweighted, h->sampler.source->nbeamlets, a->nsampled);

    double dropped = a->totalWeight - a->sampledWeight;

    if (dropped > 1.0E-3*a->totalWeight) {
        omcLog(OMC_LOG_WARNING,
            "%d of %d weighted beamlets are too weak to be given a history "
            "each batch, which leaves out %.2f%% of the fluence. Raise the "
            "number of histories or lower the number of batches.",
            a->nweighted - a->nsampled, a->nweighted,
            100.0*dropped/a->totalWeight);
    }

    /* The particle weights sum a batch to the fluence of nperbatch histories
     rather than to the fluence the caller asked for, so the ratio between
     the two rides on the batch -- once per batch, rather than on every
     particle. What comes out is then the dose for exactly these weights. */
    self->batchScale = a->totalWeight/(double)nperbatch;
    self->incidentFluence = 1.0;

    return;
}

static int beamletSample(const struct OmcSource *self, uint64_t ihist,
                         int ihistInBatch, struct OmcSourceParticle *particle) {

    const struct OmcBeamletHistories *h =
        (const struct OmcBeamletHistories *)self->impl;
    const struct Allocation *a = (const struct Allocation *)h->allocation;

    (void)ihist;

    int ibeamlet = findBeamlet(a->offset, h->sampler.source->nbeamlets,
                               ihistInBatch);

    return omcBeamletProduce(&h->sampler, ibeamlet, a->weight[ibeamlet],
                             particle);
}

static void beamletRelease(struct OmcSource *self) {

    struct OmcBeamletHistories *h = (struct OmcBeamletHistories *)self->impl;
    struct Allocation *a = (struct Allocation *)h->allocation;

    if (a != NULL) {
        freeAllocation(a);
        free(a);
        h->allocation = NULL;
    }

    return;
}

void omcBeamletHistoriesAsSource(struct OmcBeamletHistories *histories,
                                 struct OmcSource *source) {

    histories->allocation = NULL;
    memset(&histories->stats, 0, sizeof(histories->stats));

    source->check = beamletCheck;
    source->prepare = beamletPrepare;
    source->sample = beamletSample;
    source->release = beamletRelease;
    source->impl = histories;
    source->batchScale = 1.0;
    source->incidentFluence = 1.0;

    return;
}

void omcBeamletHistoriesStats(const struct OmcBeamletHistories *histories,
                              struct OmcBeamletStats *stats) {

    *stats = histories->stats;

    return;
}

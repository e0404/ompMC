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

#include "omc_source_phsp.h"

#include "omc_geom.h"
#include "omc_host.h"
#include "omc_phsp.h"
#include "omc_random.h"

#include <math.h>
#include <stddef.h>             /* NULL; math.h happens to bring it on some
                                   platforms and not on glibc */

/******************************************************************************/

void omcPhspTransformIdentity(struct OmcPhspTransform *transform) {

    for (int i = 0; i < 9; i++) {
        transform->rotation[i] = (i % 4) == 0 ? 1.0 : 0.0;
    }
    for (int i = 0; i < 3; i++) {
        transform->translation[i] = 0.0;
    }

    return;
}

void omcPhspSourceCheck(const struct OmcPhspSampler *sampler) {

    if (sampler->phsp == NULL || omcPhspCount(sampler->phsp) == 0) {
        omcFail("ompMC:phspSource:noParticles",
            "The phase space source has no particles to draw from.");
    }

    if (sampler->order != OMC_PHSP_REPLAY &&
        sampler->order != OMC_PHSP_RANDOM) {
        omcFail("ompMC:phspSource:badOrder",
            "The phase space source was given order %d, and knows %d for "
            "replaying the file and %d for drawing from it at random.",
            (int)sampler->order, (int)OMC_PHSP_REPLAY, (int)OMC_PHSP_RANDOM);
    }

    if (geometry.xbounds == NULL || geometry.ybounds == NULL ||
        geometry.zbounds == NULL) {
        omcFail("ompMC:phspSource:noGeometry",
            "The phase space source needs the phantom set up before it can "
            "work out where its particles enter one.");
    }

    /* Whether the rotation is one. A caller who meant to turn the phase
     space by 30 degrees and mistyped a matrix element would otherwise find
     out from the dose distribution.

     The determinant alone does not settle it: a shear such as
     {{1,1,0},{0,1,0},{0,0,1}} has determinant 1 and still stretches what it
     turns, and a matrix holding a NaN passes any comparison asked of it
     because every comparison against NaN is false. What makes a matrix a
     rotation is that its rows are unit vectors at right angles to each
     other -- R times its transpose is the identity -- with the determinant
     then telling a rotation from a reflection. */
    const double *r = sampler->transform.rotation;

    for (int i = 0; i < 9; i++) {
        if (!isfinite(r[i])) {
            omcFail("ompMC:phspSource:notARotation",
                "Element %d of the phase space to phantom rotation is not a "
                "finite number.", i);
        }
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double dot = r[3*i]*r[3*j] + r[3*i + 1]*r[3*j + 1]
                       + r[3*i + 2]*r[3*j + 2];
            double want = (i == j) ? 1.0 : 0.0;

            if (fabs(dot - want) > 1.0e-6) {
                omcFail("ompMC:phspSource:notARotation",
                    "Rows %d and %d of the phase space to phantom rotation "
                    "have dot product %g, and a rotation's rows are unit "
                    "vectors at right angles, so it should be %g. A matrix "
                    "that is not a rotation would stretch the directions it "
                    "turns, and they have to stay unit vectors.",
                    i, j, dot, want);
            }
        }
    }

    double det = r[0]*(r[4]*r[8] - r[5]*r[7])
               - r[1]*(r[3]*r[8] - r[5]*r[6])
               + r[2]*(r[3]*r[7] - r[4]*r[6]);

    if (fabs(det - 1.0) > 1.0e-6) {
        omcFail("ompMC:phspSource:notARotation",
            "The phase space to phantom rotation has determinant %g, and a "
            "rotation has 1. A determinant of -1 with orthonormal rows is a "
            "reflection, which turns a right handed coordinate system into a "
            "left handed one.", det);
    }

    if (sampler->phsp->newHistories == 0) {
        omcLog(OMC_LOG_WARNING, "The phase space marks no histories, so every "
               "particle is drawn as a history of its own. The dose is right; "
               "the uncertainty this run reports for it will be smaller than "
               "the truth by however much the particles of one original "
               "history are correlated.");
    }

    omcLog(OMC_LOG_INFO, "Phase space source: %llu particles, %s.",
           omcPhspCount(sampler->phsp),
           sampler->order == OMC_PHSP_REPLAY ? "replayed in order" :
           "drawn at random");

    return;
}

/* Turn and move one point of the phase space into the phantom's world. */
static void applyTransform(const struct OmcPhspTransform *transform,
                           double *x, double *y, double *z, int isDirection) {

    const double *r = transform->rotation;
    double px = *x, py = *y, pz = *z;

    *x = r[0]*px + r[1]*py + r[2]*pz;
    *y = r[3]*px + r[4]*py + r[5]*pz;
    *z = r[6]*px + r[7]*py + r[8]*pz;

    /* A direction is turned but not moved. */
    if (!isDirection) {
        *x += transform->translation[0];
        *y += transform->translation[1];
        *z += transform->translation[2];
    }

    return;
}

/* Which particle of the file this history gets.

 @warning Depends on ihist and nothing else, which is what keeps a run from
 depending on how its histories were scheduled. Note what it does NOT do:
 read the next particle. That read position belongs to whoever is stepping
 through the file serially, and there is one of it for all the threads. */
static unsigned long long recordFor(const struct OmcPhspSampler *sampler,
                                    uint64_t ihist) {

    unsigned long long count = omcPhspCount(sampler->phsp);

    if (sampler->order == OMC_PHSP_RANDOM) {
        /* setRandom() is in (0,1), so this is in range; the clamp is for the
         rounding at the very top of it rather than for the mathematics. */
        unsigned long long index = (unsigned long long)(setRandom()*count);
        return index < count ? index : count - 1;
    }

    return (sampler->first + (unsigned long long)ihist) % count;
}

int omcPhspProduce(const struct OmcPhspSampler *sampler, uint64_t ihist,
                   double weight, struct OmcSourceParticle *particle) {

    struct OmcPhspRecord record;
    int charge;

    /* omcPhspSourceCheck() turns an empty phase space away before any of
     this runs, so this is only a backstop for a caller who skipped it: the
     wrap in recordFor() would divide by zero, and omcFail() from a worker
     thread would call the host from a place the host cannot expect. */
    if (omcPhspCount(sampler->phsp) == 0) {
        return 0;
    }

    omcPhspGet(sampler->phsp, recordFor(sampler, ihist), &record);

    switch (record.type) {
        case OMC_PHSP_PHOTON:
            charge = 0;
            break;
        case OMC_PHSP_ELECTRON:
            charge = -1;
            break;
        case OMC_PHSP_POSITRON:
            charge = 1;
            break;
        default:
            /* A neutron or a proton. ompMC transports neither, and a phase
             space holding them is not wrong for it -- this history simply
             has nothing in it. */
            return 0;
    }

    double x = record.x, y = record.y, z = record.z;
    double u = record.u, v = record.v, w = record.w;

    applyTransform(&sampler->transform, &x, &y, &z, 0);
    applyTransform(&sampler->transform, &u, &v, &w, 1);

    particle->charge = charge;
    particle->energy = record.energy;

    particle->x = x;
    particle->y = y;
    particle->z = z;

    particle->u = u;
    particle->v = v;
    particle->w = w;

    particle->weight = record.weight*weight;

    return 1;
}

/******************************************************************************/
/* The source interface */

static void phspCheck(const struct OmcSource *self) {

    omcPhspSourceCheck((const struct OmcPhspSampler *)self->impl);

    return;
}

static void phspPrepare(struct OmcSource *self, int nperbatch) {

    const struct OmcPhspSampler *sampler =
        (const struct OmcPhspSampler *)self->impl;

    /* The particles carry the weights the file gave them and nothing
     rescales a batch, so what comes out is the dose per history once the
     accumulated energy is divided by how many there were. */
    self->batchScale = 1.0;
    self->incidentFluence = (double)nperbatch;

    if ((unsigned long long)nperbatch > omcPhspCount(sampler->phsp)) {
        omcLog(OMC_LOG_WARNING, "A batch of %d histories is more than the "
               "%llu particles the phase space holds, so some are used more "
               "than once per batch. That buys less than the history count "
               "suggests: a particle used twice tells you no more the second "
               "time about what the beam does, only about what this phantom "
               "does with it.", nperbatch, omcPhspCount(sampler->phsp));
    }

    return;
}

static int phspSample(const struct OmcSource *self, uint64_t ihist,
                      int ihistInBatch, struct OmcSourceParticle *particle) {

    (void)ihistInBatch;

    return omcPhspProduce((const struct OmcPhspSampler *)self->impl, ihist,
                          1.0, particle);
}

void omcPhspSamplerAsSource(struct OmcPhspSampler *sampler,
                            struct OmcSource *source) {

    source->check = phspCheck;
    source->prepare = phspPrepare;
    source->sample = phspSample;
    source->release = NULL;     /* nothing was taken */
    source->impl = sampler;
    source->batchScale = 1.0;
    source->incidentFluence = 1.0;

    return;
}

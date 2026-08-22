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

#include "omc_source_pencil.h"

#include "omc_geom.h"
#include "omc_host.h"
#include "omc_random.h"
#include "omc_spectrum.h"

#include <math.h>
#include <stddef.h>

/*! How far upstream of the front face a parallel pencil starts.

 Anywhere upstream would do: the particle flies through region 0, which holds
 no medium, so the distance costs nothing and changes nothing. What matters is
 that it is upstream at all rather than on the face -- see the warning in
 omc_source.h about a particle that starts where it should have arrived. */
#define PENCIL_STANDOFF 1.0

/******************************************************************************/

static double fieldRadiusOf(const struct OmcPencilSource *pencil) {

    /* Read from the geometry rather than resolved once at prepare() time
     because sample() sees a read-only source; the geometry is read-only for
     the whole run, which is what makes reading it here safe from a worker
     thread. */
    return pencil->fieldRadius > 0.0 ? pencil->fieldRadius
                                     : geometry.rbounds[geometry.isize];
}

/******************************************************************************/
/* The source interface */

static void pencilCheck(const struct OmcSource *self) {

    const struct OmcPencilSource *pencil =
        (const struct OmcPencilSource *)self->impl;

    if (pencil->kind != OMC_PENCIL_PARALLEL &&
        pencil->kind != OMC_PENCIL_SSD) {
        omcFail("ompMC:pencil:badKind",
            "The beam is of kind %d, which is neither a parallel pencil (%d) "
            "nor a point source (%d).", (int)pencil->kind,
            (int)OMC_PENCIL_PARALLEL, (int)OMC_PENCIL_SSD);
    }

    if (pencil->spectrum == NULL) {
        omcFail("ompMC:pencil:noSpectrum",
            "The beam has no spectrum to draw energies from.");
    }

    if (pencil->charge < -1 || pencil->charge > 1) {
        omcFail("ompMC:pencil:badCharge",
            "The beam has charge %d; ompMC transports photons (0), electrons "
            "(-1) and positrons (+1).", pencil->charge);
    }

    /* It aims down an axis, and only the cylinder has one. */
    if (geometry.mode != OMC_GEOM_CYLINDRICAL) {
        omcFail("ompMC:pencil:notCylindrical",
            "A pencil beam aims down the axis of a cylinder, and the phantom "
            "is a rectilinear voxel grid. Set the geometry up with "
            "omcGeomCylInit().");
    }

    if (pencil->kind == OMC_PENCIL_SSD) {
        if (!(pencil->ssd > 0.0)) {
            omcFail("ompMC:pencil:badSsd",
                "The point source sits %g cm from the front face; the "
                "distance has to be positive.", pencil->ssd);
        }

        if (pencil->fieldRadius < 0.0) {
            omcFail("ompMC:pencil:badFieldRadius",
                "The field radius is %g cm. Give a positive one, or 0 for the "
                "whole front face.", pencil->fieldRadius);
        }
    }

    return;
}

static void pencilPrepare(struct OmcSource *self, int nperbatch) {

    /* Every particle carries weight 1 and nothing rescales a batch, so
     dividing the accumulated energy by the number of histories is what makes
     the result the dose one incident particle delivers. */
    self->batchScale = 1.0;
    self->incidentFluence = (double)nperbatch;

    return;
}

static int pencilSample(const struct OmcSource *self, uint64_t ihist,
                        int ihistInBatch, struct OmcSourceParticle *particle) {

    (void)ihist;
    (void)ihistInBatch;

    const struct OmcPencilSource *pencil =
        (const struct OmcPencilSource *)self->impl;

    /* The energy first, always, so that a run with a spectrum and one without
     differ by the spectrum's own draws and by nothing else. */
    particle->energy = omcSpectrumSample(pencil->spectrum);
    particle->charge = pencil->charge;
    particle->weight = 1.0;

    double zface = geometry.zbounds[0];

    if (pencil->kind == OMC_PENCIL_PARALLEL) {
        /* No width and no divergence, and no random numbers to give it any */
        particle->x = 0.0;
        particle->y = 0.0;
        particle->z = zface - PENCIL_STANDOFF;

        particle->u = 0.0;
        particle->v = 0.0;
        particle->w = 1.0;

        return 1;
    }

    /* A point on the axis, aimed at a disc on the front face. Exactly two
     draws, in this order, whatever the field is: the count is part of the
     source's contract, since the random stream is indexed per history. */
    double rnno1 = setRandom();
    double rnno2 = setRandom();

    /* sqrt() rather than the random number itself: the area of the disc grows
     as r^2, so a uniform radius would crowd the particles onto the axis. */
    double radius = fieldRadiusOf(pencil)*sqrt(rnno1);
    double phi = 2.0*M_PI*rnno2;

    particle->x = 0.0;
    particle->y = 0.0;
    particle->z = zface - pencil->ssd;

    /* From the source point to where it meets the face, normalized. The
     transport takes the direction as a unit vector and never renormalizes
     it. */
    double dx = radius*cos(phi);
    double dy = radius*sin(phi);
    double dz = pencil->ssd;
    double norm = 1.0/sqrt(dx*dx + dy*dy + dz*dz);

    particle->u = dx*norm;
    particle->v = dy*norm;
    particle->w = dz*norm;

    return 1;
}

/******************************************************************************/

void omcPencilSourceAsSource(struct OmcPencilSource *pencil,
                             struct OmcSource *source) {

    source->check = pencilCheck;
    source->prepare = pencilPrepare;
    source->sample = pencilSample;
    source->release = NULL;     /* nothing was taken */
    source->impl = pencil;
    source->batchScale = 1.0;
    source->incidentFluence = 1.0;

    return;
}

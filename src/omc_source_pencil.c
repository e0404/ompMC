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

/*! Turn a direction by the projected angles @p a and @p b, about two
 perpendicular directions of its own.

 The turned direction is `d + a*e1 + b*e2` brought back to unit length, so a
 and b are the tangents of the angle onto the two planes through d -- the
 angles themselves, to the accuracy a pencil beam's divergence cares about.
 Written this way rather than as a spherical rotation because it cannot
 degenerate: it stays a unit vector for any a and b, with no pole to avoid. */
static void turnDirection(double *u, double *v, double *w,
                          double a, double b) {

    double du = *u, dv = *v, dw = *w;

    /* Any unit vector perpendicular to d. Crossing d with whichever axis it
     leans on least keeps the cross product well away from zero. */
    double tx = 0.0, ty = 0.0, tz = 0.0;
    double au = fabs(du), av = fabs(dv), aw = fabs(dw);

    if (au <= av && au <= aw) {
        tx = 1.0;
    }
    else if (av <= aw) {
        ty = 1.0;
    }
    else {
        tz = 1.0;
    }

    double e1u = dv*tz - dw*ty;
    double e1v = dw*tx - du*tz;
    double e1w = du*ty - dv*tx;

    double norm = 1.0/sqrt(e1u*e1u + e1v*e1v + e1w*e1w);
    e1u *= norm;
    e1v *= norm;
    e1w *= norm;

    /* e2 = d x e1, already unit since both are and they are perpendicular */
    double e2u = dv*e1w - dw*e1v;
    double e2v = dw*e1u - du*e1w;
    double e2w = du*e1v - dv*e1u;

    double nu = du + a*e1u + b*e2u;
    double nv = dv + a*e1v + b*e2v;
    double nw = dw + a*e1w + b*e2w;

    norm = 1.0/sqrt(nu*nu + nv*nv + nw*nw);

    *u = nu*norm;
    *v = nv*norm;
    *w = nw*norm;
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

    /* Zero is not refused: it is the delta each of these widens from. */
    if (pencil->spotSigma < 0.0) {
        omcFail("ompMC:pencil:badSpotSigma",
            "The beam has a spot width of %g cm. Give a positive one, or 0 "
            "for a beam of no width.", pencil->spotSigma);
    }

    if (pencil->divergenceSigma < 0.0) {
        omcFail("ompMC:pencil:badDivergenceSigma",
            "The beam has a divergence of %g rad. Give a positive one, or 0 "
            "for a beam that does not diverge.", pencil->divergenceSigma);
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
    int parallel = pencil->kind == OMC_PENCIL_PARALLEL;

    /* How far upstream of the front face the particle is emitted. For the
     point source that is the SSD, a real distance; for a parallel pencil it
     is arbitrary -- far enough that the particle arrives rather than starting
     inside, see omc_source.h -- which is why the two are treated differently
     below. */
    double standoff = parallel ? PENCIL_STANDOFF : pencil->ssd;

    /* Where the nominal beam meets the front face. */
    double aimX = 0.0;
    double aimY = 0.0;

    if (!parallel) {
        /* Aimed at a disc on the face. Exactly two draws, in this order,
         whatever the field is: the count is part of the source's contract,
         since the random stream is indexed per history. */
        double rnno1 = setRandom();
        double rnno2 = setRandom();

        /* sqrt() rather than the random number itself: the area of the disc
         grows as r^2, so a uniform radius would crowd the particles onto the
         axis. */
        double radius = fieldRadiusOf(pencil)*sqrt(rnno1);
        double phi = 2.0*M_PI*rnno2;

        aimX = radius*cos(phi);
        aimY = radius*sin(phi);
    }

    /* A finite width, spread round the axis. What it displaces differs
     between the two beams, because what the position MEANS differs: a
     parallel pencil is specified where it meets the phantom, a point source
     by where the source is. So this is the width of the beam on the front
     face in the first case, and the size of the focal spot in the second. */
    double spotX = 0.0;
    double spotY = 0.0;

    if (pencil->spotSigma > 0.0) {
        double offset[2];
        boxMuller(offset);

        spotX = pencil->spotSigma*offset[0];
        spotY = pencil->spotSigma*offset[1];
    }

    /* The nominal direction. The transport takes it as a unit vector and
     never renormalizes it. */
    if (parallel) {
        particle->u = 0.0;
        particle->v = 0.0;
        particle->w = 1.0;
    }
    else {
        /* From wherever on the focal spot this history starts, to the point
         on the disc it is aimed at. */
        double dx = aimX - spotX;
        double dy = aimY - spotY;
        double dz = pencil->ssd;
        double norm = 1.0/sqrt(dx*dx + dy*dy + dz*dz);

        particle->u = dx*norm;
        particle->v = dy*norm;
        particle->w = dz*norm;
    }

    /* A finite divergence, about whatever direction the beam already had. */
    if (pencil->divergenceSigma > 0.0) {
        double angle[2];
        boxMuller(angle);

        turnDirection(&particle->u, &particle->v, &particle->w,
                      pencil->divergenceSigma*angle[0],
                      pencil->divergenceSigma*angle[1]);
    }

    particle->z = zface - standoff;

    if (parallel) {
        /* Emitted from wherever it has to start to cross the face at the
         spot. Without this the particle would drift sideways over the
         standoff, and an arbitrary internal distance would quietly widen
         every diverging beam. */
        particle->x = spotX - standoff*(particle->u/particle->w);
        particle->y = spotY - standoff*(particle->v/particle->w);
    }
    else {
        /* The focal spot is a real place; the particle starts on it. */
        particle->x = spotX;
        particle->y = spotY;
    }

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

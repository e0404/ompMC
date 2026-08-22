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

/******************************************************************************/

static double fieldRadiusOf(const struct OmcPencilSource *pencil) {

    /* Read from the geometry rather than resolved once at prepare() time
     because sample() sees a read-only source; the geometry is read-only for
     the whole run, which is what makes reading it here safe from a worker
     thread. */
    return pencil->fieldRadius > 0.0 ? pencil->fieldRadius
                                     : geometry.rbounds[geometry.isize];
}

/*! Tilt a direction by the projected angles @p ax and @p ay, measured in the
 phantom's own transverse plane.

 The tilted direction is `d + (ax, ay, 0)` brought back to unit length, so for
 a beam travelling along +z the tangents of the angle onto the xz and yz
 planes are exactly ax and ay -- the angles themselves, to the accuracy a
 pencil beam's divergence cares about, and to first order for a beam merely
 near +z.

 The lab frame and not a frame built perpendicular to d, which is the obvious
 alternative and is wrong here: the correlation between where a particle
 starts and where it is going is stated per transverse AXIS, so the angle has
 to be perturbed along the same axes the position was. A basis constructed
 from d alone comes out rotated -- for d = +z it is (y, -x) -- which pairs the
 x position with the y angle and quietly moves the correlation into the cross
 term, where it is neither what was asked for nor visible in a round beam. */
static void tiltDirection(double *u, double *v, double *w,
                          double ax, double ay) {

    double nu = *u + ax;
    double nv = *v + ay;
    double nw = *w;

    double norm = 1.0/sqrt(nu*nu + nv*nv + nw*nw);

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

    if (!(pencil->correlation >= -1.0 && pencil->correlation <= 1.0)) {
        omcFail("ompMC:pencil:badCorrelation",
            "The beam has a position to angle correlation of %g. It is a "
            "correlation coefficient, so it lies between -1 and 1.",
            pencil->correlation);
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
     face in the first case, and the size of the focal spot in the second.

     The deviates are kept, because the divergence below may be correlated
     with them. */
    double spotX = 0.0;
    double spotY = 0.0;
    double offset[2] = {0.0, 0.0};
    int hasSpot = pencil->spotSigma > 0.0;

    if (hasSpot) {
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
        double independent[2];
        boxMuller(independent);

        double ax = independent[0];
        double ay = independent[1];

        /* Correlated with where the particle started, if it was asked for and
         there is a width for it to relate to. The Cholesky factor of the two
         by two covariance: one part of the angle is the position's own
         deviate, the rest is fresh. Same correlation in both planes, which is
         what keeps the beam round.

         Written so that a correlation of 0 leaves ax and ay exactly the
         deviates they already were, rather than the same numbers arrived at
         through an arithmetically equivalent detour: an uncorrelated beam
         gives bit for bit what it gave before this existed. */
        if (hasSpot && pencil->correlation != 0.0) {
            double rho = pencil->correlation;
            double rest = sqrt(1.0 - rho*rho);

            ax = rho*offset[0] + rest*independent[0];
            ay = rho*offset[1] + rest*independent[1];
        }

        tiltDirection(&particle->u, &particle->v, &particle->w,
                      pencil->divergenceSigma*ax,
                      pencil->divergenceSigma*ay);
    }

    particle->x = spotX;
    particle->y = spotY;

    /* A parallel pencil is DEFINED on the front face, so that is where its
     particles start -- there is no upstream position for a beam that has no
     source point. omc_source.h warns against handing omcSourcePlace() a
     particle already sitting on the surface unless that is genuinely where
     the source put it, and for a beam specified on that plane it is: what
     the warning is really about is starting inside the phantom, at depth,
     which skips the build up a particle should have travelled through.
     Entering exactly on the face is what every other source ends up doing
     too, once omcSourcePlace() has carried it there.

     The point source does have somewhere to be, an SSD upstream, and starts
     on its focal spot. */
    particle->z = parallel ? zface : zface - pencil->ssd;

    return 1;
}

/******************************************************************************/

void omcPencilWaist(double waistSigma, double divergenceSigma,
                    double waistDepth,
                    double *spotSigma, double *correlation) {

    /* var(s) is smallest at s = -rho sigma / sigma', where it equals
     sigma^2 (1 - rho^2). Solving those two for sigma and rho gives the pair
     below -- and |rho| < 1 falls out for any positive waist, so there is no
     combination of a real waist and a real divergence this cannot express. */
    double drift = waistDepth*divergenceSigma;
    double sigma = sqrt(waistSigma*waistSigma + drift*drift);

    if (spotSigma != NULL) {
        *spotSigma = sigma;
    }
    if (correlation != NULL) {
        *correlation = sigma > 0.0 ? -drift/sigma : 0.0;
    }

    return;
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

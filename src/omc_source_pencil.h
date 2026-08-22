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

/*!
 @file
 omc_source_pencil - A beam down the axis of a cylinder.

 Two of them, because the two questions asked of an r-z geometry are
 different. A dose kernel is defined for a beam of no width at all, every
 particle entering at r = 0 travelling along +z; that is
 #OMC_PENCIL_PARALLEL. A depth dose measurement is made under a machine at
 some distance, the beam diverging onto a field of finite size; that is
 #OMC_PENCIL_SSD.

     struct OmcPencilSource pencil = {0};
     pencil.kind = OMC_PENCIL_PARALLEL;
     pencil.spectrum = &spectrum;
     pencil.charge = 0;

     struct OmcSource source;
     omcPencilSourceAsSource(&pencil, &source);

     omcCalcRadial(&options, &source, NULL, dose, uncertainty, NULL, &summary);

 Both report per incident history: what comes out of omcCalcRadial() is the
 dose one particle of the beam delivers, not the dose per unit fluence that
 omc_dosxyz's collimated source reports. There is no field for a pencil beam
 to have a fluence over.

 The kind above picks the NOMINAL beam. Either of the two deltas a real beam
 does not have can then be widened into a Gaussian, independently:

     pencil.spotSigma = 0.15;         // cm, a beam of finite width
     pencil.divergenceSigma = 0.01;   // rad, a beam that is not quite parallel

 Both are measured on the front face of the phantom, which is the plane a
 pencil beam is specified on, and both default to 0 -- the delta they widen
 from. A zero draws no random numbers at all, so a beam that asks for neither
 gives exactly the result it gave before they existed.

 They are drawn independently of each other, which makes this a blurred pencil
 rather than a beam with emittance: where a particle starts says nothing about
 where it is going. A beam whose width and divergence are correlated -- a
 waist somewhere other than the phantom surface -- is not what this models.

 @warning The SSD source spreads its particles evenly over the disc it
 illuminates -- uniform FLUENCE on the entrance plane, which is the convention
 omc_dosxyz's rectangular source follows too. That is not the same thing as an
 isotropic point source, whose fluence would fall off with the inverse square
 across the field, and the difference shows at short SSD.
*****************************************************************************/

#ifndef OMC_SOURCE_PENCIL_H
#define OMC_SOURCE_PENCIL_H

#include "omc_source.h"

struct OmcSpectrum;

/*! Which of the two beams. */
enum OmcPencilKind {
    /*! A beam of no width, every particle entering on the axis travelling
     along +z. Draws no random numbers of its own at all. */
    OMC_PENCIL_PARALLEL = 0,

    /*! A point source on the axis, an SSD upstream of the front face,
     illuminating a disc on it. Costs two random numbers per history. */
    OMC_PENCIL_SSD = 1
};

/*! A beam down the axis. */
struct OmcPencilSource {
    enum OmcPencilKind kind;            ///< which of the two beams

    /*! Where the energies come from. Monoenergetic or a histogram, as
     omc_spectrum.h makes it; must outlive the source. */
    const struct OmcSpectrum *spectrum;

    int charge;                         ///< 0 : photon, -1 : electron, +1 : positron

    /*! #OMC_PENCIL_SSD only: how far upstream of the front face the point
     source sits, in cm. Must be positive. */
    double ssd;

    /*! #OMC_PENCIL_SSD only: radius of the disc illuminated on the front
     face, in cm. 0 means the whole face, i.e. the cylinder's own radius. */
    double fieldRadius;

    /*! Standard deviation of the starting position, in cm, spread as a round
     two dimensional Gaussian across the beam. 0 is a beam of no width.

     For a parallel pencil this is the width where the beam meets the front
     face; for a point source it is the size of the focal spot. Costs one
     Box-Muller pair, i.e. two random numbers, and none at all when 0. */
    double spotSigma;

    /*! Standard deviation of the direction, in radians, spread as a round two
     dimensional Gaussian about the nominal one. 0 is a beam that does not
     diverge at all.

     It is the PROJECTED angles that are Gaussian -- the tangents of the angle
     onto two perpendicular planes through the beam -- which for the small
     divergences a pencil beam has is the angle itself. Costs one Box-Muller
     pair, i.e. two random numbers, and none at all when 0. */
    double divergenceSigma;
};

/*! Present the beam to an engine as a source.

 @param pencil The beam. Must outlive @p source.
 @param source Filled in with the source interface.

 @pre The geometry is a cylinder (omc_geom_cyl.h). The source aims down its
 axis, and struct OmcSource::check reports it if there is not one. */
void omcPencilSourceAsSource(struct OmcPencilSource *pencil,
                             struct OmcSource *source);

#endif

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
};

/*! Present the beam to an engine as a source.

 @param pencil The beam. Must outlive @p source.
 @param source Filled in with the source interface.

 @pre The geometry is a cylinder (omc_geom_cyl.h). The source aims down its
 axis, and struct OmcSource::check reports it if there is not one. */
void omcPencilSourceAsSource(struct OmcPencilSource *pencil,
                             struct OmcSource *source);

#endif

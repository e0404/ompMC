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
 omc_geom_cyl - A cylinder in concentric rings, stacked in depth.

 The shape a pencil beam wants. Dose around a narrow beam falls by orders of
 magnitude over the first few millimetres off the axis, and a rectilinear grid
 fine enough to follow that near the axis is far finer than it needs to be
 anywhere else. Rings answer the same question with a handful of bins, and --
 because the scoring regions ARE the transport regions -- each one gets its
 batch variance from the same machinery every other ompMC geometry uses,
 rather than from summing voxels afterwards and hoping the correlations
 between them come out in the wash.

 The geometry is a single solid cylinder about the z axis, so:

   - ring 0 reaches the axis, i.e. `rbounds[0]` is 0. There is no hollow
     middle, because region 0 already means "outside the phantom" and a hole
     would need a second thing for it to mean.
   - there is no azimuthal binning. A ring is a ring, all the way round.
   - the axis is the beam's axis by convention, and depth runs along +z.

 Regions are numbered `1 + ir + iz*nr`, region 0 being outside, which is the
 rectilinear numbering with the y index held at zero. That is not a
 coincidence and it is not cosmetic: it is what lets initRegions(),
 struct Score, ausgab(), the region memo in omc_utilities.h and everything
 else that indexes by region serve a cylinder without knowing there is one.
 See struct Geom.

 A host fills in the bounds and calls omcGeomCylInit():

     geometry.isize = nr;                // rings
     geometry.ksize = nz;                // depth slabs
     geometry.rbounds = rbounds;         // nr+1 values, rbounds[0] == 0
     geometry.zbounds = zbounds;         // nz+1 values
     geometry.med_indices = ...;         // nr*nz entries, as ever
     geometry.med_densities = ...;

     omcGeomCylInit();
     initRegions();

 after which howfar(), hownear() and regionIndex() answer for the cylinder,
 and omcSourcePlace() carries source particles into it.
*****************************************************************************/

#ifndef OMC_GEOM_CYL_H
#define OMC_GEOM_CYL_H

/*! Declare the phantom cylindrical: check the bounds over, pin the unused
 azimuthal index, and work out the reciprocal spacings the fast point location
 uses.

 The caller sets struct Geom::isize (rings), struct Geom::ksize (depth slabs),
 struct Geom::rbounds and struct Geom::zbounds first. struct Geom::jsize is
 set here rather than asked of the caller -- there is no azimuthal binning to
 have an opinion about, and the rectilinear grid it is borrowed from is an
 implementation detail a host should not have to know.

 Reports through omcFail(): `ompMC:geomCyl:badGrid` for a grid with no rings
 or no slabs, `ompMC:geomCyl:badRadialBounds` for bounds that are missing, do
 not start at the axis or do not ascend, and
 `ompMC:geomCyl:badDepthBounds` likewise for the depth.

 @pre struct Geom::isize, ::ksize, ::rbounds and ::zbounds are filled in. */
void omcGeomCylInit(void);

/*! The cylindrical howfar(), which the dispatcher in omc_geom.c calls when
 struct Geom::mode says so. Same contract as howfar(): truncate @p ustep at
 the first boundary the particle meets, and name the region beyond it.

 @param idisc Set to 1 if the particle is already outside the geometry.
 @param irnew The region on the far side of whatever truncated the step; 0 if
 that is outside the cylinder. Only meaningful if @p ustep was truncated.
 @param ustep The proposed step, truncated in place. */
void omcCylHowfar(int *idisc, int *irnew, double *ustep);

/*! The cylindrical hownear(): the perpendicular distance from the particle to
 the nearest boundary of the ring and slab it is in, which is the lower bound
 the condensed history stepping needs.

 @return That distance, or 0 for a particle outside the geometry. */
double omcCylHownear(void);

/*! The cylindrical regionIndex(): the region a point falls in.

 @param x Position, in cm.
 @param y Position, in cm.
 @param z Position, in cm.
 @return `1 + ir + iz*nr`, or 0 outside the cylinder. Points exactly on the
 outer boundaries count as inside, as they do in the rectilinear geometry. */
int omcCylRegionIndex(double x, double y, double z);

/*! How far along a ray the cylinder begins, for omcSourcePlace().

 @param x Where the ray starts, in cm.
 @param y Where the ray starts, in cm.
 @param z Where the ray starts, in cm.
 @param u Direction, a unit vector.
 @param v Direction, a unit vector.
 @param w Direction, a unit vector.
 @param tenter Set to the distance at which the ray enters the cylinder, 0 for
 a ray that starts inside it.
 @return 1 if the ray is inside the cylinder over an interval of nonzero
 length, 0 if it misses, only touches it, or is heading away from it. */
int omcCylClipRay(double x, double y, double z, double u, double v, double w,
                  double *tenter);

#endif

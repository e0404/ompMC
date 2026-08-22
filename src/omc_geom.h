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
 omc_geom - The phantom every ompMC user code transports in.

 howfar(), hownear() and regionIndex() are the geometry side of the contract
 ompmc.h declares: the transport code calls them, but does not define them.
 They used to be copied verbatim into each user code, which meant two
 identical implementations that only a diff could confirm were still in step.
 They live here now, so the transport, the dose engines and any future host
 all see one phantom.

 There are two shapes of phantom. The rectilinear voxel grid is the original
 and still the default; omc_geom_cyl.h adds concentric rings stacked in depth,
 for the r-z dose distributions a pencil beam wants. Which one is in play is
 struct Geom::mode, and the three functions above dispatch on it -- rather
 than one or the other being linked in -- because a single ompmc_core has to
 serve a host that does both, sometimes in the same process.

 What a user code still owns is FILLING the geometry: omc_dosxyz reads an
 .egsphant file, the matRad interface takes cubes from MATLAB, and the Python
 host takes numpy arrays. Whoever fills it must set every field of struct
 Geom below, including the reciprocal spacings, before calling initRegions().
*****************************************************************************/

#ifndef OMC_GEOM_H
#define OMC_GEOM_H

/*! Which shape struct Geom is holding. */
enum OmcGeomMode {
    /*! The rectilinear voxel grid. Zero on purpose: struct Geom is a
     zero-initialized global, so a host that has never heard of any other
     shape gets this one without doing anything. */
    OMC_GEOM_CARTESIAN = 0,

    /*! Concentric rings stacked in depth; see omc_geom_cyl.h. */
    OMC_GEOM_CYLINDRICAL = 1
};

/*! The phantom. A host fills every field, including the reciprocal spacings
 (see omcGeomDetectSpacing(), or omcGeomCylInit() for a cylinder), before
 calling initRegions().

 The cylindrical geometry reuses this struct rather than bringing its own,
 which is what lets initRegions(), the scoring arrays and the region memo in
 omc_utilities.h serve both without knowing which is in play. It carries
 #isize rings and #ksize slabs with #jsize pinned to 1, so that the region
 numbering `1 + ir + iz*isize` is the rectilinear `1 + ix + iy*isize +
 iz*isize*jsize` with iy = 0 -- the same arithmetic, the same region 0 for
 outside, and no second copy of any of it. */
struct Geom {
    int *med_indices;           ///< index of the medium in each voxel
    double *med_densities;      ///< density of the medium in each voxel

    int isize;                  ///< voxels along x; rings, when cylindrical
    int jsize;                  ///< voxels along y; 1, when cylindrical
    int ksize;                  ///< voxels along z; depth slabs, when cylindrical

    double *xbounds;            ///< boundaries of voxels along x, isize+1 values
    double *ybounds;            ///< boundaries of voxels along y, jsize+1 values
    double *zbounds;            ///< boundaries of voxels along z, ksize+1 values

    /*! Reciprocal grid spacing along x when that axis is uniform, 0.0
     otherwise; lets regionIndex() locate a point with one multiplication
     instead of a binary search. Filled by omcGeomDetectSpacing(). */
    double dxi;
    double dyi;                 ///< reciprocal grid spacing along y, see #dxi
    double dzi;                 ///< reciprocal grid spacing along z, see #dxi

    /*! One of enum OmcGeomMode. Set by omcGeomDetectSpacing() (rectilinear)
     or omcGeomCylInit() (cylindrical), not by hand. */
    int mode;

    /*! Cylindrical only: the ring boundaries, isize+1 ascending values
     starting at 0. #xbounds and #ybounds are unused in that mode and may be
     NULL. */
    double *rbounds;

    double dri;                 ///< reciprocal ring spacing, see #dxi
};

/*! The phantom every user code fills in and passes to initRegions(). */
extern struct Geom geometry;

/*! Fill dxi/dyi/dzi from the bounds already stored in the struct, and declare
 the phantom rectilinear. Every loader of a voxel grid has to do this, and
 doing it in one place keeps a new one from forgetting and quietly losing the
 fast point location.

 Setting the mode here rather than leaving it to each host is what keeps a
 resident host -- a MEX file, a Python module -- from carrying a cylinder over
 into the next run: every loader of a rectilinear phantom already calls this,
 so none of them has to remember. */
void omcGeomDetectSpacing(void);

/*! Set up the per region transport parameters from the filled geometry:
 medium index and density scaling per voxel, per medium cut-offs clamped to
 what the PEGS data supports, and the per medium maximum density ratio the
 Woodcock majorant needs. Reads the "global ecut" and "global pcut" input
 items.

 @pre The global #geometry is completely filled in, including a call to
 omcGeomDetectSpacing(). */
void initRegions(void);

#endif

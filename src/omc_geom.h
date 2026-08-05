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
 omc_geom - The rectilinear voxel phantom every ompMC user code transports in.

 howfar(), hownear() and regionIndex() are the geometry side of the contract
 ompmc.h declares: the transport code calls them, but does not define them.
 They used to be copied verbatim into each user code, which meant two
 identical implementations that only a diff could confirm were still in step.
 They live here now, so the transport, the dose engines and any future host
 all see one phantom.

 What a user code still owns is FILLING the geometry: omc_dosxyz reads an
 .egsphant file, the matRad interface takes cubes from MATLAB, and the Python
 host takes numpy arrays. Whoever fills it must set every field of struct
 Geom below, including the reciprocal spacings, before calling initRegions().
*****************************************************************************/

#ifndef OMC_GEOM_H
#define OMC_GEOM_H

/*! The rectilinear voxel phantom. A host fills every field, including the
 reciprocal spacings (see omcGeomDetectSpacing()), before calling
 initRegions(). */
struct Geom {
    int *med_indices;           ///< index of the medium in each voxel
    double *med_densities;      ///< density of the medium in each voxel

    int isize;                  ///< number of voxels along x
    int jsize;                  ///< number of voxels along y
    int ksize;                  ///< number of voxels along z

    double *xbounds;            ///< boundaries of voxels along x, isize+1 values
    double *ybounds;            ///< boundaries of voxels along y, jsize+1 values
    double *zbounds;            ///< boundaries of voxels along z, ksize+1 values

    /*! Reciprocal grid spacing along x when that axis is uniform, 0.0
     otherwise; lets regionIndex() locate a point with one multiplication
     instead of a binary search. Filled by omcGeomDetectSpacing(). */
    double dxi;
    double dyi;                 ///< reciprocal grid spacing along y, see #dxi
    double dzi;                 ///< reciprocal grid spacing along z, see #dxi
};

/*! The phantom every user code fills in and passes to initRegions(). */
extern struct Geom geometry;

/*! Fill dxi/dyi/dzi from the bounds already stored in the struct. Every
 loader has to do this, and doing it in one place keeps a new one from
 forgetting and quietly losing the fast point location. */
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

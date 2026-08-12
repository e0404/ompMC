function [dij, dijVar, summary] = omc_matrad(cubeRho, cubeMatIx, mcGeo, mcSrc, mcOpt)
%OMC_MATRAD Monte Carlo dose calculation for matRad.
%
%   [dij, dijVar, summary] = OMC_MATRAD(cubeRho, cubeMatIx, mcGeo, mcSrc, mcOpt)
%
%   This is a compiled MEX file (.mexw64/.mexa64/.mexmaca64/...); this .m
%   file exists only to document it -- MATLAB reads help text from a
%   plain-text .m file of the same name next to a MEX file, but always
%   executes the MEX file itself, never this one. See BUILDING.md for how
%   it is built.
%
%   Returns either a sparse dose-influence matrix (mcOpt.mode = 'dij',
%   the default) or a dense dose cube -- of one weighted field
%   (mcOpt.mode = 'forward_beamlet') or of the particles of a phase space
%   file (mcOpt.mode = 'forward_phsp'); see "Mode" below.
%
%   version = OMC_MATRAD('version') returns the ompMC version string
%   without touching any dose calculation state; OMC_MATRAD('-v') and
%   OMC_MATRAD('--version') with no output argument print it instead.
%
%   The MEX file locks itself in memory on first use (mexLock) because an
%   OpenMP-using MEX file cannot safely be unloaded once a parallel
%   region has run. A rebuilt MEX file is therefore only picked up after
%   restarting MATLAB; see BUILDING.md.
%
%   Inputs:
%
%   * cubeRho -- 3D double cube of mass densities, g/cm^3.
%   * cubeMatIx -- 3D int32 cube of material indices into mcGeo.material, same size as cubeRho; 0 is not a valid index here (unlike the Python interface, there is no vacuum sentinel).
%   * mcGeo -- dose grid, a struct with fields material (cell array of medium names, matching entries in the PEGS file), xBounds, yBounds, zBounds (voxel boundaries along each axis, cm, ascending, one more value than voxels along that axis).
%   * mcSrc -- source, a struct. The beamlet fields are required except in mode 'forward_phsp', which brings its own particles: nBixels (number of beamlets), iBeam (index of the beam of each beamlet, counting from 1), xSource/ySource/zSource (source position of each beam), xCorner/yCorner/zCorner (corner of each beamlet's aperture rectangle), xSide1/ySide1/zSide1 and xSide2/ySide2/zSide2 (the two edge vectors of that rectangle). Also bixelWeights (one non-negative weight per beamlet), required by mode 'forward_beamlet'; phaseSpace, required by mode 'forward_phsp', see "Phase space" below; and collimator, optional in either forward mode, see "Collimator" below.
%   * mcOpt -- run settings, see "Options" below.
%
%   Outputs, mode 'dij' (default):
%
%   * dij -- sparse double matrix, one column per beamlet, one row per dose-grid voxel, dose in Gy per incident particle. Entries below mcOpt.relDoseThreshold (relative to the column maximum) are dropped.
%   * dijVar -- sparse matrix of the same shape and sparsity pattern: variance of the mean, dose units squared. Only computed if requested (nargout >= 2).
%
%   Mode 'dij' has no third output: a beamlet that started nothing comes
%   back as a column of zeros, which says so already. Asking for one is an
%   error rather than an empty.
%
%   Outputs, modes 'forward_beamlet' and 'forward_phsp':
%
%   * dij -- dense double cube, the size of cubeRho: the dose (or mean deposited energy, see mcOpt.outputDose) of the whole weighted field in mcSrc.bixelWeights, or of the phase space, in Gy.
%   * dijVar -- dense cube of the same size: the relative uncertainty of each voxel, 0.9999999 where nothing was deposited -- the convention omc_dosxyz writes into a .3ddose file. Only computed if requested (nargout >= 2).
%   * summary -- struct describing what became of the histories, with fields nHistories (histories actually run, nHistories rounded down to a whole number of batches), nStarted (how many put a particle into the phantom), nBlocked (how many the collimator stopped, before the particle was carried anywhere) and energyFraction (the fraction of the energy that entered the phantom which stayed in it). The histories in none of nStarted or nBlocked got past the collimator but missed the phantom, or offered a particle ompMC does not transport.
%
%   Example::
%
%     addpath('build/bin');
%     [dij, dijVar] = omc_matrad(cubeRho, cubeMatIx, mcGeo, mcSrc, mcOpt);
%
%   Options (fields of mcOpt). nSplit, charge, global_ecut, global_pcut,
%   randomSeeds, pegsFile, pgs4formFile, dataFolder and outputFolder are
%   required; the rest have defaults.
%
%   * nHistories -- histories per beamlet ('dij') or in total (either forward mode).
%   * nBatches -- statistical batches, at least 2.
%   * nSplit -- photon splitting factor at the source; 1 disables it.
%   * charge -- -1 electrons, 0 photons, +1 positrons.
%   * global_ecut -- global electron transport cut-off, MeV.
%   * global_pcut -- global photon transport cut-off, MeV.
%   * randomSeeds -- [seed1 seed2] for the Philox4x32-10 RNG.
%   * pegsFile -- path to the .pegs4dat file.
%   * pgs4formFile -- path to the pgs4form.dat file.
%   * dataFolder -- path to the cross section data directory.
%   * outputFolder -- path ompMC's own diagnostics may be written to.
%   * mode -- 'dij' (default), 'forward_beamlet' or 'forward_phsp', see "Mode" below.
%   * spectrum -- struct(energy, fluence, eMin, mode), overrides spectrumFile; see "Spectrum" below.
%   * spectrumFile -- path to a .spectrum file; default ./spectra/mohan6.spectrum.
%   * monoEnergy -- single kinetic energy in MeV, used if neither spectrum nor spectrumFile is given.
%   * sourceGeometry -- 'point' (default) or 'gaussian'.
%   * sourceGaussianWidth -- standard deviation in cm, 'gaussian' only.
%   * relDoseThreshold -- mode 'dij' only: voxels below this fraction of a beamlet's maximum dose are dropped from its column.
%   * outputDose -- 1 (default) for dose in Gy, 0 for mean deposited energy.
%   * verbose -- 0, 1 or 2; 2 also shows a waitbar unless progressCallback is given.
%   * progressCallback -- function handle called with a scalar in [0,1] once per batch and once per finished beamlet, replacing the built-in waitbar; see recordProgressCallback.m for an example.
%   * esave, e_rr, f_rr -- variance reduction, see "Variance reduction" below.
%
%   Mode: dose-influence matrix or forward dose. Both forward modes
%   compute one dense dose cube instead of one sparse column per
%   beamlet, and differ only in where the particles come from --
%   'forward_beamlet' from a spectrum through matRad's own beamlet
%   apertures, 'forward_phsp' from a phase space file, see "Phase space"
%   below.
%
%   'forward_beamlet'
%   computes the dose of a whole weighted field in one go instead of one
%   sparse column per beamlet. The collimation is given as
%   mcSrc.bixelWeights, a non-negative vector of length mcSrc.nBixels: a
%   blocked beamlet gets 0, an open one its fluence, a partly
%   transmitting one a fraction of it -- the fluence map matRad already
%   optimises, so no new geometry is needed. The result is what dij*w
%   would have been, reached directly instead of through the matrix.
%
%   Two things change meaning in either forward mode. First, nHistories
%   counts the whole calculation, not one beamlet: switching a 'dij' run
%   over unchanged divides the statistics by nBixels, so multiply
%   nHistories by nBixels to keep them. Second, relDoseThreshold does
%   nothing: it prunes columns of a sparse matrix, and there is no matrix
%   here -- it is the 'dij' result that is pruned, so set it to 0 for a
%   like-for-like comparison of the modes.
%
%   The weights modulate fluence, not spectrum: a leaf transmitting 2%
%   starts 2% of the particles, with the spectrum unhardened. Attenuation
%   in the collimator, its scatter and the beam hardening that goes with
%   it are not modelled.
%
%   Phase space. 'forward_phsp' starts each history from one particle of
%   an IAEA phase space file -- a record of everything that crossed a
%   plane in an earlier simulation of a treatment head, such as the sets
%   published at https://www-nds.iaea.org/phsp/. Using one starts
%   histories from the machine's own particles rather than from a
%   spectrum through an aperture, which is the difference between
%   modelling the beam and describing it. The mode reads no spectrum and
%   no beamlets: the file carries the energy, position and direction of
%   every particle it holds, and none of mcSrc's aperture fields are
%   asked for.
%
%   mcSrc.phaseSpace is a struct:
%
%   * file -- base name of the dataset pair (name.IAEAheader and name.IAEAphsp), with or without either extension. Required.
%   * order -- 'replay' (default) walks the file in order from first, wrapping at the end, and draws no random numbers; 'random' picks a particle per history, costing one random number, which is worth it when a run is much shorter than the file and a contiguous stretch of it would sample only one part of the beam.
%   * first -- particle the replay starts at, default 0. Ignored when order is 'random'.
%   * rotation -- 3x3 rotation carrying the phase space's coordinate system into the phantom's, applied to positions and directions alike. Default eye(3). Handed over in the usual MATLAB orientation: rotation*[x;y;z].
%   * translation -- 1x3 offsets in cm, added after rotation. Default zeros.
%
%   Two things about this mode are worth knowing. The whole file goes
%   into memory, so it costs about its size on disk -- gigabytes for a
%   published dataset. And it is recorded wherever the original
%   simulation scored it, not aimed at your phantom, so it is normal for
%   most histories to start nothing; the summary output is what tells
%   that apart from a transform that is wrong.
%
%   One particle per history. A phase space records which particles a
%   single original history left behind, and those are correlated.
%   Drawing them one at a time still gets the dose right on average, but
%   the uncertainty a run reports comes out smaller than the truth by
%   however much they are correlated. ompMC transports photons,
%   electrons and positrons; a neutron or proton in the file starts no
%   history.
%
%   Collimator. mcSrc.collimator, optional in either forward mode, puts
%   a transmission mask on a plane across the beam. It works by back
%   projection from wherever the source put the particle, so it composes
%   with either source and does not care which side of the plane the
%   particle started on -- which is what lets a field be cut out of a
%   phase space recorded above the jaws, as the published ones are. With
%   beamlets it is usually unnecessary, the collimation already being in
%   bixelWeights; it earns its place where the weights cannot say what is
%   wanted, such as a block cutting across beamlets or a leaf that
%   transmits.
%
%   * z -- the plane the mask sits on, in cm. Required.
%   * x0, y0 -- lower corner of the grid, in cm. Required.
%   * dx, dy -- cell size in cm, both positive. Required.
%   * transmission -- matrix of fractions in [0,1], one per cell, x down the rows and y across the columns. A 1x1 matrix of 1 is a rectangular aperture, which is most of the use. Required.
%   * outside -- what gets through beside the grid, in [0,1]. Default 0, which is what a field stop does.
%   * roulette -- how a partly transmitting cell is paid for. False (default) multiplies the particle's weight by the fraction and transports it regardless, so a 2% leaf costs a full shower for a fiftieth of the dose -- but draws no random numbers at all, leaving every history's random stream where it would have been with the beam open. True lets the particle through with that probability at full weight instead, spending the time where the dose is, at the price of one random number and more noise per history. Both give the same dose in the mean; cells that are fully open or fully shut are decided without drawing either way, so an all-or-nothing aperture behaves identically under both.
%
%   This is a mask, not a collimator: it attenuates and blocks, but does
%   not scatter and does not harden the spectrum of what it lets through.
%   Good for the fluence, poor for the penumbra -- the same
%   simplification the beamlet weights make.
%
%   Spectrum. The source spectrum is tried in this order: mcOpt.spectrum,
%   then mcOpt.spectrumFile, then mcOpt.monoEnergy; whichever loses is
%   announced rather than silently dropped. Giving none of them uses
%   spectra/mohan6.spectrum. mcOpt.spectrum is a struct with fields
%   energy (upper energy of each bin in MeV, strictly ascending), fluence
%   (relative number of particles per bin, same length, non-negative),
%   eMin (lower energy of the first bin in MeV, optional, default 0) and
%   mode (0 for counts per bin, the default, or 1 for counts per MeV).
%   Within a bin the energy is sampled uniformly, as it is for a spectrum
%   read from file.
%
%   Variance reduction:
%
%   * nSplit -- uniform photon splitting at the source; > 1 enables it.
%   * esave -- electron range rejection: electrons whose residual CSDA range cannot carry them out of the current voxel are terminated below this total energy (MeV). 0 or absent disables it.
%   * e_rr, f_rr -- unbiased Russian roulette of newly created electrons below total energy e_rr (MeV), with survival probability 1/f_rr. Both must be set (f_rr > 1) to take effect.
%
%   Photon transport uses Woodcock (delta) tracking, so photon steps are
%   not stopped at voxel boundaries.
%
%   See also RECORDPROGRESSCALLBACK

% This file is help text only, kept in step with omc_matrad.c by hand: the
% MEX file has no .m source of its own to generate this from. Do not add
% code here -- MATLAB always runs the MEX file, so a body here would never
% execute, and its presence would only make that non-obvious.
end

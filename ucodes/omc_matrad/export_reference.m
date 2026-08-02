%EXPORT_REFERENCE Save a MEX-computed Dij for the Python test suite to check against.
%
%   Runs test_fixture.mat through the omc_matrad MEX file and writes the
%   resulting sparse matrix, in the compressed sparse column arrays scipy
%   uses, to python/tests/mex_reference.mat. python/tests/test_ompmc.py picks
%   it up and holds the Python interface against it; without it that one test
%   skips.
%
%   Run with the directory holding omc_matrad.mex* on the path, from the
%   repository root:
%
%       addpath('build/bin'); addpath('ucodes/omc_matrad');
%       export_reference
%
%   Both interfaces drive the same engine with the same per-history random
%   streams, so the sparsity pattern has to come out identical and the values
%   to within the order in which threads accumulated into a voxel.

thisDir = fileparts(mfilename('fullpath'));
omcRoot = fileparts(fileparts(thisDir));

fixture = load(fullfile(thisDir, 'test_fixture.mat'));

mcOpt = fixture.mcOpt;
mcOpt.spectrumFile = fullfile(omcRoot, 'spectra', fixture.meta.spectrumFile);
mcOpt.dataFolder   = [fullfile(omcRoot, 'data') filesep];
mcOpt.pegsFile     = fullfile(omcRoot, 'pegs4', fixture.meta.pegsFile);
mcOpt.pgs4formFile = fullfile(omcRoot, 'pegs4', 'pgs4form.dat');
mcOpt.outputFolder = [fullfile(omcRoot, 'output') filesep];

dij = omc_matrad(fixture.cubeRho, fixture.cubeMatIx, ...
    fixture.mcGeo, fixture.mcSrc, mcOpt);

% find() walks the stored arrays, so it comes back in compressed sparse
% column order: all of column 1, then column 2, and ascending rows within each.
[rows, cols, values] = find(dij);

nCols = size(dij, 2);
indices = int32(rows - 1);                             % scipy counts from 0
indptr = int64([0; cumsum(accumarray(cols, 1, [nCols 1]))]);
shape = int64(size(dij));

outFile = fullfile(omcRoot, 'ucodes', 'omc_python', 'tests', 'mex_reference.mat');
save(outFile, 'indices', 'indptr', 'values', 'shape', '-v7');

fprintf('Wrote %d nonzeros of a %dx%d matrix to %s\n', ...
    nnz(dij), shape(1), shape(2), outFile);

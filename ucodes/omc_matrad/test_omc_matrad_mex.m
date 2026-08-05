%TEST_OMC_MATRAD_MEX Smoke test for the omc_matrad MEX file.
%
%   Checks that the compiled MEX file loads on this platform, runs a real dose
%   calculation, and can then be released without taking MATLAB down with it.
%   Run it with the directory holding omc_matrad.mex* on the path:
%
%       addpath('build/bin'); addpath('ucodes/omc_matrad');
%       test_omc_matrad_mex
%
%   The inputs come from test_fixture.mat, captured from matRad's
%   matRad_PhotonOmpMCEngine for a small BOXPHANTOM photon plan. The data,
%   pegs4 and spectrum paths are stripped from the fixture and filled in below
%   from the ompMC source tree this test lives in, so the test is independent
%   of where matRad happens to be installed.
%
%   Only structural properties of the result are checked. ompMC seeds its
%   random number generator per thread, so the numbers depend on how many
%   threads the machine runs and are not comparable across platforms.

thisDir = fileparts(mfilename('fullpath'));
omcRoot = fileparts(fileparts(thisDir));       % ucodes/omc_matrad -> repo root

%% The MEX file is there and can be entered

if exist('omc_matrad', 'file') ~= 3
    error('ompMC:test:mexNotFound', ...
        'omc_matrad is not on the MATLAB path as a MEX file.');
end
fprintf('MEX file: %s\n', which('omc_matrad'));

% Calling with no arguments trips the input check at the top of mexFunction,
% which proves the MEX file loaded and resolved all of its symbols.
try
    omc_matrad();
    error('ompMC:test:noError', ...
        'omc_matrad accepted a call with no arguments, which it should reject.');
catch err
    if ~strcmp(err.identifier, 'matRad:matRad_ompInterface:invalidNumInputs')
        rethrow(err);
    end
end
fprintf('Entered mexFunction and got the expected argument check.\n');

%% Version query

v = omc_matrad('version');
if ~ischar(v) || isempty(regexp(v, '^\d+\.\d+\.\d+$', 'once'))
    error('ompMC:test:badVersion', ...
        'omc_matrad(''version'') returned %s, expected a MAJOR.MINOR.PATCH string.', ...
        mat2str(v));
end
fprintf('omc_matrad(''version'') returned %s.\n', v);

%% A real dose calculation

fixture = load(fullfile(thisDir, 'test_fixture.mat'));
fprintf('Fixture: %s\n', fixture.meta.description);

mcOpt = fixture.mcOpt;
mcOpt.spectrumFile = fullfile(omcRoot, 'spectra', fixture.meta.spectrumFile);
mcOpt.dataFolder   = [fullfile(omcRoot, 'data') filesep];
mcOpt.pegsFile     = fullfile(omcRoot, 'pegs4', fixture.meta.pegsFile);
mcOpt.pgs4formFile = fullfile(omcRoot, 'pegs4', 'pgs4form.dat');
mcOpt.outputFolder = [fullfile(omcRoot, 'output') filesep];

tStart = tic;
[dij, dijVar] = omc_matrad(fixture.cubeRho, fixture.cubeMatIx, ...
    fixture.mcGeo, fixture.mcSrc, mcOpt);
fprintf('Dose calculation with %d histories took %.1f s.\n', ...
    mcOpt.nHistories, toc(tStart));

expectedSize = [numel(fixture.cubeRho), fixture.mcSrc.nBixels];

if ~issparse(dij)
    error('ompMC:test:notSparse', 'The dose influence matrix is not sparse.');
end
if ~isequal(size(dij), expectedSize)
    error('ompMC:test:wrongSize', ...
        'Dose influence matrix is %s, expected %s.', ...
        mat2str(size(dij)), mat2str(expectedSize));
end

dose = nonzeros(dij);
if isempty(dose)
    error('ompMC:test:noDose', 'The dose influence matrix is all zero.');
end
if ~all(isfinite(dose))
    error('ompMC:test:notFinite', ...
        '%d of %d nonzero dose entries are Inf or NaN.', ...
        sum(~isfinite(dose)), numel(dose));
end
if any(dose < 0)
    error('ompMC:test:negativeDose', ...
        '%d dose entries are negative.', sum(dose < 0));
end

% Every beamlet has to deposit something; a beamlet that scores nothing means
% the source geometry or the beamlet loop is broken.
beamletDose = full(sum(dij, 1));
if ~all(beamletDose > 0)
    error('ompMC:test:emptyBeamlet', ...
        '%d of %d beamlets deposited no dose at all.', ...
        sum(beamletDose <= 0), numel(beamletDose));
end

fprintf('dij: %s sparse, %d nonzeros, max %.4g, all %d beamlets scored.\n', ...
    mat2str(size(dij)), nnz(dij), full(max(dose)), numel(beamletDose));

% A CSC matrix has to hold ascending row indices within each column. The MEX
% file fills the columns from the list of voxels the beamlet deposited in,
% which is built in whatever order the transport happened to reach them and
% only becomes ascending because it is sorted before use; get that wrong and
% the matrix is quietly malformed rather than obviously broken. find() walks
% the stored arrays in order, so a column whose rows come back out of order
% is the symptom.
[rowIdx, colIdx] = find(dij);
for k = 1:size(dij, 2)
    rowsInColumn = rowIdx(colIdx == k);
    if ~issorted(rowsInColumn)
        error('ompMC:test:unsortedColumn', ...
            'Row indices of column %d are not ascending; the sparse matrix is malformed.', k);
    end
end
fprintf('All %d columns hold ascending row indices.\n', size(dij, 2));

if ~isequal(size(dijVar), expectedSize) || nnz(dijVar) == 0
    error('ompMC:test:badVariance', ...
        'The variance output is %s with %d nonzeros, expected %s and nonzero.', ...
        mat2str(size(dijVar)), nnz(dijVar), mat2str(expectedSize));
end
fprintf('Variance output: %d nonzeros.\n', nnz(dijVar));

%% mcOpt.spectrum replaces mcOpt.spectrumFile

% The same spectrum, read in MATLAB and handed over as arrays, has to give the
% same dose as letting the MEX file read the file itself. Both runs use the
% same seeds and the same per-history random streams, so they only differ by
% the order in which threads accumulate energy into a voxel.
fid = fopen(mcOpt.spectrumFile, 'r');
if fid < 0
    error('ompMC:test:noSpectrumFile', ...
        'Could not open the spectrum file %s.', mcOpt.spectrumFile);
end
cleanupSpectrum = onCleanup(@() fclose(fid));
fgetl(fid);                                  % title line
header = sscanf(fgetl(fid), '%d %f %d', 3);  % nBins, lower edge, mode
bins = fscanf(fid, '%f %f', [2, header(1)])';
clear cleanupSpectrum;

if size(bins, 1) ~= header(1)
    error('ompMC:test:badSpectrumFile', ...
        'Read %d of the %d bins announced by %s.', ...
        size(bins, 1), header(1), mcOpt.spectrumFile);
end

mcOptSpec = rmfield(mcOpt, 'spectrumFile');
mcOptSpec.spectrum = struct('energy', bins(:, 1), 'fluence', bins(:, 2), ...
    'eMin', header(2), 'mode', header(3));

dijSpec = omc_matrad(fixture.cubeRho, fixture.cubeMatIx, ...
    fixture.mcGeo, fixture.mcSrc, mcOptSpec);

if ~isequal(size(dijSpec), expectedSize)
    error('ompMC:test:spectrumWrongSize', ...
        'Dose influence matrix computed from a passed spectrum is %s, expected %s.', ...
        mat2str(size(dijSpec)), mat2str(expectedSize));
end

totalDose = full(sum(dij(:)));
relDiff = abs(full(sum(dijSpec(:))) - totalDose)/totalDose;
if ~(relDiff < 1e-3)
    error('ompMC:test:spectrumMismatch', ...
        ['Passing the spectrum gave a total dose differing by %.3g from ', ...
         'reading the same spectrum from file.'], relDiff);
end
fprintf('Passed spectrum with %d bins reproduced the file result to %.3g relative.\n', ...
    header(1), relDiff);

% A malformed spectrum has to be rejected up front rather than sampled.
mcOptBad = mcOptSpec;
mcOptBad.spectrum.energy = flipud(mcOptBad.spectrum.energy);
try
    omc_matrad(fixture.cubeRho, fixture.cubeMatIx, ...
        fixture.mcGeo, fixture.mcSrc, mcOptBad);
    error('ompMC:test:badSpectrumAccepted', ...
        'A spectrum with descending bin energies was accepted.');
catch err
    if ~strcmp(err.identifier, 'matRad:omc_matrad:invalidSpectrum')
        rethrow(err);
    end
end
fprintf('A spectrum with descending bin energies was rejected.\n');

%% mcOpt.monoEnergy replaces the spectrum with a single energy

% 6 MeV photons are far more penetrating than the 6 MV bremsstrahlung
% spectrum they are compared against here, whose mean energy is around 2 MeV,
% so the two runs must not agree.
mcOptMono = rmfield(mcOpt, 'spectrumFile');
mcOptMono.monoEnergy = 6;

dijMono = omc_matrad(fixture.cubeRho, fixture.cubeMatIx, ...
    fixture.mcGeo, fixture.mcSrc, mcOptMono);

monoDose = nonzeros(dijMono);
if isempty(monoDose) || ~all(isfinite(monoDose)) || any(monoDose < 0)
    error('ompMC:test:badMonoDose', ...
        'The monoenergetic source produced no usable dose (%d nonzeros).', ...
        numel(monoDose));
end

monoRatio = full(sum(dijMono(:)))/totalDose;
if ~(monoRatio > 1.05)
    error('ompMC:test:monoEnergyIgnored', ...
        ['A 6 MeV monoenergetic source deposited %.3fx the dose of the 6 MV ', ...
         'spectrum; it should deposit noticeably more, so monoEnergy looks ignored.'], ...
        monoRatio);
end
fprintf('Monoenergetic 6 MeV source deposited %.2fx the dose of the 6 MV spectrum.\n', ...
    monoRatio);

% A spectrum still wins over monoEnergy, and a nonsensical energy is rejected.
mcOptBoth = mcOptSpec;
mcOptBoth.monoEnergy = 6;
dijBoth = omc_matrad(fixture.cubeRho, fixture.cubeMatIx, ...
    fixture.mcGeo, fixture.mcSrc, mcOptBoth);
bothRatio = full(sum(dijBoth(:)))/totalDose;
if ~(abs(bothRatio - 1) < 1e-3)
    error('ompMC:test:precedenceWrong', ...
        ['With both a spectrum and monoEnergy given the result is %.3fx the ', ...
         'spectrum-only dose; the spectrum should have won.'], bothRatio);
end
fprintf('A passed spectrum took precedence over monoEnergy.\n');

mcOptBadEnergy = mcOptMono;
mcOptBadEnergy.monoEnergy = 0;
try
    omc_matrad(fixture.cubeRho, fixture.cubeMatIx, ...
        fixture.mcGeo, fixture.mcSrc, mcOptBadEnergy);
    error('ompMC:test:badEnergyAccepted', 'A monoEnergy of 0 was accepted.');
catch err
    if ~strcmp(err.identifier, 'matRad:omc_matrad:invalidMonoEnergy')
        rethrow(err);
    end
end
fprintf('A monoEnergy of 0 was rejected.\n');

%% mcOpt.charge selects the source particle

% charge used to be parsed and then ignored, so everything ran as photons.
% An electron source deposits its energy in a completely different place than
% a photon source of the same spectrum, which is what makes this detectable.
if mcOpt.charge ~= 0
    error('ompMC:test:fixtureNotPhotons', ...
        'The fixture uses charge %d, this check assumes a photon fixture.', ...
        mcOpt.charge);
end

mcOptElectron = mcOpt;
mcOptElectron.charge = -1;
dijElectron = omc_matrad(fixture.cubeRho, fixture.cubeMatIx, ...
    fixture.mcGeo, fixture.mcSrc, mcOptElectron);

electronDose = nonzeros(dijElectron);
if isempty(electronDose) || ~all(isfinite(electronDose)) || any(electronDose < 0)
    error('ompMC:test:badElectronDose', ...
        'The electron source produced no usable dose (%d nonzeros).', ...
        numel(electronDose));
end

% Same voxel grid, same beamlets, same spectrum: if charge were still ignored
% the two runs would agree to within the accumulation order.
sharedVoxels = nnz(dij & dijElectron)/nnz(dij | dijElectron);
if sharedVoxels > 0.9
    error('ompMC:test:chargeIgnored', ...
        ['An electron source deposited in %.1f%% of the same voxels as the ', ...
         'photon source; mcOpt.charge looks ignored.'], 100*sharedVoxels);
end
fprintf('Electron source scored in a different region (%.1f%% voxel overlap with photons).\n', ...
    100*sharedVoxels);

% An out-of-range charge has to be rejected rather than truncated to a photon.
mcOptBadCharge = mcOpt;
mcOptBadCharge.charge = 2;
try
    omc_matrad(fixture.cubeRho, fixture.cubeMatIx, ...
        fixture.mcGeo, fixture.mcSrc, mcOptBadCharge);
    error('ompMC:test:badChargeAccepted', 'A charge of 2 was accepted.');
catch err
    if ~strcmp(err.identifier, 'matRad:omc_matrad:invalidCharge')
        rethrow(err);
    end
end
fprintf('A charge of 2 was rejected.\n');

%% mcOpt.progressCallback replaces the built-in waitbar

% verbose = 2 would normally pop up a waitbar; a progressCallback should
% take over that reporting instead and no figure should be created for it.
global progressLog; %#ok<GVMIS>
progressLog = [];

mcOptCb = mcOpt;
mcOptCb.verbose = 2;
mcOptCb.progressCallback = @recordProgressCallback;

figuresBefore = findall(0, 'Type', 'figure');
dijCb = omc_matrad(fixture.cubeRho, fixture.cubeMatIx, ...
    fixture.mcGeo, fixture.mcSrc, mcOptCb);
figuresAfter = findall(0, 'Type', 'figure');

if numel(figuresAfter) > numel(figuresBefore)
    error('ompMC:test:waitbarNotSuppressed', ...
        'A figure was created even though a progressCallback was supplied; the built-in waitbar was not suppressed.');
end
if isempty(progressLog)
    error('ompMC:test:callbackNotCalled', 'progressCallback was never invoked.');
end
if any(progressLog < 0 | progressLog > 1)
    error('ompMC:test:callbackOutOfRange', 'progressCallback received a value outside [0,1].');
end
if ~issorted(progressLog)
    error('ompMC:test:callbackNotMonotonic', 'progressCallback values are not non-decreasing.');
end
if progressLog(end) ~= 1
    error('ompMC:test:callbackDidNotFinish', ...
        'Last progressCallback value was %.4f, expected 1.', progressLog(end));
end
if ~isequal(size(dijCb), expectedSize)
    error('ompMC:test:callbackWrongSize', ...
        'Dose influence matrix computed with a progressCallback is %s, expected %s.', ...
        mat2str(size(dijCb)), mat2str(expectedSize));
end

fprintf('progressCallback invoked %d times, from %.4f to %.4f, no waitbar figure created.\n', ...
    numel(progressLog), progressLog(1), progressLog(end));

clear global progressLog;

%% mcOpt.mode = 'forward_beamlet' is dij*w computed directly

% The forward mode spreads its histories over the beamlets in proportion to
% the weights instead of giving every beamlet the same number, and adds them
% up into one cube. What comes out has to be what dij*w would have been, so
% that is what it is held against here.
%
% Two things have to be lined up for the comparison to mean anything. The
% reference dij must not be pruned, because relDoseThreshold drops low dose
% voxels that the forward cube keeps; and nHistories counts the whole
% calculation in the forward mode but one beamlet in the dij mode, so the
% forward run gets nBixels times as many to reach the same statistics.

nBixels = fixture.mcSrc.nBixels;

weights = zeros(nBixels, 1);
weights(1:2:end) = 1;        % every other beamlet open
weights(2) = 0.25;           % and one that only partly transmits

mcSrcWeighted = fixture.mcSrc;
mcSrcWeighted.bixelWeights = weights;

mcOptForward = mcOpt;
mcOptForward.mode = 'forward_beamlet';
mcOptForward.nHistories = mcOpt.nHistories*nBixels;

tStart = tic;
[doseCube, relUnc] = omc_matrad(fixture.cubeRho, fixture.cubeMatIx, ...
    fixture.mcGeo, mcSrcWeighted, mcOptForward);
fprintf('Forward calculation with %d histories took %.1f s.\n', ...
    mcOptForward.nHistories, toc(tStart));

if ~isequal(size(doseCube), size(fixture.cubeRho))
    error('ompMC:test:forwardWrongSize', ...
        'The forward dose cube is %s, expected %s.', ...
        mat2str(size(doseCube)), mat2str(size(fixture.cubeRho)));
end
if issparse(doseCube)
    error('ompMC:test:forwardSparse', 'The forward dose cube is sparse.');
end
if ~all(isfinite(doseCube(:))) || any(doseCube(:) < 0)
    error('ompMC:test:forwardBadDose', ...
        'The forward dose cube holds %d non-finite and %d negative entries.', ...
        sum(~isfinite(doseCube(:))), sum(doseCube(:) < 0));
end
if ~(max(doseCube(:)) > 0)
    error('ompMC:test:forwardNoDose', 'The forward dose cube is all zero.');
end

% The unpruned reference
mcOptRef = mcOpt;
mcOptRef.relDoseThreshold = 0;
dijRef = omc_matrad(fixture.cubeRho, fixture.cubeMatIx, ...
    fixture.mcGeo, fixture.mcSrc, mcOptRef);
refCube = reshape(full(dijRef*weights), size(fixture.cubeRho));

% The total is the part that does not depend on where the dose landed, so it
% is the tightest thing to compare. The two runs draw different random
% streams, so what is left is the statistical spread of a 50000 history
% calculation, which is well inside a percent.
forwardTotal = sum(doseCube(:));
refTotal = sum(refCube(:));
totalDiff = abs(forwardTotal - refTotal)/refTotal;

if ~(totalDiff < 0.02)
    error('ompMC:test:forwardTotalMismatch', ...
        ['The forward cube holds %.4g Gy against %.4g Gy for dij*w, a ', ...
         'relative difference of %.3g.'], forwardTotal, refTotal, totalDiff);
end
fprintf('Forward total dose agrees with dij*w to %.3g relative.\n', totalDiff);

% Totals agreeing is not enough on its own: a cube written out with its axes
% in the wrong order would still total the same. Comparing the profile along
% each axis pins down where the dose actually went.
%
% This is a shape check, not a statistics one, and the tolerance is loose on
% purpose. A single profile bin is a far noisier quantity than the total: at
% the history count used here the two runs differ by up to 6% in one, and that
% is genuine statistical spread rather than disagreement -- raising both runs
% by a factor of 16 brings it down to 1.5%, which is the 1/sqrt(N) a Monte
% Carlo owes you. What this catches is a cube whose axes came out permuted,
% which is wrong by whole factors rather than by percents.
for iAxis = 1:3
    other = setdiff(1:3, iAxis);
    forwardProfile = sum(sum(doseCube, other(1)), other(2));
    refProfile = sum(sum(refCube, other(1)), other(2));

    forwardProfile = forwardProfile(:);
    refProfile = refProfile(:);

    % Only where there is dose to compare; the tails are all noise.
    hot = refProfile > 0.05*max(refProfile);
    profileDiff = max(abs(forwardProfile(hot) - refProfile(hot)) ...
                      ./refProfile(hot));

    if ~(profileDiff < 0.15)
        error('ompMC:test:forwardProfileMismatch', ...
            ['The forward dose profile along axis %d differs from dij*w by ', ...
             'up to %.3g where there is dose.'], iAxis, profileDiff);
    end
    fprintf('Axis %d profile agrees with dij*w to %.3g.\n', iAxis, profileDiff);
end

% The uncertainty cube follows the .3ddose convention omc_dosxyz uses: a
% relative uncertainty everywhere there is dose, and 0.9999999 where there is
% none.
if ~isequal(size(relUnc), size(doseCube))
    error('ompMC:test:forwardUncSize', ...
        'The forward uncertainty cube is %s, expected %s.', ...
        mat2str(size(relUnc)), mat2str(size(doseCube)));
end
if any(relUnc(:) < 0) || any(relUnc(:) > 1)
    error('ompMC:test:forwardUncRange', ...
        'The relative uncertainty leaves [0,1] in %d voxels.', ...
        sum(relUnc(:) < 0 | relUnc(:) > 1));
end
if ~all(abs(relUnc(doseCube == 0) - 0.9999999) < 1e-9)
    error('ompMC:test:forwardUncConvention', ...
        'Voxels without dose do not carry the 0.9999999 uncertainty.');
end
fprintf('Uncertainty cube: median %.3g where there is dose.\n', ...
    median(relUnc(doseCube > 0)));

%% The forward mode rejects what it cannot use

badCases = { ...
    'wrongLength',  'matRad:omc_matrad:invalidField', ones(nBixels + 1, 1); ...
    'negative',     'ompMC:forward:invalidWeight',    -ones(nBixels, 1); ...
    'allZero',      'ompMC:forward:noWeight',         zeros(nBixels, 1)};

for iCase = 1:size(badCases, 1)
    mcSrcBad = fixture.mcSrc;
    mcSrcBad.bixelWeights = badCases{iCase, 3};

    try
        omc_matrad(fixture.cubeRho, fixture.cubeMatIx, ...
            fixture.mcGeo, mcSrcBad, mcOptForward);
        error('ompMC:test:badWeightsAccepted', ...
            'Weights of case "%s" were accepted.', badCases{iCase, 1});
    catch err
        if ~strcmp(err.identifier, badCases{iCase, 2})
            rethrow(err);
        end
    end
end
fprintf('Weights that are the wrong length, negative or all zero were rejected.\n');

% Without any weights at all the mode cannot run.
try
    omc_matrad(fixture.cubeRho, fixture.cubeMatIx, ...
        fixture.mcGeo, fixture.mcSrc, mcOptForward);
    error('ompMC:test:missingWeightsAccepted', ...
        'The forward mode ran without mcSrc.bixelWeights.');
catch err
    if ~strcmp(err.identifier, 'matRad:omc_matrad:missingField')
        rethrow(err);
    end
end
fprintf('The forward mode without bixelWeights was rejected.\n');

% And an unknown mode must not quietly fall back to the default.
mcOptBadMode = mcOpt;
mcOptBadMode.mode = 'forward_shape';
try
    omc_matrad(fixture.cubeRho, fixture.cubeMatIx, ...
        fixture.mcGeo, fixture.mcSrc, mcOptBadMode);
    error('ompMC:test:badModeAccepted', 'An unknown mcOpt.mode was accepted.');
catch err
    if ~strcmp(err.identifier, 'matRad:omc_matrad:invalidMode')
        rethrow(err);
    end
end
fprintf('An unknown mcOpt.mode was rejected.\n');

%% Releasing the MEX file after a parallel region

% This is the part that used to bring MATLAB down. Once an OpenMP parallel
% region has run, the runtime's worker threads outlive the MEX file, so
% unloading it -- on "clear mex" or when MATLAB exits -- unloads the runtime
% from under them. mexFunction therefore locks itself.
if ~mislocked('omc_matrad')
    error('ompMC:test:notLocked', ...
        'omc_matrad did not lock itself in memory.');
end

clear omc_matrad mex
if ~mislocked('omc_matrad')
    error('ompMC:test:unloaded', ...
        'omc_matrad was unloaded by "clear mex" despite being locked.');
end
fprintf('MEX file stayed locked across "clear mex".\n');

fprintf('omc_matrad MEX smoke test passed.\n');

function recordProgressCallback(p)
%RECORDPROGRESSCALLBACK Append a progress value reported by omc_matrad.
%   Passed to omc_matrad as mcOpt.progressCallback by test_omc_matrad_mex;
%   logs to a global because the MEX file invokes it outside the caller's
%   workspace, so an ordinary local variable would not be reachable.
%
%   This lives in its own file rather than as a local function of the test.
%   MATLAB resolves a handle to a script-local function when the MEX file
%   calls it back, but Octave does not, and fails the callback with
%   "invalid function handle, unable to find function for
%   @recordProgressCallback".

global progressLog; %#ok<GVMIS>
progressLog(end+1) = p;
end

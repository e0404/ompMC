---
applyTo: "tests/**,ucodes/omc_python/tests/**,ucodes/omc_matrad/test_*.m"
---

- **C tests (`tests/`) use a small homegrown harness**, not a framework like
  Unity or CMocka: `CHECK`/`CHECK_CLOSE`/`RUN`/`EXPECT_FAIL`/`EXPECT_OK` macros
  defined locally in each test file, run through CTest which reads stdout for
  pass/fail. Don't suggest pulling in an external C test framework.
- **Failure-path tests exercise `omcFail()` via `setjmp`/`longjmp`.** On
  MinGW/x86-64 this requires the non-SEH `_setjmp` (`__USE_MINGW_SETJMP_NON_SEH`,
  defined before `<setjmp.h>` is included) because the default SEH-unwinding
  `longjmp` can crash when jumping out of `omcFail()` in an LTO-optimized
  static library. If a new test file exercises a failure path with
  `setjmp`/`longjmp`, check it follows the same pattern as the existing test
  files rather than including `<setjmp.h>` directly.
- **Test fixtures are synthesized in code where practical**, not committed as
  large binary blobs: a test builds a struct describing what a header/record
  should say and packs the bytes by hand, so a single field can be varied
  independently. Small fixtures (a few hundred bytes) are committed under
  `tests/data/` only to validate the synthesizers themselves against a real
  file. Don't ask for large committed binary test fixtures — prefer a
  synthesized one, following the existing pattern.
- **Numerical/physics tests should tolerate run-to-run noise correctly.**
  `omc_dosxyz` output is deterministically byte-identical run-to-run (the RNG
  is keyed per-history), so a `cmp`-style exact-match assertion is valid
  there. The matRad MEX path is **not** byte-reproducible (`#pragma omp atomic`
  accumulation order varies), so tests against it must use a statistical or
  relative-tolerance comparison, not exact equality — check which case a new
  numerical test is in before judging its tolerance as too loose or too
  strict.
- **Cross-host validation matters more than either host's tests alone.** A
  change to shared engine behavior should ideally be reflected in both the
  MATLAB (`ucodes/omc_matrad/test_omc_matrad_mex.m`) and Python
  (`ucodes/omc_python/tests/`) test suites, since keeping those two hosts from
  drifting apart is a stated goal of this codebase.

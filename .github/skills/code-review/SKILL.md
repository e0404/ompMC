---
name: code-review
description: Checklist-driven review guidance for ompMC — a Monte Carlo dose-calculation engine where the main risk is a silent physics/numerical regression, not a crash. Use when reviewing changes to the physics core (src/), the MATLAB MEX host (ucodes/omc_matrad), the Python extension (ucodes/omc_python), or tests, to check them against this project's known correctness invariants and past bug classes.
---

# ompMC code review skill

ompMC computes radiotherapy dose. A change can compile, pass a smoke test,
and still be wrong in a way that only shows up as a small, plausible-looking
bias in the output — that is the failure mode this project has been bitten by
before, and what this skill exists to catch.

Before finishing a review of any diff touching `src/`, `ucodes/omc_matrad/`,
or `ucodes/omc_python/`, work through [checklist.md](checklist.md). It is
short by design — it encodes specific bug classes this codebase has actually
shipped, not a generic C review checklist.

Also read [AGENTS.md](../../../AGENTS.md) at the repository root first: it
lists patterns that deliberately look like these bug classes but are not
(intentional asymmetries between engines, an RNG-keying scheme that looks
unusual, a variance-reduction technique that is unbiased but off by default).
Cross-check anything the checklist flags against that list before raising it.

For path-specific detail beyond the checklist, see:

- `.github/instructions/core-physics.instructions.md` — the transport engine
- `.github/instructions/mex-host.instructions.md` — the MATLAB/Octave MEX host
- `.github/instructions/python-bindings.instructions.md` — the nanobind extension
- `.github/instructions/tests.instructions.md` — test conventions

If the checklist raises an item you can't resolve from the diff alone (for
example, "was unbiasedness checked?" when the PR description doesn't say),
say so explicitly in the review rather than assuming it's fine.

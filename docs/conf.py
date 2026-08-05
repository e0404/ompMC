"""Sphinx configuration for the ompMC documentation.

Builds with no compiler and no CMake configure step: the C API comes from
Doxygen (invoked below as a subprocess) and the Python API from the pure
Python ``ompmc`` package under ``ucodes/omc_python``, imported straight from
source with its compiled ``_ompmc`` extension mocked out. See
docs/README.md for how to build this locally.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
ROOT_DIR = DOCS_DIR.parent

# So `import ompmc` finds the pure Python package without installing the
# compiled extension (see autodoc_mock_imports below).
sys.path.insert(0, str(ROOT_DIR / "ucodes" / "omc_python"))

# -- Project information -----------------------------------------------------

project = "ompMC"
copyright = "2018, Edgardo Doerner"
author = "Edgardo Doerner"


def _version_from_cmake() -> str:
    """The single source of truth for the version is CMakeLists.txt's
    project() call; pyproject.toml reads it with the same regex."""
    text = (ROOT_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(r"project\(ompMC\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)", text)
    return match.group(1) if match else "0.0.0"


release = _version_from_cmake()
version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "breathe",
    "sphinxcontrib.matlab",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = []
# README.md is about building the *docs*, for someone reading the repo
# source; it is not a page of the published site.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "doxygen", "README.md"]

# BUILDING.md is included verbatim into getting-started/installation.md (see
# there) so it stays a single source of truth, but its GitHub-relative links
# to files outside docs/ (src/, LICENSE, .github/...) are not part of this
# site and MyST cannot resolve them as cross-references. That is a known,
# accepted gap -- see installation.md's note -- rather than a broken link in
# hand-authored content, which is why it is silenced globally instead of
# fixed link by link.
suppress_warnings = [
    "myst.xref_missing",
    # Breathe's `doxygenfile` directive emits a self-referencing permalink
    # for the file compound itself (distinct from the functions/structs
    # inside it, which link fine); that target is never actually registered,
    # so nitpicky mode (-n) reports it as broken on every c-api page
    # regardless of what the file documents. Tracked upstream in breathe;
    # nothing on our side to fix.
    "ref.ref",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosectionlabel_prefix_document = True

# -- MyST ----------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
myst_heading_anchors = 3

# -- Autodoc / Napoleon ----------------------------------------------------

# The compiled extension needs a C++ toolchain, CMake and OpenMP to build.
# Nothing in the public API requires it to be importable: the pure Python
# wrapper in ompmc/__init__.py is what is documented, so the compiled
# submodule is mocked instead of built.
autodoc_mock_imports = ["ompmc._ompmc"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_attr_annotations = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Doxygen / Breathe ---------------------------------------------------

DOXYGEN_XML_DIR = DOCS_DIR / "doxygen" / "xml"


def _run_doxygen() -> None:
    """Regenerate the Doxygen XML that breathe reads from.

    Run unconditionally on every build (local or Read the Docs) rather than
    from CMake, so that `sphinx-build docs docs/_build/html` is the entire
    command and docs never depend on a configured CMake tree. Read the Docs
    installs the `doxygen` binary via `build.apt_packages` in
    .readthedocs.yaml.
    """
    doxyfile = DOCS_DIR / "Doxyfile"
    env = dict(os.environ, OMPMC_DOXYGEN_VERSION=release)
    subprocess.run(["doxygen", str(doxyfile)], cwd=DOCS_DIR, env=env, check=True)


if os.environ.get("SKIP_DOXYGEN") != "1":
    _run_doxygen()

breathe_projects = {"ompMC": str(DOXYGEN_XML_DIR)}
breathe_default_project = "ompMC"
breathe_default_members = ("members", "undoc-members")
# Without this, breathe renders .h files under the C++ domain (a header could
# be either), and OPTIMIZE_OUTPUT_FOR_C in the Doxyfile is wasted.
breathe_domain_by_extension = {"h": "c"}

# The C domain tries to cross-reference every type name it sees in a
# signature. <stdint.h>'s fixed-width types are never going to be Doxygen
# output (they are not part of this project's INPUT), so nitpick mode (-n)
# would otherwise fail the build over them.
nitpick_ignore_regex = [
    ("c:identifier", r"u?int(8|16|32|64)_t"),
    ("c:identifier", r"size_t"),
    # Documented as c:macro entries (see omc_utilities.h), which the
    # automatic xref generated for an array-size expression in a struct
    # member's signature does not look up under.
    ("c:identifier", r"BUFFER_SIZE"),
    ("c:identifier", r"INPUT_PAIRS"),
    # Napoleon renders a numpydoc "type, optional" field through the Python
    # domain's TypedField, which cross-references every comma/"or"-separated
    # token in it -- including words that describe the type rather than name
    # one. These are the vocabulary used across ompmc/__init__.py's
    # docstrings, not real classes.
    ("py:class", r"optional"),
    ("py:class", r"callable"),
    ("py:class", r"array_like"),
    ("py:class", r"sequence"),
]

# -- MATLAB domain -------------------------------------------------------

# omc_matrad is a compiled MEX file with no .m source, so docs/_matlab/
# carries a help-text-only omc_matrad.m stub for this to autodocument.
# It deliberately does NOT live in ucodes/omc_matrad/ alongside the real
# MEX file: build.yml's and CMakeLists.txt's MEX smoke test does
# `addpath('build/bin'); addpath('ucodes/omc_matrad')`, and addpath()
# prepends by default, so a same-named .m file placed there ends up
# ahead of build/bin on the path and permanently shadows the compiled
# MEX -- `exist('omc_matrad', 'file')` stops reporting 3 (MEX-file) and
# every MATLAB/Octave CI job fails. docs/_matlab/ is never added to that
# path, so the stub can only ever be seen by Sphinx.
matlab_src_dir = str(DOCS_DIR / "_matlab")

# -- HTML output ------------------------------------------------------------

html_theme = "furo"
# No custom CSS/JS yet -- an empty _static/ directory is invisible to git
# (it tracks no empty directories) and disappears on a fresh checkout,
# which -W turns into a build failure. Add html_static_path back along
# with the directory once there is an actual asset to put in it.
html_title = f"ompMC {version}"

html_theme_options = {
    "source_repository": "https://github.com/e0404/ompMC",
    "source_branch": "master",
    "source_directory": "docs/",
}

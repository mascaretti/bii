"""Sphinx configuration for the bii documentation.

The heavy numerical backends (JAX, blackjax, optax) are mocked so the docs build
on Read the Docs without installing them or a GPU -- autodoc only needs to import
the package to read signatures and docstrings, not to run anything.
"""
from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

# Make ``import bii`` resolve from the source tree (no install needed on RTD).
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.fspath(_ROOT / "src"))

# -- Project information ------------------------------------------------------
_meta = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]
project = "bii"
author = _meta["authors"][0]["name"]
copyright = f"2026, {author}"
release = _meta["version"]
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",        # Google/NumPy-style docstrings
    "sphinx.ext.viewcode",        # [source] links
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",         # render the LaTeX in docstrings/theory
    "myst_parser",                # Markdown pages
]

# JAX et al. are not installed in the docs environment; mock them for autodoc.
autodoc_mock_imports = ["jax", "jaxlib", "blackjax", "optax"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False

myst_enable_extensions = ["deflist", "dollarmath", "amsmath", "colon_fence"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = f"bii {release}"
html_static_path = ["_static"]

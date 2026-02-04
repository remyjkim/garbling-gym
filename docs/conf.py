# Configuration file for the Sphinx documentation builder.

import sys
from pathlib import Path

# Add source to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# -- Project information -----------------------------------------------------
project = "Garbling Gym"
copyright = "2025, Remy"
author = "Remy"
version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx_click",
    "myst_parser",
]

# Source files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "Garbling Gym Documentation"
html_short_title = "Garbling Gym"
html_logo = None
html_favicon = None

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# -- Options for autosummary -------------------------------------------------
autosummary_generate = True

# -- Options for napoleon (Google/NumPy style docstrings) -------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True

# -- Options for MyST --------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

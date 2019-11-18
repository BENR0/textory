# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
from datetime import datetime
from pkg_resources import get_distribution

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# -- Project information -----------------------------------------------------

project = 'Textory'
copyright = '2019, Benjamin Rösner'
author = 'Benjamin Rösner'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.intersphinx', 'sphinx.ext.todo', 'sphinx.ext.coverage',
        'sphinx.ext.doctest', 'sphinx.ext.napoleon', 'sphinx.ext.autosummary',
        'sphinx.ext.viewcode']

templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The suffix of the source filenames
source_suffix = '.rst'

# The master toctree document
master_doc = 'index'

# The name of the Pygments (syntax highlighting) style to use
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'# 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'dask': ('https://docs.dask.org/en/latest', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'python': ('https://docs.python.org/3', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'satpy': ('https://satpy.readthedocs.io/en/stable', None),
    'xarray': ('https://xarray.pydata.org/en/stable', None)
}

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import datetime
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(f'{os.path.dirname(__file__)}/../..'))


# -- Project information -----------------------------------------------------

project = 'CAME'
_copyright = 'Academy of Mathematics and Systems Science, CAS'
copyright = f'{datetime.now():%Y}, {_copyright}.'
author = 'Xingyan Liu'


# -- General configuration ---------------------------------------------------
pygments_style = 'sphinx'  # coloring code blocks

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # different doc-styles (Google, Numpy)
    'sphinx.ext.doctest',  
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',  # enable github-pages
    'sphinx_autodoc_typehints',  # needs to be after napoleon
    "nbsphinx",  # notebooks
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
autosummary_generate = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '*PARAMETERS.py'
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

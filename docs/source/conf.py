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
author = 'Xingyan Liu'
_copyright = 'Academy of Mathematics and Systems Science, CAS'
copyright = f'{datetime.now():%Y}, {_copyright}.'


# -- General configuration ---------------------------------------------------
pygments_style = 'sphinx'  # coloring code blocks

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # different doc-styles (Google, Numpy)
    # 'myst_parser',  # use Markdown using MyST
    "nbsphinx",  # notebooks
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',  # enable github-pages
    # 'sphinx_autodoc_typehints',  # needs to be after napoleon
]
autosummary_generate = True
autodoc_member_order = 'bysource'
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
todo_include_todos = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

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

# source_suffix = '.rst'
from recommonmark.parser import CommonMarkParser
source_parsers = {
    '.md': CommonMarkParser,
}
source_suffix = ['.rst', '.md']

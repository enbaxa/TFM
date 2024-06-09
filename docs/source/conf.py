# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sphinx_rtd_theme

project = 'repoclass'
copyright = '2024, Enric Basso'
author = 'Enric Basso'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',

]
autosummary_generate = True  # Enable autosummary

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster' NO GO
#html_theme = 'classic'  NO GO
#html_theme = 'sphinxdoc'
#html_theme = 'scrolls'
#html_theme = 'agogo' NO GO
# html_theme = 'nature' NO GO
#html_theme = 'haiku'
#html_theme = 'pyramid' NO GO
#html_theme = 'bizstyle' NO GO
#html_theme = 'traditional'
#html_theme = 'basic'
html_theme = 'sphinx_rtd_theme'


html_static_path = ['_static']

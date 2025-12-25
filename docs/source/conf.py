import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
project = 'bssunfold'
copyright = '2025'
author = 'Konstantin Chizhov'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
html_theme = "pydata_sphinx_theme"
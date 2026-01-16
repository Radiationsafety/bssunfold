import os
import sys


import tomllib

sys.path.insert(0, os.path.abspath("../../"))

# from pyproject.toml get version
with open("../pyproject.toml", "rb") as f:
    data = tomllib.load(f)

release = data["project"]["version"]

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "bssunfold"
copyright = "2025, Konstantin Chizhov"
author = "Konstantin Chizhov"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# Расширения
extensions = [
    "sphinx.ext.autodoc",  # Для автодокументации
    "sphinx.ext.napoleon",  # Для поддержки Google/Numpy docstrings
    "sphinx.ext.viewcode",  # Показывать исходный код
    "sphinx.ext.autosummary",  # Авто-сводка
    "sphinx.ext.intersphinx",  # Ссылки на другую документацию
]

# Настройки autodoc
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Настройки napoleon (для Google/Numpy стиля)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True

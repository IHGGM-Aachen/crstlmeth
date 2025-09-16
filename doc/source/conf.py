import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "crstlmeth"
copyright = "2025"
author = "Carlos Classen"
release = "v0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_click.ext",
]

autosummary_generate = True

autodoc_typehints = "description"
# Don’t import Streamlit at build time
autodoc_mock_imports = ["streamlit"]

html_theme = "furo"

todo_include_todos = True

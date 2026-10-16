import os
import sys
import runpy
from pathlib import Path
sys.path.insert(0, os.path.abspath('..'))

project = 'FiberNet'
copyright = '2026, FiberNet Contributors'
author = 'FiberNet Contributors'
# Read metadata without importing optional numerical or GUI dependencies.
release = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                            'fibernet' / 'version.py'))['__version__']
version = '.'.join(release.split('.')[:2])

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': False,
    'show-inheritance': True,
    'special-members': '__init__',
}

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
todo_include_todos = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

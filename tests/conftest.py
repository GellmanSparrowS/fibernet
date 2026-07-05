"""Pytest configuration for FiberNet tests."""

import os
import sys

# Set matplotlib backend to non-interactive for CI
os.environ.setdefault('MPLBACKEND', 'Agg')

# PyVista headless rendering setup for CI
os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')

# Try to start xvfb for headless rendering in CI
try:
    import pyvista
    if hasattr(pyvista, 'start_xvfb'):
        pyvista.start_xvfb()
except (ImportError, Exception):
    pass

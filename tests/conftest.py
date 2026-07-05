"""Pytest configuration for FiberNet tests."""

import os
import sys

# Set matplotlib backend to non-interactive for CI
os.environ.setdefault('MPLBACKEND', 'Agg')

# PyVista headless rendering setup for CI
os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')

# Block pyvista and vtk imports in CI to prevent segfaults on macOS/Windows
if os.environ.get("CI") == "true":
    class _BlockVTK:
        """Import hook to block pyvista and vtk modules in CI."""
        def find_module(self, name, path=None):
            if name in ('pyvista', 'vtk', 'vtkmodules') or \
               name.startswith('pyvista.') or \
               name.startswith('vtk.') or \
               name.startswith('vtkmodules.'):
                return self
            return None
        
        def load_module(self, name):
            raise ImportError(f"Blocked in CI: {name}")
    
    sys.meta_path.insert(0, _BlockVTK())
else:
    # Outside CI: try to start xvfb for headless rendering
    try:
        import pyvista
        if hasattr(pyvista, 'start_xvfb'):
            pyvista.start_xvfb()
    except (ImportError, Exception):
        pass

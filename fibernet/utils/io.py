"""
File I/O utilities for fiber networks.

Supports:
- JSON (built-in)
- HDF5 (built-in)
- VTK export (for ParaView)
- STL export (for 3D printing)
- CSV export (for data analysis)
"""

import numpy as np
from typing import Optional
from fibernet.core.network import FiberNetwork


def export_vtk(network: FiberNetwork, filepath: str):
    """Export fiber network to VTK format for ParaView visualization."""
    try:
        import pyvista as pv
        
        meshes = []
        for fiber in network.fibers:
            pts = fiber.centerline
            if len(pts) < 2:
                continue
            spline = pv.Spline(pts, n_points=max(len(pts), 10))
            tube = spline.tube(radius=fiber.radius, capping=True)
            meshes.append(tube)
        
        if meshes:
            combined = meshes[0]
            for m in meshes[1:]:
                combined = combined.merge(m)
            combined.save(filepath)
            print(f"Exported to {filepath}")
    except ImportError:
        print("PyVista required for VTK export. Install with: pip install pyvista")


def export_csv(network: FiberNetwork, filepath: str):
    """Export fiber data to CSV."""
    import csv
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fiber_id', 'length', 'radius', 'start_x', 'start_y', 'start_z',
                         'end_x', 'end_y', 'end_z', 'material', 'curvature_max', 'tortuosity'])
        
        for fiber in network.fibers:
            writer.writerow([
                fiber.fiber_id, fiber.length, fiber.radius,
                *fiber.start_point, *fiber.end_point,
                fiber.material.name, np.max(fiber.curvature()), fiber.tortuosity(),
            ])


def export_stl(network: FiberNetwork, filepath: str):
    """Export fiber network to STL for 3D printing."""
    try:
        import pyvista as pv
        
        meshes = []
        for fiber in network.fibers:
            pts = fiber.centerline
            if len(pts) < 2:
                continue
            spline = pv.Spline(pts, n_points=max(len(pts), 10))
            tube = spline.tube(radius=fiber.radius, capping=True)
            meshes.append(tube)
        
        if meshes:
            combined = meshes[0]
            for m in meshes[1:]:
                combined = combined.merge(m)
            combined.save(filepath)
    except ImportError:
        print("PyVista required for STL export.")

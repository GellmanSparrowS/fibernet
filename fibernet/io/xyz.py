"""
XYZ file format I/O.

Simple atomic coordinate format for visualization and exchange.
"""

import numpy as np
from fibernet.core.network import FiberNetwork


def to_xyz(
    network: FiberNetwork,
    filename: str,
    bead_spacing: float = None,
    comment: str = None,
) -> str:
    """Export fiber network to XYZ format.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    filename : str
        Output file path.
    bead_spacing : float
        Spacing between points along fibers.
    comment : str, optional
        Comment line.
    """
    if bead_spacing is None:
        bead_spacing = network.mean_radius * 3
    
    all_points = []
    all_elements = []
    
    for fiber in network.fibers:
        length = fiber.length
        num_pts = max(2, int(length / bead_spacing) + 1)
        
        if num_pts == 2:
            pts = np.array([fiber.start_point, fiber.end_point])
        else:
            pts = fiber.resample(num_pts).centerline
        
        for pt in pts:
            all_points.append(pt)
            all_elements.append(fiber.material.name[:2].upper())
    
    if comment is None:
        comment = f"FiberNet: {network.num_fibers} fibers, {len(all_points)} points"
    
    with open(filename, 'w') as f:
        f.write(f"{len(all_points)}\n")
        f.write(f"{comment}\n")
        for elem, pt in zip(all_elements, all_points):
            f.write(f"{elem:2s} {pt[0]:12.6f} {pt[1]:12.6f} {pt[2]:12.6f}\n")
    
    return filename


def from_xyz(filename: str) -> np.ndarray:
    """Read points from XYZ file.
    
    Returns
    -------
    np.ndarray
        Array of shape (N, 3) with point coordinates.
    """
    points = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 3:
        return np.array([]).reshape(0, 3)
    
    num_atoms = int(lines[0].strip())
    
    for line in lines[2:2+num_atoms]:
        parts = line.split()
        if len(parts) >= 4:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            points.append([x, y, z])
    
    return np.array(points) if points else np.array([]).reshape(0, 3)

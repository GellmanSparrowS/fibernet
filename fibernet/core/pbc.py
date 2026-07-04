"""
Periodic boundary conditions for fiber networks.

Implements minimum image convention and periodic wrapping
for infinite/repeating fiber network structures.
"""

import numpy as np
from typing import Optional, Tuple, List
from copy import deepcopy

from fibernet.core.network import FiberNetwork
from fibernet.core.fiber import Fiber


class PeriodicBox:
    """Orthorhombic periodic simulation box.
    
    Parameters
    ----------
    box_size : array-like
        Box dimensions [Lx, Ly, Lz].
    origin : array-like, optional
        Box origin. Defaults to [0, 0, 0].
    """
    
    def __init__(self, box_size, origin=None):
        self.box_size = np.asarray(box_size, dtype=float)
        self.origin = np.asarray(origin, dtype=float) if origin is not None else np.zeros(3)
    
    def wrap_point(self, point: np.ndarray) -> np.ndarray:
        """Wrap a point into the periodic box."""
        rel = np.asarray(point) - self.origin
        wrapped = rel % self.box_size
        return wrapped + self.origin
    
    def wrap_points(self, points: np.ndarray) -> np.ndarray:
        """Wrap multiple points into the periodic box."""
        rel = np.asarray(points) - self.origin
        wrapped = rel % self.box_size
        return wrapped + self.origin
    
    def minimum_image(self, r: np.ndarray) -> np.ndarray:
        """Apply minimum image convention to displacement vector."""
        r = np.asarray(r)
        return r - self.box_size * np.round(r / self.box_size)
    
    def distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute minimum-image distance between two points."""
        dr = self.minimum_image(np.asarray(p2) - np.asarray(p1))
        return float(np.linalg.norm(dr))
    
    def distance_matrix(self, points: np.ndarray) -> np.ndarray:
        """Compute pairwise minimum-image distance matrix."""
        n = len(points)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance(points[i], points[j])
                D[i, j] = d
                D[j, i] = d
        return D
    
    def nearest_image(self, point: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Find the nearest periodic image of point to target."""
        dr = np.asarray(point) - np.asarray(target)
        dr_mic = self.minimum_image(dr)
        return np.asarray(target) + dr_mic
    
    def volume(self) -> float:
        """Box volume."""
        return float(np.prod(self.box_size))
    
    def wrap_network(self, network: FiberNetwork) -> FiberNetwork:
        """Wrap all fiber centerlines into the periodic box.
        
        Fiber segments that cross the boundary are split
        at the boundary, creating wrapped image segments.
        """
        result = deepcopy(network)
        
        for fiber in result.fibers:
            pts = fiber.centerline.copy()
            wrapped = self.wrap_points(pts)
            fiber.centerline = wrapped
        
        result.metadata['periodic'] = True
        result.metadata['box'] = self.box_size.tolist()
        result.metadata['origin'] = self.origin.tolist()
        return result
    
    def replicate(self, network: FiberNetwork, repeats: Tuple[int, int, int] = (1, 1, 1)) -> FiberNetwork:
        """Create a supercell by replicating the network.
        
        Parameters
        ----------
        repeats : tuple of int
            Number of repetitions in each direction.
        """
        wrapped = self.wrap_network(network)
        result = FiberNetwork(dimension=network.dimension)
        
        fiber_id = 0
        for ix in range(repeats[0]):
            for iy in range(repeats[1]):
                for iz in range(repeats[2]):
                    offset = np.array([ix, iy, iz]) * self.box_size
                    for fiber in wrapped.fibers:
                        new_fiber = deepcopy(fiber)
                        new_fiber.centerline = fiber.centerline + offset
                        new_fiber.fiber_id = fiber_id
                        fiber_id += 1
                        result.add_fiber(new_fiber)
        
        result.auto_crosslink()
        return result


def apply_pbc(
    network: FiberNetwork,
    box_size=None,
    origin=None,
) -> Tuple[FiberNetwork, PeriodicBox]:
    """Apply periodic boundary conditions to a fiber network.
    
    Parameters
    ----------
    network : FiberNetwork
        Input network.
    box_size : array-like, optional
        Box size. Defaults to network bounding box.
    origin : array-like, optional
        Box origin. Defaults to bounding box minimum.
    
    Returns
    -------
    network : FiberNetwork
        Wrapped network.
    box : PeriodicBox
        Periodic box object.
    """
    if box_size is None:
        bb_min, bb_max = network.bounding_box()
        box_size = bb_max - bb_min
        origin = bb_min
    
    box = PeriodicBox(box_size, origin)
    wrapped = box.wrap_network(network)
    return wrapped, box


def pbc_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    box: PeriodicBox,
) -> float:
    """Compute distance between two points with PBC."""
    return box.distance(p1, p2)


def compute_rdf(
    network: FiberNetwork,
    box: PeriodicBox,
    r_max: float = None,
    num_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radial distribution function with PBC.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    box : PeriodicBox
        Periodic box.
    r_max : float, optional
        Maximum distance. Defaults to half box length.
    num_bins : int
        Number of histogram bins.
    
    Returns
    -------
    r : np.ndarray
        Distance values.
    g : np.ndarray
        g(r) values.
    """
    if r_max is None:
        r_max = min(box.box_size) / 2
    
    # Collect all node positions
    positions = []
    for fiber in network.fibers:
        for pt in fiber.centerline:
            positions.append(pt)
    
    if len(positions) < 2:
        return np.linspace(0, r_max, num_bins), np.zeros(num_bins)
    
    positions = np.array(positions)
    n = len(positions)
    
    # Compute pairwise distances
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            d = box.distance(positions[i], positions[j])
            if d < r_max:
                distances.append(d)
    
    if not distances:
        return np.linspace(0, r_max, num_bins), np.zeros(num_bins)
    
    # Histogram
    r_edges = np.linspace(0, r_max, num_bins + 1)
    hist, _ = np.histogram(distances, bins=r_edges)
    
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = r_edges[1] - r_edges[0]
    
    # Normalize by ideal gas
    rho = n / box.volume()
    dim = network.dimension
    
    if dim == 2:
        ideal = 2 * np.pi * r_centers * dr * rho
    else:
        ideal = 4 * np.pi * r_centers**2 * dr * rho
    
    ideal[ideal < 1e-15] = 1e-15
    g = hist / (n * ideal)
    
    return r_centers, g

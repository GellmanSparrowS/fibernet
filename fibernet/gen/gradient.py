"""
Gradient Network Generator Module

Provides generators for fiber networks with spatially varying properties:
- Density gradient networks
- Property gradient networks
- Multi-zone networks

These are useful for studying functionally graded materials (FGMs),
tissue engineering scaffolds, and other applications requiring
spatial control of network properties.

References:
- Pompe, W. et al., "Functionally graded materials", Materials Science and Engineering A, 2003
- Bohner, M. et al., "Functionally graded materials for biomedical applications", Acta Biomaterialia, 2012
"""

import numpy as np
from typing import Tuple, Optional, List, Callable
import warnings

from fibernet.core.network import FiberNetwork
from fibernet.core.fiber import Fiber
from fibernet.core.material import Material


def density_gradient_2d(
    num_fibers: int = 100,
    fiber_length: float = 10.0,
    box_size: Tuple[float, float] = (50.0, 50.0),
    gradient_direction: str = 'x',
    gradient_profile: str = 'linear',
    min_density: float = 0.5,
    max_density: float = 2.0,
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate a 2D fiber network with density gradient.
    
    Parameters
    ----------
    num_fibers : int
        Base number of fibers.
    fiber_length : float
        Mean fiber length.
    box_size : tuple
        (Lx, Ly) domain size.
    gradient_direction : str
        Direction of gradient: 'x', 'y', or 'radial'.
    gradient_profile : str
        Profile of gradient: 'linear', 'exponential', or 'sinusoidal'.
    min_density : float
        Minimum density factor (relative to base).
    max_density : float
        Maximum density factor (relative to base).
    radius : float
        Fiber radius.
    material : Material, optional
        Fiber material.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    network : FiberNetwork
        Fiber network with density gradient.
    
    Examples
    --------
    >>> from fibernet.gen.gradient import density_gradient_2d
    >>> net = density_gradient_2d(num_fibers=100, gradient_direction='x')
    >>> print(f"Fibers: {net.num_fibers}")
    """
    if seed is not None:
        np.random.seed(seed)
    
    net = FiberNetwork(dimension=2)
    Lx, Ly = box_size
    
    # Define density function
    def get_density(x, y):
        if gradient_direction == 'x':
            t = x / Lx
        elif gradient_direction == 'y':
            t = y / Ly
        elif gradient_direction == 'radial':
            cx, cy = Lx / 2, Ly / 2
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            r_max = np.sqrt(cx**2 + cy**2)
            t = r / r_max
        else:
            t = 0.5
        
        if gradient_profile == 'linear':
            factor = min_density + (max_density - min_density) * t
        elif gradient_profile == 'exponential':
            factor = min_density * np.exp(t * np.log(max_density / min_density))
        elif gradient_profile == 'sinusoidal':
            factor = min_density + (max_density - min_density) * (1 + np.sin(np.pi * t)) / 2
        else:
            factor = (min_density + max_density) / 2
        
        return factor
    
    # Generate fibers with acceptance-rejection sampling
    fiber_id = 0
    attempts = 0
    max_attempts = num_fibers * 20
    
    while fiber_id < num_fibers and attempts < max_attempts:
        attempts += 1
        
        # Random position
        x = np.random.uniform(0, Lx)
        y = np.random.uniform(0, Ly)
        
        # Accept based on local density
        density = get_density(x, y)
        if np.random.uniform(0, max_density) > density:
            continue
        
        # Random orientation
        theta = np.random.uniform(0, 2 * np.pi)
        
        # Fiber endpoints
        length = fiber_length * np.random.uniform(0.8, 1.2)
        dx = length / 2 * np.cos(theta)
        dy = length / 2 * np.sin(theta)
        
        start = np.array([x - dx, y - dy, 0.0])
        end = np.array([x + dx, y + dy, 0.0])
        
        # Clip to box
        start[0] = np.clip(start[0], 0, Lx)
        start[1] = np.clip(start[1], 0, Ly)
        end[0] = np.clip(end[0], 0, Lx)
        end[1] = np.clip(end[1], 0, Ly)
        
        # Add fiber
        net.add_fiber(Fiber.straight(start, end, radius=radius, material=material, fiber_id=fiber_id))
        fiber_id += 1
    
    return net


def property_gradient_2d(
    num_fibers: int = 100,
    fiber_length: float = 10.0,
    box_size: Tuple[float, float] = (50.0, 50.0),
    gradient_direction: str = 'x',
    gradient_profile: str = 'linear',
    min_property: float = 1e9,
    max_property: float = 1e10,
    radius: float = 0.1,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate a 2D fiber network with material property gradient.
    
    Parameters
    ----------
    num_fibers : int
        Number of fibers.
    fiber_length : float
        Mean fiber length.
    box_size : tuple
        (Lx, Ly) domain size.
    gradient_direction : str
        Direction of gradient: 'x', 'y', or 'radial'.
    gradient_profile : str
        Profile of gradient: 'linear', 'exponential', or 'sinusoidal'.
    min_property : float
        Minimum Young's modulus (Pa).
    max_property : float
        Maximum Young's modulus (Pa).
    radius : float
        Fiber radius.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    network : FiberNetwork
        Fiber network with property gradient.
    
    Examples
    --------
    >>> from fibernet.gen.gradient import property_gradient_2d
    >>> net = property_gradient_2d(num_fibers=100, min_property=1e9, max_property=1e10)
    >>> print(f"Fibers: {net.num_fibers}")
    """
    if seed is not None:
        np.random.seed(seed)
    
    net = FiberNetwork(dimension=2)
    Lx, Ly = box_size
    
    # Define property function
    def get_property(x, y):
        if gradient_direction == 'x':
            t = x / Lx
        elif gradient_direction == 'y':
            t = y / Ly
        elif gradient_direction == 'radial':
            cx, cy = Lx / 2, Ly / 2
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            r_max = np.sqrt(cx**2 + cy**2)
            t = r / r_max
        else:
            t = 0.5
        
        if gradient_profile == 'linear':
            E = min_property + (max_property - min_property) * t
        elif gradient_profile == 'exponential':
            E = min_property * np.exp(t * np.log(max_property / min_property))
        elif gradient_profile == 'sinusoidal':
            E = min_property + (max_property - min_property) * (1 + np.sin(np.pi * t)) / 2
        else:
            E = (min_property + max_property) / 2
        
        return E
    
    # Generate fibers
    for fiber_id in range(num_fibers):
        # Random position
        x = np.random.uniform(0, Lx)
        y = np.random.uniform(0, Ly)
        
        # Get local property
        E = get_property(x, y)
        
        # Create material with local property
        mat = Material(
            youngs_modulus=E,
            shear_modulus=E / (2 * (1 + 0.3)),  # Assume Poisson's ratio 0.3
            name=f"material_{fiber_id}"
        )
        
        # Random orientation
        theta = np.random.uniform(0, 2 * np.pi)
        
        # Fiber endpoints
        length = fiber_length * np.random.uniform(0.8, 1.2)
        dx = length / 2 * np.cos(theta)
        dy = length / 2 * np.sin(theta)
        
        start = np.array([x - dx, y - dy, 0.0])
        end = np.array([x + dx, y + dy, 0.0])
        
        # Clip to box
        start[0] = np.clip(start[0], 0, Lx)
        start[1] = np.clip(start[1], 0, Ly)
        end[0] = np.clip(end[0], 0, Lx)
        end[1] = np.clip(end[1], 0, Ly)
        
        # Add fiber
        net.add_fiber(Fiber.straight(start, end, radius=radius, material=mat, fiber_id=fiber_id))
    
    return net


def multi_zone_2d(
    zones: List[dict],
    box_size: Tuple[float, float] = (50.0, 50.0),
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate a 2D fiber network with multiple zones.
    
    Parameters
    ----------
    zones : list of dict
        List of zone specifications. Each zone is a dict with:
        - 'region': tuple of (x, y, width, height) for rectangular region
        - 'num_fibers': number of fibers in this zone
        - 'fiber_length': mean fiber length
        - 'radius': fiber radius (optional)
        - 'material': Material object (optional)
    box_size : tuple
        (Lx, Ly) domain size.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    network : FiberNetwork
        Multi-zone fiber network.
    
    Examples
    --------
    >>> from fibernet.gen.gradient import multi_zone_2d
    >>> zones = [
    ...     {'region': (0, 0, 25, 50), 'num_fibers': 50, 'fiber_length': 8.0},
    ...     {'region': (25, 0, 25, 50), 'num_fibers': 50, 'fiber_length': 12.0},
    ... ]
    >>> net = multi_zone_2d(zones, box_size=(50, 50))
    >>> print(f"Fibers: {net.num_fibers}")
    """
    if seed is not None:
        np.random.seed(seed)
    
    net = FiberNetwork(dimension=2)
    fiber_id = 0
    
    for zone in zones:
        x0, y0, width, height = zone['region']
        num_fibers = zone['num_fibers']
        fiber_length = zone['fiber_length']
        radius = zone.get('radius', 0.1)
        material = zone.get('material', None)
        
        # Generate fibers in this zone
        for _ in range(num_fibers):
            # Random position within zone
            x = np.random.uniform(x0, x0 + width)
            y = np.random.uniform(y0, y0 + height)
            
            # Random orientation
            theta = np.random.uniform(0, 2 * np.pi)
            
            # Fiber endpoints
            length = fiber_length * np.random.uniform(0.8, 1.2)
            dx = length / 2 * np.cos(theta)
            dy = length / 2 * np.sin(theta)
            
            start = np.array([x - dx, y - dy, 0.0])
            end = np.array([x + dx, y + dy, 0.0])
            
            # Clip to zone
            start[0] = np.clip(start[0], x0, x0 + width)
            start[1] = np.clip(start[1], y0, y0 + height)
            end[0] = np.clip(end[0], x0, x0 + width)
            end[1] = np.clip(end[1], y0, y0 + height)
            
            # Add fiber
            net.add_fiber(Fiber.straight(start, end, radius=radius, material=material, fiber_id=fiber_id))
            fiber_id += 1
    
    return net



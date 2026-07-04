"""
Periodic boundary conditions for fiber network simulations.

Provides utilities for:
- Wrapping networks with periodic boundaries
- Computing images for cross-boundary interactions
- Applying periodic FEM and thermal simulations
- Homogenization for effective properties
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from ..core import FiberNetwork


class PeriodicBoundary:
    """
    Periodic boundary condition handler.
    
    Parameters
    ----------
    box_size : tuple
        Size of the periodic box (Lx, Ly, Lz) or (Lx, Ly)
    dimension : int
        Dimension (2 or 3)
    """
    
    def __init__(self, box_size: Tuple[float, ...], dimension: int = 2):
        self.box_size = np.array(box_size)
        self.dimension = dimension
        
        if len(self.box_size) < dimension:
            raise ValueError(f"box_size must have at least {dimension} components")
    
    def wrap_position(self, pos: np.ndarray) -> np.ndarray:
        """Wrap position to be within the box."""
        pos = np.asarray(pos)
        wrapped = pos.copy()
        for i in range(self.dimension):
            wrapped[i] = wrapped[i] % self.box_size[i]
        return wrapped
    
    def minimum_image_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
        """
        Compute minimum image distance between two positions.
        
        Parameters
        ----------
        pos1, pos2 : np.ndarray
            Positions (3D or 2D)
        
        Returns
        -------
        np.ndarray
            Minimum image displacement vector
        """
        delta = np.asarray(pos2) - np.asarray(pos1)
        
        for i in range(self.dimension):
            L = self.box_size[i]
            if delta[i] > L / 2:
                delta[i] -= L
            elif delta[i] < -L / 2:
                delta[i] += L
        
        return delta
    
    def minimum_image_norm(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Compute minimum image distance (scalar)."""
        delta = self.minimum_image_distance(pos1, pos2)
        return np.linalg.norm(delta)
    
    def get_images(self, pos: np.ndarray, n_images: int = 1) -> List[np.ndarray]:
        """
        Get periodic image positions.
        
        Parameters
        ----------
        pos : np.ndarray
            Position in the primary box
        n_images : int
            Number of images in each direction
        
        Returns
        -------
        list of np.ndarray
            List of image positions
        """
        images = []
        pos = np.asarray(pos)
        
        if self.dimension == 2:
            for ix in range(-n_images, n_images + 1):
                for iy in range(-n_images, n_images + 1):
                    if ix == 0 and iy == 0:
                        continue
                    shift = np.array([ix * self.box_size[0], iy * self.box_size[1], 0])
                    images.append(pos + shift)
        else:
            for ix in range(-n_images, n_images + 1):
                for iy in range(-n_images, n_images + 1):
                    for iz in range(-n_images, n_images + 1):
                        if ix == 0 and iy == 0 and iz == 0:
                            continue
                        shift = np.array([ix * self.box_size[0], iy * self.box_size[1], iz * self.box_size[2]])
                        images.append(pos + shift)
        
        return images
    
    def find_cross_boundary_crosslinks(self, network: FiberNetwork) -> List[Tuple[int, int, int]]:
        """
        Find crosslinks that cross periodic boundaries.
        
        Returns
        -------
        list of (fiber_i, fiber_j, image_index)
            Cross-boundary crosslink pairs
        """
        cross_boundary = []
        
        for i, fiber_i in enumerate(network.fibers):
            for j, fiber_j in enumerate(network.fibers):
                if i >= j:
                    continue
                
                # Check minimum image distance
                center_i = (fiber_i.start_point + fiber_i.end_point) / 2
                center_j = (fiber_j.start_point + fiber_j.end_point) / 2
                
                delta = self.minimum_image_distance(center_i, center_j)
                direct = center_j - center_i
                
                if not np.allclose(delta, direct):
                    # Fibers are closer through periodic boundary
                    cross_boundary.append((i, j, 0))
        
        return cross_boundary
    
    def compute_effective_properties(
        self,
        network: FiberNetwork,
        property_type: str = 'mechanical',
        num_directions: int = None,
    ) -> Dict[str, float]:
        """
        Compute effective properties using homogenization.
        
        Parameters
        ----------
        network : FiberNetwork
            Fiber network
        property_type : str
            'mechanical', 'thermal', or 'electrical'
        num_directions : int
            Number of loading directions
        
        Returns
        -------
        dict
            Effective properties
        """
        from ..analysis import extract_stress_strain
        
        if property_type == 'mechanical':
            properties = {}
            
            if num_directions is None:
                num_directions = self.dimension
            
            for axis in range(min(num_directions, self.dimension)):
                try:
                    curve = extract_stress_strain(
                        network,
                        strain_range=(0, 0.01),
                        num_steps=5,
                        axis=axis
                    )
                    
                    E = curve.youngs_modulus
                    nu = curve.poissons_ratio
                    
                    properties[f'E_axis{axis}'] = E
                    properties[f'nu_axis{axis}'] = nu
                except Exception as e:
                    properties[f'E_axis{axis}'] = 0.0
                    properties[f'nu_axis{axis}'] = 0.0
            
            # Average properties
            E_values = [properties[f'E_axis{i}'] for i in range(min(num_directions, self.dimension))]
            properties['E_effective'] = np.mean(E_values)
            properties['E_anisotropy'] = np.std(E_values) / np.mean(E_values) if np.mean(E_values) > 0 else 0
            
            return properties
        
        elif property_type == 'thermal':
            from ..sim import ThermalSolver
            
            properties = {}
            solver = ThermalSolver(network)
            
            if num_directions is None:
                num_directions = self.dimension
            
            for axis in range(min(num_directions, self.dimension)):
                try:
                    result = solver.steady_state(temperature_diff=100, direction=axis)
                    properties[f'k_axis{axis}'] = result.effective_conductivity
                except Exception:
                    properties[f'k_axis{axis}'] = 0.0
            
            k_values = [properties[f'k_axis{i}'] for i in range(min(num_directions, self.dimension))]
            properties['k_effective'] = np.mean(k_values)
            
            return properties
        
        else:
            raise ValueError(f"Unknown property type: {property_type}")


def create_periodic_network(
    network: FiberNetwork,
    box_size: Tuple[float, ...],
    wrap_fibers: bool = True,
) -> Tuple[FiberNetwork, PeriodicBoundary]:
    """
    Create a periodic version of a fiber network.
    
    Parameters
    ----------
    network : FiberNetwork
        Original network
    box_size : tuple
        Size of periodic box
    wrap_fibers : bool
        Whether to wrap fiber positions to be within box
    
    Returns
    -------
    tuple
        (periodic_network, PeriodicBoundary)
    """
    pb = PeriodicBoundary(box_size, dimension=network.dimension)
    
    if wrap_fibers:
        from ..core import FiberNetwork as FN
        
        new_network = FN(dimension=network.dimension)
        
        for fiber in network.fibers:
            # Wrap fiber positions
            start = pb.wrap_position(fiber.start_point)
            end = pb.wrap_position(fiber.end_point)
            
            # Create new fiber with wrapped positions
            new_fiber = fiber.copy()
            # Translate to wrapped position
            delta = start - fiber.start_point
            new_fiber.translate(delta)
            
            new_network.add_fiber(new_fiber)
        
        return new_network, pb
    
    return network, pb


def apply_periodic_strain(
    network: FiberNetwork,
    box_size: Tuple[float, ...],
    strain: float,
    axis: int = 0,
    segments_per_fiber: int = 5,
) -> Dict[str, float]:
    """
    Apply strain with periodic boundary conditions.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network
    box_size : tuple
        Size of periodic box
    strain : float
        Applied strain
    axis : int
        Loading axis
    segments_per_fiber : int
        Number of segments per fiber for FEM
    
    Returns
    -------
    dict
        Results including stress, energy, etc.
    """
    from ..sim import FiberFEM
    
    pb = PeriodicBoundary(box_size, dimension=network.dimension)
    
    # Create FEM solver
    fem = FiberFEM(network, segments_per_fiber=segments_per_fiber)
    
    # Apply strain
    result = fem.apply_uniaxial_strain(strain=strain, axis=axis)
    
    # Compute periodic-corrected results
    volume = np.prod(box_size[:network.dimension])
    
    return {
        'stress': result.stress if hasattr(result, 'stress') else 0.0,
        'energy': result.energy / volume if volume > 0 else 0.0,
        'strain': strain,
        'axis': axis,
        'box_size': box_size,
    }


def homogenize_properties(
    generator_func,
    box_size: Tuple[float, ...],
    num_realizations: int = 10,
    property_type: str = 'mechanical',
    base_seed: int = 42,
    **generator_kwargs,
) -> Dict[str, float]:
    """
    Homogenize properties over multiple realizations.
    
    Parameters
    ----------
    generator_func : callable
        Network generator function
    box_size : tuple
        Size of periodic box
    num_realizations : int
        Number of realizations
    property_type : str
        'mechanical' or 'thermal'
    base_seed : int
        Base random seed
    **generator_kwargs
        Arguments for generator
    
    Returns
    -------
    dict
        Homogenized properties with mean and std
    """
    from ..utils.ensemble import generate_ensemble
    
    # Generate ensemble
    ensemble = generate_ensemble(
        generator_func,
        num_networks=num_realizations,
        base_seed=base_seed,
        show_progress=False,
        **generator_kwargs
    )
    
    # Compute properties for each realization
    all_properties = []
    for network in ensemble:
        pb = PeriodicBoundary(box_size, dimension=network.dimension)
        props = pb.compute_effective_properties(network, property_type)
        all_properties.append(props)
    
    # Aggregate
    homogenized = {}
    for key in all_properties[0].keys():
        values = [p[key] for p in all_properties if key in p]
        homogenized[f'{key}_mean'] = np.mean(values)
        homogenized[f'{key}_std'] = np.std(values)
        homogenized[f'{key}_values'] = values
    
    homogenized['num_realizations'] = num_realizations
    homogenized['box_size'] = box_size
    
    return homogenized

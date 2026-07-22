"""
Composite Laminate Generators

Generate multi-layered composite fiber structures:
- Unidirectional laminates (UD)
- Cross-ply laminates ([0/90])
- Angle-ply laminates ([+θ/-θ])
- Quasi-isotropic laminates ([0/±45/90])
- Sandwich structures (face sheets + core)
- Woven fabric laminates

References:
- Jones, R.M. "Mechanics of Composite Materials", CRC Press, 2019
- Kaw, A.K. "Mechanics of Composite Materials", CRC Press, 2006
- Halpin & Tsai, "Effects of environmental factors on composites", 1969
"""

import numpy as np
from math import comb
from typing import Tuple, Optional, List
from fibernet.core.network import FiberNetwork
from fibernet.core.fiber import Fiber
from fibernet.core.material import Material



def _add_laminate_crosslinks(net, tie_spacing=8.0, tie_radius=0.05):
    """Add transverse cross-tie fibers to connect laminate layers."""
    import numpy as np
    from fibernet.core.fiber import Fiber
    from fibernet.core.material import Material
    
    if net.num_fibers == 0:
        return
    
    all_pts = np.vstack([f.centerline for f in net.fibers])
    pmin = all_pts.min(axis=0)
    pmax = all_pts.max(axis=0)
    
    mat = net.fibers[0].material or Material(name="cross_tie")
    fiber_id = net.num_fibers
    
    if net.dimension == 2:
        # Add vertical ties
        for x in np.arange(pmin[0] + tie_spacing, pmax[0], tie_spacing):
            fiber = Fiber.straight(
                np.array([x, pmin[1] - 0.5, 0.0]),
                np.array([x, pmax[1] + 0.5, 0.0]),
                radius=tie_radius, material=mat, fiber_id=fiber_id
            )
            net.add_fiber(fiber)
            fiber_id += 1
    else:
        # Add z-direction ties
        z_range = pmax[2] - pmin[2]
        if z_range > 0.1:
            for x in np.arange(pmin[0] + tie_spacing, pmax[0], tie_spacing):
                for y in np.arange(pmin[1] + tie_spacing, pmax[1], tie_spacing):
                    fiber = Fiber.straight(
                        np.array([x, y, pmin[2] - 0.5]),
                        np.array([x, y, pmax[2] + 0.5]),
                        radius=tie_radius, material=mat, fiber_id=fiber_id
                    )
                    net.add_fiber(fiber)
                    fiber_id += 1
    
    net.auto_crosslink(threshold=tie_spacing * 0.3)


def unidirectional_laminate(
    num_layers: int = 4,
    fibers_per_layer: int = 20,
    layer_thickness: float = 1.0,
    fiber_length: float = 50.0,
    fiber_radius: float = 0.1,
    fiber_spacing: float = 0.5,
    orientation: float = 0.0,
    box_width: float = 50.0,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """
    Generate a unidirectional (UD) composite laminate.
    
    All layers have fibers oriented in the same direction.
    This is the simplest and strongest laminate configuration.
    
    Parameters
    ----------
    num_layers : int
        Number of layers (plies)
    fibers_per_layer : int
        Number of fibers per layer
    layer_thickness : float
        Thickness of each layer
    fiber_length : float
        Length of fibers
    fiber_radius : float
        Fiber cross-section radius
    fiber_spacing : float
        Spacing between fibers
    orientation : float
        Fiber orientation angle (radians, 0 = x-axis)
    box_width : float
        Width of simulation box
    material : Material, optional
        Fiber material
    seed : int, optional
        Random seed
    
    Returns
    -------
    FiberNetwork
        3D network of UD laminate
    
    Examples
    --------
    >>> net = unidirectional_laminate(num_layers=4, fibers_per_layer=20)
    """
    if seed is not None:
        np.random.seed(seed)
    
    net = FiberNetwork(dimension=3)
    
    # Direction vectors
    direction = np.array([np.cos(orientation), np.sin(orientation), 0.0])
    perp = np.array([-np.sin(orientation), np.cos(orientation), 0.0])
    
    fiber_id = 0
    for layer in range(num_layers):
        z_offset = layer * layer_thickness
        
        for i in range(fibers_per_layer):
            # Position across layer width
            offset = (i / max(fibers_per_layer - 1, 1) - 0.5) * box_width
            
            # Center of fiber
            center = np.array([box_width / 2, box_width / 2, z_offset])
            center += offset * perp
            
            # Fiber endpoints
            start = center - 0.5 * fiber_length * direction
            end = center + 0.5 * fiber_length * direction
            
            fiber = Fiber.straight(
                start, end,
                radius=fiber_radius,
                material=material,
                fiber_id=fiber_id
            )
            net.add_fiber(fiber)
            fiber_id += 1
    
    # Add cross-ties and crosslinks
    _add_laminate_crosslinks(net)

    # Ensure connected
    from fibernet.gen.disordered import _ensure_connected
    _ensure_connected(net, max_gap_factor=5.0)
    
    return net


def crossply_laminate(
    num_layers: int = 4,
    fibers_per_layer: int = 20,
    layer_thickness: float = 1.0,
    fiber_length: float = 50.0,
    fiber_radius: float = 0.1,
    box_width: float = 50.0,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """
    Generate a cross-ply laminate [0/90/0/90/...].
    
    Alternating layers with 0° and 90° fiber orientations.
    Common for bidirectional reinforcement.
    
    Parameters
    ----------
    num_layers : int
        Number of layers (should be even for symmetric laminate)
    fibers_per_layer : int
        Number of fibers per layer
    layer_thickness : float
        Thickness of each layer
    fiber_length : float
        Length of fibers
    fiber_radius : float
        Fiber cross-section radius
    box_width : float
        Width of simulation box
    material : Material, optional
        Fiber material
    seed : int, optional
        Random seed
    
    Returns
    -------
    FiberNetwork
        3D network of cross-ply laminate
    
    Examples
    --------
    >>> net = crossply_laminate(num_layers=4, fibers_per_layer=20)
    """
    if seed is not None:
        np.random.seed(seed)
    
    net = FiberNetwork(dimension=3)
    
    fiber_id = 0
    for layer in range(num_layers):
        z_offset = layer * layer_thickness
        
        # Alternate between 0° and 90°
        orientation = 0.0 if layer % 2 == 0 else np.pi / 2
        
        direction = np.array([np.cos(orientation), np.sin(orientation), 0.0])
        perp = np.array([-np.sin(orientation), np.cos(orientation), 0.0])
        
        for i in range(fibers_per_layer):
            offset = (i / max(fibers_per_layer - 1, 1) - 0.5) * box_width
            
            center = np.array([box_width / 2, box_width / 2, z_offset])
            center += offset * perp
            
            start = center - 0.5 * fiber_length * direction
            end = center + 0.5 * fiber_length * direction
            
            fiber = Fiber.straight(
                start, end,
                radius=fiber_radius,
                material=material,
                fiber_id=fiber_id
            )
            net.add_fiber(fiber)
            fiber_id += 1
    
    # Add cross-ties and crosslinks
    _add_laminate_crosslinks(net)

    return net


def angle_ply_laminate(
    num_layers: int = 4,
    angle: float = np.pi / 4,
    fibers_per_layer: int = 20,
    layer_thickness: float = 1.0,
    fiber_length: float = 50.0,
    fiber_radius: float = 0.1,
    box_width: float = 50.0,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """
    Generate an angle-ply laminate [+θ/-θ/+θ/-θ/...].
    
    Alternating layers with +θ and -θ orientations.
    Common for shear-resistant structures.
    
    Parameters
    ----------
    num_layers : int
        Number of layers
    angle : float
        Ply angle (radians)
    fibers_per_layer : int
        Number of fibers per layer
    layer_thickness : float
        Thickness of each layer
    fiber_length : float
        Length of fibers
    fiber_radius : float
        Fiber cross-section radius
    box_width : float
        Width of simulation box
    material : Material, optional
        Fiber material
    seed : int, optional
        Random seed
    
    Returns
    -------
    FiberNetwork
        3D network of angle-ply laminate
    
    Examples
    --------
    >>> net = angle_ply_laminate(num_layers=4, angle=np.pi/4)  # ±45°
    """
    if seed is not None:
        np.random.seed(seed)
    
    net = FiberNetwork(dimension=3)
    
    fiber_id = 0
    for layer in range(num_layers):
        z_offset = layer * layer_thickness
        
        # Alternate between +θ and -θ
        orientation = angle if layer % 2 == 0 else -angle
        
        direction = np.array([np.cos(orientation), np.sin(orientation), 0.0])
        perp = np.array([-np.sin(orientation), np.cos(orientation), 0.0])
        
        for i in range(fibers_per_layer):
            offset = (i / max(fibers_per_layer - 1, 1) - 0.5) * box_width
            
            center = np.array([box_width / 2, box_width / 2, z_offset])
            center += offset * perp
            
            start = center - 0.5 * fiber_length * direction
            end = center + 0.5 * fiber_length * direction
            
            fiber = Fiber.straight(
                start, end,
                radius=fiber_radius,
                material=material,
                fiber_id=fiber_id
            )
            net.add_fiber(fiber)
            fiber_id += 1
    
    # Add cross-ties and crosslinks
    _add_laminate_crosslinks(net)

    # Ensure connected
    from fibernet.gen.disordered import _ensure_connected
    _ensure_connected(net, max_gap_factor=5.0)
    
    return net


def quasi_isotropic_laminate(
    num_fibers_per_layer: int = 20,
    layer_thickness: float = 1.0,
    fiber_length: float = 50.0,
    fiber_radius: float = 0.1,
    box_width: float = 50.0,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """
    Generate a quasi-isotropic laminate [0/±45/90].
    
    Standard quasi-isotropic stacking sequence with 4 layers:
    - Layer 1: 0°
    - Layer 2: +45°
    - Layer 3: -45°
    - Layer 4: 90°
    
    Provides approximately isotropic in-plane properties.
    
    Parameters
    ----------
    num_fibers_per_layer : int
        Number of fibers per layer
    layer_thickness : float
        Thickness of each layer
    fiber_length : float
        Length of fibers
    fiber_radius : float
        Fiber cross-section radius
    box_width : float
        Width of simulation box
    material : Material, optional
        Fiber material
    seed : int, optional
        Random seed
    
    Returns
    -------
    FiberNetwork
        3D network of quasi-isotropic laminate
    
    Examples
    --------
    >>> net = quasi_isotropic_laminate(num_fibers_per_layer=20)
    """
    if seed is not None:
        np.random.seed(seed)
    
    net = FiberNetwork(dimension=3)
    
    # Standard quasi-isotropic stacking: [0/+45/-45/90]
    angles = [0.0, np.pi / 4, -np.pi / 4, np.pi / 2]
    
    fiber_id = 0
    for layer, orientation in enumerate(angles):
        z_offset = layer * layer_thickness
        
        direction = np.array([np.cos(orientation), np.sin(orientation), 0.0])
        perp = np.array([-np.sin(orientation), np.cos(orientation), 0.0])
        
        for i in range(num_fibers_per_layer):
            offset = (i / max(num_fibers_per_layer - 1, 1) - 0.5) * box_width
            
            center = np.array([box_width / 2, box_width / 2, z_offset])
            center += offset * perp
            
            start = center - 0.5 * fiber_length * direction
            end = center + 0.5 * fiber_length * direction
            
            fiber = Fiber.straight(
                start, end,
                radius=fiber_radius,
                material=material,
                fiber_id=fiber_id
            )
            net.add_fiber(fiber)
            fiber_id += 1
    
    # Add cross-ties and crosslinks
    _add_laminate_crosslinks(net)

    # Ensure connected
    from fibernet.gen.disordered import _ensure_connected
    _ensure_connected(net, max_gap_factor=5.0)
    
    return net


def custom_laminate(
    stacking_sequence: List[float],
    fibers_per_layer: int = 20,
    layer_thickness: float = 1.0,
    fiber_length: float = 50.0,
    fiber_radius: float = 0.1,
    box_width: float = 50.0,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """
    Generate a custom laminate with arbitrary stacking sequence.
    
    Parameters
    ----------
    stacking_sequence : list of float
        List of layer orientations in radians
        e.g., [0, np.pi/4, -np.pi/4, np.pi/2] for quasi-isotropic
    fibers_per_layer : int
        Number of fibers per layer
    layer_thickness : float
        Thickness of each layer
    fiber_length : float
        Length of fibers
    fiber_radius : float
        Fiber cross-section radius
    box_width : float
        Width of simulation box
    material : Material, optional
        Fiber material
    seed : int, optional
        Random seed
    
    Returns
    -------
    FiberNetwork
        3D network with custom stacking
    
    Examples
    --------
    >>> # Custom [0/30/60/90] laminate
    >>> angles = [0, np.pi/6, np.pi/3, np.pi/2]
    >>> net = custom_laminate(angles, fibers_per_layer=20)
    >>> 
    >>> # Symmetric [0/45/45/0] laminate
    >>> angles = [0, np.pi/4, np.pi/4, 0]
    >>> net = custom_laminate(angles)
    """
    if seed is not None:
        np.random.seed(seed)
    
    net = FiberNetwork(dimension=3)
    
    fiber_id = 0
    for layer, orientation in enumerate(stacking_sequence):
        z_offset = layer * layer_thickness
        
        direction = np.array([np.cos(orientation), np.sin(orientation), 0.0])
        perp = np.array([-np.sin(orientation), np.cos(orientation), 0.0])
        
        for i in range(fibers_per_layer):
            offset = (i / max(fibers_per_layer - 1, 1) - 0.5) * box_width
            
            center = np.array([box_width / 2, box_width / 2, z_offset])
            center += offset * perp
            
            start = center - 0.5 * fiber_length * direction
            end = center + 0.5 * fiber_length * direction
            
            fiber = Fiber.straight(
                start, end,
                radius=fiber_radius,
                material=material,
                fiber_id=fiber_id
            )
            net.add_fiber(fiber)
            fiber_id += 1
    
    return net


def sandwich_laminate(
    face_fibers_per_layer: int = 20,
    num_face_layers: int = 2,
    core_thickness: float = 5.0,
    face_thickness: float = 1.0,
    fiber_length: float = 50.0,
    fiber_radius: float = 0.1,
    box_width: float = 50.0,
    face_material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """
    Generate a sandwich structure (face sheets + core).
    
    Sandwich structures have thin, stiff face sheets bonded to a thick,
    lightweight core. Common in aerospace and marine applications.
    
    Parameters
    ----------
    face_fibers_per_layer : int
        Number of fibers per face sheet layer
    num_face_layers : int
        Number of layers in each face sheet
    core_thickness : float
        Thickness of the core region
    face_thickness : float
        Thickness of each face sheet layer
    fiber_length : float
        Length of fibers
    fiber_radius : float
        Fiber cross-section radius
    box_width : float
        Width of simulation box
    face_material : Material, optional
        Face sheet material
    seed : int, optional
        Random seed
    
    Returns
    -------
    FiberNetwork
        3D network of sandwich structure
    
    Examples
    --------
    >>> net = sandwich_laminate(
    ...     face_fibers_per_layer=20,
    ...     core_thickness=10.0
    ... )
    """
    if seed is not None:
        np.random.seed(seed)
    
    net = FiberNetwork(dimension=3)
    
    fiber_id = 0
    
    # Bottom face sheet
    for layer in range(num_face_layers):
        z_offset = layer * face_thickness
        orientation = 0.0 if layer % 2 == 0 else np.pi / 2
        
        direction = np.array([np.cos(orientation), np.sin(orientation), 0.0])
        perp = np.array([-np.sin(orientation), np.cos(orientation), 0.0])
        
        for i in range(face_fibers_per_layer):
            offset = (i / max(face_fibers_per_layer - 1, 1) - 0.5) * box_width
            
            center = np.array([box_width / 2, box_width / 2, z_offset])
            center += offset * perp
            
            start = center - 0.5 * fiber_length * direction
            end = center + 0.5 * fiber_length * direction
            
            fiber = Fiber.straight(
                start, end,
                radius=fiber_radius,
                material=face_material,
                fiber_id=fiber_id
            )
            net.add_fiber(fiber)
            fiber_id += 1
    
    # Top face sheet
    z_base = num_face_layers * face_thickness + core_thickness
    for layer in range(num_face_layers):
        z_offset = z_base + layer * face_thickness
        orientation = 0.0 if layer % 2 == 0 else np.pi / 2
        
        direction = np.array([np.cos(orientation), np.sin(orientation), 0.0])
        perp = np.array([-np.sin(orientation), np.cos(orientation), 0.0])
        
        for i in range(face_fibers_per_layer):
            offset = (i / max(face_fibers_per_layer - 1, 1) - 0.5) * box_width
            
            center = np.array([box_width / 2, box_width / 2, z_offset])
            center += offset * perp
            
            start = center - 0.5 * fiber_length * direction
            end = center + 0.5 * fiber_length * direction
            
            fiber = Fiber.straight(
                start, end,
                radius=fiber_radius,
                material=face_material,
                fiber_id=fiber_id
            )
            net.add_fiber(fiber)
            fiber_id += 1
    
    # Add cross-ties and crosslinks
    _add_laminate_crosslinks(net)

    return net



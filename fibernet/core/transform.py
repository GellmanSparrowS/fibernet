"""
Network transformation operations.

Provides powerful tools for manipulating fiber networks:
- Mirror/reflect
- Rotate (arbitrary axis)
- Scale
- Merge/combine with anchor alignment
- Tile/repeat (periodic structures)
- Trim to bounding box
- Duplicate with transformations
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from copy import deepcopy
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.utils.geometry import rotation_matrix_axis_angle


def mirror(
    network: FiberNetwork,
    axis: Union[int, str] = 0,
    origin: Optional[np.ndarray] = None,
    inplace: bool = False,
) -> FiberNetwork:
    """Mirror/reflect network about a plane.
    
    Parameters
    ----------
    network : FiberNetwork
        Input network.
    axis : int or str
        Normal axis of mirror plane: 0=x, 1=y, 2=z, or 'x', 'y', 'z'.
    origin : np.ndarray, optional
        Point on the mirror plane. Defaults to network center.
    inplace : bool
        If True, modify network in place. Otherwise return new network.
    
    Returns
    -------
    FiberNetwork
        Mirrored network.
    """
    net = network if inplace else deepcopy(network)
    
    if isinstance(axis, str):
        axis = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
    
    if origin is None:
        bb_min, bb_max = net.bounding_box()
        origin = 0.5 * (bb_min + bb_max)
    
    origin = np.asarray(origin)
    
    for fiber in net.fibers:
        fiber.centerline[:, axis] = 2 * origin[axis] - fiber.centerline[:, axis]
    
    for cl in net.crosslinks:
        cl.position[axis] = 2 * origin[axis] - cl.position[axis]
    
    return net


def rotate(
    network: FiberNetwork,
    angle: float,
    axis: np.ndarray,
    origin: Optional[np.ndarray] = None,
    inplace: bool = False,
) -> FiberNetwork:
    """Rotate network around arbitrary axis.
    
    Parameters
    ----------
    network : FiberNetwork
        Input network.
    angle : float
        Rotation angle in radians.
    axis : np.ndarray
        Rotation axis (3D vector).
    origin : np.ndarray, optional
        Rotation center. Defaults to network center.
    inplace : bool
        If True, modify network in place.
    
    Returns
    -------
    FiberNetwork
        Rotated network.
    """
    net = network if inplace else deepcopy(network)
    
    if origin is None:
        bb_min, bb_max = net.bounding_box()
        origin = 0.5 * (bb_min + bb_max)
    
    origin = np.asarray(origin)
    R = rotation_matrix_axis_angle(axis, angle)
    
    for fiber in net.fibers:
        centered = fiber.centerline - origin
        fiber.centerline = (R @ centered.T).T + origin
    
    for cl in net.crosslinks:
        centered = cl.position - origin
        cl.position = (R @ centered.T) + origin
    
    return net


def scale(
    network: FiberNetwork,
    factor: Union[float, np.ndarray],
    origin: Optional[np.ndarray] = None,
    inplace: bool = False,
) -> FiberNetwork:
    """Scale network uniformly or anisotropically.
    
    Parameters
    ----------
    network : FiberNetwork
        Input network.
    factor : float or np.ndarray
        Scale factor(s). Scalar for uniform, array for anisotropic.
    origin : np.ndarray, optional
        Scaling center. Defaults to network center.
    inplace : bool
        If True, modify network in place.
    
    Returns
    -------
    FiberNetwork
        Scaled network.
    """
    net = network if inplace else deepcopy(network)
    
    if origin is None:
        bb_min, bb_max = net.bounding_box()
        origin = 0.5 * (bb_min + bb_max)
    
    origin = np.asarray(origin)
    factor = np.asarray(factor)
    
    for fiber in net.fibers:
        centered = fiber.centerline - origin
        fiber.centerline = centered * factor + origin
        if np.isscalar(factor):
            fiber.radius *= factor
        else:
            fiber.radius *= np.mean(factor)
    
    for cl in net.crosslinks:
        centered = cl.position - origin
        cl.position = centered * factor + origin
    
    if net.box_size is not None:
        net.box_size = net.box_size * factor
    
    return net


def translate(
    network: FiberNetwork,
    offset: np.ndarray,
    inplace: bool = False,
) -> FiberNetwork:
    """Translate network by offset vector.
    
    Parameters
    ----------
    network : FiberNetwork
        Input network.
    offset : np.ndarray
        Translation vector.
    inplace : bool
        If True, modify network in place.
    
    Returns
    -------
    FiberNetwork
        Translated network.
    """
    net = network if inplace else deepcopy(network)
    offset = np.asarray(offset)
    
    for fiber in net.fibers:
        fiber.centerline += offset
    
    for cl in net.crosslinks:
        cl.position += offset
    
    return net


def merge(
    networks: List[FiberNetwork],
    offsets: Optional[List[np.ndarray]] = None,
    merge_crosslinks: bool = True,
    proximity_threshold: float = None,
) -> FiberNetwork:
    """Merge multiple networks into one.
    
    Parameters
    ----------
    networks : list of FiberNetwork
        Networks to merge.
    offsets : list of np.ndarray, optional
        Translation offset for each network before merging.
    merge_crosslinks : bool
        If True, detect and add crosslinks between networks.
    proximity_threshold : float, optional
        Distance threshold for crosslink detection.
    
    Returns
    -------
    FiberNetwork
        Merged network.
    """
    if not networks:
        return FiberNetwork()
    
    merged = FiberNetwork(
        dimension=networks[0].dimension,
        periodic=any(n.periodic for n in networks),
    )
    
    fiber_offset = 0
    
    for i, net in enumerate(networks):
        offset = offsets[i] if offsets and i < len(offsets) else np.zeros(3)
        
        for fiber in net.fibers:
            new_fiber = deepcopy(fiber)
            new_fiber.centerline += offset
            new_fiber.fiber_id = fiber_offset + fiber.fiber_id
            merged.add_fiber(new_fiber)
        
        for cl in net.crosslinks:
            new_cl = Crosslink(
                fiber_i=fiber_offset + cl.fiber_i,
                fiber_j=fiber_offset + cl.fiber_j,
                param_i=cl.param_i,
                param_j=cl.param_j,
                position=cl.position + offset,
                crosslink_type=cl.crosslink_type,
                strength=cl.strength,
                stiffness=cl.stiffness,
            )
            merged.add_crosslink(new_cl)
        
        fiber_offset += net.num_fibers
    
    if merge_crosslinks and proximity_threshold:
        merged.auto_crosslink(threshold=proximity_threshold)
    
    return merged


def tile(
    network: FiberNetwork,
    repeats: Tuple[int, int, int] = (2, 2, 2),
    spacing: Optional[np.ndarray] = None,
) -> FiberNetwork:
    """Tile/repeat network to create periodic structure.
    
    Parameters
    ----------
    network : FiberNetwork
        Base unit cell.
    repeats : tuple of int
        Number of repetitions in x, y, z directions.
    spacing : np.ndarray, optional
        Spacing between tiles. Defaults to network bounding box size.
    
    Returns
    -------
    FiberNetwork
        Tiled network.
    """
    if spacing is None:
        bb_min, bb_max = network.bounding_box()
        spacing = bb_max - bb_min + 0.1
    
    spacing = np.asarray(spacing)
    
    tiled_networks = []
    offsets = []
    
    for i in range(repeats[0]):
        for j in range(repeats[1]):
            for k in range(repeats[2]):
                offset = np.array([i, j, k]) * spacing
                tiled_networks.append(deepcopy(network))
                offsets.append(offset)
    
    result = merge(tiled_networks, offsets=offsets, merge_crosslinks=True, proximity_threshold=0.5)
    result.periodic = True
    
    if network.box_size is not None:
        result.box_size = network.box_size * np.array(repeats)
    
    return result


def trim_to_box(
    network: FiberNetwork,
    box_min: np.ndarray,
    box_max: np.ndarray,
    remove_outside: bool = True,
) -> FiberNetwork:
    """Trim network to bounding box.
    
    Parameters
    ----------
    network : FiberNetwork
        Input network.
    box_min : np.ndarray
        Minimum corner of trim box.
    box_max : np.ndarray
        Maximum corner of trim box.
    remove_outside : bool
        If True, remove fibers completely outside box.
        If False, also trim fibers that partially extend outside.
    
    Returns
    -------
    FiberNetwork
        Trimmed network.
    """
    box_min = np.asarray(box_min)
    box_max = np.asarray(box_max)
    
    result = FiberNetwork(
        dimension=network.dimension,
        box_size=box_max - box_min,
        periodic=network.periodic,
    )
    
    for fiber in network.fibers:
        pts = fiber.centerline
        inside = np.all((pts >= box_min) & (pts <= box_max), axis=1)
        
        if remove_outside:
            if np.any(inside):
                new_fiber = deepcopy(fiber)
                new_fiber.fiber_id = result.num_fibers
                result.add_fiber(new_fiber)
        else:
            inside_pts = pts[inside]
            if len(inside_pts) >= 2:
                new_fiber = Fiber(
                    centerline=inside_pts,
                    radius=fiber.radius,
                    material=fiber.material,
                    cross_section=fiber.cross_section,
                    cross_section_params=fiber.cross_section_params,
                    fiber_id=result.num_fibers,
                )
                result.add_fiber(new_fiber)
    
    return result


def duplicate_and_transform(
    network: FiberNetwork,
    num_copies: int,
    transform_func: callable,
) -> FiberNetwork:
    """Create multiple transformed copies and merge.
    
    Parameters
    ----------
    network : FiberNetwork
        Base network.
    num_copies : int
        Number of copies to create.
    transform_func : callable
        Function(index, network) -> transformed_network.
    
    Returns
    -------
    FiberNetwork
        Merged network with all copies.
    """
    copies = []
    for i in range(num_copies):
        copy = deepcopy(network)
        transformed = transform_func(i, copy)
        copies.append(transformed)
    
    return merge(copies, merge_crosslinks=True, proximity_threshold=0.5)


def align_by_anchor(
    network1: FiberNetwork,
    network2: FiberNetwork,
    anchor1: np.ndarray,
    anchor2: np.ndarray,
) -> Tuple[FiberNetwork, FiberNetwork]:
    """Align two networks by matching anchor points.
    
    Parameters
    ----------
    network1 : FiberNetwork
        First network (reference, not modified).
    network2 : FiberNetwork
        Second network (will be translated).
    anchor1 : np.ndarray
        Anchor point in network1.
    anchor2 : np.ndarray
        Anchor point in network2 (will be moved to anchor1).
    
    Returns
    -------
    tuple of FiberNetwork
        (network1, translated_network2)
    """
    offset = np.asarray(anchor1) - np.asarray(anchor2)
    net2_aligned = translate(network2, offset)
    return network1, net2_aligned


def create_pattern(
    base_network: FiberNetwork,
    pattern_type: str,
    num_units: int = 4,
    radius: float = 10.0,
    **kwargs,
) -> FiberNetwork:
    """Create patterned structures from base network.
    
    Parameters
    ----------
    base_network : FiberNetwork
        Base unit to pattern.
    pattern_type : str
        'circular', 'linear', 'grid', 'spiral'.
    num_units : int
        Number of units in pattern.
    radius : float
        Pattern radius (for circular/spiral).
    **kwargs
        Additional pattern parameters.
    
    Returns
    -------
    FiberNetwork
        Patterned network.
    """
    copies = []
    offsets = []
    
    if pattern_type == "circular":
        for i in range(num_units):
            angle = 2 * np.pi * i / num_units
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            offsets.append(np.array([x, y, 0]))
            
            copy = deepcopy(base_network)
            rotated = rotate(copy, angle, np.array([0, 0, 1]), origin=np.zeros(3))
            copies.append(rotated)
    
    elif pattern_type == "linear":
        spacing = kwargs.get("spacing", 10.0)
        for i in range(num_units):
            offsets.append(np.array([i * spacing, 0, 0]))
            copies.append(deepcopy(base_network))
    
    elif pattern_type == "grid":
        nx = kwargs.get("nx", int(np.sqrt(num_units)))
        ny = kwargs.get("ny", num_units // nx)
        spacing = kwargs.get("spacing", 10.0)
        
        for i in range(nx):
            for j in range(ny):
                offsets.append(np.array([i * spacing, j * spacing, 0]))
                copies.append(deepcopy(base_network))
    
    elif pattern_type == "spiral":
        turns = kwargs.get("turns", 2.0)
        for i in range(num_units):
            t = i / num_units
            angle = turns * 2 * np.pi * t
            r = radius * t
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = radius * t * 0.5
            
            offsets.append(np.array([x, y, z]))
            copy = deepcopy(base_network)
            rotated = rotate(copy, angle, np.array([0, 0, 1]), origin=np.zeros(3))
            copies.append(rotated)
    
    if not copies:
        return base_network
    
    return merge(copies, offsets=offsets, merge_crosslinks=True, proximity_threshold=0.5)

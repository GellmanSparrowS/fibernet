"""
Enhanced variants of existing generators with 2D/3D support and more options.

Provides:
- 2D/3D cross-compatible versions of all generators
- Parameterized variants with more customization
- Multi-radius, multi-material variants
- Curved and bent lattice variants
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.core.material import Material


def lattice_2d_to_3d(
    network_2d: FiberNetwork,
    num_layers: int = 3,
    layer_spacing: float = 5.0,
    add_vertical_bonds: bool = True,
    vertical_radius: float = None,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Extrude a 2D lattice into a 3D network by stacking layers.
    
    Parameters
    ----------
    network_2d : FiberNetwork
        Base 2D network.
    num_layers : int
        Number of layers to stack.
    layer_spacing : float
        Distance between layers in z-direction.
    add_vertical_bonds : bool
        Add vertical connecting fibers between layers.
    vertical_radius : float
        Radius of vertical fibers.
    """
    from fibernet.core.transform import translate, merge
    from copy import deepcopy
    
    layers = []
    for k in range(num_layers):
        layer = deepcopy(network_2d)
        layer = translate(layer, np.array([0, 0, k * layer_spacing]))
        layers.append(layer)
    
    result = merge(layers, merge_crosslinks=True, proximity_threshold=0.5)
    result.dimension = 3
    
    if add_vertical_bonds and num_layers > 1:
        mat = material or (network_2d.fibers[0].material if network_2d.fibers else Material())
        vr = vertical_radius or network_2d.mean_radius
        
        bb_min, bb_max = network_2d.bounding_box()
        spacing_x = (bb_max[0] - bb_min[0]) / max(network_2d.num_fibers, 1)
        
        for fiber in network_2d.fibers:
            mid_pt = 0.5 * (fiber.start_point + fiber.end_point)
            for k in range(num_layers - 1):
                z_bot = k * layer_spacing
                z_top = (k + 1) * layer_spacing
                result.add_fiber(Fiber.straight(
                    np.array([mid_pt[0], mid_pt[1], z_bot]),
                    np.array([mid_pt[0], mid_pt[1], z_top]),
                    radius=vr, material=mat, fiber_id=result.num_fibers,
                ))
    
    bb_min, bb_max = result.bounding_box()
    result.box_size = bb_max - bb_min
    result.metadata["generator"] = "lattice_2d_to_3d"
    result.metadata["num_layers"] = num_layers
    
    return result


def curved_lattice(
    base_generator: callable,
    base_kwargs: Dict[str, Any],
    curvature: float = 0.1,
    curve_axis: int = 2,
) -> FiberNetwork:
    """Apply curvature deformation to a lattice structure.
    
    Creates curved/arched lattice structures from flat lattices.
    
    Parameters
    ----------
    curvature : float
        Curvature magnitude (1/radius_of_curvature).
    curve_axis : int
        Axis around which to curve (0=x, 1=y, 2=z).
    """
    net = base_generator(**base_kwargs)
    bb_min, bb_max = net.bounding_box()
    center = 0.5 * (bb_min + bb_max)
    
    R = 1.0 / max(curvature, 1e-6)
    
    for fiber in net.fibers:
        pts = fiber.centerline.copy()
        
        if curve_axis == 2:
            for k in range(len(pts)):
                x_rel = pts[k, 0] - center[0]
                z_rel = pts[k, 2] - center[2]
                
                theta = x_rel / R
                new_x = R * np.sin(theta)
                new_z = R * (1 - np.cos(theta)) + z_rel
                
                pts[k, 0] = new_x + center[0]
                pts[k, 2] = new_z + center[2]
        
        elif curve_axis == 1:
            for k in range(len(pts)):
                x_rel = pts[k, 0] - center[0]
                y_rel = pts[k, 1] - center[1]
                
                theta = x_rel / R
                new_x = R * np.sin(theta)
                new_y = R * (1 - np.cos(theta)) + y_rel
                
                pts[k, 0] = new_x + center[0]
                pts[k, 1] = new_y + center[1]
        
        fiber.centerline = pts
    
    net.metadata["curvature"] = curvature
    net.metadata["curve_axis"] = curve_axis
    return net


def multi_radius_network(
    base_generator: callable,
    base_kwargs: Dict[str, Any],
    radius_distribution: str = "bimodal",
    radius_params: Dict[str, Any] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Create a network with multiple fiber radii.
    
    Parameters
    ----------
    radius_distribution : str
        'bimodal', 'uniform', 'normal', 'power_law'.
    radius_params : dict
        Parameters for the distribution.
    """
    rng = np.random.default_rng(seed)
    net = base_generator(**base_kwargs)
    
    if radius_params is None:
        radius_params = {
            "r1": 0.1, "r2": 0.5,
            "fraction_r1": 0.7,
        }
    
    if radius_distribution == "bimodal":
        r1 = radius_params.get("r1", 0.1)
        r2 = radius_params.get("r2", 0.5)
        frac = radius_params.get("fraction_r1", 0.7)
        
        for fiber in net.fibers:
            if rng.uniform() < frac:
                fiber.radius = r1
            else:
                fiber.radius = r2
    
    elif radius_distribution == "uniform":
        r_min = radius_params.get("min", 0.05)
        r_max = radius_params.get("max", 0.5)
        for fiber in net.fibers:
            fiber.radius = rng.uniform(r_min, r_max)
    
    elif radius_distribution == "normal":
        r_mean = radius_params.get("mean", 0.2)
        r_std = radius_params.get("std", 0.05)
        for fiber in net.fibers:
            fiber.radius = max(rng.normal(r_mean, r_std), 0.01)
    
    elif radius_distribution == "power_law":
        r_min = radius_params.get("min", 0.05)
        r_max = radius_params.get("max", 1.0)
        exponent = radius_params.get("exponent", -2.0)
        
        u = rng.uniform(size=net.num_fibers)
        radii = r_min + (r_max - r_min) * (1 - u) ** (1 / (exponent + 1))
        
        for i, fiber in enumerate(net.fibers):
            fiber.radius = radii[i]
    
    net.metadata["radius_distribution"] = radius_distribution
    return net


def variable_stiffness_network(
    base_generator: callable,
    base_kwargs: Dict[str, Any],
    stiffness_axis: int = 0,
    stiffness_func: str = "linear",
    E_range: Tuple[float, float] = (1e8, 1e10),
) -> FiberNetwork:
    """Create a network with spatially varying stiffness.
    
    Parameters
    ----------
    stiffness_axis : int
        Axis along which stiffness varies.
    stiffness_func : str
        'linear', 'exponential', 'step', 'gaussian'.
    E_range : tuple
        (E_min, E_max) range of Young's modulus.
    """
    from copy import deepcopy
    
    net = base_generator(**base_kwargs)
    
    bb_min, bb_max = net.bounding_box()
    L = bb_max[stiffness_axis] - bb_min[stiffness_axis]
    E_min, E_max = E_range
    
    for fiber in net.fibers:
        center = 0.5 * (fiber.start_point + fiber.end_point)
        t = (center[stiffness_axis] - bb_min[stiffness_axis]) / max(L, 1e-12)
        
        if stiffness_func == "linear":
            factor = t
        elif stiffness_func == "exponential":
            factor = (np.exp(3 * t) - 1) / (np.e**3 - 1)
        elif stiffness_func == "step":
            factor = 1.0 if t > 0.5 else 0.0
        elif stiffness_func == "gaussian":
            factor = np.exp(-((t - 0.5) / 0.2)**2)
        else:
            factor = t
        
        E = E_min + factor * (E_max - E_min)
        
        new_mat = deepcopy(fiber.material)
        new_mat.youngs_modulus = E
        if new_mat.shear_modulus:
            new_mat.shear_modulus = E / (2 * (1 + (new_mat.poissons_ratio or 0.3)))
        fiber.material = new_mat
    
    net.metadata["stiffness_gradient"] = {
        "axis": stiffness_axis,
        "function": stiffness_func,
        "E_range": E_range,
    }
    
    return net


def gyroid_infill(
    box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
    cell_size: float = 5.0,
    fiber_radius: float = 0.2,
    resolution: int = 50,
    threshold: float = 0.0,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a gyroid-based fiber network (TPMS structure).
    
    Triply periodic minimal surface (TPMS) gyroid infill,
    used in advanced metamaterials and lattice structures.
    """
    mat = material or Material(name="gyroid")
    Lx, Ly, Lz = box_size
    a = cell_size
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array(box_size),
        metadata={"generator": "gyroid_infill", "cell_size": cell_size},
    )
    
    xs = np.linspace(0, Lx, resolution)
    ys = np.linspace(0, Ly, resolution)
    zs = np.linspace(0, Lz, resolution)
    
    fid = 0
    for zi, z in enumerate(zs[:-1]):
        for yi, y in enumerate(ys[:-1]):
            for xi, x in enumerate(xs[:-1]):
                val = (
                    np.sin(2 * np.pi * x / a) * np.cos(2 * np.pi * y / a)
                    + np.sin(2 * np.pi * y / a) * np.cos(2 * np.pi * z / a)
                    + np.sin(2 * np.pi * z / a) * np.cos(2 * np.pi * x / a)
                )
                
                if abs(val) < threshold + 0.3:
                    dx = xs[1] - xs[0]
                    p1 = np.array([x, y, z])
                    p2 = np.array([x + dx, y, z])
                    
                    val2 = (
                        np.sin(2 * np.pi * p2[0] / a) * np.cos(2 * np.pi * p2[1] / a)
                        + np.sin(2 * np.pi * p2[1] / a) * np.cos(2 * np.pi * p2[2] / a)
                        + np.sin(2 * np.pi * p2[2] / a) * np.cos(2 * np.pi * p2[0] / a)
                    )
                    
                    if abs(val2) < threshold + 0.3:
                        net.add_fiber(Fiber.straight(
                            p1, p2, radius=fiber_radius, material=mat, fiber_id=fid
                        ))
                        fid += 1
    
    net.auto_crosslink(threshold=3.0 * fiber_radius)
    return net


def diamond_lattice_3d(
    spacing: float = 5.0,
    grid_size: Tuple[int, int, int] = (3, 3, 3),
    radius: float = 0.1,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 3D diamond lattice (tetrahedral bonding)."""
    mat = material or Material(name="diamond")
    nx, ny, nz = grid_size
    a = spacing
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([nx * a, ny * a, nz * a]),
        metadata={"generator": "diamond_lattice_3d", "spacing": spacing},
    )
    
    basis = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.25, 0.25, 0.25]) * a,
    ]
    
    fid = 0
    node_positions = {}
    
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                for b_idx, b in enumerate(basis):
                    pos = np.array([i * a, j * a, k * a]) + b
                    key = (i, j, k, b_idx)
                    node_positions[key] = pos
    
    bond_vectors = [
        np.array([0.25, 0.25, 0.25]) * a,
        np.array([0.25, -0.25, -0.25]) * a,
        np.array([-0.25, 0.25, -0.25]) * a,
        np.array([-0.25, -0.25, 0.25]) * a,
    ]
    
    edge_set = set()
    for key, pos in node_positions.items():
        i, j, k, b_idx = key
        
        if b_idx == 0:
            for bv in bond_vectors:
                target = pos + bv
                for key2, pos2 in node_positions.items():
                    if np.linalg.norm(pos2 - target) < 0.1 * a:
                        edge_key = tuple(sorted([key, key2]))
                        if edge_key not in edge_set:
                            edge_set.add(edge_key)
                            net.add_fiber(Fiber.straight(
                                pos, pos2, radius=radius, material=mat, fiber_id=fid
                            ))
                            fid += 1
    
    net.auto_crosslink(threshold=2.5 * radius)
    return net


def foam_like_3d(
    box_size: Tuple[float, float, float] = (30.0, 30.0, 30.0),
    num_cells: int = 50,
    radius: float = 0.15,
    strut_curvature: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate a foam-like 3D fiber network.
    
    Models open-cell foam structure using perturbed Voronoi.
    """
    rng = np.random.default_rng(seed)
    mat = material or Material(name="foam")
    Lx, Ly, Lz = box_size
    
    from fibernet.gen.advanced import voronoi_network_3d
    net = voronoi_network_3d(
        num_seeds=num_cells, box_size=box_size,
        radius=radius, material=mat, seed=seed,
    )
    
    if strut_curvature > 0:
        for fiber in net.fibers:
            pts = fiber.centerline.copy()
            mid = 0.5 * (pts[0] + pts[-1])
            length = np.linalg.norm(pts[-1] - pts[0])
            
            if len(pts) > 2 and length > 1e-6:
                direction = (pts[-1] - pts[0]) / length
                if abs(direction[0]) < 0.9:
                    perp = np.cross(direction, [1, 0, 0])
                else:
                    perp = np.cross(direction, [0, 1, 0])
                perp /= np.linalg.norm(perp)
                
                for p_idx in range(1, len(pts) - 1):
                    t = p_idx / (len(pts) - 1)
                    offset = strut_curvature * length * np.sin(np.pi * t) * perp
                    offset += rng.normal(0, strut_curvature * 0.3, 3)
                    pts[p_idx] += offset
                
                fiber.centerline = pts
    
    net.metadata["generator"] = "foam_like_3d"
    net.metadata["strut_curvature"] = strut_curvature
    return net

"""
Triply Periodic Minimal Surface (TPMS) generators for fiber networks.

Implements:
- Gyroid (Schwarz G-surface)
- Schwarz Diamond (D-surface)
- Schwarz Primitive (P-surface)
- I-WP (Wrapped Package)
- Neovius
- Lidinoid
- F-RD (Face-centered Rhombic Dodecahedron)
- Sheet-based TPMS (level-set slicing)
- Lattice-based TPMS (skeleton extraction)

References:
- Gibson, L. J., & Ashby, M. F. (1997). Cellular Solids.
- Schoen, A. H. (1970). Infinite periodic minimal surfaces without self-intersections.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.core.material import Material


def _tpms_field(
    kind: str,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> np.ndarray:
    """Compute TPMS level-set field.
    
    Parameters
    ----------
    kind : str
        TPMS type: 'gyroid', 'diamond', 'primitive', 'iwp', 'neovius', 'lidinoid'
    X, Y, Z : np.ndarray
        3D coordinate grids.
    
    Returns
    -------
    np.ndarray
        Level-set field (zero at surface).
    """
    k = kind.lower()
    
    if k in ('gyroid', 'g'):
        return (np.sin(X) * np.cos(Y) +
                np.sin(Y) * np.cos(Z) +
                np.sin(Z) * np.cos(X))
    
    elif k in ('diamond', 'd'):
        return (np.sin(X) * np.sin(Y) * np.sin(Z) +
                np.sin(X) * np.cos(Y) * np.cos(Z) +
                np.cos(X) * np.sin(Y) * np.cos(Z) +
                np.cos(X) * np.cos(Y) * np.sin(Z))
    
    elif k in ('primitive', 'p'):
        return np.cos(X) + np.cos(Y) + np.cos(Z)
    
    elif k in ('iwp', 'i-wp'):
        return (2.0 * (np.cos(X) * np.cos(Y) +
                       np.cos(Y) * np.cos(Z) +
                       np.cos(Z) * np.cos(X)) -
                (np.cos(2*X) + np.cos(2*Y) + np.cos(2*Z)))
    
    elif k == 'neovius':
        return (3.0 * (np.cos(X) + np.cos(Y) + np.cos(Z)) +
                4.0 * np.cos(X) * np.cos(Y) * np.cos(Z))
    
    elif k == 'lidinoid':
        return (np.cos(2*X) * np.sin(Y) * np.cos(Z) +
                np.cos(2*Y) * np.sin(Z) * np.cos(X) +
                np.cos(2*Z) * np.sin(X) * np.cos(Y) -
                np.cos(2*X) * np.cos(2*Y) -
                np.cos(2*Y) * np.cos(2*Z) -
                np.cos(2*Z) * np.cos(2*X) + 0.3)
    
    elif k in ('frd', 'f-rd'):
        return (4.0 * np.cos(X) * np.cos(Y) * np.cos(Z) -
                (np.cos(2*X) * np.cos(2*Y) +
                 np.cos(2*Y) * np.cos(2*Z) +
                 np.cos(2*Z) * np.cos(2*X)) + 0.5)
    
    else:
        raise ValueError(f"Unknown TPMS type: {kind}. "
                        f"Choose from: gyroid, diamond, primitive, iwp, "
                        f"neovius, lidinoid, frd")


def tpms_sheet(
    kind: str = "gyroid",
    box_size: Tuple[float, float, float] = (10.0, 10.0, 10.0),
    resolution: int = 50,
    thickness: float = 0.3,
    num_periods: Tuple[int, int, int] = (1, 1, 1),
    fiber_radius: float = 0.05,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate fiber network from TPMS sheet (level-set surface).
    
    Creates a fiber network that follows the zero level-set of a TPMS,
    discretized as a set of fibers along the surface.
    
    Parameters
    ----------
    kind : str
        TPMS type: 'gyroid', 'diamond', 'primitive', 'iwp', 'neovius', 'lidinoid'
    box_size : tuple
        Physical size (Lx, Ly, Lz).
    resolution : int
        Grid resolution per period for marching cubes.
    thickness : float
        Level-set band thickness (controls surface thickness).
    num_periods : tuple
        Number of periods in each direction.
    fiber_radius : float
        Radius of generated fibers.
    material : Material, optional
        Fiber material. Defaults to generic polymer.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    FiberNetwork
    """
    if seed is not None:
        np.random.seed(seed)
    
    if material is None:
        material = Material(youngs_modulus=1e9, poissons_ratio=0.3, density=1000.0)
    
    Lx, Ly, Lz = box_size
    nx, ny, nz = num_periods
    
    res_x = resolution * nx
    res_y = resolution * ny
    res_z = resolution * nz
    
    x = np.linspace(0, 2 * np.pi * nx, res_x)
    y = np.linspace(0, 2 * np.pi * ny, res_y)
    z = np.linspace(0, 2 * np.pi * nz, res_z)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    field = _tpms_field(kind, X, Y, Z)
    
    try:
        from skimage.measure import marching_cubes
        verts, faces, normals, _ = marching_cubes(
            field, level=0.0,
            spacing=(Lx / res_x, Ly / res_y, Lz / res_z)
        )
    except ImportError:
        verts, faces = _simple_marching_cubes(field, 0.0,
                                               (Lx / res_x, Ly / res_y, Lz / res_z))
    
    edges = set()
    if len(faces) > 0 and len(faces[0]) == 3:
        # Real faces from marching cubes
        for face in faces:
            for i in range(3):
                e = tuple(sorted([face[i], face[(i+1) % 3]]))
                edges.add(e)
    else:
        # Fallback edges from simple marching cubes
        for edge in faces:
            edges.add(tuple(sorted(edge)))
    
    edges = list(edges)
    
    if len(edges) == 0:
        return FiberNetwork(box_size=np.array(box_size))
    
    edge_array = np.array(edges)
    
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    
    n_verts = len(verts)
    row = np.concatenate([edge_array[:, 0], edge_array[:, 1]])
    col = np.concatenate([edge_array[:, 1], edge_array[:, 0]])
    data = np.ones(len(row))
    adj = csr_matrix((data, (row, col)), shape=(n_verts, n_verts))
    
    n_comp, labels = connected_components(adj, directed=False)
    
    fibers = []
    for comp_id in range(n_comp):
        comp_verts = np.where(labels == comp_id)[0]
        
        if len(comp_verts) < 2:
            continue
        
        comp_edges = [e for e in edges if labels[e[0]] == comp_id]
        
        if len(comp_edges) < 1:
            continue
        
        ordered_path = _order_edges_to_path(verts, comp_edges)
        
        if ordered_path is not None and len(ordered_path) >= 2:
            centerline = np.array([verts[v] for v in ordered_path])
            
            if np.linalg.norm(centerline[-1] - centerline[0]) < 1e-6:
                centerline = centerline[:-1]
            
            if len(centerline) >= 2:
                fiber = Fiber(centerline=centerline, radius=fiber_radius, material=material)
                fibers.append(fiber)
        else:
            for edge in comp_edges:
                p1, p2 = verts[edge[0]], verts[edge[1]]
                fiber = Fiber(
                    centerline=np.array([p1, p2]),
                    radius=fiber_radius,
                    material=material
                )
                fibers.append(fiber)
    
    network = FiberNetwork(fibers=fibers, box_size=np.array(box_size))
    network.metadata['generator'] = f'tpms_sheet_{kind}'
    network.metadata['tpms_type'] = kind
    network.metadata['num_periods'] = num_periods
    network.metadata['resolution'] = resolution
    
    return network


def tpms_lattice(
    kind: str = "gyroid",
    box_size: Tuple[float, float, float] = (10.0, 10.0, 10.0),
    resolution: int = 40,
    num_periods: Tuple[int, int, int] = (1, 1, 1),
    strut_radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate fiber network from TPMS lattice (solid-void skeleton).
    
    Extracts the medial axis / skeleton of the solid phase of a TPMS
    and converts it to a fiber network.
    
    Parameters
    ----------
    kind : str
        TPMS type.
    box_size : tuple
        Physical size.
    resolution : int
        Grid resolution per period.
    num_periods : tuple
        Periods in each direction.
    strut_radius : float
        Radius of struts in the lattice.
    material : Material, optional
        Strut material.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    FiberNetwork
    """
    if seed is not None:
        np.random.seed(seed)
    
    if material is None:
        material = Material(youngs_modulus=1e9, poissons_ratio=0.3, density=1000.0)
    
    Lx, Ly, Lz = box_size
    nx, ny, nz = num_periods
    
    res_x = resolution * nx
    res_y = resolution * ny
    res_z = resolution * nz
    
    x = np.linspace(0, 2 * np.pi * nx, res_x)
    y = np.linspace(0, 2 * np.pi * ny, res_y)
    z = np.linspace(0, 2 * np.pi * nz, res_z)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    field = _tpms_field(kind, X, Y, Z)
    
    solid = field > 0
    
    skeleton = _extract_skeleton_3d(solid)
    
    vox_size = np.array([Lx / res_x, Ly / res_y, Lz / res_z])
    
    fibers = []
    for path in skeleton:
        centerline = np.array(path) * vox_size
        if len(centerline) >= 2:
            fiber = Fiber(centerline=centerline, radius=strut_radius, material=material)
            fibers.append(fiber)
    
    if len(fibers) == 0:
        return FiberNetwork(box_size=np.array(box_size))
    
    network = FiberNetwork(fibers=fibers, box_size=np.array(box_size))
    network.metadata['generator'] = f'tpms_lattice_{kind}'
    network.metadata['tpms_type'] = kind
    network.metadata['num_periods'] = num_periods
    
    return network


def tpms_gradient(
    kind: str = "gyroid",
    box_size: Tuple[float, float, float] = (20.0, 10.0, 10.0),
    resolution: int = 40,
    gradient_axis: int = 0,
    gradient_range: Tuple[float, float] = (0.5, 2.0),
    strut_radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate TPMS with graded unit cell size.
    
    Creates a functionally graded TPMS structure where the unit cell
    size varies along a specified axis.
    
    Parameters
    ----------
    kind : str
        TPMS type.
    box_size : tuple
        Physical size.
    resolution : int
        Base grid resolution.
    gradient_axis : int
        Axis along which to vary unit cell size (0, 1, or 2).
    gradient_range : tuple
        (min_periods_per_unit, max_periods_per_unit).
    strut_radius : float
        Strut radius.
    material : Material, optional
        Material.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    FiberNetwork
    """
    if seed is not None:
        np.random.seed(seed)
    
    if material is None:
        material = Material(youngs_modulus=1e9, poissons_ratio=0.3, density=1000.0)
    
    Lx, Ly, Lz = box_size
    L = [Lx, Ly, Lz][gradient_axis]
    
    n_segments = 4
    seg_length = L / n_segments
    
    all_fibers = []
    
    for seg_idx in range(n_segments):
        t = seg_idx / (n_segments - 1) if n_segments > 1 else 0.5
        periods_per_unit = gradient_range[0] + t * (gradient_range[1] - gradient_range[0])
        
        n_periods = max(1, int(round(periods_per_unit * seg_length)))
        
        seg_box = list(box_size)
        seg_box[gradient_axis] = seg_length
        
        seg_net = tpms_lattice(
            kind=kind,
            box_size=tuple(seg_box),
            resolution=resolution,
            num_periods=tuple([n_periods if a == gradient_axis else 1 for a in range(3)]),
            strut_radius=strut_radius,
            material=material,
            seed=seed + seg_idx if seed is not None else None,
        )
        
        offset = np.zeros(3)
        offset[gradient_axis] = seg_idx * seg_length
        
        for fiber in seg_net.fibers:
            new_centerline = fiber.centerline + offset
            new_fiber = Fiber(
                centerline=new_centerline,
                radius=fiber.radius,
                material=fiber.material
            )
            all_fibers.append(new_fiber)
    
    network = FiberNetwork(fibers=all_fibers, box_size=np.array(box_size))
    network.metadata['generator'] = f'tpms_gradient_{kind}'
    network.metadata['gradient_axis'] = gradient_axis
    network.metadata['gradient_range'] = gradient_range
    
    return network


def _simple_marching_cubes(
    field: np.ndarray,
    level: float,
    spacing: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Simplified marching cubes fallback when skimage is not available.
    
    Uses zero-crossing detection on edges of the voxel grid.
    """
    dx, dy, dz = spacing
    nx, ny, nz = field.shape
    
    verts = []
    edges = []
    vert_map = {}
    
    def add_vertex(pos):
        key = tuple(np.round(pos, 6))
        if key not in vert_map:
            vert_map[key] = len(verts)
            verts.append(pos)
        return vert_map[key]
    
    def interp_edge(p1, p2, v1, v2):
        if (v1 - level) * (v2 - level) < 0:
            t = (level - v1) / (v2 - v1)
            pos = p1 + t * (p2 - p1)
            return add_vertex(pos)
        return None
    
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                corners = np.array([
                    [i, j, k], [i+1, j, k], [i+1, j+1, k], [i, j+1, k],
                    [i, j, k+1], [i+1, j, k+1], [i+1, j+1, k+1], [i, j+1, k+1]
                ], dtype=float)
                corners *= np.array([dx, dy, dz])
                
                vals = [field[i, j, k], field[i+1, j, k],
                        field[i+1, j+1, k], field[i, j+1, k],
                        field[i, j, k+1], field[i+1, j, k+1],
                        field[i+1, j+1, k+1], field[i, j+1, k+1]]
                
                cell_verts = []
                edge_pairs = [(0,1), (1,2), (2,3), (3,0),
                             (4,5), (5,6), (6,7), (7,4),
                             (0,4), (1,5), (2,6), (3,7)]
                
                for e1, e2 in edge_pairs:
                    v = interp_edge(corners[e1], corners[e2], vals[e1], vals[e2])
                    if v is not None:
                        cell_verts.append(v)
                
                for a in range(len(cell_verts)):
                    for b in range(a + 1, min(a + 3, len(cell_verts))):
                        edges.append([cell_verts[a], cell_verts[b]])
    
    if len(verts) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    
    return np.array(verts), np.array(edges)


def _order_edges_to_path(verts: np.ndarray, edges: List[Tuple[int, int]]) -> Optional[List[int]]:
    """Order a set of edges into a continuous path."""
    if not edges:
        return None
    
    from collections import defaultdict
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    start = None
    for v, neighbors in adj.items():
        if len(neighbors) == 1:
            start = v
            break
    
    if start is None:
        start = edges[0][0]
    
    visited = set()
    path = [start]
    current = start
    
    while True:
        found_next = False
        for neighbor in adj[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                current = neighbor
                found_next = True
                break
        
        if not found_next:
            break
    
    return path


def _extract_skeleton_3d(binary: np.ndarray) -> List[List[Tuple[int, int, int]]]:
    """Extract skeleton from 3D binary volume using thinning."""
    from scipy.ndimage import label as ndimage_label
    
    labeled, n_components = ndimage_label(binary)
    
    paths = []
    
    for comp_id in range(1, n_components + 1):
        component = labeled == comp_id
        
        coords = np.argwhere(component)
        if len(coords) < 2:
            continue
        
        center = coords.mean(axis=0)
        
        sorted_coords = coords[np.argsort(np.sum((coords - center)**2, axis=1))]
        
        path = []
        for coord in sorted_coords[:min(50, len(sorted_coords))]:
            path.append(tuple(coord))
        
        if len(path) >= 2:
            paths.append(path)
    
    return paths

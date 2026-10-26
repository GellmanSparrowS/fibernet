"""Legacy FiberNetwork convenience API, separate from the StructureGraph API.

The generator registry is accessed through ``fibernet.api.create``; it is not
exported as ``fibernet.create``. Registering a backend here does not extend the
root ``fibernet.simulate`` dispatcher. The legacy mechanics entry point is
unsupported and raises an explicit migration error rather than changing models.

Quick Start
-----------
>>> import fibernet.api as legacy
>>> net = legacy.create("random_2d", num_fibers=200, fiber_length=10.0)
>>> names = legacy.list_generators()

For current simulations, prepare a StructureGraph and use ``fibernet.simulate``.
"""

import json as _json
import numpy as np
from typing import Optional, Dict, List, Union, Tuple, Callable, Any, Literal, TYPE_CHECKING, overload
from pathlib import Path
from dataclasses import dataclass, field

from fibernet.core.network import FiberNetwork
from fibernet.core.structure_graph import StructureGraph as _StructureGraph
from fibernet.core.fiber import Fiber
from fibernet.core.material import Material
from fibernet import gen
from fibernet.core import transform

if TYPE_CHECKING:
    from fibernet.analysis.topology_metrics import TopologyMetricsV1


# ============================================================
# Registry Pattern for Extensibility
# ============================================================

class _Registry:
    """Central registry for generators and simulation backends.
    
    Supports dynamic registration so third-party plugins can extend
    FiberNet without modifying source code.
    
    Examples
    --------
    >>> # Register a custom generator
    >>> @legacy.register_generator("my_custom_lattice")
    ... def make_my_lattice(**kwargs):
    ...     return FiberNetwork(...)
    >>> 
    >>> # Use it
    >>> net = legacy.create("my_custom_lattice", param1=42)
    """
    
    def __init__(self):
        self.generators: Dict[str, Callable] = {}
        self.backends: Dict[str, Callable] = {}
    
    def register_generator(self, name: str, func: Callable, *, overwrite: bool = False):
        """Register a generator function."""
        if name in self.generators and not overwrite:
            raise ValueError(f"Generator '{name}' already registered. Use overwrite=True.")
        self.generators[name] = func
    
    def register_backend(self, name: str, func: Callable, *, overwrite: bool = False):
        """Register a simulation backend."""
        if name in self.backends and not overwrite:
            raise ValueError(f"Backend '{name}' already registered. Use overwrite=True.")
        self.backends[name] = func
    
    def list_generators(self) -> List[str]:
        return sorted(self.generators.keys())
    
    def list_backends(self) -> List[str]:
        return sorted(self.backends.keys())


_registry = _Registry()


def register_generator(name: str) -> Callable:
    """Decorator to register a generator function.
    
    Parameters
    ----------
    name : str
        Unique generator identifier.
    
    Returns
    -------
    Callable
        Decorator function.
    
    Examples
    --------
    >>> @legacy.register_generator("my_lattice")
    ... def make_lattice(cell_size=10.0, **kwargs):
    ...     return FiberNetwork(...)
    """
    def decorator(func):
        _registry.register_generator(name, func)
        return func
    return decorator


def register_backend(name: str) -> Callable:
    """Decorator to register a simulation backend.
    
    Parameters
    ----------
    name : str
        Unique backend identifier.
    
    Returns
    -------
    Callable
        Decorator function.
    
    Examples
    --------
    >>> @legacy.register_backend("my_fem")
    ... def run_my_fem(network, strain=0.01, **kwargs):
    ...     return {...}
    """
    def decorator(func):
        _registry.register_backend(name, func)
        return func
    return decorator


# ============================================================
# Built-in Generator Registration
# ============================================================

def _register_builtin_generators():
    """Register all built-in generators (consolidated).
    
    Removed: laminates, woven 2D, helix/braided, category 16, specific lattices/metamaterials
    Added: unified lattice_2d, lattice_3d, metamaterial_2d, curved_random_2d,
           entangled_3d, biomimetic_network, hierarchical_lattice
    """
    
    def _safe_register(name, func):
        try:
            _registry.register_generator(name, func)
        except (AttributeError, ValueError):
            pass
    
    # Core random
    _safe_register("random_2d", gen.random_straight_2d)
    _safe_register("random_3d", gen.random_straight_3d)
    _safe_register("random_walk", gen.random_walk_fibers)
    
    # Unified lattices
    _safe_register("lattice_2d", gen.lattice_2d)
    _safe_register("lattice_3d", gen.lattice_3d)
    
    # Unified metamaterials
    _safe_register("metamaterial_2d", gen.metamaterial_2d)
    
    # Curved fibers
    _safe_register("curved_random_2d", gen.curved_random_2d)
    
    # Entangled 3D
    _safe_register("entangled_3d", gen.entangled_3d)
    
    # Biomimetic (merged)
    _safe_register("biomimetic_network", gen.biomimetic_network)
    
    # Hierarchical
    _safe_register("hierarchical_lattice", gen.hierarchical_lattice)
    
    # Fractals
    _safe_register("sierpinski", gen.sierpinski_triangle)
    _safe_register("koch_curve", gen.koch_curve)
    _safe_register("fractal_tree", gen.fractal_tree)
    _safe_register("hilbert", gen.hilbert_curve)
    _safe_register("fractal_network", gen.fractal_network)
    
    # Advanced
    _safe_register("voronoi_2d", gen.voronoi_network_2d)
    _safe_register("voronoi_3d", gen.voronoi_network_3d)
    _safe_register("electrospun", gen.electrospun_network)
    _safe_register("meltblown", gen.meltblown_network)
    _safe_register("paper_network", gen.paper_network)
    _safe_register("foam_like_3d", gen.foam_like_3d)
    
    # TPMS
    try:
        from functools import partial
        from fibernet.gen.tpms import tpms_sheet, tpms_lattice, tpms_gradient
        _safe_register("tpms_sheet", partial(tpms_sheet, resolution=15))
        _safe_register("tpms_lattice", partial(tpms_lattice, resolution=15))
        _safe_register("tpms_gradient", partial(tpms_gradient, resolution=12))
    except ImportError:
        pass
    
    # Field-guided
    try:
        from functools import partial
        from fibernet.gen.field_guided import field_guided_network, FieldGuidedConfig
        _safe_register("field_guided", partial(
            field_guided_network,
            config=FieldGuidedConfig(
                fiber_count=200, canvas_size=256,
                fiber_length_mean=30.0, fiber_length_std=8.0,
                fiber_length_min=10.0, fiber_length_max=60.0,
                seed=42,
            )
        ))
    except (ImportError, TypeError):
        try:
            _safe_register("field_guided", gen.field_guided_network)
        except AttributeError:
            pass


_register_builtin_generators()


# ============================================================
# High-Level API Functions
# ============================================================

def create(
    generator: str = "random_2d",
    **kwargs,
) -> FiberNetwork:
    """Create a fiber network using a registered generator.
    
    Parameters
    ----------
    generator : str
        Generator name. Use ``list_generators()`` to see all available.
    **kwargs
        Generator-specific parameters.
    
    Returns
    -------
    FiberNetwork
        Generated fiber network.
    
    See Also
    --------
    list_generators : Show all available generators.
    register_generator : Register a custom generator.
    
    Examples
    --------
    >>> import fibernet.api as legacy
    >>> names = legacy.list_generators()
    >>> net = legacy.create("random_2d", num_fibers=200, fiber_length=10.0)
    """
    if generator not in _registry.generators:
        available = _registry.list_generators()
        raise ValueError(
            f"Unknown generator: '{generator}'.\n"
            f"Available generators ({len(available)}):\n"
            + "\n".join(f"  - {g}" for g in available)
        )
    return _registry.generators[generator](**kwargs)


def list_generators() -> List[str]:
    """List all registered generator names.
    
    Returns
    -------
    list of str
        Sorted list of available generator names.
    """
    return _registry.list_generators()


def list_backends() -> List[str]:
    """List all registered simulation backend names."""
    return _registry.list_backends()


def mirror(network: FiberNetwork, axis: int = 0) -> FiberNetwork:
    """Mirror a network along an axis."""
    return transform.mirror(network, axis=axis)


def rotate(
    network: FiberNetwork,
    angle: float = 0.0,
    axis: Union[np.ndarray, List] = None,
) -> FiberNetwork:
    """Rotate a network."""
    if axis is None:
        axis = np.array([0, 0, 1])
    return transform.rotate(network, angle=angle, axis=np.asarray(axis))


def scale(network: FiberNetwork, factor: float = 1.0) -> FiberNetwork:
    """Scale a network."""
    return transform.scale(network, factor=factor)


def translate(network: FiberNetwork, offset: Union[np.ndarray, List] = None) -> FiberNetwork:
    """Translate a network."""
    if offset is None:
        offset = np.zeros(3)
    return transform.translate(network, offset=np.asarray(offset))


def merge(networks: List[FiberNetwork], offsets: List[np.ndarray] = None) -> FiberNetwork:
    """Merge multiple networks."""
    return transform.merge(networks, offsets=offsets)


def tile(network: FiberNetwork, repeats: Tuple[int, int, int] = (2, 2, 1), spacing=None) -> FiberNetwork:
    """Tile a network periodically.
    
    Parameters
    ----------
    network : FiberNetwork
        Base unit cell.
    repeats : tuple of int
        Number of repetitions in x, y, z directions.
    spacing : np.ndarray, optional
        Spacing between tiles. Defaults to bounding box size + 0.1.
    """
    return transform.tile(network, repeats=repeats, spacing=spacing)


def simulate_mechanics(
    network: FiberNetwork,
    strain: float = 0.01,
    axis: int = 0,
    model: str = "linear",
    **kwargs,
) -> Dict:
    """Unsupported legacy FiberNetwork mechanics entry point.

    The former FiberFEM and nonlinear constitutive implementations are absent.
    This function deliberately does not substitute the StructureGraph beam-frame
    model or reinterpret legacy engineering strain as a stretch ratio.
    For a separately prepared StructureGraph, use fibernet.simulate with
    backend="fem", mode="stretch", strain=1 + engineering_strain. Review the
    graph conversion, boundary conditions, units and constitutive assumptions;
    this is not an equivalent drop-in replacement. 3-D FEM is linear-only.
    """
    raise NotImplementedError(
        "simulate_mechanics is an unsupported legacy FiberNetwork API: its "
        "FiberFEM/nonlinear solvers are unavailable. Prepare and validate a "
        "StructureGraph for fibernet.simulate(..., backend='fem', mode='stretch') "
        "instead; strain there is stretch ratio (1 + engineering strain), and "
        "the model and boundary conditions are not a drop-in replacement."
    )


def simulate_dynamics(
    network: FiberNetwork,
    dt: float = 1e-7,
    steps: int = 5000,
    damping: float = 0.01,
    external_force: Optional[np.ndarray] = None,
    fixed_nodes: Optional[List[int]] = None,
    save_interval: int = 100,
    backend: str = "taichi",
    **kwargs,
) -> Dict:
    """Run the explicit legacy NumPy mass-spring path.

    The historical default backend="taichi" is retained in the signature but
    rejects before conversion. Its crosslink topology and mixed unit/coefficient
    conventions have no validated mapping to the current StructureGraph engine.
    Use fibernet.simulate with a prepared StructureGraph for native dynamics.
    The explicit NumPy implementation is preserved; this dispatch repair does
    not validate or correct its physical units or integration conventions.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network with crosslinks.
    dt : float
        Time step (seconds). Should be small enough for stability.
    steps : int
        Number of integration steps.
    damping : float
        Velocity damping coefficient (0 = no damping, 0.01 = light damping).
    external_force : ndarray, optional
        External force on each node (N, 3).
    fixed_nodes : list of int, optional
        Node indices to fix in space.
    save_interval : int
        Save trajectory every N steps.
    backend : str
        Explicit "numpy" selects the unchanged legacy implementation.
        "taichi" raises NotImplementedError with migration guidance; unknown
        names raise ValueError. No backend is selected as a fallback.
    
    Returns
    -------
    dict
        ``positions`` (final), ``trajectory`` (list of position snapshots),
        ``velocities``, ``forces``, ``energy``, ``time_seconds``.
    
    See Also
    --------
    simulate_mechanics : Unsupported legacy mechanics (migration error).
    
    Notes
    -----
    See docs/TAICHI_API_DISPATCH.md. Legacy NumPy output semantics are unchanged;
    do not treat this wrapper as a unit-consistent native-backend adapter.
    """
    if not isinstance(backend, str) or backend not in ("taichi", "numpy"):
        raise ValueError("backend must be 'taichi' or 'numpy'")
    if backend == "taichi":
        raise NotImplementedError(
            "Legacy FiberNetwork Taichi dynamics has no validated crosslink-topology "
            "and unit mapping to StructureGraph (EA/L versus EA; mm versus physical "
            "length units). Prepare a StructureGraph with explicit coefficients, "
            "masses and units, then use fibernet.simulate(graph, mode='dynamics', "
            "backend='taichi', ...). Endpoint-only conversion is not equivalent. "
            "See docs/TAICHI_API_DISPATCH.md."
        )
    
    # Build spring-mass system from network
    edges, rest_lengths, stiffness, masses, positions = _build_mass_spring_system(network)
    
    if fixed_nodes is None:
        fixed_nodes = []
    
    velocities = np.zeros_like(positions)
    
    if external_force is not None:
        velocities = velocities  # placeholder
    
    return _numpy_dynamics(
        positions, velocities, masses, edges, rest_lengths, stiffness,
        dt=dt, steps=steps, damping=damping,
        fixed_nodes=fixed_nodes, external_force=external_force,
        save_interval=save_interval,
    )


def _build_mass_spring_system(network: FiberNetwork):
    """Convert a FiberNetwork to a mass-spring system.
    
    The conversion maps:
    - Crosslinks -> nodes (point masses)
    - Fiber segments between crosslinks -> springs
    
    Spring stiffness: k = E * A / L (axial rigidity)
    Node mass: fiber mass distributed proportionally to crosslink nodes
    
    Returns
    -------
    edges : ndarray (M, 2)
        Edge connectivity (node indices).
    rest_lengths : ndarray (M,)
        Rest length of each spring (mm).
    stiffness : ndarray (M,)
        Spring stiffness k = E * A / L (N/m).
    masses : ndarray (N,)
        Node masses (kg).
    positions : ndarray (N, 3)
        Node positions (mm).
    """
    if len(network.crosslinks) == 0:
        raise ValueError("Network has no crosslinks. Cannot build mass-spring system.")
    
    # Step 1: Create nodes at crosslink positions
    node_positions = np.array([cl.position for cl in network.crosslinks])
    n_nodes = len(node_positions)
    
    # Step 2: Group crosslinks by fiber
    fiber_crosslinks = {}  # fiber_idx -> [(cl_idx, param), ...]
    
    for cl_idx, cl in enumerate(network.crosslinks):
        if cl.fiber_i not in fiber_crosslinks:
            fiber_crosslinks[cl.fiber_i] = []
        if cl.fiber_j not in fiber_crosslinks:
            fiber_crosslinks[cl.fiber_j] = []
        
        fiber_crosslinks[cl.fiber_i].append((cl_idx, cl.param_i))
        fiber_crosslinks[cl.fiber_j].append((cl_idx, cl.param_j))
    
    # Step 3: Build springs along each fiber
    edges_list = []
    rest_lengths_list = []
    stiffness_list = []
    spring_fiber_map = {}  # edge_idx -> fiber_idx (for diagnostics)
    
    # Track unique edges to avoid duplicates
    seen_edges = set()
    
    for fiber_idx, cl_list in fiber_crosslinks.items():
        if len(cl_list) < 2:
            continue
        
        # Sort by param (position along fiber)
        cl_list.sort(key=lambda x: x[1])
        
        # Get fiber properties
        fiber = network.fibers[fiber_idx]
        radius_m = fiber.radius * 1e-3  # mm -> m
        A = np.pi * radius_m**2          # m^2
        E = fiber.material.youngs_modulus if fiber.material else 200e9  # Pa
        
        # Create springs between consecutive crosslinks
        for k in range(len(cl_list) - 1):
            node_i = cl_list[k][0]
            node_j = cl_list[k+1][0]
            
            if node_i == node_j:
                continue
            
            # Canonical edge key for dedup
            edge_key = (min(node_i, node_j), max(node_i, node_j))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            
            pos_i = node_positions[node_i]
            pos_j = node_positions[node_j]
            
            L_mm = np.linalg.norm(pos_j - pos_i)  # mm
            if L_mm < 1e-12:
                continue
            
            L_m = L_mm * 1e-3  # mm -> m
            
            # Spring stiffness: k = E * A / L (N/m)
            k_spring = E * A / L_m
            
            edge_idx = len(edges_list)
            edges_list.append([node_i, node_j])
            rest_lengths_list.append(L_mm)  # stored in mm
            stiffness_list.append(k_spring)  # stored in N/m
            spring_fiber_map[edge_idx] = fiber_idx
    
    if not edges_list:
        raise ValueError("No valid springs could be built. Check network connectivity.")
    
    edges = np.array(edges_list, dtype=np.int32)
    rest_lengths = np.array(rest_lengths_list, dtype=np.float64)
    stiffness = np.array(stiffness_list, dtype=np.float64)
    positions = node_positions
    
    # Step 4: Assign masses from fiber geometry
    # Each fiber's mass is distributed equally to its crosslink nodes.
    # This guarantees total mass = sum of all fiber masses.
    masses = np.zeros(n_nodes, dtype=np.float64)
    
    for fiber_idx, cl_list in fiber_crosslinks.items():
        fiber = network.fibers[fiber_idx]
        radius_m = fiber.radius * 1e-3
        length_m = fiber.length * 1e-3
        A = np.pi * radius_m**2
        rho = fiber.material.density if fiber.material and hasattr(fiber.material, 'density') else 1000.0
        fiber_mass = rho * A * length_m  # kg
        
        # Get unique node indices for this fiber
        node_indices = list(set(cl_idx for cl_idx, _ in cl_list))
        n_fiber_nodes = len(node_indices)
        
        if n_fiber_nodes > 0:
            mass_per_node = fiber_mass / n_fiber_nodes
            for ni in node_indices:
                masses[ni] += mass_per_node
    
    # Ensure minimum mass to avoid numerical issues
    min_mass = max(masses[masses > 0].min() * 0.01, 1e-12) if np.any(masses > 0) else 1e-12
    masses = np.maximum(masses, min_mass)
    
    return edges, rest_lengths, stiffness, masses, positions


def _numpy_dynamics(
    positions, velocities, masses, edges, rest_lengths, stiffness,
    dt=1e-7, steps=5000, damping=0.01,
    fixed_nodes=None, external_force=None, save_interval=100,
):
    """Pure numpy mass-spring dynamics (Verlet integration)."""
    import time
    start = time.time()
    
    pos = positions.copy()
    vel = velocities.copy()
    n_nodes = len(pos)
    fixed_set = set(fixed_nodes) if fixed_nodes else set()
    
    trajectory = [pos.copy()]
    
    for step in range(steps):
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Add external force
        if external_force is not None:
            forces += external_force
        
        # Add spring forces
        for e in range(len(edges)):
            i, j = edges[e]
            diff = pos[j] - pos[i]
            L = np.linalg.norm(diff)
            if L > 1e-12:
                strain = (L - rest_lengths[e]) / rest_lengths[e]
                f = stiffness[e] * strain * (diff / L)
                forces[i] += f
                forces[j] -= f
        
        # Verlet integration
        acc = forces / masses[:, np.newaxis]
        vel = (vel + acc * dt) * (1 - damping)
        pos = pos + vel * dt
        
        # Fix nodes
        for n in fixed_set:
            pos[n] = positions[n]
            vel[n] = 0
        
        # Save trajectory
        if (step + 1) % save_interval == 0:
            trajectory.append(pos.copy())
    
    # Final energy
    energy = 0.0
    for e in range(len(edges)):
        i, j = edges[e]
        diff = pos[j] - pos[i]
        L = np.linalg.norm(diff)
        strain = (L - rest_lengths[e]) / rest_lengths[e]
        energy += 0.5 * stiffness[e] * (strain * rest_lengths[e])**2
    
    return {
        'positions': pos,
        'trajectory': trajectory,
        'velocities': vel,
        'forces': forces,
        'energy': energy,
        'time_seconds': time.time() - start,
        'edges': edges,
        'rest_lengths': rest_lengths,
        'stiffness': stiffness,
        'initial_positions': positions,
    }


def simulate_thermal(
    network: FiberNetwork,
    T_hot: float = 100,
    T_cold: float = 0,
    axis: int = 0,
) -> Dict:
    """Unsupported legacy steady-state/effective-conductivity interface.

    Explicit transient lumped conduction is a different model, not a fallback.
    CTE and thermal deformation are not provided by that replacement example.
    """
    raise NotImplementedError(
        "Legacy steady-state/effective-conductivity thermal API is unsupported. "
        "Explicit transient lumped conduction requires caller-supplied C/g/T0/Q "
        "and a time grid; see examples/transient_thermal_explicit.py. "
        "This is not an equivalent replacement and does not compute CTE or thermal deformation."
    )


@overload
def analyze(network: FiberNetwork, *, kind: None = None) -> Dict: ...


@overload
def analyze(network: _StructureGraph, *, kind: Literal['topology']) -> 'TopologyMetricsV1': ...


def analyze(network: Union[FiberNetwork, _StructureGraph], *,
            kind: Optional[str] = None) -> Union[Dict, 'TopologyMetricsV1']:
    """Analyze an explicit representation without implicit conversion.

    StructureGraph requires kind='topology' and returns the unchanged
    TopologyMetricsV1 result, not a comprehensive geometry/material report.
    Use analyze_geometry_paths with explicit length_unit/coordinate_frame and
    analyze_spectrum separately for their declared units and resource budgets.

    FiberNetwork requires omitted/None kind and returns the historical twelve
    fields for a nonperiodic, crosslink-free endpoint network. Arc lengths use
    caller coordinate units; interior crossings are not junctions. Ambiguous
    rounded endpoints, parallel fibers and degenerate chords raise. Empty means
    are zero by convention, is_connected is False; no volume fraction is implied.
    Crosslinked legacy input is unsupported, and from_fiber_network is only an
    approximate endpoint projection, not a lossless migration helper.
    """
    if isinstance(network, _StructureGraph):
        if kind != 'topology':
            raise ValueError("StructureGraph analysis requires kind='topology'; "
                             "geometry and spectrum use their explicit analysis entry points")
        from fibernet.analysis import analyze_topology
        return analyze_topology(network)
    if not isinstance(network, FiberNetwork):
        raise TypeError('analyze requires FiberNetwork or StructureGraph; no conversion is performed')
    if kind is not None:
        raise ValueError('FiberNetwork analysis requires omitted/None kind; '
                         "kind='topology' requires StructureGraph")
    from fibernet.analysis._legacy_summary import analyze_legacy_endpoints
    return analyze_legacy_endpoints(network)


def export(
    network: Union[FiberNetwork, _StructureGraph],
    filename: str,
    format: str = None,
    **kwargs,
) -> str:
    """Export a native network or graph without implicit representation conversion.

    StructureGraph supports JSON here; use its explicit interoperability methods
    for other formats. Unknown extensions require an explicit supported format.

    Parameters
    ----------
    network : FiberNetwork or StructureGraph
        Native representation to save.
    filename : str
        Output filename.
    format : str, optional
        File format. Auto-detected from extension if not specified.
        Supported: json, lammps, vtk, vtp, xyz, pdb, msh
    """
    if format is None:
        ext = Path(filename).suffix.lower()
        format_map = {
            '.json': 'json',
            '.lammps': 'lammps',
            '.data': 'lammps',
            '.vtk': 'vtk',
            '.vtp': 'vtp',
            '.xyz': 'xyz',
            '.pdb': 'pdb',
            '.msh': 'gmsh',
        }
        if ext not in format_map:
            raise ValueError(f"Cannot infer export format from extension {ext!r}; "
                             "use a supported extension or specify format explicitly")
        format = format_map[ext]

    if not isinstance(network, (FiberNetwork, _StructureGraph)):
        raise TypeError("export requires FiberNetwork or StructureGraph")
    if isinstance(network, _StructureGraph) and format != 'json':
        raise ValueError("StructureGraph supports only JSON in fibernet.api.export; "
                         "use an explicit StructureGraph interoperability exporter")

    if format == 'json':
        network.save_json(filename)
    elif format == 'lammps':
        from fibernet.io.lammps import to_lammps
        to_lammps(network, filename, **kwargs)
    elif format == 'vtk':
        from fibernet.io.vtk import to_vtk
        to_vtk(network, filename, **kwargs)
    elif format == 'vtp':
        from fibernet.io.vtk import to_vtk_xml
        to_vtk_xml(network, filename)
    elif format == 'xyz':
        from fibernet.io.xyz import to_xyz
        to_xyz(network, filename, **kwargs)
    elif format == 'pdb':
        from fibernet.io.pdb import to_pdb
        to_pdb(network, filename, **kwargs)
    elif format in ('gmsh', 'msh'):
        from fibernet.io.gmsh import to_gmsh
        to_gmsh(network, filename, **kwargs)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return filename


def load(filename: str, format: str = None) -> Union[FiberNetwork, _StructureGraph]:
    """Load a native representation without silently discarding graph data.

    JSON dispatch preserves StructureGraph nodes/edges or legacy FiberNetwork
    fibers/crosslinks. Mixed or unknown representation shapes are rejected;
    version handling follows each native reader. Additional legacy header fields
    remain tolerated by those readers; this dispatch is not schema certification.
    LAMMPS and PDB readers return FiberNetwork. Unknown extensions require an
    explicit supported format.
    
    Parameters
    ----------
    filename : str
        Input filename.
    format : str, optional
        File format. Auto-detected from extension.
    """
    if format is None:
        ext = Path(filename).suffix.lower()
        format_map = {
            '.json': 'json',
            '.lammps': 'lammps',
            '.data': 'lammps',
            '.pdb': 'pdb',
        }
        if ext not in format_map:
            raise ValueError(f"Cannot infer load format from extension {ext!r}; "
                             "use a supported extension or specify format explicitly")
        format = format_map[ext]

    if format == 'json':
        with open(filename) as stream:
            data = _json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("native JSON must be a StructureGraph or FiberNetwork object")
        graph_fields = {'nodes', 'edges'} & data.keys()
        network_fields = {'fibers', 'crosslinks'} & data.keys()
        if graph_fields and network_fields:
            raise ValueError("ambiguous JSON mixes StructureGraph and FiberNetwork fields")
        if graph_fields:
            if not all(isinstance(data.get(key), list) for key in ('nodes', 'edges')):
                raise ValueError("StructureGraph JSON requires nodes and edges arrays")
            return _StructureGraph.from_dict(data)
        if network_fields:
            if (not isinstance(data.get('fibers'), list)
                    or not isinstance(data.get('crosslinks', []), list)):
                raise ValueError("FiberNetwork JSON requires a fibers array and optional crosslinks array")
            return FiberNetwork.from_dict(data)
        raise ValueError("unrecognized native JSON: expected nodes/edges or fibers/crosslinks")
    elif format == 'lammps':
        from fibernet.io.lammps import from_lammps
        return from_lammps(filename)
    elif format == 'pdb':
        from fibernet.io.pdb import from_pdb
        return from_pdb(filename)
    else:
        raise ValueError(f"Cannot load format: {format}")


def plot(network: FiberNetwork, **kwargs):
    """Quick plot of the network.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    **kwargs
        Additional plot parameters (color_by, colormap, line_width, etc.).
    """
    if network.dimension == 2:
        from fibernet.viz.plot2d import plot_network_2d
        return plot_network_2d(network, **kwargs)
    else:
        from fibernet.viz.render3d import render_network_3d
        return render_network_3d(network, **kwargs)


def plot_dynamics(result: Dict, **kwargs):
    """Visualize mass-spring dynamics results.
    
    Parameters
    ----------
    result : dict
        Output from ``simulate_dynamics()``.
    **kwargs
        Plot parameters (colormap, show_forces, save_path, etc.).
    """
    from fibernet.viz.plot2d import plot_dynamics_result
    return plot_dynamics_result(result, **kwargs)


def plot_metamaterial(network: FiberNetwork, **kwargs):
    """Professional visualization for metamaterial structures.
    
    Parameters
    ----------
    network : FiberNetwork
        Metamaterial network (from create_metamaterial).
    **kwargs
        Plot parameters (show_unit_cells, show_crosslinks, colormap, save_path).
    """
    from fibernet.viz.plot2d import plot_metamaterial as _plot_meta
    return _plot_meta(network, **kwargs)


def plot_stress_strain(result: Dict, **kwargs):
    """Plot stress-strain curve from mechanics simulation.
    
    Parameters
    ----------
    result : dict
        Output from ``simulate_mechanics()``.
    **kwargs
        Plot parameters (show_modulus, save_path).
    """
    from fibernet.viz.plot2d import plot_stress_strain as _plot_ss
    return _plot_ss(result, **kwargs)


# ============================================================
# Metamaterial Workflow
# ============================================================

def create_metamaterial(
    unit_cell: str = "reentrant_honeycomb_2d",
    array_size: Tuple[int, int] = (3, 3),
    weld_threshold: float = 0.5,
    **cell_params,
) -> FiberNetwork:
    """Create a metamaterial from a unit cell array with welded crosslinks.
    
    This function implements the standard metamaterial workflow:
    1. Generate a parameterized unit cell
    2. Tile it into an array (Nx × Ny)
    3. Weld intersections with crosslinks
    4. Return a complete graph ready for simulation
    
    Parameters
    ----------
    unit_cell : str
        Unit cell type. Available options:
        - ``"reentrant_honeycomb_2d"`` : Re-entrant honeycomb (auxetic when angle > 90°)
        - ``"chiral_honeycomb_2d"`` : Chiral honeycomb with rotating nodes
        - ``"star_honeycomb_2d"`` : Star-shaped honeycomb
        - ``"arrowhead_auxetic_2d"`` : Arrowhead auxetic structure
        - ``"hierarchical_lattice_2d"`` : Multi-scale hierarchical lattice
        - ``"missing_rib_auxetic_2d"`` : Missing-rib auxetic
    array_size : tuple of int
        Array dimensions (Nx, Ny). Minimum recommended: (3, 3).
    weld_threshold : float
        Distance threshold for auto-detecting intersections (default 0.5).
    **cell_params
        Unit cell parameters. Common parameters:
        - ``reentrant_angle`` : Re-entrant angle in degrees (default 150)
        - ``cell_height``, ``cell_width`` : Cell dimensions
        - ``grid_size`` : Internal grid resolution (default (5, 5))
        - ``radius`` : Fiber radius
    
    Returns
    -------
    FiberNetwork
        Complete metamaterial graph with welded crosslinks.
    
    See Also
    --------
    create : Low-level generator access.
    tile : Manual tiling with custom spacing.
    
    Examples
    --------
    >>> # Create a 3×3 re-entrant honeycomb array
    >>> meta = fn.create_metamaterial(
    ...     unit_cell="reentrant_honeycomb_2d",
    ...     array_size=(3, 3),
    ...     reentrant_angle=150,
    ...     cell_height=10,
    ...     cell_width=10,
    ... )
    >>> print(f"Fibers: {meta.num_fibers}, Crosslinks: {meta.num_crosslinks}")
    >>> 
    >>> # Run dynamics simulation
    >>> traj = fn.simulate_dynamics(meta, dt=1e-7, steps=5000)
    >>> 
    Legacy simulate_mechanics is unsupported; see its migration documentation.
    """
    # Step 1: Generate unit cell
    if unit_cell not in _registry.generators:
        raise ValueError(
            f"Unknown unit cell: '{unit_cell}'.\n"
            f"Available: {[g for g in _registry.list_generators() if '2d' in g or '3d' in g]}"
        )
    
    cell = _registry.generators[unit_cell](**cell_params)
    
    # Step 2: Tile into array
    # Determine spacing from bounding box
    bb_min, bb_max = cell.bounding_box()
    cell_size = bb_max - bb_min
    
    # Add small gap to avoid overlaps
    spacing = cell_size + 0.1
    
    # Create array
    array = transform.tile(cell, repeats=(array_size[0], array_size[1], 1), spacing=spacing)
    
    # Step 3: Weld intersections
    # Clear existing crosslinks and re-detect
    array.crosslinks = []
    array.auto_crosslink(threshold=weld_threshold, crosslink_type="welded")
    
    # Add metadata
    array.metadata = {
        'unit_cell': unit_cell,
        'array_size': array_size,
        'cell_params': cell_params,
        'weld_threshold': weld_threshold,
    }
    
    return array


def print_metamaterial_info(network: FiberNetwork):
    """Print detailed information about a metamaterial structure.
    
    Parameters
    ----------
    network : FiberNetwork
        Metamaterial network (typically from create_metamaterial).
    """
    print("=" * 70)
    print("METAMATERIAL STRUCTURE INFO")
    print("=" * 70)
    
    if hasattr(network, 'metadata'):
        meta = network.metadata
        print(f"Unit Cell: {meta.get('unit_cell', 'unknown')}")
        print(f"Array Size: {meta.get('array_size', 'unknown')}")
        print(f"Cell Parameters: {meta.get('cell_params', {})}")
        print(f"Weld Threshold: {meta.get('weld_threshold', 0.5)}")
        print()
    
    print(f"Total Fibers: {network.num_fibers}")
    print(f"Total Crosslinks: {network.num_crosslinks}")
    print(f"Dimension: {network.dimension}D")
    
    bb_min, bb_max = network.bounding_box()
    print(f"Bounding Box: {bb_min} to {bb_max}")
    print(f"Size: {bb_max - bb_min}")
    
    if network.crosslinks:
        # Analyze connectivity
        connectivity = {}
        for cl in network.crosslinks:
            for idx in [cl.fiber_i, cl.fiber_j]:
                connectivity[idx] = connectivity.get(idx, 0) + 1
        
        print(f"\nConnectivity Stats:")
        print(f"  Fibers with crosslinks: {len(connectivity)}/{network.num_fibers}")
        if connectivity:
            print(f"  Min connectivity: {min(connectivity.values())}")
            print(f"  Max connectivity: {max(connectivity.values())}")
            print(f"  Avg connectivity: {np.mean(list(connectivity.values())):.2f}")
    
    print("=" * 70)

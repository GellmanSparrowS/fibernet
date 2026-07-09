"""
High-level convenience API for FiberNet.

Design Philosophy
-----------------
FiberNet uses a **registry pattern** to make the API extensible:
- ``create()`` dispatches to registered generators
- ``simulate()`` dispatches to registered simulation backends
- Third-party plugins can register via ``register_generator()`` / ``register_backend()``

This makes it easy to add new structure types, simulation methods (e.g. truss FEM,
continuum FEM, LAMMPS, GROMACS), and ML models without modifying core code.

Quick Start
-----------
>>> import fibernet as fn
>>> 
>>> # 1. Create a structure
>>> net = fn.create("reentrant_honeycomb_2d", reentrant_angle=150)
>>> 
>>> # 2. Run mechanics (beam FEM)
>>> result = fn.simulate_mechanics(net, strain=0.001)
>>> print(f"E = {result['modulus']:.2e} Pa")
>>> 
>>> # 3. Run dynamics (mass-spring, Taichi-accelerated)
>>> traj = fn.simulate_dynamics(net, dt=1e-7, steps=5000)
>>> 
>>> # 4. Analyze structure
>>> stats = fn.analyze(net)
>>> 
>>> # 5. Visualize
>>> fn.plot(net, color_by="orientation")
"""

import numpy as np
from typing import Optional, Dict, List, Union, Tuple, Callable, Any
from pathlib import Path
from dataclasses import dataclass, field

from fibernet.core.network import FiberNetwork
from fibernet.core.fiber import Fiber
from fibernet.core.material import Material
from fibernet import gen
from fibernet.core import transform


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
    >>> @fn.register_generator("my_custom_lattice")
    ... def make_my_lattice(**kwargs):
    ...     return FiberNetwork(...)
    >>> 
    >>> # Use it
    >>> net = fn.create("my_custom_lattice", param1=42)
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
    >>> @fn.register_generator("my_lattice")
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
    >>> @fn.register_backend("my_fem")
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
    >>> net = fn.create("reentrant_honeycomb_2d", reentrant_angle=150, grid_size=(5, 5))
    >>> net = fn.create("random_2d", num_fibers=200, fiber_length=10.0)
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
    """Run mechanical simulation on a fiber network.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    strain : float
        Applied strain (default 0.01 = 1%).
    axis : int
        Loading direction (0=x, 1=y, 2=z).
    model : str
        Constitutive model:
        - ``"linear"`` : Linear elastic beam FEM (Euler-Bernoulli)
        - ``"bilinear"`` : Bilinear elastoplastic
        - ``"neo_hookean"`` : Hyperelastic (large deformation)
    **kwargs
        Additional model parameters (e.g. ``segments_per_fiber``).
    
    Returns
    -------
    dict
        ``modulus``, ``max_stress``, ``max_displacement``, ``energy``,
        ``stress_strain``, ``fem``.
    """
    from fibernet.sim.mechanical import FiberFEM
    
    if model == "linear":
        seg = kwargs.get('segments_per_fiber', 5)
        fem = FiberFEM(network, segments_per_fiber=seg)
        result = fem.apply_uniaxial_strain(strain=strain, axis=axis)
        E = fem.effective_modulus(strain=strain, axis=axis)
        return {
            'modulus': E,
            'max_stress': result.max_stress() if hasattr(result, 'max_stress') else 0,
            'max_displacement': result.max_displacement() if hasattr(result, 'max_displacement') else 0,
            'energy': result.energy if hasattr(result, 'energy') else 0,
            'displacements': result.displacements,
            'node_positions': fem.node_positions,
            'fem': fem,
        }
    else:
        from fibernet.sim.nonlinear import (
            NonlinearFEM, LinearElastic, BilinearPlasticity, HyperelasticNeoHookean,
        )
        model_map = {
            'bilinear': BilinearPlasticity(),
            'neo_hookean': HyperelasticNeoHookean(),
        }
        constitutive = model_map.get(model, LinearElastic())
        fem = NonlinearFEM(network, constitutive_model=constitutive)
        strains, stresses = fem.compute_stress_strain(
            max_strain=strain, num_steps=kwargs.get('num_steps', 20)
        )
        return {
            'stress_strain': (strains, stresses),
            'max_stress': stresses[-1] if len(stresses) > 0 else 0,
            'fem': fem,
        }


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
    """Run mass-spring dynamics simulation.
    
    Models the fiber network as a spring-mass system where:
    - Crosslinks become point masses
    - Fibers become springs with stiffness derived from Young's modulus
    - Spring rest length = fiber segment length
    - Spring stiffness k = E * A / L (axial rigidity)
    
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
        ``"taichi"`` for GPU/CPU parallel, ``"numpy"`` for pure Python.
    
    Returns
    -------
    dict
        ``positions`` (final), ``trajectory`` (list of position snapshots),
        ``velocities``, ``forces``, ``energy``, ``time_seconds``.
    
    See Also
    --------
    simulate_mechanics : Static FEM analysis.
    
    Examples
    --------
    >>> net = fn.create("reentrant_honeycomb_2d", reentrant_angle=150)
    >>> traj = fn.simulate_dynamics(net, dt=1e-7, steps=5000, damping=0.05)
    >>> print(f"Final energy: {traj['energy']:.2e} J")
    """
    from fibernet.sim.accelerated import TaichiEngine, HAS_TAICHI
    
    if backend == "taichi" and not HAS_TAICHI:
        raise ImportError(
            "Taichi required for GPU/CPU parallel dynamics. "
            "Install: pip install taichi\n"
            "Or use backend='numpy' for pure Python."
        )
    
    # Build spring-mass system from network
    edges, rest_lengths, stiffness, masses, positions = _build_mass_spring_system(network)
    
    if fixed_nodes is None:
        fixed_nodes = []
    
    velocities = np.zeros_like(positions)
    
    if external_force is not None:
        velocities = velocities  # placeholder
    
    if backend == "taichi":
        engine = TaichiEngine(arch=kwargs.get("arch", "cpu"))
        result = engine.parallel_dynamics(
            positions=positions,
            velocities=velocities,
            masses=masses,
            rest_lengths=rest_lengths,
            stiffness=stiffness,
            edges=edges,
            dt=dt,
            damping=damping,
            num_steps=steps,
            save_interval=save_interval,
            fixed_nodes=fixed_nodes,
            external_force=external_force,
        )
        return {
            'positions': result.positions[-1] if result.positions else positions,
            'trajectory': result.positions or [],
            'velocities': result.forces if result.forces is not None else np.zeros_like(positions),
            'energy': result.energy,
            'time_seconds': result.time_seconds,
            'edges': edges,
            'rest_lengths': rest_lengths,
            'stiffness': stiffness,
            'initial_positions': positions,
        }
    else:
        # Pure numpy fallback
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
    """Run thermal simulation."""
    from fibernet.sim.thermal import ThermalSolver
    
    solver = ThermalSolver(network)
    result = solver.solve_steady_state(T_hot=T_hot, T_cold=T_cold, axis=axis)
    
    return {
        'conductivity': result.effective_conductivity,
        'temperatures': result.temperatures,
    }


def analyze(network: FiberNetwork) -> Dict:
    """Analyze network structure.
    
    Returns
    -------
    dict
        Analysis results with keys: num_fibers, num_crosslinks,
        nematic_order, mean_length, total_length, volume_fraction, etc.
    """
    from fibernet.analysis import MorphologyAnalyzer, TopologyAnalyzer
    
    morph = MorphologyAnalyzer(network)
    topo = TopologyAnalyzer(network)
    
    morph_report = morph.full_report()
    try:
        topo_report = topo.full_report()
    except ImportError:
        topo_report = {}
    
    return {
        'num_fibers': network.num_fibers,
        'num_crosslinks': network.num_crosslinks,
        'dimension': network.dimension,
        'nematic_order': morph_report.get('nematic_order', 0),
        'mean_length': morph_report.get('mean_length', 0),
        'total_length': morph_report.get('total_length', 0),
        'mean_tortuosity': morph_report.get('mean_tortuosity', 1.0),
        'num_nodes': topo_report.get('num_nodes', 0),
        'num_edges': topo_report.get('num_edges', 0),
        'mean_degree': topo_report.get('degree_stats', {}).get('mean', 0),
        'is_connected': topo_report.get('is_connected', False),
        'num_components': topo_report.get('num_components', 0),
    }


def export(
    network: FiberNetwork,
    filename: str,
    format: str = None,
    **kwargs,
) -> str:
    """Export network to file.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
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
        format = format_map.get(ext, 'json')
    
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


def load(filename: str, format: str = None) -> FiberNetwork:
    """Load network from file.
    
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
        format = format_map.get(ext, 'json')
    
    if format == 'json':
        return FiberNetwork.load_json(filename)
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
    >>> # Run mechanics
    >>> result = fn.simulate_mechanics(meta, strain=0.001)
    >>> print(f"Modulus: {result['modulus']:.2e} Pa")
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

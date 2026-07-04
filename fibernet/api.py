"""
High-level convenience API for FiberNet.

Provides easy access to common operations:
- fibernet.create() - Create networks from generators
- fibernet.simulate() - Run simulations
- fibernet.analyze() - Analyze structures
- fibernet.export() - Export to various formats
- fibernet.load() - Load from files
"""

import numpy as np
from fibernet.utils.exceptions import (
    EmptyNetworkError, InvalidParameterError,
    check_nonempty, check_positive, check_range, check_integer
)
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path

from fibernet.core.network import FiberNetwork
from fibernet.core.fiber import Fiber
from fibernet.core.material import Material
from fibernet import gen
from fibernet.core import transform


def create(
    generator: str = "random_2d",
    **kwargs,
) -> FiberNetwork:
    """Create a fiber network using a generator.
    
    Parameters
    ----------
    generator : str
        Generator name. Available:
        - "random_2d", "random_3d" - Random Mikado model
        - "random_walk" - Random walk polymer
        - "oriented_2d", "oriented_3d" - Oriented random
        - "square_2d" - Square lattice
        - "honeycomb_2d" - Honeycomb lattice
        - "triangular_2d" - Triangular lattice
        - "cubic_3d" - Cubic lattice
        - "octet_3d" - Octet truss
        - "kagome_2d" - Kagome lattice
        - "helix" - Single helix
        - "double_helix" - DNA-like double helix
        - "plain_weave" - Plain weave
        - "twill_weave" - Twill weave
        - "voronoi_2d" - Voronoi tessellation
        - "electrospun" - Electrospun nanofiber
        - "biomimetic_collagen" - Biomimetic collagen
    **kwargs
        Generator parameters.
    
    Returns
    -------
    FiberNetwork
        Generated network.
    """
    generators = {
        "random_2d": gen.random_straight_2d,
        "random_3d": gen.random_straight_3d,
        "random_walk": gen.random_walk_fibers,
        "oriented_2d": gen.oriented_random_2d,
        "square_2d": gen.square_lattice_2d,
        "honeycomb_2d": gen.honeycomb_lattice_2d,
        "triangular_2d": gen.triangular_lattice_2d,
        "cubic_3d": gen.cubic_lattice_3d,
        "octet_3d": gen.octet_truss_3d,
        "kagome_2d": gen.kagome_lattice_2d,
        "helix": gen.single_helix,
        "double_helix": gen.double_helix,
        "plain_weave": gen.plain_weave_2d,
        "twill_weave": gen.twill_weave_2d,
    }
    
    if generator not in generators:
        raise ValueError(f"Unknown generator: {generator}. Available: {list(generators.keys())}")
    
    return generators[generator](**kwargs)


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


def tile(network: FiberNetwork, repeats: Tuple[int, int, int] = (2, 2, 1)) -> FiberNetwork:
    """Tile a network periodically."""
    return transform.tile(network, repeats=repeats)


def simulate_mechanics(
    network: FiberNetwork,
    strain: float = 0.01,
    axis: int = 0,
    model: str = "linear",
    **kwargs,
) -> Dict:
    """Run mechanical simulation.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    strain : float
        Applied strain.
    axis : int
        Loading direction.
    model : str
        Constitutive model: "linear", "bilinear", "neo_hookean".
    
    Returns
    -------
    dict
        Simulation results with keys: modulus, max_stress, energy, etc.
    """
    from fibernet.sim.mechanical import FiberFEM
    from fibernet.sim.nonlinear import (
        NonlinearFEM, LinearElastic, BilinearPlasticity, HyperelasticNeoHookean,
    )
    
    if model == "linear":
        fem = FiberFEM(network, segments_per_fiber=kwargs.get('segments', 5))
        result = fem.apply_uniaxial_strain(strain=strain, axis=axis)
        E = fem.effective_modulus(strain=strain, axis=axis)
        
        return {
            'modulus': E,
            'max_stress': result.max_stress(),
            'max_displacement': result.max_displacement(),
            'energy': result.energy,
            'displacements': result.displacements,
            'fem': fem,
        }
    
    elif model in ["bilinear", "plastic", "neo_hookean"]:
        models = {
            "bilinear": BilinearPlasticity,
            "plastic": BilinearPlasticity,
            "neo_hookean": HyperelasticNeoHookean,
        }
        
        model_cls = models[model]
        model_params = {k: v for k, v in kwargs.items() if k not in ['segments']}
        constitutive = model_cls(**model_params)
        
        fem = NonlinearFEM(
            network,
            constitutive_model=constitutive,
            segments_per_fiber=kwargs.get('segments', 5),
        )
        
        strains, stresses, energies = fem.stress_strain_curve(
            axis=axis, max_strain=strain, num_steps=kwargs.get('steps', 20),
        )
        
        return {
            'strains': strains,
            'stresses': stresses,
            'energies': energies,
            'modulus': stresses[0] / strains[0] if len(strains) > 0 else 0,
            'max_stress': stresses[-1] if len(stresses) > 0 else 0,
            'fem': fem,
        }
    
    else:
        raise ValueError(f"Unknown model: {model}")


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
    # Validate input
    check_nonempty(network, 'analyze')

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
    elif format == 'gmsh' or format == 'msh':
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
        Additional plot parameters.
    """
    if network.dimension == 2:
        from fibernet.viz.plot2d import plot_network_2d
        return plot_network_2d(network, **kwargs)
    else:
        from fibernet.viz.render3d import render_network_3d
        return render_network_3d(network, **kwargs)

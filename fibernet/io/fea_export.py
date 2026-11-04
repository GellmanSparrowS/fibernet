"""
Legacy FEA templates: independent straight fibers, not validated solver decks

Provides export capabilities for:
- Abaqus (.inp)
- ANSYS (.dat, .cdb)
- Gmsh (.msh)
- LAMMPS (.lmp): unsupported; raises before opening a destination

Curves, backtracking and declared crosslinks are rejected before file I/O.
Coincident endpoints remain separate nodes. Abaqus/ANSYS require uniform
circular isotropic scalar beam properties; Gmsh writes geometry only.
These checks do not establish units, boundary-condition suitability or solver
acceptance. Numeric values use round-trip float64 text precision. Abaqus
rejects fibers parallel to its fixed Z-axis section orientation (within 64
machine eps of the normalized XY direction). Preserve source data using JSON.

References:
- Abaqus Documentation: https://help.3ds.com/
- ANSYS Documentation: https://www.ansys.com/
- Gmsh: https://gmsh.info/
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional
from numbers import Real
from fibernet.core.network import FiberNetwork


def _beam_properties(fiber):
    """Only the circular, isotropic scalar properties written by these templates."""
    from fibernet.core.fiber import CrossSection
    if fiber.cross_section != CrossSection.CIRCULAR:
        raise ValueError('legacy FEA beam templates support circular sections only')
    material = fiber.material
    if material is not None and material.stiffness_tensor is not None:
        raise ValueError('legacy FEA beam templates do not export anisotropic stiffness')
    values = (fiber.radius, 1e9 if material is None else material.youngs_modulus,
              0.3 if material is None else material.poissons_ratio,
              1000.0 if material is None else material.density)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
           or not np.isfinite(v) for v in values):
        raise ValueError('legacy FEA beam properties must be finite real scalars')
    radius, young, poisson, density = values
    if radius <= 0 or young <= 0 or density <= 0 or not -1 < poisson < 0.5:
        raise ValueError('legacy FEA beam properties are outside the supported domain')
    return tuple(float(v) for v in values)


def _preflight(network, segments_per_fiber, *, beam_properties=False, abaqus_orientation=False):
    """Reject unsupported source data before any destination can be opened.

    This is input admission, not commercial solver/deck validation. Fibers stay
    independent even when endpoints coincide. Only floating-point roundoff is
    allowed when checking ordered straight sampled centerlines.
    """
    if not isinstance(network, FiberNetwork):
        raise TypeError('legacy FEA export requires FiberNetwork; preserve native JSON instead')
    if (isinstance(segments_per_fiber, (bool, np.bool_))
            or not isinstance(segments_per_fiber, (int, np.integer))
            or segments_per_fiber <= 0):
        raise ValueError('segments_per_fiber must be a positive integer')
    if network.periodic:
        raise ValueError('legacy FEA export does not encode periodic boundary conditions or adjacency')
    if beam_properties and not network.fibers:
        raise ValueError('legacy beam export requires at least one fiber')
    if network.crosslinks:
        raise ValueError('legacy FEA export cannot encode crosslinks or welded incidence; preserve native JSON')
    first_properties = None
    for fiber in network.fibers:
        points = np.asarray(fiber.centerline, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                or not np.isfinite(points).all()):
            raise ValueError('legacy FEA export requires finite 3D centerlines')
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            chord = points[-1] - points[0]
            length = np.linalg.norm(chord)
            if not np.isfinite(length) or length <= 0:
                raise ValueError('legacy FEA export requires nondegenerate straight fibers')
            relative = points - points[0]
            distance = relative @ (chord / length)
            residual = relative - distance[:, None] * (chord / length)
            eps = 64 * np.finfo(float).eps * max(float(np.max(np.abs(relative))), float(length))
        if (not np.isfinite(distance).all() or not np.isfinite(residual).all()
                or np.any(np.abs(residual) > eps) or np.any(distance < -eps)
                or np.any(distance > length + eps) or np.any(np.diff(distance) < -eps)):
            raise ValueError('legacy FEA export requires ordered straight centerlines; preserve native JSON')
        if abaqus_orientation and np.linalg.norm((chord / length)[:2]) <= 64 * np.finfo(float).eps:
            raise ValueError('legacy Abaqus fixed orientation (0, 0, 1) cannot support Z-parallel fibers')
        if beam_properties:
            properties = _beam_properties(fiber)
            if first_properties is None:
                first_properties = properties
            elif properties != first_properties:
                raise ValueError('legacy FEA beam templates require uniform radius, E, nu and density')


def export_to_abaqus(
    network: FiberNetwork,
    filename: Union[str, Path],
    job_name: str = "FiberNetwork",
    material_name: str = "FIBER_MATERIAL",
    segments_per_fiber: int = 5,
    beam_type: str = "B31",
    element_size: float = 1.0,
) -> None:
    """
    Export fiber network to Abaqus input file (.inp).
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to export
    filename : str or Path
        Output filename (should end with .inp)
    job_name : str
        Abaqus job name
    material_name : str
        Material name in Abaqus
    segments_per_fiber : int
        Number of beam elements per fiber
    beam_type : str
        Only two-node B31 is supported.
    element_size : float
        Unused legacy placeholder; only the default 1.0 is accepted.
        Use segments_per_fiber to choose the number of straight elements.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.io.fea_export import export_to_abaqus
    >>> 
    >>> net = gen.random_straight_3d(num_fibers=50, box_size=(50, 50, 50))
    >>> export_to_abaqus(net, "network.inp", job_name="fiber_analysis")
    """
    if beam_type != 'B31':
        raise ValueError('legacy Abaqus export supports only two-node B31 elements')
    if (isinstance(element_size, (bool, np.bool_)) or not isinstance(element_size, Real)
            or not np.isfinite(element_size) or element_size != 1.0):
        raise ValueError('element_size is an unused legacy placeholder; only its default 1.0 is accepted')
    _preflight(network, segments_per_fiber, beam_properties=True, abaqus_orientation=True)
    path = Path(filename)
    
    with open(path, 'w') as f:
        # Header
        f.write(f"*HEADING\n")
        f.write(f"** Abaqus Input File Generated by FiberNet\n")
        f.write(f"** Network: {len(network.fibers)} fibers\n")
        f.write(f"** Job Name: {job_name}\n")
        f.write(f"**\n")
        
        # Part definition
        f.write(f"*PART, NAME=FIBER_NETWORK\n")
        
        # Nodes
        f.write(f"*NODE\n")
        node_id = 1
        node_map = {}  # Map (fiber_id, segment_id) -> node_id
        
        for fiber_idx, fiber in enumerate(network.fibers):
            # Generate nodes along fiber
            for seg_id in range(segments_per_fiber + 1):
                t = seg_id / segments_per_fiber
                # Linear interpolation
                position = (fiber.start_point if seg_id == 0 else
                            fiber.end_point if seg_id == segments_per_fiber else
                            fiber.start_point + t * (fiber.end_point - fiber.start_point))
                f.write(f"{node_id}, {position[0]:.17g}, {position[1]:.17g}, {position[2]:.17g}\n")
                node_map[(fiber_idx, seg_id)] = node_id
                node_id += 1
        
        # Elements
        f.write(f"*ELEMENT, TYPE={beam_type}, ELSET=ALL_FIBERS\n")
        element_id = 1
        for fiber_idx in range(len(network.fibers)):
            for seg_id in range(segments_per_fiber):
                node1 = node_map[(fiber_idx, seg_id)]
                node2 = node_map[(fiber_idx, seg_id + 1)]
                f.write(f"{element_id}, {node1}, {node2}\n")
                element_id += 1
        
        # Beam sections
        f.write(f"*BEAM SECTION, SECTION=CIRC, ELSET=ALL_FIBERS, MATERIAL={material_name}\n")
        if len(network.fibers) > 0:
            radius = network.fibers[0].radius
            f.write(f"{radius:.17g}\n")
        else:
            f.write(f"0.5\n")
        f.write(f"0.0, 0.0, 1.0\n")  # Beam orientation
        
        f.write(f"*END PART\n")
        f.write(f"**\n")
        
        # Assembly
        f.write(f"*ASSEMBLY, NAME=ASSEMBLY\n")
        f.write(f"*INSTANCE, NAME=FIBER_NETWORK-1, PART=FIBER_NETWORK\n")
        f.write(f"*END INSTANCE\n")
        f.write(f"*END ASSEMBLY\n")
        f.write(f"**\n")
        
        # Material definition
        f.write(f"*MATERIAL, NAME={material_name}\n")
        f.write(f"*ELASTIC\n")
        if len(network.fibers) > 0 and network.fibers[0].material is not None:
            E = network.fibers[0].material.youngs_modulus
            nu = network.fibers[0].material.poissons_ratio
            f.write(f"{E:.17g}, {nu:.17g}\n")
        else:
            f.write(f"1.0e9, 0.3\n")  # Default values
        
        f.write(f"*DENSITY\n")
        if len(network.fibers) > 0 and network.fibers[0].material is not None:
            rho = network.fibers[0].material.density
            f.write(f"{rho:.17g}\n")
        else:
            f.write(f"1000.0\n")
        
        f.write(f"**\n")
        
        # Step definition
        f.write(f"*STEP, NAME=StaticAnalysis\n")
        f.write(f"*STATIC\n")
        f.write(f"0.1, 1.0, 1e-5, 0.1\n")
        f.write(f"**\n")
        
        # Boundary conditions (fix first node)
        f.write(f"*BOUNDARY\n")
        f.write(f"ASSEMBLY.FIBER_NETWORK-1.1, ENCASTRE\n")
        f.write(f"**\n")
        
        # Output requests
        f.write(f"*OUTPUT, FIELD\n")
        f.write(f"*NODE OUTPUT\n")
        f.write(f"U, RF\n")
        f.write(f"*ELEMENT OUTPUT\n")
        f.write(f"S, E\n")
        f.write(f"**\n")
        
        f.write(f"*END STEP\n")


def export_to_ansys(
    network: FiberNetwork,
    filename: Union[str, Path],
    segments_per_fiber: int = 5,
    element_type: str = "BEAM188",
) -> None:
    """
    Export fiber network to ANSYS input file (.dat or .cdb).
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to export
    filename : str or Path
        Output filename
    segments_per_fiber : int
        Number of beam elements per fiber
    element_type : str
        Only two-node BEAM188 is supported.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.io.fea_export import export_to_ansys
    >>> 
    >>> net = gen.random_straight_3d(num_fibers=50, box_size=(50, 50, 50))
    >>> export_to_ansys(net, "network.dat")
    """
    if element_type != 'BEAM188':
        raise ValueError('legacy ANSYS export supports only two-node BEAM188 elements')
    _preflight(network, segments_per_fiber, beam_properties=True)
    path = Path(filename)
    
    with open(path, 'w') as f:
        f.write(f"! ANSYS Input File Generated by FiberNet\n")
        f.write(f"! Network: {len(network.fibers)} fibers\n")
        f.write(f"!\n")
        
        # Preprocessor
        f.write(f"/PREP7\n")
        f.write(f"!\n")
        
        # Element type
        f.write(f"ET, 1, {element_type}\n")
        f.write(f"!\n")
        
        # Material properties
        if len(network.fibers) > 0 and network.fibers[0].material is not None:
            E = network.fibers[0].material.youngs_modulus
            nu = network.fibers[0].material.poissons_ratio
            rho = network.fibers[0].material.density
        else:
            E, nu, rho = 1.0e9, 0.3, 1000.0
        radius = network.fibers[0].radius if network.fibers else 0.5
        
        f.write(f"MP, EX, 1, {E:.17g}\n")
        f.write(f"MP, PRXY, 1, {nu:.17g}\n")
        f.write(f"MP, DENS, 1, {rho:.17g}\n")
        f.write(f"!\n")
        
        # Beam section
        f.write(f"SECTYPE, 1, BEAM, CIRC\n")
        f.write(f"SECDATA, {radius:.17g}\n")
        f.write(f"!\n")
        
        # Nodes
        node_id = 1
        node_map = {}
        
        for fiber_idx, fiber in enumerate(network.fibers):
            for seg_id in range(segments_per_fiber + 1):
                t = seg_id / segments_per_fiber
                position = (fiber.start_point if seg_id == 0 else
                            fiber.end_point if seg_id == segments_per_fiber else
                            fiber.start_point + t * (fiber.end_point - fiber.start_point))
                f.write(f"N, {node_id}, {position[0]:.17g}, {position[1]:.17g}, {position[2]:.17g}\n")
                node_map[(fiber_idx, seg_id)] = node_id
                node_id += 1
        
        f.write(f"!\n")
        
        # Elements
        element_id = 1
        for fiber_idx in range(len(network.fibers)):
            for seg_id in range(segments_per_fiber):
                node1 = node_map[(fiber_idx, seg_id)]
                node2 = node_map[(fiber_idx, seg_id + 1)]
                f.write(f"E, {node1}, {node2}\n")
                element_id += 1
        
        f.write(f"!\n")
        
        # Boundary conditions
        f.write(f"D, 1, ALL, 0\n")  # Fix first node
        f.write(f"!\n")
        
        f.write(f"FINISH\n")
        f.write(f"!\n")
        
        # Solution
        f.write(f"/SOLU\n")
        f.write(f"ANTYPE, STATIC\n")
        f.write(f"SOLVE\n")
        f.write(f"FINISH\n")
        f.write(f"!\n")
        
        # Post-processing
        f.write(f"/POST1\n")
        f.write(f"PLNSOL, U, SUM\n")
        f.write(f"FINISH\n")


def export_to_gmsh(
    network: FiberNetwork,
    filename: Union[str, Path],
    segments_per_fiber: int = 5,
) -> None:
    """
    Export independent straight fibers as Gmsh MSH 2.2 ASCII line geometry.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to export
    filename : str or Path
        Output filename
    segments_per_fiber : int
        Number of segments per fiber
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.io.fea_export import export_to_gmsh
    >>> 
    >>> net = gen.random_straight_3d(num_fibers=50, box_size=(50, 50, 50))
    >>> export_to_gmsh(net, "network.msh")
    """
    _preflight(network, segments_per_fiber)
    if not network.fibers:
        raise ValueError('legacy Gmsh line export requires at least one fiber')
    path = Path(filename)
    
    with open(path, 'w') as f:
        # Gmsh format header
        f.write(f"$MeshFormat\n")
        f.write(f"2.2 0 8\n")
        f.write(f"$EndMeshFormat\n")
        f.write(f"\n")
        
        # Nodes
        f.write(f"$Nodes\n")
        
        # Count total nodes
        total_nodes = len(network.fibers) * (segments_per_fiber + 1)
        f.write(f"{total_nodes}\n")
        
        node_id = 1
        node_map = {}
        
        for fiber_idx, fiber in enumerate(network.fibers):
            for seg_id in range(segments_per_fiber + 1):
                t = seg_id / segments_per_fiber
                position = (fiber.start_point if seg_id == 0 else
                            fiber.end_point if seg_id == segments_per_fiber else
                            fiber.start_point + t * (fiber.end_point - fiber.start_point))
                f.write(f"{node_id} {position[0]:.17g} {position[1]:.17g} {position[2]:.17g}\n")
                node_map[(fiber_idx, seg_id)] = node_id
                node_id += 1
        
        f.write(f"$EndNodes\n")
        f.write(f"\n")
        
        # Elements (lines)
        f.write(f"$Elements\n")
        total_elements = len(network.fibers) * segments_per_fiber
        f.write(f"{total_elements}\n")
        
        element_id = 1
        for fiber_idx in range(len(network.fibers)):
            for seg_id in range(segments_per_fiber):
                node1 = node_map[(fiber_idx, seg_id)]
                node2 = node_map[(fiber_idx, seg_id + 1)]
                f.write(f"{element_id} 1 2 0 {fiber_idx + 1} {node1} {node2}\n")
                element_id += 1
        
        f.write(f"$EndElements\n")


def export_to_lammps(
    network: FiberNetwork,
    filename: Union[str, Path],
    atom_style: str = "bond",
    segments_per_fiber: int = 5,
) -> None:
    """Unsupported legacy FEA-LAMMPS exporter; no destination is opened.

    Its historical density-derived placeholder mass and unspecified units did
    not define a validated physical model. No automatic replacement is made.
    Preserve the original FiberNetwork with save_json. The separate explicit
    StructureGraph LAMMPS backend requires caller-specified units, masses and
    bond parameters; it is not an equivalent conversion of this network.
    """
    raise NotImplementedError(
        'legacy FEA-LAMMPS export has unvalidated units and placeholder masses; '
        'no file was opened. Preserve FiberNetwork with save_json; prepare a '
        'separate explicit model for the StructureGraph LAMMPS backend.'
    )

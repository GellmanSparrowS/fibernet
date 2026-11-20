"""
GMSH mesh file export.

Exports fiber networks as beam element meshes for FEM solvers.
"""

from fibernet.core.network import FiberNetwork


def to_gmsh(
    network: FiberNetwork,
    filename: str,
    segments_per_fiber: int = 5,
    dimension: int = 2,
) -> str:
    """Export fiber network to GMSH .msh format (v2 ASCII).
    
    Only exactly coincident sampled positions share a mesh node. Coordinates
    are written with round-trip float64 precision; previously a fixed decimal
    tolerance could merge nearby fibers or collapse short line elements.
    This does not search for intersections or apply declared crosslinks.
    A segment with identical sampled endpoints is rejected before writing.
    This is a geometry export, not a stable-ID or material format.

    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    filename : str
        Output .msh file path.
    segments_per_fiber : int
        Number of beam elements per fiber.
    dimension : int
        Coordinate dimension (2 or 3), retained for compatibility. The file
        always stores XYZ coordinates and the fibers form a 1D physical group.
    """
    # Build mesh
    nodes = []
    node_map = {}
    elements = []
    
    nid = 1
    eid = 1
    
    for f_idx, fiber in enumerate(network.fibers):
        resampled = fiber.resample(segments_per_fiber + 1)
        pts = resampled.centerline
        # The interpolation endpoint can round differently on adjacent fibers.
        # Use the authoritative coordinates on both ends.
        if len(pts) > 1:
            pts[[0, -1]] = fiber.centerline[[0, -1]]
        
        fiber_nids = []
        for pt in pts:
            key = tuple(pt)
            if key not in node_map:
                node_map[key] = nid
                nodes.append(pt)
                nid += 1
            fiber_nids.append(node_map[key])
        
        for i in range(len(fiber_nids) - 1):
            if fiber_nids[i] == fiber_nids[i + 1]:
                raise ValueError('Gmsh resampling produced a zero-length line; '
                                 'use fewer segments or check the centerline')
            elements.append({
                'id': eid,
                'type': 1,  # Line element
                'n1': fiber_nids[i],
                'n2': fiber_nids[i + 1],
                'fiber_idx': f_idx,
            })
            eid += 1
    
    with open(filename, 'w') as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        
        # Physical groups
        f.write("$PhysicalNames\n")
        f.write(f"{1}\n")
        f.write('1 1 "fibers"\n')
        f.write("$EndPhysicalNames\n")
        
        # Nodes
        f.write("$Nodes\n")
        f.write(f"{len(nodes)}\n")
        for i, pt in enumerate(nodes):
            f.write(f"{i+1} {pt[0]:.17g} {pt[1]:.17g} {pt[2]:.17g}\n")
        f.write("$EndNodes\n")
        
        # Elements
        f.write("$Elements\n")
        f.write(f"{len(elements)}\n")
        for elem in elements:
            # type 1 = 2-node line, 2 tags: physical + elementary
            f.write(f"{elem['id']} 1 2 1 {elem['fiber_idx']+1} {elem['n1']} {elem['n2']}\n")
        f.write("$EndElements\n")
    
    return filename

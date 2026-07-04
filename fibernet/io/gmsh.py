"""
GMSH mesh file export.

Exports fiber networks as beam element meshes for FEM solvers.
"""

import numpy as np
from fibernet.core.network import FiberNetwork


def to_gmsh(
    network: FiberNetwork,
    filename: str,
    segments_per_fiber: int = 5,
    dimension: int = 2,
) -> str:
    """Export fiber network to GMSH .msh format (v2 ASCII).
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    filename : str
        Output .msh file path.
    segments_per_fiber : int
        Number of beam elements per fiber.
    dimension : int
        Mesh dimension (2 or 3).
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
        
        fiber_nids = []
        for pt in pts:
            key = tuple(np.round(pt, 8))
            if key not in node_map:
                node_map[key] = nid
                nodes.append(pt)
                nid += 1
            fiber_nids.append(node_map[key])
        
        for i in range(len(fiber_nids) - 1):
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
        f.write(f'{dimension} 1 "fibers"\n')
        f.write("$EndPhysicalNames\n")
        
        # Nodes
        f.write("$Nodes\n")
        f.write(f"{len(nodes)}\n")
        for i, pt in enumerate(nodes):
            f.write(f"{i+1} {pt[0]:.10e} {pt[1]:.10e} {pt[2]:.10e}\n")
        f.write("$EndNodes\n")
        
        # Elements
        f.write("$Elements\n")
        f.write(f"{len(elements)}\n")
        for elem in elements:
            # type 1 = 2-node line, 2 tags: physical + elementary
            f.write(f"{elem['id']} 1 2 1 {elem['fiber_idx']+1} {elem['n1']} {elem['n2']}\n")
        f.write("$EndElements\n")
    
    return filename

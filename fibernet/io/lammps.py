"""
LAMMPS data file I/O.

The legacy FiberNetwork writer is unsupported and fails before file I/O.
The existing reader remains available unchanged; this does not validate it.
"""

import numpy as np
from typing import Optional, Dict
from fibernet.core.network import FiberNetwork
from fibernet.core.fiber import Fiber
from fibernet.core.material import Material


def to_lammps(
    network: FiberNetwork,
    filename: str,
    atom_style: str = "bond",
    bead_spacing: float = None,
    bond_type: str = "harmonic",
    units: str = "real",
) -> str:
    """Unsupported legacy FiberNetwork writer; rejects without touching output.

    Historical atom records, coefficient sections and material/unit mapping
    were not a validated LAMMPS contract. Neither unlinked inputs nor choosing
    another atom_style/bond_type establishes a supported physical model.
    Preserve source data using FiberNetwork.save_json. The separate explicit
    StructureGraph backend requires caller-defined units, masses and bonds;
    it is not an automatic conversion or equivalent replacement.
    """
    raise NotImplementedError(
        'legacy to_lammps export is unsupported: atom records, bond coefficients '
        'and units/material mapping are not validated; no file was opened. '
        'Preserve FiberNetwork with save_json. For a separately prepared '
        'StructureGraph, use the explicit LAMMPS backend with length_unit, '
        'masses_kg and bond parameters; no automatic conversion is performed.'
    )


def from_lammps(filename: str, bead_spacing: float = 1.0) -> FiberNetwork:
    """Import fiber network from LAMMPS data file.
    
    Reads bonded chains and converts them to fibers.
    
    Parameters
    ----------
    filename : str
        Input LAMMPS data file.
    bead_spacing : float
        Used to estimate fiber radius if not available.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    atoms = {}
    bonds = []
    atom_types = {}
    
    section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        if 'atoms' in line and not section:
            continue
        if 'bonds' in line and not section:
            continue
        
        if line.startswith('Atoms'):
            section = 'atoms'
            continue
        elif line.startswith('Bonds'):
            section = 'bonds'
            continue
        elif line.startswith('Masses') or line.startswith('Bond') or line.startswith('Angles'):
            section = None
            continue
        
        if section == 'atoms':
            parts = line.split()
            if len(parts) >= 5:
                try:
                    aid = int(parts[0])
                    atype = int(parts[1])
                    x = float(parts[-3])
                    y = float(parts[-2])
                    z = float(parts[-1])
                    atoms[aid] = {'type': atype, 'pos': np.array([x, y, z])}
                    if atype not in atom_types:
                        atom_types[atype] = atype
                except (ValueError, IndexError):
                    pass
        
        elif section == 'bonds':
            parts = line.split()
            if len(parts) >= 4:
                try:
                    bid = int(parts[0])
                    btype = int(parts[1])
                    i = int(parts[2])
                    j = int(parts[3])
                    bonds.append({'i': i, 'j': j, 'type': btype})
                except (ValueError, IndexError):
                    pass
    
    if not atoms:
        return FiberNetwork()
    
    # Build adjacency
    adj = {}
    for b in bonds:
        i, j = b['i'], b['j']
        if i not in adj:
            adj[i] = []
        if j not in adj:
            adj[j] = []
        adj[i].append(j)
        adj[j].append(i)
    
    # Trace chains
    visited = set()
    chains = []
    
    for start_id in atoms:
        if start_id in visited:
            continue
        
        chain = [start_id]
        visited.add(start_id)
        
        current = start_id
        while True:
            neighbors = [n for n in adj.get(current, []) if n not in visited]
            if not neighbors:
                break
            next_id = neighbors[0]
            chain.append(next_id)
            visited.add(next_id)
            current = next_id
        
        if len(chain) >= 2:
            chains.append(chain)
    
    # Create network
    net = FiberNetwork(dimension=3)
    
    for c_idx, chain in enumerate(chains):
        positions = np.array([atoms[aid]['pos'] for aid in chain])
        radius = bead_spacing * 0.5
        
        fiber = Fiber(
            centerline=positions,
            radius=radius,
            fiber_id=c_idx,
        )
        net.add_fiber(fiber)
    
    net.auto_crosslink(threshold=3.0 * bead_spacing)
    return net

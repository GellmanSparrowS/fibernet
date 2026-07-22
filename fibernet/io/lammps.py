"""
LAMMPS data file I/O.

Exports fiber networks to LAMMPS data format for molecular dynamics.
Fibers are represented as chains of beads connected by bonds.
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
    """Export fiber network to LAMMPS data file.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to export.
    filename : str
        Output file path.
    atom_style : str
        LAMMPS atom style ('bond', 'molecular', 'full').
    bead_spacing : float
        Spacing between beads along fibers. Defaults to mean fiber radius.
    bond_type : str
        Bond potential type ('harmonic', 'fene', 'morse').
    units : str
        LAMMPS units ('real', 'lj', 'si', 'metal').
    
    Returns
    -------
    str
        Output filename.
    """
    if bead_spacing is None:
        bead_spacing = network.mean_radius * 2
    
    # Collect all atoms and bonds
    atoms = []
    bonds = []
    atom_types = {}
    bond_types = {}
    
    atom_id = 1
    bond_id = 1
    
    for f_idx, fiber in enumerate(network.fibers):
        length = fiber.length
        num_beads = max(2, int(length / bead_spacing) + 1)
        
        if num_beads == 2:
            positions = np.array([fiber.start_point, fiber.end_point])
        else:
            positions = fiber.resample(num_beads).centerline
        
        mat_name = fiber.material.name
        if mat_name not in atom_types:
            atom_types[mat_name] = len(atom_types) + 1
        
        atype = atom_types[mat_name]
        
        fiber_atom_ids = []
        for pos in positions:
            atoms.append({
                'id': atom_id,
                'type': atype,
                'x': pos[0], 'y': pos[1], 'z': pos[2],
                'fiber_id': f_idx,
                'radius': fiber.radius,
            })
            fiber_atom_ids.append(atom_id)
            atom_id += 1
        
        # Create bonds between consecutive beads
        bond_key = f"{mat_name}_{bond_type}"
        if bond_key not in bond_types:
            bond_types[bond_key] = len(bond_types) + 1
        
        btype = bond_types[bond_key]
        for i in range(len(fiber_atom_ids) - 1):
            bonds.append({
                'id': bond_id,
                'type': btype,
                'i': fiber_atom_ids[i],
                'j': fiber_atom_ids[i + 1],
                'rest_length': bead_spacing,
                'E': fiber.material.youngs_modulus,
                'A': fiber.cross_section_area,
            })
            bond_id += 1
    
    # Compute box
    bb_min, bb_max = network.bounding_box()
    box_lo = bb_min - 1.0
    box_hi = bb_max + 1.0
    
    # Write file
    with open(filename, 'w') as f:
        f.write(f"# FiberNet LAMMPS data file\n")
        f.write(f"# Generated from {network.num_fibers} fibers\n\n")
        
        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{len(bonds)} bonds\n")
        f.write(f"0 angles\n")
        f.write(f"0 dihedrals\n")
        f.write(f"0 impropers\n\n")
        
        f.write(f"{len(atom_types)} atom types\n")
        f.write(f"{len(bond_types)} bond types\n\n")
        
        f.write(f"{box_lo[0]:.6f} {box_hi[0]:.6f} xlo xhi\n")
        f.write(f"{box_lo[1]:.6f} {box_hi[1]:.6f} ylo yhi\n")
        f.write(f"{box_lo[2]:.6f} {box_hi[2]:.6f} zlo zhi\n\n")
        
        # Masses
        f.write("Masses\n\n")
        for name, atype in sorted(atom_types.items(), key=lambda x: x[1]):
            # Estimate mass from density and volume
            mat = None
            for fiber in network.fibers:
                if fiber.material.name == name:
                    mat = fiber.material
                    break
            
            density = mat.density if mat and mat.density else 1000.0
            vol = np.pi * (fiber.radius**2) * bead_spacing
            mass = density * vol
            f.write(f"{atype} {mass:.6e} # {name}\n")
        f.write("\n")
        
        # Bond coefficients
        f.write(f"# Bond Coeffs ({bond_type})\n\n")
        for key, btype in sorted(bond_types.items(), key=lambda x: x[1]):
            # Find representative bond
            for b in bonds:
                if b['type'] == btype:
                    K = b['E'] * b['A'] / b['rest_length']
                    r0 = b['rest_length']
                    f.write(f"{btype} {K:.6e} {r0:.6f} # {key}\n")
                    break
        f.write("\n")
        
        # Atoms
        f.write("Atoms\n\n")
        for atom in atoms:
            if atom_style == "bond":
                f.write(f"{atom['id']} {atom['type']} {atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}\n")
            elif atom_style == "molecular":
                mol_id = atom['fiber_id'] + 1
                f.write(f"{atom['id']} {mol_id} {atom['type']} {atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}\n")
            else:
                q = 0.0
                mol_id = atom['fiber_id'] + 1
                f.write(f"{atom['id']} {mol_id} {atom['type']} {q:.4f} {atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}\n")
        f.write("\n")
        
        # Bonds
        if bonds:
            f.write("Bonds\n\n")
            for bond in bonds:
                f.write(f"{bond['id']} {bond['type']} {bond['i']} {bond['j']}\n")
    
    return filename


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

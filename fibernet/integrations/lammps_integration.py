"""
LAMMPS Integration for FiberNet

Provides integration with LAMMPS for molecular dynamics simulations:
- Convert FiberNetwork to LAMMPS data file
- Run MD simulations with various force fields
- Extract trajectories and properties
- Coarse-grained fiber models

LAMMPS is GPL v2 licensed: https://github.com/lammps/lammps
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import os
import warnings

from fibernet.core.network import FiberNetwork


@dataclass
class LAMMPSResult:
    """Result of LAMMPS MD simulation."""
    trajectory: List[np.ndarray] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)
    pressures: List[float] = field(default_factory=list)
    final_structure: np.ndarray = None
    simulation_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'num_frames': len(self.trajectory),
            'final_energy': self.energies[-1] if self.energies else 0.0,
            'final_temperature': self.temperatures[-1] if self.temperatures else 0.0,
            'simulation_time': self.simulation_time,
        }


class LAMMPSBridge:
    """Bridge between FiberNetwork and LAMMPS.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to simulate.
    """
    
    def __init__(self, network: FiberNetwork):
        self.network = network
        self._lammps_available = False
        try:
            from lammps import lammps
            self._lammps_available = True
        except ImportError:
            warnings.warn(
                "LAMMPS Python interface not available. "
                "Install with: pip install lammps"
            )
    
    def write_data_file(
        self,
        filename: str,
        atom_style: str = 'bond',
        segments_per_fiber: int = 10,
    ) -> str:
        """Write LAMMPS data file from FiberNetwork.
        
        Parameters
        ----------
        filename : str
            Output filename.
        atom_style : str
            LAMMPS atom style.
        segments_per_fiber : int
            Number of segments per fiber.
        
        Returns
        -------
        filename : str
            Path to written file.
        """
        lines = []
        
        # Header
        lines.append("FiberNet LAMMPS data file")
        lines.append("")
        
        # Collect atoms
        atoms = []
        bonds = []
        atom_id = 1
        bond_id = 1
        fiber_atom_ranges = {}
        
        for fi, fiber in enumerate(self.network.fibers):
            cl = fiber.centerline
            # Resample centerline to segments_per_fiber points
            t = np.linspace(0, 1, segments_per_fiber)
            if len(cl) > 1:
                from scipy.interpolate import interp1d
                t_orig = np.linspace(0, 1, len(cl))
                interp = interp1d(t_orig, cl, axis=0)
                points = interp(t)
            else:
                points = np.tile(cl[0], (segments_per_fiber, 1))
            
            start_id = atom_id
            for pi, point in enumerate(points):
                # atom_id type x y z
                atoms.append(f"{atom_id} 1 {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}")
                atom_id += 1
            fiber_atom_ranges[fi] = (start_id, atom_id - 1)
            
            # Bonds between consecutive atoms in fiber
            for bi in range(start_id, atom_id - 1):
                bonds.append(f"{bond_id} 1 {bi} {bi + 1}")
                bond_id += 1
        
        n_atoms = len(atoms)
        n_bonds = len(bonds)
        n_atom_types = 1
        n_bond_types = 1
        
        # Compute bounds
        all_coords = np.array([a.split()[2:5] for a in atoms], dtype=float)
        xlo, xhi = all_coords[:, 0].min() - 1, all_coords[:, 0].max() + 1
        ylo, yhi = all_coords[:, 1].min() - 1, all_coords[:, 1].max() + 1
        zlo, zhi = all_coords[:, 2].min() - 1, all_coords[:, 2].max() + 1
        
        lines.append(f"{n_atoms} atoms")
        lines.append(f"{n_bonds} bonds")
        lines.append(f"0 angles")
        lines.append(f"0 dihedrals")
        lines.append(f"0 impropers")
        lines.append("")
        lines.append(f"{n_atom_types} atom types")
        lines.append(f"{n_bond_types} bond types")
        lines.append(f"0 angle types")
        lines.append(f"0 dihedral types")
        lines.append(f"0 improper types")
        lines.append("")
        lines.append(f"{xlo:.6f} {xhi:.6f} xlo xhi")
        lines.append(f"{ylo:.6f} {yhi:.6f} ylo yhi")
        lines.append(f"{zlo:.6f} {zhi:.6f} zlo zhi")
        lines.append("")
        lines.append("Masses")
        lines.append("")
        lines.append("1 1.0")
        lines.append("")
        lines.append("Atoms")
        lines.append("")
        lines.extend(atoms)
        lines.append("")
        lines.append("Bonds")
        lines.append("")
        lines.extend(bonds)
        
        with open(filename, 'w') as f:
            f.write('\n'.join(lines))
        
        return filename
    
    def write_input_script(
        self,
        filename: str,
        data_file: str,
        temperature: float = 300.0,
        timestep: float = 0.001,
        nsteps: int = 10000,
        dump_freq: int = 100,
        force_field: str = 'harmonic',
    ) -> str:
        """Write LAMMPS input script.
        
        Parameters
        ----------
        filename : str
            Output script filename.
        data_file : str
            LAMMPS data file.
        temperature : float
            Simulation temperature (K).
        timestep : float
            Time step (ps).
        nsteps : int
            Number of simulation steps.
        dump_freq : int
            Dump frequency for trajectory.
        force_field : str
            Force field type: 'harmonic', 'fene', 'morse'
        
        Returns
        -------
        filename : str
            Path to written script.
        """
        script = f"""# FiberNet LAMMPS Input Script
# Generated automatically

units           real
atom_style      bond
boundary        p p p

read_data       {data_file}

# Force field
pair_style      lj/cut 10.0
pair_coeff      * * 1.0 1.0 10.0

"""
        if force_field == 'harmonic':
            script += """bond_style      harmonic
bond_coeff      * 100.0 1.0
"""
        elif force_field == 'fene':
            script += """bond_style      fene
bond_coeff      * 30.0 1.5 1.0 1.0
"""
        elif force_field == 'morse':
            script += """bond_style      morse
bond_coeff      * 10.0 2.0 1.0
"""
        
        script += f"""
# Thermostat
fix             1 all nvt temp {temperature} {temperature} 100.0

# Output
timestep        {timestep}
thermo          100
thermo_style    custom step temp pe ke etotal press vol

dump            1 all custom {dump_freq} dump.lammpstrj id type x y z
dump_modify     1 sort id

# Run
run             {nsteps}
"""
        
        with open(filename, 'w') as f:
            f.write(script)
        
        return filename
    
    def run_simulation(
        self,
        temperature: float = 300.0,
        nsteps: int = 1000,
        workdir: str = None,
    ) -> LAMMPSResult:
        """Run LAMMPS MD simulation.
        
        Parameters
        ----------
        temperature : float
            Simulation temperature (K).
        nsteps : int
            Number of steps.
        workdir : str, optional
            Working directory. Uses temp dir if None.
        
        Returns
        -------
        result : LAMMPSResult
            Simulation results.
        """
        if not self._lammps_available:
            warnings.warn("LAMMPS not available. Returning empty result.")
            return LAMMPSResult()
        
        if workdir is None:
            workdir = tempfile.mkdtemp()
        
        data_file = os.path.join(workdir, "fiber.data")
        script_file = os.path.join(workdir, "in.fiber")
        
        # Write files
        self.write_data_file(data_file)
        self.write_input_script(
            script_file,
            data_file,
            temperature=temperature,
            nsteps=nsteps,
        )
        
        # Run LAMMPS
        try:
            from lammps import lammps
            lmp = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
            lmp.file(script_file)
            
            # Extract results
            n_atoms = lmp.get_natoms()
            positions = lmp.gather_atoms("x", 1, 3)
            final_structure = np.array(positions).reshape(-1, 3)
            
            result = LAMMPSResult(
                final_structure=final_structure,
                simulation_time=nsteps * 0.001,
            )
            
            lmp.close()
            
        except Exception as e:
            warnings.warn(f"LAMMPS simulation failed: {e}")
            result = LAMMPSResult()
        
        return result


def run_lammps_md(
    network: FiberNetwork,
    temperature: float = 300.0,
    nsteps: int = 1000,
    **kwargs,
) -> LAMMPSResult:
    """Convenience function for LAMMPS MD simulation.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to simulate.
    temperature : float
        Temperature (K).
    nsteps : int
        Number of MD steps.
    
    Returns
    -------
    result : LAMMPSResult
        Simulation results.
    """
    bridge = LAMMPSBridge(network)
    return bridge.run_simulation(
        temperature=temperature,
        nsteps=nsteps,
        **kwargs,
    )

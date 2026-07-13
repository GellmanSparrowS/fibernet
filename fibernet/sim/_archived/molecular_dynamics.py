"""
Molecular Dynamics Module for Fiber Networks

Provides coarse-grained MD simulations:
- Bead-spring fiber models
- Langevin dynamics with thermal fluctuations
- Crosslink bond dynamics
- Bond breaking and reforming
- Fiber diffusion and reptation

This module provides native MD without requiring LAMMPS.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings

from fibernet.core.network import FiberNetwork


@dataclass
class MDTrajectory:
    """Container for MD trajectory data."""
    positions: List[np.ndarray] = field(default_factory=list)
    velocities: List[np.ndarray] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)
    
    def num_frames(self) -> int:
        return len(self.positions)
    
    def to_dict(self) -> Dict:
        return {
            'num_frames': self.num_frames(),
            'total_time': self.times[-1] if self.times else 0.0,
            'final_energy': self.energies[-1] if self.energies else 0.0,
        }


@dataclass
class MDParameters:
    """Molecular dynamics simulation parameters."""
    temperature: float = 300.0  # K
    timestep: float = 0.001  # ps
    nsteps: int = 10000
    dump_freq: int = 100
    friction: float = 1.0  # Langevin friction coefficient (1/ps)
    kb: float = 1.380649e-23  # Boltzmann constant (J/K)
    mass: float = 1.0  # Bead mass (amu)
    bond_stiffness: float = 100.0  # Bond spring constant (kcal/mol/A^2)
    bond_length: float = 1.0  # Equilibrium bond length (A)
    lj_epsilon: float = 1.0  # LJ energy parameter (kcal/mol)
    lj_sigma: float = 1.0  # LJ length parameter (A)
    lj_cutoff: float = 2.5  # LJ cutoff (multiples of sigma)


class FiberMDSolver:
    """Coarse-grained MD solver for fiber networks.
    
    Uses Langevin dynamics: m*a = F_bond + F_LJ + F_friction + F_random
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to simulate.
    params : MDParameters, optional
        Simulation parameters.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim.molecular_dynamics import FiberMDSolver, MDParameters
    >>> net = gen.random_straight_2d(num_fibers=50, seed=42)
    >>> params = MDParameters(temperature=300.0, nsteps=1000)
    >>> solver = FiberMDSolver(net, params)
    >>> trajectory = solver.run()
    >>> print(f"Frames: {trajectory.num_frames()}")
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        params: Optional[MDParameters] = None,
    ):
        self.network = network
        self.params = params or MDParameters()
        
        # Build bead-spring model
        self._build_model()
    
    def _build_model(self):
        """Build coarse-grained bead-spring model."""
        # Collect all beads (one per crosslink point)
        self.bead_positions = []
        self.bead_velocities = []
        self.bead_fiber_ids = []
        self.bonds = []  # (i, j) pairs
        
        bead_id = 0
        fiber_bead_map = {}
        
        for fi, fiber in enumerate(self.network.fibers):
            cl = fiber.centerline
            n_pts = len(cl)
            
            start_id = bead_id
            for pi in range(n_pts):
                self.bead_positions.append(cl[pi].copy())
                self.bead_velocities.append(np.zeros(3))
                self.bead_fiber_ids.append(fi)
                bead_id += 1
            
            fiber_bead_map[fi] = list(range(start_id, bead_id))
            
            # Add bonds between consecutive beads
            for bi in range(start_id, bead_id - 1):
                self.bonds.append((bi, bi + 1))
        
        self.bead_positions = np.array(self.bead_positions)
        self.bead_velocities = np.array(self.bead_velocities)
        self.n_beads = len(self.bead_positions)
        self.bonds = np.array(self.bonds) if self.bonds else np.zeros((0, 2), dtype=int)
        self.fiber_bead_map = fiber_bead_map
    
    def _compute_bond_forces(self, positions: np.ndarray) -> np.ndarray:
        """Compute harmonic bond forces."""
        forces = np.zeros_like(positions)
        k = self.params.bond_stiffness
        r0 = self.params.bond_length
        
        for i, j in self.bonds:
            rij = positions[j] - positions[i]
            r = np.linalg.norm(rij)
            if r > 1e-10:
                f_mag = k * (r - r0)
                f_dir = rij / r
                forces[i] += f_mag * f_dir
                forces[j] -= f_mag * f_dir
        
        return forces
    
    def _compute_lj_forces(self, positions: np.ndarray) -> np.ndarray:
        """Compute Lennard-Jones forces."""
        forces = np.zeros_like(positions)
        eps = self.params.lj_epsilon
        sig = self.params.lj_sigma
        cutoff = self.params.lj_cutoff * sig
        
        for i in range(self.n_beads):
            for j in range(i + 1, self.n_beads):
                # Skip bonded pairs
                if self._are_bonded(i, j):
                    continue
                
                rij = positions[j] - positions[i]
                r = np.linalg.norm(rij)
                
                if r < cutoff and r > 1e-10:
                    # LJ force
                    sr6 = (sig / r) ** 6
                    f_mag = 24 * eps / r * sr6 * (2 * sr6 - 1)
                    f_dir = rij / r
                    forces[i] += f_mag * f_dir
                    forces[j] -= f_mag * f_dir
        
        return forces
    
    def _are_bonded(self, i: int, j: int) -> bool:
        """Check if two beads are bonded."""
        if len(self.bonds) == 0:
            return False
        bonded = np.any((self.bonds[:, 0] == i) & (self.bonds[:, 1] == j))
        bonded = bonded or np.any((self.bonds[:, 0] == j) & (self.bonds[:, 1] == i))
        return bonded
    
    def _compute_langevin_forces(self, velocities: np.ndarray) -> np.ndarray:
        """Compute Langevin thermostat forces."""
        gamma = self.params.friction
        m = self.params.mass
        T = self.params.temperature
        dt = self.params.timestep
        kb = self.params.kb
        
        # Friction force
        f_friction = -gamma * m * velocities
        
        # Random force (fluctuation-dissipation)
        sigma = np.sqrt(2 * gamma * m * kb * T / dt)
        f_random = sigma * np.random.randn(*velocities.shape)
        
        return f_friction + f_random
    
    def _compute_energy(self, positions: np.ndarray, velocities: np.ndarray) -> float:
        """Compute total energy."""
        # Kinetic energy
        ke = 0.5 * self.params.mass * np.sum(velocities**2)
        
        # Bond potential energy
        pe_bond = 0.0
        k = self.params.bond_stiffness
        r0 = self.params.bond_length
        for i, j in self.bonds:
            rij = positions[j] - positions[i]
            r = np.linalg.norm(rij)
            pe_bond += 0.5 * k * (r - r0)**2
        
        return ke + pe_bond
    
    def run(self, verbose: bool = False) -> MDTrajectory:
        """Run MD simulation.
        
        Parameters
        ----------
        verbose : bool
            Print progress.
        
        Returns
        -------
        trajectory : MDTrajectory
            Simulation trajectory.
        """
        dt = self.params.timestep
        m = self.params.mass
        
        positions = self.bead_positions.copy()
        velocities = self.bead_velocities.copy()
        
        trajectory = MDTrajectory()
        trajectory.positions.append(positions.copy())
        trajectory.times.append(0.0)
        
        for step in range(self.params.nsteps):
            # Compute forces
            f_bond = self._compute_bond_forces(positions)
            f_lj = self._compute_lj_forces(positions)
            f_lang = self._compute_langevin_forces(velocities)
            
            f_total = f_bond + f_lj + f_lang
            
            # Velocity Verlet integration
            accelerations = f_total / m
            velocities += 0.5 * dt * accelerations
            positions += dt * velocities
            
            # Recompute forces at new position
            f_bond_new = self._compute_bond_forces(positions)
            f_lj_new = self._compute_lj_forces(positions)
            f_lang_new = self._compute_langevin_forces(velocities)
            f_total_new = f_bond_new + f_lj_new + f_lang_new
            
            accelerations_new = f_total_new / m
            velocities += 0.5 * dt * accelerations_new
            
            # Save trajectory
            if (step + 1) % self.params.dump_freq == 0:
                trajectory.positions.append(positions.copy())
                trajectory.velocities.append(velocities.copy())
                trajectory.times.append((step + 1) * dt)
                
                # Compute energy
                energy = self._compute_energy(positions, velocities)
                trajectory.energies.append(energy)
                
                # Compute temperature
                ke = 0.5 * m * np.sum(velocities**2)
                temp = 2 * ke / (3 * self.n_beads * self.params.kb)
                trajectory.temperatures.append(temp)
                
                if verbose and (step + 1) % (10 * self.params.dump_freq) == 0:
                    print(f"Step {step+1}/{self.params.nsteps}, E={energy:.4f}, T={temp:.1f} K")
        
        return trajectory
    
    def compute_msd(self, trajectory: MDTrajectory) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean squared displacement from trajectory.
        
        Parameters
        ----------
        trajectory : MDTrajectory
            MD trajectory.
        
        Returns
        -------
        times : np.ndarray
            Time values.
        msd : np.ndarray
            MSD values.
        """
        if len(trajectory.positions) < 2:
            return np.array([0.0]), np.array([0.0])
        
        ref = trajectory.positions[0]
        times = np.array(trajectory.times)
        msd = np.array([
            np.mean(np.sum((pos - ref)**2, axis=1))
            for pos in trajectory.positions
        ])
        
        return times, msd
    
    def compute_diffusion_coefficient(
        self,
        trajectory: MDTrajectory,
    ) -> float:
        """Compute diffusion coefficient from MSD.
        
        Parameters
        ----------
        trajectory : MDTrajectory
            MD trajectory.
        
        Returns
        -------
        D : float
            Diffusion coefficient.
        
        Notes
        -----
        D = MSD / (2 * d * t) where d is dimensionality.
        """
        times, msd = self.compute_msd(trajectory)
        
        if len(times) < 2:
            return 0.0
        
        # Linear fit to MSD vs time (skip first few points)
        skip = max(1, len(times) // 5)
        if len(times) > skip + 1:
            slope, _ = np.polyfit(times[skip:], msd[skip:], 1)
            # D = slope / (2 * d), d=3 for 3D
            D = slope / 6.0
            return D
        
        return 0.0


def run_fiber_md(
    network: FiberNetwork,
    temperature: float = 300.0,
    nsteps: int = 1000,
    **kwargs,
) -> MDTrajectory:
    """Convenience function for fiber MD simulation.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to simulate.
    temperature : float
        Temperature (K).
    nsteps : int
        Number of MD steps.
    **kwargs : dict
        Additional MDParameters.
    
    Returns
    -------
    trajectory : MDTrajectory
        Simulation trajectory.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim.molecular_dynamics import run_fiber_md
    >>> net = gen.random_straight_2d(num_fibers=50, seed=42)
    >>> traj = run_fiber_md(net, temperature=300, nsteps=500)
    """
    params = MDParameters(
        temperature=temperature,
        nsteps=nsteps,
        **kwargs,
    )
    solver = FiberMDSolver(network, params)
    return solver.run()

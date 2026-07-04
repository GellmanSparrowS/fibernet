"""
Dynamics simulation engine for fiber networks.

Implements:
- Molecular dynamics (Verlet integration)
- Brownian dynamics (Langevin equation)
- Damped dynamics for energy minimization
- Time-dependent loading scenarios
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork, Crosslink


@dataclass
class DynamicsResult:
    """Container for dynamics simulation results."""
    time: np.ndarray = None
    positions: List[np.ndarray] = field(default_factory=list)
    velocities: List[np.ndarray] = field(default_factory=list)
    kinetic_energy: np.ndarray = None
    potential_energy: np.ndarray = None
    total_energy: np.ndarray = None
    temperature: np.ndarray = None
    stress_history: List[np.ndarray] = field(default_factory=list)


class FiberDynamics:
    """Dynamics engine for fiber network simulations.
    
    Uses lumped mass model where each node has mass proportional
    to the fiber segments connected to it.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to simulate.
    dt : float
        Time step.
    damping : float
        Damping coefficient (for damped dynamics).
    temperature : float
        Temperature for Brownian dynamics (Kelvin).
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        dt: float = 1e-6,
        damping: float = 0.01,
        temperature: float = 0.0,
    ):
        self.network = network
        self.dt = dt
        self.damping = damping
        self.temperature = temperature
        self.kb = 1.380649e-23
        
        self._build_nodes()
    
    def _build_nodes(self):
        """Build lumped-mass node model."""
        self.positions = []
        self.velocities = []
        self.masses = []
        self.node_to_fiber = []
        self.node_to_idx = []
        
        for f_idx, fiber in enumerate(self.network.fibers):
            pts = fiber.centerline
            for p_idx, pt in enumerate(pts):
                self.positions.append(pt.copy())
                self.velocities.append(np.zeros(3))
                
                seg_length = 0.0
                if p_idx > 0:
                    seg_length += np.linalg.norm(pts[p_idx] - pts[p_idx - 1]) / 2
                if p_idx < len(pts) - 1:
                    seg_length += np.linalg.norm(pts[p_idx + 1] - pts[p_idx]) / 2
                
                volume = seg_length * fiber.cross_section_area
                mass = fiber.material.density * volume
                self.masses.append(max(mass, 1e-20))
                self.node_to_fiber.append(f_idx)
                self.node_to_idx.append(p_idx)
        
        self.num_nodes = len(self.positions)
        if self.num_nodes > 0:
            self.positions = np.array(self.positions)
            self.velocities = np.array(self.velocities)
            self.masses = np.array(self.masses)
        else:
            self.positions = np.zeros((0, 3))
            self.velocities = np.zeros((0, 3))
            self.masses = np.zeros(0)
    
    def _compute_forces(self, positions: np.ndarray) -> np.ndarray:
        """Compute internal elastic forces on all nodes."""
        forces = np.zeros_like(positions)
        
        offset = 0
        for f_idx, fiber in enumerate(self.network.fibers):
            n_pts = len(fiber.centerline)
            E = fiber.material.youngs_modulus
            A = fiber.cross_section_area
            
            for i in range(1, n_pts):
                p_prev = positions[offset + i - 1]
                p_curr = positions[offset + i]
                
                p_ref_prev = fiber.centerline[i - 1]
                p_ref_curr = fiber.centerline[i]
                
                L0 = np.linalg.norm(p_ref_curr - p_ref_prev)
                L = np.linalg.norm(p_curr - p_prev)
                
                if L0 > 1e-12 and L > 1e-12:
                    strain = (L - L0) / L0
                    force_mag = E * A * strain
                    direction = (p_curr - p_prev) / L
                    
                    forces[offset + i - 1] += force_mag * direction
                    forces[offset + i] -= force_mag * direction
            
            if n_pts > 2:
                EI = E * np.pi * fiber.radius**4 / 4.0
                for i in range(1, n_pts - 1):
                    p_prev = positions[offset + i - 1]
                    p_curr = positions[offset + i]
                    p_next = positions[offset + i + 1]
                    
                    p_ref_prev = fiber.centerline[i - 1]
                    p_ref_curr = fiber.centerline[i]
                    p_ref_next = fiber.centerline[i + 1]
                    
                    bending_force = EI * (p_prev - 2 * p_curr + p_next)
                    ref_bending = EI * (p_ref_prev - 2 * p_ref_curr + p_ref_next)
                    
                    delta_bending = bending_force - ref_bending
                    ds = np.linalg.norm(p_ref_curr - p_ref_prev)
                    if ds > 1e-12:
                        forces[offset + i] -= delta_bending / ds**2
            
            offset += n_pts
        
        return forces
    
    def run_verlet(
        self,
        num_steps: int,
        fixed_nodes: Optional[List[int]] = None,
        external_force: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
        save_interval: int = 10,
    ) -> DynamicsResult:
        """Run Verlet integration dynamics simulation.
        
        Parameters
        ----------
        num_steps : int
            Number of time steps.
        fixed_nodes : list of int
            Node indices to keep fixed.
        external_force : callable, optional
            Function(step, positions) -> force_array.
        save_interval : int
            Save trajectory every N steps.
        """
        pos = self.positions.copy()
        vel = self.velocities.copy()
        masses = self.masses.copy()
        
        fixed_set = set(fixed_nodes) if fixed_nodes else set()
        
        result = DynamicsResult(
            time=np.arange(0, num_steps * self.dt, self.dt * save_interval),
        )
        ke_list = []
        pe_list = []
        
        for step in range(num_steps):
            f_int = self._compute_forces(pos)
            
            if external_force:
                f_ext = external_force(step, pos)
                f_int += f_ext
            
            f_int -= self.damping * vel
            
            if self.temperature > 0:
                sigma = np.sqrt(2 * self.damping * self.kb * self.temperature / self.dt)
                f_int += sigma * np.random.randn(*pos.shape) * masses[:, None]
            
            acc = f_int / masses[:, None]
            
            vel_half = vel + 0.5 * self.dt * acc
            pos += self.dt * vel_half
            vel = vel_half + 0.5 * self.dt * acc
            
            for n in fixed_set:
                pos[n] = self.positions[n]
                vel[n] = 0.0
            
            if step % save_interval == 0:
                ke = 0.5 * np.sum(masses[:, None] * vel**2)
                ke_list.append(ke)
                result.positions.append(pos.copy())
                result.velocities.append(vel.copy())
        
        result.kinetic_energy = np.array(ke_list)
        return result
    
    def minimize_energy(
        self,
        max_steps: int = 10000,
        tol: float = 1e-8,
        fixed_nodes: Optional[List[int]] = None,
    ) -> DynamicsResult:
        """Energy minimization using damped dynamics (steepest descent with momentum)."""
        old_damping = self.damping
        self.damping = 0.5
        
        result = self.run_verlet(
            num_steps=max_steps,
            fixed_nodes=fixed_nodes,
            save_interval=max(1, max_steps // 100),
        )
        
        self.damping = old_damping
        return result


class TimeDependentLoading:
    """
    Time-dependent loading for dynamics simulations.
    
    Supports various loading protocols:
    - Constant force/displacement
    - Linear ramp
    - Cyclic/sinusoidal loading
    - Step loading
    - Custom loading profiles
    
    Examples
    --------
    >>> # Sinusoidal loading
    >>> loading = TimeDependentLoading.sinusoidal(
    ...     amplitude=1.0, frequency=100.0, direction=np.array([1, 0, 0])
    ... )
    >>> 
    >>> # Linear ramp
    >>> loading = TimeDependentLoading.ramp(
    ...     rate=0.1, direction=np.array([1, 0, 0])
    ... )
    """
    
    @staticmethod
    def constant(
        force: np.ndarray,
        node_indices: Optional[List[int]] = None
    ) -> Callable:
        """
        Create constant force loading.
        
        Parameters
        ----------
        force : np.ndarray
            Force vector (3D)
        node_indices : list, optional
            Nodes to apply force to (all if None)
        
        Returns
        -------
        loading : callable
            Loading function
        """
        def loading_func(step, positions):
            forces = np.zeros_like(positions)
            if node_indices is None:
                for i in range(len(positions)):
                    forces[i] += force
            else:
                for i in node_indices:
                    forces[i] += force
            return forces
        
        return loading_func
    
    @staticmethod
    def ramp(
        rate: float,
        direction: np.ndarray,
        node_indices: Optional[List[int]] = None,
        start_time: float = 0.0,
        dt: float = 1e-6
    ) -> Callable:
        """
        Create linear ramp loading.
        
        Parameters
        ----------
        rate : float
            Loading rate (force per time)
        direction : np.ndarray
            Loading direction (will be normalized)
        node_indices : list, optional
            Nodes to apply force to
        start_time : float
            Start time for loading
        dt : float
            Time step
        
        Returns
        -------
        loading : callable
            Loading function
        """
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        
        def loading_func(step, positions):
            t = step * dt
            forces = np.zeros_like(positions)
            if t >= start_time:
                magnitude = rate * (t - start_time)
                force = magnitude * direction
                if node_indices is None:
                    for i in range(len(positions)):
                        forces[i] += force
                else:
                    for i in node_indices:
                        forces[i] += force
            return forces
        
        return loading_func
    
    @staticmethod
    def sinusoidal(
        amplitude: float,
        frequency: float,
        direction: np.ndarray,
        phase: float = 0.0,
        node_indices: Optional[List[int]] = None,
        dt: float = 1e-6
    ) -> Callable:
        """
        Create sinusoidal loading.
        
        Parameters
        ----------
        amplitude : float
            Force amplitude
        frequency : float
            Frequency (Hz)
        direction : np.ndarray
            Loading direction
        phase : float
            Phase offset (radians)
        node_indices : list, optional
            Nodes to apply force to
        dt : float
            Time step
        
        Returns
        -------
        loading : callable
            Loading function
        """
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        
        def loading_func(step, positions):
            t = step * dt
            magnitude = amplitude * np.sin(2 * np.pi * frequency * t + phase)
            force = magnitude * direction
            forces = np.zeros_like(positions)
            if node_indices is None:
                for i in range(len(positions)):
                    forces[i] += force
            else:
                for i in node_indices:
                    forces[i] += force
            return forces
        
        return loading_func
    
    @staticmethod
    def step_loading(
        force: np.ndarray,
        step_time: int,
        node_indices: Optional[List[int]] = None
    ) -> Callable:
        """
        Create step loading (applied after specified step).
        
        Parameters
        ----------
        force : np.ndarray
            Force vector
        step_time : int
            Step at which to apply force
        node_indices : list, optional
            Nodes to apply force to
        
        Returns
        -------
        loading : callable
            Loading function
        """
        def loading_func(step, positions):
            forces = np.zeros_like(positions)
            if step >= step_time:
                if node_indices is None:
                    for i in range(len(positions)):
                        forces[i] += force
                else:
                    for i in node_indices:
                        forces[i] += force
            return forces
        
        return loading_func


def compute_kinetic_energy(velocities: np.ndarray, masses: np.ndarray) -> float:
    """
    Compute kinetic energy.
    
    Parameters
    ----------
    velocities : np.ndarray
        Velocity array (N, 3)
    masses : np.ndarray
        Mass array (N,)
    
    Returns
    -------
    ke : float
        Total kinetic energy
    """
    return 0.5 * np.sum(masses[:, None] * velocities**2)


def compute_temperature(
    velocities: np.ndarray,
    masses: np.ndarray,
    degrees_of_freedom: int
) -> float:
    """
    Compute instantaneous temperature.
    
    Parameters
    ----------
    velocities : np.ndarray
        Velocity array (N, 3)
    masses : np.ndarray
        Mass array (N,)
    degrees_of_freedom : int
        Number of degrees of freedom
    
    Returns
    -------
    T : float
        Temperature (Kelvin)
    """
    kb = 1.380649e-23
    ke = compute_kinetic_energy(velocities, masses)
    
    if degrees_of_freedom > 0:
        T = 2 * ke / (degrees_of_freedom * kb)
    else:
        T = 0.0
    
    return T



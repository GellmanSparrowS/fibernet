"""
Coupled multi-physics simulations for fiber networks.

Implements thermo-mechanical, electro-mechanical, and chemo-mechanical coupling.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ..core import FiberNetwork


@dataclass
class ThermoMechanicalResult:
    """Result of a thermo-mechanical simulation."""
    temperature_field: np.ndarray
    displacement_field: np.ndarray
    stress_field: np.ndarray
    strain_field: np.ndarray
    thermal_strain: np.ndarray
    mechanical_strain: np.ndarray
    total_energy: float
    thermal_energy: float
    mechanical_energy: float
    time_steps: np.ndarray
    
    @property
    def max_temperature(self) -> float:
        return np.max(self.temperature_field)
    
    @property
    def min_temperature(self) -> float:
        return np.min(self.temperature_field)
    
    @property
    def max_stress(self) -> float:
        return np.max(np.abs(self.stress_field))
    
    @property
    def max_displacement(self) -> float:
        return np.max(np.linalg.norm(self.displacement_field, axis=-1))


@dataclass
class ElectroMechanicalResult:
    """Result of an electro-mechanical simulation."""
    electric_potential: np.ndarray
    electric_field: np.ndarray
    displacement_field: np.ndarray
    stress_field: np.ndarray
    piezoelectric_strain: np.ndarray
    polarization: np.ndarray
    time_steps: np.ndarray


class ThermoMechanicalSolver:
    """
    Coupled thermo-mechanical solver for fiber networks.
    
    Solves the coupled system:
    - Mechanical equilibrium: div(σ) + f = 0
    - Heat equation: ρc dT/dt = div(k∇T) + Q
    
    where:
    - σ = C:(ε - αΔT)  (constitutive law with thermal strain)
    - Q = β σ:ε̇       (mechanical heating from dissipation)
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to simulate
    youngs_modulus : float
        Young's modulus (Pa)
    poisson_ratio : float
        Poisson's ratio
    thermal_expansion : float
        Coefficient of thermal expansion (1/K)
    thermal_conductivity : float
        Thermal conductivity (W/m·K)
    specific_heat : float
        Specific heat capacity (J/kg·K)
    density : float
        Mass density (kg/m³)
    mechanical_to_thermal_fraction : float
        Fraction of mechanical work converted to heat (0-1)
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        youngs_modulus: float = 1e9,
        poisson_ratio: float = 0.3,
        thermal_expansion: float = 1e-5,
        thermal_conductivity: float = 0.5,
        specific_heat: float = 1000.0,
        density: float = 1000.0,
        mechanical_to_thermal_fraction: float = 0.9,
    ):
        self.network = network
        self.youngs_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio
        self.thermal_expansion = thermal_expansion
        self.thermal_conductivity = thermal_conductivity
        self.specific_heat = specific_heat
        self.density = density
        self.beta = mechanical_to_thermal_fraction
    
    def solve_steady_state(
        self,
        mechanical_load: Dict[str, Any],
        thermal_boundary: Dict[str, float],
        reference_temperature: float = 293.15,
    ) -> ThermoMechanicalResult:
        """
        Solve steady-state coupled problem.
        
        Parameters
        ----------
        mechanical_load : dict
            Mechanical loading specification (e.g., {'strain': 0.01, 'axis': 0})
        thermal_boundary : dict
            Temperature boundary conditions (e.g., {'top': 400, 'bottom': 300})
        reference_temperature : float
            Reference temperature for thermal strain calculation (K)
        
        Returns
        -------
        ThermoMechanicalResult
            Coupled simulation result
        """
        # Step 1: Solve thermal problem first
        num_fibers = self.network.num_fibers
        temperature_field = self._solve_thermal_steady_state(thermal_boundary)
        
        # Step 2: Compute thermal strain (isotropic, same in all directions)
        delta_T = temperature_field - reference_temperature
        thermal_strain_scalar = self.thermal_expansion * delta_T
        thermal_strain = np.zeros((num_fibers, 3))
        for i in range(3):
            thermal_strain[:, i] = thermal_strain_scalar
        
        # Step 3: Solve mechanical problem with thermal strain
        displacement, stress, mechanical_strain = self._solve_mechanical_with_thermal(
            mechanical_load, thermal_strain
        )
        
        # Step 4: Compute total strain
        total_strain = mechanical_strain + thermal_strain
        
        # Step 5: Compute energies
        thermal_energy = self._compute_thermal_energy(temperature_field)
        mechanical_energy = self._compute_mechanical_energy(stress, mechanical_strain)
        total_energy = thermal_energy + mechanical_energy
        
        return ThermoMechanicalResult(
            temperature_field=temperature_field,
            displacement_field=displacement,
            stress_field=stress,
            strain_field=total_strain,
            thermal_strain=thermal_strain,
            mechanical_strain=mechanical_strain,
            total_energy=total_energy,
            thermal_energy=thermal_energy,
            mechanical_energy=mechanical_energy,
            time_steps=np.array([0.0]),
        )
    
    def solve_transient(
        self,
        mechanical_load: Dict[str, Any],
        thermal_boundary: Dict[str, float],
        initial_temperature: float = 293.15,
        total_time: float = 10.0,
        num_steps: int = 100,
        mechanical_load_rate: Optional[float] = None,
    ) -> ThermoMechanicalResult:
        """
        Solve transient coupled problem.
        
        Parameters
        ----------
        mechanical_load : dict
            Final mechanical loading specification
        thermal_boundary : dict
            Temperature boundary conditions
        initial_temperature : float
            Initial uniform temperature (K)
        total_time : float
            Total simulation time (s)
        num_steps : int
            Number of time steps
        mechanical_load_rate : float, optional
            Rate of mechanical loading application
        
        Returns
        -------
        ThermoMechanicalResult
            Final state of coupled simulation
        """
        dt = total_time / num_steps
        time_steps = np.linspace(0, total_time, num_steps + 1)
        
        # Initialize fields
        num_fibers = self.network.num_fibers
        temperature = np.ones(num_fibers) * initial_temperature
        displacement = np.zeros((num_fibers, 3))
        
        # Storage for history
        temp_history = [temperature.copy()]
        disp_history = [displacement.copy()]
        
        for step in range(num_steps):
            t = time_steps[step + 1]
            
            # Scale mechanical load
            if mechanical_load_rate is not None:
                load_factor = min(1.0, t * mechanical_load_rate)
            else:
                load_factor = t / total_time
            
            scaled_load = {k: v * load_factor if isinstance(v, (int, float)) and k != 'axis' else v 
                          for k, v in mechanical_load.items()}
            
            # Compute thermal strain from current temperature (isotropic)
            delta_T = temperature - initial_temperature
            thermal_strain_scalar = self.thermal_expansion * delta_T
            thermal_strain = np.zeros((num_fibers, 3))
            for i in range(3):
                thermal_strain[:, i] = thermal_strain_scalar
            
            # Solve mechanical problem
            displacement, stress, mechanical_strain = self._solve_mechanical_with_thermal(
                scaled_load, thermal_strain
            )
            
            # Compute mechanical heating
            strain_rate = mechanical_strain / dt if step > 0 else np.zeros_like(mechanical_strain)
            mechanical_heating = self.beta * np.sum(stress * strain_rate, axis=-1) if stress.ndim > 1 else self.beta * stress * strain_rate
            
            # Update temperature (explicit Euler)
            laplacian_T = self._compute_temperature_laplacian(temperature, thermal_boundary)
            temperature += dt * (
                self.thermal_conductivity / (self.density * self.specific_heat) * laplacian_T +
                mechanical_heating / (self.density * self.specific_heat)
            )
            
            # Apply boundary conditions
            temperature = self._apply_thermal_bc(temperature, thermal_boundary)
            
            temp_history.append(temperature.copy())
            disp_history.append(displacement.copy())
        
        # Final state
        delta_T = temperature - initial_temperature
        thermal_strain_scalar = self.thermal_expansion * delta_T
        thermal_strain = np.zeros((num_fibers, 3))
        for i in range(3):
            thermal_strain[:, i] = thermal_strain_scalar
        displacement, stress, mechanical_strain = self._solve_mechanical_with_thermal(
            mechanical_load, thermal_strain
        )
        total_strain = mechanical_strain + thermal_strain
        
        thermal_energy = self._compute_thermal_energy(temperature)
        mechanical_energy = self._compute_mechanical_energy(stress, mechanical_strain)
        
        return ThermoMechanicalResult(
            temperature_field=temperature,
            displacement_field=displacement,
            stress_field=stress,
            strain_field=total_strain,
            thermal_strain=thermal_strain,
            mechanical_strain=mechanical_strain,
            total_energy=thermal_energy + mechanical_energy,
            thermal_energy=thermal_energy,
            mechanical_energy=mechanical_energy,
            time_steps=time_steps,
        )
    
    def _solve_thermal_steady_state(self, thermal_boundary: Dict[str, float]) -> np.ndarray:
        """Solve steady-state heat equation."""
        num_fibers = self.network.num_fibers
        temperature = np.ones(num_fibers) * 293.15  # Initial guess
        
        # Simple iterative solver
        for _ in range(100):
            laplacian_T = self._compute_temperature_laplacian(temperature, thermal_boundary)
            temperature += 0.01 * laplacian_T
            temperature = self._apply_thermal_bc(temperature, thermal_boundary)
        
        return temperature
    
    def _compute_temperature_laplacian(
        self, temperature: np.ndarray, thermal_boundary: Dict[str, float]
    ) -> np.ndarray:
        """Compute discrete Laplacian of temperature field."""
        num_fibers = len(temperature)
        laplacian = np.zeros(num_fibers)
        
        for i, fiber_i in enumerate(self.network.fibers):
            center_i = (fiber_i.start_point + fiber_i.end_point) / 2
            
            neighbors = []
            for j, fiber_j in enumerate(self.network.fibers):
                if i == j:
                    continue
                center_j = (fiber_j.start_point + fiber_j.end_point) / 2
                dist = np.linalg.norm(center_i - center_j)
                if dist < 5.0:  # Neighbor cutoff
                    neighbors.append((j, dist))
            
            if neighbors:
                for j, dist in neighbors:
                    laplacian[i] += (temperature[j] - temperature[i]) / (dist**2 + 0.1)
                laplacian[i] /= len(neighbors)
        
        return laplacian
    
    def _apply_thermal_bc(
        self, temperature: np.ndarray, thermal_boundary: Dict[str, float]
    ) -> np.ndarray:
        """Apply thermal boundary conditions."""
        if not thermal_boundary:
            return temperature
        
        for i, fiber in enumerate(self.network.fibers):
            center = (fiber.start_point + fiber.end_point) / 2
            for bc_name, bc_temp in thermal_boundary.items():
                if bc_name == 'top' and center[1] > 0.9 * (np.max([f.start_point[1] for f in self.network.fibers]) if self.network.num_fibers > 0 else 10.0):
                    temperature[i] = bc_temp
                elif bc_name == 'bottom' and center[1] < 0.1 * (np.max([f.start_point[1] for f in self.network.fibers]) if self.network.num_fibers > 0 else 10.0):
                    temperature[i] = bc_temp
                elif bc_name == 'left' and center[0] < 0.1 * (np.max([f.start_point[0] for f in self.network.fibers]) if self.network.num_fibers > 0 else 10.0):
                    temperature[i] = bc_temp
                elif bc_name == 'right' and center[0] > 0.9 * (np.max([f.start_point[0] for f in self.network.fibers]) if self.network.num_fibers > 0 else 10.0):
                    temperature[i] = bc_temp
        
        return temperature
    
    def _solve_mechanical_with_thermal(
        self, mechanical_load: Dict[str, Any], thermal_strain: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve mechanical problem accounting for thermal strain."""
        num_fibers = self.network.num_fibers
        
        # Compute effective strain from load
        strain_val = mechanical_load.get('strain', 0.0)
        axis = int(mechanical_load.get('axis', 0))
        
        # Mechanical strain = total strain - thermal strain
        total_strain = np.zeros((num_fibers, 3) if num_fibers > 0 else (0, 3))
        if num_fibers > 0:
            total_strain[:, axis] = strain_val
        
        # Mechanical strain = total - thermal
        if isinstance(thermal_strain, np.ndarray) and thermal_strain.shape == (num_fibers, 3):
            mechanical_strain = total_strain - thermal_strain
        elif isinstance(thermal_strain, np.ndarray) and thermal_strain.ndim == 1:
            thermal_strain_3d = np.zeros((num_fibers, 3))
            for i in range(3):
                thermal_strain_3d[:, i] = thermal_strain
            mechanical_strain = total_strain - thermal_strain_3d
        else:
            mechanical_strain = total_strain
        
        # Stress from mechanical strain only
        stress = self.youngs_modulus * mechanical_strain
        
        # Simple displacement estimate
        displacement = np.zeros((num_fibers, 3))
        for i, fiber in enumerate(self.network.fibers):
            length = fiber.length
            displacement[i] = mechanical_strain[i] * length
        
        return displacement, stress, mechanical_strain
    
    def _compute_thermal_energy(self, temperature: np.ndarray) -> float:
        """Compute thermal energy: integral of ρcT over volume."""
        return np.sum(self.density * self.specific_heat * temperature) * 1e-18  # Scale factor
    
    def _compute_mechanical_energy(self, stress: np.ndarray, strain: np.ndarray) -> float:
        """Compute mechanical strain energy: 0.5 * σ:ε integrated over volume."""
        if stress.ndim > 1 and strain.ndim > 1:
            energy_density = 0.5 * np.sum(stress * strain, axis=-1)
        else:
            energy_density = 0.5 * stress * strain
        return np.sum(energy_density) * 1e-18  # Scale factor


class ElectroMechanicalSolver:
    """
    Coupled electro-mechanical solver for piezoelectric fiber networks.
    
    Solves:
    - Mechanical equilibrium with piezoelectric coupling
    - Electric field equation: div(ε∇φ) = div(d:σ)
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network
    youngs_modulus : float
        Young's modulus (Pa)
    piezoelectric_coefficient : float
        Piezoelectric coefficient d33 (C/N)
    dielectric_constant : float
        Relative dielectric constant
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        youngs_modulus: float = 1e9,
        piezoelectric_coefficient: float = 1e-10,
        dielectric_constant: float = 1000.0,
    ):
        self.network = network
        self.youngs_modulus = youngs_modulus
        self.d33 = piezoelectric_coefficient
        self.dielectric = dielectric_constant
    
    def solve(
        self,
        mechanical_load: Dict[str, Any],
        voltage_boundary: Dict[str, float] = None,
    ) -> ElectroMechanicalResult:
        """
        Solve electro-mechanical coupled problem.
        
        Parameters
        ----------
        mechanical_load : dict
            Mechanical loading
        voltage_boundary : dict
            Voltage boundary conditions
        
        Returns
        -------
        ElectroMechanicalResult
        """
        num_fibers = self.network.num_fibers
        
        # Solve mechanical problem
        strain_val = mechanical_load.get('strain', 0.0)
        axis = mechanical_load.get('axis', 0)
        
        strain = np.zeros((num_fibers, 3))
        strain[:, axis] = strain_val
        stress = self.youngs_modulus * strain
        
        # Compute piezoelectric polarization
        polarization = self.d33 * np.sum(stress, axis=-1)
        
        # Compute electric potential from polarization
        potential = self._compute_potential(polarization, voltage_boundary)
        
        # Compute electric field (negative gradient of potential)
        electric_field = self._compute_electric_field(potential)
        
        # Piezoelectric strain contribution
        piezo_strain = self.d33 * electric_field
        
        # Total displacement
        displacement = np.zeros((num_fibers, 3))
        for i, fiber in enumerate(self.network.fibers):
            displacement[i] = strain[i] * fiber.length
            displacement[i, axis] += np.sum(piezo_strain[i]) * fiber.length
        
        return ElectroMechanicalResult(
            electric_potential=potential,
            electric_field=electric_field,
            displacement_field=displacement,
            stress_field=stress,
            piezoelectric_strain=piezo_strain,
            polarization=polarization,
            time_steps=np.array([0.0]),
        )
    
    def _compute_potential(
        self, polarization: np.ndarray, voltage_boundary: Optional[Dict[str, float]]
    ) -> np.ndarray:
        """Compute electric potential from polarization."""
        num_fibers = len(polarization)
        potential = np.zeros(num_fibers)
        
        # Simple model: potential proportional to polarization
        potential = polarization / (self.dielectric * 8.854e-12)  # ε0
        
        if voltage_boundary:
            for i, fiber in enumerate(self.network.fibers):
                center = (fiber.start_point + fiber.end_point) / 2
                for bc_name, voltage in voltage_boundary.items():
                    if bc_name == 'top' and center[1] > 0.9 * (np.max([f.start_point[1] for f in self.network.fibers]) if self.network.num_fibers > 0 else 10.0):
                        potential[i] = voltage
                    elif bc_name == 'bottom' and center[1] < 0.1 * (np.max([f.start_point[1] for f in self.network.fibers]) if self.network.num_fibers > 0 else 10.0):
                        potential[i] = voltage
        
        return potential
    
    def _compute_electric_field(self, potential: np.ndarray) -> np.ndarray:
        """Compute electric field from potential gradient."""
        num_fibers = len(potential)
        electric_field = np.zeros((num_fibers, 3))
        
        for i, fiber_i in enumerate(self.network.fibers):
            center_i = (fiber_i.start_point + fiber_i.end_point) / 2
            
            for j, fiber_j in enumerate(self.network.fibers):
                if i == j:
                    continue
                center_j = (fiber_j.start_point + fiber_j.end_point) / 2
                delta = center_j - center_i
                dist = np.linalg.norm(delta)
                
                if dist < 5.0 and dist > 0.1:
                    electric_field[i] -= (potential[j] - potential[i]) / (dist**2 + 0.1) * delta / dist
        
        return electric_field


def run_thermo_mechanical_analysis(
    network: FiberNetwork,
    strain: float = 0.01,
    axis: int = 0,
    temperature_diff: float = 100.0,
    **solver_kwargs
) -> ThermoMechanicalResult:
    """
    Convenience function for thermo-mechanical analysis.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network
    strain : float
        Applied strain
    axis : int
        Loading axis
    temperature_diff : float
        Temperature difference for thermal boundary
    **solver_kwargs
        Additional solver parameters
    
    Returns
    -------
    ThermoMechanicalResult
    """
    solver = ThermoMechanicalSolver(network, **solver_kwargs)
    
    mechanical_load = {'strain': strain, 'axis': axis}
    thermal_boundary = {
        'top': 293.15 + temperature_diff,
        'bottom': 293.15,
    }
    
    return solver.solve_steady_state(mechanical_load, thermal_boundary)

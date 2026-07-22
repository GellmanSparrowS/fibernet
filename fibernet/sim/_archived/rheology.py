"""
Rheology module for fiber suspensions.

Provides tools for simulating the rheological behavior of fiber suspensions:
- Shear viscosity
- Normal stress differences
- Orientation tensor evolution
- Fiber-fiber interactions in flow
- Jeffery orbit calculations
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ..core import FiberNetwork


@dataclass
class RheologyResult:
    """Result of a rheological simulation."""
    shear_rate: np.ndarray
    viscosity: np.ndarray
    first_normal_stress_diff: np.ndarray
    second_normal_stress_diff: np.ndarray
    shear_stress: np.ndarray
    orientation_tensor: np.ndarray
    time: np.ndarray


@dataclass
class JefferyOrbit:
    """Jeffery orbit for a single fiber in shear flow."""
    orientation: np.ndarray
    angular_velocity: np.ndarray
    period: float
    orbit_constant: float


class FiberSuspensionRheology:
    """
    Simulate rheology of fiber suspensions.
    
    Models dilute and semi-dilute fiber suspensions in Newtonian fluids,
    including fiber orientation evolution and effective viscosity.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network representing the suspension
    fluid_viscosity : float
        Background fluid viscosity (Pa·s)
    aspect_ratio : float
        Fiber aspect ratio (length / diameter)
    volume_fraction : float
        Fiber volume fraction
    interaction_parameter : float
        Fiber-fiber interaction coefficient (Ci)
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        fluid_viscosity: float = 1.0,
        aspect_ratio: float = 20.0,
        volume_fraction: float = 0.01,
        interaction_parameter: float = 0.01,
    ):
        self.network = network
        self.eta_s = fluid_viscosity
        self.ar = aspect_ratio
        self.phi = volume_fraction
        self.Ci = interaction_parameter
        
        # Shape factor for slender fibers
        self.xi = (self.ar**2 - 1) / (self.ar**2 + 1)
    
    def compute_effective_viscosity(
        self,
        shear_rate: float,
    ) -> float:
        """
        Compute effective viscosity of fiber suspension.
        
        Uses Batchelor's theory for dilute suspensions:
        eta_eff = eta_s * (1 + [eta] * phi)
        
        where [eta] is the intrinsic viscosity depending on aspect ratio.
        
        Parameters
        ----------
        shear_rate : float
            Applied shear rate (1/s)
        
        Returns
        -------
        float
            Effective viscosity (Pa·s)
        """
        # Intrinsic viscosity for slender fibers (Batchelor)
        # [eta] = (ar^2) / (6 * ln(2*ar) - 1.8) for ar >> 1
        if self.ar > 1:
            intrinsic_viscosity = self.ar**2 / (6 * np.log(2 * self.ar) - 1.8)
        else:
            intrinsic_viscosity = 2.5  # Einstein result for spheres
        
        # Effective viscosity
        eta_eff = self.eta_s * (1 + intrinsic_viscosity * self.phi)
        
        # Add fiber-fiber interaction contribution (semi-dilute)
        if self.phi > 1.0 / (self.ar**2):
            eta_interaction = self.eta_s * self.Ci * self.phi * self.ar**2
            eta_eff += eta_interaction
        
        return eta_eff
    
    def compute_shear_stress(
        self,
        shear_rate: float,
    ) -> float:
        """Compute shear stress at given shear rate."""
        eta_eff = self.compute_effective_viscosity(shear_rate)
        return eta_eff * shear_rate
    
    def compute_normal_stress_differences(
        self,
        shear_rate: float,
        orientation_tensor: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """
        Compute first and second normal stress differences.
        
        N1 = sigma_11 - sigma_22
        N2 = sigma_22 - sigma_33
        
        Parameters
        ----------
        shear_rate : float
            Applied shear rate (1/s)
        orientation_tensor : np.ndarray, optional
            Second-order orientation tensor a_ij
        
        Returns
        -------
        Tuple[float, float]
            (N1, N2) normal stress differences (Pa)
        """
        if orientation_tensor is None:
            # Default: isotropic orientation
            orientation_tensor = np.eye(3) / 3.0
        
        # Stress from fiber orientation (Dinh-Armstrong model)
        # sigma_ij = 2 * eta_s * D_ij + Np * phi * a_ijkl * D_kl
        # where Np = ar^2 / (3 * ln(2*ar))
        
        Np = self.ar**2 / (3 * np.log(2 * self.ar)) if self.ar > 1 else 0
        
        # Rate of deformation tensor for simple shear
        D = np.zeros((3, 3))
        D[0, 1] = D[1, 0] = shear_rate / 2.0
        
        # Fourth-order orientation tensor (quadratic closure)
        a4 = np.outer(orientation_tensor.flatten(), orientation_tensor.flatten()).reshape(3, 3, 3, 3)
        
        # Fiber stress contribution
        sigma_fiber = Np * self.phi * np.einsum('ijkl,kl->ij', a4, D)
        
        # Total stress
        sigma = 2 * self.eta_s * D + sigma_fiber
        
        # Normal stress differences
        N1 = sigma[0, 0] - sigma[1, 1]
        N2 = sigma[1, 1] - sigma[2, 2]
        
        return N1, N2
    
    def jeffery_orbit(
        self,
        initial_orientation: np.ndarray,
        shear_rate: float,
        total_time: float = 10.0,
        num_steps: int = 1000,
    ) -> JefferyOrbit:
        """
        Compute Jeffery orbit for a single fiber in shear flow.
        
        The orientation of a spheroidal particle in shear flow follows:
        dp/dt = Omega.p + xi*(E.p - E:ppp)
        
        Parameters
        ----------
        initial_orientation : np.ndarray
            Initial orientation vector (3D unit vector)
        shear_rate : float
            Applied shear rate (1/s)
        total_time : float
            Total simulation time (s)
        num_steps : int
            Number of time steps
        
        Returns
        -------
        JefferyOrbit
            Jeffery orbit result
        """
        dt = total_time / num_steps
        p = np.array(initial_orientation, dtype=float)
        p /= np.linalg.norm(p)
        
        orientations = [p.copy()]
        angular_velocities = []
        
        # Vorticity tensor
        Omega = np.zeros((3, 3))
        Omega[0, 1] = shear_rate / 2.0
        Omega[1, 0] = -shear_rate / 2.0
        
        # Rate of strain tensor
        E = np.zeros((3, 3))
        E[0, 1] = E[1, 0] = shear_rate / 2.0
        
        for _ in range(num_steps):
            # Jeffery equation
            dp_dt = Omega @ p + self.xi * (E @ p - np.dot(p, E @ p) * p)
            angular_velocities.append(dp_dt.copy())
            
            # Forward Euler integration
            p = p + dt * dp_dt
            p /= np.linalg.norm(p)  # Normalize
            
            orientations.append(p.copy())
        
        orientations = np.array(orientations)
        angular_velocities = np.array(angular_velocities)
        
        # Compute period (T = 2*pi*(ar + 1/ar) / shear_rate)
        period = 2 * np.pi * (self.ar + 1.0 / self.ar) / shear_rate
        
        # Orbit constant
        C = np.tan(np.arctan2(p[1], p[0])) * self.ar
        
        return JefferyOrbit(
            orientation=orientations,
            angular_velocity=angular_velocities,
            period=period,
            orbit_constant=C,
        )
    
    def orientation_evolution(
        self,
        initial_orientation_tensor: Optional[np.ndarray] = None,
        shear_rate: float = 1.0,
        total_time: float = 10.0,
        num_steps: int = 100,
    ) -> np.ndarray:
        """
        Compute evolution of second-order orientation tensor.
        
        da_ij/dt = (Omega_ik*a_kj - a_ik*Omega_kj)
                 + xi*(E_ik*a_kj + a_ik*E_kj - 2*E_kl*a_ijkl)
                 + 2*Ci*gamma_dot*(delta_ij - 3*a_ij)
        
        Parameters
        ----------
        initial_orientation_tensor : np.ndarray, optional
            Initial orientation tensor (3x3)
        shear_rate : float
            Applied shear rate
        total_time : float
            Total time
        num_steps : int
            Number of steps
        
        Returns
        -------
        np.ndarray
            Orientation tensor history (num_steps+1, 3, 3)
        """
        dt = total_time / num_steps
        
        if initial_orientation_tensor is None:
            a = np.eye(3) / 3.0  # Isotropic
        else:
            a = np.array(initial_orientation_tensor)
        
        # Vorticity and strain rate tensors
        Omega = np.zeros((3, 3))
        Omega[0, 1] = shear_rate / 2.0
        Omega[1, 0] = -shear_rate / 2.0
        
        E = np.zeros((3, 3))
        E[0, 1] = E[1, 0] = shear_rate / 2.0
        
        history = [a.copy()]
        
        for _ in range(num_steps):
            # Quadratic closure for 4th order tensor
            a4 = np.einsum('ij,kl->ijkl', a, a)
            
            # Folgar-Tucker equation
            da = np.zeros((3, 3))
            
            # Vorticity contribution
            da += Omega @ a - a @ Omega
            
            # Strain rate contribution
            da += self.xi * (E @ a + a @ E - 2 * np.einsum('kl,ijkl->ij', E, a4))
            
            # Diffusion (fiber interaction)
            da += 2 * self.Ci * shear_rate * (np.eye(3) - 3 * a)
            
            a = a + dt * da
            
            # Ensure symmetry
            a = (a + a.T) / 2.0
            
            # Ensure trace = 1
            a /= np.trace(a)
            
            history.append(a.copy())
        
        return np.array(history)
    
    def shear_flow_sweep(
        self,
        shear_rates: np.ndarray,
    ) -> RheologyResult:
        """
        Compute rheological properties over a range of shear rates.
        
        Parameters
        ----------
        shear_rates : np.ndarray
            Array of shear rates (1/s)
        
        Returns
        -------
        RheologyResult
            Rheological properties at each shear rate
        """
        viscosities = []
        shear_stresses = []
        N1_values = []
        N2_values = []
        
        for gamma_dot in shear_rates:
            eta = self.compute_effective_viscosity(gamma_dot)
            tau = self.compute_shear_stress(gamma_dot)
            N1, N2 = self.compute_normal_stress_differences(gamma_dot)
            
            viscosities.append(eta)
            shear_stresses.append(tau)
            N1_values.append(N1)
            N2_values.append(N2)
        
        return RheologyResult(
            shear_rate=shear_rates,
            viscosity=np.array(viscosities),
            first_normal_stress_diff=np.array(N1_values),
            second_normal_stress_diff=np.array(N2_values),
            shear_stress=np.array(shear_stresses),
            orientation_tensor=np.eye(3) / 3.0,
            time=np.zeros(len(shear_rates)),
        )
    
    def transient_shear(
        self,
        shear_rate: float,
        total_time: float = 10.0,
        num_steps: int = 100,
    ) -> RheologyResult:
        """
        Compute transient response to startup of steady shear.
        
        Parameters
        ----------
        shear_rate : float
            Applied shear rate (1/s)
        total_time : float
            Total simulation time (s)
        num_steps : int
            Number of time steps
        
        Returns
        -------
        RheologyResult
            Time-dependent rheological response
        """
        dt = total_time / num_steps
        time = np.linspace(0, total_time, num_steps + 1)
        
        # Compute orientation evolution
        a_history = self.orientation_evolution(
            initial_orientation_tensor=None,
            shear_rate=shear_rate,
            total_time=total_time,
            num_steps=num_steps,
        )
        
        viscosities = []
        shear_stresses = []
        N1_values = []
        N2_values = []
        
        for a in a_history:
            N1, N2 = self.compute_normal_stress_differences(shear_rate, a)
            eta = self.compute_effective_viscosity(shear_rate)
            tau = eta * shear_rate
            
            viscosities.append(eta)
            shear_stresses.append(tau)
            N1_values.append(N1)
            N2_values.append(N2)
        
        return RheologyResult(
            shear_rate=np.full(num_steps + 1, shear_rate),
            viscosity=np.array(viscosities),
            first_normal_stress_diff=np.array(N1_values),
            second_normal_stress_diff=np.array(N2_values),
            shear_stress=np.array(shear_stresses),
            orientation_tensor=a_history,
            time=time,
        )


def compute_intrinsic_viscosity(aspect_ratio: float) -> float:
    """
    Compute intrinsic viscosity [eta] for a given aspect ratio.
    
    Parameters
    ----------
    aspect_ratio : float
        Fiber aspect ratio (length / diameter)
    
    Returns
    -------
    float
        Intrinsic viscosity
    """
    if aspect_ratio <= 1:
        return 2.5  # Einstein result
    
    # Batchelor's result for slender fibers
    return aspect_ratio**2 / (6 * np.log(2 * aspect_ratio) - 1.8)


def compute_dilute_limit_viscosity(
    fluid_viscosity: float,
    aspect_ratio: float,
    volume_fraction: float,
) -> float:
    """
    Compute viscosity in the dilute limit.
    
    eta = eta_s * (1 + [eta] * phi)
    
    Parameters
    ----------
    fluid_viscosity : float
        Solvent viscosity (Pa·s)
    aspect_ratio : float
        Fiber aspect ratio
    volume_fraction : float
        Fiber volume fraction
    
    Returns
    -------
    float
        Effective viscosity (Pa·s)
    """
    intrinsic = compute_intrinsic_viscosity(aspect_ratio)
    return fluid_viscosity * (1 + intrinsic * volume_fraction)

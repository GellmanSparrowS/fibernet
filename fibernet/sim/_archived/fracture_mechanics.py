"""
Fracture mechanics module for fiber networks.

Provides tools for:
- Crack propagation simulation
- J-integral calculation
- Energy release rate computation
- Stress intensity factors
- Fracture toughness characterization
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from ..core import FiberNetwork


@dataclass
class CrackTip:
    """Represents a crack tip in the network."""
    position: np.ndarray
    direction: np.ndarray
    length: float
    k_I: float  # Mode I stress intensity factor
    k_II: float  # Mode II stress intensity factor
    energy_release_rate: float


@dataclass
class FractureResult:
    """Results from fracture mechanics analysis."""
    crack_tips: List[CrackTip]
    j_integral: float
    critical_load: float
    fracture_toughness: float
    crack_path: np.ndarray
    energy_dissipated: float


class CrackPropagationSolver:
    """
    Simulates crack propagation in fiber networks.
    
    Uses maximum circumferential stress criterion for crack growth direction
    and energy-based criterion for crack initiation.
    
    Parameters
    ----------
    network : FiberNetwork
        The fiber network to analyze
    fracture_toughness : float
        Critical energy release rate G_c (J/m²)
    element_length : float
        Characteristic element length for crack growth increment
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        fracture_toughness: float = 100.0,
        element_length: float = 0.1,
    ):
        self.network = network
        self.G_c = fracture_toughness
        self.da = element_length
        self.crack_tips: List[CrackTip] = []
        self.broken_fibers: List[int] = []
    
    def initialize_crack(
        self,
        tip_position: np.ndarray,
        tip_direction: np.ndarray,
        initial_length: float = 0.0,
    ) -> CrackTip:
        """
        Initialize a crack tip.
        
        Parameters
        ----------
        tip_position : np.ndarray
            Initial position of crack tip
        tip_direction : np.ndarray
            Initial crack growth direction
        initial_length : float
            Initial crack length
            
        Returns
        -------
        CrackTip
            Initialized crack tip
        """
        tip = CrackTip(
            position=np.array(tip_position),
            direction=np.array(tip_direction) / np.linalg.norm(tip_direction),
            length=initial_length,
            k_I=0.0,
            k_II=0.0,
            energy_release_rate=0.0,
        )
        self.crack_tips.append(tip)
        return tip
    
    def compute_stress_intensity_factors(
        self,
        crack_tip: CrackTip,
        stress_field: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute Mode I and Mode II stress intensity factors.
        
        Uses displacement correlation method.
        
        Parameters
        ----------
        crack_tip : CrackTip
            The crack tip to analyze
        stress_field : np.ndarray
            Stress field around the crack tip
            
        Returns
        -------
        Tuple[float, float]
            (K_I, K_II) stress intensity factors
        """
        # Simplified SIF calculation based on stress field
        # In practice, would use more sophisticated methods
        
        tip_pos = crack_tip.position
        tip_dir = crack_tip.direction
        
        # Find nodes near crack tip
        r_char = self.da * 2  # Characteristic radius
        
        # Sample stress at points ahead of crack tip
        stress_ahead = []
        for r in [r_char * 0.5, r_char, r_char * 1.5]:
            sample_point = tip_pos + r * tip_dir
            stress = self._interpolate_stress(sample_point, stress_field)
            stress_ahead.append(stress)
        
        # Mode I: opening stress perpendicular to crack
        normal = np.array([-tip_dir[1], tip_dir[0], 0]) if len(tip_dir) > 2 else np.array([-tip_dir[1], tip_dir[0]])
        
        sigma_yy_avg = np.mean([np.dot(s, normal) if len(s) == len(normal) else 0 for s in stress_ahead])
        
        # Mode II: shear stress parallel to crack
        sigma_xy_avg = np.mean([np.dot(s, tip_dir) if len(s) == len(tip_dir) else 0 for s in stress_ahead])
        
        # K = sigma * sqrt(pi * a)
        a = crack_tip.length if crack_tip.length > 0 else self.da
        
        K_I = sigma_yy_avg * np.sqrt(np.pi * a)
        K_II = sigma_xy_avg * np.sqrt(np.pi * a)
        
        return K_I, K_II
    
    def compute_j_integral(
        self,
        crack_tip: CrackTip,
        stress_field: np.ndarray,
        strain_field: np.ndarray,
        displacement_field: np.ndarray,
    ) -> float:
        """
        Compute J-integral around crack tip.
        
        J = integral(W*dy - T*du/dx*ds) along contour
        
        Parameters
        ----------
        crack_tip : CrackTip
            The crack tip
        stress_field : np.ndarray
            Stress field
        strain_field : np.ndarray
            Strain field
        displacement_field : np.ndarray
            Displacement field
            
        Returns
        -------
        float
            J-integral value
        """
        # Simplified J-integral calculation
        # J = G = (K_I^2 + K_II^2) / E' for plane stress
        
        K_I, K_II = self.compute_stress_intensity_factors(crack_tip, stress_field)
        
        # Assume plane stress
        E = self.network.fibers[0].material.youngs_modulus if len(self.network.fibers) > 0 else 1e9
        nu = self.network.fibers[0].material.poissons_ratio if len(self.network.fibers) > 0 else 0.3
        
        E_prime = E  # Plane stress: E' = E
        
        J = (K_I**2 + K_II**2) / E_prime
        
        return J
    
    def compute_crack_growth_direction(
        self,
        crack_tip: CrackTip,
    ) -> np.ndarray:
        """
        Compute crack growth direction using maximum circumferential stress criterion.
        
        theta_c = 2*arctan((K_I - sqrt(K_I^2 + 8*K_II^2)) / (4*K_II))
        
        Parameters
        ----------
        crack_tip : CrackTip
            The crack tip
            
        Returns
        -------
        np.ndarray
            Growth direction vector
        """
        K_I = crack_tip.k_I
        K_II = crack_tip.k_II
        
        if abs(K_II) < 1e-10:
            # Pure Mode I: continue in current direction
            return crack_tip.direction
        
        # Maximum circumferential stress criterion
        discriminant = K_I**2 + 8 * K_II**2
        if discriminant < 0:
            theta_c = 0.0
        else:
            numerator = K_I - np.sqrt(discriminant)
            denominator = 4 * K_II
            if abs(denominator) < 1e-10:
                theta_c = 0.0
            else:
                theta_c = 2 * np.arctan(numerator / denominator)
        
        # Rotate current direction by theta_c
        cos_theta = np.cos(theta_c)
        sin_theta = np.sin(theta_c)
        
        current_dir = crack_tip.direction
        if len(current_dir) == 3:
            # 3D: rotate in x-y plane
            rotation_matrix = np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])
        else:
            # 2D
            rotation_matrix = np.array([
                [cos_theta, -sin_theta],
                [sin_theta, cos_theta]
            ])
        
        new_direction = rotation_matrix @ current_dir
        return new_direction / np.linalg.norm(new_direction)
    
    def propagate_crack(
        self,
        crack_tip: CrackTip,
        stress_field: np.ndarray,
        strain_field: Optional[np.ndarray] = None,
        displacement_field: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Propagate crack by one increment.
        
        Parameters
        ----------
        crack_tip : CrackTip
            The crack tip to propagate
        stress_field : np.ndarray
            Current stress field
        strain_field : np.ndarray, optional
            Current strain field
        displacement_field : np.ndarray, optional
            Current displacement field
            
        Returns
        -------
        bool
            True if crack propagated, False if arrested
        """
        # Compute SIFs
        K_I, K_II = self.compute_stress_intensity_factors(crack_tip, stress_field)
        crack_tip.k_I = K_I
        crack_tip.k_II = K_II
        
        # Compute energy release rate
        if strain_field is not None and displacement_field is not None:
            G = self.compute_j_integral(crack_tip, stress_field, strain_field, displacement_field)
        else:
            # Use K-based estimate
            E = self.network.fibers[0].material.youngs_modulus if len(self.network.fibers) > 0 else 1e9
            G = (K_I**2 + K_II**2) / E
        
        crack_tip.energy_release_rate = G
        
        # Check if crack should propagate
        if G < self.G_c:
            return False  # Crack arrested
        
        # Compute growth direction
        growth_dir = self.compute_crack_growth_direction(crack_tip)
        
        # Update crack tip
        crack_tip.position = crack_tip.position + self.da * growth_dir
        crack_tip.direction = growth_dir
        crack_tip.length += self.da
        
        # Break fibers in crack path
        self._break_fibers_in_path(crack_tip.position, self.da)
        
        return True
    
    def _break_fibers_in_path(
        self,
        crack_position: np.ndarray,
        radius: float,
    ):
        """Break fibers within radius of crack tip."""
        for i, fiber in enumerate(self.network.fibers):
            if i in self.broken_fibers:
                continue
            
            fiber_center = np.mean(fiber.centerline, axis=0)
            distance = np.linalg.norm(fiber_center - crack_position)
            
            if distance < radius:
                self.broken_fibers.append(i)
    
    def _interpolate_stress(
        self,
        point: np.ndarray,
        stress_field: np.ndarray,
    ) -> np.ndarray:
        """Interpolate stress at given point."""
        # Simplified: return average stress
        if len(stress_field) > 0:
            return np.mean(stress_field, axis=0)
        return np.zeros(3)
    
    def simulate_propagation(
        self,
        stress_field: np.ndarray,
        max_steps: int = 100,
        strain_field: Optional[np.ndarray] = None,
        displacement_field: Optional[np.ndarray] = None,
    ) -> FractureResult:
        """
        Simulate crack propagation over multiple steps.
        
        Parameters
        ----------
        stress_field : np.ndarray
            Applied stress field
        max_steps : int
            Maximum number of propagation steps
        strain_field : np.ndarray, optional
            Strain field
        displacement_field : np.ndarray, optional
            Displacement field
            
        Returns
        -------
        FractureResult
            Results of fracture simulation
        """
        crack_path = []
        
        for step in range(max_steps):
            propagated = False
            
            for tip in self.crack_tips:
                crack_path.append(tip.position.copy())
                
                if self.propagate_crack(tip, stress_field, strain_field, displacement_field):
                    propagated = True
            
            if not propagated:
                break  # All cracks arrested
        
        if len(crack_path) > 0:
            crack_path = np.array(crack_path)
        else:
            crack_path = np.array([])
        
        # Compute final metrics
        final_J = 0.0
        final_K = 0.0
        
        if len(self.crack_tips) > 0:
            tip = self.crack_tips[0]
            if strain_field is not None and displacement_field is not None:
                final_J = self.compute_j_integral(tip, stress_field, strain_field, displacement_field)
            else:
                E = self.network.fibers[0].material.youngs_modulus if len(self.network.fibers) > 0 else 1e9
                final_J = (tip.k_I**2 + tip.k_II**2) / E
            
            final_K = np.sqrt(tip.k_I**2 + tip.k_II**2)
        
        # Energy dissipated
        energy_dissipated = self.G_c * len(self.broken_fibers) * self.da
        
        return FractureResult(
            crack_tips=self.crack_tips,
            j_integral=final_J,
            critical_load=0.0,  # Would need load tracking
            fracture_toughness=final_K,
            crack_path=crack_path,
            energy_dissipated=energy_dissipated,
        )


def compute_energy_release_rate(
    network: FiberNetwork,
    crack_length: float,
    applied_stress: float,
    geometry_factor: float = 1.12,
) -> float:
    """
    Compute energy release rate G for a crack.
    
    G = (K^2) / E' where K = Y * sigma * sqrt(pi * a)
    
    Parameters
    ----------
    network : FiberNetwork
        The fiber network
    crack_length : float
        Crack length a
    applied_stress : float
        Applied remote stress sigma
    geometry_factor : float
        Geometry correction factor Y
        
    Returns
    -------
    float
        Energy release rate G (J/m²)
    """
    if len(network.fibers) == 0:
        return 0.0
    
    E = network.fibers[0].material.youngs_modulus
    nu = network.fibers[0].material.poissons_ratio
    
    # Plane stress
    E_prime = E
    
    K = geometry_factor * applied_stress * np.sqrt(np.pi * crack_length)
    G = K**2 / E_prime
    
    return G


def compute_fracture_toughness(
    network: FiberNetwork,
    critical_load: float,
    crack_length: float,
    specimen_width: float,
    geometry_factor: Optional[float] = None,
) -> float:
    """
    Compute fracture toughness K_c from critical load.
    
    K_c = Y * sigma_c * sqrt(pi * a)
    
    Parameters
    ----------
    network : FiberNetwork
        The fiber network
    critical_load : float
        Critical load at fracture (N)
    crack_length : float
        Initial crack length (m)
    specimen_width : float
        Specimen width (m)
    geometry_factor : float, optional
        Geometry factor Y (default: compact tension formula)
        
    Returns
    -------
    float
        Fracture toughness K_c (Pa·m^0.5)
    """
    # Compute critical stress
    if len(network.fibers) == 0:
        return 0.0
    
    # Assume unit thickness
    thickness = 1.0
    sigma_c = critical_load / (specimen_width * thickness)
    
    # Default geometry factor for center-cracked plate
    if geometry_factor is None:
        a_over_w = crack_length / specimen_width
        # Secant formula
        geometry_factor = np.sqrt(1.0 / np.cos(np.pi * a_over_w / 2))
    
    K_c = geometry_factor * sigma_c * np.sqrt(np.pi * crack_length)
    
    return K_c

"""
Multi-scale modeling framework for fiber networks.

Provides tools for:
- Homogenization of fiber network properties
- Representative Volume Element (RVE) analysis
- Scale bridging between micro and macro scales
- Effective property computation
- Hierarchical modeling
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import sparse
from ..core import FiberNetwork


@dataclass
class HomogenizedProperties:
    """Effective (homogenized) material properties."""
    # Elastic properties
    youngs_modulus_x: float
    youngs_modulus_y: float
    youngs_modulus_z: float
    poissons_ratio_xy: float
    poissons_ratio_xz: float
    poissons_ratio_yz: float
    shear_modulus_xy: float
    shear_modulus_xz: float
    shear_modulus_yz: float
    
    # Thermal properties
    thermal_conductivity_x: float
    thermal_conductivity_y: float
    thermal_conductivity_z: float
    thermal_expansion_x: float
    thermal_expansion_y: float
    thermal_expansion_z: float
    
    # Density and porosity
    density: float
    porosity: float
    
    @property
    def is_isotropic(self) -> bool:
        """Check if properties are approximately isotropic."""
        E_mean = np.mean([self.youngs_modulus_x, self.youngs_modulus_y, self.youngs_modulus_z])
        E_std = np.std([self.youngs_modulus_x, self.youngs_modulus_y, self.youngs_modulus_z])
        return E_std / E_mean < 0.1
    
    @property
    def effective_youngs_modulus(self) -> float:
        """Average Young's modulus."""
        return np.mean([self.youngs_modulus_x, self.youngs_modulus_y, self.youngs_modulus_z])


@dataclass
class RVEResult:
    """Result of RVE analysis."""
    homogenized_properties: HomogenizedProperties
    stress_field: np.ndarray
    strain_field: np.ndarray
    displacement_field: np.ndarray
    convergence_history: List[float]


class HomogenizationSolver:
    """
    Compute homogenized (effective) properties of fiber networks.
    
    Uses computational homogenization theory to derive effective
    macroscopic properties from microscopic fiber network structure.
    
    Parameters
    ----------
    network : FiberNetwork
        Representative fiber network (RVE)
    fiber_youngs_modulus : float
        Young's modulus of individual fibers (Pa)
    fiber_poissons_ratio : float
        Poisson's ratio of fibers
    fiber_thermal_conductivity : float
        Thermal conductivity of fibers (W/m·K)
    fiber_density : float
        Density of fiber material (kg/m³)
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        fiber_youngs_modulus: float = 1e9,
        fiber_poissons_ratio: float = 0.3,
        fiber_thermal_conductivity: float = 0.5,
        fiber_density: float = 1000.0,
    ):
        self.network = network
        self.E_f = fiber_youngs_modulus
        self.nu_f = fiber_poissons_ratio
        self.k_f = fiber_thermal_conductivity
        self.rho_f = fiber_density
    
    def compute_elastic_properties(self) -> Tuple[float, float, float, float, float, float]:
        """
        Compute effective elastic properties using orientation averaging.
        
        Returns
        -------
        Tuple
            (E_x, E_y, E_z, nu_xy, G_xy, G_xz)
        """
        if self.network.num_fibers == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Compute orientation distribution
        orientations = []
        for fiber in self.network.fibers:
            centerline = fiber.centerline
            direction = centerline[-1] - centerline[0]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                orientations.append(direction / direction_norm)
        
        if len(orientations) == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        orientations = np.array(orientations)
        
        # Compute volume fraction
        V_f = self._compute_volume_fraction()
        
        # Second-order orientation tensor
        a_ij = np.einsum('ij,ik->jk', orientations, orientations) / len(orientations)
        
        # Effective stiffness using orientation averaging
        # Simplified: use rule of mixtures with orientation factor
        
        # Effective Young's modulus in each direction
        E_x = self.E_f * V_f * a_ij[0, 0]
        E_y = self.E_f * V_f * a_ij[1, 1]
        E_z = self.E_f * V_f * (a_ij[2, 2] if a_ij.shape[0] > 2 else 0.0)
        
        # Ensure minimum values
        E_x = max(E_x, 1e3)
        E_y = max(E_y, 1e3)
        E_z = max(E_z, 1e3)
        
        # Effective Poisson's ratio
        nu_xy = self.nu_f * V_f
        
        # Effective shear modulus
        G_xy = E_x / (2 * (1 + nu_xy))
        G_xz = E_x / (2 * (1 + nu_xy))
        
        return (E_x, E_y, E_z, nu_xy, G_xy, G_xz)
    
    def compute_thermal_properties(self) -> Tuple[float, float, float, float, float, float]:
        """
        Compute effective thermal properties.
        
        Returns
        -------
        Tuple
            (k_x, k_y, k_z, alpha_x, alpha_y, alpha_z)
        """
        if self.network.num_fibers == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        V_f = self._compute_volume_fraction()
        
        # Thermal conductivity (rule of mixtures)
        k_x = self.k_f * V_f
        k_y = self.k_f * V_f
        k_z = self.k_f * V_f
        
        # Thermal expansion (simplified)
        alpha_f = 1e-5  # Fiber CTE
        alpha_x = alpha_f * V_f
        alpha_y = alpha_f * V_f
        alpha_z = alpha_f * V_f
        
        return (k_x, k_y, k_z, alpha_x, alpha_y, alpha_z)
    
    def _compute_volume_fraction(self) -> float:
        """Compute fiber volume fraction."""
        if self.network.num_fibers == 0:
            return 0.0
        
        # Compute total fiber volume
        fiber_volume = 0.0
        for fiber in self.network.fibers:
            # Assume cylindrical fibers
            radius = fiber.radius
            length = fiber.length
            fiber_volume += np.pi * radius**2 * length
        
        # Compute RVE volume
        if hasattr(self.network, 'box_size') and self.network.box_size is not None:
            box = self.network.box_size
            if len(box) == 3:
                rve_volume = box[0] * box[1] * box[2]
            else:
                # 2D: assume unit thickness
                rve_volume = box[0] * box[1] * 1.0
        else:
            # Compute from fiber positions
            all_points = np.vstack([f.centerline for f in self.network.fibers])
            bbox = np.max(all_points, axis=0) - np.min(all_points, axis=0)
            if len(bbox) == 3:
                rve_volume = bbox[0] * bbox[1] * bbox[2]
            else:
                rve_volume = bbox[0] * bbox[1] * 1.0
            rve_volume = max(rve_volume, 1.0)
        
        if rve_volume <= 0:
            return 0.0
        V_f = fiber_volume / rve_volume
        return min(V_f, 0.9)  # Cap at 90%
    
    def homogenize(self) -> HomogenizedProperties:
        """
        Compute all homogenized properties.
        
        Returns
        -------
        HomogenizedProperties
            Effective material properties
        """
        # Elastic properties
        E_x, E_y, E_z, nu_xy, G_xy, G_xz = self.compute_elastic_properties()
        
        # Thermal properties
        k_x, k_y, k_z, alpha_x, alpha_y, alpha_z = self.compute_thermal_properties()
        
        # Density and porosity
        V_f = self._compute_volume_fraction()
        rho_eff = self.rho_f * V_f
        porosity = 1.0 - V_f
        
        return HomogenizedProperties(
            youngs_modulus_x=E_x,
            youngs_modulus_y=E_y,
            youngs_modulus_z=E_z,
            poissons_ratio_xy=nu_xy,
            poissons_ratio_xz=nu_xy,
            poissons_ratio_yz=nu_xy,
            shear_modulus_xy=G_xy,
            shear_modulus_xz=G_xz,
            shear_modulus_yz=G_xz,
            thermal_conductivity_x=k_x,
            thermal_conductivity_y=k_y,
            thermal_conductivity_z=k_z,
            thermal_expansion_x=alpha_x,
            thermal_expansion_y=alpha_y,
            thermal_expansion_z=alpha_z,
            density=rho_eff,
            porosity=porosity,
        )


class RVEAnalyzer:
    """
    Analyze Representative Volume Elements (RVEs) for fiber networks.
    
    Performs:
    - RVE size convergence studies
    - Boundary condition application
    - Stress/strain field computation
    - Effective property extraction
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network RVE
    youngs_modulus : float
        Fiber Young's modulus (Pa)
    poisson_ratio : float
        Fiber Poisson's ratio
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        youngs_modulus: float = 1e9,
        poisson_ratio: float = 0.3,
    ):
        self.network = network
        self.E = youngs_modulus
        self.nu = poisson_ratio
    
    def apply_periodic_bc(
        self,
        macro_strain: np.ndarray,
    ) -> np.ndarray:
        """
        Apply periodic boundary conditions for given macroscopic strain.
        
        Parameters
        ----------
        macro_strain : np.ndarray
            Macroscopic strain tensor (3x3 or 6-vector)
        
        Returns
        -------
        np.ndarray
            Displacement field
        """
        if macro_strain.shape == (6,):
            # Voigt notation to tensor
            eps = np.array([
                [macro_strain[0], macro_strain[5]/2, macro_strain[4]/2],
                [macro_strain[5]/2, macro_strain[1], macro_strain[3]/2],
                [macro_strain[4]/2, macro_strain[3]/2, macro_strain[2]],
            ])
        else:
            eps = macro_strain
        
        # Compute displacement for each fiber
        displacements = np.zeros((self.network.num_fibers, 3))
        
        for i, fiber in enumerate(self.network.fibers):
            center = np.mean(fiber.centerline, axis=0)
            displacements[i] = eps @ center
        
        return displacements
    
    def compute_effective_stiffness(
        self,
        num_tests: int = 6,
    ) -> np.ndarray:
        """
        Compute effective stiffness tensor using numerical tests.
        
        Parameters
        ----------
        num_tests : int
            Number of independent tests (6 for full 3D)
        
        Returns
        -------
        np.ndarray
            Effective stiffness tensor (6x6 in Voigt notation)
        """
        C_eff = np.zeros((6, 6))
        
        # Apply 6 independent strain
        unit_strains = np.eye(6)
        
        for i in range(6):
            # Apply unit strain
            displacements = self.apply_periodic_bc(unit_strains[i])
            
            # Compute average stress (simplified)
            stress = self._compute_average_stress(displacements)
            
            # Store in stiffness matrix
            C_eff[:, i] = stress
        
        return C_eff
    
    def _compute_average_stress(
        self,
        displacements: np.ndarray,
    ) -> np.ndarray:
        """Compute volume-averaged stress."""
        # Simplified: use linear elasticity
        stress = np.zeros(6)
        
        for i, fiber in enumerate(self.network.fibers):
            # Fiber strain from displacement
            centerline = fiber.centerline
            direction = centerline[-1] - centerline[0]
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > 0:
                direction = direction / direction_norm
                fiber_strain = np.dot(displacements[i], direction) / fiber.length
                fiber_stress = self.E * fiber_strain
                
                # Add to average (Voigt notation)
                stress[0] += fiber_stress * direction[0]**2
                stress[1] += fiber_stress * direction[1]**2
                stress[2] += fiber_stress * (direction[2]**2 if len(direction) > 2 else 0)
                stress[3] += fiber_stress * direction[1] * (direction[2] if len(direction) > 2 else 0)
                stress[4] += fiber_stress * direction[0] * (direction[2] if len(direction) > 2 else 0)
                stress[5] += fiber_stress * direction[0] * direction[1]
        
        # Normalize by volume
        V_f = self._compute_volume_fraction()
        if V_f > 0:
            stress /= V_f
        
        return stress
    
    def _compute_volume_fraction(self) -> float:
        """Compute fiber volume fraction."""
        solver = HomogenizationSolver(self.network)
        return solver._compute_volume_fraction()
    
    def convergence_study(
        self,
        generator_func,
        generator_params: Dict,
        sizes: List[float],
        num_samples: int = 3,
    ) -> Dict:
        """
        Perform RVE size convergence study.
        
        Parameters
        ----------
        generator_func : callable
            Network generator function
        generator_params : Dict
            Generator parameters
        sizes : List[float]
            RVE sizes to test
        num_samples : int
            Number of samples per size
        
        Returns
        -------
        Dict
            Convergence study results
        """
        results = {
            'sizes': sizes,
            'youngs_modulus_mean': [],
            'youngs_modulus_std': [],
            'porosity_mean': [],
        }
        
        for size in sizes:
            E_values = []
            porosity_values = []
            
            for sample in range(num_samples):
                params = generator_params.copy()
                params['box_size'] = (size, size, size) if '3d' in generator_func.__name__.lower() else (size, size)
                params['seed'] = sample
                
                net = generator_func(**params)
                
                homogenizer = HomogenizationSolver(net, fiber_youngs_modulus=self.E)
                props = homogenizer.homogenize()
                
                E_values.append(props.effective_youngs_modulus)
                porosity_values.append(props.porosity)
            
            results['youngs_modulus_mean'].append(np.mean(E_values))
            results['youngs_modulus_std'].append(np.std(E_values))
            results['porosity_mean'].append(np.mean(porosity_values))
        
        return results


def compute_effective_properties(
    network: FiberNetwork,
    fiber_properties: Optional[Dict] = None,
) -> HomogenizedProperties:
    """
    Convenience function to compute effective properties.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network RVE
    fiber_properties : Dict, optional
        Dictionary of fiber material properties
    
    Returns
    -------
    HomogenizedProperties
        Effective material properties
    """
    if fiber_properties is None:
        fiber_properties = {
            'fiber_youngs_modulus': 1e9,
            'fiber_poissons_ratio': 0.3,
            'fiber_thermal_conductivity': 0.5,
            'fiber_density': 1000.0,
        }
    
    solver = HomogenizationSolver(network, **fiber_properties)
    return solver.homogenize()


def estimate_rve_size(
    generator_func,
    generator_params: Dict,
    fiber_youngs_modulus: float = 1e9,
    tolerance: float = 0.05,
    size_range: Tuple[float, float] = (10.0, 100.0),
    num_points: int = 5,
) -> float:
    """
    Estimate minimum RVE size for convergence.
    
    Parameters
    ----------
    generator_func : callable
        Network generator
    generator_params : Dict
        Generator parameters
    fiber_youngs_modulus : float
        Fiber Young's modulus
    tolerance : float
        Convergence tolerance (relative std)
    size_range : Tuple[float, float]
        Range of sizes to test
    num_points : int
        Number of size points
    
    Returns
    -------
    float
        Estimated minimum RVE size
    """
    sizes = np.linspace(size_range[0], size_range[1], num_points)
    
    analyzer = RVEAnalyzer(None, youngs_modulus=fiber_youngs_modulus)
    results = analyzer.convergence_study(
        generator_func,
        generator_params,
        sizes.tolist(),
        num_samples=3,
    )
    
    # Find where relative std < tolerance
    rel_std = np.array(results['youngs_modulus_std']) / np.array(results['youngs_modulus_mean'])
    
    converged_idx = np.where(rel_std < tolerance)[0]
    
    if len(converged_idx) > 0:
        return sizes[converged_idx[0]]
    else:
        return sizes[-1]  # Return largest size

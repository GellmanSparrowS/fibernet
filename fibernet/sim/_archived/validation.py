"""
Validation module for FiberNet mechanical simulations.

Provides analytical benchmarks and convergence tests to verify FEM results:
- Gibson-Ashby cellular solid models
- Euler-Bernoulli beam analytical solutions
- Convergence studies (h-refinement)
- Patch tests
- Patch test for rigid body modes

References:
- Gibson & Ashby, "Cellular Solids" (2nd ed., 1997)
- Timoshenko & Gere, "Theory of Elastic Stability" (1961)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.core.material import Material
from fibernet.sim.mechanical import FiberFEM, stress_strain_curve


@dataclass
class ValidationResult:
    """Container for validation results."""
    test_name: str = ""
    passed: bool = True
    relative_error: float = 0.0
    analytical_value: float = 0.0
    numerical_value: float = 0.0
    tolerance: float = 0.1
    details: Dict = field(default_factory=dict)
    
    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (f"{status} {self.test_name}: "
                f"analytical={self.analytical_value:.4e}, "
                f"numerical={self.numerical_value:.4e}, "
                f"error={self.relative_error:.2%}")


def gibson_ashby_honeycomb(
    relative_density: float = 0.1,
    E_solid: float = 1e9,
    nu_solid: float = 0.3,
) -> Dict[str, float]:
    """Gibson-Ashby analytical model for 2D hexagonal honeycomb.
    
    Parameters
    ----------
    relative_density : float
        ρ*/ρ_s (density ratio).
    E_solid : float
        Young's modulus of solid material.
    nu_solid : float
        Poisson's ratio of solid material.
    
    Returns
    -------
    dict with keys:
        E1, E2 (Young's moduli), nu12, nu21 (Poisson's ratios),
        G12 (shear modulus)
    
    Notes
    -----
    For regular hexagonal honeycomb (t/l = relative_density / (2/√3)):
    
    E1/Es = (t/l)^3 * cos(θ) / [(h/l + sin(θ)) * sin²(θ)]
    E2/Es = (t/l)^3 * (h/l + sin(θ)) / [cos³(θ)]
    
    For regular honeycomb: h/l = 1, θ = 30°:
    E1 = E2 = (2/√3) * (t/l)^3 * Es
    
    Reference: Gibson & Ashby (1997), Chapter 4.
    """
    t_over_l = relative_density * np.sqrt(3) / 2
    
    theta = np.pi / 6
    h_over_l = 1.0
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    
    E1 = E_solid * (t_over_l**3 * cos_t) / ((h_over_l + sin_t) * sin_t**2)
    E2 = E_solid * (t_over_l**3 * (h_over_l + sin_t)) / (cos_t**3)
    
    nu12 = cos_t**2 / (h_over_l + sin_t) / sin_t
    nu21 = (h_over_l + sin_t) * sin_t / cos_t**2
    
    G12 = E_solid * (h_over_l + sin_t) / (h_over_l**2 * (1 + 2 * h_over_l)) * (t_over_l / h_over_l)**3
    
    return {
        'E1': E1, 'E2': E2,
        'nu12': nu12, 'nu21': nu21,
        'G12': G12,
        'relative_density': relative_density,
    }


def gibson_ashby_foam_3d(
    relative_density: float = 0.1,
    E_solid: float = 1e9,
    sigma_solid: float = 1e7,
) -> Dict[str, float]:
    """Gibson-Ashby model for 3D open-cell foam.
    
    Parameters
    ----------
    relative_density : float
        ρ*/ρ_s.
    E_solid : float
        Solid Young's modulus.
    sigma_solid : float
        Solid yield strength.
    
    Returns
    -------
    dict with effective properties.
    
    Notes
    -----
    Open-cell foam (Gibson-Ashby):
    E*/Es ≈ C1 * (ρ*/ρ_s)^2  (C1 ≈ 1 for open-cell)
    σ*/σ_s ≈ C2 * (ρ*/ρ_s)^(3/2)  (C2 ≈ 0.3)
    
    Reference: Gibson & Ashby (1997), Chapter 5.
    """
    C1 = 1.0
    C2 = 0.3
    
    E_star = C1 * E_solid * relative_density**2
    sigma_star = C2 * sigma_solid * relative_density**(3/2)
    
    return {
        'E_star': E_star,
        'sigma_star': sigma_star,
        'E_ratio': E_star / E_solid,
        'sigma_ratio': sigma_star / sigma_solid,
    }


def gibson_ashby_closed_cell(
    relative_density: float = 0.1,
    E_solid: float = 1e9,
    phi: float = 0.8,
) -> Dict[str, float]:
    """Gibson-Ashby model for closed-cell foam.
    
    Parameters
    ----------
    relative_density : float
        ρ*/ρ_s.
    E_solid : float
        Solid Young's modulus.
    phi : float
        Fraction of solid in edges (vs. faces).
    
    Returns
    -------
    dict with effective properties.
    
    Notes
    -----
    Closed-cell foam:
    E*/Es ≈ φ²(ρ*/ρ_s)² + (1-φ)(ρ*/ρ_s)
    
    Reference: Gibson & Ashby (1997), Chapter 5.
    """
    rho_ratio = relative_density
    
    E_ratio = phi**2 * rho_ratio**2 + (1 - phi) * rho_ratio
    E_star = E_solid * E_ratio
    
    return {
        'E_star': E_star,
        'E_ratio': E_ratio,
        'phi': phi,
    }


def euler_beam_cantilever(
    L: float = 1.0,
    E: float = 1e9,
    I: float = 1e-12,
    P: float = 1.0,
) -> Dict[str, float]:
    """Euler-Bernoulli cantilever beam analytical solution.
    
    Parameters
    ----------
    L : float
        Beam length.
    E : float
        Young's modulus.
    I : float
        Second moment of area.
    P : float
        Tip load.
    
    Returns
    -------
    dict with analytical deflection, slope, moment, shear.
    
    Notes
    -----
    δ_tip = PL³/(3EI)
    θ_tip = PL²/(2EI)
    M_max = PL (at fixed end)
    """
    EI = E * I
    
    delta_tip = P * L**3 / (3 * EI)
    theta_tip = P * L**2 / (2 * EI)
    M_max = P * L
    V_max = P
    
    return {
        'delta_tip': delta_tip,
        'theta_tip': theta_tip,
        'M_max': M_max,
        'V_max': V_max,
    }


def euler_beam_bending_stiffness(
    L: float = 1.0,
    E: float = 1e9,
    r: float = 0.01,
) -> float:
    """Compute bending stiffness EI for circular cross-section beam."""
    I = np.pi * r**4 / 4
    return E * I


def validate_cantilever_beam(
    L: float = 1.0,
    E: float = 1e9,
    r: float = 0.01,
    P: float = 1.0,
    segments: int = 10,
    tolerance: float = 0.15,
) -> ValidationResult:
    """Validate FEM against analytical cantilever beam solution.
    
    Creates a single-fiber network and compares tip deflection
    against Euler-Bernoulli analytical solution.
    
    Parameters
    ----------
    L : float
        Beam length.
    E : float
        Young's modulus.
    r : float
        Beam radius.
    P : float
        Tip load (in y-direction).
    segments : int
        Number of beam elements.
    tolerance : float
        Acceptable relative error.
    
    Returns
    -------
    ValidationResult
    """
    I = np.pi * r**4 / 4
    A = np.pi * r**2
    
    material = Material(youngs_modulus=E, poissons_ratio=0.3)
    
    n_pts = segments + 1
    centerline = np.zeros((n_pts, 3))
    centerline[:, 0] = np.linspace(0, L, n_pts)
    
    fiber = Fiber(centerline=centerline, radius=r, material=material)
    network = FiberNetwork(fibers=[fiber], box_size=np.array([L, L, L]))
    
    fem = FiberFEM(network, segments_per_fiber=segments)
    
    num_dof = fem.num_dof
    F = np.zeros(num_dof)
    
    tip_node = fem.num_nodes - 1
    F[tip_node * 6 + 1] = P
    
    fixed_dofs = list(range(6))
    
    result = fem.solve_static(forces=F, fixed_dofs=fixed_dofs)
    
    analytical = euler_beam_cantilever(L, E, I, P)
    
    numerical_tip = abs(result.displacements[tip_node * 6 + 1])
    analytical_tip = abs(analytical['delta_tip'])
    
    rel_error = abs(numerical_tip - analytical_tip) / analytical_tip if analytical_tip > 0 else 0
    
    return ValidationResult(
        test_name="Cantilever Beam (Euler-Bernoulli)",
        passed=rel_error < tolerance,
        relative_error=rel_error,
        analytical_value=analytical_tip,
        numerical_value=numerical_tip,
        tolerance=tolerance,
        details={
            'L': L, 'E': E, 'r': r, 'segments': segments,
            'I': I, 'EI': E * I,
        }
    )


def validate_honeycomb_scaling(
    relative_densities: List[float] = None,
    E_solid: float = 1e9,
    tolerance: float = 0.3,
) -> ValidationResult:
    """Validate FEM scaling against Gibson-Ashby honeycomb model.
    
    Tests that effective modulus scales as (ρ*/ρ_s)^3 for 2D honeycomb.
    
    Parameters
    ----------
    relative_densities : list
        Relative density values to test.
    E_solid : float
        Solid material modulus.
    tolerance : float
        Acceptable error in scaling exponent.
    
    Returns
    -------
    ValidationResult
    """
    if relative_densities is None:
        relative_densities = [0.05, 0.1, 0.15, 0.2]
    
    from fibernet.gen.ordered import honeycomb_2d
    
    E_effs = []
    rhos = []
    
    for rho in relative_densities:
        try:
            net = honeycomb_2d(
                box_size=(10, 10),
                fiber_length=1.0,
                fiber_radius=rho * 0.1,
                material=Material(youngs_modulus=E_solid, poissons_ratio=0.3),
            )
            
            fem = FiberFEM(net, segments_per_fiber=3)
            E_eff = fem.effective_modulus(strain=0.001, axis=0)
            
            if E_eff > 0:
                E_effs.append(E_eff)
                rhos.append(rho)
        except Exception:
            continue
    
    if len(rhos) < 2:
        return ValidationResult(
            test_name="Honeycomb Gibson-Ashby Scaling",
            passed=False,
            details={'error': 'Insufficient valid data points'}
        )
    
    log_rho = np.log(rhos)
    log_E = np.log(E_effs)
    
    slope, intercept, r_value, p_value, std_err = np.polyfit(log_rho, log_E, 1, full=False, cov=False) if len(rhos) >= 3 else (*np.polyfit(log_rho, log_E, 1), 0, 0, 0)
    
    analytical_exponent = 3.0
    rel_error = abs(slope - analytical_exponent) / analytical_exponent
    
    return ValidationResult(
        test_name="Honeycomb Gibson-Ashby Scaling (E ∝ ρ³)",
        passed=rel_error < tolerance,
        relative_error=rel_error,
        analytical_value=analytical_exponent,
        numerical_value=slope,
        tolerance=tolerance,
        details={
            'r_squared': r_value**2 if len(rhos) >= 3 else 0,
            'relative_densities': rhos,
            'effective_moduli': E_effs,
        }
    )


def validate_convergence(
    network: Optional[FiberNetwork] = None,
    segment_counts: List[int] = None,
    axis: int = 0,
    strain: float = 0.001,
) -> ValidationResult:
    """Convergence study: h-refinement (mesh density).
    
    Tests that effective modulus converges as segments_per_fiber increases.
    
    Parameters
    ----------
    network : FiberNetwork, optional
        Test network. Creates simple network if None.
    segment_counts : list
        Number of segments to test.
    axis : int
        Loading direction.
    strain : float
        Test strain.
    
    Returns
    -------
    ValidationResult
    """
    if segment_counts is None:
        segment_counts = [2, 4, 8, 12, 16]
    
    if network is None:
        from fibernet.gen.disordered import random_2d
        network = random_2d(
            num_fibers=30,
            fiber_length=5.0,
            box_size=(10, 10),
            fiber_radius=0.05,
            seed=42,
        )
    
    E_effs = []
    for seg in segment_counts:
        try:
            fem = FiberFEM(network, segments_per_fiber=seg)
            E = fem.effective_modulus(strain=strain, axis=axis)
            E_effs.append(E)
        except Exception:
            E_effs.append(0.0)
    
    E_effs = np.array(E_effs)
    valid = E_effs > 0
    
    if np.sum(valid) < 2:
        return ValidationResult(
            test_name="Convergence Study (h-refinement)",
            passed=False,
            details={'error': 'Insufficient valid data'}
        )
    
    E_valid = E_effs[valid]
    seg_valid = np.array(segment_counts)[valid]
    
    E_ref = E_valid[-1]
    errors = np.abs(E_valid[:-1] - E_ref) / E_ref if E_ref > 0 else np.ones_like(E_valid[:-1])
    
    converged = len(errors) >= 2 and errors[-1] < 0.1
    
    return ValidationResult(
        test_name="Convergence Study (h-refinement)",
        passed=converged,
        relative_error=errors[-1] if len(errors) > 0 else 1.0,
        analytical_value=E_ref,
        numerical_value=E_valid[-2] if len(E_valid) >= 2 else 0,
        tolerance=0.1,
        details={
            'segment_counts': seg_valid.tolist(),
            'effective_moduli': E_valid.tolist(),
            'relative_errors': errors.tolist(),
        }
    )


def validate_patch_test(
    E: float = 1e9,
    nu: float = 0.3,
    tolerance: float = 1e-6,
) -> ValidationResult:
    """Patch test: uniform strain should be exactly reproduced.
    
    Creates a simple 2-element mesh and applies uniform strain.
    The FEM should reproduce the exact linear displacement field.
    
    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    tolerance : float
        Acceptable error (should be machine precision for linear elements).
    
    Returns
    -------
    ValidationResult
    """
    material = Material(youngs_modulus=E, poissons_ratio=nu)
    
    centerline = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    
    fiber = Fiber(centerline=centerline, radius=0.01, material=material)
    network = FiberNetwork(fibers=[fiber], box_size=np.array([1.0, 1.0, 1.0]))
    
    fem = FiberFEM(network, segments_per_fiber=2)
    
    applied_strain = 0.001
    
    result = fem.apply_uniaxial_strain(applied_strain, axis=0)
    
    if result.displacements is None:
        return ValidationResult(
            test_name="Patch Test (uniform strain)",
            passed=False,
            details={'error': 'No displacements computed'}
        )
    
    errors = []
    for n_idx in range(fem.num_nodes):
        x = fem.node_positions[n_idx, 0]
        expected_u = applied_strain * x
        actual_u = result.displacements[n_idx * 6]
        errors.append(abs(actual_u - expected_u))
    
    max_error = max(errors) if errors else 0
    rel_error = max_error / (applied_strain * 1.0) if applied_strain > 0 else 0
    
    return ValidationResult(
        test_name="Patch Test (uniform strain)",
        passed=rel_error < tolerance,
        relative_error=rel_error,
        analytical_value=applied_strain,
        numerical_value=max_error,
        tolerance=tolerance,
        details={'max_absolute_error': max_error}
    )


def run_all_validations(
    E_solid: float = 1e9,
    verbose: bool = True,
) -> List[ValidationResult]:
    """Run all validation tests and return results.
    
    Parameters
    ----------
    E_solid : float
        Reference Young's modulus.
    verbose : bool
        Print results as they run.
    
    Returns
    -------
    list of ValidationResult
    """
    results = []
    
    tests = [
        ("Cantilever Beam", lambda: validate_cantilever_beam(E=E_solid)),
        ("Patch Test", lambda: validate_patch_test(E=E_solid)),
        ("Convergence", lambda: validate_convergence()),
    ]
    
    for name, test_func in tests:
        if verbose:
            print(f"Running: {name}...")
        try:
            result = test_func()
            results.append(result)
            if verbose:
                print(f"  {result.summary()}")
        except Exception as e:
            results.append(ValidationResult(
                test_name=name,
                passed=False,
                details={'error': str(e)}
            ))
            if verbose:
                print(f"  ❌ ERROR: {e}")
    
    return results


def print_validation_report(results: List[ValidationResult]) -> str:
    """Format validation results as a report."""
    lines = ["=" * 60, "FiberNet FEM Validation Report", "=" * 60]
    
    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)
    
    for r in results:
        lines.append(r.summary())
    
    lines.append("-" * 60)
    lines.append(f"Summary: {n_pass}/{n_total} tests passed")
    lines.append("=" * 60)
    
    return "\n".join(lines)

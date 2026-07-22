"""
Example 6: Uncertainty Quantification via Monte Carlo Simulation

Demonstrates:
- Monte Carlo ensemble analysis
- Statistical property estimation
- Confidence intervals
- Sensitivity analysis

Usage:
    python examples/06_uncertainty_quantification.py
"""

import numpy as np
from fibernet import gen
from fibernet.sim import FiberFEM
from fibernet.sim.uncertainty import (
    monte_carlo_ensemble, sensitivity_analysis, convergence_study
)
from fibernet.analysis import MorphologyAnalyzer

print("=" * 60)
print("Example 6: Uncertainty Quantification")
print("=" * 60)

# Define property function
def compute_modulus(net):
    """Compute effective modulus."""
    fem = FiberFEM(net)
    return fem.effective_modulus()

def compute_order(net):
    """Compute nematic order parameter."""
    morph = MorphologyAnalyzer(net)
    return morph.nematic_order_parameter()

# Monte Carlo ensemble
print("\n[1/3] Monte Carlo ensemble (n=30)...")
result = monte_carlo_ensemble(
    gen.random_straight_2d,
    compute_modulus,
    num_samples=30,
    generator_kwargs={'num_fibers': 40, 'fiber_length': 10.0},
)

print(f"  Modulus: {result.mean:.2e} ± {result.std:.2e} Pa")
print(f"  CV: {result.cv:.3f}")
print(f"  95% CI: [{result.confidence_interval[0]:.2e}, {result.confidence_interval[1]:.2e}]")
print(f"  Converged: {result.converged}")

# Order parameter statistics
print("\n[2/3] Order parameter statistics (n=20)...")
result_order = monte_carlo_ensemble(
    gen.random_straight_2d,
    compute_order,
    num_samples=20,
    generator_kwargs={'num_fibers': 40},
)

print(f"  Order: {result_order.mean:.3f} ± {result_order.std:.3f}")

# Sensitivity analysis
print("\n[3/3] Sensitivity to fiber count...")
sa_result = sensitivity_analysis(
    gen.random_straight_2d,
    compute_modulus,
    'num_fibers',
    [20, 40, 60, 80],
    num_samples_per_value=5,
    generator_kwargs={'fiber_length': 10.0},
)

for n, E, std in zip(sa_result['values'], sa_result['means'], sa_result['stds']):
    print(f"  N={n:3d}: E = {E:.2e} ± {std:.2e} Pa")

print("\n" + "=" * 60)
print("UQ analysis complete!")

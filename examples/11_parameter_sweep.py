"""
Example 11: Parameter Sweep and Sensitivity Analysis

Demonstrates:
- Grid search (full factorial design)
- Latin hypercube sampling
- Random parameter sampling
- Sensitivity analysis
- Result visualization and export

Usage:
    python examples/11_parameter_sweep.py
"""

import numpy as np
from fibernet import gen
from fibernet.sim import FiberFEM
from fibernet.doe import DesignOfExperiments, run_parameter_sweep

print("=" * 60)
print("Example 11: Parameter Sweep and Sensitivity Analysis")
print("=" * 60)

# Define output function
def compute_properties(net):
    """Compute mechanical properties from network."""
    fem = FiberFEM(net)
    E = fem.effective_modulus()
    return {
        'modulus': E,
        'total_length': net.total_length,
        'num_crosslinks': net.num_crosslinks,
    }

# Test 1: Grid search (full factorial)
print("\n[1/4] Grid search (full factorial)...")
params_grid = {
    'num_fibers': [30, 50, 70],
    'fiber_length': [5.0, 10.0, 15.0],
}

result_grid = run_parameter_sweep(
    gen.random_straight_2d,
    params_grid,
    output_func=compute_properties,
    method='grid',
    seed=42,
)

print(f"  Ran {len(result_grid.results)} experiments")
print("  Results:")
print("  num_fibers | length | modulus (Pa) | total_length")
print("  " + "-" * 55)
for r in result_grid.results[:6]:  # Show first 6
    n = r.parameters['num_fibers']
    L = r.parameters['fiber_length']
    E = r.outputs['modulus']
    TL = r.outputs['total_length']
    print(f"  {n:>10d} | {L:>6.1f} | {E:>12.2e} | {TL:>11.1f}")

# Test 2: Latin hypercube sampling
print("\n[2/4] Latin hypercube sampling (15 samples)...")
params_lhs = {
    'num_fibers': (30.0, 100.0),
    'fiber_length': (5.0, 20.0),
}

result_lhs = run_parameter_sweep(
    gen.random_straight_2d,
    params_lhs,
    output_func=compute_properties,
    method='lhs',
    num_samples=15,
    seed=42,
)

print(f"  Ran {len(result_lhs.results)} experiments")
print(f"  Modulus range: {min(r.outputs['modulus'] for r in result_lhs.results):.2e} - "
      f"{max(r.outputs['modulus'] for r in result_lhs.results):.2e} Pa")

# Test 3: Random sampling
print("\n[3/4] Random sampling (10 samples)...")
result_random = run_parameter_sweep(
    gen.random_straight_2d,
    params_lhs,
    output_func=compute_properties,
    method='random',
    num_samples=10,
    seed=42,
)

print(f"  Ran {len(result_random.results)} experiments")

# Test 4: Sensitivity analysis
print("\n[4/4] Sensitivity analysis...")
doe = DesignOfExperiments(gen.random_straight_2d, {'seed': 42}, compute_properties)
result_full = doe.grid_search(params_grid)

sensitivity_modulus = doe.sensitivity_analysis(result_full, 'modulus')
sensitivity_length = doe.sensitivity_analysis(result_full, 'total_length')

print("  Parameter sensitivities (correlation with outputs):")
print("  Parameter     | Modulus | Total Length")
print("  " + "-" * 45)
for param in ['num_fibers', 'fiber_length']:
    s_mod = sensitivity_modulus.get(param, 0)
    s_len = sensitivity_length.get(param, 0)
    print(f"  {param:14s} | {s_mod:>7.3f} | {s_len:>12.3f}")

# Export to DataFrame (if pandas available)
print("\n[BONUS] Exporting results to DataFrame...")
try:
    import pandas as pd
    df = result_grid.to_dataframe()
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print("\n  Summary statistics:")
    print(df.describe().to_string())
except ImportError:
    print("  Pandas not available, skipping DataFrame export")

print("\n" + "=" * 60)
print("Parameter sweep complete!")
print("\nKey insights:")
print("  - Grid search explores all combinations (full factorial)")
print("  - Latin hypercube provides better coverage with fewer samples")
print("  - Sensitivity analysis identifies most influential parameters")
print("  - Results can be exported for further analysis")

"""
Example 3: Thermal Analysis - CTE and Heat Transfer

Demonstrates:
- Computing coefficient of thermal expansion (CTE)
- Analyzing thermal deformation
- Comparing CTE for different network types

Usage:
    python examples/03_thermal_analysis.py
"""

import numpy as np
from fibernet import gen
from fibernet.sim.thermal import ThermalAnalyzer

print("=" * 60)
print("Example 3: Thermal Analysis")
print("=" * 60)

# Generate network
print("\n[1/3] Generating fiber network...")
net = gen.random_straight_2d(
    num_fibers=50,
    fiber_length=10.0,
    box_size=(50.0, 50.0),
    seed=42,
)
print(f"  Network: {net.num_fibers} fibers")

# Compute CTE
print("\n[2/3] Computing CTE tensor...")
thermal = ThermalAnalyzer(net)
cte = thermal.compute_cte(fiber_cte=1.2e-5)

print(f"  CTE tensor (1/K):")
for i in range(3):
    row = "    " + "  ".join([f"{cte[i,j]:.2e}" for j in range(3)])
    print(row)

# Principal CTEs
eigenvalues = np.linalg.eigvalsh(cte)
print(f"\n  Principal CTEs:")
for i, val in enumerate(sorted(eigenvalues)):
    print(f"    α_{i+1} = {val:.2e} 1/K")

# Thermal deformation
print("\n[3/3] Simulating thermal deformation...")
delta_T = 100.0  # 100 K temperature increase
deformed = thermal.apply_temperature_change(net, delta_T=delta_T)

print(f"  Temperature change: {delta_T} K")
print(f"  Deformed network: {deformed.num_fibers} fibers")

# Compute average deformation
original_centers = np.array([f.centerline for f in net.fibers])
deformed_centers = np.array([f.centerline for f in deformed.fibers])

# Compare lengths
orig_lengths = [f.length for f in net.fibers]
def_lengths = [f.length for f in deformed.fibers]
avg_strain = np.mean([(d - o) / o for o, d in zip(orig_lengths, def_lengths)])

print(f"  Average thermal strain: {avg_strain:.2e}")
print(f"  Expected (isotropic): {np.mean(eigenvalues) * delta_T:.2e}")

print("\n" + "=" * 60)
print("Thermal analysis complete!")

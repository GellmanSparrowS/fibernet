"""
Example 7: Electromagnetic Analysis of Fiber Networks

Demonstrates:
- Effective conductivity computation
- Percolation threshold estimation
- Anisotropic conductivity
- Comparison of different network types

Usage:
    python examples/07_electromagnetic_analysis.py
"""

import numpy as np
from fibernet import gen
from fibernet.core.material import Material
from fibernet.sim.electromagnetic import EMSolver

print("=" * 60)
print("Example 7: Electromagnetic Analysis")
print("=" * 60)

# Create conductive fiber material (e.g., carbon fiber)
fiber_material = Material(
    name='carbon_fiber',
    youngs_modulus=230e9,
    poissons_ratio=0.3,
    density=1750.0,
    electrical_conductivity=1e5,  # S/m (typical for carbon fiber)
    permittivity=10.0,
)

# Generate different networks
print("\n[1/4] Generating networks with conductive fibers...")

networks = {
    'Sparse random': gen.random_straight_2d(num_fibers=20, seed=42, material=fiber_material),
    'Dense random': gen.random_straight_2d(num_fibers=80, seed=42, material=fiber_material),
    'Square lattice': gen.square_lattice_2d(spacing=2.0, grid_size=(15, 15), material=fiber_material),
}

for name, net in networks.items():
    print(f"  {name:15s}: {net.num_fibers} fibers, {net.num_crosslinks} crosslinks")

# Conductivity analysis
print("\n[2/4] Computing effective conductivity (x-direction)...")

for name, net in networks.items():
    solver = EMSolver(net)
    result = solver.solve_conductivity(voltage=1.0, axis=0)
    print(f"  {name:15s}: σ_eff = {result.effective_conductivity:.2e} S/m, "
          f"Percolating = {result.is_percolating}")

# Anisotropy analysis
print("\n[3/4] Conductivity anisotropy (dense random)...")
net_dense = networks['Dense random']
solver = EMSolver(net_dense)

for axis, label in enumerate(['x', 'y', 'z']):
    result = solver.solve_conductivity(voltage=1.0, axis=axis)
    print(f"  σ_{label} = {result.effective_conductivity:.2e} S/m")

# Fiber density sweep for percolation study
print("\n[4/4] Percolation study (varying fiber density)...")
densities = [10, 20, 40, 60, 80, 100]

for n_fibers in densities:
    net = gen.random_straight_2d(num_fibers=n_fibers, seed=42, material=fiber_material)
    solver = EMSolver(net)
    result = solver.solve_conductivity(axis=0)
    print(f"  N={n_fibers:3d}: σ = {result.effective_conductivity:.2e} S/m, "
          f"Percolating = {result.is_percolating}")

print("\n" + "=" * 60)
print("Electromagnetic analysis complete!")
print("\nNote: Percolation transition occurs when the network becomes")
print("connected enough to support current flow across the sample.")

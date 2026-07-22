"""
Example 1: Mechanical Analysis of Random Fiber Network

Demonstrates:
- Generating a random 2D fiber network
- FEM uniaxial strain testing
- Effective modulus computation
- Stress-strain analysis

Usage:
    python examples/01_mechanical_analysis.py
"""

import numpy as np
from fibernet import gen
from fibernet.sim import FiberFEM
from fibernet.analysis import MorphologyAnalyzer, StressStrainCurve

np.random.seed(42)

print("=" * 60)
print("Example 1: Mechanical Analysis of Random Fiber Network")
print("=" * 60)

# Step 1: Generate network
print("\n[1/4] Generating random 2D fiber network...")
net = gen.random_straight_2d(
    num_fibers=50,
    fiber_length=10.0,
    box_size=(50.0, 50.0),
    seed=42,
)
print(f"  Fibers: {net.num_fibers}")
print(f"  Crosslinks: {net.num_crosslinks}")
print(f"  Total length: {net.total_length:.1f}")

# Step 2: FEM analysis
print("\n[2/4] Running FEM analysis...")
fem = FiberFEM(net)

# Apply uniaxial strain in x-direction
result_x = fem.apply_uniaxial_strain(strain=0.01, axis=0)
print(f"  Strain (x): 0.01")
print(f"  Max stress: {result_x.stresses.max():.2e} Pa")
print(f"  Mean stress: {result_x.stresses.mean():.2e} Pa")

# Step 3: Effective modulus
print("\n[3/4] Computing effective modulus...")
E = fem.effective_modulus()
print(f"  Young's modulus: {E:.2e} Pa")

# Step 4: Morphology
print("\n[4/4] Computing morphology metrics...")
morph = MorphologyAnalyzer(net)
order = morph.nematic_order_parameter()
print(f"  Nematic order parameter: {order:.3f}")

print("\n" + "=" * 60)
print("Analysis complete!")

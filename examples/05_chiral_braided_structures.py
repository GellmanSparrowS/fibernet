"""
Example 5: Advanced Structures - Chiral and Braided Networks

Demonstrates:
- Generating chiral metamaterials
- Creating braided fiber ropes
- Analyzing their unique mechanical properties

Usage:
    python examples/05_chiral_braided_structures.py
"""

import numpy as np
from fibernet import gen
from fibernet.sim import FiberFEM
from fibernet.analysis import MorphologyAnalyzer

print("=" * 60)
print("Example 5: Advanced Structures")
print("=" * 60)

# Chiral metamaterial
print("\n[1/3] Generating chiral metamaterial...")
net_chiral = gen.chiral_metamaterial(
    unit_cell_size=10.0,
    grid_size=(3, 3, 1),
    helix_radius=2.0,
    fiber_radius=0.2,
    turns_per_cell=1.0,
)
print(f"  Chiral network: {net_chiral.num_fibers} fibers")
print(f"  Crosslinks: {net_chiral.num_crosslinks}")

morph = MorphologyAnalyzer(net_chiral)
order = morph.nematic_order_parameter()
print(f"  Nematic order: {order:.3f}")

# Braided rope
print("\n[2/3] Generating braided rope structure...")
net_braided = gen.braided_rope(
    num_strands=6,
    rope_radius=5.0,
    pitch=20.0,
    num_turns=3.0,
    fiber_radius=0.3,
)
print(f"  Braided rope: {net_braided.num_fibers} fibers")
print(f"  Crosslinks: {net_braided.num_crosslinks}")

# Compare mechanical properties
print("\n[3/3] Comparing mechanical properties...")

structures = {
    'Chiral': net_chiral,
    'Braided': net_braided,
}

for name, net in structures.items():
    fem = FiberFEM(net)
    E = fem.effective_modulus()
    print(f"  {name:10s}: E = {E:.2e} Pa")

print("\n" + "=" * 60)
print("Structure generation complete!")
print("\nNote: These structures exhibit unique properties:")
print("  - Chiral: Negative Poisson's ratio, rotational coupling")
print("  - Braided: High torsional stiffness, energy absorption")

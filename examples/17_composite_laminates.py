"""
Example 17: Composite Laminate Structures

Demonstrates generation and analysis of multi-layered composite laminates:
- Unidirectional (UD) laminates
- Cross-ply [0/90] laminates
- Angle-ply [±θ] laminates
- Quasi-isotropic [0/±45/90] laminates
- Custom stacking sequences
- Sandwich structures

These are fundamental building blocks for aerospace, automotive, and marine composites.

Usage:
    python examples/17_composite_laminates.py
"""

import numpy as np
from fibernet.gen.laminates import (
    unidirectional_laminate,
    crossply_laminate,
    angle_ply_laminate,
    quasi_isotropic_laminate,
    custom_laminate,
    sandwich_laminate,
)
from fibernet.core.material import Material

print("=" * 70)
print("  FiberNet - Composite Laminate Examples")
print("=" * 70)

# Define carbon fiber material
carbon_fiber = Material(
    name="carbon_fiber",
    youngs_modulus=230e9,  # Pa
    density=1800.0,  # kg/m³
    poissons_ratio=0.3,
)

# 1. Unidirectional Laminate
print("\n[1/6] Unidirectional (UD) Laminate")
print("-" * 70)

ud_laminate = unidirectional_laminate(
    num_layers=4,
    fibers_per_layer=20,
    layer_thickness=0.125e-3,  # 0.125 mm (typical ply thickness)
    fiber_length=0.1,  # 100 mm
    fiber_radius=3.5e-6,  # 7 μm diameter carbon fiber
    orientation=0.0,  # All fibers at 0°
    material=carbon_fiber,
    seed=42
)

print(f"  Fibers: {len(ud_laminate.fibers)}")
print(f"  Layers: 4")
print(f"  Orientation: [0/0/0/0]")
print(f"  Application: Maximum stiffness in one direction")

# 2. Cross-ply Laminate
print("\n[2/6] Cross-ply Laminate [0/90/0/90]")
print("-" * 70)

crossply = crossply_laminate(
    num_layers=4,
    fibers_per_layer=20,
    layer_thickness=0.125e-3,
    fiber_length=0.1,
    fiber_radius=3.5e-6,
    material=carbon_fiber,
    seed=42
)

print(f"  Fibers: {len(crossply.fibers)}")
print(f"  Layers: 4")
print(f"  Orientation: [0/90/0/90]")
print(f"  Application: Bidirectional reinforcement")

# 3. Angle-ply Laminate
print("\n[3/6] Angle-ply Laminate [±45]")
print("-" * 70)

angle_ply = angle_ply_laminate(
    num_layers=4,
    angle=np.pi / 4,  # 45°
    fibers_per_layer=20,
    layer_thickness=0.125e-3,
    fiber_length=0.1,
    fiber_radius=3.5e-6,
    material=carbon_fiber,
    seed=42
)

print(f"  Fibers: {len(angle_ply.fibers)}")
print(f"  Layers: 4")
print(f"  Orientation: [+45/-45/+45/-45]")
print(f"  Application: Shear-resistant structures")

# 4. Quasi-isotropic Laminate
print("\n[4/6] Quasi-isotropic Laminate [0/±45/90]")
print("-" * 70)

quasi_iso = quasi_isotropic_laminate(
    num_fibers_per_layer=20,
    layer_thickness=0.125e-3,
    fiber_length=0.1,
    fiber_radius=3.5e-6,
    material=carbon_fiber,
    seed=42
)

print(f"  Fibers: {len(quasi_iso.fibers)}")
print(f"  Layers: 4")
print(f"  Orientation: [0/+45/-45/90]")
print(f"  Application: Approximately isotropic in-plane properties")

# 5. Custom Laminate
print("\n[5/6] Custom Laminate [0/30/60/90]")
print("-" * 70)

custom_angles = [0.0, np.pi / 6, np.pi / 3, np.pi / 2]  # [0/30/60/90]

custom = custom_laminate(
    stacking_sequence=custom_angles,
    fibers_per_layer=20,
    layer_thickness=0.125e-3,
    fiber_length=0.1,
    fiber_radius=3.5e-6,
    material=carbon_fiber,
    seed=42
)

print(f"  Fibers: {len(custom.fibers)}")
print(f"  Layers: {len(custom_angles)}")
print(f"  Orientation: [0/30/60/90]")
print(f"  Application: Custom stiffness distribution")

# 6. Sandwich Structure
print("\n[6/6] Sandwich Structure")
print("-" * 70)

sandwich = sandwich_laminate(
    face_fibers_per_layer=20,
    num_face_layers=2,
    core_thickness=5e-3,  # 5 mm core
    face_thickness=0.125e-3,  # 0.125 mm per face ply
    fiber_length=0.1,
    fiber_radius=3.5e-6,
    face_material=carbon_fiber,
    seed=42
)

print(f"  Fibers: {len(sandwich.fibers)}")
print(f"  Face sheets: 2 × 2 layers")
print(f"  Core thickness: 5 mm")
print(f"  Application: High bending stiffness, low weight (aerospace)")

# Comparison
print("\n" + "=" * 70)
print("  Laminate Comparison")
print("=" * 70)

laminates = {
    'UD': ud_laminate,
    'Cross-ply': crossply,
    'Angle-ply': angle_ply,
    'Quasi-iso': quasi_iso,
    'Custom': custom,
    'Sandwich': sandwich,
}

print(f"\n  {'Type':<15s} | {'Fibers':>8s} | {'Layers':>7s} | {'Application'}")
print("  " + "-" * 65)

for name, net in laminates.items():
    num_layers = len(net.fibers) // 20  # Assuming 20 fibers per layer
    print(f"  {name:<15s} | {len(net.fibers):>8d} | {num_layers:>7d} | See above")

print("\n" + "=" * 70)
print("  Composite Laminate Examples Complete!")
print("=" * 70)
print("""
Next steps:
  - Analyze mechanical properties using FEM
  - Compute effective moduli (E1, E2, G12, ν12)
  - Visualize with PyVista
  - Export to FEA software (Abaqus, ANSYS)

References:
  - Jones, R.M. "Mechanics of Composite Materials", CRC Press, 2019
  - Kaw, A.K. "Mechanics of Composite Materials", CRC Press, 2006
""")

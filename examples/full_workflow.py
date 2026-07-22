"""
Full FiberNet Workflow Example
===============================

This example demonstrates the complete pipeline:
1. Generate a fiber network
2. Analyze its structure
3. Simulate mechanical properties
4. Export to multiple formats
5. Visualize the results

Run: python examples/full_workflow.py
"""

import numpy as np
import fibernet as fn
from fibernet import gen, sim, analysis
import tempfile
import os

print("="*70)
print("FiberNet Full Workflow Example")
print("="*70)

# 1. Generate a fiber network
print("\n1. Generating fiber network...")
print("-" * 70)

net = gen.random_straight_2d(
    num_fibers=150,
    fiber_length=12.0,
    box_size=(50, 50),
    seed=42
)

print(f"✓ Generated network with {net.num_fibers} fibers")
print(f"  - Fiber length: {net.fibers[0].length:.1f}")
print(f"  - Box size: {net.box_size}")
print(f"  - Crosslinks: {net.num_crosslinks}")

# 2. Analyze the structure
print("\n2. Analyzing network structure...")
print("-" * 70)

# Morphology analysis
morph = analysis.MorphologyAnalyzer(net)
nematic = morph.nematic_order_parameter()
porosity = morph.porosity()
tortuosity = morph.tortuosity_distribution()

print(f"✓ Morphology analysis:")
print(f"  - Nematic order parameter: {nematic:.3f}")
print(f"  - Porosity: {porosity:.3f}")
print(f"  - Mean tortuosity: {np.mean(tortuosity):.3f}")

# Length distribution
lengths = [f.length for f in net.fibers]
print(f"  - Mean fiber length: {np.mean(lengths):.2f}")
print(f"  - Std fiber length: {np.std(lengths):.2f}")

# Topology analysis (if networkx available)
try:
    topo = analysis.TopologyAnalyzer(net)
    num_components = topo.num_connected_components()
    largest = topo.largest_component_fraction()
    is_conn = topo.is_connected()
    print(f"✓ Topology analysis:")
    print(f"  - Connected: {is_conn}")
    print(f"  - Connected components: {num_components}")
    print(f"  - Largest component fraction: {largest:.3f}")
except ImportError:
    print("  (Skipping topology analysis - networkx not installed)")

# 3. Simulate mechanical properties
print("\n3. Running mechanical simulation...")
print("-" * 70)

fem = sim.FiberFEM(net, segments_per_fiber=5)

# Compute effective modulus
print("  Computing effective modulus...")
E_eff = fem.effective_modulus(strain=0.001)
print(f"  ✓ Effective Young's modulus: {E_eff:.2e} Pa")

# Apply uniaxial strain
print("  Applying uniaxial strain (ε = 0.01)...")
result = fem.apply_uniaxial_strain(strain=0.01, axis=0)
print(f"  ✓ Simulation complete:")
print(f"    - Energy: {result.energy:.2e} J")
print(f"    - Max stress: {result.max_stress():.2e} Pa")
print(f"    - Max displacement: {result.max_displacement():.3f}")

# Thermal simulation
print("\n4. Running thermal simulation...")
print("-" * 70)

thermal = fn.simulate_thermal(net, T_hot=100.0, T_cold=0.0)
print(f"✓ Thermal simulation complete:")
print(f"  - Effective conductivity: {thermal['conductivity']:.2e} W/(m·K)")

# 5. Export to multiple formats
print("\n5. Exporting to multiple formats...")
print("-" * 70)

with tempfile.TemporaryDirectory() as tmpdir:
    # JSON export
    json_path = os.path.join(tmpdir, 'network.json')
    fn.export(net, json_path, format='json')
    json_size = os.path.getsize(json_path) / 1024
    print(f"✓ JSON: {json_size:.1f} KB")
    
    # LAMMPS export
    lammps_path = os.path.join(tmpdir, 'network.lammps')
    fn.export(net, lammps_path, format='lammps')
    lammps_size = os.path.getsize(lammps_path) / 1024
    print(f"✓ LAMMPS: {lammps_size:.1f} KB")
    
    # VTK export (if pyvista available)
    try:
        vtk_path = os.path.join(tmpdir, 'network.vtk')
        fn.export(net, vtk_path, format='vtk')
        vtk_size = os.path.getsize(vtk_path) / 1024
        print(f"✓ VTK: {vtk_size:.1f} KB")
    except Exception as e:
        print(f"  (VTK export failed: {e})")
    
    # Reload and verify
    net_loaded = fn.load(json_path, format='json')
    print(f"✓ Verified: reloaded {net_loaded.num_fibers} fibers from JSON")

# 6. Create comparison networks
print("\n6. Comparing different network types...")
print("-" * 70)

networks = {
    'Random 2D': net,
    'Square lattice': gen.square_lattice_2d(spacing=5.0, grid_size=(5, 5)),
    'Honeycomb': gen.honeycomb_lattice_2d(cell_size=5.0, grid_size=(4, 4)),
}

print("Network comparison:")
for name, n in networks.items():
    morph = analysis.MorphologyAnalyzer(n)
    order = morph.nematic_order_parameter()
    print(f"  {name:20s}: {n.num_fibers:3d} fibers, order={order:.3f}")

# 7. Network transformations
print("\n7. Demonstrating network transformations...")
print("-" * 70)

# Scale
net_scaled = fn.scale(net, factor=2.0)
print(f"✓ Scaled by 2.0: {net_scaled.num_fibers} fibers")

# Rotate
net_rotated = fn.rotate(net, angle=np.pi/4, axis=[0, 0, 1])
print(f"✓ Rotated by 45°: {net_rotated.num_fibers} fibers")

# Merge
net_merged = fn.merge([net, net_scaled])
print(f"✓ Merged two networks: {net_merged.num_fibers} fibers")

# Summary
print("\n" + "="*70)
print("Workflow Summary")
print("="*70)
print(f"✓ Generated: {net.num_fibers} fibers with {net.num_crosslinks} crosslinks")
print(f"✓ Analyzed: morphology, topology, structure")
print(f"✓ Simulated: mechanical (E={E_eff:.2e} Pa), thermal")
print(f"✓ Exported: JSON, LAMMPS, VTK")
print(f"✓ Transformed: scale, rotate, merge")
print("\nNext steps:")
print("  - Try different generator types (3D, chiral, woven)")
print("  - Explore damage mechanics and fatigue simulation")
print("  - Use ML features to predict properties")
print("  - Visualize with matplotlib or pyvista")
print("\n🎉 Full workflow complete!")

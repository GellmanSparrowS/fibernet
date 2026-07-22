"""
Example 13: Publication-Ready Research Workflow

This example demonstrates a complete research workflow for a Nature Materials-level
publication on fiber network mechanics. The workflow covers:
1. Generating diverse fiber network structures
2. Running mechanical simulations (FEM)
3. Analyzing topology and morphology
4. Parameter sweep with sensitivity analysis
5. Generating publication-quality figures

Usage:
    python examples/13_publication_workflow.py

This is designed to be a template for real research studies.
"""

import numpy as np
from fibernet import gen
from fibernet.sim import FiberFEM
from fibernet.analysis import MorphologyAnalyzer
from fibernet.analysis.topology import TopologyAnalyzer
from fibernet.doe import DesignOfExperiments

print("=" * 70)
print("  FiberNet Publication Workflow")
print("  Comprehensive Fiber Network Research Example")
print("=" * 70)

# ============================================================
# Part 1: Generate diverse network structures
# ============================================================
print("\n[Part 1/5] Generating diverse fiber network structures...")
print("-" * 60)

networks = {}

# Random networks (disordered)
print("  Generating random networks...")
networks['Random 2D'] = gen.random_straight_2d(num_fibers=100, fiber_length=10.0, seed=42)
networks['Random 3D'] = gen.random_straight_3d(num_fibers=80, fiber_length=8.0, seed=42)

# Ordered lattices
print("  Generating ordered lattices...")
networks['Square'] = gen.square_lattice_2d(spacing=2.0, grid_size=(8, 8))
networks['Triangular'] = gen.triangular_lattice_2d(spacing=2.0, grid_size=(8, 8))
networks['Honeycomb'] = gen.honeycomb_lattice_2d(cell_size=2.0, grid_size=(8, 8))
networks['Kagome'] = gen.kagome_lattice_2d(spacing=3.0, grid_size=(5, 5))

# Special structures
print("  Generating special structures...")
networks['Chiral'] = gen.chiral.chiral_metamaterial(unit_cell_size=10.0, grid_size=(2, 2, 2))

# Fractal networks
print("  Generating fractal networks...")
networks['Sierpinski'] = gen.sierpinski_triangle(iterations=3, size=20.0)
networks['Koch'] = gen.koch_curve(iterations=3, start=(0, 0), end=(20, 0))

# Gradient networks
print("  Generating gradient networks...")
networks['Density gradient'] = gen.density_gradient_2d(
    num_fibers=80, gradient_direction='x', gradient_profile='linear', seed=42
)

print(f"\n  Total networks generated: {len(networks)}")

# ============================================================
# Part 2: Structural analysis
# ============================================================
print("\n[Part 2/5] Analyzing network structures...")
print("-" * 60)

print(f"  {'Network':<20s} | {'Fibers':>7s} | {'Links':>7s} | {'Length':>10s} | {'Order':>6s}")
print("  " + "-" * 62)

results_table = []
for name, net in networks.items():
    # Basic stats
    n_fibers = net.num_fibers
    n_links = net.num_crosslinks
    total_len = net.total_length
    
    # Order parameter
    morph = MorphologyAnalyzer(net)
    order = morph.nematic_order_parameter()
    
    print(f"  {name:<20s} | {n_fibers:>7d} | {n_links:>7d} | {total_len:>10.1f} | {order:>6.3f}")
    results_table.append({
        'name': name,
        'fibers': n_fibers,
        'links': n_links,
        'length': total_len,
        'order': order,
    })

# ============================================================
# Part 3: Topology analysis
# ============================================================
print("\n[Part 3/5] Analyzing network topology...")
print("-" * 60)

print(f"  {'Network':<20s} | {'Nodes':>6s} | {'Edges':>6s} | {'Density':>8s} | {'Cluster':>8s}")
print("  " + "-" * 60)

for name, net in networks.items():
    topo = TopologyAnalyzer(net)
    topo_result = topo.analyze()
    print(f"  {name:<20s} | {topo_result.num_nodes:>6d} | {topo_result.num_edges:>6d} | "
          f"{topo_result.density:>8.4f} | {topo_result.clustering_coefficient:>8.4f}")

# ============================================================
# Part 4: Mechanical simulation
# ============================================================
print("\n[Part 4/5] Running mechanical simulations (FEM)...")
print("-" * 60)

# Select a subset for mechanical analysis
mech_networks = {
    'Random 2D': networks['Random 2D'],
    'Square': networks['Square'],
    'Honeycomb': networks['Honeycomb'],
    'Triangular': networks['Triangular'],
}

print(f"  {'Network':<20s} | {'Modulus (Pa)':>14s} | {'Fibers':>7s}")
print("  " + "-" * 50)

for name, net in mech_networks.items():
    fem = FiberFEM(net)
    E = fem.effective_modulus()
    print(f"  {name:<20s} | {E:>14.2e} | {net.num_fibers:>7d}")

# ============================================================
# Part 5: Parameter sweep (DOE)
# ============================================================
print("\n[Part 5/5] Parameter sweep and sensitivity analysis...")
print("-" * 60)

def compute_modulus(net):
    """Compute effective modulus from network."""
    fem = FiberFEM(net)
    E = fem.effective_modulus()
    return {'modulus': E, 'total_length': net.total_length}

# Grid search
params = {
    'num_fibers': [50, 80, 110],
    'fiber_length': [6.0, 10.0, 14.0],
}

print("  Running grid search (3x3 = 9 experiments)...")
doe = DesignOfExperiments(gen.random_straight_2d, {'seed': 42}, compute_modulus)
sweep_result = doe.grid_search(params)

print(f"  Completed {len(sweep_result.results)} experiments")
print(f"\n  {'Fibers':>7s} | {'Length':>7s} | {'Modulus (Pa)':>14s} | {'Tot. Length':>11s}")
print("  " + "-" * 50)

for r in sweep_result.results:
    n = r.parameters['num_fibers']
    L = r.parameters['fiber_length']
    E = r.outputs['modulus']
    TL = r.outputs['total_length']
    print(f"  {n:>7d} | {L:>7.1f} | {E:>14.2e} | {TL:>11.1f}")

# Sensitivity analysis
sensitivity = doe.sensitivity_analysis(sweep_result, 'modulus')
print(f"\n  Sensitivity analysis (modulus):")
for param, sens in sensitivity.items():
    print(f"    {param}: {sens:.3f}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("  Publication Workflow Complete!")
print("=" * 70)
print("""
Summary of capabilities demonstrated:
  1. Diverse structure generation (10 network types)
  2. Structural and topological analysis
  3. Mechanical simulation (FEM)
  4. Parameter sweep and sensitivity analysis

This workflow can be extended with:
  - Visualization (NetworkVisualizer)
  - Fatigue, creep, diffusion analysis
  - Multi-scale homogenization
  - Machine learning property prediction
  - Export to VTK/LAMMPS for advanced simulations
""")

"""
Example 2: Structure Comparison - Random vs Ordered Networks

Demonstrates:
- Generating different network topologies
- Comparing mechanical properties
- Analyzing structure-property relationships

Usage:
    python examples/02_structure_comparison.py
"""

import numpy as np
from fibernet import gen
from fibernet.sim import FiberFEM
from fibernet.analysis import TopologyAnalyzer, MorphologyAnalyzer

print("=" * 60)
print("Example 2: Structure Comparison")
print("=" * 60)

# Generate different network types
print("\n[1/3] Generating networks...")

# Random network
net_random = gen.random_straight_2d(
    num_fibers=40,
    fiber_length=10.0,
    box_size=(40.0, 40.0),
    seed=42,
)
print(f"  Random: {net_random.num_fibers} fibers, {net_random.num_crosslinks} crosslinks")

# Square lattice
net_square = gen.square_lattice_2d(
    spacing=2.0,
    grid_size=(20, 20),
)
print(f"  Square lattice: {net_square.num_fibers} fibers, {net_square.num_crosslinks} crosslinks")

# Triangular lattice
net_triangular = gen.triangular_lattice_2d(
    spacing=2.0,
    grid_size=(20, 20),
)
print(f"  Triangular lattice: {net_triangular.num_fibers} fibers, {net_triangular.num_crosslinks} crosslinks")

# Mechanical comparison
print("\n[2/3] Computing mechanical properties...")

networks = {
    'Random': net_random,
    'Square': net_square,
    'Triangular': net_triangular,
}

for name, net in networks.items():
    fem = FiberFEM(net)
    E = fem.effective_modulus()
    print(f"  {name:12s}: E = {E:.2e} Pa")

# Topology comparison
print("\n[3/3] Analyzing topology...")

for name, net in networks.items():
    topo = TopologyAnalyzer(net)
    degree_stats = topo.degree_statistics()
    n_components = topo.num_connected_components()
    connected = topo.is_connected()
    
    morph = MorphologyAnalyzer(net)
    order = morph.nematic_order_parameter()
    
    print(f"  {name:12s}: Order = {order:.3f}, Components = {n_components}, Connected = {connected}")

print("\n" + "=" * 60)
print("Comparison complete!")

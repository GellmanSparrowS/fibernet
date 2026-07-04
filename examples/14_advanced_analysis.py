"""
Example 14: Advanced Structural Analysis

Demonstrates advanced analysis capabilities:
1. Spatial statistics (Ripley's K, pair correlation)
2. Effective property computation (homogenization)
3. Network comparison and clustering
4. Combined analysis workflow

Usage:
    python examples/14_advanced_analysis.py
"""

import numpy as np
from fibernet import gen
from fibernet.analysis.spatial import (
    SpatialStatistics, OrientationAnalysis, LengthAnalysis,
    AnisotropyAnalysis, compute_spatial_statistics
)
from fibernet.analysis.homogenization import compute_effective_properties
from fibernet.analysis.comparison import (
    NetworkComparator, compare_networks, network_similarity
)

print("=" * 70)
print("  FiberNet Advanced Structural Analysis")
print("=" * 70)

# ============================================================
# Part 1: Spatial Statistics
# ============================================================
print("\n[Part 1/4] Spatial Statistics Analysis...")
print("-" * 60)

net = gen.random_straight_2d(num_fibers=100, fiber_length=8.0, seed=42)
print(f"  Generated random 2D network: {net.num_fibers} fibers")

# Compute comprehensive statistics
stats = compute_spatial_statistics(net)
print(f"  Nematic order: {stats['nematic_order']:.3f}")
print(f"  Anisotropy index: {stats['anisotropy_index']:.3f}")
print(f"  Mean connectivity: {stats['mean_connectivity']:.1f}")
print(f"  Length: mean={stats['length']['mean']:.2f}, std={stats['length']['std']:.2f}")

# Ripley's K function
spatial = SpatialStatistics(net)
r = np.linspace(1, 15, 10)
K = spatial.ripley_k(r)
print(f"\n  Ripley's K function:")
for ri, Ki in zip(r[::2], K[::2]):
    print(f"    K({ri:.1f}) = {Ki:.2f}")

# Pair correlation
g = spatial.pair_correlation(r)
print(f"\n  Pair correlation function:")
for ri, gi in zip(r[::2], g[::2]):
    print(f"    g({ri:.1f}) = {gi:.3f}")

# Nearest neighbor distances
nn = spatial.nearest_neighbor_distances()
if len(nn) > 0:
    print(f"\n  Nearest neighbor distances:")
    print(f"    mean={np.mean(nn):.2f}, std={np.std(nn):.2f}")

# ============================================================
# Part 2: Effective Properties (Homogenization)
# ============================================================
print("\n[Part 2/4] Effective Property Computation...")
print("-" * 60)

props = compute_effective_properties(net)

print(f"  Elastic properties:")
if 'E_x' in props['elastic']:
    print(f"    E_x = {props['elastic']['E_x']:.2e} Pa")
    print(f"    E_y = {props['elastic']['E_y']:.2e} Pa")
    print(f"    nu  = {props['elastic']['nu']:.3f}")
    print(f"    G   = {props['elastic']['G']:.2e} Pa")
else:
    print(f"    E = {props['elastic']['E']:.2e} Pa")

print(f"  Thermal properties:")
print(f"    conductivity = {props['thermal']['conductivity']:.4f} W/(m·K)")
print(f"    expansion    = {props['thermal']['expansion']:.2e} 1/K")

print(f"  Electrical properties:")
print(f"    conductivity = {props['electrical']['conductivity']:.2e} S/m")

# ============================================================
# Part 3: Network Comparison
# ============================================================
print("\n[Part 3/4] Network Comparison and Similarity...")
print("-" * 60)

# Generate diverse networks
networks = [
    gen.random_straight_2d(num_fibers=80, fiber_length=8.0, seed=42),
    gen.random_straight_2d(num_fibers=80, fiber_length=12.0, seed=42),
    gen.random_straight_2d(num_fibers=120, fiber_length=8.0, seed=42),
    gen.square_lattice_2d(spacing=2.0, grid_size=(8, 8)),
    gen.triangular_lattice_2d(spacing=2.0, grid_size=(8, 8)),
]

names = ['Random-8', 'Random-12', 'Random-120', 'Square', 'Triangular']

# Pairwise similarity
print(f"  Pairwise similarity matrix:")
print(f"  {'':>15s}", end='')
for name in names:
    print(f" {name:>12s}", end='')
print()

for i, net1 in enumerate(networks):
    print(f"  {names[i]:>15s}", end='')
    for j, net2 in enumerate(networks):
        sim = network_similarity(net1, net2)
        print(f" {sim:>12.3f}", end='')
    print()

# Compare with different metrics
comparator = NetworkComparator(networks)
distances = comparator.pairwise_distances(metric='euclidean')
print(f"\n  Pairwise distance matrix (Euclidean):")
print(f"  {'':>15s}", end='')
for name in names:
    print(f" {name:>12s}", end='')
print()

for i in range(len(networks)):
    print(f"  {names[i]:>15s}", end='')
    for j in range(len(networks)):
        print(f" {distances[i, j]:>12.3f}", end='')
    print()

# Most similar to random-8
similar = comparator.most_similar(query_idx=0, top_k=3)
print(f"\n  Most similar to '{names[0]}':")
for idx, dist in similar:
    print(f"    {names[idx]} (distance: {dist:.3f})")

# ============================================================
# Part 4: Compare network types
# ============================================================
print("\n[Part 4/4] Cross-Type Comparison...")
print("-" * 60)

net_types = {
    'Random': gen.random_straight_2d(num_fibers=80, seed=42),
    'Square': gen.square_lattice_2d(spacing=2.0, grid_size=(8, 8)),
    'Triangular': gen.triangular_lattice_2d(spacing=2.0, grid_size=(8, 8)),
    'Honeycomb': gen.honeycomb_lattice_2d(cell_size=2.0, grid_size=(5, 5)),
}

print(f"  {'Type':<15s} | {'Fibers':>7s} | {'Links':>7s} | {'S':>6s} | {'AI':>6s} | {'E_eff (Pa)':>12s}")
print("  " + "-" * 65)

for name, net in net_types.items():
    stats = compute_spatial_statistics(net)
    props = compute_effective_properties(net)
    E = props['elastic'].get('E_x', props['elastic'].get('E', 0))
    
    print(f"  {name:<15s} | {len(net.fibers):>7d} | {len(net.crosslinks):>7d} | "
          f"{stats['nematic_order']:>6.3f} | {stats['anisotropy_index']:>6.3f} | "
          f"{E:>12.2e}")

# ============================================================
print("\n" + "=" * 70)
print("  Advanced Analysis Complete!")
print("=" * 70)
print("""
Key capabilities demonstrated:
  1. Spatial statistics (Ripley's K, pair correlation, nearest neighbor)
  2. Effective property computation (elastic, thermal, electrical)
  3. Network comparison and similarity search
  4. Cross-type structural comparison

These tools enable quantitative comparison of fiber network structures
for materials design and optimization studies.
""")

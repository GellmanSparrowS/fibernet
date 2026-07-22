"""
Example 12: Network Topology Analysis

Demonstrates graph-theoretic analysis of fiber networks:
- Basic topology metrics (degree, density, clustering)
- Centrality analysis (betweenness, closeness, eigenvector)
- Degree distribution
- Community detection
- Critical node identification

Usage:
    python examples/12_topology_analysis.py
"""

import numpy as np
from fibernet import gen
from fibernet.analysis.topology import TopologyAnalyzer, analyze_topology

print("=" * 60)
print("Example 12: Network Topology Analysis")
print("=" * 60)

# Generate different network types
print("\n[1/6] Generating fiber networks...")
networks = {
    'Random 2D': gen.random_straight_2d(num_fibers=80, seed=42),
    'Square lattice': gen.square_lattice_2d(spacing=2.0, grid_size=(8, 8)),
    'Honeycomb': gen.honeycomb_lattice_2d(cell_size=2.0, grid_size=(8, 8)),
}

print("  Generated 3 network types")

# Basic topology analysis
print("\n[2/6] Computing basic topology metrics...")
print("\n  Network Type      | Nodes | Edges | Density | Avg Degree | Clustering")
print("  " + "-" * 75)

for name, net in networks.items():
    result = analyze_topology(net)
    print(f"  {name:18s} | {result.num_nodes:5d} | {result.num_edges:5d} | "
          f"{result.density:7.4f} | {result.avg_degree:10.2f} | {result.clustering_coefficient:10.4f}")

# Degree distribution
print("\n[3/6] Analyzing degree distribution...")
net = networks['Random 2D']
topo = TopologyAnalyzer(net)
degrees, counts = topo.degree_distribution()

print("  Degree | Count | Percentage")
print("  " + "-" * 35)
total_nodes = sum(counts)
for deg, count in zip(degrees, counts):
    pct = 100 * count / total_nodes
    print(f"  {deg:>6d} | {count:>5d} | {pct:>9.1f}%")

print(f"\n  Degree statistics:")
print(f"    Mean: {topo.analyze().avg_degree:.2f}")
print(f"    Std:  {topo.analyze().degree_std:.2f}")
print(f"    Min:  {topo.analyze().min_degree}")
print(f"    Max:  {topo.analyze().max_degree}")

# Centrality analysis
print("\n[4/6] Computing centrality measures...")
centrality = topo.compute_centrality()

# Find top 5 nodes by betweenness centrality
sorted_betweenness = sorted(
    centrality.betweenness_centrality.items(),
    key=lambda x: x[1],
    reverse=True
)

print("  Top 5 nodes by betweenness centrality:")
print("  Node ID | Betweenness | Degree")
print("  " + "-" * 35)
for node, bet in sorted_betweenness[:5]:
    deg = centrality.degree_centrality[node]
    print(f"  {node:>7d} | {bet:>11.4f} | {deg:>6.4f}")

# Find top 5 nodes by degree centrality
sorted_degree = sorted(
    centrality.degree_centrality.items(),
    key=lambda x: x[1],
    reverse=True
)

print("\n  Top 5 nodes by degree centrality:")
print("  Node ID | Degree | Betweenness")
print("  " + "-" * 35)
for node, deg in sorted_degree[:5]:
    bet = centrality.betweenness_centrality[node]
    print(f"  {node:>7d} | {deg:>6.4f} | {bet:>11.4f}")

# Community detection
print("\n[5/6] Detecting communities...")
try:
    communities = topo.find_communities(method='louvain')
    num_communities = len(set(communities.values()))
    print(f"  Number of communities detected: {num_communities}")
    
    # Count nodes per community
    comm_sizes = {}
    for node, comm_id in communities.items():
        comm_sizes[comm_id] = comm_sizes.get(comm_id, 0) + 1
    
    print("\n  Community sizes:")
    for comm_id, size in sorted(comm_sizes.items())[:10]:  # Show top 10
        print(f"    Community {comm_id}: {size} nodes")
except Exception as e:
    print(f"  Community detection failed: {e}")

# Critical nodes
print("\n[6/6] Identifying critical nodes...")
critical_nodes = topo.get_critical_nodes(metric='betweenness', top_k=10)

print(f"  Top 10 critical nodes (by betweenness):")
print(f"  {critical_nodes}")

# Compare with lattice networks
print("\n[BONUS] Comparing topology across network types...")
print("\n  Network Type      | Components | Diameter | Avg Path | Assortativity")
print("  " + "-" * 75)

for name, net in networks.items():
    result = analyze_topology(net)
    print(f"  {name:18s} | {result.num_components:>10d} | {result.diameter:>8d} | "
          f"{result.avg_path_length:>8.2f} | {result.assortativity:>13.4f}")

print("\n" + "=" * 60)
print("Topology analysis complete!")
print("\nKey insights:")
print("  - Random networks have broader degree distributions")
print("  - Lattice networks have higher clustering coefficients")
print("  - Betweenness centrality identifies bridge nodes")
print("  - Community detection reveals modular structure")
print("  - Critical nodes are potential failure points")

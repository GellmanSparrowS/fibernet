"""
Example 4: Buckling Analysis of Fiber Networks

Demonstrates:
- Euler buckling analysis of individual fibers
- Network-level buckling modes
- Critical load prediction

Usage:
    python examples/04_buckling_analysis.py
"""

import numpy as np
from fibernet import gen
from fibernet.sim import BucklingAnalyzer

print("=" * 60)
print("Example 4: Buckling Analysis")
print("=" * 60)

# Generate network
print("\n[1/4] Generating fiber network...")
net = gen.random_straight_2d(
    num_fibers=40,
    fiber_length=15.0,
    box_size=(50.0, 50.0),
    radius=0.05,
    seed=42,
)
print(f"  Network: {net.num_fibers} fibers, {net.num_crosslinks} crosslinks")

# Buckling analysis
print("\n[2/4] Performing buckling analysis...")
buckling = BucklingAnalyzer(net)

# Euler buckling for individual fibers
print("\n[3/4] Computing Euler buckling loads...")
euler_loads = buckling.euler_buckling_loads(boundary='pinned-pinned')

print(f"  Critical loads (pinned-pinned):")
print(f"    Min: {euler_loads.min():.2e} N")
print(f"    Max: {euler_loads.max():.2e} N")
print(f"    Mean: {euler_loads.mean():.2e} N")

# Network buckling modes
print("\n[4/4] Computing network buckling modes...")
eigenvalues, modes = buckling.compute_buckling_modes(num_modes=5)

print(f"  First 5 buckling eigenvalues:")
for i, eig in enumerate(eigenvalues[:5]):
    print(f"    Mode {i+1}: λ = {eig:.2e}")

print("\n" + "=" * 60)
print("Buckling analysis complete!")

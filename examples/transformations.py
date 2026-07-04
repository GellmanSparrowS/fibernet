"""
Network transformations example - demonstrating powerful network manipulation tools.
"""
import sys
sys.path.insert(0, '/home/codex/projects/codex_test/fibernet')

import numpy as np
from fibernet.core.transform import (
    mirror, rotate, scale, translate, merge, tile,
    trim_to_box, align_by_anchor, create_pattern,
)
from fibernet.gen import square_lattice_2d, random_straight_2d, single_helix

def main():
    print("=" * 60)
    print("Network Transformations Example")
    print("=" * 60)
    
    # 1. Basic transformations
    print("\n1. Basic Transformations")
    net = square_lattice_2d(spacing=5, grid_size=(3, 3))
    print(f"   Original: {net.num_fibers} fibers")
    
    mirrored = mirror(net, axis=0)
    print(f"   Mirrored (x-axis): {mirrored.num_fibers} fibers")
    
    rotated = rotate(net, angle=np.pi/4, axis=np.array([0, 0, 1]))
    print(f"   Rotated (45deg): {rotated.num_fibers} fibers")
    
    scaled = scale(net, factor=2.0)
    print(f"   Scaled (2x): {scaled.num_fibers} fibers")
    
    translated_net = translate(net, offset=np.array([10, 10, 0]))
    print(f"   Translated: {translated_net.num_fibers} fibers")
    
    # 2. Merge networks
    print("\n2. Merging Networks")
    net1 = random_straight_2d(50, 15, (30, 30), seed=42)
    net2 = random_straight_2d(30, 15, (30, 30), seed=43)
    merged = merge([net1, net2], offsets=[np.zeros(3), np.array([35, 0, 0])])
    print(f"   Net1: {net1.num_fibers}, Net2: {net2.num_fibers}")
    print(f"   Merged: {merged.num_fibers} fibers")
    
    # 3. Tiling (periodic structures)
    print("\n3. Tiling (Periodic Structures)")
    unit_cell = square_lattice_2d(spacing=5, grid_size=(2, 2))
    tiled = tile(unit_cell, repeats=(3, 3, 1))
    print(f"   Unit cell: {unit_cell.num_fibers} fibers")
    print(f"   3x3 tiled: {tiled.num_fibers} fibers")
    
    # 4. Trim to bounding box
    print("\n4. Trimming to Bounding Box")
    large_net = random_straight_2d(200, 20, (100, 100), seed=42)
    trimmed = trim_to_box(large_net, box_min=np.array([20, 20, -1]), box_max=np.array([80, 80, 1]))
    print(f"   Original: {large_net.num_fibers} fibers in 100x100 box")
    print(f"   Trimmed: {trimmed.num_fibers} fibers in 60x60 box")
    
    # 5. Pattern generation
    print("\n5. Pattern Generation")
    base_unit = random_straight_2d(10, 8, (10, 10), seed=42)
    
    circular = create_pattern(base_unit, pattern_type="circular", num_units=8, radius=30)
    print(f"   Circular (8 copies): {circular.num_fibers} fibers")
    
    linear = create_pattern(base_unit, pattern_type="linear", num_units=5, spacing=15)
    print(f"   Linear (5 copies): {linear.num_fibers} fibers")
    
    grid = create_pattern(base_unit, pattern_type="grid", num_units=9, spacing=15)
    print(f"   Grid (3x3): {grid.num_fibers} fibers")
    
    spiral = create_pattern(base_unit, pattern_type="spiral", num_units=12, radius=40, turns=2.0)
    print(f"   Spiral (12 units): {spiral.num_fibers} fibers")
    
    # 6. Align by anchor points
    print("\n6. Aligning Networks by Anchor Points")
    net_a = square_lattice_2d(spacing=5, grid_size=(3, 3))
    net_b = square_lattice_2d(spacing=5, grid_size=(3, 3))
    
    net_a, net_b_aligned = align_by_anchor(
        net_a, net_b,
        anchor1=np.array([15, 15, 0]),
        anchor2=np.array([0, 0, 0]),
    )
    combined = merge([net_a, net_b_aligned])
    print(f"   Combined aligned networks: {combined.num_fibers} fibers")
    
    # 7. Complex workflow
    print("\n7. Complex Workflow: Hierarchical Structure")
    unit = random_straight_2d(20, 10, (20, 20), seed=42)
    grid_structure = create_pattern(unit, pattern_type="grid", num_units=9, spacing=25)
    rotated_grid = rotate(grid_structure, angle=np.pi/6, axis=np.array([0, 0, 1]))
    
    helix_top = single_helix(helix_radius=2, pitch=3, num_turns=10)
    helix_top = translate(helix_top, offset=np.array([0, 75, 0]))
    
    hierarchical = merge([rotated_grid, helix_top])
    print(f"   Hierarchical structure: {hierarchical.num_fibers} fibers")
    
    print("\n✓ Transformations example complete")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FiberNet API Demo + 卍 Structure Generation
=============================================
This script demonstrates:
1. The generation paradigm (polyline → unit → transform → tile)
2. Various API usage examples
3. 卍 (swastika/manji) structure generation
4. Multi-level composition

Run: python3 analysis_scripts/api_demo_and_swastika.py

Output: Saved to output_viz/swastika_demo.png
"""

import sys, os, json, time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fibernet as fn
from fibernet.core.structure_graph import StructureGraph

OUT = Path(__file__).parent.parent / 'output_viz'
OUT.mkdir(exist_ok=True)

CHECKPOINT_FILE = OUT / '_demo_checkpoint.json'


def checkpoint(stage, data):
    """Save checkpoint for resume capability."""
    state = {'stage': stage, 'data': data, 'timestamp': time.time()}
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    print(f"  ✅ Checkpoint: {stage}")


def load_checkpoint():
    """Load checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return None


# ======================================================================
# Part 1: Generation Paradigm Demo
# ======================================================================

def demo_paradigm():
    """Demonstrate the 4-level generation paradigm."""
    print("\n" + "="*60)
    print("PART 1: Generation Paradigm Demo")
    print("="*60)
    
    results = {}
    
    # Level 1: Single line segment
    print("\n[Level 1] Single straight line segment")
    line = fn.pattern_2d(
        points=[(0, 5), (10, 5)],
        closed=False,
        box=(10, 10),
        grid=(1, 1),
    )
    print(f"  Line: {line}")
    results['line'] = line
    
    # Level 1b: Line with intermediate nodes (becomes a curve)
    print("\n[Level 1b] Line with intermediate nodes → curve")
    curved = fn.pattern_2d(
        points=[(0, 5), (10, 5)],
        closed=False,
        box=(10, 10),
        grid=(1, 1),
        n_pts_per_side=3,
        perturbation=0.3,
        seed=42,
    )
    print(f"  Curved: {curved}")
    results['curved'] = curved
    
    # Level 2: Unit cell assembly
    print("\n[Level 2] Unit cell (square frame)")
    sq_unit = fn.pattern_2d(unit='square', box=(10, 10), grid=(1, 1))
    print(f"  Square unit: {sq_unit}")
    results['sq_unit'] = sq_unit
    
    # Level 2b: Custom polyline unit
    print("\n[Level 2b] Custom V-shape unit")
    v_unit = fn.pattern_2d(
        points=[(0, 0), (5, 10), (10, 0)],
        closed=False,
        box=(10, 10),
        grid=(1, 1),
    )
    print(f"  V-shape unit: {v_unit}")
    results['v_unit'] = v_unit
    
    # Level 3: Transforms
    print("\n[Level 3] Transform: mirror + rotation")
    transformed = fn.pattern_2d(
        unit='square',
        box=(10, 10),
        grid=(1, 1),
        mirror_x=True,
        mirror_y=True,
        n_pts_per_side=2,
        perturbation=0.1,
        seed=7,
    )
    print(f"  Transformed: {transformed}")
    results['transformed'] = transformed
    
    # Level 4: Tiling + Welding
    print("\n[Level 4] Tile into 4×4 grid")
    tiled = fn.pattern_2d(
        unit='honeycomb',
        box=(10, 10),
        grid=(4, 4),
    )
    print(f"  Tiled: {tiled}")
    results['tiled'] = tiled
    
    checkpoint('paradigm', {k: f"{v}" for k, v in results.items()})
    return results


# ======================================================================
# Part 2: API Examples
# ======================================================================

def demo_api_examples():
    """Demonstrate various API usage patterns."""
    print("\n" + "="*60)
    print("PART 2: API Examples")
    print("="*60)
    
    results = {}
    
    # Example: All built-in units
    print("\n[Example] All built-in unit types")
    units = fn.list_units()
    print(f"  Available units: {units}")
    for u in units:
        try:
            g = fn.pattern_2d(unit=u, box=(10, 10), grid=(2, 2))
            print(f"    {u}: {g}")
        except Exception as e:
            print(f"    {u}: ERROR - {e}")
    
    # Example: Custom points with fit_to_box
    print("\n[Example] Custom points with fit_to_box")
    star_pts = [(50,0),(65,35),(100,50),(65,65),(50,100),(35,65),(0,50),(35,35)]
    star = fn.pattern_2d(
        points=star_pts,
        closed=True,
        box=(10, 10),
        fit_to_box=True,
        grid=(3, 3),
        mirror_x=True,
        mirror_y=True,
    )
    print(f"  Star: {star}")
    results['star'] = star
    
    # Example: Open curve with boundary_mode='extend'
    print("\n[Example] Open curve with boundary extend")
    wave = fn.pattern_2d(
        points=[(i, 5+3*np.sin(i*0.8)) for i in np.linspace(0, 10, 12)],
        closed=False,
        box=(10, 10),
        grid=(3, 3),
        boundary_mode='extend',
    )
    print(f"  Wave: {wave}")
    results['wave'] = wave
    
    # Example: Register custom unit
    print("\n[Example] Register custom X-shape unit")
    def x_unit_factory(box, n_internal=0, radius=0.1, material=None, **kw):
        w, h = box
        g = StructureGraph(dimension=2, box_size=[w, h])
        n00 = g.add_node([0, 0])
        n10 = g.add_node([w, 0])
        n01 = g.add_node([0, h])
        n11 = g.add_node([w, h])
        g.add_edge(n00, n11, radius=radius, material=material, n_internal=n_internal)
        g.add_edge(n10, n01, radius=radius, material=material, n_internal=n_internal)
        g._metadata["unit_type"] = "x_shape"
        return g
    
    fn.register_unit('x_shape', x_unit_factory)
    x_grid = fn.pattern_2d(unit='x_shape', box=(10, 10), grid=(3, 3))
    print(f"  X-grid: {x_grid}")
    results['x_grid'] = x_grid
    
    # Example: 3D
    print("\n[Example] 3D cubic lattice")
    cubic = fn.pattern_3d(unit='cubic', box=(10, 10, 10), grid=(2, 2, 2))
    print(f"  Cubic: {cubic}")
    results['cubic'] = cubic
    
    checkpoint('api_examples', {k: f"{v}" for k, v in results.items()})
    return results


# ======================================================================
# Part 3: 卍 Structure Generation
# ======================================================================

def create_swastika_unit(box, arm_frac=0.5, **kwargs):
    """Create a 卍 (manji/swastika) unit cell.
    
    The structure has 4 arms extending from center, each bending 90°.
    Arms touch cell boundaries for proper welding during tiling.
    
    Parameters
    ----------
    box : (w, h)
        Cell dimensions.
    arm_frac : float
        How far the arm extends (fraction of half-cell).
        1.0 = arm reaches boundary.
    """
    w, h = box
    cx, cy = w / 2, h / 2
    radius = kwargs.get('radius', 0.1)
    material = kwargs.get('material', None)
    n_internal = kwargs.get('n_internal', 0)
    
    g = StructureGraph(dimension=2, box_size=[w, h])
    
    # Center node
    nc = g.add_node([cx, cy])
    
    # 4 arm endpoints on cell boundaries
    # Arm 1: goes UP from center, then bends RIGHT to right boundary
    n1_mid = g.add_node([cx, cy + h * arm_frac / 2])   # middle of vertical part
    n1_bend = g.add_node([cx, h])                        # top boundary
    n1_end = g.add_node([w, h])                          # top-right corner
    
    # Arm 2: goes RIGHT from center, then bends DOWN to bottom boundary
    n2_mid = g.add_node([cx + w * arm_frac / 2, cy])   # middle of horizontal part
    n2_bend = g.add_node([w, cy])                        # right boundary
    n2_end = g.add_node([w, 0])                          # bottom-right corner
    
    # Arm 3: goes DOWN from center, then bends LEFT to left boundary
    n3_mid = g.add_node([cx, cy - h * arm_frac / 2])   # middle of vertical part
    n3_bend = g.add_node([cx, 0])                        # bottom boundary
    n3_end = g.add_node([0, 0])                          # bottom-left corner
    
    # Arm 4: goes LEFT from center, then bends UP to top boundary
    n4_mid = g.add_node([cx - w * arm_frac / 2, cy])   # middle of horizontal part
    n4_bend = g.add_node([0, cy])                        # left boundary
    n4_end = g.add_node([0, h])                          # top-left corner
    
    # Connect arm 1: center → up → bend → end
    g.add_edge(nc, n1_mid, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n1_mid, n1_bend, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n1_bend, n1_end, radius=radius, material=material, n_internal=n_internal)
    
    # Connect arm 2: center → right → bend → end
    g.add_edge(nc, n2_mid, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n2_mid, n2_bend, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n2_bend, n2_end, radius=radius, material=material, n_internal=n_internal)
    
    # Connect arm 3: center → down → bend → end
    g.add_edge(nc, n3_mid, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n3_mid, n3_bend, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n3_bend, n3_end, radius=radius, material=material, n_internal=n_internal)
    
    # Connect arm 4: center → left → bend → end
    g.add_edge(nc, n4_mid, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n4_mid, n4_bend, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n4_bend, n4_end, radius=radius, material=material, n_internal=n_internal)
    
    g._metadata["unit_type"] = "swastika"
    return g


def create_swastika_v2(box, **kwargs):
    """Improved 卍 unit: arms go to boundary midpoints for better welding.
    
    Each arm goes: center → boundary_midpoint → corner
    This ensures boundary contact for welding.
    """
    w, h = box
    cx, cy = w / 2, h / 2
    radius = kwargs.get('radius', 0.1)
    material = kwargs.get('material', None)
    n_internal = kwargs.get('n_internal', 0)
    
    g = StructureGraph(dimension=2, box_size=[w, h])
    
    # Center
    nc = g.add_node([cx, cy])
    
    # Arm 1 (up-right): center → top-mid → top-right
    n_up = g.add_node([cx, h])      # top boundary midpoint
    n_tr = g.add_node([w, h])       # top-right corner
    g.add_edge(nc, n_up, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n_up, n_tr, radius=radius, material=material, n_internal=n_internal)
    
    # Arm 2 (right-down): center → right-mid → bottom-right
    n_right = g.add_node([w, cy])   # right boundary midpoint
    n_br = g.add_node([w, 0])       # bottom-right corner
    g.add_edge(nc, n_right, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n_right, n_br, radius=radius, material=material, n_internal=n_internal)
    
    # Arm 3 (down-left): center → bottom-mid → bottom-left
    n_down = g.add_node([cx, 0])    # bottom boundary midpoint
    n_bl = g.add_node([0, 0])       # bottom-left corner
    g.add_edge(nc, n_down, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n_down, n_bl, radius=radius, material=material, n_internal=n_internal)
    
    # Arm 4 (left-up): center → left-mid → top-left
    n_left = g.add_node([0, cy])    # left boundary midpoint
    n_tl = g.add_node([0, h])       # top-left corner
    g.add_edge(nc, n_left, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n_left, n_tl, radius=radius, material=material, n_internal=n_internal)
    
    g._metadata["unit_type"] = "swastika_v2"
    return g


def demo_swastika():
    """Generate and verify 卍 structures."""
    print("\n" + "="*60)
    print("PART 3: 卍 (Swastika) Structure Generation")
    print("="*60)
    
    results = {}
    
    # Register both variants
    fn.register_unit('swastika', create_swastika_unit)
    fn.register_unit('swastika_v2', create_swastika_v2)
    print("  Registered 'swastika' and 'swastika_v2' units")
    
    # Test single unit
    print("\n[Test] Single swastika unit")
    unit1 = create_swastika_v2((10, 10))
    print(f"  Unit: {unit1}")
    print(f"  Nodes: {len(unit1.nodes)}, Edges: {len(unit1.edges)}")
    
    # Check boundary contacts
    boundary_nodes = []
    for nid, node in unit1.nodes.items():
        p = node.position
        on_boundary = (abs(p[0]) < 0.01 or abs(p[0] - 10) < 0.01 or 
                       abs(p[1]) < 0.01 or abs(p[1] - 10) < 0.01)
        if on_boundary:
            boundary_nodes.append((nid, p[:2]))
    print(f"  Boundary nodes: {len(boundary_nodes)}")
    for nid, pos in boundary_nodes:
        print(f"    Node {nid}: ({pos[0]:.1f}, {pos[1]:.1f})")
    
    # Generate tiled swastika grids
    configs = [
        ('swastika_v2 3×3', dict(unit='swastika_v2', box=(10, 10), grid=(3, 3))),
        ('swastika_v2 5×5', dict(unit='swastika_v2', box=(10, 10), grid=(5, 5))),
        ('swastika_v2 4×4 +mirror_x', dict(unit='swastika_v2', box=(10, 10), grid=(4, 4), mirror_x=True)),
        ('swastika_v2 4×4 +mirror_xy', dict(unit='swastika_v2', box=(10, 10), grid=(4, 4), mirror_x=True, mirror_y=True)),
        ('swastika 3×3 arm_frac=0.7', dict(unit='swastika', box=(10, 10), grid=(3, 3), unit_kwargs={'arm_frac': 0.7})),
    ]
    
    for name, params in configs:
        print(f"\n[Generate] {name}")
        try:
            g = fn.pattern_2d(**params)
            print(f"  Result: {g}")
            
            # Check connectivity
            connected = _check_connected(g)
            print(f"  Connected: {connected}")
            
            results[name] = {
                'graph': g,
                'nodes': len(g.nodes),
                'edges': len(g.edges),
                'connected': connected,
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {'error': str(e)}
    
    # Test with n_pts_per_side (wavy arms)
    print("\n[Generate] swastika_v2 3×3 + wavy arms")
    wavy = fn.pattern_2d(
        unit='swastika_v2',
        box=(10, 10),
        grid=(3, 3),
        n_pts_per_side=2,
        perturbation=0.15,
        seed=42,
    )
    print(f"  Wavy: {wavy}")
    results['wavy'] = {'graph': wavy, 'nodes': len(wavy.nodes), 'edges': len(wavy.edges)}
    
    checkpoint('swastika', {k: {kk: vv for kk, vv in v.items() if kk != 'graph'} 
                for k, v in results.items()})
    return results


# ======================================================================
# Part 4: Multi-level composition
# ======================================================================

def demo_multilevel():
    """Demonstrate multi-level structure building."""
    print("\n" + "="*60)
    print("PART 4: Multi-level Composition")
    print("="*60)
    
    results = {}
    
    # Level 1+2: Custom polyline → square arrangement
    print("\n[Multi] Z-shape polyline → 4×4 grid")
    z_grid = fn.pattern_2d(
        points=[(0, 10), (10, 10), (0, 0), (10, 0)],
        closed=False,
        box=(10, 10),
        grid=(4, 4),
        mirror_x=True,
        mirror_y=True,
    )
    print(f"  Z-grid: {z_grid}")
    results['z_grid'] = z_grid
    
    # Level 3: Rotate the whole thing
    print("\n[Multi] Honeycomb + rotation 30°")
    rotated_hc = fn.pattern_2d(
        unit='honeycomb',
        box=(10, 10),
        grid=(3, 3),
        rotation=30.0,
    )
    print(f"  Rotated honeycomb: {rotated_hc}")
    results['rotated_hc'] = rotated_hc
    
    # Cross pattern (already built-in, but show the concept)
    print("\n[Multi] Cross unit → 5×5 with mirror")
    cross = fn.pattern_2d(
        unit='cross',
        box=(10, 10),
        grid=(5, 5),
        mirror_x=True,
        mirror_y=True,
        n_pts_per_side=1,
        seed=10,
    )
    print(f"  Cross grid: {cross}")
    results['cross'] = cross
    
    checkpoint('multilevel', {k: f"{v}" for k, v in results.items()})
    return results


# ======================================================================
# Utility: Connectivity check
# ======================================================================

def _check_connected(graph):
    """Check if graph is connected using BFS."""
    if not graph.nodes:
        return True
    adj = {}
    for nid in graph.nodes:
        adj[nid] = set()
    for e in graph.edges:
        adj[e.node_i].add(e.node_j)
        adj[e.node_j].add(e.node_i)
    
    start = next(iter(graph.nodes))
    visited = {start}
    queue = [start]
    while queue:
        n = queue.pop(0)
        for nb in adj.get(n, set()):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == len(graph.nodes)


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 60)
    print("FiberNet API Demo + 卍 Structure Generation")
    print("=" * 60)
    print(f"Version: {fn.__version__}")
    print(f"Available units: {fn.list_units()}")
    
    # Run demos
    r1 = demo_paradigm()
    r2 = demo_api_examples()
    r3 = demo_swastika()
    r4 = demo_multilevel()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_results = {**r1, **r2, **r3, **r4}
    for name, result in all_results.items():
        if isinstance(result, dict):
            if 'error' in result:
                print(f"  ❌ {name}: {result['error']}")
            elif 'graph' in result:
                g = result['graph']
                conn = result.get('connected', '?')
                print(f"  ✅ {name}: {g} (connected={conn})")
            else:
                print(f"  ✅ {name}: {result}")
        else:
            print(f"  ✅ {name}: {result}")
    
    # Cleanup checkpoint
    if CHECKPOINT_FILE.exists():
        os.remove(CHECKPOINT_FILE)
        print(f"\n  🧹 Cleaned up checkpoint file")
    
    print(f"\nAnalysis document: analysis_scripts/API_ANALYSIS.md")
    print("Done!")


if __name__ == '__main__':
    main()

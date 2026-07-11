#!/usr/bin/env python3
"""
FiberNet 生成范式 v2 Demo
==========================
演示四级流水线 + 多层级组合 + 连通性验证。

运行: python3 analysis_scripts/paradigm_v2_demo.py

所有生成结果打印统计信息 + 连通性检查。
支持断点续跑（checkpoint）。
"""
import sys, os, json, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import fibernet as fn

CHECKPOINT = Path(__file__).parent.parent / 'output_viz' / '_paradigm_v2_checkpoint.json'

def save_checkpoint(stage, results):
    state = {'stage': stage, 'results': results, 'timestamp': time.time()}
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT, 'w') as f:
        json.dump(state, f, indent=2, default=str)

def load_checkpoint():
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return None

def check_connected(g):
    if not g.nodes:
        return True
    adj = {nid: set() for nid in g.nodes}
    for e in g.edges.values():
        adj[e.node_i].add(e.node_j)
        adj[e.node_j].add(e.node_i)
    start = next(iter(g.nodes))
    visited = {start}
    queue = [start]
    while queue:
        n = queue.pop(0)
        for nb in adj.get(n, set()):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == len(g.nodes)

def test(name, g, results):
    conn = check_connected(g)
    sym = '✅' if conn else '⚠️'
    print(f"  {sym} {name:40s}: {g}")
    results[name] = {'str': str(g), 'connected': conn, 'nodes': len(g.nodes), 'edges': len(g.edges)}
    return conn

def demo():
    print("=" * 70)
    print("FiberNet 生成范式 v2 Demo")
    print(f"Version: {fn.__version__}, Units: {fn.list_units()}")
    print("=" * 70)

    results = {}

    # --- Level 1: Polyline Primitives ---
    print("\n--- Level 1: 折线基元 ---")
    test("L1: straight line", fn.pattern_2d(points=[(0,5),(10,5)], closed=False, box=(10,10), grid=(1,1)), results)
    test("L1: curved (n_pts=3)", fn.pattern_2d(points=[(0,5),(10,5)], closed=False, box=(10,10), grid=(1,1), n_pts_per_side=3, perturbation=0.3, seed=42), results)
    test("L1: V-shape", fn.pattern_2d(points=[(0,0),(5,10),(10,0)], closed=False, box=(10,10), grid=(1,1)), results)
    test("L1: closed polygon", fn.pattern_2d(points=[(2,2),(8,2),(8,8),(2,8)], closed=True, box=(10,10), grid=(1,1), n_pts_per_side=2), results)
    save_checkpoint('level1', results)

    # --- Level 2: Shapes + Tiling ---
    print("\n--- Level 2: 形状 + 平铺 ---")
    for u in fn.list_units():
        g = fn.pattern_2d(unit=u, box=(10,10), grid=(3,3))
        test(f"L2: {u}", g, results)
    save_checkpoint('level2', results)

    # --- Level 2: Custom shape ---
    print("\n--- Level 2: 自定义形状 ---")
    star_pts = [(50,0),(65,35),(100,50),(65,65),(50,100),(35,65),(0,50),(35,35)]
    test("L2: custom star", fn.pattern_2d(points=star_pts, closed=True, fit_to_box=True, box=(10,10), grid=(3,3)), results)

    # Register zigzag
    def zigzag_factory(box, radius=0.1, material=None, **kw):
        w, h = box
        g = fn.StructureGraph(dimension=2, box_size=[w, h])
        n0 = g.add_node([0, h/2])
        n1 = g.add_node([w/3, h])
        n2 = g.add_node([2*w/3, 0])
        n3 = g.add_node([w, h/2])
        g.add_edge(n0, n1, radius=radius, material=material)
        g.add_edge(n1, n2, radius=radius, material=material)
        g.add_edge(n2, n3, radius=radius, material=material)
        return g
    fn.register_unit("zigzag", zigzag_factory)
    test("L2: registered zigzag", fn.pattern_2d(unit="zigzag", box=(10,10), grid=(4,4)), results)
    save_checkpoint('level2b', results)

    # --- Level 3: Transforms ---
    print("\n--- Level 3: 变换 ---")
    for rot in [15, 30, 45, 60]:
        test(f"L3: honeycomb rot={rot}", fn.pattern_2d(unit="honeycomb", box=(10,10), grid=(2,2), rotation=rot), results)
    test("L3: square mirror_x+rot45", fn.pattern_2d(unit="square", box=(10,10), grid=(2,2), mirror_x=True, rotation=45), results)
    save_checkpoint('level3', results)

    # --- Level 4: Multi-level ---
    print("\n--- Level 4: 多层级组合 ---")
    from fibernet.core.tiling import fit_unit_to_box

    def nested_honeycomb(box, **kw):
        inner = fn.pattern_2d(unit="honeycomb", box=(box[0]/2, box[1]/2), grid=(2,2))
        return fit_unit_to_box(inner, target_box=list(box) + [0.0])
    fn.register_unit("nested_honeycomb", nested_honeycomb)
    test("L4: nested honeycomb 3x3", fn.pattern_2d(unit="nested_honeycomb", box=(10,10), grid=(3,3)), results)

    # Merge two different regions
    g1 = fn.pattern_2d(unit="square", box=(10,10), grid=(3,3))
    g2 = fn.pattern_2d(unit="honeycomb", box=(10,10), grid=(3,3))
    g2 = fn.translate(g2, [30, 0, 0])
    combined = g1.merge(g2)
    test("L4: square+honeycomb merged", combined, results)
    save_checkpoint('level4', results)

    # --- 3D ---
    print("\n--- 3D Structures ---")
    for u in ['cubic', 'octet', 'diamond_3d']:
        test(f"3D: {u}", fn.pattern_3d(unit=u, box=(5,5,5), grid=(2,2,2)), results)
    save_checkpoint('3d', results)

    # --- FEM ---
    print("\n--- FEM Simulation ---")
    g = fn.pattern_2d(unit="honeycomb", box=(10,10), grid=(3,3))
    fem = fn.BeamFEM(g)
    result = fem.uniaxial_tension(strain=0.01)
    print(f"  Honeycomb FEM: E*={result.effective_youngs_modulus:.2e}, ν*={result.effective_poissons_ratio:.3f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    total = len(results)
    passed = sum(1 for v in results.values() if v.get('connected', True))
    print(f"SUMMARY: {passed}/{total} connected")
    if passed < total:
        print("Disconnected:")
        for name, v in results.items():
            if not v.get('connected', True):
                print(f"  ⚠️ {name}")
    print("=" * 70)

    if CHECKPOINT.exists():
        os.remove(CHECKPOINT)
        print("🧹 Checkpoint cleaned up")

if __name__ == '__main__':
    demo()

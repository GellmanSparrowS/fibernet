#!/usr/bin/env python3
"""
FiberNet 生成范式 Demo
=======================
演示四级流水线 + 多层级组合能力。

运行: python3 analysis_scripts/generation_paradigm_demo.py

所有生成结果打印统计信息，不生成图片。
支持断点续跑（checkpoint）。
"""

import sys, os, json, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fibernet as fn
from fibernet.core.structure_graph import StructureGraph

CHECKPOINT = Path(__file__).parent.parent / 'output_viz' / '_paradigm_checkpoint.json'


def save_checkpoint(stage, results_summary):
    state = {'stage': stage, 'results': results_summary, 'timestamp': time.time()}
    with open(CHECKPOINT, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def load_checkpoint():
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return None


def check_connected(graph):
    """BFS connectivity check."""
    if not graph.nodes:
        return True
    adj = {}
    for nid in graph.nodes:
        adj[nid] = set()
    for e in graph.edges.values():
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


def demo_level1():
    """Level 1: 单条直线段 → 变形折线"""
    print("\n" + "=" * 60)
    print("LEVEL 1: 单条直线段 → 变形折线")
    print("=" * 60)

    results = {}

    # 1a. 纯直线
    print("\n[1a] 单条直线段: (0,5) → (10,5)")
    line = fn.pattern_2d(
        points=[(0, 5), (10, 5)],
        closed=False, box=(10, 10), grid=(1, 1),
    )
    conn = check_connected(line)
    print(f"  结果: {line}, 连通={conn}")
    results['line'] = str(line)

    # 1b. 直线 + 中间节点（变成曲线）
    print("\n[1b] 直线 + 3个中间节点 + 位移 → 变形曲线")
    curved = fn.pattern_2d(
        points=[(0, 5), (10, 5)],
        closed=False, box=(10, 10), grid=(1, 1),
        n_pts_per_side=3, perturbation=0.3, seed=42,
    )
    conn = check_connected(curved)
    print(f"  结果: {curved}, 连通={conn}")
    results['curved'] = str(curved)

    # 1c. V 形折线
    print("\n[1c] V 形折线: (0,0)→(5,10)→(10,0)")
    vshape = fn.pattern_2d(
        points=[(0, 0), (5, 10), (10, 0)],
        closed=False, box=(10, 10), grid=(1, 1),
    )
    conn = check_connected(vshape)
    print(f"  结果: {vshape}, 连通={conn}")
    results['vshape'] = str(vshape)

    save_checkpoint('level1', results)
    return results


def demo_level2():
    """Level 2: Unit Cell 组装"""
    print("\n" + "=" * 60)
    print("LEVEL 2: Unit Cell 组装（12种内置 + 自定义）")
    print("=" * 60)

    results = {}

    # 2a. 所有内置 unit
    print("\n[2a] 12种内置 unit cell (3×3 grid):")
    for u in fn.list_units():
        try:
            g = fn.pattern_2d(unit=u, box=(10, 10), grid=(3, 3))
            conn = check_connected(g)
            sym = '✅' if conn else '⚠️'
            print(f"  {sym} {u:15s}: {g}")
            results[u] = {'str': str(g), 'connected': conn}
        except Exception as e:
            print(f"  ❌ {u:15s}: {e}")
            results[u] = {'error': str(e)}

    # 2b. 自定义点 + fit_to_box
    print("\n[2b] 自定义 5 角星 + fit_to_box=True:")
    star_pts = [(50, 0), (65, 35), (100, 50), (65, 65), (50, 100), (35, 65), (0, 50), (35, 35)]
    star = fn.pattern_2d(
        points=star_pts, closed=True,
        fit_to_box=True, box=(10, 10), grid=(3, 3),
    )
    conn = check_connected(star)
    print(f"  结果: {star}, 连通={conn}")
    results['star_custom'] = {'str': str(star), 'connected': conn}

    # 2c. 内部形状 + boundary_mode='extend'
    print("\n[2c] 内部正方形 + boundary_mode='extend':")
    inner_sq = fn.pattern_2d(
        points=[(3, 3), (7, 3), (7, 7), (3, 7)],
        closed=True, box=(10, 10), grid=(3, 3),
        boundary_mode='extend',
    )
    conn = check_connected(inner_sq)
    print(f"  结果: {inner_sq}, 连通={conn}")
    results['inner_extend'] = {'str': str(inner_sq), 'connected': conn}

    save_checkpoint('level2', results)
    return results


def demo_level3():
    """Level 3: Transform (变换)"""
    print("\n" + "=" * 60)
    print("LEVEL 3: Transform（镜像 + 旋转）")
    print("=" * 60)

    results = {}

    configs = [
        ('square 无变换', dict(unit='square', box=(10, 10), grid=(2, 2))),
        ('square + mirror_x', dict(unit='square', box=(10, 10), grid=(2, 2), mirror_x=True)),
        ('square + mirror_y', dict(unit='square', box=(10, 10), grid=(2, 2), mirror_y=True)),
        ('square + mirror_xy', dict(unit='square', box=(10, 10), grid=(2, 2), mirror_x=True, mirror_y=True)),
        ('honeycomb + rot30', dict(unit='honeycomb', box=(10, 10), grid=(2, 2), rotation=30.0)),
        ('hexagon + rot45 + mirror', dict(unit='hexagon', box=(10, 10), grid=(2, 2), rotation=45.0, mirror_x=True)),
    ]

    for name, params in configs:
        g = fn.pattern_2d(**params)
        conn = check_connected(g)
        sym = '✅' if conn else '⚠️'
        print(f"  {sym} {name:30s}: {g}")
        results[name] = {'str': str(g), 'connected': conn}

    save_checkpoint('level3', results)
    return results


def demo_level4():
    """Level 4: Tiling + Welding"""
    print("\n" + "=" * 60)
    print("LEVEL 4: Tiling + Welding（不同 grid 尺寸）")
    print("=" * 60)

    results = {}

    for grid in [(2, 2), (3, 3), (4, 4), (5, 5), (3, 5)]:
        g = fn.pattern_2d(unit='honeycomb', box=(10, 10), grid=grid)
        conn = check_connected(g)
        sym = '✅' if conn else '⚠️'
        print(f"  {sym} grid={grid}: {g}")
        results[f'grid_{grid}'] = {'str': str(g), 'connected': conn}

    save_checkpoint('level4', results)
    return results


def demo_multilevel():
    """多层级组合：register_unit 嵌套"""
    print("\n" + "=" * 60)
    print("MULTILEVEL: 多层级组合（嵌套 pattern）")
    print("=" * 60)

    results = {}

    # 注册一个 "小网格作为 unit" 的工厂
    def _unit_nested_honeycomb(box, **kwargs):
        """内部生成 2×2 honeycomb，然后缩放适配到 box。"""
        inner = fn.pattern_2d(unit='honeycomb', box=(box[0] / 2, box[1] / 2), grid=(2, 2))
        from fibernet.core.tiling import fit_unit_to_box
        return fit_unit_to_box(inner, target_box=list(box) + [0.0])

    fn.register_unit('nested_honeycomb', _unit_nested_honeycomb)

    print("\n[Multi] 嵌套蜂窝（2×2蜂窝作为unit） → 3×3 grid:")
    g = fn.pattern_2d(unit='nested_honeycomb', box=(10, 10), grid=(3, 3))
    conn = check_connected(g)
    print(f"  结果: {g}, 连通={conn}")
    results['nested'] = {'str': str(g), 'connected': conn}

    # 注册一个 Z 字形 unit
    def _unit_zigzag(box, **kwargs):
        from fibernet.core.material import Material
        w, h = box
        r = kwargs.get('radius', 0.1)
        mat = kwargs.get('material', Material())
        g = StructureGraph(dimension=2, box_size=[w, h, 0])
        n0 = g.add_node([0, h / 2])
        n1 = g.add_node([w / 2, h])
        n2 = g.add_node([w / 2, 0])
        n3 = g.add_node([w, h / 2])
        g.add_edge(n0, n1, radius=r, material=mat)
        g.add_edge(n1, n2, radius=r, material=mat)
        g.add_edge(n2, n3, radius=r, material=mat)
        return g

    fn.register_unit('zigzag', _unit_zigzag)

    print("\n[Multi] Z 字形 unit → 4×4 grid + mirror:")
    g = fn.pattern_2d(unit='zigzag', box=(10, 10), grid=(4, 4), mirror_x=True, mirror_y=True)
    conn = check_connected(g)
    print(f"  结果: {g}, 连通={conn}")
    results['zigzag'] = {'str': str(g), 'connected': conn}

    # merge 两个不同区域
    print("\n[Multi] 手动 merge: square grid + honeycomb grid 并排:")
    g1 = fn.pattern_2d(unit='square', box=(10, 10), grid=(3, 3))
    g2 = fn.pattern_2d(unit='honeycomb', box=(10, 10), grid=(3, 3))
    from fibernet.core.transforms import translate
    g2_shifted = translate(g2, [30, 0, 0])
    combined = g1.merge(g2_shifted)
    print(f"  结果: {combined}")
    results['merged'] = str(combined)

    save_checkpoint('multilevel', results)
    return results


def main():
    print("=" * 60)
    print("FiberNet 生成范式 Demo")
    print(f"Version: {fn.__version__}")
    print(f"Units: {fn.list_units()}")
    print("=" * 60)

    cp = load_checkpoint()
    if cp:
        print(f"\n  📋 Found checkpoint at stage: {cp['stage']}")

    r1 = demo_level1()
    r2 = demo_level2()
    r3 = demo_level3()
    r4 = demo_level4()
    r5 = demo_multilevel()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_results = {**r1, **r2, **r3, **r4, **r5}
    errors = 0
    for name, val in all_results.items():
        if isinstance(val, dict):
            if 'error' in val:
                print(f"  ❌ {name}: {val['error']}")
                errors += 1
            elif 'connected' in val:
                sym = '✅' if val['connected'] else '⚠️'
                print(f"  {sym} {name}: {val.get('str', '')}")
        else:
            print(f"  ✅ {name}: {val}")

    print(f"\n  Total: {len(all_results)} tests, {errors} errors")

    if CHECKPOINT.exists():
        os.remove(CHECKPOINT)
        print("  🧹 Cleaned up checkpoint")

    print("\nDone! 分析文档: analysis_scripts/generation_paradigm_analysis.md")


if __name__ == '__main__':
    main()

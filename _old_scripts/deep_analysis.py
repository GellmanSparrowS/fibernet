"""
深度分析 FiberNet 生成模块的质量
从 Graph 可行性、仿真接入、可编程性等角度评估
"""
import sys, traceback
import numpy as np
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import fibernet as fn
from fibernet.core.network import FiberNetwork

# ============================================================
# 1. Graph 可行性分析
# ============================================================

def analyze_graph_quality(net, name):
    """分析一个网络的图论质量"""
    r = {"name": name}
    
    # --- 基本连通性 ---
    adj = defaultdict(set)
    for cl in net.crosslinks:
        adj[cl.fiber_i].add(cl.fiber_j)
        adj[cl.fiber_j].add(cl.fiber_i)
    
    visited = set()
    components = []
    for s in range(net.num_fibers):
        if s not in visited:
            comp = set()
            q = [s]
            while q:
                n = q.pop(0)
                if n in visited: continue
                visited.add(n); comp.add(n)
                q.extend(adj[n] - visited)
            components.append(comp)
    
    r["num_components"] = len(components)
    r["largest_component_pct"] = max(len(c) for c in components) / net.num_fibers if net.num_fibers > 0 else 0
    r["is_connected"] = len(components) == 1
    r["isolated_fibers"] = sum(1 for c in components if len(c) == 1)
    
    # --- 度分布 ---
    degrees = net.degree_distribution()
    r["mean_degree"] = float(np.mean(degrees))
    r["max_degree"] = int(np.max(degrees)) if len(degrees) > 0 else 0
    r["degree_zero_fibers"] = int(np.sum(degrees == 0))
    r["degree_std"] = float(np.std(degrees))
    
    # --- 邻接矩阵稀疏度 ---
    n = net.num_fibers
    if n > 0:
        total_possible = n * (n - 1) / 2
        actual_edges = len(set((min(cl.fiber_i, cl.fiber_j), max(cl.fiber_i, cl.fiber_j)) for cl in net.crosslinks))
        r["edge_density"] = actual_edges / total_possible if total_possible > 0 else 0
    else:
        r["edge_density"] = 0
    
    # --- Crosslink 位置合理性 ---
    if net.crosslinks:
        cl_positions = np.array([cl.position for cl in net.crosslinks])
        r["crosslink_bbox"] = (cl_positions.max(axis=0) - cl_positions.min(axis=0)).tolist()
        
        # 检查 crosslink 是否在纤维上
        off_fiber_count = 0
        for cl in net.crosslinks[:100]:  # sample
            fi = net.fibers[cl.fiber_i]
            fj = net.fibers[cl.fiber_j]
            # Distance from crosslink position to nearest point on fiber
            pi = fi.centerline[int(cl.param_i * (fi.num_points - 1))] if cl.param_i <= 1.0 else fi.centerline[-1]
            pj = fj.centerline[int(cl.param_j * (fj.num_points - 1))] if cl.param_j <= 1.0 else fj.centerline[-1]
            d_i = np.linalg.norm(cl.position - pi)
            d_j = np.linalg.norm(cl.position - pj)
            if d_i > fi.radius * 5 or d_j > fj.radius * 5:
                off_fiber_count += 1
        r["off_fiber_crosslinks_pct"] = off_fiber_count / min(100, len(net.crosslinks)) * 100
    else:
        r["crosslink_bbox"] = [0, 0, 0]
        r["off_fiber_crosslinks_pct"] = 0
    
    return r


# ============================================================
# 2. 仿真接入可行性
# ============================================================

def analyze_simulation_readiness(net, name):
    """分析一个网络是否适合仿真"""
    r = {"name": name}
    
    # --- 密度合理性 ---
    r["density"] = net.density()
    r["total_length"] = net.total_length
    
    # --- 长度分布 ---
    if net.num_fibers > 0:
        lengths = np.array([f.length for f in net.fibers])
        r["mean_length"] = float(np.mean(lengths))
        r["cv_length"] = float(np.std(lengths) / np.mean(lengths)) if np.mean(lengths) > 0 else 0
        r["min_length"] = float(np.min(lengths))
        r["max_length"] = float(np.max(lengths))
        r["length_ratio"] = float(np.max(lengths) / np.min(lengths)) if np.min(lengths) > 0 else float('inf')
    else:
        r["mean_length"] = 0
        r["cv_length"] = 0
        r["length_ratio"] = 0
    
    # --- 半径分布 ---
    if net.num_fibers > 0:
        radii = np.array([f.radius for f in net.fibers])
        r["mean_radius"] = float(np.mean(radii))
        r["radius_cv"] = float(np.std(radii) / np.mean(radii)) if np.mean(radii) > 0 else 0
    else:
        r["mean_radius"] = 0
        r["radius_cv"] = 0
    
    # --- Bounding box 合理性 ---
    bb_min, bb_max = net.bounding_box()
    bb = bb_max - bb_min
    r["bbox"] = bb.tolist()
    r["aspect_ratio"] = float(max(bb[:2]) / max(min(bb[:2]), 1e-10)) if net.dimension == 2 else 0
    
    # --- FEM 可解性预估 ---
    # 条件: 连通 + 足够多的 crosslinks + 合理密度
    r["fem_ready"] = (
        r.get("density", 0) > 0.001 and
        net.num_crosslinks > 0 and
        (net.num_crosslinks / max(net.num_fibers, 1)) > 0.5
    )
    
    return r


# ============================================================
# 3. 可编程性分析
# ============================================================

def analyze_programmability():
    """分析 API 的可编程性"""
    import inspect
    
    results = {}
    
    # 列出所有 generator 的参数签名
    all_gens = fn.list_generators()
    for name in all_gens:
        func = fn._registry.generators.get(name)
        if func:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            results[name] = {
                "params": params,
                "n_params": len(params),
                "has_grid_size": "grid_size" in params,
                "has_radius": "radius" in params,
                "has_material": "material" in params,
            }
    
    return results


# ============================================================
# 4. 多样性分析
# ============================================================

def analyze_diversity():
    """分析生成器产生的结构多样性"""
    all_gens = fn.list_generators()
    
    stats = []
    for name in all_gens:
        try:
            net = fn.create(name)
            s = {"name": name}
            s["fibers"] = net.num_fibers
            s["crosslinks"] = net.num_crosslinks
            s["dimension"] = net.dimension
            s["density"] = net.density()
            s["mean_length"] = net.mean_fiber_length
            
            if net.num_fibers > 0:
                dirs = net.fiber_orientations()
                if net.dimension == 2 and len(dirs) > 1:
                    angles = np.arctan2(dirs[:, 1], dirs[:, 0])
                    # Nematic order parameter
                    cos2 = np.mean(np.cos(2 * angles))
                    sin2 = np.mean(np.sin(2 * angles))
                    s["nematic_order"] = float(np.sqrt(cos2**2 + sin2**2))
                else:
                    s["nematic_order"] = 0
                
                lengths = np.array([f.length for f in net.fibers])
                s["length_cv"] = float(np.std(lengths) / np.mean(lengths)) if np.mean(lengths) > 0 else 0
            else:
                s["nematic_order"] = 0
                s["length_cv"] = 0
            
            stats.append(s)
        except Exception as e:
            stats.append({"name": name, "error": str(e)})
    
    return stats


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("FIBERNET 生成模块深度分析")
print("=" * 80)

# --- Graph 可行性 ---
print("\n\n🔗 1. GRAPH 可行性分析")
print("-" * 80)
print(f"{'Generator':30s} | {'Comp':>4s} | {'Conn%':>6s} | {'Deg':>5s} | {'Iso':>4s} | {'Off%':>5s} | {'EdgD':>5s}")
print("-" * 80)

all_gens = fn.list_generators()
graph_results = []
for name in all_gens:
    try:
        net = fn.create(name)
        g = analyze_graph_quality(net, name)
        graph_results.append(g)
        conn_pct = g["largest_component_pct"] * 100
        print(f"  {name:30s} | {g['num_components']:4d} | {conn_pct:5.1f}% | {g['mean_degree']:5.2f} | {g['isolated_fibers']:4d} | {g['off_fiber_crosslinks_pct']:5.1f} | {g['edge_density']:.3f}")
    except Exception as e:
        print(f"  {name:30s} | FAIL: {e}")

# 统计
connected = sum(1 for g in graph_results if g["is_connected"])
high_conn = sum(1 for g in graph_results if g["largest_component_pct"] > 0.9)
print(f"\n  完全连通: {connected}/{len(graph_results)}")
print(f"  >90%连通: {high_conn}/{len(graph_results)}")

# --- 仿真接入 ---
print("\n\n⚙️  2. 仿真接入可行性")
print("-" * 80)
print(f"{'Generator':30s} | {'Dens':>7s} | {'Lmean':>7s} | {'Lcv':>5s} | {'Lratio':>8s} | {'FEM':>4s}")
print("-" * 80)

sim_results = []
for name in all_gens:
    try:
        net = fn.create(name)
        s = analyze_simulation_readiness(net, name)
        sim_results.append(s)
        fem_str = "✓" if s["fem_ready"] else "✗"
        lratio = f"{s['length_ratio']:.1f}" if s['length_ratio'] < 1e6 else "INF"
        print(f"  {name:30s} | {s['density']:7.4f} | {s['mean_length']:7.3f} | {s['cv_length']:5.2f} | {lratio:>8s} | {fem_str:>4s}")
    except Exception as e:
        print(f"  {name:30s} | FAIL: {e}")

fem_ready = sum(1 for s in sim_results if s.get("fem_ready"))
print(f"\n  FEM-ready: {fem_ready}/{len(sim_results)}")

# --- 可编程性 ---
print("\n\n🔧 3. 可编程性分析")
print("-" * 80)

prog = analyze_programmability()
has_grid = sum(1 for v in prog.values() if v["has_grid_size"])
has_radius = sum(1 for v in prog.values() if v["has_radius"])
has_material = sum(1 for v in prog.values() if v["has_material"])
avg_params = np.mean([v["n_params"] for v in prog.values()])

print(f"  总生成器数: {len(prog)}")
print(f"  平均参数数: {avg_params:.1f}")
print(f"  有 grid_size 参数: {has_grid}/{len(prog)} ({has_grid/len(prog)*100:.0f}%)")
print(f"  有 radius 参数: {has_radius}/{len(prog)} ({has_radius/len(prog)*100:.0f}%)")
print(f"  有 material 参数: {has_material}/{len(prog)} ({has_material/len(prog)*100:.0f}%)")

# 参数覆盖分析
param_sets = defaultdict(int)
for name, v in prog.items():
    for p in v["params"]:
        param_sets[p] += 1

print(f"\n  最常用参数:")
for p, c in sorted(param_sets.items(), key=lambda x: -x[1])[:10]:
    print(f"    {p}: {c} 个生成器使用")

# --- 多样性 ---
print("\n\n📊 4. 结构多样性分析")
print("-" * 80)
print(f"{'Generator':30s} | {'Fibers':>6s} | {'CL':>6s} | {'Dim':>3s} | {'Dens':>6s} | {'Nemat':>6s} | {'Lcv':>5s}")
print("-" * 80)

div = analyze_diversity()
for s in div:
    if "error" in s:
        print(f"  {s['name']:30s} | FAIL: {s['error']}")
    else:
        print(f"  {s['name']:30s} | {s['fibers']:6d} | {s['crosslinks']:6d} | {s['dimension']:3d} | {s['density']:6.4f} | {s['nematic_order']:6.4f} | {s['length_cv']:5.2f}")

# 多样性指标
if div:
    densities = [s.get("density", 0) for s in div if "density" in s]
    nematics = [s.get("nematic_order", 0) for s in div if "nematic_order" in s]
    fiber_counts = [s.get("fibers", 0) for s in div if "fibers" in s]
    
    print(f"\n  密度范围: {min(densities):.4f} ~ {max(densities):.4f}")
    print(f"  向序参数范围: {min(nematics):.4f} ~ {max(nematics):.4f}")
    print(f"  纤维数范围: {min(fiber_counts)} ~ {max(fiber_counts)}")
    print(f"  2D 生成器: {sum(1 for s in div if s.get('dimension') == 2)}")
    print(f"  3D 生成器: {sum(1 for s in div if s.get('dimension') == 3)}")

# ============================================================
# 5. 关键问题总结
# ============================================================
print("\n\n" + "=" * 80)
print("关键问题总结")
print("=" * 80)

# 问题1: 不连通的生成器
disconnected = [g for g in graph_results if not g["is_connected"]]
print(f"\n❌ 不连通的生成器 ({len(disconnected)}):")
for g in disconnected:
    print(f"  - {g['name']}: {g['num_components']} 组件, 最大连通分量 {g['largest_component_pct']*100:.1f}%")

# 问题2: FEM 不可用
not_fem = [s for s in sim_results if not s.get("fem_ready")]
print(f"\n❌ FEM 不可用 ({len(not_fem)}):")
for s in not_fem:
    print(f"  - {s['name']}: density={s['density']:.4f}, crosslinks={0 if 'crosslinks' not in s else 'N/A'}")

# 问题3: 无参数化
no_grid = [name for name, v in prog.items() if not v["has_grid_size"]]
print(f"\n⚠️  无 grid_size 参数 ({len(no_grid)}):")
for name in no_grid[:5]:
    print(f"  - {name}: params={prog[name]['params']}")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)

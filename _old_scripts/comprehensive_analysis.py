"""
FiberNet v2.0 Comprehensive Data Analysis Pipeline
===================================================
Structural diversity, mechanics/deformation, ML/RL — all data output (no images).

Phases:
  1. Structural diversity: generate diverse structures, compare graph metrics
  2. Weld graph: crossing detection statistics
  3. Feature extraction: 94-dim features across structures
  4. Mechanics simulation: FEM on different structures, stress-strain
  5. ML proxy model: feature->property mapping with sklearn
  6. RL environment: simple optimization demo

Usage:
  python comprehensive_analysis.py [--phase N] [--resume]

Saves results to analysis_results/ directory.
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
OUTPUT_DIR = Path(__file__).parent / "analysis_results"
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"


def save_phase(phase_name, data):
    path = OUTPUT_DIR / f"{phase_name}.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=_json_default)
    ckpt = {}
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            ckpt = json.load(f)
    ckpt[phase_name] = {"status": "completed", "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(ckpt, f, indent=2)
    print(f"  [✓] {phase_name} saved → {path.name}")


def _json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, set): return list(obj)
    return str(obj)


def is_phase_done(phase_name):
    if not CHECKPOINT_FILE.exists(): return False
    with open(CHECKPOINT_FILE) as f:
        ckpt = json.load(f)
    return ckpt.get(phase_name, {}).get("status") == "completed"


def print_table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=4)) + 2
                      for i, h in enumerate(headers)]
    hdr = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    sep = "─" * len(hdr)
    print(f"  {sep}")
    print(f"  {hdr}")
    print(f"  {sep}")
    for row in rows:
        print(f"  {''.join(str(v).ljust(w) for v, w in zip(row, col_widths))}")
    print(f"  {sep}")


def _to_nx(net):
    """Convert FiberNetwork to nx.Graph with pos attribute.
    
    Handles both crosslink-based and centerline-based conversion.
    """
    import networkx as nx
    from fibernet.graph.io import to_networkx as cl_to_nx
    
    if isinstance(net, nx.Graph):
        return net
    
    # Try crosslink-based conversion first
    G = cl_to_nx(net)
    if G.number_of_nodes() > 0:
        return G
    
    # Fallback: build spatial graph from fiber centerlines
    G = nx.Graph()
    node_idx = 0
    for fi, fiber in enumerate(net.fibers):
        cl = fiber.centerline
        n_pts = len(cl)
        # Sample every Nth point (reduce density)
        step = max(1, n_pts // 10)
        fiber_nodes = []
        for j in range(0, n_pts, step):
            pt = cl[j]
            G.add_node(node_idx, pos=tuple(pt))
            fiber_nodes.append(node_idx)
            node_idx += 1
        # Connect consecutive nodes in fiber
        for k in range(len(fiber_nodes) - 1):
            G.add_edge(fiber_nodes[k], fiber_nodes[k + 1])
    
    return G


# ============================================================
# Phase 1: Structural Diversity
# ============================================================
def phase1_structural_diversity():
    print("\n" + "="*70)
    print("  PHASE 1: Structural Diversity Analysis")
    print("="*70)
    
    import networkx as nx
    from fibernet.gen.regular import RegularNetworkGenerator
    from fibernet.gen.zigzag import ZigZagGenerator
    from fibernet.gen.ordered import (
        square_lattice_2d, triangular_lattice_2d, honeycomb_lattice_2d, kagome_lattice_2d
    )
    from fibernet.gen.disordered import random_straight_2d, oriented_random_2d
    from fibernet.gen.chiral import single_helix, chiral_metamaterial
    from fibernet.gen.woven import plain_weave_2d, twill_weave_2d
    from fibernet.gen.metamaterials import (
        reentrant_honeycomb_2d, star_honeycomb_2d, arrowhead_auxetic_2d
    )
    from fibernet.gen.bundles import parallel_bundle_2d, twisted_bundle_2d
    
    structures = {}
    
    def try_gen(name, fn):
        try:
            result = fn()
            G = _to_nx(result)
            structures[name] = G
            return True
        except Exception as e:
            print(f"    [!] {name}: {e}")
            return False
    
    # 1. Regular (P1-based) with varying perturbations
    print("  [1/8] Regular networks with perturbation sweep...")
    for p in [0.0, 0.1, 0.3, 0.5]:
        gen = RegularNetworkGenerator(
            side_length=10, num_points_per_side=2,
            perturbations=[(p, -p), (p, p), (-p, p)] if p > 0 else [],
            tiling=3
        )
        structures[f"regular_p{p:.1f}"] = gen.generate()
    
    # 2. ZigZag variants
    print("  [2/8] ZigZag networks (mirror combinations)...")
    for mx, my in [(True, True), (True, False), (False, True), (False, False)]:
        gen = ZigZagGenerator(n_cols=3, n_rows=5, mirror_x=mx, mirror_y=my)
        structures[f"zigzag_mx{mx}_my{my}"] = gen.generate()
    
    # 3. Ordered lattices
    print("  [3/8] Ordered lattices...")
    try_gen("ordered_square", lambda: square_lattice_2d(grid_size=(6,6), spacing=1.0))
    try_gen("ordered_triangular", lambda: triangular_lattice_2d(grid_size=(6,6), spacing=1.0))
    try_gen("ordered_honeycomb", lambda: honeycomb_lattice_2d(grid_size=(6,6), cell_size=1.0))
    try_gen("ordered_kagome", lambda: kagome_lattice_2d(grid_size=(5,5), spacing=1.5))
    
    # 4. Disordered
    print("  [4/8] Disordered networks...")
    try_gen("disordered_random", lambda: random_straight_2d(num_fibers=40, fiber_length=8, box_size=(20,20), seed=42))
    try_gen("disordered_oriented", lambda: oriented_random_2d(num_fibers=40, fiber_length=8, box_size=(20,20), preferred_angle=45, angular_spread=15, seed=42))
    
    # 5. Chiral
    print("  [5/8] Chiral structures...")
    try_gen("chiral_helix", lambda: single_helix(helix_radius=3.0, pitch=2.0, num_turns=4.0))
    try_gen("meta_chiral", lambda: chiral_metamaterial(unit_cell_size=5.0, grid_size=(3,3,3)))
    
    # 6. Woven
    print("  [6/8] Woven structures...")
    try_gen("woven_plain", lambda: plain_weave_2d(grid_size=(8,8), spacing=2.0, amplitude=0.3))
    try_gen("woven_twill", lambda: twill_weave_2d(grid_size=(8,8), spacing=2.0))
    
    # 7. Metamaterials
    print("  [7/8] Metamaterial structures...")
    try_gen("meta_reentrant", lambda: reentrant_honeycomb_2d(grid_size=(4,4), cell_height=5, cell_width=5))
    try_gen("meta_star", lambda: star_honeycomb_2d(grid_size=(3,3), star_arm_length=3.0, num_arms=4))
    try_gen("meta_arrowhead", lambda: arrowhead_auxetic_2d(grid_size=(4,4)))
    
    # 8. Bundles
    print("  [8/8] Bundle structures...")
    try_gen("bundle_parallel", lambda: parallel_bundle_2d(num_fibers=10, bundle_length=20, bundle_width=5))
    try_gen("bundle_twisted", lambda: twisted_bundle_2d(num_fibers=8, bundle_length=20, twist_pitch=10))
    
    # Compute graph metrics
    print(f"\n  Total structures: {len(structures)}")
    print("\n  Graph Metrics Comparison:")
    
    results = {}
    rows = []
    for name, G in sorted(structures.items()):
        if not isinstance(G, nx.Graph):
            continue
        
        n_n = G.number_of_nodes()
        n_e = G.number_of_edges()
        degs = [d for _, d in G.degree()]
        avg_deg = np.mean(degs) if degs else 0
        deg_std = np.std(degs) if degs else 0
        
        components = nx.number_connected_components(G) if n_n > 0 else 0
        density = nx.density(G) if n_n > 1 else 0
        
        pos = nx.get_node_attributes(G, 'pos')
        if pos and n_e > 0:
            elens = []
            for u, v in G.edges():
                pu = np.array(pos[u][:2] if len(pos[u]) >= 2 else pos[u], dtype=float)
                pv = np.array(pos[v][:2] if len(pos[v]) >= 2 else pos[v], dtype=float)
                elens.append(float(np.linalg.norm(pu - pv)))
            mean_len = np.mean(elens)
            std_len = np.std(elens)
            total_len = sum(elens)
        else:
            mean_len = std_len = total_len = 0
        
        try:
            cc = nx.average_clustering(G)
        except:
            cc = 0
        
        results[name] = {
            "nodes": n_n, "edges": n_e,
            "avg_degree": round(avg_deg, 2), "deg_std": round(deg_std, 2),
            "components": components, "density": round(density, 4),
            "mean_edge_len": round(mean_len, 3), "std_edge_len": round(std_len, 3),
            "total_length": round(total_len, 2),
            "clustering_coeff": round(cc, 4),
        }
        
        rows.append([name[:28], n_n, n_e, f"{avg_deg:.2f}",
                     f"{deg_std:.2f}", components, f"{density:.4f}",
                     f"{mean_len:.3f}", f"{cc:.4f}"])
    
    print_table(
        ["Structure", "Nodes", "Edges", "AvgDeg", "DStd", "Comp", "Density", "MeanLen", "CC"],
        rows,
        col_widths=[30, 7, 7, 8, 7, 6, 9, 9, 8]
    )
    
    save_phase("phase1_structural_diversity", results)
    return structures


# ============================================================
# Phase 2: Weld Graph Analysis
# ============================================================
def phase2_weld_analysis(structures):
    print("\n" + "="*70)
    print("  PHASE 2: Weld Graph Crossing Analysis")
    print("="*70)
    
    import networkx as nx
    from fibernet.graph.weld import weld_graph, find_intersections
    
    results = {}
    rows = []
    
    for name, G in sorted(structures.items()):
        if not isinstance(G, nx.Graph) or G.number_of_edges() < 2:
            continue
        
        n_before = G.number_of_nodes()
        e_before = G.number_of_edges()
        
        # Check if pos exists
        pos = nx.get_node_attributes(G, 'pos')
        if not pos:
            continue
        
        t0 = time.time()
        intersections = find_intersections(G)
        t_find = time.time() - t0
        
        n_crossings = sum(len(pts) for pts in intersections.values()) // 2
        n_crossed_edges = len(intersections)
        
        if n_crossings > 0 and n_before < 200:  # limit weld to small graphs
            t0 = time.time()
            G_welded = weld_graph(G)
            t_weld = time.time() - t0
            n_after = G_welded.number_of_nodes()
            e_after = G_welded.number_of_edges()
        else:
            t_weld = 0
            n_after = n_before + n_crossings
            e_after = e_before + n_crossings * 2
        
        results[name] = {
            "nodes_before": n_before, "edges_before": e_before,
            "crossings": n_crossings, "crossed_edges": n_crossed_edges,
            "nodes_after": n_after, "edges_after": e_after,
            "new_nodes": n_after - n_before, "new_edges": e_after - e_before,
            "find_time_ms": round(t_find * 1000, 2),
            "weld_time_ms": round(t_weld * 1000, 2),
        }
        
        rows.append([name[:28], n_before, e_before, n_crossings,
                     n_after, e_after, f"{t_find*1000:.1f}ms", f"{t_weld*1000:.1f}ms"])
    
    print("\n  Weld Graph Statistics:")
    print_table(
        ["Structure", "N_bef", "E_bef", "Cross", "N_aft", "E_aft", "FindT", "WeldT"],
        rows,
        col_widths=[30, 7, 7, 7, 7, 7, 10, 10]
    )
    
    save_phase("phase2_weld_analysis", results)


# ============================================================
# Phase 3: Feature Extraction (94-dim)
# ============================================================
def phase3_feature_extraction(structures):
    print("\n" + "="*70)
    print("  PHASE 3: 94-Dimensional Feature Extraction")
    print("="*70)
    
    import networkx as nx
    from fibernet.analysis.graph_features import GraphFeatureExtractor
    
    ext = GraphFeatureExtractor(canvas_size=256, thick=5)
    
    results = {}
    feature_matrix = []
    names = []
    
    for name, G in sorted(structures.items()):
        if not isinstance(G, nx.Graph) or G.number_of_nodes() < 3:
            continue
        pos = nx.get_node_attributes(G, 'pos')
        if not pos:
            continue
        
        try:
            t0 = time.time()
            features = ext.extract(G)
            elapsed = time.time() - t0
            
            results[name] = {
                "features": {k: round(v, 6) for k, v in features.items()},
                "n_features": len(features),
                "extraction_time_ms": round(elapsed * 1000, 1),
            }
            
            vec = ext.extract_vector(G)
            feature_matrix.append(vec)
            names.append(name)
        except Exception as e:
            print(f"    [!] {name}: {e}")
    
    feature_matrix = np.array(feature_matrix) if feature_matrix else np.array([])
    
    print(f"\n  Features extracted: {len(results)} structures × {feature_matrix.shape[1] if feature_matrix.ndim == 2 else 0} dims")
    
    if feature_matrix.ndim == 2 and feature_matrix.shape[0] > 1:
        feat_names = ext.FEATURE_COLS
        
        means = np.mean(feature_matrix, axis=0)
        stds = np.std(feature_matrix, axis=0)
        mins = np.min(feature_matrix, axis=0)
        maxs = np.max(feature_matrix, axis=0)
        
        variability = stds / (np.abs(means) + 1e-10)
        top_indices = np.argsort(variability)[-20:][::-1]
        
        print("\n  Top 20 Most Variable Features:")
        rows = []
        for idx in top_indices:
            fname = feat_names[idx] if idx < len(feat_names) else f"f{idx}"
            rows.append([fname[:32], f"{means[idx]:.4f}", f"{stds[idx]:.4f}",
                        f"{mins[idx]:.4f}", f"{maxs[idx]:.4f}", f"{variability[idx]:.3f}"])
        
        print_table(["Feature", "Mean", "Std", "Min", "Max", "CV"],
                     rows, col_widths=[34, 12, 12, 12, 12, 10])
        
        # Category summary
        print("\n  Feature Category Coverage:")
        cat_struct = [i for i, f in enumerate(feat_names)
                      if not any(k in f.lower() for k in ['pore', 'mesh', 'hole', 'contact', 'overlap', 'cross'])]
        cat_pore = [i for i, f in enumerate(feat_names)
                    if any(k in f.lower() for k in ['pore', 'mesh', 'hole'])]
        cat_contact = [i for i, f in enumerate(feat_names)
                       if any(k in f.lower() for k in ['contact', 'overlap', 'cross'])]
        
        for cat_name, indices in [("structural", cat_struct), ("pore", cat_pore), ("contact", cat_contact)]:
            if indices:
                vals = feature_matrix[:, indices]
                nz = np.count_nonzero(vals)
                print(f"    {cat_name:15s}: {len(indices):3d} features, {nz}/{vals.size} nonzero ({100*nz/max(vals.size,1):.0f}%)")
        
        np.save(OUTPUT_DIR / "feature_matrix.npy", feature_matrix)
        with open(OUTPUT_DIR / "feature_names.json", 'w') as f:
            json.dump({"structure_names": names, "feature_cols": list(feat_names)}, f, indent=2)
    
    save_phase("phase3_features", results)
    return results


# ============================================================
# Phase 4: Mechanics Simulation
# ============================================================
def phase4_mechanics(structures):
    print("\n" + "="*70)
    print("  PHASE 4: Mechanics Simulation & Deformation Analysis")
    print("="*70)
    
    import networkx as nx
    from fibernet.core.material import Material
    from fibernet.sim.mechanical import FiberFEM, stress_strain_curve
    from fibernet.graph.io import from_networkx
    
    material = Material(name="polymer", youngs_modulus=1e9, poissons_ratio=0.35, density=1200)
    
    # Select manageable-size structures
    sim_structures = {}
    for name, G in structures.items():
        if not isinstance(G, nx.Graph):
            continue
        pos = nx.get_node_attributes(G, 'pos')
        if not pos or G.number_of_nodes() < 5 or G.number_of_edges() < 3:
            continue
        if G.number_of_nodes() > 300:
            continue
        sim_structures[name] = G
    
    print(f"  Selected {len(sim_structures)} structures for FEM")
    
    results = {}
    rows = []
    
    for name, G in sorted(sim_structures.items()):
        try:
            network = from_networkx(G, material=material)
            if len(network.fibers) < 2:
                continue
            
            n_fibers = len(network.fibers)
            fem = FiberFEM(network, segments_per_fiber=3)
            
            positions = np.array([cl.position for cl in network.crosslinks])
            if len(positions) < 2:
                continue
            
            x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
            L0 = x_max - x_min
            if L0 < 1e-6:
                continue
            
            tol = 0.15 * L0
            left_nodes = [i for i, p in enumerate(positions) if p[0] <= x_min + tol]
            right_nodes = [i for i, p in enumerate(positions) if p[0] >= x_max - tol]
            
            if not left_nodes or not right_nodes:
                continue
            
            # Static analysis
            F = np.zeros(fem.num_dof)
            for n in right_nodes:
                F[n * 6] = 1.0
            
            t0 = time.time()
            result = fem.solve_static(forces=F, fixed_nodes=left_nodes)
            t_solve = time.time() - t0
            
            max_disp = result.max_displacement()
            max_stress = result.max_stress()
            energy = result.energy
            
            # Stress-strain (abbreviated)
            E_eff = 0
            try:
                strains, stresses = stress_strain_curve(
                    network, max_strain=0.02, num_steps=5, axis=0, segments_per_fiber=3
                )
                if len(strains) >= 2 and strains[-1] > 0 and stresses[-1] > 0:
                    E_eff = stresses[-1] / strains[-1]
            except:
                pass
            
            results[name] = {
                "n_fibers": n_fibers,
                "n_elements": fem.num_elements,
                "n_dof": fem.num_dof,
                "max_displacement_m": round(max_disp, 10),
                "max_stress_Pa": round(max_stress, 2),
                "strain_energy_J": round(energy, 8),
                "effective_modulus_Pa": round(E_eff, 2),
                "solve_time_ms": round(t_solve * 1000, 1),
            }
            
            rows.append([name[:25], n_fibers, fem.num_elements, f"{max_disp:.2e}",
                        f"{max_stress:.2e}", f"{energy:.2e}", f"{E_eff:.2e}",
                        f"{t_solve*1000:.0f}ms"])
            
        except Exception as e:
            print(f"    [!] {name}: {e}")
    
    print("\n  FEM Simulation Results:")
    print_table(
        ["Structure", "Fib", "Elem", "MaxDisp", "MaxStrs", "Energy", "E_eff", "Time"],
        rows,
        col_widths=[27, 5, 6, 12, 12, 12, 12, 8]
    )
    
    save_phase("phase4_mechanics", results)
    return results


# ============================================================
# Phase 5: ML Proxy Model
# ============================================================
def phase5_ml_proxy(feature_data, mechanics_data, structures):
    print("\n" + "="*70)
    print("  PHASE 5: ML Proxy Model (Feature → Property)")
    print("="*70)
    
    from fibernet.gen.regular import RegularNetworkGenerator
    from fibernet.analysis.graph_features import GraphFeatureExtractor
    from fibernet.core.material import Material
    from fibernet.sim.mechanical import FiberFEM
    from fibernet.graph.io import from_networkx
    
    ext = GraphFeatureExtractor(canvas_size=256, thick=5)
    material = Material(name="polymer", youngs_modulus=1e9, poissons_ratio=0.35, density=1200)
    
    print("  Generating parametric sweep dataset...")
    
    all_features = []
    all_targets = []
    param_grid = []
    
    np.random.seed(42)
    n_samples = 60
    
    for i in range(n_samples):
        dx = np.random.uniform(-0.4, 0.4)
        dy = np.random.uniform(-0.4, 0.4)
        tiling = np.random.choice([2, 3, 4])
        n_pts = np.random.choice([1, 2, 3])
        
        try:
            gen = RegularNetworkGenerator(
                side_length=10, num_points_per_side=n_pts,
                perturbations=[(dx, dy), (-dy, dx), (dx*0.5, dy*0.5)],
                tiling=tiling
            )
            G = gen.generate()
            
            features = ext.extract_vector(G)
            
            # Compute mechanical property via FEM
            try:
                net = from_networkx(G, material=material)
                if len(net.fibers) < 2:
                    continue
                fem = FiberFEM(net, segments_per_fiber=2)
                
                positions = np.array([cl.position for cl in net.crosslinks])
                x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
                L0 = x_max - x_min
                if L0 < 1e-6:
                    continue
                
                tol = 0.15 * L0
                left = [j for j, p in enumerate(positions) if p[0] <= x_min + tol]
                right = [j for j, p in enumerate(positions) if p[0] >= x_max - tol]
                
                if not left or not right:
                    continue
                
                F = np.zeros(fem.num_dof)
                for n in right:
                    F[n * 6] = 1.0
                
                result = fem.solve_static(forces=F, fixed_nodes=left)
                target = result.energy  # strain energy as target property
            except:
                target = 0.0
            
            if target > 0:
                all_features.append(features)
                all_targets.append(target)
                param_grid.append({"dx": round(dx, 3), "dy": round(dy, 3),
                                  "tiling": int(tiling), "n_pts": int(n_pts)})
            
        except Exception:
            continue
    
    X = np.array(all_features)
    y = np.array(all_targets)
    
    print(f"\n  Dataset: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Target (strain energy): mean={np.mean(y):.4e}, std={np.std(y):.4e}")
    print(f"  Target range: [{np.min(y):.4e}, {np.max(y):.4e}]")
    
    if X.shape[0] < 5:
        print("  [!] Insufficient data for ML")
        save_phase("phase5_ml", {"status": "insufficient_data"})
        return
    
    # Feature selection
    var = np.var(X, axis=0)
    active_mask = var > 1e-10
    X_active = X[:, active_mask]
    n_active = X_active.shape[1]
    print(f"  Active features: {n_active}/{X.shape[1]}")
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.metrics import r2_score, mean_absolute_error
    
    scaler_X = StandardScaler()
    X_norm = scaler_X.fit_transform(X_active)
    y_mean, y_std = y.mean(), y.std() + 1e-10
    y_norm = (y - y_mean) / y_std
    
    models = {
        "Ridge(α=1)": Ridge(alpha=1.0),
        "Lasso(α=0.01)": Lasso(alpha=0.01, max_iter=5000),
        "RF(100trees)": RandomForestRegressor(n_estimators=100, random_state=42),
        "GBR(100trees)": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    
    ml_results = {}
    rows = []
    
    for mname, model in models.items():
        try:
            cv_scores = cross_val_score(model, X_norm, y_norm, cv=min(5, X.shape[0]-1), scoring='r2')
            model.fit(X_norm, y_norm)
            y_pred = model.predict(X_norm)
            r2 = r2_score(y_norm, y_pred)
            mae = mean_absolute_error(y_norm, y_pred)
            
            ml_results[mname] = {
                "cv_r2": f"{np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}",
                "train_r2": round(r2, 4),
                "train_mae": round(mae, 4),
            }
            rows.append([mname, f"{np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}",
                        f"{r2:.4f}", f"{mae:.4f}"])
        except Exception as e:
            print(f"    [!] {mname}: {e}")
            rows.append([mname, "FAILED", "-", "-"])
    
    print("\n  ML Model Performance (strain energy prediction):")
    print_table(["Model", "CV-R²", "Train-R²", "MAE"], rows, col_widths=[20, 20, 12, 10])
    
    # Feature importance
    best_model = models["RF(100trees)"]
    best_model.fit(X_norm, y_norm)
    importance = best_model.feature_importances_
    
    active_indices = np.where(active_mask)[0]
    feat_names = ext.FEATURE_COLS
    top_idx = np.argsort(importance)[-10:][::-1]
    
    print("\n  Top 10 Important Features (RandomForest):")
    imp_rows = []
    for idx in top_idx:
        orig_idx = active_indices[idx]
        fname = feat_names[orig_idx] if orig_idx < len(feat_names) else f"f{orig_idx}"
        imp_rows.append([fname[:35], f"{importance[idx]:.4f}"])
    print_table(["Feature", "Importance"], imp_rows, col_widths=[37, 12])
    
    # Parametric sensitivity
    print("\n  Parametric Sensitivity Analysis:")
    tiling_groups = {}
    for i, p in enumerate(param_grid):
        if i < len(y):
            t = p["tiling"]
            tiling_groups.setdefault(t, []).append(y[i])
    
    sens_rows = []
    for t in sorted(tiling_groups.keys()):
        vals = tiling_groups[t]
        sens_rows.append([f"tiling={t}", len(vals), f"{np.mean(vals):.4e}",
                         f"{np.std(vals):.4e}", f"{np.min(vals):.4e}", f"{np.max(vals):.4e}"])
    
    print_table(["Group", "N", "Mean_E", "Std_E", "Min_E", "Max_E"],
                sens_rows, col_widths=[12, 5, 12, 12, 12, 12])
    
    save_phase("phase5_ml", {
        "models": ml_results,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_active_features": n_active,
        "target_stats": {"mean": float(y_mean), "std": float(y_std)},
        "param_grid_sample": param_grid[:10],
    })


# ============================================================
# Phase 6: RL Environment Demo
# ============================================================
def phase6_rl_demo():
    print("\n" + "="*70)
    print("  PHASE 6: RL Environment Demonstration")
    print("="*70)
    
    import networkx as nx
    from fibernet.gen.regular import RegularNetworkGenerator
    from fibernet.analysis.graph_features import GraphFeatureExtractor
    
    ext = GraphFeatureExtractor(canvas_size=256, thick=5)
    
    class FiberNetOptEnv:
        """RL environment: optimize perturbation to maximize structural quality.
        
        State: 10-dim (graph summary features)
        Action: 3-dim (perturbation deltas in [-0.1, 0.1])
        Reward: connectivity + edge density - perturbation penalty
        """
        def __init__(self):
            self.perturb = [0.0, 0.0, 0.0]
            self.step_count = 0
            self.max_steps = 20
            self.history = []
            
        def reset(self):
            self.perturb = [0.0, 0.0, 0.0]
            self.step_count = 0
            self.history = []
            return self._get_state()
        
        def _generate(self):
            gen = RegularNetworkGenerator(
                side_length=10, num_points_per_side=2,
                perturbations=[
                    (self.perturb[0], self.perturb[1]),
                    (-self.perturb[1], self.perturb[0]),
                    (self.perturb[2], -self.perturb[2]),
                ],
                tiling=3
            )
            return gen.generate()
        
        def _get_state(self):
            G = self._generate()
            n_n = G.number_of_nodes()
            n_e = G.number_of_edges()
            avg_deg = sum(dict(G.degree()).values()) / max(n_n, 1)
            density = nx.density(G)
            components = nx.number_connected_components(G)
            try:
                feats = ext.extract_vector(G)[:5]
            except:
                feats = np.zeros(5)
            return np.array([n_n/100, n_e/200, avg_deg/4, density*10, components/10, *feats])
        
        def _reward(self, G):
            n_n = G.number_of_nodes()
            n_e = G.number_of_edges()
            avg_deg = sum(dict(G.degree()).values()) / max(n_n, 1)
            components = nx.number_connected_components(G)
            density = nx.density(G)
            # Reward: connectivity (fewer components), higher density, higher avg degree
            # Normalize: components ratio, density bonus, degree bonus
            comp_penalty = max(0, components - 1) * 5.0  # penalize disconnected
            reward = avg_deg * 2.0 + density * 500 - comp_penalty
            penalty = sum(abs(p) for p in self.perturb) * 0.5
            return reward - penalty
        
        def step(self, action):
            action = np.clip(action, -0.1, 0.1)
            self.perturb = [
                np.clip(self.perturb[0] + action[0], -0.5, 0.5),
                np.clip(self.perturb[1] + action[1], -0.5, 0.5),
                np.clip(self.perturb[2] + action[2], -0.5, 0.5),
            ]
            self.step_count += 1
            G = self._generate()
            reward = self._reward(G)
            done = self.step_count >= self.max_steps
            state = self._get_state()
            self.history.append({
                "step": self.step_count,
                "perturb": [round(p, 4) for p in self.perturb],
                "reward": round(reward, 4),
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
            })
            return state, reward, done, {}
    
    print("\n  Environment: FiberNet perturbation optimization")
    print("  State dim: 10, Action dim: 3, Max steps: 20")
    print("  Goal: Maximize connectivity + edge density via perturbation tuning")
    
    env = FiberNetOptEnv()
    np.random.seed(42)
    
    all_episodes = []
    
    for episode in range(5):
        state = env.reset()
        total_reward = 0
        step_rewards = []
        
        # Epsilon-greedy exploration with decay
        epsilon = 1.0 / (episode + 1)
        
        for step in range(env.max_steps):
            magnitude = 0.08 * (1 - step / env.max_steps)
            if np.random.random() < epsilon:
                action = np.random.uniform(-magnitude, magnitude, size=3)
            else:
                # Greedy: try small positive perturbation in best direction
                action = np.random.uniform(-magnitude/2, magnitude/2, size=3)
            
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            step_rewards.append(reward)
            state = next_state
            if done:
                break
        
        all_episodes.append({
            "episode": episode + 1,
            "total_reward": round(total_reward, 4),
            "avg_reward": round(total_reward / len(step_rewards), 4),
            "max_reward": round(max(step_rewards), 4),
            "min_reward": round(min(step_rewards), 4),
            "final_perturb": [round(p, 4) for p in env.perturb],
            "steps": len(step_rewards),
            "trajectory": env.history.copy(),
        })
        
        print(f"\n  Episode {episode+1}:")
        print(f"    Total reward: {total_reward:.4f} | Steps: {len(step_rewards)}")
        print(f"    Reward trajectory: {step_rewards[0]:.4f} → {step_rewards[-1]:.4f}")
        print(f"    Final perturbation: [{env.perturb[0]:.4f}, {env.perturb[1]:.4f}, {env.perturb[2]:.4f}]")
    
    # Summary
    print(f"\n  Episode Summary:")
    print_table(
        ["Episode", "TotalRwd", "AvgRwd", "MaxRwd", "Steps", "FinalPerturb"],
        [[f"Ep{e['episode']}", f"{e['total_reward']:.4f}", f"{e['avg_reward']:.4f}",
          f"{e['max_reward']:.4f}", e['steps'],
          f"[{e['final_perturb'][0]:.3f},{e['final_perturb'][1]:.3f},{e['final_perturb'][2]:.3f}]"]
         for e in all_episodes],
        col_widths=[10, 12, 12, 12, 8, 35]
    )
    
    save_phase("phase6_rl", {
        "env": {"state_dim": 10, "action_dim": 3, "max_steps": 20},
        "episodes": all_episodes,
    })


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="FiberNet v2.0 Comprehensive Analysis")
    parser.add_argument("--phase", type=int, default=0, help="Run specific phase (0=all)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    print("="*70)
    print("  FiberNet v2.0 Comprehensive Data Analysis Pipeline")
    print("  Structural Diversity | Mechanics | ML | RL")
    print("="*70)
    print(f"  Output: {OUTPUT_DIR}")
    
    t_start = time.time()
    structures = None
    
    # Phase 1
    if args.phase in [0, 1] and (not args.resume or not is_phase_done("phase1_structural_diversity")):
        structures = phase1_structural_diversity()
    
    if structures is None and args.phase in [0, 2, 3, 4, 5]:
        print("  Re-generating structures for later phases...")
        from fibernet.gen.regular import RegularNetworkGenerator
        from fibernet.gen.zigzag import ZigZagGenerator
        structures = {}
        for p in [0.0, 0.1, 0.3, 0.5]:
            gen = RegularNetworkGenerator(
                side_length=10, num_points_per_side=2,
                perturbations=[(p, -p), (p, p), (-p, p)] if p > 0 else [],
                tiling=3
            )
            structures[f"regular_p{p:.1f}"] = gen.generate()
        for mx, my in [(True, True), (True, False), (False, True)]:
            gen = ZigZagGenerator(n_cols=3, n_rows=5, mirror_x=mx, mirror_y=my)
            structures[f"zigzag_mx{mx}_my{my}"] = gen.generate()
    
    # Phase 2
    if args.phase in [0, 2] and (not args.resume or not is_phase_done("phase2_weld_analysis")):
        if structures:
            phase2_weld_analysis(structures)
    
    # Phase 3
    feature_data = None
    if args.phase in [0, 3] and (not args.resume or not is_phase_done("phase3_features")):
        if structures:
            feature_data = phase3_feature_extraction(structures)
    
    # Phase 4
    mechanics_data = None
    if args.phase in [0, 4] and (not args.resume or not is_phase_done("phase4_mechanics")):
        if structures:
            mechanics_data = phase4_mechanics(structures)
    
    # Phase 5
    if args.phase in [0, 5] and (not args.resume or not is_phase_done("phase5_ml")):
        if structures:
            phase5_ml_proxy(feature_data or {}, mechanics_data or {}, structures)
    
    # Phase 6
    if args.phase in [0, 6] and (not args.resume or not is_phase_done("phase6_rl")):
        phase6_rl_demo()
    
    elapsed = time.time() - t_start
    print("\n" + "="*70)
    print(f"  Analysis Complete! Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            print(f"    {f.name:40s} ({size:,} bytes)")
    print("="*70)


if __name__ == "__main__":
    main()

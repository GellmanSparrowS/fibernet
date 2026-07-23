"""
Phase 6: Comprehensive Beam FEM Validation
============================================
Tests the corrected solver on real fiber network structures:
1. Deformed structures (pattern_2d, 5 pts/side, ±0.4 amplitude)
2. Large deformation (10x10cm → stretch 10cm, compress 5cm)
3. Deformation propagation analysis
4. Multi-radius fiber tests
5. 3D complex structures
6. Graph-level physics analysis
7. All-in-one visualization

Features:
- Checkpoint/resume for long runs
- Memory-safe (process structures one at a time)
- JSON results saved incrementally
"""
import sys, json, time, gc, os
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fibernet import pattern_2d
from fibernet.ml.gnn import graph_from_structure
from fibernet.ml.beam_frame_fem import BeamFrameFEM

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_FILE = RESULTS_DIR / "phase6_checkpoint.json"
FINAL_FILE = RESULTS_DIR / "phase6_comprehensive_results.json"

def save_checkpoint(data, step_name):
    data['_last_step'] = step_name
    data['_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=_json_default)

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {'_last_step': 'start'}

def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

def deduplicate_edge_count(edge_index):
    seen = set()
    for e in range(edge_index.shape[1]):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        seen.add((min(i,j), max(i,j)))
    return len(seen)

# ============================================================
# TEST 1: Deformed Structures from pattern_2d
# ============================================================
def test_deformed_structures():
    print("\n" + "="*70)
    print("TEST 1: Deformed Structures (5 pts/side, ±0.4 amplitude)")
    print("="*70)
    
    results = {}
    disp_5 = [(0.4, 0.0), (-0.3, 0.2), (0.0, -0.4), (0.4, 0.4), (-0.4, -0.3)]
    
    for unit in ['honeycomb', 'kagome', 'reentrant', 'triangle', 'square']:
        print(f"\n  [{unit}]")
        try:
            g = pattern_2d(unit=unit, box=(10, 10), grid=(5, 5),
                          n_pts_per_side=5, point_displacements=disp_5)
            gd = graph_from_structure(g)
            node_pos = gd['node_features'].numpy()[:, :2]
            edge_index = gd['edge_index'].numpy()
            n_nodes = node_pos.shape[0]
            n_edges = deduplicate_edge_count(edge_index)
            
            print(f"    Nodes: {n_nodes}, Unique edges: {n_edges}")
            print(f"    Position range: x=[{node_pos[:,0].min():.2f}, {node_pos[:,0].max():.2f}], y=[{node_pos[:,1].min():.2f}, {node_pos[:,1].max():.2f}]")
            
            r = 0.01
            radii = np.full(edge_index.shape[1], r)
            solver = BeamFrameFEM(E=1e9, nu=0.3)
            
            x = node_pos[:, 0]
            y = node_pos[:, 1]
            tol_x = (x.max() - x.min()) * 0.05
            tol_y = (y.max() - y.min()) * 0.05
            
            left = np.where(np.abs(x - x.min()) < tol_x)[0].tolist()
            right = np.where(np.abs(x - x.max()) < tol_x)[0].tolist()
            bottom = np.where(np.abs(y - y.min()) < tol_y)[0].tolist()
            top = np.where(np.abs(y - y.max()) < tol_y)[0].tolist()
            
            # Stretch test: right edge +10%
            stretch = (x.max() - x.min()) * 0.1
            prescribed = {int(n): (stretch, 0.0) for n in right}
            res = solver.solve_2d(edge_index, node_pos, radii,
                                 fixed_nodes=left, prescribed_disp=prescribed)
            
            disps = np.linalg.norm(res['u'][:, :2], axis=1)
            max_d = disps.max()
            max_s = res['sigma_total'].max()
            
            # Propagation: ratio of far-side to near-side displacement
            far_d = disps[left].mean() if left else 0
            near_d = disps[right].mean() if right else 1
            prop = far_d / near_d if near_d > 1e-15 else 0
            
            results[unit] = {
                'n_nodes': n_nodes, 'n_edges': n_edges,
                'max_disp': float(max_d), 'max_stress': float(max_s),
                'propagation': float(prop),
                'boundary_counts': {'left': len(left), 'right': len(right),
                                    'bottom': len(bottom), 'top': len(top)},
                'node_pos_range': {'x_min': float(x.min()), 'x_max': float(x.max()),
                                   'y_min': float(y.min()), 'y_max': float(y.max())}
            }
            print(f"    Max disp: {max_d:.4e}, Max stress: {max_s:.4e} Pa")
            print(f"    Propagation: {prop:.1%}")
            
        except Exception as ex:
            print(f"    ERROR: {ex}")
            results[unit] = {'error': str(ex)}
    
    return results

# ============================================================
# TEST 2: Large Deformation (10x10cm → stretch/compress)
# ============================================================
def test_large_deformation():
    print("\n" + "="*70)
    print("TEST 2: Large Deformation (10x10cm structure)")
    print("="*70)
    
    results = {}
    L = 10.0  # 10cm = 0.1m (using cm units for clarity)
    E = 1e9   # 1 GPa
    
    for unit in ['honeycomb', 'kagome', 'triangle', 'square']:
        print(f"\n  [{unit}] Generating 10x10cm structure...")
        try:
            g = pattern_2d(unit=unit, box=(L, L), grid=(6, 6))
            gd = graph_from_structure(g)
            node_pos = gd['node_features'].numpy()[:, :2]
            edge_index = gd['edge_index'].numpy()
            n_nodes = node_pos.shape[0]
            n_edges = deduplicate_edge_count(edge_index)
            
            print(f"    {n_nodes} nodes, {n_edges} edges")
            
            x = node_pos[:, 0]
            y = node_pos[:, 1]
            tol_x = (x.max() - x.min()) * 0.05
            tol_y = (y.max() - y.min()) * 0.05
            
            left = np.where(np.abs(x - x.min()) < tol_x)[0].tolist()
            right = np.where(np.abs(x - x.max()) < tol_x)[0].tolist()
            bottom = np.where(np.abs(y - y.min()) < tol_y)[0].tolist()
            top = np.where(np.abs(y - y.max()) < tol_y)[0].tolist()
            
            results[unit] = {'n_nodes': n_nodes, 'n_edges': n_edges, 'tests': {}}
            
            # Test A: Stretch right by 5cm (50% strain)
            print(f"    Test A: Stretch right by 5cm (50%)...")
            r = 0.01
            radii = np.full(edge_index.shape[1], r)
            solver = BeamFrameFEM(E=E, nu=0.3)
            
            stretch_50 = L * 0.5
            prescribed_stretch = {int(n): (stretch_50, 0.0) for n in right}
            
            res_a = solver.solve_2d(edge_index, node_pos, radii,
                                    fixed_nodes=left, prescribed_disp=prescribed_stretch)
            
            disps_a = np.linalg.norm(res_a['u'][:, :2], axis=1)
            results[unit]['tests']['stretch_50pct'] = {
                'max_disp': float(disps_a.max()),
                'max_axial_stress': float(np.max(np.abs(res_a['sigma_axial']))),
                'max_bending_stress': float(np.max(res_a['sigma_bending'])),
                'max_total_stress': float(np.max(res_a['sigma_total'])),
                'mean_disp': float(disps_a.mean()),
            }
            print(f"      Max disp: {disps_a.max():.4e}, Max stress: {res_a['sigma_total'].max():.4e}")
            
            # Test B: Compress top by 5cm (50% strain)
            print(f"    Test B: Compress top by 5cm (50%)...")
            compress_50 = -L * 0.5
            prescribed_compress = {int(n): (0.0, compress_50) for n in top}
            
            res_b = solver.solve_2d(edge_index, node_pos, radii,
                                    fixed_nodes=bottom, prescribed_disp=prescribed_compress)
            
            disps_b = np.linalg.norm(res_b['u'][:, :2], axis=1)
            results[unit]['tests']['compress_50pct'] = {
                'max_disp': float(disps_b.max()),
                'max_axial_stress': float(np.max(np.abs(res_b['sigma_axial']))),
                'max_bending_stress': float(np.max(res_b['sigma_bending'])),
                'max_total_stress': float(np.max(res_b['sigma_total'])),
            }
            print(f"      Max disp: {disps_b.max():.4e}, Max stress: {res_b['sigma_total'].max():.4e}")
            
            # Test C: Combined stretch + compress (biaxial)
            print(f"    Test C: Biaxial (stretch x + compress y)...")
            prescribed_biaxial = {}
            for n in right:
                prescribed_biaxial[int(n)] = (stretch_50, 0.0)
            for n in top:
                if int(n) not in prescribed_biaxial:
                    prescribed_biaxial[int(n)] = (0.0, compress_50)
                else:
                    prescribed_biaxial[int(n)] = (stretch_50, compress_50)
            
            fixed_biaxial = list(set(left + bottom))
            
            res_c = solver.solve_2d(edge_index, node_pos, radii,
                                    fixed_nodes=fixed_biaxial,
                                    prescribed_disp=prescribed_biaxial)
            
            disps_c = np.linalg.norm(res_c['u'][:, :2], axis=1)
            results[unit]['tests']['biaxial_50pct'] = {
                'max_disp': float(disps_c.max()),
                'max_stress': float(np.max(res_c['sigma_total'])),
            }
            print(f"      Max disp: {disps_c.max():.4e}, Max stress: {res_c['sigma_total'].max():.4e}")
            
            gc.collect()
            
        except Exception as ex:
            print(f"    ERROR: {ex}")
            import traceback; traceback.print_exc()
            results[unit] = {'error': str(ex)}
    
    return results

# ============================================================
# TEST 3: Multi-Radius Tests
# ============================================================
def test_multi_radius():
    print("\n" + "="*70)
    print("TEST 3: Multi-Radius Fiber Tests")
    print("="*70)
    
    results = {}
    radii_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    
    for unit in ['honeycomb', 'kagome', 'triangle']:
        print(f"\n  [{unit}]")
        g = pattern_2d(unit=unit, box=(10, 10), grid=(5, 5))
        gd = graph_from_structure(g)
        node_pos = gd['node_features'].numpy()[:, :2]
        edge_index = gd['edge_index'].numpy()
        
        x = node_pos[:, 0]
        y = node_pos[:, 1]
        tol = (x.max() - x.min()) * 0.05
        left = np.where(np.abs(x - x.min()) < tol)[0].tolist()
        right = np.where(np.abs(x - x.max()) < tol)[0].tolist()
        
        stretch = (x.max() - x.min()) * 0.1
        prescribed = {int(n): (stretch, 0.0) for n in right}
        
        solver = BeamFrameFEM(E=1e9, nu=0.3)
        results[unit] = {}
        
        print(f"    {'r':>8s} {'max_u':>12s} {'sigma_ax':>12s} {'sigma_bend':>12s} {'sigma_tot':>12s}")
        for r in radii_values:
            radii = np.full(edge_index.shape[1], r)
            res = solver.solve_2d(edge_index, node_pos, radii,
                                 fixed_nodes=left, prescribed_disp=prescribed)
            
            max_u = np.max(np.linalg.norm(res['u'][:, :2], axis=1))
            max_sa = np.max(np.abs(res['sigma_axial']))
            max_sb = np.max(res['sigma_bending'])
            max_st = np.max(res['sigma_total'])
            
            results[unit][f'r{r}'] = {
                'max_disp': float(max_u),
                'sigma_axial': float(max_sa),
                'sigma_bending': float(max_sb),
                'sigma_total': float(max_st)
            }
            print(f"    {r:8.4f} {max_u:12.4e} {max_sa:12.4e} {max_sb:12.4e} {max_st:12.4e}")
    
    return results

# ============================================================
# TEST 4: 3D Structures
# ============================================================
def test_3d_structures():
    print("\n" + "="*70)
    print("TEST 4: 3D Complex Structures")
    print("="*70)
    
    results = {}
    
    def make_cube_lattice(nx, ny, nz, spacing=1.0):
        """Create a 3D cube lattice."""
        nodes = []
        node_map = {}
        idx = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    nodes.append([i*spacing, j*spacing, k*spacing])
                    node_map[(i,j,k)] = idx
                    idx += 1
        
        edges = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    n = node_map[(i,j,k)]
                    if i < nx-1:
                        m = node_map[(i+1,j,k)]
                        edges.extend([[n,m],[m,n]])
                    if j < ny-1:
                        m = node_map[(i,j+1,k)]
                        edges.extend([[n,m],[m,n]])
                    if k < nz-1:
                        m = node_map[(i,j,k+1)]
                        edges.extend([[n,m],[m,n]])
        
        return np.array(nodes), np.array(edges).T
    
    for name, (nx, ny, nz) in [('3x3x3', (3,3,3)), ('5x5x5', (5,5,5)), ('4x4x6', (4,4,6))]:
        print(f"\n  [{name}] Building 3D lattice...")
        node_pos, edge_index = make_cube_lattice(nx, ny, nz)
        n_nodes = node_pos.shape[0]
        n_edges = deduplicate_edge_count(edge_index)
        print(f"    {n_nodes} nodes, {n_edges} edges")
        
        r = 0.01
        radii = np.full(edge_index.shape[1], r)
        solver = BeamFrameFEM(E=1e9, nu=0.3)
        
        # Fix bottom face (z=0)
        z = node_pos[:, 2]
        bottom = np.where(z < 0.01)[0].tolist()
        top = np.where(z > (nz-1)*0.99)[0].tolist()
        
        # Apply displacement on top face
        disp_z = -0.5  # compress by 0.5 units
        prescribed = {int(n): (0.0, 0.0, disp_z) for n in top}
        
        try:
            res = solver.solve_3d(edge_index, node_pos, radii,
                                  fixed_nodes=bottom, prescribed_disp=prescribed)
            
            u_trans = np.linalg.norm(res['u'][:, :3], axis=1)
            results[name] = {
                'n_nodes': n_nodes, 'n_edges': n_edges,
                'max_disp': float(u_trans.max()),
                'mean_disp': float(u_trans.mean()),
                'max_axial_stress': float(np.max(np.abs(res['sigma_axial']))),
                'max_bending_stress': float(np.max(res['sigma_bending'])),
                'max_total_stress': float(np.max(res['sigma_total'])),
                'node_pos': node_pos.tolist(),
                'edge_index': edge_index.tolist(),
                'displacement': res['u'][:, :3].tolist(),
                'node_stress': res['node_stress'].tolist()
            }
            print(f"    Max disp: {u_trans.max():.4e}")
            print(f"    Max stress: {res['sigma_total'].max():.4e} Pa")
            
        except Exception as ex:
            print(f"    ERROR: {ex}")
            import traceback; traceback.print_exc()
            results[name] = {'error': str(ex)}
        
        gc.collect()
    
    return results

# ============================================================
# TEST 5: Graph-Level Physics Analysis
# ============================================================
def test_graph_physics():
    print("\n" + "="*70)
    print("TEST 5: Graph-Level Physics Analysis")
    print("="*70)
    
    import networkx as nx
    
    results = {}
    
    for unit in ['honeycomb', 'kagome', 'triangle', 'square', 'reentrant']:
        print(f"\n  [{unit}]")
        g = pattern_2d(unit=unit, box=(10, 10), grid=(5, 5))
        gd = graph_from_structure(g)
        node_pos = gd['node_features'].numpy()[:, :2]
        edge_index = gd['edge_index'].numpy()
        n_nodes = node_pos.shape[0]
        
        # Build networkx graph for analysis
        G = nx.Graph()
        for e in range(edge_index.shape[1]):
            i, j = int(edge_index[0, e]), int(edge_index[1, e])
            if i < j:
                L = np.linalg.norm(node_pos[j] - node_pos[i])
                G.add_edge(i, j, length=L)
        
        # Graph metrics
        degrees = dict(G.degree())
        degree_values = list(degrees.values())
        
        # Connectivity
        is_connected = nx.is_connected(G)
        
        # Betweenness centrality (stress path indicator)
        betweenness = nx.betweenness_centrality(G, weight='length')
        
        # Spectral gap (algebraic connectivity)
        L_matrix = nx.laplacian_matrix(G).toarray()
        eigenvalues = np.sort(np.linalg.eigvalsh(L_matrix.astype(float)))
        spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 0
        
        results[unit] = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'is_connected': is_connected,
            'avg_degree': float(np.mean(degree_values)),
            'min_degree': int(min(degree_values)),
            'max_degree': int(max(degree_values)),
            'degree_distribution': {str(k): int(v) for k, v in 
                np.unique(degree_values, return_counts=True).__iter__()}
            if hasattr(np.unique(degree_values, return_counts=True), '__iter__') else {},
            'spectral_gap': float(spectral_gap),
            'avg_betweenness': float(np.mean(list(betweenness.values()))),
            'max_betweenness': float(max(betweenness.values())),
            'diameter': nx.diameter(G) if is_connected else -1,
            'clustering_coeff': float(nx.average_clustering(G)),
        }
        
        # FEM stress path analysis
        r = 0.01
        radii = np.full(edge_index.shape[1], r)
        solver = BeamFrameFEM(E=1e9, nu=0.3)
        
        x = node_pos[:, 0]
        tol = (x.max() - x.min()) * 0.05
        left = np.where(np.abs(x - x.min()) < tol)[0].tolist()
        right = np.where(np.abs(x - x.max()) < tol)[0].tolist()
        
        stretch = (x.max() - x.min()) * 0.1
        prescribed = {int(n): (stretch, 0.0) for n in right}
        
        res = solver.solve_2d(edge_index, node_pos, radii,
                             fixed_nodes=left, prescribed_disp=prescribed)
        
        # Stress path: identify high-stress edges (force chains)
        edge_stress = res['sigma_total']
        edge_list = res['edge_list']
        mean_stress = edge_stress.mean()
        high_stress_mask = edge_stress > 2 * mean_stress
        
        n_high = np.sum(high_stress_mask)
        n_total = len(edge_stress)
        
        results[unit]['stress_path'] = {
            'mean_stress': float(mean_stress),
            'max_stress': float(edge_stress.max()),
            'stress_concentration_factor': float(edge_stress.max() / mean_stress) if mean_stress > 0 else 0,
            'n_high_stress_edges': int(n_high),
            'high_stress_fraction': float(n_high / n_total) if n_total > 0 else 0,
        }
        
        print(f"    Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print(f"    Avg degree: {np.mean(degree_values):.1f}, Spectral gap: {spectral_gap:.4f}")
        print(f"    Connected: {is_connected}, Diameter: {nx.diameter(G) if is_connected else 'N/A'}")
        print(f"    SCF: {edge_stress.max()/mean_stress:.2f}" if mean_stress > 0 else "    SCF: N/A")
        print(f"    High-stress edges: {n_high}/{n_total} ({n_high/n_total:.0%})" if n_total > 0 else "")
    
    return results

# ============================================================
# MAIN
# ============================================================
def main():
    print("="*70)
    print("Phase 6: Comprehensive Beam FEM Validation")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    checkpoint = load_checkpoint()
    all_results = {}
    
    # Check what's already done
    done_steps = set()
    if '_completed' in checkpoint:
        done_steps = set(checkpoint['_completed'])
        all_results = {k: v for k, v in checkpoint.items() if not k.startswith('_')}
    
    steps = [
        ('deformed_structures', test_deformed_structures),
        ('large_deformation', test_large_deformation),
        ('multi_radius', test_multi_radius),
        ('3d_structures', test_3d_structures),
        ('graph_physics', test_graph_physics),
    ]
    
    for step_name, step_func in steps:
        if step_name in done_steps:
            print(f"\n  Skipping {step_name} (already done)")
            continue
        
        print(f"\n>>> Running {step_name}...")
        t0 = time.time()
        try:
            result = step_func()
            all_results[step_name] = result
            done_steps.add(step_name)
            elapsed = time.time() - t0
            print(f"  {step_name} completed in {elapsed:.1f}s")
        except Exception as ex:
            print(f"  {step_name} FAILED: {ex}")
            import traceback; traceback.print_exc()
            all_results[step_name] = {'error': str(ex)}
            done_steps.add(step_name)
        
        # Save checkpoint after each step
        save_data = dict(all_results)
        save_data['_completed'] = list(done_steps)
        save_checkpoint(save_data, step_name)
        gc.collect()
    
    # Save final results
    all_results['_completed'] = list(done_steps)
    all_results['_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(FINAL_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    
    print(f"\n{'='*70}")
    print(f"All tests completed!")
    print(f"Results: {FINAL_FILE}")
    print(f"{'='*70}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

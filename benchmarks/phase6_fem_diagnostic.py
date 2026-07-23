"""Phase 6: FEM Diagnostic - identify issues with current beam FEM."""
import sys, json, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fibernet.ml.beam_frame_fem_sparse import SparseBeamFrameFEM

def test_cantilever_analytical():
    print("\n" + "="*70)
    print("TEST 1: Cantilever Beam Analytical Validation")
    print("="*70)
    E, nu, L_total, r = 200e9, 0.3, 1.0, 0.01
    n_nodes, P = 11, 1000.0
    node_pos = np.zeros((n_nodes, 2))
    node_pos[:, 0] = np.linspace(0, L_total, n_nodes)
    edges = []
    for i in range(n_nodes - 1):
        edges.extend([[i, i+1], [i+1, i]])
    edge_index = np.array(edges).T
    radii = np.full(edge_index.shape[1], r)
    solver = SparseBeamFrameFEM(E=E, nu=nu)
    forces = np.zeros((n_nodes, 2))
    forces[-1, 1] = -P
    u, sigma, moments, edge_list = solver.solve_2d(edge_index, node_pos, radii, forces, [0])
    I_val = np.pi * r**4 / 4
    delta_analytical = P * L_total**3 / (3 * E * I_val)
    delta_fem = abs(u[-1, 1])
    theta_analytical = P * L_total**2 / (2 * E * I_val)
    theta_fem = abs(u[-1, 2])
    print(f"  delta_tip analytical = {delta_analytical:.6e} m")
    print(f"  delta_tip FEM        = {delta_fem:.6e} m")
    print(f"  Ratio = {delta_fem/delta_analytical:.6f}")
    print(f"  theta_tip analytical = {theta_analytical:.6e} rad")
    print(f"  theta_tip FEM        = {theta_fem:.6e} rad")
    print(f"  Ratio = {theta_fem/theta_analytical:.6f}")
    x = node_pos[:, 0]
    delta_prof = P * x**2 * (3*L_total - x) / (6 * E * I_val)
    max_err = np.max(np.abs(-u[:, 1] - delta_prof)) / delta_analytical
    print(f"  Max profile error = {max_err:.4%}")
    ok = abs(delta_fem/delta_analytical - 1.0) < 0.05
    print(f"  Verdict: {'PASS' if ok else 'FAIL'}")
    return {'delta_ratio': delta_fem/delta_analytical, 'theta_ratio': theta_fem/theta_analytical, 'max_profile_error': max_err}

def test_portal_frame():
    print("\n" + "="*70)
    print("TEST 2: Portal Frame Welded Joint Moment Transfer")
    print("="*70)
    E, nu, r, h, w = 200e9, 0.3, 0.02, 2.0, 3.0
    node_pos = np.array([[0, 0], [0, h], [w, h], [w, 0]])
    edges = [[0,1],[1,0],[1,2],[2,1],[2,3],[3,2]]
    edge_index = np.array(edges).T
    radii = np.full(edge_index.shape[1], r)
    solver = SparseBeamFrameFEM(E=E, nu=nu)
    forces = np.zeros((4, 2))
    forces[1, 0] = 10000.0
    u, sigma, moments, edge_list = solver.solve_2d(edge_index, node_pos, radii, forces, [0, 3])
    print(f"  Node displacements:")
    for i in range(4):
        print(f"    Node {i}: ux={u[i,0]:.6e}, uy={u[i,1]:.6e}, theta={u[i,2]:.6e}")
    print(f"  Edge stresses and moments:")
    for idx, e in enumerate(edge_list):
        i, j = edge_index[0, e], edge_index[1, e]
        print(f"    Edge {i}-{j}: sigma={sigma[idx]:.4e}, M=[{moments[idx,0]:.4e}, {moments[idx,1]:.4e}]")
    has_moment = abs(u[1, 2]) > 1e-10
    print(f"  Moment transfer at joint: {'YES' if has_moment else 'NO'}")
    return {'theta_1': u[1,2], 'moment_transfer': has_moment}

def test_propagation():
    print("\n" + "="*70)
    print("TEST 3: Deformation Propagation in 6x6 Grid")
    print("="*70)
    E, nu, r, n, spacing = 1e9, 0.3, 0.01, 6, 1.0
    node_pos = np.array([[i*spacing, j*spacing] for j in range(n) for i in range(n)])
    n_nodes = len(node_pos)
    edges = []
    for j in range(n):
        for i in range(n):
            node = j * n + i
            if i < n-1: edges.extend([[node, node+1], [node+1, node]])
            if j < n-1: edges.extend([[node, node+n], [node+n, node]])
    edge_index = np.array(edges).T
    radii = np.full(edge_index.shape[1], r)
    solver = SparseBeamFrameFEM(E=E, nu=nu)
    left_nodes = [j * n for j in range(n)]
    right_nodes = [j * n + (n-1) for j in range(n)]
    forces = np.zeros((n_nodes, 2))
    for rn in right_nodes: forces[rn, 0] = 100.0
    u, sigma, moments, edge_list = solver.solve_2d(edge_index, node_pos, radii, forces, left_nodes)
    print(f"  Middle row displacements (y={n//2}):")
    for i in range(n):
        node = (n//2) * n + i
        print(f"    x={i*spacing:.1f}: ux={u[node,0]:.6e}, uy={u[node,1]:.6e}")
    mid_disps = [abs(u[j*n + n//2, 0]) for j in range(n)]
    edge_disps = [abs(u[j*n + n-1, 0]) for j in range(n)]
    prop = np.mean(mid_disps) / np.mean(edge_disps) if np.mean(edge_disps) > 0 else 0
    print(f"  Propagation ratio: {prop:.4f} ({prop:.1%})")
    forces2 = np.zeros((n_nodes, 2))
    target = n//2 * n + n-1
    forces2[target, 0] = 1000.0
    u2, _, _, _ = solver.solve_2d(edge_index, node_pos, radii, forces2, left_nodes)
    dmags = np.sqrt(u2[:,0]**2 + u2[:,1]**2)
    n_affected = np.sum(dmags > dmags.max() * 0.01)
    print(f"  Concentrated load: {n_affected}/{n_nodes} nodes affected ({n_affected/n_nodes:.0%})")
    print(f"  Disp field (magnitude):")
    for j in range(n-1, -1, -1):
        row = ""
        for i in range(n):
            val = dmags[j*n+i]
            if val > dmags.max()*0.5: row += "X "
            elif val > dmags.max()*0.1: row += "+ "
            elif val > dmags.max()*0.01: row += ". "
            else: row += "  "
        print(f"    y={j}: {row}")
    return {'propagation': prop, 'n_affected': int(n_affected), 'n_total': n_nodes}

def test_radius_scaling():
    print("\n" + "="*70)
    print("TEST 4: Fiber Radius Scaling Laws")
    print("="*70)
    E, nu, L, n_nodes, P = 200e9, 0.3, 1.0, 11, 1000.0
    node_pos = np.zeros((n_nodes, 2))
    node_pos[:, 0] = np.linspace(0, L, n_nodes)
    edges = []
    for i in range(n_nodes-1): edges.extend([[i,i+1],[i+1,i]])
    edge_index = np.array(edges).T
    solver = SparseBeamFrameFEM(E=E, nu=nu)
    radii_vals = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    results = []
    print(f"  {'r':>8s} {'delta':>12s} {'sigma':>12s} {'d_ratio':>10s} {'s_ratio':>10s}")
    prev_d, prev_s, prev_r = None, None, None
    for r in radii_vals:
        radii = np.full(edge_index.shape[1], r)
        forces = np.zeros((n_nodes, 2))
        forces[-1, 1] = -P
        u, sigma, moments, _ = solver.solve_2d(edge_index, node_pos, radii, forces, [0])
        d = abs(u[-1, 1])
        s = np.max(np.abs(sigma))
        dr = f"{d/prev_d / (prev_r/r)**4:.3f}" if prev_d else "-"
        sr = f"{s/prev_s / (prev_r/r)**2:.3f}" if prev_s else "-"
        print(f"  {r:8.4f} {d:12.4e} {s:12.4e} {dr:>10s} {sr:>10s}")
        results.append({'r': r, 'delta': d, 'sigma': s})
        prev_d, prev_s, prev_r = d, s, r
    return results

def main():
    print("="*70)
    print("Phase 6: FEM Diagnostic Tests")
    print("="*70)
    r1 = test_cantilever_analytical()
    r2 = test_portal_frame()
    r3 = test_propagation()
    r4 = test_radius_scaling()
    output = Path(__file__).parent / "results" / "phase6_diagnostic.json"
    output.parent.mkdir(exist_ok=True)
    with open(output, 'w') as f:
        json.dump({'cantilever': r1, 'portal_frame': r2, 'propagation': r3, 'radius': r4}, f, indent=2, default=str)
    print(f"\nResults: {output}")

if __name__ == '__main__':
    main()

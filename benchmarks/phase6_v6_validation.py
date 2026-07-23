"""Validate v6 solver against analytical solutions and v4 results."""
import sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fibernet.ml.beam_frame_fem_v6 import BeamFrameFEM_v6

def test_cantilever():
    print("="*70)
    print("V6 TEST 1: Cantilever with Bending Stress")
    print("="*70)
    E, nu, L, r = 200e9, 0.3, 1.0, 0.01
    n = 11
    P = 1000.0
    node_pos = np.zeros((n, 2))
    node_pos[:, 0] = np.linspace(0, L, n)
    edges = []
    for i in range(n-1): edges.extend([[i,i+1],[i+1,i]])
    edge_index = np.array(edges).T
    radii = np.full(edge_index.shape[1], r)
    forces = np.zeros((n, 2))
    forces[-1, 1] = -P
    
    solver = BeamFrameFEM_v6(E=E, nu=nu)
    res = solver.solve_2d(edge_index, node_pos, radii, forces=forces, fixed_nodes=[0])
    
    I_val = np.pi * r**4 / 4
    delta_ana = P * L**3 / (3 * E * I_val)
    delta_fem = abs(res['u'][-1, 1])
    
    # Analytical max bending stress: sigma_max = M_max * c / I = (P*L) * r / I
    M_max = P * L  # at fixed end
    sigma_bending_ana = M_max * r / I_val
    
    print(f"  Tip displacement:")
    print(f"    Analytical: {delta_ana:.6e} m")
    print(f"    FEM:        {delta_fem:.6e} m")
    print(f"    Ratio:      {delta_fem/delta_ana:.6f} {'PASS' if abs(delta_fem/delta_ana - 1) < 0.01 else 'FAIL'}")
    
    print(f"\n  Max bending stress:")
    print(f"    Analytical: {sigma_bending_ana:.6e} Pa")
    print(f"    FEM axial:  {np.max(np.abs(res['sigma_axial'])):.6e} Pa")
    print(f"    FEM bending:{np.max(res['sigma_bending']):.6e} Pa")
    print(f"    FEM total:  {np.max(res['sigma_total']):.6e} Pa")
    
    bending_ratio = np.max(res['sigma_bending']) / sigma_bending_ana
    print(f"    Ratio:      {bending_ratio:.6f} {'PASS' if abs(bending_ratio - 1) < 0.05 else 'FAIL'}")
    
    print(f"\n  Moments at fixed end:")
    print(f"    M_i (FEM):   {res['moments'][0, 0]:.6e} N*m")
    print(f"    M_i (ana):   {M_max:.6e} N*m")
    
    # Check moment distribution along beam
    print(f"\n  Moment distribution:")
    for idx in range(len(res['edge_list'])):
        x_mid = (node_pos[idx, 0] + node_pos[idx+1, 0]) / 2
        M_avg = (res['moments'][idx, 0] + res['moments'][idx, 1]) / 2
        M_ana = P * (L - x_mid)
        print(f"    x={x_mid:.2f}: M_FEM={M_avg:.2f} N*m, M_ana={M_ana:.2f} N*m, ratio={M_avg/M_ana:.4f}" if abs(M_ana) > 1e-6 else f"    x={x_mid:.2f}: M_FEM={M_avg:.2f} N*m (near-free)")
    
    return {
        'delta_ratio': delta_fem/delta_ana,
        'bending_stress_ratio': bending_ratio,
        'max_sigma_bending': float(np.max(res['sigma_bending'])),
        'max_sigma_axial': float(np.max(np.abs(res['sigma_axial'])))
    }

def test_displacement_bc():
    print("\n" + "="*70)
    print("V6 TEST 2: Displacement BC - Prescribed Stretch")
    print("="*70)
    E, nu, L, r = 200e9, 0.3, 1.0, 0.01
    n = 11
    node_pos = np.zeros((n, 2))
    node_pos[:, 0] = np.linspace(0, L, n)
    edges = []
    for i in range(n-1): edges.extend([[i,i+1],[i+1,i]])
    edge_index = np.array(edges).T
    radii = np.full(edge_index.shape[1], r)
    
    solver = BeamFrameFEM_v6(E=E, nu=nu)
    
    # Prescribe 0.01m displacement at right end (1% strain)
    delta = 0.01
    prescribed = {n-1: (delta, 0.0)}
    
    res = solver.solve_2d(edge_index, node_pos, radii,
                         fixed_nodes=[0],
                         prescribed_disp=prescribed)
    
    # Analytical: uniform strain = delta/L
    strain_ana = delta / L
    sigma_ana = E * strain_ana
    A = np.pi * r**2
    force_ana = sigma_ana * A
    
    print(f"  Prescribed displacement: {delta} m at right end")
    print(f"  Analytical uniform strain: {strain_ana:.6f}")
    print(f"  Analytical stress: {sigma_ana:.6e} Pa")
    print(f"  Analytical reaction force: {force_ana:.6e} N")
    print(f"  ---")
    print(f"  FEM tip displacement: {res['u'][-1, 0]:.6e} m")
    print(f"  FEM max axial stress: {np.max(np.abs(res['sigma_axial'])):.6e} Pa")
    print(f"  FEM max bending stress: {np.max(res['sigma_bending']):.6e} Pa")
    
    # Check: for pure axial stretch, bending stress should be ~0
    stress_ratio = np.max(np.abs(res['sigma_axial'])) / sigma_ana
    bending_check = np.max(res['sigma_bending']) / sigma_ana
    
    print(f"  Axial stress ratio: {stress_ratio:.6f} {'PASS' if abs(stress_ratio - 1) < 0.05 else 'FAIL'}")
    print(f"  Bending/Axial ratio: {bending_check:.6e} {'PASS (near-zero)' if bending_check < 0.01 else 'FAIL (unexpected bending)'}")
    
    # Check displacement profile (should be linear)
    print(f"\n  Displacement profile (should be linear):")
    for i in range(n):
        x = node_pos[i, 0]
        ux = res['u'][i, 0]
        ux_ana = strain_ana * x
        print(f"    x={x:.2f}: ux_FEM={ux:.6e}, ux_ana={ux_ana:.6e}")
    
    # Check reaction force
    rxn = res['reactions'][0, 0]
    print(f"\n  Reaction force at fixed end: {rxn:.6e} N")
    print(f"  Analytical reaction: {-force_ana:.6e} N")
    print(f"  Ratio: {abs(rxn)/force_ana:.6f}")
    
    return {
        'stress_ratio': stress_ratio,
        'bending_ratio': bending_check,
        'reaction_ratio': abs(rxn)/force_ana
    }

def test_nonlinear():
    print("\n" + "="*70)
    print("V6 TEST 3: Nonlinear Large Deformation")
    print("="*70)
    E, nu, L, r = 1e9, 0.3, 1.0, 0.01
    n = 11
    node_pos = np.zeros((n, 2))
    node_pos[:, 0] = np.linspace(0, L, n)
    edges = []
    for i in range(n-1): edges.extend([[i,i+1],[i+1,i]])
    edge_index = np.array(edges).T
    radii = np.full(edge_index.shape[1], r)
    
    solver = BeamFrameFEM_v6(E=E, nu=nu)
    
    # Large prescribed displacement (50% stretch)
    delta = 0.5
    prescribed = {n-1: (delta, 0.0)}
    
    print(f"  Applying 50% stretch (delta={delta}m on L={L}m)")
    print(f"  Using {10} load steps")
    
    res = solver.solve_2d_nonlinear(
        edge_index, node_pos, radii,
        prescribed_disp=prescribed,
        fixed_nodes=[0],
        n_steps=10, tol=1e-8, max_iter=5
    )
    
    print(f"\n  Nonlinear history:")
    for h in res['history']:
        print(f"    Step {h['step']}: max_u={h['max_disp']:.4e}, max_stress={h['max_stress']:.4e}, iters={h['iterations']}")
    
    print(f"\n  Final results:")
    print(f"    Total tip displacement: {res['u_total'][-1, 0]:.6e} m")
    print(f"    Max axial stress: {np.max(np.abs(res['sigma_axial'])):.6e} Pa")
    print(f"    Max bending stress: {np.max(res['sigma_bending']):.6e} Pa")
    
    # Compare with linear solution
    res_linear = solver.solve_2d(edge_index, node_pos, radii,
                                 fixed_nodes=[0],
                                 prescribed_disp=prescribed)
    
    print(f"\n  Linear solution:")
    print(f"    Max axial stress: {np.max(np.abs(res_linear['sigma_axial'])):.6e} Pa")
    
    strain_ana = delta / L
    sigma_ana = E * strain_ana
    print(f"\n  Analytical uniform stress: {sigma_ana:.6e} Pa")
    
    return {
        'nonlinear_stress': float(np.max(res['sigma_total'])),
        'linear_stress': float(np.max(res_linear['sigma_total'])),
        'analytical_stress': sigma_ana
    }

def test_complex_structure():
    print("\n" + "="*70)
    print("V6 TEST 4: Complex Honeycomb with Disp BC")
    print("="*70)
    from fibernet import pattern_2d
    from fibernet.ml.gnn import graph_from_structure
    
    g = pattern_2d(unit='honeycomb', box=(10, 10), grid=(4, 4))
    gd = graph_from_structure(g)
    node_pos = gd['node_features'].numpy()[:, :2]
    edge_index = gd['edge_index'].numpy()
    n_nodes = node_pos.shape[0]
    n_edges = edge_index.shape[1]
    
    print(f"  Honeycomb: {n_nodes} nodes, {n_edges} edges")
    
    r = 0.01
    radii = np.full(n_edges, r)
    solver = BeamFrameFEM_v6(E=1e9, nu=0.3)
    
    # Find boundary nodes
    x = node_pos[:, 0]
    y = node_pos[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    tol = (x_max - x_min) * 0.05
    
    left_nodes = np.where(np.abs(x - x_min) < tol)[0].tolist()
    right_nodes = np.where(np.abs(x - x_max) < tol)[0].tolist()
    bottom_nodes = np.where(np.abs(y - y_min) < tol)[0].tolist()
    top_nodes = np.where(np.abs(y - y_max) < tol)[0].tolist()
    
    print(f"  Boundary: left={len(left_nodes)}, right={len(right_nodes)}, bottom={len(bottom_nodes)}, top={len(top_nodes)}")
    
    # Test A: Stretch right edge by 10%
    stretch = (x_max - x_min) * 0.1  # 10% of width
    prescribed = {}
    for node in right_nodes:
        prescribed[int(node)] = (stretch, 0.0)
    
    print(f"\n  Test A: Stretch right edge by {stretch:.2f} units (10%)")
    res = solver.solve_2d(edge_index, node_pos, radii,
                         fixed_nodes=left_nodes,
                         prescribed_disp=prescribed)
    
    print(f"    Max displacement: {np.max(np.linalg.norm(res['u'][:, :2], axis=1)):.6e}")
    print(f"    Max axial stress: {np.max(np.abs(res['sigma_axial'])):.6e} Pa")
    print(f"    Max bending stress: {np.max(res['sigma_bending']):.6e} Pa")
    print(f"    Max total stress: {np.max(res['sigma_total']):.6e} Pa")
    
    # Check propagation
    disps = np.linalg.norm(res['u'][:, :2], axis=1)
    x_sorted = np.argsort(x)
    
    # Bin by x-position
    n_bins = 5
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_means = []
    for b in range(n_bins):
        mask = (x >= bin_edges[b]) & (x < bin_edges[b+1])
        if mask.any():
            bin_means.append(np.mean(disps[mask]))
        else:
            bin_means.append(0)
    
    prop = bin_means[0] / bin_means[-1] if bin_means[-1] > 0 else 0
    print(f"    Propagation: {prop:.1%}")
    for b in range(n_bins):
        print(f"      Bin {b} (x={bin_edges[b]:.1f}-{bin_edges[b+1]:.1f}): mean_disp={bin_means[b]:.6e}")
    
    # Test B: Compress top by 10%
    compress = -(y_max - y_min) * 0.1
    prescribed_b = {}
    for node in top_nodes:
        prescribed_b[int(node)] = (0.0, compress)
    
    print(f"\n  Test B: Compress top by {abs(compress):.2f} units (10%)")
    res_b = solver.solve_2d(edge_index, node_pos, radii,
                           fixed_nodes=bottom_nodes,
                           prescribed_disp=prescribed_b)
    
    print(f"    Max displacement: {np.max(np.linalg.norm(res_b['u'][:, :2], axis=1)):.6e}")
    print(f"    Max bending stress: {np.max(res_b['sigma_bending']):.6e} Pa")
    
    return {
        'stretch': {'max_disp': float(np.max(np.linalg.norm(res['u'][:, :2], axis=1))),
                    'max_stress': float(np.max(res['sigma_total'])),
                    'propagation': prop},
        'compress': {'max_disp': float(np.max(np.linalg.norm(res_b['u'][:, :2], axis=1))),
                     'max_stress': float(np.max(res_b['sigma_total']))}
    }

def main():
    r1 = test_cantilever()
    r2 = test_displacement_bc()
    r3 = test_nonlinear()
    r4 = test_complex_structure()
    
    output = Path(__file__).parent / "results" / "phase6_v6_validation.json"
    output.parent.mkdir(exist_ok=True)
    with open(output, 'w') as f:
        json.dump({'cantilever': r1, 'disp_bc': r2, 'nonlinear': r3, 'complex': r4}, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"V6 Validation Complete — Results: {output}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()

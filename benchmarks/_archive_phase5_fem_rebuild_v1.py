#!/usr/bin/env python3
"""
Phase 5 Rebuild: Proper FEM Integration with Graph-Level Verification
=====================================================================
Key fixes:
1. Build scipy-based REFERENCE truss solver (ground truth for spring model)
2. Proper skfem 2D continuum FEM with correct BC enforcement
3. Fair comparison: same physics model → same answer
4. Multi-complexity: small (2x2), medium (5x5), large (8x8)
5. Graph-level node-by-node analysis
6. Visualization for visual inspection
7. Cross-platform dependency audit

Usage:
    python benchmarks/phase5_fem_rebuild.py
"""

import sys, os, json, time, gc, platform
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "benchmarks" / "results"

# ============================================================
# 1. Reference Truss FEM Solver (Ground Truth)
# ============================================================

def reference_truss_solve(node_pos, edge_index, radii, forces, fixed_nodes, E=1e9):
    """
    Reference 2D truss FEM solver using scipy.sparse.
    This is the EXACT same physics as DifferentiableSpringNetwork.
    
    Each truss element contributes:
        K_e = (EA/L) * [cc  -cc; -cc  cc]
    where c = direction cosines, A = πr², L = element length.
    
    Returns:
        u: (n_nodes, 2) displacement array
        sigma: (n_edges,) axial stress array
    """
    n_nodes = node_pos.shape[0]
    n_edges = edge_index.shape[1]
    n_dof = n_nodes * 2
    
    # Build global stiffness matrix
    rows, cols, vals = [], [], []
    
    areas = np.pi * radii ** 2  # (n_edges,)
    
    for e in range(n_edges):
        i = edge_index[0, e]
        j = edge_index[1, e]
        
        # Element geometry
        dx = node_pos[j, 0] - node_pos[i, 0]
        dy = node_pos[j, 1] - node_pos[i, 1]
        L = np.sqrt(dx**2 + dy**2)
        if L < 1e-12:
            continue
        
        c = dx / L  # cos(θ)
        s = dy / L  # sin(θ)
        
        # Axial stiffness
        k = E * areas[e] / L
        
        # Element stiffness in global coords (4x4)
        # K_e = k * [c²  cs  -c²  -cs; cs  s²  -cs  -s²; -c²  -cs  c²  cs; -cs  -s²  cs  s²]
        cc = c * c
        cs = c * s
        ss = s * s
        
        ke = k * np.array([
            [cc,  cs, -cc, -cs],
            [cs,  ss, -cs, -ss],
            [-cc, -cs, cc,  cs],
            [-cs, -ss, cs,  ss],
        ])
        
        # DOF indices
        dofs = [2*i, 2*i+1, 2*j, 2*j+1]
        
        for a in range(4):
            for b in range(4):
                rows.append(dofs[a])
                cols.append(dofs[b])
                vals.append(ke[a, b])
    
    K = sparse.csr_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    
    # Add small regularization for numerical stability
    K = K + 1e-8 * sparse.eye(n_dof)
    
    # Force vector
    f = np.zeros(n_dof)
    for n in range(n_nodes):
        f[2*n] = forces[n, 0]
        f[2*n+1] = forces[n, 1]
    
    # Apply boundary conditions
    fixed_dofs = []
    for n in fixed_nodes:
        fixed_dofs.extend([2*n, 2*n+1])
    fixed_dofs = np.array(sorted(set(fixed_dofs)))
    
    all_dofs = np.arange(n_dof)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
    
    # Reduce system
    K_ff = K[free_dofs][:, free_dofs]
    f_f = f[free_dofs]
    
    # Solve
    u_f = spsolve(K_ff, f_f)
    
    # Expand solution
    u = np.zeros(n_dof)
    u[free_dofs] = u_f
    u = u.reshape(n_nodes, 2)
    
    # Compute stresses
    sigma = np.zeros(n_edges)
    for e in range(n_edges):
        i = edge_index[0, e]
        j = edge_index[1, e]
        dx = node_pos[j, 0] - node_pos[i, 0]
        dy = node_pos[j, 1] - node_pos[i, 1]
        L = np.sqrt(dx**2 + dy**2)
        if L < 1e-12:
            continue
        c = dx / L
        s = dy / L
        # Axial strain = (u_j - u_i) · direction / L
        du_x = u[j, 0] - u[i, 0]
        du_y = u[j, 1] - u[i, 1]
        strain = (du_x * c + du_y * s) / L
        sigma[e] = E * strain
    
    return u, sigma


# ============================================================
# 2. Proper skfem 2D Continuum FEM
# ============================================================

def skfem_continuum_solve(node_pos, triangles, forces, fixed_nodes, E=1e9, nu=0.3):
    """
    2D plane-stress continuum FEM using skfem.
    
    This is a DIFFERENT physical model from the truss (continuum vs truss).
    Used for cross-validation, not direct comparison.
    """
    import skfem
    from skfem import MeshTri, Basis, ElementTriP1, ElementVector
    from skfem.assembly import BilinearForm
    from skfem.helpers import ddot, sym_grad
    
    mesh = MeshTri(node_pos.T, triangles.T)
    elem = ElementVector(ElementTriP1())
    basis = Basis(mesh, elem)
    
    # Plane stress Lamé parameters
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    
    @BilinearForm
    def stiffness(u, v, w):
        eps_u = sym_grad(u)
        eps_v = sym_grad(v)
        div_u = u.grad[0, 0] + u.grad[1, 1]
        div_v = v.grad[0, 0] + v.grad[1, 1]
        return lam * div_u * div_v + 2 * mu * ddot(eps_u, eps_v)
    
    K = stiffness.assemble(basis)
    
    # Force vector
    n_nodes = node_pos.shape[0]
    f_vec = np.zeros(2 * n_nodes)
    for n in range(n_nodes):
        f_vec[2*n] = forces[n, 0]
        f_vec[2*n+1] = forces[n, 1]
    
    # Boundary conditions — CRITICAL FIX: use proper DOF indexing
    fixed_dofs = []
    for n in fixed_nodes:
        fixed_dofs.extend([2*n, 2*n+1])
    fixed_dofs = np.array(sorted(set(fixed_dofs)))
    
    # Use skfem.condense properly
    K_bc, f_bc, x0, I = skfem.condense(K, f_vec, D=fixed_dofs)
    
    # Solve
    u_reduced = skfem.solve(K_bc, f_bc)
    
    # Reconstruct full solution using expand=True output
    u_full = np.zeros(2 * n_nodes)
    u_full[fixed_dofs] = 0.0  # Fixed nodes = 0
    u_full[I] = u_reduced
    
    return u_full.reshape(n_nodes, 2)


# ============================================================
# 3. Multi-Complexity Benchmark Suite
# ============================================================

def run_benchmarks():
    """Run FEM benchmarks across complexity levels and unit types."""
    from fibernet import pattern_2d, list_units
    from fibernet.ml.gnn import graph_from_structure
    from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
    from scipy.spatial import Delaunay
    
    units = ['honeycomb', 'kagome', 'reentrant', 'diamond', 'square', 'triangle']
    complexity_levels = {
        'small': (2, 3),   # grid sizes
        'medium': (5, 5),
        'large': (8, 8),
    }
    
    results = {}
    
    for level_name, (gx, gy) in complexity_levels.items():
        print(f"\n{'='*60}")
        print(f"Complexity Level: {level_name} (grid {gx}x{gy})")
        print(f"{'='*60}")
        results[level_name] = {}
        
        for unit in units:
            gc.collect()
            
            try:
                g = pattern_2d(unit=unit, box=(10, 10), grid=(gx, gy))
                gd = graph_from_structure(g)
            except Exception as e:
                print(f"  {unit}: SKIP (generation failed: {e})")
                continue
            
            n_nodes = gd['node_features'].shape[0]
            n_edges = gd['edge_index'].shape[1]
            positions = gd['node_features'][:, :2].numpy()
            edge_index = gd['edge_index'].numpy()
            
            # Standard loading
            radii = np.ones(n_edges) * 0.01
            forces = np.zeros((n_nodes, 2))
            forces[-1, 0] = 500.0  # Horizontal load at last node
            fixed_nodes = [0, 1]
            
            print(f"\n  {unit}: {n_nodes} nodes, {n_edges} edges")
            
            unit_results = {
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'grid': f"{gx}x{gy}",
            }
            
            # --- A. Our DifferentiableSpringNetwork ---
            try:
                physics = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)
                radii_t = torch.tensor(radii, dtype=torch.float32)
                forces_t = torch.tensor(forces, dtype=torch.float32)
                fixed_t = torch.tensor(fixed_nodes, dtype=torch.long)
                
                t0 = time.time()
                with torch.no_grad():
                    u_ds, sigma_ds = physics.solve(
                        gd['edge_index'], gd['node_features'][:, :2],
                        radii_t, forces_t, fixed_t)
                t_ds = time.time() - t0
                
                u_ds_np = u_ds.numpy()
                sigma_ds_np = sigma_ds.numpy()
                
                unit_results['spring_network'] = {
                    'max_disp': float(np.max(np.sqrt(u_ds_np[:,0]**2 + u_ds_np[:,1]**2))),
                    'max_stress': float(np.max(np.abs(sigma_ds_np))),
                    'time_ms': round(t_ds * 1000, 2),
                    'fixed_node_disp': [float(np.sqrt(u_ds_np[n,0]**2 + u_ds_np[n,1]**2)) for n in fixed_nodes],
                }
                print(f"    Spring:  max_u={unit_results['spring_network']['max_disp']:.6e}, "
                      f"time={t_ds*1000:.1f}ms, fixed_nodes={[f'{d:.2e}' for d in unit_results['spring_network']['fixed_node_disp']]}")
            except Exception as e:
                print(f"    Spring: FAILED ({e})")
                unit_results['spring_network'] = {'error': str(e)}
                u_ds_np = None
            
            # --- B. Reference Truss FEM (scipy) ---
            try:
                t0 = time.time()
                u_ref, sigma_ref = reference_truss_solve(
                    positions, edge_index, radii, forces, fixed_nodes, E=1e9)
                t_ref = time.time() - t0
                
                unit_results['reference_truss'] = {
                    'max_disp': float(np.max(np.sqrt(u_ref[:,0]**2 + u_ref[:,1]**2))),
                    'max_stress': float(np.max(np.abs(sigma_ref))),
                    'time_ms': round(t_ref * 1000, 2),
                    'fixed_node_disp': [float(np.sqrt(u_ref[n,0]**2 + u_ref[n,1]**2)) for n in fixed_nodes],
                }
                print(f"    RefTruss: max_u={unit_results['reference_truss']['max_disp']:.6e}, "
                      f"time={t_ref*1000:.1f}ms, fixed_nodes={[f'{d:.2e}' for d in unit_results['reference_truss']['fixed_node_disp']]}")
                
                # --- Compare Spring vs Reference ---
                if u_ds_np is not None:
                    disp_diff = np.sqrt((u_ds_np[:,0] - u_ref[:,0])**2 + (u_ds_np[:,1] - u_ref[:,1])**2)
                    max_diff = float(np.max(disp_diff))
                    mean_diff = float(np.mean(disp_diff))
                    max_u_ref = float(np.max(np.sqrt(u_ref[:,0]**2 + u_ref[:,1]**2)))
                    relative_error = max_diff / (max_u_ref + 1e-20)
                    
                    stress_diff = np.abs(sigma_ds_np - sigma_ref)
                    max_stress_ref = float(np.max(np.abs(sigma_ref)))
                    stress_relative_error = float(np.max(stress_diff)) / (max_stress_ref + 1e-20)
                    
                    unit_results['spring_vs_reference'] = {
                        'max_disp_diff': max_diff,
                        'mean_disp_diff': mean_diff,
                        'relative_disp_error': relative_error,
                        'relative_stress_error': stress_relative_error,
                        'match': relative_error < 0.01,  # <1% relative error
                    }
                    match_str = "✓ MATCH" if relative_error < 0.01 else "✗ MISMATCH"
                    print(f"    Spring vs Ref: rel_err={relative_error:.6f} {match_str}, "
                          f"stress_rel_err={stress_relative_error:.6f}")
            except Exception as e:
                print(f"    RefTruss: FAILED ({e})")
                unit_results['reference_truss'] = {'error': str(e)}
                u_ref = None
            
            # --- C. skfem Continuum FEM ---
            try:
                tri = Delaunay(positions)
                triangles = tri.simplices
                
                t0 = time.time()
                u_fem = skfem_continuum_solve(
                    positions, triangles, forces, fixed_nodes, E=1e9, nu=0.3)
                t_fem = time.time() - t0
                
                max_u_fem = float(np.max(np.sqrt(u_fem[:,0]**2 + u_fem[:,1]**2)))
                
                unit_results['skfem_continuum'] = {
                    'max_disp': max_u_fem,
                    'n_triangles': len(triangles),
                    'time_ms': round(t_fem * 1000, 2),
                    'fixed_node_disp': [float(np.sqrt(u_fem[n,0]**2 + u_fem[n,1]**2)) for n in fixed_nodes],
                }
                print(f"    skfem:   max_u={max_u_fem:.6e}, "
                      f"triangles={len(triangles)}, time={t_fem*1000:.1f}ms, "
                      f"fixed_nodes={[f'{d:.2e}' for d in unit_results['skfem_continuum']['fixed_node_disp']]}")
                
                # Cross-model comparison (continuum vs truss — different physics!)
                if u_ref is not None:
                    max_u_ref = unit_results['reference_truss']['max_disp']
                    ratio = max_u_fem / (max_u_ref + 1e-20)
                    unit_results['continuum_vs_truss_ratio'] = round(ratio, 4)
                    print(f"    Continuum/Truss ratio: {ratio:.4f} (expected ≠ 1, different physics)")
            except Exception as e:
                print(f"    skfem: FAILED ({e})")
                unit_results['skfem_continuum'] = {'error': str(e)}
            
            # --- D. Graph-level analysis ---
            if u_ref is not None:
                import networkx as nx
                nx_g = g.to_networkx()
                
                # Degree distribution
                degrees = [d for _, d in nx_g.degree()]
                
                # Stress-weighted graph metrics
                mean_stress = float(np.mean(np.abs(sigma_ref)))
                max_stress = float(np.max(np.abs(sigma_ref)))
                scf = max_stress / (mean_stress + 1e-20)
                
                # Force chain detection
                stress_abs = np.abs(sigma_ref)
                p90 = float(np.percentile(stress_abs, 90))
                chain_mask = stress_abs > p90
                chain_fraction = float(np.sum(chain_mask)) / n_edges
                stress_in_chain = float(np.sum(stress_abs[chain_mask])) / (float(np.sum(stress_abs)) + 1e-20)
                
                # Displacement-degree correlation
                disp_mag = np.sqrt(u_ref[:,0]**2 + u_ref[:,1]**2)
                degree_arr = np.array(degrees)
                if len(disp_mag) == len(degree_arr) and np.std(disp_mag) > 1e-12 and np.std(degree_arr) > 1e-12:
                    disp_degree_corr = float(np.corrcoef(disp_mag, degree_arr)[0, 1])
                else:
                    disp_degree_corr = 0.0
                
                # Compliance
                compliance = float(np.sum(forces * u_ref))
                
                # Spectral
                L = nx.laplacian_matrix(nx_g).toarray()
                eigvals = np.sort(np.linalg.eigvalsh(L))
                alg_conn = float(eigvals[1]) if len(eigvals) > 1 else 0.0
                
                unit_results['graph_analysis'] = {
                    'avg_degree': float(np.mean(degrees)),
                    'max_degree': int(max(degrees)),
                    'stress_concentration_factor': round(scf, 3),
                    'force_chain_fraction': round(chain_fraction, 4),
                    'stress_carried_by_chains': round(stress_in_chain, 4),
                    'disp_degree_correlation': round(disp_degree_corr, 4),
                    'compliance': round(compliance, 6),
                    'algebraic_connectivity': round(alg_conn, 6),
                }
                ga = unit_results['graph_analysis']
                print(f"    Graph: SCF={ga['stress_concentration_factor']}, "
                      f"chain_frac={ga['force_chain_fraction']:.3f}, "
                      f"alg_conn={ga['algebraic_connectivity']:.4f}, "
                      f"compliance={ga['compliance']:.4f}")
            
            results[level_name][unit] = unit_results
    
    return results


# ============================================================
# 4. Cross-Platform Dependency Audit
# ============================================================

def cross_platform_audit():
    """Audit all dependencies for cross-platform compatibility."""
    deps = {
        'torch': {'package': 'torch', 'platforms': ['linux', 'macos', 'windows'],
                  'notes': 'CPU version universally available; CUDA Linux/Windows only'},
        'numpy': {'package': 'numpy', 'platforms': ['linux', 'macos', 'windows'],
                  'notes': 'Universal'},
        'scipy': {'package': 'scipy', 'platforms': ['linux', 'macos', 'windows'],
                  'notes': 'Universal, requires BLAS/LAPACK'},
        'scikit-fem': {'package': 'scikit-fem', 'platforms': ['linux', 'macos', 'windows'],
                       'notes': 'Pure Python + scipy, cross-platform'},
        'networkx': {'package': 'networkx', 'platforms': ['linux', 'macos', 'windows'],
                     'notes': 'Pure Python, universal'},
        'taichi': {'package': 'taichi', 'platforms': ['linux', 'macos', 'windows'],
                   'notes': 'GPU compute, all platforms supported'},
        'fenics': {'package': 'fenics-dolfinx', 'platforms': ['linux', 'macos'],
                   'notes': 'NOT available on Windows natively; Docker only'},
    }
    
    audit_results = {}
    for name, info in deps.items():
        try:
            mod = __import__(info['package'].replace('-', '_'))
            version = getattr(mod, '__version__', 'unknown')
            audit_results[name] = {
                'installed': True,
                'version': version,
                'platforms': info['platforms'],
                'notes': info['notes'],
                'current_platform_ok': True,
            }
        except ImportError:
            audit_results[name] = {
                'installed': False,
                'version': None,
                'platforms': info['platforms'],
                'notes': info['notes'],
                'current_platform_ok': False,
            }
    
    # Check current platform
    current = platform.system().lower()
    if current == 'darwin':
        current = 'macos'
    
    audit_results['current_platform'] = current
    audit_results['python_version'] = platform.python_version()
    
    return audit_results


# ============================================================
# 5. Visualization Generator
# ============================================================

def generate_visualization(results, output_path):
    """Generate a comprehensive visualization image for visual inspection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from fibernet import pattern_2d
    from fibernet.ml.gnn import graph_from_structure
    
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    units = ['honeycomb', 'kagome', 'reentrant', 'diamond']
    
    # Row 1: Structure + displacement field (medium complexity)
    for col, unit in enumerate(units):
        ax = fig.add_subplot(gs[0, col])
        
        try:
            g = pattern_2d(unit=unit, box=(10, 10), grid=(5, 5))
            gd = graph_from_structure(g)
            
            n_nodes = gd['node_features'].shape[0]
            n_edges = gd['edge_index'].shape[1]
            positions = gd['node_features'][:, :2].numpy()
            edge_index = gd['edge_index'].numpy()
            
            # Solve with reference truss
            radii = np.ones(n_edges) * 0.01
            forces = np.zeros((n_nodes, 2))
            forces[-1, 0] = 500.0
            fixed = [0, 1]
            
            u_ref, sigma_ref = reference_truss_solve(
                positions, edge_index, radii, forces, fixed, E=1e9)
            
            disp_mag = np.sqrt(u_ref[:,0]**2 + u_ref[:,1]**2)
            
            # Draw deformed structure
            scale = 50  # amplification factor
            pos_def = positions + u_ref * scale
            
            # Edges colored by stress
            for e in range(n_edges):
                i, j = edge_index[0, e], edge_index[1, e]
                stress = sigma_ref[e]
                color = plt.cm.RdYlBu_r(np.clip((stress + 1e5) / 2e5, 0, 1))
                ax.plot([pos_def[i,0], pos_def[j,0]], [pos_def[i,1], pos_def[j,1]],
                       color=color, linewidth=1.5, alpha=0.8)
            
            # Nodes colored by displacement
            sc = ax.scatter(pos_def[:,0], pos_def[:,1], c=disp_mag, cmap='hot',
                          s=30, edgecolors='black', linewidths=0.5, zorder=5)
            
            # Mark fixed nodes
            for n in fixed:
                ax.scatter(pos_def[n,0], pos_def[n,1], marker='s', s=100,
                          facecolors='none', edgecolors='blue', linewidths=2, zorder=6)
            
            # Force arrow
            ax.annotate('', xy=(pos_def[-1,0] + 0.5, pos_def[-1,1]),
                       xytext=(pos_def[-1,0], pos_def[-1,1]),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
            
            ax.set_title(f'{unit} ({n_nodes}n/{n_edges}e)\n'
                        f'max_u={disp_mag.max():.4e}', fontsize=10)
            ax.set_aspect('equal')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
        except Exception as e:
            ax.text(0.5, 0.5, f'{unit}\nFAILED\n{str(e)[:80]}',
                   transform=ax.transAxes, ha='center', va='center', fontsize=9)
    
    # Row 2: Solver comparison (spring vs reference vs skfem)
    for col, unit in enumerate(units):
        ax = fig.add_subplot(gs[1, col])
        
        try:
            g = pattern_2d(unit=unit, box=(10, 10), grid=(4, 4))
            gd = graph_from_structure(g)
            
            n_nodes = gd['node_features'].shape[0]
            n_edges = gd['edge_index'].shape[1]
            positions = gd['node_features'][:, :2].numpy()
            edge_index = gd['edge_index'].numpy()
            
            radii = np.ones(n_edges) * 0.01
            forces = np.zeros((n_nodes, 2))
            forces[-1, 0] = 500.0
            fixed = [0, 1]
            
            # Reference truss
            u_ref, _ = reference_truss_solve(positions, edge_index, radii, forces, fixed)
            disp_ref = np.sqrt(u_ref[:,0]**2 + u_ref[:,1]**2)
            
            # Our spring network
            from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
            physics = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)
            with torch.no_grad():
                u_ds, _ = physics.solve(
                    gd['edge_index'], gd['node_features'][:, :2],
                    torch.tensor(radii, dtype=torch.float32),
                    torch.tensor(forces, dtype=torch.float32),
                    torch.tensor(fixed, dtype=torch.long))
            disp_ds = np.sqrt(u_ds.numpy()[:,0]**2 + u_ds.numpy()[:,1]**2)
            
            # skfem continuum
            from scipy.spatial import Delaunay
            tri = Delaunay(positions)
            u_fem = skfem_continuum_solve(positions, tri.simplices, forces, fixed)
            disp_fem = np.sqrt(u_fem[:,0]**2 + u_fem[:,1]**2)
            
            # Plot sorted displacements
            node_ids = np.arange(n_nodes)
            ax.plot(sorted(disp_ref), label='Reference Truss', linewidth=2)
            ax.plot(sorted(disp_ds), '--', label='Our Spring', linewidth=2)
            
            # skfem has different physics, scale for comparison
            if disp_fem.max() > 0:
                scale_factor = disp_ref.max() / (disp_fem.max() + 1e-20)
                ax.plot(sorted(disp_fem * scale_factor), ':', label=f'skfem (×{scale_factor:.1f})', linewidth=2)
            
            ax.set_title(f'{unit}: Solver Comparison', fontsize=10)
            ax.set_xlabel('Node rank')
            ax.set_ylabel('Displacement (m)')
            ax.legend(fontsize=7)
            ax.set_yscale('log')
        except Exception as e:
            ax.text(0.5, 0.5, f'{unit}\nFAILED\n{str(e)[:80]}',
                   transform=ax.transAxes, ha='center', va='center', fontsize=9)
    
    # Row 3: Graph-level metrics
    # 3a: Stress distribution histogram
    ax = fig.add_subplot(gs[2, 0])
    all_stresses = []
    labels = []
    for unit in units:
        try:
            g = pattern_2d(unit=unit, box=(10, 10), grid=(5, 5))
            gd = graph_from_structure(g)
            n_nodes = gd['node_features'].shape[0]
            n_edges = gd['edge_index'].shape[1]
            positions = gd['node_features'][:, :2].numpy()
            edge_index = gd['edge_index'].numpy()
            radii = np.ones(n_edges) * 0.01
            forces = np.zeros((n_nodes, 2))
            forces[-1, 0] = 500.0
            _, sigma = reference_truss_solve(positions, edge_index, radii, forces, [0,1])
            all_stresses.append(np.abs(sigma))
            labels.append(unit)
        except:
            pass
    
    if all_stresses:
        ax.boxplot(all_stresses, labels=labels)
        ax.set_title('Stress Distribution by Unit Type', fontsize=10)
        ax.set_ylabel('|σ| (Pa)')
        ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=30)
    
    # 3b: Compliance vs Algebraic Connectivity
    ax = fig.add_subplot(gs[2, 1])
    import networkx as nx
    alg_conns = []
    compliances = []
    unit_labels = []
    for unit in ['honeycomb', 'kagome', 'reentrant', 'diamond', 'square', 'triangle']:
        try:
            g = pattern_2d(unit=unit, box=(10, 10), grid=(5, 5))
            gd = graph_from_structure(g)
            n_nodes = gd['node_features'].shape[0]
            n_edges = gd['edge_index'].shape[1]
            positions = gd['node_features'][:, :2].numpy()
            edge_index = gd['edge_index'].numpy()
            radii = np.ones(n_edges) * 0.01
            forces = np.zeros((n_nodes, 2))
            forces[-1, 0] = 500.0
            u, _ = reference_truss_solve(positions, edge_index, radii, forces, [0,1])
            compliance = float(np.sum(forces * u))
            
            nx_g = g.to_networkx()
            L_mat = nx.laplacian_matrix(nx_g).toarray()
            eigvals = np.sort(np.linalg.eigvalsh(L_mat))
            alg_conn = float(eigvals[1]) if len(eigvals) > 1 else 0.0
            
            alg_conns.append(alg_conn)
            compliances.append(compliance)
            unit_labels.append(unit)
        except:
            pass
    
    if alg_conns:
        ax.scatter(alg_conns, compliances, s=100, zorder=5)
        for i, label in enumerate(unit_labels):
            ax.annotate(label, (alg_conns[i], compliances[i]), fontsize=8,
                       xytext=(5, 5), textcoords='offset points')
        corr = np.corrcoef(alg_conns, compliances)[0, 1] if len(alg_conns) > 2 else 0
        ax.set_title(f'Algebraic Connectivity vs Compliance\n(corr={corr:.3f})', fontsize=10)
        ax.set_xlabel('λ₂ (algebraic connectivity)')
        ax.set_ylabel('Compliance')
    
    # 3c: Force chain analysis
    ax = fig.add_subplot(gs[2, 2])
    chain_fracs = []
    chain_stresses = []
    chain_labels = []
    for unit in units:
        try:
            g = pattern_2d(unit=unit, box=(10, 10), grid=(5, 5))
            gd = graph_from_structure(g)
            n_nodes = gd['node_features'].shape[0]
            n_edges = gd['edge_index'].shape[1]
            positions = gd['node_features'][:, :2].numpy()
            edge_index = gd['edge_index'].numpy()
            radii = np.ones(n_edges) * 0.01
            forces = np.zeros((n_nodes, 2))
            forces[-1, 0] = 500.0
            _, sigma = reference_truss_solve(positions, edge_index, radii, forces, [0,1])
            
            stress_abs = np.abs(sigma)
            p90 = np.percentile(stress_abs, 90)
            chain_mask = stress_abs > p90
            chain_frac = np.sum(chain_mask) / n_edges
            stress_in_chain = np.sum(stress_abs[chain_mask]) / (np.sum(stress_abs) + 1e-20)
            
            chain_fracs.append(chain_frac)
            chain_stresses.append(stress_in_chain)
            chain_labels.append(unit)
        except:
            pass
    
    if chain_fracs:
        x = np.arange(len(chain_labels))
        width = 0.35
        ax.bar(x - width/2, chain_fracs, width, label='Chain fraction', color='steelblue')
        ax.bar(x + width/2, chain_stresses, width, label='Stress carried', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(chain_labels, rotation=30)
        ax.set_title('Force Chain Analysis (P90 threshold)', fontsize=10)
        ax.set_ylabel('Fraction')
        ax.legend(fontsize=8)
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='10% baseline')
    
    # 3d: Solver accuracy summary
    ax = fig.add_subplot(gs[2, 3])
    solver_data = {}
    for unit in units:
        try:
            g = pattern_2d(unit=unit, box=(10, 10), grid=(4, 4))
            gd = graph_from_structure(g)
            n_nodes = gd['node_features'].shape[0]
            n_edges = gd['edge_index'].shape[1]
            positions = gd['node_features'][:, :2].numpy()
            edge_index = gd['edge_index'].numpy()
            radii = np.ones(n_edges) * 0.01
            forces = np.zeros((n_nodes, 2))
            forces[-1, 0] = 500.0
            
            u_ref, _ = reference_truss_solve(positions, edge_index, radii, forces, [0,1])
            
            from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
            physics = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)
            with torch.no_grad():
                u_ds, _ = physics.solve(
                    gd['edge_index'], gd['node_features'][:, :2],
                    torch.tensor(radii, dtype=torch.float32),
                    torch.tensor(forces, dtype=torch.float32),
                    torch.tensor([0, 1], dtype=torch.long))
            
            diff = np.sqrt((u_ds.numpy()[:,0] - u_ref[:,0])**2 + (u_ds.numpy()[:,1] - u_ref[:,1])**2)
            max_u = np.max(np.sqrt(u_ref[:,0]**2 + u_ref[:,1]**2))
            rel_err = np.max(diff) / (max_u + 1e-20)
            solver_data[unit] = rel_err
        except:
            pass
    
    if solver_data:
        bars = ax.bar(solver_data.keys(), solver_data.values(), color='seagreen')
        ax.axhline(y=0.01, color='red', linestyle='--', label='1% threshold')
        ax.set_title('Spring vs Reference: Relative Error', fontsize=10)
        ax.set_ylabel('Max Relative Error')
        ax.set_yscale('log')
        ax.legend()
        ax.tick_params(axis='x', rotation=30)
        for bar, val in zip(bars, solver_data.values()):
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.2,
                   f'{val:.1e}', ha='center', fontsize=8)
    
    fig.suptitle('FiberNet FEM Integration — Comprehensive Benchmark Dashboard',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nVisualization saved to: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Phase 5 Rebuild: Proper FEM with Graph-Level Verification")
    print("=" * 70)
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {platform.python_version()}")
    
    # Run benchmarks
    results = run_benchmarks()
    
    # Cross-platform audit
    print(f"\n{'='*60}")
    print("Cross-Platform Dependency Audit")
    print(f"{'='*60}")
    audit = cross_platform_audit()
    print(f"  Current platform: {audit['current_platform']}")
    print(f"  Python: {audit['python_version']}")
    for name, info in audit.items():
        if name in ('current_platform', 'python_version'):
            continue
        status = "✓" if info.get('installed') else "✗"
        ver = info.get('version', 'N/A')
        plats = ', '.join(info.get('platforms', []))
        print(f"  {status} {name:15s} v{ver:12s} platforms=[{plats}] — {info.get('notes', '')}")
    
    # Generate visualization
    print(f"\n{'='*60}")
    print("Generating Visualization")
    print(f"{'='*60}")
    viz_path = str(RESULTS_DIR / "phase5_fem_dashboard.png")
    generate_visualization(results, viz_path)
    
    # Save results
    output = {
        'benchmarks': results,
        'cross_platform_audit': audit,
        'visualization_path': viz_path,
    }
    
    output_file = RESULTS_DIR / "phase5_fem_rebuild_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for level, units in results.items():
        for unit, data in units.items():
            match_data = data.get('spring_vs_reference', {})
            match_str = ""
            if 'relative_disp_error' in match_data:
                match_str = f"err={match_data['relative_disp_error']:.2e} {'✓' if match_data.get('match') else '✗'}"
            ga = data.get('graph_analysis', {})
            print(f"  [{level:6s}] {unit:12s} {data['n_nodes']:4d}n | {match_str} | "
                  f"SCF={ga.get('stress_concentration_factor', 'N/A')}")
    
    return output

if __name__ == '__main__':
    main()

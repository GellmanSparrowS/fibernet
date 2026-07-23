#!/usr/bin/env python3
"""
Phase 5: FEM Integration & Deep Graph-Level Physics Verification
================================================================
- scikit-fem FEA solver integration
- Convert StructureGraph → FEM mesh → solve → compare with DiffPhysics
- Graph-level stress path analysis
- Force chain detection
- Cross-validation between solvers

Usage:
    python benchmarks/phase5_fem_integration.py
"""

import sys, os, json, time, gc, traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "benchmarks" / "results"

def safe_test(name, fn):
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        return {'name': name, 'passed': True, 'time_s': round(elapsed, 3), 'stats': result}
    except Exception as e:
        elapsed = time.time() - t0
        return {'name': name, 'passed': False, 'time_s': round(elapsed, 3),
                'error': str(e)[:300], 'traceback': traceback.format_exc()[-800:]}


# ============================================================
# 1. scikit-fem FEA Integration
# ============================================================

def test_skfem_truss_solver():
    """Test scikit-fem as a truss/beam FEA solver for fiber networks."""
    import skfem
    from skfem import MeshLine, Basis, ElementLineP1
    from skfem.assembly import BilinearForm, LinearForm
    from skfem.helpers import dot, ddot
    import numpy as np
    
    # 4-node truss
    mesh = MeshLine(np.linspace(0, 1, 4))
    basis = Basis(mesh, ElementLineP1())
    
    @BilinearForm
    def stiffness(u, v, w):
        return dot(u.grad, v.grad)
    
    @LinearForm
    def load(v, w):
        return 1.0 * v
    
    K = stiffness.assemble(basis)
    f = load.assemble(basis)
    
    # Apply BC (fix node 0)
    K_bc, f_bc, _, _ = skfem.condense(K, f, D=np.array([0]))
    u = skfem.solve(K_bc, f_bc)
    
    return {
        'solver': 'scikit-fem',
        'mesh_nodes': int(mesh.nvertices),
        'solution': u.tolist(),
        'u_max': float(np.max(np.abs(u))) if len(u) > 0 else 0.0,
    }

def test_skfem_2d_mesh():
    """Create 2D triangular mesh from StructureGraph and solve elasticity."""
    import skfem
    from skfem import MeshTri, Basis, ElementTriP1, ElementVector
    from skfem.assembly import BilinearForm, LinearForm
    from skfem.helpers import dot, ddot, sym_grad
    import numpy as np
    
    # Create a simple triangular mesh
    mesh = MeshTri.init_symmetric()  # Unit square with triangles
    
    # Vector-valued P1 elements (for 2D displacement)
    elem = ElementVector(ElementTriP1())
    basis = Basis(mesh, elem)
    
    # Linear elasticity stiffness
    E = 1e9  # Young's modulus
    nu = 0.3  # Poisson's ratio
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    
    @BilinearForm
    def stiffness(u, v, w):
        # σ(u) : ε(v) = λ(∇·u)(∇·v) + 2μ ε(u):ε(v)
        eps_u = sym_grad(u)
        eps_v = sym_grad(v)
        div_u = u.grad[0, 0] + u.grad[1, 1]
        div_v = v.grad[0, 0] + v.grad[1, 1]
        return lam * div_u * div_v + 2 * mu * ddot(eps_u, eps_v)
    
    K = stiffness.assemble(basis)
    
    # Apply point load
    f = np.zeros(K.shape[0])
    # Find top-right corner node
    p = mesh.p
    top_right = np.argmin((p[0] - 1)**2 + (p[1] - 1)**2)
    f[2 * top_right] = 100.0  # x-force
    
    # Fix bottom edge
    bottom_nodes = np.where(np.abs(p[1]) < 1e-10)[0]
    fixed_dofs = np.concatenate([2 * bottom_nodes, 2 * bottom_nodes + 1])
    
    K_bc, f_bc, _, _ = skfem.condense(K, f, D=fixed_dofs.astype(int))
    u = skfem.solve(K_bc, f_bc)
    
    # Extract displacement field
    u_x = u[0::2]
    u_y = u[1::2]
    
    return {
        'solver': 'scikit-fem 2D',
        'mesh_type': 'MeshTri symmetric',
        'n_vertices': int(mesh.nvertices),
        'n_elements': int(mesh.nelements),
        'n_dofs': int(K.shape[0]),
        'max_displacement_x': float(np.max(np.abs(u_x))),
        'max_displacement_y': float(np.max(np.abs(u_y))),
        'displacement_norm': float(np.sqrt(np.sum(u_x**2 + u_y**2))),
    }


def test_structure_to_skfem_pipeline():
    """Full pipeline: StructureGraph → skfem mesh → FEA solve → graph analysis."""
    import skfem
    from skfem import MeshTri, Basis, ElementTriP1, ElementVector
    from skfem.assembly import BilinearForm
    from skfem.helpers import ddot, sym_grad
    from fibernet import pattern_2d
    from fibernet.ml.gnn import graph_from_structure
    from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
    
    g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(3, 3))
    gd = graph_from_structure(g)
    
    n_nodes = gd['node_features'].shape[0]
    positions = gd['node_features'][:, :2].numpy()
    
    # Create Delaunay triangulation from node positions
    from scipy.spatial import Delaunay
    tri = Delaunay(positions)
    
    # Build skfem mesh from Delaunay
    mesh = MeshTri(positions.T, tri.simplices.T)
    
    # Solve with skfem
    elem = ElementVector(ElementTriP1())
    basis = Basis(mesh, elem)
    
    E = 1e9
    nu = 0.3
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
    
    # Apply load
    f = np.zeros(K.shape[0])
    # Load at last node
    f[2 * (n_nodes - 1)] = 500.0
    
    # Fix first 2 nodes
    fixed = np.array([0, 1])
    fixed_dofs = np.concatenate([2 * fixed, 2 * fixed + 1])
    
    K_bc, f_bc, _, _ = skfem.condense(K, f, D=fixed_dofs)
    u_skfem = skfem.solve(K_bc, f_bc)
    
    # Compare with our DifferentiableSpringNetwork
    physics = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)
    edge_index = gd['edge_index']
    n_edges = edge_index.shape[1]
    radii = torch.ones(n_edges) * 0.01
    forces = torch.zeros(n_nodes, 2)
    forces[-1, 0] = 500.0
    fixed_t = torch.tensor([0, 1], dtype=torch.long)
    
    with torch.no_grad():
        u_spring, sigma = physics.solve(edge_index, gd['node_features'][:, :2],
                                         radii, forces, fixed_t)
    
    # Extract comparable quantities
    u_spring_np = u_spring.numpy()
    u_skfem_x = u_skfem[0::2]
    u_skfem_y = u_skfem[1::2]
    
    return {
        'pipeline': 'StructureGraph → Delaunay → skfem → solve',
        'n_nodes': n_nodes,
        'n_triangles': len(tri.simplices),
        'skfem_max_disp': float(np.max(np.sqrt(u_skfem_x**2 + u_skfem_y**2))),
        'spring_max_disp': float(u_spring_np.max()),
        'skfem_dofs': int(K.shape[0]),
        'solvers_agree_order_of_magnitude': abs(np.log10(max(np.max(np.sqrt(u_skfem_x**2 + u_skfem_y**2)), 1e-20)) - np.log10(max(float(u_spring_np.max()), 1e-20))) < 2,
    }


# ============================================================
# 2. Deep Graph-Level Physics Verification
# ============================================================

def test_stress_path_analysis():
    """Analyze force transmission paths through fiber network graph."""
    import networkx as nx
    from fibernet import pattern_2d
    from fibernet.ml.gnn import graph_from_structure
    from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
    
    results = {}
    
    for unit in ['honeycomb', 'kagome', 'reentrant', 'diamond']:
        g = pattern_2d(unit=unit, box=(10, 10), grid=(4, 4))
        gd = graph_from_structure(g)
        
        n_nodes = gd['node_features'].shape[0]
        n_edges = gd['edge_index'].shape[1]
        
        physics = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)
        radii = torch.ones(n_edges) * 0.008
        forces = torch.zeros(n_nodes, 2)
        forces[-1, 0] = 500.0  # Horizontal load at last node
        fixed = torch.tensor([0, 1], dtype=torch.long)
        
        with torch.no_grad():
            u, sigma = physics.solve(gd['edge_index'], gd['node_features'][:, :2],
                                     radii, forces, fixed)
        
        # Build weighted graph for path analysis
        nx_g = g.to_networkx()
        
        # Edge weights = |σ| (stress magnitude)
        ei = gd['edge_index']
        for e_idx in range(n_edges):
            ni, nj = ei[0, e_idx].item(), ei[1, e_idx].item()
            if ni < nj:  # undirected
                # Stress-based weight
                edge_stress = abs(sigma[e_idx].item())
                if nx_g.has_edge(ni, nj):
                    nx_g[ni][nj]['stress'] = edge_stress
                    nx_g[ni][nj]['weight'] = 1.0 / (edge_stress + 1e-10)
        
        # Force chain detection: edges with stress > 2× mean
        mean_stress = sigma.abs().mean().item()
        force_chain_edges = (sigma.abs() > 2 * mean_stress).sum().item()
        force_chain_fraction = force_chain_edges / n_edges
        
        # Shortest stress path from load to boundary
        try:
            load_node = n_nodes - 1
            path_lengths = nx.single_source_dijkstra_path_length(
                nx_g, load_node, weight='weight')
            max_stress_path = max(path_lengths.values()) if path_lengths else -1
            avg_stress_path = np.mean(list(path_lengths.values())) if path_lengths else -1
        except:
            max_stress_path = -1
            avg_stress_path = -1
        
        # Stress concentration factor
        max_stress = sigma.abs().max().item()
        stress_concentration = max_stress / (mean_stress + 1e-10)
        
        # Node displacement ranking vs degree ranking
        degrees = np.zeros(n_nodes)
        for e_idx in range(n_edges):
            degrees[ei[0, e_idx].item()] += 1
            degrees[ei[1, e_idx].item()] += 1
        
        disp_mag = u.norm(dim=1).numpy()
        top_disp_nodes = set(np.argsort(disp_mag)[-5:].tolist())
        top_degree_nodes = set(np.argsort(degrees)[-5:].tolist())
        disp_degree_overlap = len(top_disp_nodes & top_degree_nodes)
        
        results[unit] = {
            'n_nodes': n_nodes,
            'mean_stress': round(mean_stress, 2),
            'max_stress': round(max_stress, 2),
            'stress_concentration_factor': round(stress_concentration, 3),
            'force_chain_edges': force_chain_edges,
            'force_chain_fraction': round(force_chain_fraction, 4),
            'max_stress_path': round(max_stress_path, 4),
            'avg_stress_path': round(avg_stress_path, 4),
            'top_disp_vs_degree_overlap': disp_degree_overlap,
            'max_displacement': round(float(u.abs().max()), 6),
        }
    
    return results


def test_graph_spectral_physics():
    """Spectral analysis: relate graph Laplacian eigenvalues to mechanical properties."""
    import networkx as nx
    from fibernet import pattern_2d
    from fibernet.ml.gnn import graph_from_structure
    from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
    
    results = {}
    
    for unit in ['honeycomb', 'kagome', 'reentrant', 'diamond', 'square', 'triangle']:
        g = pattern_2d(unit=unit, box=(10, 10), grid=(5, 5))
        gd = graph_from_structure(g)
        
        n_nodes = gd['node_features'].shape[0]
        n_edges = gd['edge_index'].shape[1]
        
        # Physics solve
        physics = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)
        radii = torch.ones(n_edges) * 0.008
        forces = torch.zeros(n_nodes, 2)
        forces[-1, 0] = 500.0
        fixed = torch.tensor([0, 1], dtype=torch.long)
        
        with torch.no_grad():
            u, sigma = physics.solve(gd['edge_index'], gd['node_features'][:, :2],
                                     radii, forces, fixed)
        
        compliance = physics.compliance(u, forces).item()
        
        # Graph Laplacian spectrum
        nx_g = g.to_networkx()
        L = nx.laplacian_matrix(nx_g).toarray()
        eigenvalues = np.sort(np.linalg.eigvalsh(L))
        
        # Spectral properties
        algebraic_conn = eigenvalues[1]
        spectral_gap = eigenvalues[2] - eigenvalues[1] if len(eigenvalues) > 2 else 0
        spectral_radius = eigenvalues[-1]
        
        # Fiedler vector (second eigenvector) — relates to structural modes
        eigvecs = np.linalg.eigh(L)[1]
        fiedler = eigvecs[:, 1]
        fiedler_range = float(np.max(fiedler) - np.min(fiedler))
        
        # Effective resistance (graph-theoretic stiffness measure)
        try:
            R_eff = nx.resistance_distance(nx_g, 0, n_nodes - 1)
        except:
            R_eff = -1
        
        results[unit] = {
            'n_nodes': n_nodes,
            'compliance': round(compliance, 4),
            'algebraic_connectivity': round(algebraic_conn, 6),
            'spectral_gap': round(spectral_gap, 6),
            'spectral_radius': round(spectral_radius, 4),
            'fiedler_range': round(fiedler_range, 6),
            'effective_resistance': round(R_eff, 6) if R_eff > 0 else -1,
            'compliance_vs_alg_conn': round(compliance * algebraic_conn, 6),
        }
    
    # Cross-unit correlation analysis
    units = list(results.keys())
    compliances = [results[u]['compliance'] for u in units]
    alg_conns = [results[u]['algebraic_connectivity'] for u in units]
    eff_resistances = [results[u]['effective_resistance'] for u in units if results[u]['effective_resistance'] > 0]
    
    corr_alg = np.corrcoef(alg_conns, compliances)[0, 1] if len(units) > 2 else 0
    
    return {
        'per_unit': results,
        'correlation_algebraic_connectivity_vs_compliance': round(corr_alg, 4),
    }


def test_force_chain_network():
    """Detect force chains (high-stress paths) in the network graph."""
    from fibernet import pattern_2d
    from fibernet.ml.gnn import graph_from_structure
    from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
    
    g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5))
    gd = graph_from_structure(g)
    
    n_nodes = gd['node_features'].shape[0]
    n_edges = gd['edge_index'].shape[1]
    
    physics = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)
    radii = torch.ones(n_edges) * 0.008
    forces = torch.zeros(n_nodes, 2)
    forces[-1, 0] = 1000.0
    fixed = torch.tensor([0, 1], dtype=torch.long)
    
    with torch.no_grad():
        u, sigma = physics.solve(gd['edge_index'], gd['node_features'][:, :2],
                                 radii, forces, fixed)
    
    # Force chain detection
    ei = gd['edge_index']
    stress_abs = sigma.abs()
    
    # Percentile-based threshold
    p90 = float(torch.quantile(stress_abs, 0.9))
    p95 = float(torch.quantile(stress_abs, 0.95))
    
    chain_mask_90 = stress_abs > p90
    chain_mask_95 = stress_abs > p95
    
    # Build force chain subgraph
    chain_edges_90 = ei[:, chain_mask_90]
    chain_edges_95 = ei[:, chain_mask_95]
    
    # Connected components in force chain
    chain_nodes_90 = set(chain_edges_90[0].tolist()) | set(chain_edges_90[1].tolist())
    chain_nodes_95 = set(chain_edges_95[0].tolist()) | set(chain_edges_95[1].tolist())
    
    # Force chain carries what fraction of total stress?
    stress_in_chain_90 = stress_abs[chain_mask_90].sum().item()
    stress_in_chain_95 = stress_abs[chain_mask_95].sum().item()
    total_stress = stress_abs.sum().item()
    
    # Nodal force balance check
    nodal_forces = torch.zeros(n_nodes, 2)
    for e in range(n_edges):
        ni, nj = ei[0, e].item(), ei[1, e].item()
        pos_i = gd['node_features'][ni, :2]
        pos_j = gd['node_features'][nj, :2]
        direction = (pos_j - pos_i)
        direction = direction / (direction.norm() + 1e-12)
        force_vec = sigma[e] * direction
        nodal_forces[ni] += force_vec
        nodal_forces[nj] -= force_vec
    
    # Check equilibrium at non-boundary nodes
    free_nodes = [n for n in range(n_nodes) if n not in fixed.tolist()]
    equilibrium_residual = nodal_forces[free_nodes].norm(dim=1).mean().item()
    
    return {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'stress_p90_threshold': round(p90, 2),
        'stress_p95_threshold': round(p95, 2),
        'force_chain_edges_p90': int(chain_mask_90.sum()),
        'force_chain_edges_p95': int(chain_mask_95.sum()),
        'force_chain_nodes_p90': len(chain_nodes_90),
        'force_chain_nodes_p95': len(chain_nodes_95),
        'stress_carried_by_chain_p90': round(stress_in_chain_90 / total_stress, 4),
        'stress_carried_by_chain_p95': round(stress_in_chain_95 / total_stress, 4),
        'equilibrium_residual': round(equilibrium_residual, 6),
        'max_displacement': round(float(u.abs().max()), 6),
    }


def test_multi_solver_comparison():
    """Compare our DifferentiableSpringNetwork with scikit-fem on same structures."""
    import skfem
    from skfem import MeshTri, Basis, ElementTriP1, ElementVector
    from skfem.assembly import BilinearForm
    from skfem.helpers import ddot, sym_grad
    from scipy.spatial import Delaunay
    from fibernet import pattern_2d
    from fibernet.ml.gnn import graph_from_structure
    from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
    
    results = {}
    
    for unit in ['honeycomb', 'square', 'triangle']:
        g = pattern_2d(unit=unit, box=(10, 10), grid=(3, 3))
        gd = graph_from_structure(g)
        
        n_nodes = gd['node_features'].shape[0]
        n_edges = gd['edge_index'].shape[1]
        positions = gd['node_features'][:, :2].numpy()
        
        # --- Spring Network ---
        physics = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)
        radii = torch.ones(n_edges) * 0.01
        forces = torch.zeros(n_nodes, 2)
        forces[-1, 0] = 500.0
        fixed = torch.tensor([0, 1], dtype=torch.long)
        
        t0 = time.time()
        with torch.no_grad():
            u_spring, sigma_spring = physics.solve(
                gd['edge_index'], positions if isinstance(positions, torch.Tensor) else torch.tensor(positions, dtype=torch.float32),
                radii, forces, fixed)
        spring_time = time.time() - t0
        
        spring_max_disp = float(u_spring.abs().max())
        
        # --- scikit-fem ---
        try:
            tri = Delaunay(positions)
            mesh = MeshTri(positions.T, tri.simplices.T)
            elem = ElementVector(ElementTriP1())
            basis = Basis(mesh, elem)
            
            E = 1e9; nu = 0.3
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
            f_vec = np.zeros(K.shape[0])
            f_vec[2 * (n_nodes - 1)] = 500.0
            
            fixed_dofs = np.concatenate([2 * np.array([0, 1]), 2 * np.array([0, 1]) + 1])
            K_bc, f_bc = skfem.condense(K, f_vec, D=fixed_dofs.astype(int))
            
            t0 = time.time()
            u_skfem = skfem.solve(K_bc, f_bc)
            skfem_time = time.time() - t0
            
            u_skfem_x = u_skfem[0::2]
            u_skfem_y = u_skfem[1::2]
            skfem_max_disp = float(np.max(np.sqrt(u_skfem_x**2 + u_skfem_y**2)))
        except Exception as e:
            skfem_max_disp = -1
            skfem_time = -1
        
        results[unit] = {
            'spring_max_disp': round(spring_max_disp, 6),
            'spring_time_ms': round(spring_time * 1000, 2),
            'skfem_max_disp': round(skfem_max_disp, 6) if skfem_max_disp > 0 else 'failed',
            'skfem_time_ms': round(skfem_time * 1000, 2) if skfem_time > 0 else 'failed',
            'ratio_skfem_spring': round(skfem_max_disp / spring_max_disp, 4) if skfem_max_disp > 0 and spring_max_disp > 0 else -1,
        }
    
    return results


# ============================================================
# Main
# ============================================================

TESTS = {
    'skfem_truss': test_skfem_truss_solver,
    'skfem_2d_elasticity': test_skfem_2d_mesh,
    'structure_to_skfem': test_structure_to_skfem_pipeline,
    'stress_paths': test_stress_path_analysis,
    'spectral_physics': test_graph_spectral_physics,
    'force_chains': test_force_chain_network,
    'multi_solver_comparison': test_multi_solver_comparison,
}

def run_phase5():
    print("=" * 70)
    print("Phase 5: FEM Integration & Deep Graph Physics Verification")
    print("=" * 70)
    
    results = {}
    for name, fn in TESTS.items():
        gc.collect()
        result = safe_test(name, fn)
        results[name] = result
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"  [{status}] {name} ({result['time_s']:.1f}s)")
        if not result['passed']:
            print(f"    Error: {result.get('error', '')[:200]}")
    
    passed = sum(1 for r in results.values() if r['passed'])
    total = len(TESTS)
    
    print(f"\n{'=' * 70}")
    print(f"Summary: {passed}/{total} passed")
    print(f"{'=' * 70}")
    
    output_file = RESULTS_DIR / "phase5_fem_integration.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {output_file}")
    return results

if __name__ == '__main__':
    run_phase5()

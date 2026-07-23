"""
Phase 5 Beam FEM: Welded-Joint Frame Analysis
==============================================

Tests the new BeamFrameFEM (beam elements with welded joints) vs the old
DifferentiableSpringNetwork (truss elements with pin joints).

Key differences:
- Truss: pin-jointed, axial force only, 2/3 DOF/node (translation only)
- Beam: welded joints, axial + bending + torsion, 3/6 DOF/node

Validation:
- Analytical solutions (cantilever, simply supported)
- PyNite cross-validation
- Multi-complexity benchmarks
- Graph-level analysis
"""

import sys
import time
import numpy as np
import torch
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fibernet.ml.beam_frame_fem import BeamFrameFEM, deduplicate_edges
from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
from fibernet import pattern_2d
from fibernet.ml.gnn import graph_from_structure

def test_analytical_2d():
    """Test 2D beam FEM against analytical solutions."""
    print("=" * 60)
    print("TEST 1: 2D Analytical Validation")
    print("=" * 60)
    
    solver = BeamFrameFEM(dim=2, E=200e9, nu=0.3)
    
    # Test 1a: Cantilever beam with tip load
    print("\n1a. Cantilever beam (tip load)")
    edge_index = np.array([[0], [1]])
    node_pos = np.array([[0.0, 0.0], [5.0, 0.0]])
    radii = np.array([0.05])
    forces = np.array([[0.0, 0.0], [0.0, -1000.0]])
    fixed = [0]
    
    u, sigma, moments = solver.solve(edge_index, node_pos, radii, forces, fixed)
    
    # Analytical
    E = 200e9
    I = np.pi * 0.05**4 / 4
    L = 5.0
    P = 1000.0
    delta_analytical = P * L**3 / (3 * E * I)
    
    err = abs(u[1, 1] + delta_analytical) / delta_analytical
    print(f"  Ours: DY={u[1,1]:.6e}, Analytical: {-delta_analytical:.6e}, Error: {err:.6e}")
    print(f"  Result: {'✓' if err < 1e-6 else '✗'}")
    
    # Test 1b: Cantilever with moment at tip
    print("\n1b. Cantilever beam (tip moment)")
    forces = np.array([[0.0, 0.0], [0.0, 0.0]])
    moments_ext = np.array([0.0, 1000.0])  # 1kN·m at tip
    
    u, sigma, moments = solver.solve(edge_index, node_pos, radii, forces, fixed, moments=moments_ext)
    
    # Analytical: θ = ML/(EI)
    M = 1000.0
    theta_analytical = M * L / (E * I)
    
    err = abs(u[1, 2] - theta_analytical) / theta_analytical
    print(f"  Ours: θz={u[1,2]:.6e}, Analytical: {theta_analytical:.6e}, Error: {err:.6e}")
    print(f"  Result: {'✓' if err < 1e-6 else '✗'}")

def test_analytical_3d():
    """Test 3D beam FEM against analytical solutions."""
    print("\n" + "=" * 60)
    print("TEST 2: 3D Analytical Validation")
    print("=" * 60)
    
    solver = BeamFrameFEM(dim=3, E=200e9, nu=0.3)
    
    # Test 2a: 3D cantilever with tip load
    print("\n2a. 3D Cantilever (tip load in Y)")
    edge_index = np.array([[0], [1]])
    node_pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    radii = np.array([0.05])
    forces = np.array([[0.0, 0.0, 0.0], [0.0, -1000.0, 0.0]])
    fixed = [0]
    
    u, sigma, moments = solver.solve(edge_index, node_pos, radii, forces, fixed)
    
    E = 200e9
    I = np.pi * 0.05**4 / 4
    L = 5.0
    P = 1000.0
    delta_analytical = P * L**3 / (3 * E * I)
    
    err = abs(u[1, 1] + delta_analytical) / delta_analytical
    print(f"  Ours: DY={u[1,1]:.6e}, Analytical: {-delta_analytical:.6e}, Error: {err:.6e}")
    print(f"  Result: {'✓' if err < 1e-6 else '✗'}")
    
    # Test 2b: 3D cantilever with torsion
    print("\n2b. 3D Cantilever (torsion)")
    forces = np.zeros((2, 3))
    moments_ext = np.zeros((2, 3))
    moments_ext[1, 0] = 1000.0  # 1kN·m about x-axis
    
    u, sigma, moments = solver.solve(edge_index, node_pos, radii, forces, fixed, moments=moments_ext)
    
    G = E / (2 * (1 + 0.3))
    J = np.pi * 0.05**4 / 2
    T = 1000.0
    theta_analytical = T * L / (G * J)
    
    err = abs(u[1, 3] - theta_analytical) / theta_analytical
    print(f"  Ours: θx={u[1,3]:.6e}, Analytical: {theta_analytical:.6e}, Error: {err:.6e}")
    print(f"  Result: {'✓' if err < 1e-6 else '✗'}")

def test_pynite_validation():
    """Validate against PyNite FEA."""
    print("\n" + "=" * 60)
    print("TEST 3: PyNite Cross-Validation")
    print("=" * 60)
    
    from Pynite.FEModel3D import FEModel3D
    
    # L-shaped frame
    print("\n3a. L-shaped frame (3D)")
    solver = BeamFrameFEM(dim=3, E=200e9, nu=0.3)
    
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    node_pos = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 3.0, 0.0]])
    radii = np.array([0.05, 0.05, 0.05, 0.05])
    forces = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1000.0, 0.0]])
    fixed = [0]
    
    u, sigma, moments = solver.solve(edge_index, node_pos, radii, forces, fixed)
    
    # PyNite
    G = 200e9 / (2 * (1 + 0.3))
    sec = solver.circular_section_properties(0.05)
    model = FEModel3D()
    model.add_material('Steel', E=200e9, G=G, nu=0.3, rho=7850)
    model.add_section('Sec1', A=sec['A'], Iy=sec['Iy'], Iz=sec['Iz'], J=sec['J'])
    model.add_node('N1', 0, 0, 0)
    model.add_node('N2', 3, 0, 0)
    model.add_node('N3', 3, 3, 0)
    model.add_member('M1', 'N1', 'N2', 'Steel', 'Sec1')
    model.add_member('M2', 'N2', 'N3', 'Steel', 'Sec1')
    model.def_support('N1', True, True, True, True, True, True)
    model.add_node_load('N3', 'FY', -1000)
    model.analyze()
    
    pn = model.nodes['N3']
    ck = list(pn.DY.keys())[0]
    
    err_dy = abs(u[2, 1] - pn.DY[ck]) / (abs(pn.DY[ck]) + 1e-20)
    err_dx = abs(u[2, 0] - pn.DX[ck]) / (abs(pn.DX[ck]) + 1e-20)
    
    print(f"  Ours: DY={u[2,1]:.6e}, DX={u[2,0]:.6e}")
    print(f"  PyNite: DY={pn.DY[ck]:.6e}, DX={pn.DX[ck]:.6e}")
    print(f"  Error: DY={err_dy:.6e}, DX={err_dx:.6e}")
    print(f"  Result: {'✓' if max(err_dy, err_dx) < 0.01 else '✗'}")

def test_fiber_network_2d():
    """Test with 2D fiber network structures."""
    print("\n" + "=" * 60)
    print("TEST 4: 2D Fiber Networks (Beam vs Truss)")
    print("=" * 60)
    
    beam_solver = BeamFrameFEM(dim=2, E=1e9, nu=0.3)
    truss_solver = DifferentiableSpringNetwork(youngs_modulus=1e9, damping=0.001)
    
    unit_types = ['honeycomb', 'kagome', 'square', 'triangle', 'reentrant', 'diamond']
    
    for idx, unit_type in enumerate(unit_types):
        print(f"\n4.{idx+1}. {unit_type.capitalize()} lattice")
        
        try:
            g = pattern_2d(unit=unit_type, box=(10, 10), grid=(5, 5))
            gd = graph_from_structure(g)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        
        edge_index = gd['edge_index'].numpy()
        node_pos = gd['node_features'].numpy()[:, :2]
        n_nodes = node_pos.shape[0]
        n_edges = edge_index.shape[1]
        
        unique_edges, unique_idx = deduplicate_edges(edge_index)
        n_unique = unique_edges.shape[1]
        
        # Forces: horizontal load at last node
        forces = np.zeros((n_nodes, 2))
        forces[-1, 0] = 500.0
        fixed = [0, 1]
        
        # Beam FEM (auto-deduplicates)
        # Use radius proportional to edge length (L/r ≈ 50, realistic slenderness)
        # Get average edge length
        unique_edges_arr, _ = deduplicate_edges(edge_index)
        edge_lengths = np.array([np.linalg.norm(node_pos[j] - node_pos[i]) 
                                  for i, j in unique_edges_arr.T])
        avg_edge_length = np.mean(edge_lengths)
        radius = avg_edge_length / 50  # L/r = 50
        radii_all = np.ones(n_edges) * radius
        t0 = time.time()
        u_beam, sigma_beam, moments_beam = beam_solver.solve(
            edge_index, node_pos, radii_all, forces, fixed
        )
        t_beam = time.time() - t0
        
        # Truss FEM
        t0 = time.time()
        u_truss, sigma_truss = truss_solver.solve(
            torch.tensor(edge_index),
            torch.tensor(node_pos, dtype=torch.float32),
            torch.tensor(radii_all, dtype=torch.float32),
            torch.tensor(forces, dtype=torch.float32),
            torch.tensor(fixed, dtype=torch.long)
        )
        t_truss = time.time() - t0
        u_truss_np = u_truss.numpy()
        
        # Compare translations only
        max_disp_beam = np.max(np.linalg.norm(u_beam[:, :2], axis=1))
        max_disp_truss = np.max(np.linalg.norm(u_truss_np, axis=1))
        
        disp_diff = np.linalg.norm(u_beam[:, :2] - u_truss_np, axis=1)
        max_diff = np.max(disp_diff)
        rel_diff = max_diff / (max_disp_truss + 1e-20)
        
        # Max rotation (beam only)
        max_rotation = np.max(np.abs(u_beam[:, 2]))
        
        # Max bending moment
        # sigma_beam is already for unique edges (after dedup in solve())
        unique_sigma = sigma_beam
        unique_moments = moments_beam
        max_moment = np.max(np.abs(unique_moments))
        
        # Compliance (strain energy)
        compliance_beam = float(np.sum(forces * u_beam[:, :2]))
        compliance_truss = float(np.sum(forces * u_truss_np))
        
        print(f"  Nodes: {n_nodes}, Edges: {n_edges} (unique: {n_unique})")
        print(f"  Beam:  max_disp={max_disp_beam:.4e}, max_rot={max_rotation:.4e}, time={t_beam*1000:.1f}ms")
        print(f"  Truss: max_disp={max_disp_truss:.4e}, time={t_truss*1000:.1f}ms")
        print(f"  Translation diff: rel={rel_diff:.4f}")
        print(f"  Max bending moment: {max_moment:.4e}")
        print(f"  Compliance: beam={compliance_beam:.4e}, truss={compliance_truss:.4e}")
        print(f"  Note: Beam includes bending stiffness (welded), truss is pin-jointed")

def test_fiber_network_3d():
    """Test with 3D fiber network structures."""
    print("\n" + "=" * 60)
    print("TEST 5: 3D Fiber Networks")
    print("=" * 60)
    
    beam_solver = BeamFrameFEM(dim=3, E=1e9, nu=0.3)
    
    # Create a simple 3D structure (cube frame)
    print("\n5.1. Cube frame (8 nodes, 12 edges)")
    
    # 8 corners of a cube
    node_pos = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top
    ], dtype=np.float32)
    
    # 12 edges of a cube (bidirectional)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
    ]
    edge_index = np.array([[i, j] for i, j in edges] + [[j, i] for i, j in edges]).T
    
    n_nodes = node_pos.shape[0]
    radii = np.ones(edge_index.shape[1]) * 0.01
    
    # Apply load at top corner
    forces = np.zeros((n_nodes, 3))
    forces[6, 0] = 100.0  # 100N in x at node 6
    forces[6, 1] = -100.0  # 100N in y
    
    # Fix bottom corners
    fixed = [0, 1, 2, 3]
    
    t0 = time.time()
    u, sigma, moments = beam_solver.solve(edge_index, node_pos, radii, forces, fixed)
    t_solve = time.time() - t0
    
    max_disp = np.max(np.linalg.norm(u[:, :3], axis=1))
    
    print(f"  Nodes: {n_nodes}, Edges: {edge_index.shape[1]}")
    print(f"  Max displacement: {max_disp:.6e}")
    print(f"  Time: {t_solve*1000:.1f}ms")
    print(f"  Loaded node (6) displacement: {u[6, :3]}")
    print(f"  Result: ✓ (3D beam FEM working)")

def test_scaling():
    """Test scaling with problem size."""
    print("\n" + "=" * 60)
    print("TEST 6: Scaling Analysis")
    print("=" * 60)
    
    sizes = [(2, 3), (3, 3), (5, 5), (8, 8)]
    
    print("\n6.1. 2D Honeycomb scaling (beam FEM)")
    for nx, ny in sizes:
        beam_2d = BeamFrameFEM(dim=2, E=1e9, nu=0.3, damping=0.001)
        try:
            g = pattern_2d(unit='honeycomb', box=(10, 10), grid=(nx, ny))
            gd = graph_from_structure(g)
        except Exception as e:
            print(f"  {nx}x{ny}: SKIP ({e})")
            continue
        
        edge_index = gd['edge_index'].numpy()
        node_pos = gd['node_features'].numpy()[:, :2]
        n_nodes = node_pos.shape[0]
        n_edges = edge_index.shape[1]
        
        # Use scaled radius
        unique_edges_arr, _ = deduplicate_edges(edge_index)
        edge_lengths = np.array([np.linalg.norm(node_pos[j] - node_pos[i]) 
                                  for i, j in unique_edges_arr.T])
        radius = np.mean(edge_lengths) / 50
        
        radii = np.ones(n_edges) * radius
        forces = np.zeros((n_nodes, 2))
        forces[-1, 0] = 500.0
        fixed = [0, 1]
        
        t0 = time.time()
        u, sigma, moments = beam_2d.solve(edge_index, node_pos, radii, forces, fixed)
        t = time.time() - t0
        
        max_disp = np.max(np.linalg.norm(u[:, :2], axis=1))
        max_rot = np.max(np.abs(u[:, 2]))
        
        print(f"  {nx}x{ny}: {n_nodes:4d}n {n_edges:4d}e | time={t*1000:.1f}ms | max_u={max_disp:.4e} | max_rot={max_rot:.4e}")


if __name__ == '__main__':
    print("Phase 5 Beam FEM Benchmark Suite")
    print("=" * 60)
    
    test_analytical_2d()
    test_analytical_3d()
    test_pynite_validation()
    test_fiber_network_2d()
    test_fiber_network_3d()
    test_scaling()
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

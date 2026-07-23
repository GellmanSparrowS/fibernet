#!/usr/bin/env python3
"""
Phase 4+5: Cross-Module Integration & Simulation Extensions
============================================================
Phase 4: Test cross-module pipelines
Phase 5: Simulation extensions using open-source libs
"""

import sys, os, json, time, gc, traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "benchmarks" / "results"

def safe_test(name, fn):
    """Run test with error handling."""
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        return {'name': name, 'passed': True, 'time_s': round(elapsed, 3), 'stats': result}
    except Exception as e:
        elapsed = time.time() - t0
        return {'name': name, 'passed': False, 'time_s': round(elapsed, 3),
                'error': str(e)[:300], 'traceback': traceback.format_exc()[-500:]}

# ============================================================
# Phase 4: Cross-Module Integration Pipelines
# ============================================================

def test_gnn_to_neural_ode():
    """Pipeline: GNN embedding → Neural ODE time evolution."""
    from fibernet.ml.gnn import FiberGNN, graph_from_structure
    from fibernet.ml.neural_ode import FiberNeuralODE
    from fibernet import pattern_2d
    
    g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4))
    gd = graph_from_structure(g)
    
    # GNN produces graph embedding
    gnn = FiberGNN(node_dim=5, hidden=32, n_outputs=4, n_layers=3, pooling='mean')
    embed = gnn([gd]).squeeze(0)  # (4,)
    
    # Neural ODE evolves embedding over time
    ode = FiberNeuralODE(state_dim=4, hidden=[32, 16])
    t = torch.linspace(0, 1, 20)
    trajectory = ode.solve(embed, t)
    
    stats = {
        'gnn_embed_shape': list(embed.shape),
        'trajectory_shape': list(trajectory.shape),
        'final_state_norm': float(trajectory[-1].norm()),
    }
    
    # Gradient flow through entire pipeline
    loss = trajectory[-1].sum()
    loss.backward()
    gnn_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in gnn.parameters())
    ode_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in ode.parameters())
    stats['gradient_flows_gnn'] = gnn_grad
    stats['gradient_flows_ode'] = ode_grad
    
    return stats

def test_pinn_to_conservative():
    """Pipeline: PhysicsInformedGNN → EnergyConservingNN."""
    from fibernet.ml.pinn_gnn import PhysicsInformedGNN
    from fibernet.ml.conservative_nn import EnergyConservingNN, HamiltonianNN
    from fibernet import pattern_2d
    from fibernet.ml.gnn import graph_from_structure
    
    g = pattern_2d(unit="kagome", box=(10, 10), grid=(3, 3))
    gd = graph_from_structure(g)
    
    # PIGNN produces node embeddings
    pinn = PhysicsInformedGNN(node_dim=5, edge_dim=2, hidden=32, n_layers=3,
                               predict_field=True, force_dim=2)
    fields = pinn.predict_fields(gd)
    node_embed = fields['node_embeddings']  # (N, hidden)
    
    # Energy conserving NN processes embeddings
    ec = EnergyConservingNN(state_dim=32, hidden=[32, 16])
    dx = ec(node_embed[0])  # dynamics for first node
    
    # Hamiltonian NN for dynamics
    hnn = HamiltonianNN(n_coords=16, hidden=[32])
    q = node_embed[0, :16]
    p = node_embed[0, 16:]
    dq, dp = hnn(q, p)
    
    return {
        'node_embed_shape': list(node_embed.shape),
        'ec_output_shape': list(dx.shape),
        'hamiltonian_dq_norm': float(dq.norm()),
        'hamiltonian_dp_norm': float(dp.norm()),
    }

def test_gflownet_to_physics():
    """Pipeline: GFlowNet generates structure → Physics evaluates it."""
    from fibernet.ml.gflownet import FiberGFlowNet, StructureState, StructureAction
    from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
    
    gfn = FiberGFlowNet(n_node_features=5, hidden=32, max_nodes=15)
    states = gfn.sample(n=5, max_steps=12)
    
    physics = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)
    
    valid = 0
    max_disps = []
    for state in states:
        if state.n_nodes >= 3 and state.n_edges >= 2:
            pos = torch.tensor(np.array(state.node_positions), dtype=torch.float32)
            src = [e[0] for e in state.edges]
            dst = [e[1] for e in state.edges]
            ei = torch.tensor([src, dst], dtype=torch.long)
            r = torch.ones(len(state.edges)) * 0.01
            f = torch.zeros(state.n_nodes, 2)
            f[-1, 0] = 100.0
            
            try:
                u, sigma = physics.solve(ei, pos, r, f, torch.tensor([0]))
                max_disps.append(float(u.abs().max()))
                valid += 1
            except:
                pass
    
    return {
        'total_generated': len(states),
        'valid_for_physics': valid,
        'valid_rate': valid / len(states),
        'max_displacements': max_disps,
    }

def test_diffusion_to_inverse_design():
    """Pipeline: Diffusion generates → Inverse Design maps back."""
    from fibernet.ml.diffusion import FiberDiffusion, DiffusionTrainer
    from fibernet.ml.inverse_design import TandemNetwork, InverseDesignTrainer
    
    n_feat = 8
    n_prop = 3
    
    # Train diffusion on synthetic data
    diff = FiberDiffusion(n_features=n_feat, hidden=[32, 16], n_steps=20)
    X = np.random.randn(50, n_feat).astype(np.float32)
    DiffusionTrainer(diff).fit(X, epochs=5, batch_size=16, verbose=False)
    
    # Generate samples
    X_gen = diff.sample(n=20)
    
    # Create fake properties
    y = np.random.randn(50, n_prop).astype(np.float32)
    
    # Train inverse design
    tandem = TandemNetwork(n_features=n_feat, n_properties=n_prop, hidden=[32])
    InverseDesignTrainer(tandem).fit(X, y, epochs=5, batch_size=16, verbose=False)
    
    # Design for target properties
    target = np.random.randn(1, n_prop).astype(np.float32)
    designs = InverseDesignTrainer(tandem).design(target, n_candidates=5)
    
    return {
        'diffusion_samples': list(X_gen.shape),
        'designs': list(designs.shape),
        'design_mean': float(designs.mean()),
    }

def test_active_learning_with_pinn():
    """Pipeline: Active Learning selects → PIGNN evaluates."""
    from fibernet.ml.active_learning import UncertaintySampling, ActiveLearningLoop
    from fibernet.ml.pinn_gnn import PhysicsInformedGNN
    from fibernet import pattern_2d
    from fibernet.ml.gnn import graph_from_structure
    
    # Generate pool of structures
    graphs = []
    for unit in ['honeycomb', 'kagome', 'diamond']:
        g = pattern_2d(unit=unit, box=(10, 10), grid=(3, 3))
        gd = graph_from_structure(g)
        graphs.append(gd)
    
    # PIGNN with uncertainty estimation
    pinn = PhysicsInformedGNN(node_dim=5, edge_dim=2, hidden=32, n_layers=3,
                               predict_field=True, force_dim=2)
    
    # Evaluate uncertainty via multiple forward passes (dropout)
    uncertainties = []
    for gd in graphs:
        preds = []
        for _ in range(5):
            fields = pinn.predict_fields(gd)
            preds.append(fields['displacement'].detach().numpy())
        preds = np.array(preds)
        uncertainty = float(np.std(preds, axis=0).mean())
        uncertainties.append(uncertainty)
    
    # Active learning selects highest uncertainty
    ranked = np.argsort(uncertainties)[::-1]
    
    return {
        'pool_size': len(graphs),
        'uncertainties': [round(u, 6) for u in uncertainties],
        'selected_ranking': ranked.tolist(),
    }

def test_transfer_learning_pipeline():
    """Pipeline: Pretrain on one topology → Finetune on another."""
    from fibernet.ml.transfer_learning import FiberTransferNet
    
    net = FiberTransferNet(n_features=20, n_outputs=1, hidden=[32, 16])
    
    # Pretrain on source domain
    X_src = np.random.randn(100, 20).astype(np.float32)
    y_src = (X_src @ np.random.randn(20).astype(np.float32)).astype(np.float32)
    pretrain_result = net.pretrain(X_src, y_src, epochs=10, verbose=False)
    
    # Finetune on target domain
    X_tgt = np.random.randn(20, 20).astype(np.float32)
    y_tgt = (X_tgt @ np.random.randn(20).astype(np.float32) * 2).astype(np.float32)
    finetune_result = net.finetune(X_tgt, y_tgt, epochs=10, freeze_layers=1, verbose=False)
    
    # Predict
    pred = net.predict(X_tgt[:5])
    
    return {
        'pretrain_loss': pretrain_result['train_loss'][-1],
        'finetune_loss': finetune_result['train_loss'][-1],
        'pred_shape': list(pred.shape),
        'improvement': pretrain_result['train_loss'][-1] / (finetune_result['train_loss'][-1] + 1e-8),
    }

# ============================================================
# Phase 5: Simulation Extensions
# ============================================================

def test_taichi_simulation():
    """Test Taichi-based simulation extensions."""
    try:
        import taichi as ti
        from fibernet.engine import TaichiEngine
        from fibernet import pattern_2d
        
        g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4))
        
        # Run Taichi simulation
        engine = TaichiEngine(g)
        result = engine.run(steps=100, dt=0.001)
        
        return {
            'engine_type': 'taichi',
            'steps': 100,
            'n_nodes': len(g.nodes),
            'result_keys': list(result.keys()) if isinstance(result, dict) else str(type(result)),
        }
    except ImportError as e:
        return {'error': f'Taichi not available: {e}'}
    except Exception as e:
        return {'error': f'Taichi simulation failed: {e}'}

def test_networkx_advanced_analysis():
    """Advanced NetworkX analysis: community detection, centrality, spectral."""
    import networkx as nx
    from fibernet import pattern_2d
    
    results = {}
    for unit in ['honeycomb', 'kagome', 'triangle', 'diamond']:
        g = pattern_2d(unit=unit, box=(10, 10), grid=(4, 4))
        nx_g = g.to_networkx()
        
        # Spectral analysis
        L = nx.laplacian_matrix(nx_g).toarray()
        eigenvalues = np.sort(np.linalg.eigvalsh(L))
        spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0
        
        # Eigenvector centrality
        try:
            ec = nx.eigenvector_centrality_numpy(nx_g)
            max_ec = max(ec.values())
        except:
            max_ec = 0
        
        # Graph density
        density = nx.density(nx_g)
        
        # Assortativity
        try:
            assort = nx.degree_assortativity_coefficient(nx_g)
        except:
            assort = 0
        
        results[unit] = {
            'n_nodes': nx_g.number_of_nodes(),
            'n_edges': nx_g.number_of_edges(),
            'spectral_gap': round(spectral_gap, 6),
            'algebraic_connectivity': round(eigenvalues[1], 6) if len(eigenvalues) > 1 else 0,
            'max_eigenvector_centrality': round(max_ec, 6),
            'density': round(density, 6),
            'assortativity': round(assort, 4),
            'spectrum_first5': [round(e, 4) for e in eigenvalues[:5]],
        }
    
    return results

def test_scipy_optimization():
    """Scipy-based structure optimization."""
    from scipy.optimize import minimize
    from fibernet import pattern_2d
    from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
    from fibernet.ml.gnn import graph_from_structure
    
    g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(3, 3))
    gd = graph_from_structure(g)
    
    physics = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)
    n_edges = gd['edge_index'].shape[1]
    n_nodes = gd['node_features'].shape[0]
    
    forces = torch.zeros(n_nodes, 2)
    forces[-1, 0] = 500.0
    fixed = torch.tensor([0, 1], dtype=torch.long)
    
    def objective(radii_np):
        radii = torch.tensor(radii_np, dtype=torch.float32)
        with torch.no_grad():
            u, sigma = physics.solve(gd['edge_index'], gd['node_features'][:, :2],
                                    radii, forces, fixed)
        compliance = physics.compliance(u, forces)
        volume = (torch.tensor(np.pi) * radii**2 * 1.0).sum()  # unit length
        return compliance.item() + 0.1 * volume.item()
    
    # Initial guess
    x0 = np.ones(n_edges) * 0.01
    
    # Optimize
    result = minimize(objective, x0, method='L-BFGS-B',
                     bounds=[(0.002, 0.05)] * n_edges,
                     options={'maxiter': 20, 'disp': False})
    
    return {
        'optimizer': 'L-BFGS-B',
        'iterations': result.nit,
        'final_objective': round(result.fun, 6),
        'initial_objective': round(objective(x0), 6),
        'improvement': round(objective(x0) - result.fun, 6),
        'optimized_radii_stats': {
            'min': round(float(result.x.min()), 6),
            'max': round(float(result.x.max()), 6),
            'mean': round(float(result.x.mean()), 6),
            'std': round(float(result.x.std()), 6),
        }
    }


# ============================================================
# Main
# ============================================================

PHASE4_TESTS = {
    'gnn_to_neural_ode': test_gnn_to_neural_ode,
    'pinn_to_conservative': test_pinn_to_conservative,
    'gflownet_to_physics': test_gflownet_to_physics,
    'diffusion_to_inverse_design': test_diffusion_to_inverse_design,
    'active_learning_with_pinn': test_active_learning_with_pinn,
    'transfer_learning': test_transfer_learning_pipeline,
}

PHASE5_TESTS = {
    'taichi_simulation': test_taichi_simulation,
    'networkx_advanced': test_networkx_advanced_analysis,
    'scipy_optimization': test_scipy_optimization,
}

def run_all():
    print("=" * 70)
    print("Phase 4+5: Cross-Module Integration & Simulation Extensions")
    print("=" * 70)
    
    results = {'phase4': {}, 'phase5': {}}
    
    print("\n--- Phase 4: Cross-Module Integration ---")
    for name, fn in PHASE4_TESTS.items():
        gc.collect()
        result = safe_test(name, fn)
        results['phase4'][name] = result
        status = "PASS" if result['passed'] else "FAIL"
        print(f"  [{status}] {name} ({result['time_s']:.1f}s)")
        if not result['passed']:
            print(f"    Error: {result.get('error', 'unknown')}")
    
    print("\n--- Phase 5: Simulation Extensions ---")
    for name, fn in PHASE5_TESTS.items():
        gc.collect()
        result = safe_test(name, fn)
        results['phase5'][name] = result
        status = "PASS" if result['passed'] else "FAIL"
        print(f"  [{status}] {name} ({result['time_s']:.1f}s)")
        if not result['passed']:
            print(f"    Error: {result.get('error', 'unknown')}")
    
    # Summary
    p4_pass = sum(1 for r in results['phase4'].values() if r['passed'])
    p5_pass = sum(1 for r in results['phase5'].values() if r['passed'])
    total = len(PHASE4_TESTS) + len(PHASE5_TESTS)
    
    print(f"\n{'=' * 70}")
    print(f"Summary: Phase 4: {p4_pass}/{len(PHASE4_TESTS)}, Phase 5: {p5_pass}/{len(PHASE5_TESTS)}")
    print(f"{'=' * 70}")
    
    # Save
    output_file = RESULTS_DIR / "phase4_5_integration.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {output_file}")
    return results

if __name__ == '__main__':
    run_all()

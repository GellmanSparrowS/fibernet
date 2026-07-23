#!/usr/bin/env python3
"""
Phase 2: Synthetic Data End-to-End Pipeline with Graph-Level Verification
=========================================================================
1. Generate diverse fiber networks with ground truth physics
2. Train GNN for graph-level property prediction (compliance, max_stress)
3. Train PhysicsInformedGNN for node-level displacement fields
4. Graph-level verification: topology-aware metrics, stress paths

Usage:
    python benchmarks/phase2_synthetic_pipeline.py [--epochs 50] [--n_samples 60]
"""

import sys, os, json, time, argparse, gc, traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# Data Generation
# ============================================================

def generate_diverse_dataset(n_samples=60, seed=42):
    """Generate diverse fiber networks with ground truth physics."""
    from fibernet import pattern_2d
    from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
    from fibernet.ml.gnn import graph_from_structure

    rng = np.random.RandomState(seed)
    physics = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)

    units = ['honeycomb', 'kagome', 'reentrant', 'diamond', 'square', 'triangle', 'voronoi']
    dataset = []

    for i in range(n_samples):
        unit = units[i % len(units)]
        grid_size = rng.randint(2, 7)  # 2-6 grid

        try:
            g = pattern_2d(unit=unit, box=(10, 10), grid=(grid_size, grid_size),
                          seed=seed + i if unit == 'voronoi' else None)
        except Exception as e:
            continue

        n_nodes = len(g.nodes)
        if n_nodes < 5:
            continue

        # Convert to graph dict
        gd = graph_from_structure(g)

        # Use edge_index size (bidirectional = 2 * undirected edges)
        n_edge_entries = gd['edge_index'].shape[1]

        # Random radii (design variables)
        radii = torch.ones(n_edge_entries) * (0.005 + rng.uniform(0, 0.01))

        # Random forces
        actual_n_nodes = gd['node_features'].shape[0]
        forces = torch.zeros(actual_n_nodes, 2)
        n_forced = max(1, rng.randint(1, 4))
        forced = rng.choice(actual_n_nodes, min(n_forced, actual_n_nodes - 2), replace=False)
        for fn in forced:
            forces[fn] = torch.tensor(rng.randn(2).astype(np.float32)) * 500.0

        # Fixed boundary
        n_fixed = max(2, actual_n_nodes // 10)
        fixed = torch.tensor(rng.choice(actual_n_nodes, min(n_fixed, actual_n_nodes - 2), replace=False), dtype=torch.long)

        # Ground truth solve
        try:
            with torch.no_grad():
                u, sigma = physics.solve(gd['edge_index'], gd['node_features'][:, :2],
                                        radii, forces, fixed)
            compliance = physics.compliance(u, forces)
        except Exception as e:
            continue

        dataset.append({
            'graph_data': gd,
            'structure_graph': g,
            'radii': radii,
            'forces': forces,
            'fixed': fixed,
            'u_true': u,
            'sigma_true': sigma,
            'n_nodes_actual': actual_n_nodes,
            'compliance': float(compliance.item()),
            'max_displacement': float(u.abs().max().item()),
            'max_stress': float(sigma.abs().max().item()),
            'unit': unit,
            'n_nodes': n_nodes,
            'n_edges': n_edge_entries,
        })

        if len(dataset) >= n_samples:
            break

        gc.collect()

    return dataset


# ============================================================
# GNN Training: Graph-Level Property Prediction
# ============================================================

def train_gnn_graph_level(dataset, epochs=80, verbose=True):
    """Train FiberGNN to predict graph-level properties (compliance, max_stress)."""
    from fibernet.ml.gnn import FiberGNN

    if verbose:
        print("\n[GNN Graph-Level Training]")
        print(f"  Dataset: {len(dataset)} structures")

    # Prepare labels
    labels = np.array([d['compliance'] for d in dataset], dtype=np.float32)
    # Use log-transform for compliance (spans orders of magnitude)
    labels_log = np.log1p(labels)  # log(1+x) to handle small values
    label_mean = labels_log.mean()
    label_std = labels_log.std() + 1e-8
    labels_norm = (labels_log - label_mean) / label_std

    # Get node_dim from data
    node_dim = dataset[0]['graph_data']['node_features'].shape[1]

    gnn = FiberGNN(node_dim=node_dim, hidden=64, n_outputs=1, n_layers=4,
                   layer_type='gcn', pooling='attention')

    optimizer = torch.optim.Adam(gnn.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n = len(dataset)
    n_train = int(n * 0.8)
    n_val = n - n_train
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    graphs = [d['graph_data'] for d in dataset]
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}

    t0 = time.time()
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        gnn.train()
        train_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, 16):
            end = min(start + 16, n_train)
            batch_g = [graphs[train_idx[j]] for j in range(start, end)]
            batch_y = torch.tensor(labels_norm[train_idx[start:end]], dtype=torch.float32)

            optimizer.zero_grad()
            pred = gnn(batch_g).squeeze(-1)
            loss = F.mse_loss(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= max(n_batches, 1)

        # Validation
        gnn.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_preds = []
            val_trues = []
            for j in range(n_val):
                g = graphs[val_idx[j]]
                p = gnn([g]).squeeze(-1)
                val_preds_log = p.item() * label_std + label_mean
                val_preds.append(np.expm1(val_preds_log))  # inverse of log1p
                val_trues.append(labels[val_idx[j]])

            val_preds = np.array(val_preds)
            val_trues = np.array(val_trues)
            val_loss = float(np.mean((val_preds - val_trues) ** 2))
            ss_res = np.sum((val_trues - val_preds) ** 2)
            ss_tot = np.sum((val_trues - val_trues.mean()) ** 2)
            val_r2 = 1 - ss_res / (ss_tot + 1e-12)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in gnn.state_dict().items()}

        if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:3d}: train={train_loss:.4f}, val_mse={val_loss:.4f}, val_R²={val_r2:.4f}")

    train_time = time.time() - t0

    # Restore best model
    if best_state is not None:
        gnn.load_state_dict(best_state)

    # Final evaluation
    gnn.eval()
    with torch.no_grad():
        preds_all = []
        trues_all = []
        for d in dataset:
            p = gnn([d['graph_data']]).squeeze(-1)
            pred_log = p.item() * label_std + label_mean
            preds_all.append(np.expm1(pred_log))  # inverse of log1p
            trues_all.append(d['compliance'])
        preds_all = np.array(preds_all)
        trues_all = np.array(trues_all)

        mae = float(np.mean(np.abs(preds_all - trues_all)))
        rmse = float(np.sqrt(np.mean((preds_all - trues_all) ** 2)))
        ss_res = np.sum((trues_all - preds_all) ** 2)
        ss_tot = np.sum((trues_all - trues_all.mean()) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-12))

    if verbose:
        print(f"  Final: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        print(f"  Train time: {train_time:.1f}s, Params: {sum(p.numel() for p in gnn.parameters())}")

    return {
        'gnn': gnn,
        'history': history,
        'final_mae': mae,
        'final_rmse': rmse,
        'final_r2': r2,
        'train_time': train_time,
        'n_params': sum(p.numel() for p in gnn.parameters()),
        'label_mean': label_mean,
        'label_std': label_std,
    }


# ============================================================
# PhysicsInformedGNN: Node-Level Field Prediction
# ============================================================

def train_pinn_gnn_node_level(dataset, epochs=60, verbose=True):
    """Train PhysicsInformedGNN for per-node displacement prediction."""
    from fibernet.ml.pinn_gnn import PhysicsInformedGNN

    if verbose:
        print("\n[PhysicsInformedGNN Node-Level Training]")

    node_dim = dataset[0]['graph_data']['node_features'].shape[1]
    edge_dim = dataset[0]['graph_data']['edge_features'].shape[1]

    # Compute displacement statistics for normalization
    all_u = torch.cat([d['u_true'].flatten() for d in dataset])
    u_mean = all_u.mean().item()
    u_std = all_u.std().item() + 1e-8
    
    # Normalize targets
    for d in dataset:
        d['u_norm'] = (d['u_true'] - u_mean) / u_std

    # Model: output mode = "node" for per-node displacement
    pinn = PhysicsInformedGNN(
        node_dim=node_dim, edge_dim=edge_dim, hidden=64, n_layers=4,
        n_outputs=2, output_mode='node', predict_field=True, force_dim=2,
        physics_weight=0.1, youngs_modulus=1e9
    )

    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n = len(dataset)
    n_train = int(n * 0.8)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    history = {'train_loss': [], 'val_loss': [], 'train_corr': []}

    t0 = time.time()
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        pinn.train()
        train_loss = 0.0
        n_batches = 0

        for j in range(n_train):
            d = dataset[train_idx[j]]
            gd = d['graph_data']
            u_norm = d['u_norm']

            optimizer.zero_grad()

            # Node-level prediction via predict_fields
            fields = pinn.predict_fields(gd)
            pred_disp = fields['displacement']

            # Data loss: MSE between predicted and normalized displacement
            data_loss = F.mse_loss(pred_disp, u_norm)

            # Physics loss
            physics_losses = pinn.compute_physics_loss(gd, d['forces'])
            physics_loss = physics_losses['total_physics']

            loss = data_loss + 0.05 * physics_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
            optimizer.step()

            train_loss += data_loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= max(n_batches, 1)

        # Validation
        val_loss = 0.0
        val_corrs = []
        with torch.no_grad():
            for j in range(len(val_idx)):
                d = dataset[val_idx[j]]
                gd = d['graph_data']
                u_norm = d['u_norm']

                fields = pinn.predict_fields(gd)
                pred = fields['displacement']

                val_loss += F.mse_loss(pred, u_norm).item()

                # Correlation
                pf = pred.flatten()
                tf = u_norm.flatten()
                if pf.std() > 1e-8 and tf.std() > 1e-8:
                    corr = torch.corrcoef(torch.stack([pf, tf]))[0, 1]
                    val_corrs.append(corr.item())

        val_loss /= max(len(val_idx), 1)
        avg_corr = float(np.mean(val_corrs)) if val_corrs else 0.0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_corr'].append(avg_corr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in pinn.state_dict().items()}

        if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:3d}: train={train_loss:.6f}, val={val_loss:.6f}, corr={avg_corr:.4f}")

    train_time = time.time() - t0

    if best_state is not None:
        pinn.load_state_dict(best_state)

    if verbose:
        print(f"  Train time: {train_time:.1f}s, Params: {sum(p.numel() for p in pinn.parameters())}")

    return {
        'pinn': pinn,
        'history': history,
        'best_val_loss': best_val_loss,
        'train_time': train_time,
        'n_params': sum(p.numel() for p in pinn.parameters()),
        'u_mean': u_mean,
        'u_std': u_std,
    }


# ============================================================
# Graph-Level Verification
# ============================================================

def graph_level_verification(dataset, pinn, u_std=1.0, u_mean=0.0, verbose=True):
    """Verify predictions at graph topology level."""
    if verbose:
        print("\n[Graph-Level Verification]")

    pinn.eval()
    results = []

    with torch.no_grad():
        for i, d in enumerate(dataset[:15]):
            gd = d['graph_data']
            u_true = d['u_true']
            sigma_true = d['sigma_true']

            fields = pinn.predict_fields(gd)
            pred_raw = fields['displacement'] * u_std + u_mean  # un-normalize
            pred = pred_raw

            # --- Metric 1: Displacement magnitude ranking ---
            pred_mag = pred.norm(dim=1)
            true_mag = u_true.norm(dim=1)

            n_top = max(3, len(pred_mag) // 5)  # Top 20%
            pred_top = set(torch.topk(pred_mag, n_top).indices.tolist())
            true_top = set(torch.topk(true_mag, n_top).indices.tolist())
            top_overlap = len(pred_top & true_top) / n_top

            # --- Metric 2: Overall correlation ---
            pf = pred.flatten()
            tf = u_true.flatten()
            if pf.std() > 1e-8 and tf.std() > 1e-8:
                corr = torch.corrcoef(torch.stack([pf, tf]))[0, 1].item()
            else:
                corr = 0.0

            # --- Metric 3: Force transmission path ---
            ei = gd['edge_index']
            pred_edge = (pred[ei[0]] - pred[ei[1]]).norm(dim=1)
            true_edge = (u_true[ei[0]] - u_true[ei[1]]).norm(dim=1)

            n_top_edge = max(3, len(pred_edge) // 10)  # Top 10%
            pred_top_e = set(torch.topk(pred_edge, n_top_edge).indices.tolist())
            true_top_e = set(torch.topk(true_edge, n_top_edge).indices.tolist())
            edge_overlap = len(pred_top_e & true_top_e) / n_top_edge

            # --- Metric 4: Degree-aware error ---
            degrees = torch.zeros(len(gd['node_features']))
            for e_idx in range(ei.shape[1]):
                degrees[ei[0, e_idx]] += 1
                degrees[ei[1, e_idx]] += 1

            high_deg_mask = degrees > degrees.median()
            low_deg_mask = ~high_deg_mask

            high_deg_error = float((pred[high_deg_mask] - u_true[high_deg_mask]).abs().mean().item()) if high_deg_mask.sum() > 0 else 0
            low_deg_error = float((pred[low_deg_mask] - u_true[low_deg_mask]).abs().mean().item()) if low_deg_mask.sum() > 0 else 0

            # --- Metric 5: Boundary vs interior ---
            fixed_mask = torch.zeros(len(gd['node_features']), dtype=torch.bool)
            fixed_mask[d['fixed']] = True
            free_mask = ~fixed_mask

            boundary_error = float((pred[fixed_mask] - u_true[fixed_mask]).abs().mean().item()) if fixed_mask.sum() > 0 else 0
            interior_error = float((pred[free_mask] - u_true[free_mask]).abs().mean().item()) if free_mask.sum() > 0 else 0

            # --- Metric 6: Sign agreement ---
            sign_agree = float((torch.sign(pred) == torch.sign(u_true)).float().mean().item())

            results.append({
                'idx': i,
                'unit': d['unit'],
                'n_nodes': d['n_nodes'],
                'n_edges': d['n_edges'],
                'top20_node_overlap': round(top_overlap, 3),
                'correlation': round(corr, 4),
                'top10_edge_overlap': round(edge_overlap, 3),
                'high_deg_error': round(high_deg_error, 6),
                'low_deg_error': round(low_deg_error, 6),
                'boundary_error': round(boundary_error, 6),
                'interior_error': round(interior_error, 6),
                'sign_agreement': round(sign_agree, 4),
                'max_true_disp': round(float(u_true.abs().max()), 6),
                'max_pred_disp': round(float(pred.abs().max()), 6),
            })

    # Summary statistics
    if verbose:
        print(f"\n  Summary over {len(results)} graphs:")
        avg_corr = np.mean([r['correlation'] for r in results])
        avg_node = np.mean([r['top20_node_overlap'] for r in results])
        avg_edge = np.mean([r['top10_edge_overlap'] for r in results])
        avg_sign = np.mean([r['sign_agreement'] for r in results])

        print(f"    Avg correlation: {avg_corr:.4f}")
        print(f"    Avg top-20% node overlap: {avg_node:.3f}")
        print(f"    Avg top-10% edge overlap: {avg_edge:.3f}")
        print(f"    Avg sign agreement: {avg_sign:.4f}")
        print(f"\n  Per-graph details:")
        for r in results:
            print(f"    [{r['unit']:12s}] {r['n_nodes']:3d}n/{r['n_edges']:3d}e | "
                  f"corr={r['correlation']:+.3f} node_ov={r['top20_node_overlap']:.2f} "
                  f"edge_ov={r['top10_edge_overlap']:.2f} sign={r['sign_agreement']:.2f}")

    return results


# ============================================================
# Main
# ============================================================

def run_phase2(epochs_gnn=80, epochs_pinn=60, n_samples=60):
    print("=" * 70)
    print("Phase 2: Synthetic Data End-to-End Pipeline")
    print("=" * 70)

    # Step 1: Generate dataset
    print(f"\n[Generating {n_samples} structures with ground truth physics...]")
    t0 = time.time()
    dataset = generate_diverse_dataset(n_samples=n_samples)
    gen_time = time.time() - t0
    print(f"  Generated {len(dataset)} structures in {gen_time:.1f}s")
    print(f"  Units: {set(d['unit'] for d in dataset)}")
    print(f"  Nodes: {min(d['n_nodes'] for d in dataset)}-{max(d['n_nodes'] for d in dataset)}")
    print(f"  Compliance range: [{min(d['compliance'] for d in dataset):.4f}, {max(d['compliance'] for d in dataset):.4f}]")

    # Step 2: GNN graph-level training
    gnn_result = train_gnn_graph_level(dataset, epochs=epochs_gnn)

    # Step 3: PINN_GNN node-level training
    pinn_result = train_pinn_gnn_node_level(dataset, epochs=epochs_pinn)

    # Step 4: Graph-level verification
    u_std = pinn_result['u_std']
    u_mean = pinn_result['u_mean']
    graph_results = graph_level_verification(dataset, pinn_result['pinn'], u_std=u_std, u_mean=u_mean)

    # Summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(dataset),
        'gnn': {
            'mae': gnn_result['final_mae'],
            'rmse': gnn_result['final_rmse'],
            'r2': gnn_result['final_r2'],
            'n_params': gnn_result['n_params'],
            'train_time': gnn_result['train_time'],
        },
        'pinn_gnn': {
            'best_val_loss': pinn_result['best_val_loss'],
            'n_params': pinn_result['n_params'],
            'train_time': pinn_result['train_time'],
        },
        'graph_verification': graph_results,
        'graph_verification_summary': {
            'avg_correlation': float(np.mean([r['correlation'] for r in graph_results])),
            'avg_node_overlap': float(np.mean([r['top20_node_overlap'] for r in graph_results])),
            'avg_edge_overlap': float(np.mean([r['top10_edge_overlap'] for r in graph_results])),
            'avg_sign_agreement': float(np.mean([r['sign_agreement'] for r in graph_results])),
        }
    }

    # Save
    output_file = RESULTS_DIR / "phase2_results.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print("Phase 2 Summary")
    print(f"{'=' * 70}")
    print(f"GNN (compliance): R²={summary['gnn']['r2']:.4f}, MAE={summary['gnn']['mae']:.4f}")
    print(f"PINN_GNN (displacement): val_loss={summary['pinn_gnn']['best_val_loss']:.6f}")
    gv = summary['graph_verification_summary']
    print(f"Graph verification: corr={gv['avg_correlation']:.4f}, "
          f"node_ov={gv['avg_node_overlap']:.3f}, edge_ov={gv['avg_edge_overlap']:.3f}, "
          f"sign={gv['avg_sign_agreement']:.4f}")
    print(f"\nResults saved to: {output_file}")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs_gnn', type=int, default=80)
    parser.add_argument('--epochs_pinn', type=int, default=60)
    parser.add_argument('--n_samples', type=int, default=60)
    args = parser.parse_args()

    run_phase2(args.epochs_gnn, args.epochs_pinn, args.n_samples)


# ============================================================
# NetworkX-based Graph Analysis (Phase 2b)
# ============================================================

def networkx_graph_analysis(dataset, verbose=True):
    """Deep graph-level analysis using NetworkX."""
    import networkx as nx
    
    if verbose:
        print("\n[NetworkX Graph Analysis]")
    
    results = []
    for i, d in enumerate(dataset[:20]):
        g = d['structure_graph']
        
        # Convert to NetworkX
        nx_g = g.to_networkx()
        
        # Topology metrics
        n_nodes = nx_g.number_of_nodes()
        n_edges = nx_g.number_of_edges()
        
        # Degree distribution
        degrees = [d for _, d in nx_g.degree()]
        avg_degree = np.mean(degrees)
        degree_std = np.std(degrees)
        max_degree = max(degrees)
        
        # Clustering coefficient
        avg_clustering = nx.average_clustering(nx_g)
        
        # Connected components
        n_components = nx.number_connected_components(nx_g)
        is_connected = nx.is_connected(nx_g)
        
        # Diameter (only for connected graphs)
        diameter = nx.diameter(nx_g) if is_connected else -1
        
        # Average shortest path
        avg_path = nx.average_shortest_path_length(nx_g) if is_connected else -1
        
        # Algebraic connectivity (Fiedler value)
        if is_connected and n_nodes > 2:
            L = nx.laplacian_matrix(nx_g).toarray()
            eigenvalues = np.sort(np.linalg.eigvalsh(L))
            algebraic_connectivity = eigenvalues[1] if len(eigenvalues) > 1 else 0
        else:
            algebraic_connectivity = 0
        
        # Betweenness centrality
        bc = nx.betweenness_centrality(nx_g)
        max_bc = max(bc.values())
        avg_bc = np.mean(list(bc.values()))
        
        results.append({
            'idx': i,
            'unit': d['unit'],
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'avg_degree': round(avg_degree, 3),
            'degree_std': round(degree_std, 3),
            'max_degree': max_degree,
            'avg_clustering': round(avg_clustering, 4),
            'n_components': n_components,
            'is_connected': is_connected,
            'diameter': diameter,
            'avg_path_length': round(avg_path, 3) if avg_path > 0 else -1,
            'algebraic_connectivity': round(algebraic_connectivity, 4),
            'max_betweenness': round(max_bc, 4),
            'avg_betweenness': round(avg_bc, 4),
            'compliance': round(d['compliance'], 4),
        })
    
    if verbose:
        print(f"\n  Topology metrics for {len(results)} graphs:")
        print(f"  {'Unit':>12s} {'Nodes':>5s} {'Edges':>5s} {'AvgDeg':>7s} {'Clust':>6s} "
              f"{'Diam':>5s} {'AlgConn':>8s} {'MaxBC':>7s} {'Compl':>10s}")
        for r in results:
            print(f"  {r['unit']:>12s} {r['n_nodes']:5d} {r['n_edges']:5d} "
                  f"{r['avg_degree']:7.2f} {r['avg_clustering']:6.3f} "
                  f"{r['diameter']:5d} {r['algebraic_connectivity']:8.4f} "
                  f"{r['max_betweenness']:7.4f} {r['compliance']:10.4f}")
        
        # Correlation analysis: topology vs compliance
        alg_conns = [r['algebraic_connectivity'] for r in results]
        compliances = [r['compliance'] for r in results]
        avg_degrees = [r['avg_degree'] for r in results]
        
        if len(alg_conns) > 3:
            corr_alg_conn = np.corrcoef(alg_conns, compliances)[0, 1]
            corr_avg_deg = np.corrcoef(avg_degrees, compliances)[0, 1]
            print(f"\n  Correlation: algebraic_connectivity vs compliance = {corr_alg_conn:.4f}")
            print(f"  Correlation: avg_degree vs compliance = {corr_avg_deg:.4f}")
    
    return results


def scipy_stress_path_analysis(dataset, verbose=True):
    """Analyze stress transmission paths using scipy.sparse."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    
    if verbose:
        print("\n[Scipy Stress Path Analysis]")
    
    results = []
    for i, d in enumerate(dataset[:15]):
        gd = d['graph_data']
        ei = gd['edge_index']
        n_nodes = gd['node_features'].shape[0]
        sigma = d['sigma_true']
        u = d['u_true']
        
        # Build weighted adjacency matrix (weight = 1/stress for shortest path)
        n_edge_entries = ei.shape[1]
        
        # Use displacement differences as edge weights
        edge_weights = []
        for e in range(n_edge_entries):
            ni, nj = ei[0, e].item(), ei[1, e].item()
            if ni < n_nodes and nj < n_nodes:
                disp_diff = torch.norm(u[ni] - u[nj]).item()
                # Invert: high displacement difference = short path
                weight = 1.0 / (disp_diff + 1e-6)
                edge_weights.append((ni, nj, weight))
        
        if not edge_weights:
            continue
        
        # Build sparse matrix
        rows = [e[0] for e in edge_weights]
        cols = [e[1] for e in edge_weights]
        vals = [e[2] for e in edge_weights]
        adj = csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
        
        # Shortest paths from fixed nodes
        fixed = d['fixed']
        if len(fixed) > 0:
            source = fixed[0].item()
            try:
                dist_matrix = shortest_path(adj, method='D', indices=source, directed=False)
                max_dist = float(np.max(dist_matrix[np.isfinite(dist_matrix)]))
                avg_dist = float(np.mean(dist_matrix[np.isfinite(dist_matrix)]))
                n_reachable = int(np.sum(np.isfinite(dist_matrix)))
            except:
                max_dist = -1
                avg_dist = -1
                n_reachable = 0
        else:
            max_dist = -1
            avg_dist = -1
            n_reachable = 0
        
        results.append({
            'idx': i,
            'unit': d['unit'],
            'n_nodes': n_nodes,
            'max_path_dist': round(max_dist, 4),
            'avg_path_dist': round(avg_dist, 4),
            'n_reachable': n_reachable,
            'reachability': round(n_reachable / n_nodes, 4),
            'compliance': round(d['compliance'], 4),
        })
    
    if verbose:
        print(f"\n  Stress path analysis for {len(results)} graphs:")
        for r in results:
            print(f"    [{r['unit']:12s}] {r['n_nodes']:4d}n | "
                  f"max_path={r['max_path_dist']:8.3f} avg_path={r['avg_path_dist']:8.3f} "
                  f"reach={r['reachability']:.3f} compl={r['compliance']:.4f}")
    
    return results


# ============================================================
# Phase 2b: Run additional analysis
# ============================================================

def run_phase2b():
    """Run Phase 2b: NetworkX + Scipy graph analysis."""
    print("=" * 70)
    print("Phase 2b: Graph-Level Analysis with NetworkX + Scipy")
    print("=" * 70)
    
    # Regenerate dataset
    dataset = generate_diverse_dataset(n_samples=50)
    print(f"  Generated {len(dataset)} structures")
    
    # NetworkX analysis
    nx_results = networkx_graph_analysis(dataset)
    
    # Scipy analysis
    scipy_results = scipy_stress_path_analysis(dataset)
    
    # Save
    output = {
        'networkx_analysis': nx_results,
        'scipy_stress_paths': scipy_results,
    }
    
    output_file = RESULTS_DIR / "phase2b_graph_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return output


if __name__ == '__main__' and 'phase2b' in sys.argv:
    run_phase2b()

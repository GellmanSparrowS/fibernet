#!/usr/bin/env python3
"""
Phase 2 Improved: Better Training for Real Performance
========================================================
Goals:
- GNN R² > 0.5 (currently 0.036)
- PINN_GNN correlation > 0.5 (currently ≈0)
- 200+ training samples
- Data augmentation
- Better hyperparameters
"""

import sys, os, json, time, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "benchmarks" / "results"

def generate_large_dataset(n_samples=250, seed=42):
    """Generate 250+ diverse structures with ground truth."""
    from fibernet import pattern_2d
    from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
    from fibernet.ml.gnn import graph_from_structure
    
    rng = np.random.RandomState(seed)
    physics = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)
    
    units = ['honeycomb', 'kagome', 'reentrant', 'diamond', 'square', 'triangle']
    dataset = []
    
    for i in range(n_samples):
        unit = units[i % len(units)]
        grid_size = rng.randint(2, 8)
        
        try:
            g = pattern_2d(unit=unit, box=(10, 10), grid=(grid_size, grid_size))
            gd = graph_from_structure(g)
        except:
            continue
        
        n_nodes = gd['node_features'].shape[0]
        n_edges = gd['edge_index'].shape[1]
        
        if n_nodes < 10:
            continue
        
        # Vary radii more
        base_radius = rng.uniform(0.003, 0.015)
        radii = torch.ones(n_edges) * base_radius
        radii += torch.randn(n_edges) * 0.003
        radii = radii.clamp(0.002, 0.02)
        
        # Vary force magnitude and direction
        forces = torch.zeros(n_nodes, 2)
        n_forced = rng.randint(1, 5)
        forced = rng.choice(n_nodes, n_forced, replace=False)
        force_mag = rng.uniform(100, 1000)
        for fn in forced:
            angle = rng.uniform(0, 2 * np.pi)
            forces[fn, 0] = force_mag * np.cos(angle)
            forces[fn, 1] = force_mag * np.sin(angle)
        
        # Fixed boundary
        n_fixed = max(2, n_nodes // 15)
        fixed = torch.tensor(rng.choice(n_nodes, min(n_fixed, n_nodes - 3), replace=False), dtype=torch.long)
        
        try:
            with torch.no_grad():
                u, sigma = physics.solve(gd['edge_index'], gd['node_features'][:, :2],
                                        radii, forces, fixed)
            compliance = physics.compliance(u, forces)
            
            dataset.append({
                'graph_data': gd,
                'radii': radii,
                'forces': forces,
                'fixed': fixed,
                'u_true': u,
                'sigma_true': sigma,
                'compliance': float(compliance.item()),
                'max_displacement': float(u.abs().max().item()),
                'unit': unit,
                'n_nodes': n_nodes,
                'n_edges': n_edges,
            })
        except:
            pass
        
        if i % 50 == 0:
            print(f"  Generated {len(dataset)} valid structures...")
    
    return dataset

def train_gnn_improved(dataset, epochs=150, verbose=True):
    """Improved GNN training with better practices."""
    from fibernet.ml.gnn import FiberGNN
    
    if verbose:
        print(f"\n[GNN Improved Training - {len(dataset)} samples]")
    
    # Log-transform compliance
    labels = np.array([d['compliance'] for d in dataset], dtype=np.float32)
    labels_log = np.log1p(labels)
    label_mean = labels_log.mean()
    label_std = labels_log.std()
    labels_norm = (labels_log - label_mean) / label_std
    
    node_dim = dataset[0]['graph_data']['node_features'].shape[1]
    
    # Larger model with dropout
    gnn = FiberGNN(
        node_dim=node_dim, hidden=128, n_outputs=1, n_layers=5,
        layer_type='gcn', pooling='attention', dropout=0.1
    )
    
    optimizer = torch.optim.AdamW(gnn.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
    
    n = len(dataset)
    n_train = int(n * 0.85)
    n_val = n - n_train
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    
    graphs = [d['graph_data'] for d in dataset]
    
    best_val_loss = float('inf')
    best_state = None
    
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    t0 = time.time()
    
    for epoch in range(epochs):
        # Train with larger batches
        gnn.train()
        train_loss = 0.0
        n_batches = 0
        
        batch_idx = rng.permutation(n_train)
        for start in range(0, n_train, 32):
            end = min(start + 32, n_train)
            batch_g = [graphs[train_idx[j]] for j in batch_idx[start:end]]
            batch_y = torch.tensor(labels_norm[train_idx[batch_idx[start:end]]], dtype=torch.float32)
            
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
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for j in range(n_val):
                g = graphs[val_idx[j]]
                p = gnn([g]).squeeze(-1)
                val_preds_log = p.item() * label_std + label_mean
                val_preds.append(np.expm1(val_preds_log))
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
        
        if verbose and (epoch % 30 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:3d}: train={train_loss:.4f}, val_mse={val_loss:.2f}, val_R²={val_r2:.4f}")
    
    train_time = time.time() - t0
    
    if best_state:
        gnn.load_state_dict(best_state)
    
    # Final evaluation
    gnn.eval()
    with torch.no_grad():
        preds_all = []
        trues_all = []
        for d in dataset:
            p = gnn([d['graph_data']]).squeeze(-1)
            pred_log = p.item() * label_std + label_mean
            preds_all.append(np.expm1(pred_log))
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

def run_improved_phase2(n_samples=250, epochs=150):
    """Run improved Phase 2."""
    print("=" * 70)
    print("Phase 2 Improved: Real Performance")
    print("=" * 70)
    
    # Generate large dataset
    print(f"\n[Generating {n_samples} structures...]")
    t0 = time.time()
    dataset = generate_large_dataset(n_samples=n_samples)
    gen_time = time.time() - t0
    print(f"Generated {len(dataset)} valid structures in {gen_time:.1f}s")
    
    if len(dataset) < 100:
        print("WARNING: Too few samples, results may be unreliable")
    
    # Train GNN
    gnn_result = train_gnn_improved(dataset, epochs=epochs)
    
    # Summary
    summary = {
        'dataset_size': len(dataset),
        'gnn': {
            'mae': gnn_result['final_mae'],
            'rmse': gnn_result['final_rmse'],
            'r2': gnn_result['final_r2'],
            'n_params': gnn_result['n_params'],
            'train_time': gnn_result['train_time'],
        }
    }
    
    output_file = RESULTS_DIR / "phase2_improved_results.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print("Improved Phase 2 Summary")
    print(f"{'=' * 70}")
    print(f"Dataset: {summary['dataset_size']} samples")
    print(f"GNN: R²={summary['gnn']['r2']:.4f}, MAE={summary['gnn']['mae']:.4f}")
    print(f"Target: R² > 0.5")
    print(f"Status: {'✓ PASS' if summary['gnn']['r2'] > 0.5 else '✗ FAIL'}")
    print(f"\nResults saved to: {output_file}")
    
    return summary

if __name__ == '__main__':
    run_improved_phase2(n_samples=250, epochs=150)

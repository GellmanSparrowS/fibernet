"""
Comprehensive validation script for all 14 3D unit types.
Tests: generation → tiling → simulation → feature extraction → visualization.

Usage:
    cd fibernet && source .venv/bin/activate
    python fibernet/scripts/validate_3d_units.py

Results are saved to output_data/3d_validation/
"""

import os
import sys
import json
import time
import traceback
import numpy as np

# Ensure fibernet is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import fibernet as fn
from fibernet.analysis.graph_features import GraphFeatureExtractor

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'output_data', '3d_validation')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'viz'), exist_ok=True)


def validate_unit(unit_name, unit_kwargs=None):
    """Validate a single 3D unit type through the full pipeline."""
    result = {
        'unit': unit_name,
        'generation': {},
        'tiling': {},
        'simulation': {},
        'features': {},
        'visualization': {},
        'errors': [],
    }
    
    ukw = unit_kwargs or {}
    
    # 1. Generation (single unit cell)
    try:
        t0 = time.time()
        g_unit = fn.pattern_3d(unit=unit_name, box=(10, 10, 10), grid=(1, 1, 1), unit_kwargs=ukw)
        gen_time = time.time() - t0
        result['generation'] = {
            'nodes': g_unit.num_nodes,
            'edges': g_unit.num_edges,
            'dimension': g_unit.dimension,
            'time_s': round(gen_time, 3),
        }
    except Exception as e:
        result['errors'].append(f'generation: {e}')
        return result
    
    # 2. Tiling (2x2x2)
    try:
        t0 = time.time()
        g_tiled = fn.pattern_3d(unit=unit_name, box=(10, 10, 10), grid=(2, 2, 2), unit_kwargs=ukw)
        tile_time = time.time() - t0
        result['tiling'] = {
            'nodes': g_tiled.num_nodes,
            'edges': g_tiled.num_edges,
            'time_s': round(tile_time, 3),
            'ratio_nodes': round(g_tiled.num_nodes / max(g_unit.num_nodes, 1), 2),
        }
    except Exception as e:
        result['errors'].append(f'tiling: {e}')
        g_tiled = g_unit
    
    # 3. Simulation (stretch test)
    try:
        engine = fn.TaichiEngine()
        t0 = time.time()
        sim_result = engine.stretch_test(g_tiled, target_stretch=1.3, num_steps=100)
        sim_time = time.time() - t0
        max_disp = float(np.max(np.abs(sim_result.displacements)))
        result['simulation'] = {
            'max_displacement': round(max_disp, 6),
            'time_s': round(sim_time, 3),
            'energy': float(sim_result.energy) if hasattr(sim_result, 'energy') else 0.0,
        }
    except Exception as e:
        result['errors'].append(f'simulation: {traceback.format_exc()}')
    
    # 4. Feature extraction
    try:
        ext = GraphFeatureExtractor(canvas_size=128)
        t0 = time.time()
        # Convert to networkx for feature extraction
        import networkx as nx
        G_nx = nx.Graph()
        for nid in g_tiled.nodes:
            pos = g_tiled.nodes[nid].position
            G_nx.add_node(nid, pos=tuple(pos[:2]) if g_tiled.dimension == 2 else tuple(pos))
        for eid in g_tiled.edges:
            e = g_tiled.edges[eid]
            G_nx.add_edge(e.node_i, e.node_j)
        
        features = ext.extract(G_nx)
        feat_time = time.time() - t0
        n_features = len(features)
        
        # Key structural features
        result['features'] = {
            'n_features': n_features,
            'n_node': features.get('n_node', 0),
            'n_edge': features.get('n_edge', 0),
            'mean_degree': round(features.get('mean_degree', 0), 2),
            'is_connected': features.get('is_connected', 0),
            'clustering_coef': round(features.get('clustering_coef', 0), 4),
            'time_s': round(feat_time, 3),
        }
    except Exception as e:
        result['errors'].append(f'features: {e}')
    
    # 5. Visualization (save PyVista render)
    try:
        viz_path = os.path.join(OUTPUT_DIR, 'viz', f'{unit_name}_3d.png')
        fig = fn.render_graph_3d(g_tiled, save_path=viz_path)
        result['visualization'] = {
            'saved': os.path.exists(viz_path),
            'path': viz_path,
        }
    except Exception as e:
        result['errors'].append(f'visualization: {e}')
    
    # 6. Test n_pts_per_side (RL feature)
    try:
        g_pts = fn.pattern_3d(unit=unit_name, box=(10, 10, 10), grid=(1, 1, 1),
                              n_pts_per_side=2, unit_kwargs=ukw)
        result['rl_feature'] = {
            'pts_nodes': g_pts.num_nodes,
            'pts_edges': g_pts.num_edges,
        }
    except Exception as e:
        result['errors'].append(f'n_pts_per_side: {e}')
    
    return result


def main():
    """Run validation for all 3D unit types."""
    units_3d = fn.list_units_3d()
    print(f"FiberNet v{fn.__version__} — 3D Unit Validation")
    print(f"Testing {len(units_3d)} unit types: {', '.join(units_3d)}\n")
    
    # Special kwargs for certain units
    special_kwargs = {
        'chiral_3d': {'chirality': 0.3},
        'reentrant_3d': {'angle': 15.0},
    }
    
    all_results = []
    for unit_name in units_3d:
        kwargs = special_kwargs.get(unit_name, {})
        print(f"Validating {unit_name}...", end=' ', flush=True)
        result = validate_unit(unit_name, kwargs)
        all_results.append(result)
        
        status = '✅' if not result['errors'] else f'⚠️ {len(result["errors"])} errors'
        gen = result.get('generation', {})
        print(f"{status} — {gen.get('nodes', '?')}n/{gen.get('edges', '?')}e")
        
        if result['errors']:
            for err in result['errors']:
                print(f"  ERROR: {err[:100]}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    
    total_errors = sum(len(r['errors']) for r in all_results)
    passed = sum(1 for r in all_results if not r['errors'])
    
    print(f"Units tested: {len(all_results)}")
    print(f"Fully passed: {passed}/{len(all_results)}")
    print(f"Total errors: {total_errors}")
    
    # Feature extraction summary
    feat_counts = [r.get('features', {}).get('n_features', 0) for r in all_results]
    if feat_counts:
        print(f"Features extracted: min={min(feat_counts)}, max={max(feat_counts)}")
    
    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    # Print table
    print(f"\n{'Unit':<15} {'Nodes':>6} {'Edges':>6} {'Tiled N':>8} {'Sim Time':>8} {'Feats':>5} {'Status':>8}")
    print('-' * 70)
    for r in all_results:
        gen = r.get('generation', {})
        tile = r.get('tiling', {})
        sim = r.get('simulation', {})
        feat = r.get('features', {})
        status = 'PASS' if not r['errors'] else 'FAIL'
        print(f"{r['unit']:<15} {gen.get('nodes',0):>6} {gen.get('edges',0):>6} "
              f"{tile.get('nodes',0):>8} {sim.get('time_s',0):>7.2f}s {feat.get('n_features',0):>5} {status:>8}")
    
    return total_errors == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

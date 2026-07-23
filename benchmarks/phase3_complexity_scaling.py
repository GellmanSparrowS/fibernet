#!/usr/bin/env python3
"""
Phase 3: Complexity Scaling Benchmark
Measure timing/memory for different graph sizes and operations.
"""

import sys, os, json, time, gc, tracemalloc
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "benchmarks" / "results"

def measure_memory():
    """Get current memory in MB."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def benchmark_gnn_forward(sizes=[10, 50, 100, 500, 1000], n_runs=5):
    """Benchmark GNN forward pass for different graph sizes."""
    from fibernet.ml.gnn import FiberGNN
    from fibernet import pattern_2d
    from fibernet.ml.gnn import graph_from_structure
    
    print("\n[GNN Forward Pass Scaling]")
    results = []
    
    for target_nodes in sizes:
        # Find grid size that gives approximately target_nodes
        grid = max(2, int(np.sqrt(target_nodes / 2)))
        
        try:
            g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(grid, grid))
            gd = graph_from_structure(g)
            actual_nodes = gd['node_features'].shape[0]
            actual_edges = gd['edge_index'].shape[1]
        except:
            continue
        
        # Create GNN
        gnn = FiberGNN(
            node_dim=gd['node_features'].shape[1],
            hidden=64, n_outputs=1, n_layers=4, pooling='attention'
        )
        gnn.eval()
        
        # Warmup
        with torch.no_grad():
            _ = gnn([gd])
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            gc.collect()
            t0 = time.perf_counter()
            with torch.no_grad():
                pred = gnn([gd])
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        avg_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000
        
        results.append({
            'target_nodes': target_nodes,
            'actual_nodes': actual_nodes,
            'actual_edges': actual_edges,
            'avg_time_ms': round(avg_time, 3),
            'std_time_ms': round(std_time, 3),
        })
        
        print(f"  {actual_nodes:4d} nodes, {actual_edges:5d} edges: "
              f"{avg_time:.2f} ± {std_time:.2f} ms")
    
    return results

def benchmark_physics_solve(sizes=[10, 50, 100, 500, 1000], n_runs=3):
    """Benchmark DifferentiableSpringNetwork solve for different graph sizes."""
    from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
    from fibernet import pattern_2d
    from fibernet.ml.gnn import graph_from_structure
    
    print("\n[Physics Solve Scaling]")
    results = []
    physics = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)
    
    for target_nodes in sizes:
        grid = max(2, int(np.sqrt(target_nodes / 2)))
        
        try:
            g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(grid, grid))
            gd = graph_from_structure(g)
            actual_nodes = gd['node_features'].shape[0]
            actual_edges = gd['edge_index'].shape[1]
        except:
            continue
        
        # Setup
        radii = torch.ones(actual_edges) * 0.01
        forces = torch.zeros(actual_nodes, 2)
        forces[-1, 0] = 500.0
        fixed = torch.tensor([0, 1], dtype=torch.long)
        
        # Warmup
        with torch.no_grad():
            _ = physics.solve(gd['edge_index'], gd['node_features'][:, :2],
                            radii, forces, fixed)
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            gc.collect()
            t0 = time.perf_counter()
            with torch.no_grad():
                u, sigma = physics.solve(gd['edge_index'], gd['node_features'][:, :2],
                                        radii, forces, fixed)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        
        results.append({
            'target_nodes': target_nodes,
            'actual_nodes': actual_nodes,
            'actual_edges': actual_edges,
            'avg_time_ms': round(avg_time, 3),
            'std_time_ms': round(std_time, 3),
        })
        
        print(f"  {actual_nodes:4d} nodes, {actual_edges:5d} edges: "
              f"{avg_time:.2f} ± {std_time:.2f} ms")
    
    return results

def benchmark_memory_scaling(sizes=[10, 50, 100, 500, 1000]):
    """Benchmark memory usage for different graph sizes."""
    from fibernet import pattern_2d
    from fibernet.ml.gnn import graph_from_structure, FiberGNN
    from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
    
    print("\n[Memory Scaling]")
    results = []
    
    for target_nodes in sizes:
        grid = max(2, int(np.sqrt(target_nodes / 2)))
        
        gc.collect()
        mem_before = measure_memory()
        
        try:
            g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(grid, grid))
            gd = graph_from_structure(g)
            actual_nodes = gd['node_features'].shape[0]
            actual_edges = gd['edge_index'].shape[1]
            
            # Create model and run
            gnn = FiberGNN(node_dim=5, hidden=64, n_outputs=1, n_layers=3)
            physics = DifferentiableSpringNetwork(youngs_modulus=1e9)
            
            with torch.no_grad():
                pred = gnn([gd])
                radii = torch.ones(actual_edges) * 0.01
                forces = torch.zeros(actual_nodes, 2)
                forces[-1, 0] = 500.0
                fixed = torch.tensor([0, 1])
                u, sigma = physics.solve(gd['edge_index'], gd['node_features'][:, :2],
                                        radii, forces, fixed)
            
            gc.collect()
            mem_after = measure_memory()
            mem_delta = mem_after - mem_before
            
            results.append({
                'actual_nodes': actual_nodes,
                'actual_edges': actual_edges,
                'mem_before_mb': round(mem_before, 2),
                'mem_after_mb': round(mem_after, 2),
                'mem_delta_mb': round(mem_delta, 2),
            })
            
            print(f"  {actual_nodes:4d} nodes: {mem_delta:.1f} MB delta "
                  f"({mem_before:.1f} → {mem_after:.1f} MB)")
        except Exception as e:
            print(f"  {target_nodes:4d} nodes: FAILED - {e}")
        
        gc.collect()
    
    return results

def benchmark_pinn_gnn_scaling(sizes=[10, 50, 100, 500], n_runs=3):
    """Benchmark PhysicsInformedGNN for different graph sizes."""
    from fibernet.ml.pinn_gnn import PhysicsInformedGNN
    from fibernet import pattern_2d
    from fibernet.ml.gnn import graph_from_structure
    
    print("\n[PhysicsInformedGNN Scaling]")
    results = []
    
    for target_nodes in sizes:
        grid = max(2, int(np.sqrt(target_nodes / 2)))
        
        try:
            g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(grid, grid))
            gd = graph_from_structure(g)
            actual_nodes = gd['node_features'].shape[0]
            actual_edges = gd['edge_index'].shape[1]
        except:
            continue
        
        pinn = PhysicsInformedGNN(
            node_dim=gd['node_features'].shape[1],
            edge_dim=gd['edge_features'].shape[1],
            hidden=64, n_layers=3, n_outputs=1, predict_field=True
        )
        pinn.eval()
        
        # Warmup
        with torch.no_grad():
            fields = pinn.predict_fields(gd)
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            gc.collect()
            t0 = time.perf_counter()
            with torch.no_grad():
                fields = pinn.predict_fields(gd)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        
        results.append({
            'actual_nodes': actual_nodes,
            'actual_edges': actual_edges,
            'avg_time_ms': round(avg_time, 3),
            'std_time_ms': round(std_time, 3),
        })
        
        print(f"  {actual_nodes:4d} nodes, {actual_edges:5d} edges: "
              f"{avg_time:.2f} ± {std_time:.2f} ms")
    
    return results

def run_phase3():
    """Run Phase 3 complexity scaling benchmarks."""
    print("=" * 70)
    print("Phase 3: Complexity Scaling Benchmark")
    print("=" * 70)
    
    results = {
        'gnn_forward': benchmark_gnn_forward(),
        'physics_solve': benchmark_physics_solve(),
        'memory_scaling': benchmark_memory_scaling(),
        'pinn_gnn': benchmark_pinn_gnn_scaling(),
    }
    
    # Save results
    output_file = RESULTS_DIR / "phase3_complexity_scaling.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print("Phase 3 Summary")
    print(f"{'=' * 70}")
    
    # Scaling analysis
    if results['gnn_forward']:
        nodes = [r['actual_nodes'] for r in results['gnn_forward']]
        times = [r['avg_time_ms'] for r in results['gnn_forward']]
        if len(nodes) >= 2:
            # Fit power law: time = a * nodes^b
            log_nodes = np.log(nodes)
            log_times = np.log(times)
            b, a = np.polyfit(log_nodes, log_times, 1)
            print(f"GNN Forward: O(n^{b:.2f}) scaling")
    
    if results['physics_solve']:
        nodes = [r['actual_nodes'] for r in results['physics_solve']]
        times = [r['avg_time_ms'] for r in results['physics_solve']]
        if len(nodes) >= 2:
            log_nodes = np.log(nodes)
            log_times = np.log(times)
            b, a = np.polyfit(log_nodes, log_times, 1)
            print(f"Physics Solve: O(n^{b:.2f}) scaling")
    
    print(f"\nResults saved to: {output_file}")
    return results

if __name__ == '__main__':
    run_phase3()

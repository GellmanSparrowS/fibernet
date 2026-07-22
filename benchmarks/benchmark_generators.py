"""
Benchmark Suite: Network Generators

Performance benchmarks for structure generation algorithms.
Tests scaling, memory usage, and generation time.

Usage:
    python benchmarks/benchmark_generators.py
"""

import numpy as np
import time
import psutil
import os
from fibernet import gen

print("=" * 70)
print("BENCHMARK: Network Generator Performance")
print("=" * 70)

process = psutil.Process(os.getpid())

def measure_generation(func, *args, **kwargs):
    """Measure generation time and memory."""
    t0 = time.time()
    net = func(*args, **kwargs)
    t1 = time.time()
    mem = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        'time': t1 - t0,
        'memory_mb': mem,
        'n_fibers': net.num_fibers,
        'n_crosslinks': net.num_crosslinks,
        'total_length': net.total_length,
    }

# Test 1: Random 2D scaling
print("\n[1/5] Random 2D generator scaling...")
sizes = [100, 500, 1000, 2000, 5000]
for n in sizes:
    result = measure_generation(gen.random_straight_2d, num_fibers=n, seed=42)
    print(f"  N={n:5d}: Time={result['time']:.3f}s, "
          f"Mem={result['memory_mb']:.1f}MB, "
          f"Fibers={result['n_fibers']}, "
          f"Crosslinks={result['n_crosslinks']}")

# Test 2: Random 3D scaling
print("\n[2/5] Random 3D generator scaling...")
sizes = [100, 500, 1000, 2000]
for n in sizes:
    result = measure_generation(gen.random_straight_3d, num_fibers=n, seed=42)
    print(f"  N={n:5d}: Time={result['time']:.3f}s, "
          f"Mem={result['memory_mb']:.1f}MB, "
          f"Fibers={result['n_fibers']}, "
          f"Crosslinks={result['n_crosslinks']}")

# Test 3: Lattice generators
print("\n[3/5] Lattice generator performance...")
lattice_gens = [
    ('Square', gen.square_lattice_2d, {'spacing': 1.0, 'grid_size': (20, 20)}),
    ('Triangular', gen.triangular_lattice_2d, {'spacing': 1.0, 'grid_size': (20, 20)}),
    ('Honeycomb', gen.honeycomb_lattice_2d, {'cell_size': 1.0, 'grid_size': (20, 20)}),
    ('Kagome', gen.kagome_lattice_2d, {'spacing': 1.0, 'grid_size': (20, 20)}),
    ('Cubic 3D', gen.cubic_lattice_3d, {'spacing': 2.0, 'grid_size': (10, 10, 10)}),
    ('Diamond 3D', gen.diamond_lattice_3d, {'spacing': 3.0, 'grid_size': (8, 8, 8)}),
    ('Octet truss', gen.octet_truss_3d, {'spacing': 2.0, 'grid_size': (10, 10, 10)}),
]

for name, func, kwargs in lattice_gens:
    result = measure_generation(func, **kwargs)
    print(f"  {name:15s}: Time={result['time']:.3f}s, "
          f"Fibers={result['n_fibers']}, "
          f"Crosslinks={result['n_crosslinks']}")

# Test 4: Special structures
print("\n[4/5] Special structure generators...")
special_gens = [
    ('Chiral', gen.chiral_metamaterial, {'unit_cell_size': 10.0, 'grid_size': (3, 3, 3)}),
    ('Braided rope', gen.braided_rope, {'num_strands': 6, 'rope_radius': 5.0}),
    ('Twisted bundle', gen.twisted_bundle, {'num_fibers': 20, 'twist_angle': 45.0}),
    ('Hierarchical', gen.hierarchical_bundle, {'levels': 3, 'fibers_per_bundle': 5}),
    ('Plain weave', gen.plain_weave_2d, {'spacing': 2.0, 'grid_size': (20, 20)}),
    ('Twill weave', gen.twill_weave_2d, {'spacing': 2.0, 'grid_size': (20, 20)}),
    ('Satin weave', gen.satin_weave_2d, {'spacing': 2.0, 'grid_size': (20, 20)}),
]

for name, func, kwargs in special_gens:
    try:
        result = measure_generation(func, **kwargs)
        print(f"  {name:15s}: Time={result['time']:.3f}s, "
              f"Fibers={result['n_fibers']}, "
              f"Crosslinks={result['n_crosslinks']}")
    except Exception as e:
        print(f"  {name:15s}: ERROR - {str(e)[:50]}")

# Test 5: Biomimetic structures
print("\n[5/5] Biomimetic structure generators...")
bio_gens = [
    ('Collagen', gen.biomimetic_collagen, {'density': 0.3, 'box_size': (50, 50)}),
    ('Fibrin', gen.biomimetic_fibrin, {'density': 0.3, 'box_size': (50, 50)}),
    ('Electrospun', gen.electrospun_network, {'num_fibers': 500, 'fiber_length': 10.0}),
    ('Paper', gen.paper_network, {'num_fibers': 500, 'fiber_length': 10.0}),
]

for name, func, kwargs in bio_gens:
    try:
        result = measure_generation(func, **kwargs)
        print(f"  {name:15s}: Time={result['time']:.3f}s, "
              f"Fibers={result['n_fibers']}, "
              f"Crosslinks={result['n_crosslinks']}")
    except Exception as e:
        print(f"  {name:15s}: ERROR - {str(e)[:50]}")

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)

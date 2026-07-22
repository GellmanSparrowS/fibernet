"""
Benchmark Suite: Mechanical Analysis

Performance benchmarks and validation tests for mechanical simulations.
Compares different network types and validates expected scaling laws.

Usage:
    python benchmarks/benchmark_mechanical.py
"""

import numpy as np
import time
from fibernet import gen
from fibernet.sim import FiberFEM
from fibernet.analysis import MorphologyAnalyzer

print("=" * 70)
print("BENCHMARK: Mechanical Analysis Performance & Validation")
print("=" * 70)

# Test 1: Scaling with network size
print("\n[1/4] Network size scaling test...")
sizes = [10, 20, 40, 80, 160]
results_size = []

for n in sizes:
    net = gen.random_straight_2d(num_fibers=n, seed=42)
    
    t0 = time.time()
    fem = FiberFEM(net)
    E = fem.effective_modulus()
    t1 = time.time()
    
    results_size.append({
        'n_fibers': n,
        'modulus': E,
        'time': t1 - t0,
    })
    print(f"  N={n:4d}: E={E:.2e} Pa, Time={t1-t0:.3f}s")

# Test 2: Convergence with fiber count
print("\n[2/4] Statistical convergence test...")
n_samples = 5
n_fibers_list = [20, 50, 100]

for n_fibers in n_fibers_list:
    moduli = []
    for seed in range(n_samples):
        net = gen.random_straight_2d(num_fibers=n_fibers, seed=seed)
        fem = FiberFEM(net)
        E = fem.effective_modulus()
        moduli.append(E)
    
    mean_E = np.mean(moduli)
    std_E = np.std(moduli)
    cv = std_E / mean_E if mean_E > 0 else 0
    
    print(f"  N={n_fibers:4d}: E={mean_E:.2e} ± {std_E:.2e} Pa (CV={cv:.3f})")

# Test 3: Network type comparison
print("\n[3/4] Network type comparison...")
networks = {
    'Random 2D': gen.random_straight_2d(num_fibers=50, seed=42),
    'Random 3D': gen.random_straight_3d(num_fibers=50, seed=42),
    'Square lattice': gen.square_lattice_2d(spacing=2.0, grid_size=(15, 15)),
    'Triangular lattice': gen.triangular_lattice_2d(spacing=2.0, grid_size=(15, 15)),
    'Honeycomb': gen.honeycomb_lattice_2d(cell_size=2.0, grid_size=(15, 15)),
}

for name, net in networks.items():
    fem = FiberFEM(net)
    E = fem.effective_modulus()
    morph = MorphologyAnalyzer(net)
    order = morph.nematic_order_parameter()
    print(f"  {name:20s}: E={E:.2e} Pa, Order={order:.3f}")

# Test 4: Anisotropy validation
print("\n[4/4] Anisotropy validation...")
net = gen.random_straight_2d(num_fibers=80, seed=42)
fem = FiberFEM(net)

E_x = fem.effective_modulus(axis=0)
E_y = fem.effective_modulus(axis=1)
E_z = fem.effective_modulus(axis=2)

print(f"  E_x = {E_x:.2e} Pa")
print(f"  E_y = {E_y:.2e} Pa")
print(f"  E_z = {E_z:.2e} Pa")
print(f"  Anisotropy ratio (E_x/E_y) = {E_x/E_y:.3f}")

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)

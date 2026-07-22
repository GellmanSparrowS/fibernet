"""
Example 16: Fiber Bundles

Demonstrates fiber bundle generators for biological and composite materials:
1. Parallel bundles (unidirectional composites)
2. Twisted bundles (ropes, cables)
3. Random bundles (loose fiber mats)
4. Braided bundles (braided composites)
5. Tendon-like bundles (biological tissues)

Usage:
    python examples/16_fiber_bundles.py
"""

import numpy as np
from fibernet import gen
from fibernet.gen.bundles import (
    parallel_bundle_2d,
    twisted_bundle_2d,
    random_bundle_3d,
    braided_bundle_3d,
    tendon_like_bundle_3d,
)
from fibernet.analysis.spatial import compute_spatial_statistics

print("=" * 70)
print("  FiberNet - Fiber Bundle Examples")
print("=" * 70)

# ============================================================
# 1. Parallel Bundle (Unidirectional Composite)
# ============================================================
print("\n[1/5] Parallel Bundle (Unidirectional Composite)")
print("-" * 60)

bundle_parallel = parallel_bundle_2d(
    num_fibers=20,
    bundle_length=100.0,
    bundle_width=10.0,
    fiber_radius=0.5,
    orientation=0.0,  # Horizontal
    seed=42
)

print(f"  Fibers: {len(bundle_parallel.fibers)}")
print(f"  Dimension: {bundle_parallel.dimension}D")

# Compute orientation order
from fibernet.analysis.spatial import OrientationAnalysis
orient = OrientationAnalysis(bundle_parallel)
S = orient.nematic_order_parameter()
print(f"  Nematic order parameter: {S:.3f}")
print(f"  Expected: close to 1.0 (highly aligned)")

# ============================================================
# 2. Twisted Bundle (Rope/Cable)
# ============================================================
print("\n[2/5] Twisted Bundle (Rope/Cable)")
print("-" * 60)

bundle_twisted = twisted_bundle_2d(
    num_fibers=12,
    bundle_length=80.0,
    twist_pitch=30.0,  # Distance for one full twist
    bundle_radius=5.0,
    fiber_radius=0.4,
    seed=42
)

print(f"  Fibers: {len(bundle_twisted.fibers)}")
print(f"  Twist pitch: 30.0 units")
print(f"  Bundle radius: 5.0 units")

# ============================================================
# 3. Random Bundle (Loose Fiber Mat)
# ============================================================
print("\n[3/5] Random Bundle (3D Loose Fiber Mat)")
print("-" * 60)

bundle_random = random_bundle_3d(
    num_fibers=30,
    bundle_length=60.0,
    bundle_radius=8.0,
    fiber_radius=0.5,
    orientation_variance=0.3,  # Angular spread
    seed=42
)

print(f"  Fibers: {len(bundle_random.fibers)}")
print(f"  Dimension: {bundle_random.dimension}D")
print(f"  Orientation variance: 0.3 rad")

# Compute statistics
stats = compute_spatial_statistics(bundle_random)
print(f"  Anisotropy index: {stats['anisotropy_index']:.3f}")

# ============================================================
# 4. Braided Bundle (Braided Composite)
# ============================================================
print("\n[4/5] Braided Bundle (Braided Composite)")
print("-" * 60)

bundle_braided = braided_bundle_3d(
    num_strands=8,
    bundle_length=100.0,
    braid_radius=6.0,
    fibers_per_strand=4,
    strand_radius=0.6,
    seed=42
)

print(f"  Total fibers: {len(bundle_braided.fibers)}")
print(f"  Strands: 8")
print(f"  Fibers per strand: 4")
print(f"  Braid radius: 6.0 units")

# ============================================================
# 5. Tendon-like Bundle (Biological Tissue)
# ============================================================
print("\n[5/5] Tendon-like Bundle (Biological Tissue)")
print("-" * 60)

bundle_tendon = tendon_like_bundle_3d(
    num_fibers=40,
    bundle_length=120.0,
    bundle_radius=10.0,
    fiber_radius=0.8,
    crimp_amplitude=1.5,  # Waviness amplitude
    crimp_wavelength=15.0,  # Crimp wavelength
    seed=42
)

print(f"  Fibers: {len(bundle_tendon.fibers)}")
print(f"  Bundle length: 120.0 units")
print(f"  Crimp amplitude: 1.5 units")
print(f"  Crimp wavelength: 15.0 units")
print(f"  Characteristic: Wavy/crimped fibers like biological tendons")

# ============================================================
# Comparison
# ============================================================
print("\n" + "=" * 70)
print("  Bundle Comparison")
print("=" * 70)

bundles = {
    'Parallel': bundle_parallel,
    'Twisted': bundle_twisted,
    'Random': bundle_random,
    'Braided': bundle_braided,
    'Tendon': bundle_tendon,
}

print(f"\n  {'Bundle Type':<15s} | {'Fibers':>8s} | {'Dim':>4s} | {'Total Length':>12s}")
print("  " + "-" * 50)

for name, bundle in bundles.items():
    total_length = sum(f.length for f in bundle.fibers)
    print(f"  {name:<15s} | {len(bundle.fibers):>8d} | {bundle.dimension:>4d}D | {total_length:>12.1f}")

# ============================================================
print("\n" + "=" * 70)
print("  Fiber Bundle Examples Complete!")
print("=" * 70)
print("""
Applications:
  - Parallel bundles: Unidirectional composites, laminates
  - Twisted bundles: Ropes, cables, yarns
  - Random bundles: Non-woven mats, insulation
  - Braided bundles: Braided composites, medical sutures
  - Tendon-like: Biological tissues (tendons, ligaments)

Next steps:
  - Analyze mechanical properties using FEM
  - Visualize with PyVista or trimesh
  - Export to external FEM solvers
""")

"""
Research Case Study: Hierarchical Carbon Fiber Composite Analysis

This example demonstrates a complete research workflow for studying
hierarchical fiber network structures with advanced mechanics.

Workflow:
1. Generate base structures (carbon fibers + polymer matrix)
2. Apply transformations (rotation, tiling, merging)
3. Run multi-physics simulations (mechanical + thermal)
4. Analyze structure-property relationships
5. Export results for publication
"""

import sys
sys.path.insert(0, '/home/codex/projects/codex_test/fibernet')

import numpy as np
from fibernet import gen, api
from fibernet.core.transform import rotate, merge, tile
from fibernet.core.pbc import PeriodicBox, apply_pbc
from fibernet.sim.nonlinear import NonlinearFEM, BilinearPlasticity
from fibernet.analysis import MorphologyAnalyzer, TopologyAnalyzer
from fibernet.io import to_vtk, to_lammps


def main():
    print("=" * 80)
    print("FiberNet Research Case Study")
    print("Hierarchical Carbon Fiber Composite")
    print("=" * 80)
    
    # ============================================================
    # 1. Generate Hierarchical Structure
    # ============================================================
    print("\n[1] Generating Hierarchical Structure")
    print("-" * 80)
    
    # Level 1: Carbon fibers (primary reinforcement)
    print("  Creating carbon fibers...")
    carbon = gen.random_straight_2d(
        num_fibers=20,
        fiber_length=10,
        box_size=(30, 30),
        radius=0.3,
        seed=42
    )
    print(f"    Carbon fibers: {carbon.num_fibers} fibers")
    
    # Level 2: Polymer matrix (secondary phase)
    print("  Creating polymer matrix...")
    polymer = gen.random_straight_2d(
        num_fibers=30,
        fiber_length=8,
        box_size=(30, 30),
        radius=0.1,
        seed=43
    )
    print(f"    Polymer fibers: {polymer.num_fibers} fibers")
    
    # Merge into composite
    print("  Merging composite...")
    composite = merge([carbon, polymer])
    print(f"    Total: {composite.num_fibers} fibers, {composite.num_crosslinks} crosslinks")
    
    # ============================================================
    # 2. Apply Transformations
    # ============================================================
    print("\n[2] Applying Transformations")
    print("-" * 80)
    
    # Rotate
    print("  Rotating by 15 degrees...")
    rotated = rotate(composite, angle=np.pi/12, axis=np.array([0, 0, 1]))
    
    # Tile
    print("  Tiling 2x2...")
    rve = tile(rotated, repeats=(2, 2, 1))
    print(f"    RVE: {rve.num_fibers} fibers")
    
    # ============================================================
    # 3. Structural Analysis
    # ============================================================
    print("\n[3] Structural Analysis")
    print("-" * 80)
    
    # Morphology
    print("  Analyzing morphology...")
    morph = MorphologyAnalyzer(rve)
    morph_report = morph.full_report()
    print(f"    Nematic order: {morph_report['nematic_order']:.3f}")
    print(f"    Mean length: {morph_report['mean_length']:.2f}")
    
    # Topology
    print("  Analyzing topology...")
    topo = TopologyAnalyzer(rve)
    topo_report = topo.full_report()
    print(f"    Nodes: {topo_report['num_nodes']}")
    print(f"    Edges: {topo_report['num_edges']}")
    print(f"    Connected: {topo_report['is_connected']}")
    
    # ============================================================
    # 4. Mechanical Simulation (Nonlinear)
    # ============================================================
    print("\n[4] Nonlinear Mechanical Simulation")
    print("-" * 80)
    
    # Use small network for demo
    print("  Setting up FEM model...")
    small = gen.random_straight_2d(num_fibers=15, fiber_length=8, box_size=(20, 20), seed=42)
    
    # Bilinear plasticity
    print("  Using bilinear plasticity model...")
    plasticity = BilinearPlasticity(E=100e9, sigma_y=2e9, Et=5e9)
    
    fem = NonlinearFEM(
        small,
        constitutive_model=plasticity,
        segments_per_fiber=4,
        large_deformation=False
    )
    print(f"    Nodes: {fem.num_nodes}, Elements: {fem.num_elements}")
    
    # Stress-strain curve
    print("  Computing stress-strain curve (10 steps)...")
    strains, stresses, energies = fem.stress_strain_curve(
        axis=0,
        max_strain=0.01,
        num_steps=10
    )
    
    print(f"    Max stress: {stresses[-1]:.2e} Pa at strain {strains[-1]:.4f}")
    if len(strains) > 1:
        E_initial = stresses[1] / strains[1]
        print(f"    Initial modulus: {E_initial:.2e} Pa")
    
    # ============================================================
    # 5. Thermal Simulation
    # ============================================================
    print("\n[5] Thermal Simulation")
    print("-" * 80)
    
    print("  Computing effective thermal conductivity...")
    thermal_results = api.simulate_thermal(small, T_hot=100, T_cold=0, axis=0)
    print(f"    Effective conductivity: {thermal_results['conductivity']:.2f} W/(m·K)")
    
    # ============================================================
    # 6. Export Results
    # ============================================================
    print("\n[6] Exporting Results")
    print("-" * 80)
    
    # VTK
    vtk_file = "/tmp/composite.vtk"
    to_vtk(rve, vtk_file)
    print(f"  VTK: {vtk_file}")
    
    # LAMMPS
    lammps_file = "/tmp/composite.lammps"
    to_lammps(rve, lammps_file, bead_spacing=1.0)
    print(f"  LAMMPS: {lammps_file}")
    
    # JSON
    json_file = "/tmp/composite.json"
    rve.save_json(json_file)
    print(f"  JSON: {json_file}")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("Research Case Study Complete")
    print("=" * 80)
    print("\nKey Results:")
    print(f"  - Composite: {rve.num_fibers} fibers, {rve.num_crosslinks} crosslinks")
    print(f"  - Nematic order: {morph_report['nematic_order']:.3f}")
    if len(strains) > 1:
        print(f"  - Modulus: {E_initial:.2e} Pa")
    print(f"  - Max stress: {stresses[-1]:.2e} Pa")
    print(f"  - Thermal conductivity: {thermal_results['conductivity']:.2f} W/(m·K)")
    print("\nOutputs saved to /tmp/")
    print("=" * 80)


if __name__ == "__main__":
    main()

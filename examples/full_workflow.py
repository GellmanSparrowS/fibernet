"""
Complete workflow example - from network generation to analysis and simulation.

Demonstrates a research workflow:
1. Generate complex fiber network
2. Apply transformations
3. Analyze structure
4. Run mechanical simulation
5. Run thermal simulation
6. Export results
"""
import sys
sys.path.insert(0, '/home/codex/projects/codex_test/fibernet')

import numpy as np
from fibernet import gen
from fibernet.core.transform import merge, rotate, tile
from fibernet.analysis import MorphologyAnalyzer, TopologyAnalyzer
from fibernet.analysis.advanced import SpectralAnalyzer, PoreAnalyzer, AnisotropyAnalyzer
from fibernet.sim.mechanical import FiberFEM
from fibernet.sim.thermal import ThermalSolver

def main():
    print("=" * 70)
    print("FiberNet Complete Workflow Example")
    print("=" * 70)
    
    # ===== Step 1: Generate Base Structures =====
    print("\n[Step 1] Generating Base Structures")
    print("-" * 70)
    
    print("Creating square lattice unit cell...")
    lattice = gen.square_lattice_2d(spacing=5, grid_size=(3, 3))
    print(f"  Lattice: {lattice.num_fibers} fibers, {lattice.num_crosslinks} crosslinks")
    
    print("Creating electrospun nanofiber mat...")
    electrospun = gen.electrospun_network(
        num_fibers=50, fiber_length=20, box_size=(20, 20),
        radius_mean=0.2, waviness=0.3, seed=42,
    )
    print(f"  Electrospun: {electrospun.num_fibers} fibers")
    
    # ===== Step 2: Apply Transformations =====
    print("\n[Step 2] Applying Transformations")
    print("-" * 70)
    
    lattice_rot = rotate(lattice, angle=np.pi/6, axis=np.array([0, 0, 1]))
    print("Rotated lattice by 30 degrees")
    
    composite = merge([lattice_rot, electrospun])
    print(f"  Composite: {composite.num_fibers} fibers")
    
    tiled = tile(composite, repeats=(2, 2, 1), spacing=np.array([30, 30, 0]))
    print(f"  Tiled (2x2): {tiled.num_fibers} fibers")
    
    # ===== Step 3: Structural Analysis =====
    print("\n[Step 3] Structural Analysis")
    print("-" * 70)
    
    morph = MorphologyAnalyzer(tiled)
    morph_report = morph.full_report()
    print("Morphology:")
    print(f"  Total length: {morph_report['total_length']:.1f}")
    print(f"  Mean length: {morph_report['mean_length']:.2f}")
    print(f"  Nematic order: {morph_report['nematic_order']:.3f}")
    
    topo = TopologyAnalyzer(tiled)
    topo_report = topo.full_report()
    print("\nTopology:")
    print(f"  Nodes: {topo_report['num_nodes']}")
    print(f"  Edges: {topo_report['num_edges']}")
    print(f"  Connected: {topo_report['is_connected']}")
    
    spectral = SpectralAnalyzer(tiled)
    print("\nSpectral:")
    print(f"  Spectral gap: {spectral.spectral_gap():.4f}")
    print(f"  Spectral entropy: {spectral.spectral_entropy():.4f}")
    
    pore = PoreAnalyzer(tiled)
    pore_stats = pore.pore_size_statistics()
    print("\nPore Structure:")
    print(f"  Mean pore size: {pore_stats['mean']:.3f}")
    print(f"  Median pore size: {pore_stats['median']:.3f}")
    
    aniso = AnisotropyAnalyzer(tiled)
    print("\nAnisotropy:")
    print(f"  Anisotropy index: {aniso.anisotropy_index():.3f}")
    
    # ===== Step 4: Mechanical Simulation =====
    print("\n[Step 4] Mechanical Simulation")
    print("-" * 70)
    
    small = gen.square_lattice_2d(spacing=5, grid_size=(3, 3))
    fem = FiberFEM(small, segments_per_fiber=3)
    print(f"  FEM: {fem.num_nodes} nodes, {fem.num_elements} elements")
    
    result = fem.apply_uniaxial_strain(strain=0.001, axis=0)
    print(f"  Max displacement: {result.max_displacement():.6f}")
    print(f"  Strain energy: {result.energy:.2e} J")
    
    E_eff = fem.effective_modulus(strain=0.001, axis=0)
    print(f"  Effective modulus: {E_eff:.2e} Pa")
    
    # ===== Step 5: Thermal Simulation =====
    print("\n[Step 5] Thermal Simulation")
    print("-" * 70)
    
    thermal = ThermalSolver(small)
    print(f"  Thermal: {thermal.num_nodes} nodes, {thermal.num_elements} elements")
    
    result_x = thermal.solve_steady_state(T_hot=100, T_cold=0, axis=0)
    result_y = thermal.solve_steady_state(T_hot=100, T_cold=0, axis=1)
    
    print(f"  k_x: {result_x.effective_conductivity:.2f} W/(m*K)")
    print(f"  k_y: {result_y.effective_conductivity:.2f} W/(m*K)")
    
    # ===== Step 6: Export Results =====
    print("\n[Step 6] Exporting Results")
    print("-" * 70)
    
    output_path = "/tmp/fibernet_workflow.json"
    tiled.save_json(output_path)
    print(f"  Network saved to: {output_path}")
    
    import json
    report = {
        "structure": {
            "num_fibers": tiled.num_fibers,
            "num_crosslinks": tiled.num_crosslinks,
            "nematic_order": morph_report['nematic_order'],
        },
        "mechanical": {
            "effective_modulus_Pa": E_eff,
            "strain_energy_J": result.energy,
        },
        "thermal": {
            "k_x_W_mK": result_x.effective_conductivity,
            "k_y_W_mK": result_y.effective_conductivity,
        },
        "pore_structure": pore_stats,
    }
    
    report_path = "/tmp/fibernet_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved to: {report_path}")
    
    # ===== Summary =====
    print("\n" + "=" * 70)
    print("Workflow Complete!")
    print("=" * 70)
    print(f"  {tiled.num_fibers} fibers, {tiled.num_crosslinks} crosslinks")
    print(f"  Effective modulus: {E_eff:.2e} Pa")
    print(f"  k_x: {result_x.effective_conductivity:.2f}, k_y: {result_y.effective_conductivity:.2f} W/(m*K)")
    print("=" * 70)

if __name__ == "__main__":
    main()

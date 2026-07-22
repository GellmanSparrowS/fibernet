"""
Basic usage example for FiberNet.

Demonstrates:
1. Generating different fiber network types
2. Running basic analyses
3. Mechanical simulation
4. Visualization
"""

import numpy as np
import fibernet as fn
from fibernet import gen, sim, viz, analysis


def main():
    print("=" * 60)
    print("FiberNet Basic Usage Example")
    print("=" * 60)
    
    # --- 1. Generate networks ---
    print("\n--- Generating Networks ---")
    
    # Random 2D
    net_2d = gen.random_straight_2d(num_fibers=100, fiber_length=10, box_size=(50, 50), radius=0.1, seed=42)
    print(f"2D Random: {net_2d}")
    
    # Random 3D
    net_3d = gen.random_straight_3d(num_fibers=150, fiber_length=12, box_size=(40, 40, 40), radius=0.15, seed=42)
    print(f"3D Random: {net_3d}")
    
    # Honeycomb
    net_hc = gen.honeycomb_lattice_2d(cell_size=5, grid_size=(5, 5))
    print(f"Honeycomb: {net_hc}")
    
    # Double helix
    net_dh = gen.double_helix(helix_radius=3, pitch=2, num_turns=3, fiber_radius=0.1)
    print(f"Double Helix: {net_dh}")
    
    # Woven
    net_woven = gen.plain_weave_2d(spacing=2, grid_size=(10, 10), radius=0.05)
    print(f"Plain Weave: {net_woven}")
    
    # --- 2. Analysis ---
    print("\n--- Morphology Analysis (3D Random) ---")
    morph = analysis.MorphologyAnalyzer(net_3d)
    report = morph.full_report()
    for k, v in report.items():
        print(f"  {k}: {v}")
    
    print("\n--- Topology Analysis (3D Random) ---")
    topo = analysis.TopologyAnalyzer(net_3d)
    topo_report = topo.full_report()
    for k, v in topo_report.items():
        print(f"  {k}: {v}")
    
    print("\n--- Property Estimation (3D Random) ---")
    props = analysis.PropertyEstimator(net_3d)
    prop_report = props.full_report()
    for k, v in prop_report.items():
        print(f"  {k}: {v}")
    
    # --- 3. Mechanical Simulation ---
    print("\n--- Mechanical Simulation (Honeycomb) ---")
    fem = sim.FiberFEM(net_hc, segments_per_fiber=3)
    print(f"  FEM nodes: {fem.num_nodes}, elements: {fem.num_elements}")
    
    result = fem.apply_uniaxial_strain(strain=0.001, axis=0)
    print(f"  Max displacement: {result.max_displacement():.6e}")
    print(f"  Max stress: {result.max_stress():.6e}")
    print(f"  Strain energy: {result.energy:.6e}")
    
    E_eff = fem.effective_modulus(strain=0.001, axis=0)
    print(f"  Effective modulus: {E_eff:.2e} Pa")
    
    # --- 4. Save results ---
    print("\n--- Saving Networks ---")
    net_3d.save_json("/tmp/fibernet_example.json")
    print("  Saved to /tmp/fibernet_example.json")
    
    loaded = fn.FiberNetwork.load_json("/tmp/fibernet_example.json")
    print(f"  Loaded: {loaded}")
    
    # --- 5. Visualization ---
    print("\n--- Visualization ---")
    try:
        fig = viz.plot_network_2d(net_2d, color_by="orientation", save_path="/tmp/fibernet_2d.png")
        print("  Saved 2D plot to /tmp/fibernet_2d.png")
        
        fig2 = viz.plot_orientation_distribution(net_2d, save_path="/tmp/fibernet_odf.png")
        print("  Saved ODF plot to /tmp/fibernet_odf.png")
        
        fig3 = viz.plot_length_distribution(net_2d, save_path="/tmp/fibernet_lengths.png")
        print("  Saved length distribution to /tmp/fibernet_lengths.png")
    except Exception as e:
        print(f"  Visualization warning: {e}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

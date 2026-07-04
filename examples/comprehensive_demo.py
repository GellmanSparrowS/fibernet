"""
Comprehensive FiberNet Demo Script
===================================

This script demonstrates the main features of the FiberNet library.
Run with: python -m fibernet.examples.comprehensive_demo

Requires: numpy, scipy (core dependencies)
Optional: matplotlib, networkx, tqdm for enhanced features
"""

import numpy as np
import fibernet as fn
from fibernet import gen, sim, analysis


def section(title):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                  FiberNet Comprehensive Demo                     ║")
    print("║          Fiber Network Generation, Simulation & Analysis         ║")
    print(f"║                        Version {fn.__version__}                          ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    # 1. Network Generation
    section("1. Network Generation")
    
    print("Creating various network types...")
    
    # Random 2D network
    net_2d = fn.create('random_2d', num_fibers=100, fiber_length=10.0, 
                       box_size=(30, 30), seed=42)
    print(f"✓ Random 2D: {net_2d.num_fibers} fibers, {net_2d.num_crosslinks} crosslinks")
    
    # Random 3D network
    net_3d = gen.random_straight_3d(num_fibers=80, fiber_length=15.0,
                                     box_size=(40, 40, 40), seed=123)
    print(f"✓ Random 3D: {net_3d.num_fibers} fibers, {net_3d.num_crosslinks} crosslinks")
    
    # Ordered structures
    square = gen.square_lattice_2d(spacing=5.0, grid_size=(5, 5))
    print(f"✓ Square lattice: {square.num_fibers} fibers")
    
    honeycomb = gen.honeycomb_lattice_2d(cell_size=5.0, grid_size=(4, 4))
    print(f"✓ Honeycomb: {honeycomb.num_fibers} fibers")
    
    # 2. Network Analysis
    section("2. Network Analysis")
    
    stats = fn.analyze(net_2d)
    print("2D Network Statistics:")
    print(f"  - Nematic order parameter: {stats['nematic_order']:.3f}")
    print(f"  - Mean fiber length: {stats['mean_length']:.2f}")
    print(f"  - Total length: {stats['total_length']:.2f}")
    
    # Advanced morphology analysis
    morph = analysis.MorphologyAnalyzer(net_2d)
    porosity = morph.porosity()
    tortuosity = morph.tortuosity_distribution()
    print(f"  - Porosity: {porosity:.3f}")
    print(f"  - Mean tortuosity: {np.mean(tortuosity):.3f}")
    
    # 3. Mechanical Simulation
    section("3. Mechanical Simulation (FEM)")
    
    print("Running uniaxial tension simulation...")
    fem = sim.FiberFEM(net_2d, segments_per_fiber=5)
    result = fem.apply_uniaxial_strain(strain=0.01, axis=0)
    
    print(f"✓ Simulation complete")
    print(f"  - Energy: {result.energy:.2e} J")
    print(f"  - Max displacement: {result.max_displacement():.4f}")
    print(f"  - Max stress: {result.max_stress():.2e} Pa")
    
    # Compute effective modulus
    E_eff = fem.effective_modulus(strain=0.001, axis=0)
    print(f"  - Effective Young's modulus: {E_eff:.2e} Pa")
    
    # 4. Damage Mechanics
    section("4. Damage Mechanics & Fatigue")
    
    print("Running progressive failure analysis...")
    damage_solver = sim.DamageMechanicsSolver(
        net_2d, youngs_modulus=1e9, tensile_strength=1e8
    )
    damage_result = damage_solver.progressive_failure(max_strain=0.05, num_steps=20)
    
    print(f"✓ Damage analysis complete")
    print(f"  - Peak load: {damage_result.peak_load:.2e} N")
    print(f"  - Energy absorbed: {damage_result.energy_absorbed:.2e} J")
    
    # 5. Thermal Simulation
    section("5. Thermal Simulation")
    
    print("Running steady-state heat conduction...")
    thermal_result = fn.simulate_thermal(net_2d, T_hot=100.0, T_cold=0.0)
    
    print(f"✓ Thermal simulation complete")
    print(f"  - Temperature field computed")
    print(f"  - Effective conductivity: {thermal_result['conductivity']:.2e} W/(m·K)")
    
    # 6. Multi-scale Homogenization
    section("6. Multi-scale Homogenization")
    
    print("Computing effective properties via homogenization...")
    homogenizer = sim.HomogenizationSolver(
        net_3d, fiber_youngs_modulus=1e9, fiber_poissons_ratio=0.3
    )
    props = homogenizer.homogenize()
    
    print(f"✓ Homogenization complete")
    print(f"  - Effective Young's modulus: {props.effective_youngs_modulus:.2e} Pa")
    print(f"  - Porosity: {props.porosity:.3f}")
    print(f"  - Is isotropic: {props.is_isotropic}")
    
    # 7. Rheology
    section("7. Fiber Suspension Rheology")
    
    print("Analyzing fiber suspension in fluid flow...")
    rheo = sim.FiberSuspensionRheology(
        net_2d, fluid_viscosity=1.0, aspect_ratio=20.0
    )
    
    eta = rheo.compute_effective_viscosity(shear_rate=10.0)
    orbit = rheo.jeffery_orbit(
        initial_orientation=np.array([1.0, 0.0, 0.0]),
        shear_rate=1.0, total_time=5.0, num_steps=100
    )
    
    print(f"✓ Rheology analysis complete")
    print(f"  - Effective viscosity: {eta:.2f} Pa·s")
    print(f"  - Jeffery orbit period: {orbit.period:.2f} s")
    
    # 8. Percolation Analysis
    section("8. Percolation Analysis")
    
    print("Analyzing network percolation...")
    perc = analysis.PercolationAnalyzer(net_2d)
    perc_result = perc.analyze()
    
    print(f"✓ Percolation analysis complete")
    print(f"  - Percolates: {perc_result.percolates}")
    print(f"  - Largest cluster: {perc_result.largest_cluster_size} fibers")
    print(f"  - Percolation probability: {perc_result.percolation_probability:.3f}")
    
    # 9. Network Transformations
    section("9. Network Transformations")
    
    print("Applying transformations...")
    net_scaled = fn.scale(net_2d, factor=2.0)
    net_rotated = fn.rotate(net_2d, angle=np.pi/4, axis=[0, 0, 1])
    net_merged = fn.merge([net_2d, net_scaled])
    
    print(f"✓ Transformations complete")
    print(f"  - Scaled: {net_scaled.num_fibers} fibers")
    print(f"  - Rotated: {net_rotated.num_fibers} fibers")
    print(f"  - Merged: {net_merged.num_fibers} fibers")
    
    # 10. Export/Import
    section("10. Export/Import")
    
    import tempfile
    import os
    
    print("Exporting network to various formats...")
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, 'network.json')
        fn.export(net_2d, json_path, format='json')
        print(f"  ✓ JSON export: {os.path.getsize(json_path)} bytes")
        
        # Import back
        net_loaded = fn.load(json_path, format='json')
        print(f"  ✓ JSON import: {net_loaded.num_fibers} fibers")
    
    # Summary
    section("Summary")
    
    print("FiberNet provides a comprehensive toolkit for:")
    print("  • Network generation (50+ generators)")
    print("  • Structural analysis (morphology, topology, percolation)")
    print("  • Mechanical simulation (FEM, damage, fatigue)")
    print("  • Multi-physics (thermal, electromagnetic, acoustic)")
    print("  • Multi-scale modeling (homogenization, RVE)")
    print("  • Rheology (fiber suspensions, Jeffery orbits)")
    print("  • Machine learning integration")
    print("  • Export to various formats")
    print("\nFor more examples, see the tutorials/ directory.")
    print("\n🎉 Demo complete!")


if __name__ == '__main__':
    main()

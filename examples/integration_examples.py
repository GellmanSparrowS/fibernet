"""
Integration examples demonstrating end-to-end workflows.

These examples show how different components of FiberNet work together
for complete research workflows.
"""

import numpy as np
from fibernet import gen, analysis, viz
from fibernet.sim import FiberFEM, ThermalSolver, DMA
from fibernet.utils import EnsembleGenerator, ExperimentConfig
from fibernet.ml import GNNFeatureExtractor, PropertyPredictor
from fibernet.sim.periodic import PeriodicBoundary


def example_mechanical_characterization():
    """
    Example: Complete mechanical characterization workflow.
    
    1. Generate network
    2. Run FEM simulation
    3. Extract stress-strain curve
    4. Compute mechanical properties
    5. Visualize results
    """
    print("=== Mechanical Characterization Workflow ===\n")
    
    # 1. Generate network
    network = gen.random_straight_2d(
        num_fibers=100,
        fiber_length=15.0,
        box_size=(50, 50),
        seed=42
    )
    print(f"Generated network: {network.num_fibers} fibers, {network.num_crosslinks} crosslinks")
    
    # 2. Run FEM simulation
    fem = FiberFEM(network, youngs_modulus=1e9, poisson_ratio=0.3)
    result = fem.uniaxial_tension(strain=0.05, axis=0)
    print(f"FEM simulation completed: {len(result.strains)} strain steps")
    
    # 3. Extract stress-strain curve
    analyzer = analysis.MorphologyAnalyzer(network)
    stress_strain = analyzer.extract_stress_strain(result)
    
    # 4. Compute properties
    properties = analyzer.compute_mechanical_properties(stress_strain)
    print(f"Young's modulus: {properties['youngs_modulus']:.2e} Pa")
    print(f"Yield strength: {properties['yield_strength']:.2e} Pa")
    print(f"Ultimate strength: {properties['ultimate_strength']:.2e} Pa")
    
    # 5. Visualize
    fig = viz.plot_stress_strain(stress_strain, title="Stress-Strain Curve")
    fig.savefig("stress_strain_curve.png", dpi=150)
    print("Saved visualization to stress_strain_curve.png\n")
    
    return properties


def example_thermal_simulation():
    """
    Example: Thermal conductivity simulation workflow.
    
    1. Generate network
    2. Assign thermal properties
    3. Run thermal simulation
    4. Compute effective conductivity
    5. Visualize temperature field
    """
    print("=== Thermal Simulation Workflow ===\n")
    
    # 1. Generate network
    network = gen.random_straight_2d(
        num_fibers=80,
        fiber_length=12.0,
        box_size=(40, 40),
        seed=123
    )
    print(f"Generated network: {network.num_fibers} fibers")
    
    # 2. Assign thermal properties
    thermal_conductivity = 0.5  # W/(m·K)
    network.set_property('thermal_conductivity', thermal_conductivity)
    
    # 3. Run thermal simulation
    solver = ThermalSolver(network)
    result = solver.steady_state_heat_conduction(
        boundary_conditions={'left': 300, 'right': 400},
        direction='x'
    )
    print(f"Thermal simulation completed")
    
    # 4. Compute effective conductivity
    k_eff = solver.compute_effective_conductivity(result)
    print(f"Effective thermal conductivity: {k_eff:.4f} W/(m·K)")
    
    # 5. Visualize
    fig = viz.plot_temperature_field(network, result.temperature_field)
    fig.savefig("temperature_field.png", dpi=150)
    print("Saved visualization to temperature_field.png\n")
    
    return k_eff


def example_dma_analysis():
    """
    Example: Dynamic Mechanical Analysis workflow.
    
    1. Generate network
    2. Run frequency sweep
    3. Run temperature sweep
    4. Compute glass transition temperature
    5. Plot master curve
    """
    print("=== DMA Analysis Workflow ===\n")
    
    # 1. Generate network
    network = gen.random_straight_2d(
        num_fibers=60,
        fiber_length=10.0,
        box_size=(30, 30),
        seed=456
    )
    print(f"Generated network: {network.num_fibers} fibers")
    
    # 2. Run frequency sweep
    dma = DMA(network, youngs_modulus=1e9, viscosity=1e6)
    freq_result = dma.frequency_sweep(
        frequencies=np.logspace(-2, 2, 50),
        strain=0.01
    )
    print(f"Frequency sweep: {len(freq_result.frequencies)} points")
    
    # 3. Run temperature sweep
    temp_result = dma.temperature_sweep(
        temperatures=np.linspace(200, 400, 100),
        frequency=1.0,
        strain=0.01
    )
    print(f"Temperature sweep: {len(temp_result.temperatures)} points")
    
    # 4. Compute Tg
    Tg = dma.compute_glass_transition(temp_result)
    print(f"Glass transition temperature: {Tg:.1f} K")
    
    # 5. Plot results
    fig = viz.plot_dma_results(freq_result, temp_result)
    fig.savefig("dma_results.png", dpi=150)
    print("Saved visualization to dma_results.png\n")
    
    return Tg


def example_ensemble_study():
    """
    Example: Statistical ensemble study workflow.
    
    1. Generate ensemble of networks
    2. Compute properties for each
    3. Analyze statistics
    4. Check convergence
    5. Report results
    """
    print("=== Ensemble Study Workflow ===\n")
    
    # 1. Generate ensemble
    ensemble_gen = EnsembleGenerator(
        generator=gen.random_straight_2d,
        params={'num_fibers': 50, 'fiber_length': 10.0, 'box_size': (30, 30)},
        num_samples=20,
        seed_range=(0, 100)
    )
    ensemble = ensemble_gen.generate()
    print(f"Generated ensemble: {len(ensemble)} networks")
    
    # 2. Compute properties
    properties_list = []
    for network in ensemble:
        analyzer = analysis.MorphologyAnalyzer(network)
        props = {
            'num_fibers': network.num_fibers,
            'num_crosslinks': network.num_crosslinks,
            'mean_length': analyzer.mean_fiber_length(),
            'nematic_order': analyzer.nematic_order_parameter(),
            'porosity': analyzer.porosity(),
        }
        properties_list.append(props)
    
    # 3. Analyze statistics
    stats = ensemble_gen.compute_statistics(properties_list)
    print(f"\nEnsemble statistics:")
    for key, value in stats['mean'].items():
        std = stats['std'][key]
        print(f"  {key}: {value:.3f} ± {std:.3f}")
    
    # 4. Check convergence
    convergence = ensemble_gen.check_convergence(properties_list)
    print(f"\nConvergence analysis:")
    for key, converged in convergence.items():
        print(f"  {key}: {'Converged' if converged else 'Not converged'}")
    
    # 5. Report
    print("\nEnsemble study completed\n")
    
    return stats


def example_periodic_boundary():
    """
    Example: Periodic boundary condition workflow.
    
    1. Generate network with periodic boundaries
    2. Apply periodic constraints
    3. Run simulation
    4. Compute bulk properties
    5. Compare with non-periodic
    """
    print("=== Periodic Boundary Workflow ===\n")
    
    # 1. Generate network with periodic boundaries
    box_size = (50, 50)
    network = gen.random_straight_2d(
        num_fibers=100,
        fiber_length=15.0,
        box_size=box_size,
        seed=789,
        periodic=True
    )
    print(f"Generated periodic network: {network.num_fibers} fibers")
    
    # 2. Set up periodic boundary conditions
    pbc = PeriodicBoundary(box_size)
    
    # 3. Run FEM with periodic BCs
    fem = FiberFEM(network, youngs_modulus=1e9, poisson_ratio=0.3)
    result = fem.uniaxial_tension(strain=0.05, axis=0, periodic=True)
    print(f"Periodic FEM simulation completed")
    
    # 4. Compute bulk properties
    analyzer = analysis.MorphologyAnalyzer(network)
    stress_strain = analyzer.extract_stress_strain(result)
    properties = analyzer.compute_mechanical_properties(stress_strain)
    
    print(f"Bulk Young's modulus: {properties['youngs_modulus']:.2e} Pa")
    
    # 5. Compare with non-periodic
    network_nonperiodic = gen.random_straight_2d(
        num_fibers=100,
        fiber_length=15.0,
        box_size=box_size,
        seed=789,
        periodic=False
    )
    fem_np = FiberFEM(network_nonperiodic, youngs_modulus=1e9, poisson_ratio=0.3)
    result_np = fem_np.uniaxial_tension(strain=0.05, axis=0, periodic=False)
    stress_strain_np = analyzer.extract_stress_strain(result_np)
    properties_np = analyzer.compute_mechanical_properties(stress_strain_np)
    
    print(f"Non-periodic Young's modulus: {properties_np['youngs_modulus']:.2e} Pa")
    print(f"Difference: {abs(properties['youngs_modulus'] - properties_np['youngs_modulus']) / properties_np['youngs_modulus'] * 100:.1f}%")
    print("\nPeriodic boundary study completed\n")
    
    return properties


def example_machine_learning():
    """
    Example: Machine learning workflow for property prediction.
    
    1. Generate training dataset
    2. Extract features
    3. Train model
    4. Evaluate on test set
    5. Make predictions
    """
    print("=== Machine Learning Workflow ===\n")
    
    # 1. Generate training dataset
    print("Generating training dataset...")
    train_networks = []
    train_labels = []
    
    for i in range(50):
        # Vary network parameters
        num_fibers = np.random.randint(50, 150)
        fiber_length = np.random.uniform(8, 15)
        
        network = gen.random_straight_2d(
            num_fibers=num_fibers,
            fiber_length=fiber_length,
            box_size=(40, 40),
            seed=i
        )
        
        # Compute target property (e.g., stiffness)
        fem = FiberFEM(network, youngs_modulus=1e9, poisson_ratio=0.3)
        result = fem.uniaxial_tension(strain=0.01, axis=0)
        analyzer = analysis.MorphologyAnalyzer(network)
        stress_strain = analyzer.extract_stress_strain(result)
        props = analyzer.compute_mechanical_properties(stress_strain)
        
        train_networks.append(network)
        train_labels.append(props['youngs_modulus'])
    
    print(f"Generated {len(train_networks)} training samples")
    
    # 2. Extract features
    print("Extracting features...")
    extractor = GNNFeatureExtractor()
    X_train = []
    
    for network in train_networks:
        features = extractor.extract_features(network)
        X_train.append(features)
    
    X_train = np.array(X_train)
    y_train = np.array(train_labels)
    print(f"Feature matrix shape: {X_train.shape}")
    
    # 3. Train model
    print("Training model...")
    predictor = PropertyPredictor(model_type='random_forest')
    predictor.fit(X_train, y_train)
    print("Model trained")
    
    # 4. Evaluate on test set
    print("Evaluating on test set...")
    test_networks = []
    test_labels = []
    
    for i in range(10):
        network = gen.random_straight_2d(
            num_fibers=100,
            fiber_length=12.0,
            box_size=(40, 40),
            seed=100 + i
        )
        
        fem = FiberFEM(network, youngs_modulus=1e9, poisson_ratio=0.3)
        result = fem.uniaxial_tension(strain=0.01, axis=0)
        analyzer = analysis.MorphologyAnalyzer(network)
        stress_strain = analyzer.extract_stress_strain(result)
        props = analyzer.compute_mechanical_properties(stress_strain)
        
        test_networks.append(network)
        test_labels.append(props['youngs_modulus'])
    
    X_test = []
    for network in test_networks:
        features = extractor.extract_features(network)
        X_test.append(features)
    
    X_test = np.array(X_test)
    y_test = np.array(test_labels)
    
    metrics = predictor.evaluate(X_test, y_test)
    print(f"Test R² score: {metrics['r2']:.3f}")
    print(f"Test RMSE: {metrics['rmse']:.2e} Pa")
    
    # 5. Make predictions
    print("Making predictions on new networks...")
    new_network = gen.random_straight_2d(
        num_fibers=120,
        fiber_length=13.0,
        box_size=(40, 40),
        seed=999
    )
    
    new_features = extractor.extract_features(new_network)
    prediction = predictor.predict(new_features.reshape(1, -1))
    print(f"Predicted Young's modulus: {prediction[0]:.2e} Pa")
    
    print("\nMachine learning workflow completed\n")
    
    return metrics


def example_config_driven_workflow():
    """
    Example: Configuration-driven reproducible workflow.
    
    1. Load configuration
    2. Generate network from config
    3. Run simulation from config
    4. Save results
    5. Verify reproducibility
    """
    print("=== Configuration-Driven Workflow ===\n")
    
    # 1. Create configuration
    config = ExperimentConfig(
        name="tensile_test",
        network={
            'generator': 'random_straight_2d',
            'num_fibers': 80,
            'fiber_length': 12.0,
            'box_size': (40, 40),
            'seed': 42
        },
        simulation={
            'type': 'fem',
            'youngs_modulus': 1e9,
            'poisson_ratio': 0.3,
            'strain': 0.05,
            'axis': 0
        },
        analysis={
            'compute_properties': True,
            'save_visualization': True
        }
    )
    
    print("Created experiment configuration")
    config.save("experiment_config.yaml")
    print("Saved configuration to experiment_config.yaml")
    
    # 2. Generate network from config
    network = gen.from_config(config.network)
    print(f"Generated network: {network.num_fibers} fibers")
    
    # 3. Run simulation from config
    if config.simulation['type'] == 'fem':
        fem = FiberFEM(
            network,
            youngs_modulus=config.simulation['youngs_modulus'],
            poisson_ratio=config.simulation['poisson_ratio']
        )
        result = fem.uniaxial_tension(
            strain=config.simulation['strain'],
            axis=config.simulation['axis']
        )
        print("FEM simulation completed")
    
    # 4. Save results
    if config.analysis['compute_properties']:
        analyzer = analysis.MorphologyAnalyzer(network)
        stress_strain = analyzer.extract_stress_strain(result)
        properties = analyzer.compute_mechanical_properties(stress_strain)
        
        import json
        with open("results.json", 'w') as f:
            json.dump(properties, f, indent=2)
        print("Saved results to results.json")
    
    if config.analysis['save_visualization']:
        fig = viz.plot_stress_strain(stress_strain, title=config.name)
        fig.savefig("config_results.png", dpi=150)
        print("Saved visualization to config_results.png")
    
    # 5. Verify reproducibility
    print("\nVerifying reproducibility...")
    network2 = gen.from_config(config.network)
    assert network2.num_fibers == network.num_fibers
    assert network2.num_crosslinks == network.num_crosslinks
    print("Reproducibility verified")
    
    print("\nConfiguration-driven workflow completed\n")
    
    return properties


def example_comprehensive_study():
    """
    Example: Comprehensive multi-physics study.
    
    Combines mechanical, thermal, and structural analysis
    for a complete material characterization.
    """
    print("=== Comprehensive Multi-Physics Study ===\n")
    
    # Generate network
    network = gen.random_straight_2d(
        num_fibers=100,
        fiber_length=15.0,
        box_size=(50, 50),
        seed=42
    )
    print(f"Network: {network.num_fibers} fibers, {network.num_crosslinks} crosslinks")
    
    # Structural analysis
    print("\n--- Structural Analysis ---")
    analyzer = analysis.MorphologyAnalyzer(network)
    print(f"Mean fiber length: {analyzer.mean_fiber_length():.2f}")
    print(f"Nematic order: {analyzer.nematic_order_parameter():.3f}")
    print(f"Porosity: {analyzer.porosity():.3f}")
    print(f"Connectivity: {analyzer.mean_connectivity():.2f}")
    
    # Mechanical analysis
    print("\n--- Mechanical Analysis ---")
    fem = FiberFEM(network, youngs_modulus=1e9, poisson_ratio=0.3)
    
    # Test in multiple directions
    for axis in [0, 1]:
        result = fem.uniaxial_tension(strain=0.05, axis=axis)
        stress_strain = analyzer.extract_stress_strain(result)
        props = analyzer.compute_mechanical_properties(stress_strain)
        print(f"Axis {axis}: E = {props['youngs_modulus']:.2e} Pa, σ_y = {props['yield_strength']:.2e} Pa")
    
    # Thermal analysis
    print("\n--- Thermal Analysis ---")
    network.set_property('thermal_conductivity', 0.5)
    thermal_solver = ThermalSolver(network)
    thermal_result = thermal_solver.steady_state_heat_conduction(
        boundary_conditions={'left': 300, 'right': 400},
        direction='x'
    )
    k_eff = thermal_solver.compute_effective_conductivity(thermal_result)
    print(f"Effective thermal conductivity: {k_eff:.4f} W/(m·K)")
    
    # DMA analysis
    print("\n--- Dynamic Mechanical Analysis ---")
    dma = DMA(network, youngs_modulus=1e9, viscosity=1e6)
    freq_result = dma.frequency_sweep(
        frequencies=np.logspace(-1, 1, 20),
        strain=0.01
    )
    print(f"Frequency sweep: {len(freq_result.frequencies)} points")
    print(f"Storage modulus range: {freq_result.storage_modulus.min():.2e} - {freq_result.storage_modulus.max():.2e} Pa")
    
    print("\nComprehensive study completed\n")
    
    return {
        'structural': {
            'mean_length': analyzer.mean_fiber_length(),
            'nematic_order': analyzer.nematic_order_parameter(),
            'porosity': analyzer.porosity()
        },
        'thermal': {'k_eff': k_eff},
        'dma': {'storage_modulus_range': (freq_result.storage_modulus.min(), freq_result.storage_modulus.max())}
    }


if __name__ == "__main__":
    # Run all examples
    print("=" * 70)
    print("FIBERNET INTEGRATION EXAMPLES")
    print("=" * 70)
    print()
    
    # Example 1: Mechanical characterization
    mech_props = example_mechanical_characterization()
    
    # Example 2: Thermal simulation
    k_eff = example_thermal_simulation()
    
    # Example 3: DMA analysis
    Tg = example_dma_analysis()
    
    # Example 4: Ensemble study
    stats = example_ensemble_study()
    
    # Example 5: Periodic boundaries
    periodic_props = example_periodic_boundary()
    
    # Example 6: Machine learning
    ml_metrics = example_machine_learning()
    
    # Example 7: Configuration-driven workflow
    config_props = example_config_driven_workflow()
    
    # Example 8: Comprehensive study
    comprehensive = example_comprehensive_study()
    
    print("=" * 70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 70)

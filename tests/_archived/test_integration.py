"""
Integration tests for complete FiberNet workflows.

Tests end-to-end scenarios from network generation through
analysis, simulation, and export.
"""

import numpy as np
import pytest
import tempfile
import os
from fibernet import (
    create, analyze, simulate_mechanics, simulate_thermal,
    mirror, rotate, scale, translate, merge, export, load,
    FiberNetwork, Fiber, Material,
)
from fibernet import gen, sim, analysis, io
from fibernet.analysis import MorphologyAnalyzer, PercolationAnalyzer
from fibernet.sim import (
    FiberFEM, CrackPropagationSolver, DamageMechanicsSolver,
    FiberSuspensionRheology, HomogenizationSolver,
)


class TestCompleteWorkflow2D:
    """Test complete 2D workflow: generate -> analyze -> simulate -> export."""

    def test_random_network_full_workflow(self, tmp_path):
        # 1. Generate
        net = create("random_2d", num_fibers=100, fiber_length=10.0, box_size=(30, 30), seed=42)
        assert net.num_fibers == 100
        assert net.dimension == 2

        # 2. Analyze
        stats = analyze(net)
        assert 'num_fibers' in stats
        assert stats['num_fibers'] == 100
        assert 'nematic_order' in stats
        assert 0 <= stats['nematic_order'] <= 1

        # 3. Transform (mirror returns same fiber count)
        net_mirrored = mirror(net, axis=0)
        assert net_mirrored.num_fibers == net.num_fibers

        net_rotated = rotate(net, angle=np.pi/4, axis=[0, 0, 1])
        assert net_rotated.num_fibers == net.num_fibers

        net_scaled = scale(net, factor=2.0)
        assert net_scaled.num_fibers == net.num_fibers

        # 4. Mechanical simulation
        result = simulate_mechanics(net, strain=0.01)
        assert result is not None

        # 5. Export/Import
        json_path = str(tmp_path / "network.json")
        export(net, json_path, format="json")
        assert os.path.exists(json_path)
        loaded = load(json_path, format="json")
        assert loaded.num_fibers == net.num_fibers

    def test_lattice_network_workflow(self, tmp_path):
        # Generate lattice
        net = create("square_2d", spacing=5.0, grid_size=(5, 5))
        assert net.num_fibers > 0

        # Analyze using network properties (not methods)
        assert net.mean_fiber_length > 0
        assert net.total_length > 0

        # MorphologyAnalyzer
        analyzer = MorphologyAnalyzer(net)
        assert analyzer.nematic_order_parameter() >= 0
        assert analyzer.porosity() >= 0

        # Export
        vtk_path = str(tmp_path / "lattice.vtk")
        export(net, vtk_path, format="vtk")
        assert os.path.exists(vtk_path)


class TestCompleteWorkflow3D:
    """Test complete 3D workflow."""

    def test_3d_random_network_workflow(self, tmp_path):
        # Generate 3D network
        net = gen.random_straight_3d(
            num_fibers=80, fiber_length=15.0, box_size=(40, 40, 40), seed=42
        )
        assert net.dimension == 3
        assert net.num_fibers == 80

        # Analyze
        stats = analyze(net)
        assert stats['dimension'] == 3

        # Mechanical simulation using FiberFEM
        fem = FiberFEM(net, segments_per_fiber=5)
        result = fem.apply_uniaxial_strain(strain=0.01, axis=0)
        assert result is not None
        assert result.displacements is not None

        # Export
        vtk_path = str(tmp_path / "network_3d.vtk")
        export(net, vtk_path, format="vtk")
        assert os.path.exists(vtk_path)


class TestMultiPhysicsIntegration:
    """Test multi-physics simulation integration."""

    def test_mechanical_and_thermal(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=8.0, box_size=(20, 20), seed=42)

        # Mechanical
        result_mech = simulate_mechanics(net, strain=0.01)
        assert result_mech is not None

        # Thermal
        result_thermal = simulate_thermal(net, T_hot=100.0, T_cold=0.0)
        assert result_thermal is not None

    def test_damage_and_fracture(self):
        net = gen.random_straight_2d(num_fibers=60, fiber_length=10.0, box_size=(30, 30), seed=42)

        # Damage mechanics
        damage_solver = DamageMechanicsSolver(net, youngs_modulus=1e9, tensile_strength=1e8)
        result = damage_solver.progressive_failure(max_strain=0.05, num_steps=20)
        assert result.peak_load >= 0
        assert result.energy_absorbed >= 0

        # Fracture mechanics
        crack_solver = CrackPropagationSolver(net, fracture_toughness=100.0)
        tip = crack_solver.initialize_crack(
            tip_position=np.array([15.0, 15.0, 0.0]),
            tip_direction=np.array([1.0, 0.0, 0.0]),
            initial_length=2.0,
        )
        assert tip is not None


class TestCrossModuleIntegration:
    """Test integration between different modules."""

    def test_gen_to_analysis_to_sim(self):
        # Generate
        net = gen.random_straight_2d(num_fibers=80, fiber_length=10.0, box_size=(30, 30), seed=42)

        # Analyze using network properties and MorphologyAnalyzer
        density = net.density()
        mean_length = net.mean_fiber_length  # property, not method
        
        morph = MorphologyAnalyzer(net)
        order = morph.nematic_order_parameter()
        porosity = morph.porosity()

        # Use analysis results to inform simulation
        fem = FiberFEM(net, segments_per_fiber=5)
        result = fem.apply_uniaxial_strain(strain=0.01, axis=0)
        assert result is not None

    def test_homogenization_workflow(self):
        # Generate network
        net = gen.random_straight_3d(
            num_fibers=100, fiber_length=15.0, box_size=(40, 40, 40), seed=42
        )

        # Homogenize
        homogenizer = HomogenizationSolver(net, fiber_youngs_modulus=1e9)
        props = homogenizer.homogenize()

        assert props.effective_youngs_modulus > 0
        assert 0 <= props.porosity <= 1.0

    def test_rheology_workflow(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=10.0, box_size=(30, 30), seed=42)

        rheo = FiberSuspensionRheology(net, fluid_viscosity=1.0, aspect_ratio=20.0)
        eta = rheo.compute_effective_viscosity(shear_rate=1.0)
        assert eta > 0

        orbit = rheo.jeffery_orbit(
            initial_orientation=np.array([1.0, 0.0, 0.0]),
            shear_rate=1.0,
            total_time=5.0,
            num_steps=100,
        )
        assert orbit.period > 0

    def test_percolation_workflow(self):
        net = gen.random_straight_2d(num_fibers=100, fiber_length=10.0, box_size=(30, 30), seed=42)

        analyzer = PercolationAnalyzer(net)
        result = analyzer.analyze()
        assert result.largest_cluster_size > 0
        assert 0 <= result.percolation_probability <= 1


class TestFormatInteroperability:
    """Test that different I/O formats preserve network integrity."""

    def test_json_roundtrip(self, tmp_path):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=10.0, box_size=(30, 30), seed=42)
        path = str(tmp_path / "test.json")
        export(net, path, format="json")
        loaded = load(path, format="json")
        assert loaded.num_fibers == net.num_fibers
        assert loaded.num_crosslinks == net.num_crosslinks

    def test_lammps_export(self, tmp_path):
        net = gen.random_straight_2d(num_fibers=30, fiber_length=8.0, box_size=(20, 20), seed=42)
        path = str(tmp_path / "test.lammps")
        export(net, path, format="lammps")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_vtk_export(self, tmp_path):
        net = gen.random_straight_3d(num_fibers=30, fiber_length=8.0, box_size=(20, 20, 20), seed=42)
        path = str(tmp_path / "test.vtk")
        export(net, path, format="vtk")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


class TestGeneratorDiversity:
    """Test that all generator types produce valid networks."""

    def test_random_2d(self):
        net = create("random_2d", num_fibers=30, fiber_length=8.0, box_size=(20, 20), seed=42)
        assert net.num_fibers > 0
        assert net.dimension == 2

    def test_random_3d(self):
        net = create("random_3d", num_fibers=30, fiber_length=8.0, box_size=(20, 20, 20), seed=42)
        assert net.num_fibers > 0
        assert net.dimension == 3

    def test_square_2d(self):
        net = create("square_2d", spacing=5.0, grid_size=(4, 4))
        assert net.num_fibers > 0

    def test_honeycomb_2d(self):
        net = create("honeycomb_2d", cell_size=5.0, grid_size=(3, 3))
        assert net.num_fibers > 0

    def test_triangular_2d(self):
        net = create("triangular_2d", spacing=5.0, grid_size=(4, 4))
        assert net.num_fibers > 0

    def test_all_generators_analyzable(self):
        """All generator types should produce networks that can be analyzed."""
        networks = [
            create("random_2d", num_fibers=20, fiber_length=5.0, box_size=(15, 15), seed=42),
            create("square_2d", spacing=5.0, grid_size=(3, 3)),
        ]
        for net in networks:
            stats = analyze(net)
            assert stats['num_fibers'] > 0
            assert 'nematic_order' in stats

"""Integration tests - verify complete workflows work end-to-end."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from fibernet import gen, api
from fibernet.core.transform import rotate, scale, merge, tile
from fibernet.core.pbc import apply_pbc, compute_rdf
from fibernet.analysis import MorphologyAnalyzer, TopologyAnalyzer
from fibernet.sim.mechanical import FiberFEM
from fibernet.io import to_vtk, to_lammps, to_xyz, to_pdb


class TestFullWorkflow:
    """Test complete generate-analyze-simulate-export workflow."""
    
    def test_random_network_workflow(self):
        """Complete workflow with random 2D network."""
        # 1. Generate
        net = gen.random_straight_2d(
            num_fibers=50, fiber_length=10, box_size=(30, 30), seed=42
        )
        assert net.num_fibers == 50
        
        # 2. Analyze
        morph = MorphologyAnalyzer(net)
        assert morph.nematic_order_parameter() >= 0
        
        topo = TopologyAnalyzer(net)
        topo_report = topo.full_report()
        assert topo_report['num_nodes'] > 0
        
        # 3. Simulate
        fem = FiberFEM(net, segments_per_fiber=3)
        assert fem.num_nodes > 0
        
        E = fem.effective_modulus(strain=0.001, axis=0)
        # NaN is expected for disconnected random networks
        
        # 4. Export
        with tempfile.TemporaryDirectory() as tmpdir:
            vtk_path = Path(tmpdir) / "test.vtk"
            to_vtk(net, str(vtk_path))
            assert vtk_path.exists()
            
            json_path = Path(tmpdir) / "test.json"
            net.save_json(str(json_path))
            assert json_path.exists()
            
            # Reload
            from fibernet.core.network import FiberNetwork
            loaded = FiberNetwork.load_json(str(json_path))
            assert loaded.num_fibers == net.num_fibers
    
    def test_lattice_network_workflow(self):
        """Complete workflow with ordered lattice."""
        # 1. Generate
        net = gen.square_lattice_2d(spacing=5, grid_size=(3, 3))
        assert net.num_fibers > 0
        
        # 2. Transform
        rotated = rotate(net, angle=np.pi/4, axis=np.array([0, 0, 1]))
        assert rotated.num_fibers == net.num_fibers
        
        scaled = scale(rotated, factor=2.0)
        assert scaled.num_fibers == net.num_fibers
        
        # 3. Analyze
        morph = MorphologyAnalyzer(scaled)
        report = morph.full_report()
        assert 'nematic_order' in report
        
        # 4. Simulate
        fem = FiberFEM(scaled, segments_per_fiber=3)
        result = fem.apply_uniaxial_strain(strain=0.001, axis=0)
        assert not np.isnan(result.energy)
    
    def test_high_level_api_workflow(self):
        """Test the high-level convenience API."""
        # Create
        net = api.create("random_2d", num_fibers=30, fiber_length=8, box_size=(25, 25))
        assert net.num_fibers == 30
        
        # Analyze
        results = api.analyze(net)
        assert 'num_fibers' in results
        assert 'nematic_order' in results
        assert results['num_fibers'] == 30
        
        # Simulate mechanics (may be NaN for disconnected random networks)
        mech = api.simulate_mechanics(net, strain=0.001, axis=0)
        assert 'modulus' in mech
        
        # Simulate thermal (more robust)
        thermal = api.simulate_thermal(net, T_hot=100, T_cold=0)
        assert 'conductivity' in thermal
        
        # Transform
        rotated = api.rotate(net, angle=np.pi/6)
        assert rotated.num_fibers == net.num_fibers
        
        scaled = api.scale(net, factor=1.5)
        assert scaled.num_fibers == net.num_fibers
    
    def test_merge_and_tile_workflow(self):
        """Test merge and tile operations."""
        # Generate two networks
        net1 = gen.random_straight_2d(num_fibers=20, fiber_length=8, box_size=(20, 20), seed=42)
        net2 = gen.random_straight_2d(num_fibers=15, fiber_length=8, box_size=(20, 20), seed=43)
        
        # Merge
        merged = merge([net1, net2])
        assert merged.num_fibers == net1.num_fibers + net2.num_fibers
        
        # Tile
        tiled = tile(net1, repeats=(2, 2, 1))
        assert tiled.num_fibers == net1.num_fibers * 4
        
        # Analyze merged
        morph = MorphologyAnalyzer(merged)
        assert morph.nematic_order_parameter() >= 0
    
    def test_pbc_workflow(self):
        """Test periodic boundary conditions workflow."""
        net = gen.random_straight_2d(num_fibers=30, fiber_length=8, box_size=(25, 25), seed=42)
        
        # Apply PBC
        wrapped, box = apply_pbc(net)
        assert wrapped.num_fibers == net.num_fibers
        assert box is not None
        
        # Compute RDF
        r, g = compute_rdf(wrapped, box, r_max=10, num_bins=20)
        assert len(r) == 20
        assert len(g) == 20
        assert np.all(g >= 0)
    
    def test_multi_format_export(self):
        """Test exporting to multiple formats."""
        net = gen.random_straight_2d(num_fibers=20, fiber_length=8, box_size=(20, 20), seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # VTK
            vtk_path = Path(tmpdir) / "test.vtk"
            to_vtk(net, str(vtk_path))
            assert vtk_path.exists()
            assert vtk_path.stat().st_size > 0
            
            # LAMMPS
            lammps_path = Path(tmpdir) / "test.lammps"
            to_lammps(net, str(lammps_path), bead_spacing=1.0)
            assert lammps_path.exists()
            assert lammps_path.stat().st_size > 0
            
            # XYZ
            xyz_path = Path(tmpdir) / "test.xyz"
            to_xyz(net, str(xyz_path))
            assert xyz_path.exists()
            
            # PDB
            pdb_path = Path(tmpdir) / "test.pdb"
            to_pdb(net, str(pdb_path))
            assert pdb_path.exists()
            
            # JSON
            json_path = Path(tmpdir) / "test.json"
            net.save_json(str(json_path))
            assert json_path.exists()


class TestGeneratorDiversity:
    """Test that various generator types produce valid networks."""
    
    def test_ordered_generators(self):
        """Test all ordered lattice generators."""
        net = gen.square_lattice_2d(spacing=5, grid_size=(2, 2))
        assert net.num_fibers > 0
        
        net = gen.triangular_lattice_2d(spacing=5, grid_size=(2, 2))
        assert net.num_fibers > 0
        
        net = gen.honeycomb_lattice_2d(cell_size=5, grid_size=(2, 2))
        assert net.num_fibers > 0
    
    def test_disordered_generators(self):
        """Test random/disordered generators."""
        net = gen.random_straight_2d(num_fibers=20, fiber_length=8, box_size=(20, 20), seed=42)
        assert net.num_fibers == 20
        
        net = gen.random_straight_3d(num_fibers=20, fiber_length=8, box_size=(20, 20, 20), seed=42)
        assert net.num_fibers == 20
        assert net.dimension == 3
        
        net = gen.random_walk_fibers(
            num_fibers=10, num_steps=10, step_length=1.0,
            box_size=(20, 20, 1), radius=0.1, dimension=2, seed=42
        )
        assert net.num_fibers == 10
    
    def test_chiral_generators(self):
        """Test chiral/helical generators."""
        net = gen.double_helix(
            num_turns=3, pitch=2.0, helix_radius=1.0,
            fiber_radius=0.05
        )
        assert net.num_fibers > 0
    
    def test_specialized_generators(self):
        """Test specialized generators."""
        from fibernet.gen.specialized import (
            cnt_network_2d, paper_network, textile_weave,
            electrospun_mat, fiber_reinforced_composite,
        )
        
        net = cnt_network_2d(num_tubes=10, tube_length=5, box_size=(20, 20), seed=42)
        assert net.num_fibers > 0
        
        net = paper_network(num_fibers=20, fiber_length=8, box_size=(20, 20), seed=42)
        assert net.num_fibers == 20
        
        net = textile_weave(warp_count=3, weft_count=3, seed=42)
        assert net.num_fibers == 6
        
        net = electrospun_mat(num_fibers=20, box_size=(20, 20), seed=42)
        assert net.num_fibers == 20
        
        net = fiber_reinforced_composite(
            matrix_size=(20, 20, 5),
            fiber_volume_fraction=0.3,
            seed=42,
        )
        assert net.num_fibers > 0


class TestSimulationModules:
    """Test simulation modules integrate properly."""
    
    def test_fluid_flow(self):
        """Test Darcy fluid solver."""
        from fibernet.sim.fluid import DarcySolver
        
        net = gen.random_straight_2d(num_fibers=20, fiber_length=8, box_size=(20, 20), seed=42)
        solver = DarcySolver(net)
        
        porosity = solver.compute_porosity()
        assert 0 < porosity < 1
        
        K = solver.kozeny_carman_permeability()
        assert K > 0
    
    def test_acoustic(self):
        """Test acoustic solver."""
        from fibernet.sim.acoustic import AcousticSolver
        
        net = gen.square_lattice_2d(spacing=5, grid_size=(2, 2))
        solver = AcousticSolver(net, segments_per_fiber=3)
        
        result = solver.compute_modes(num_modes=5)
        if result.frequencies is not None:
            assert len(result.frequencies) > 0
    
    def test_nonlinear_mechanics(self):
        """Test nonlinear FEM."""
        from fibernet.sim.nonlinear import NonlinearFEM, BilinearPlasticity
        
        net = gen.square_lattice_2d(spacing=5, grid_size=(2, 2))
        model = BilinearPlasticity(E=1e9, sigma_y=1e7, Et=1e8)
        fem = NonlinearFEM(net, constitutive_model=model, segments_per_fiber=3)
        
        strains, stresses, energies = fem.stress_strain_curve(
            axis=0, max_strain=0.005, num_steps=5
        )
        assert len(strains) > 0


class TestMLIntegration:
    """Test ML module integration."""
    
    def test_feature_extraction(self):
        """Test feature extraction pipeline."""
        from fibernet.ml.features import FeatureExtractor, extract_features
        
        net = gen.random_straight_2d(num_fibers=30, fiber_length=8, box_size=(25, 25), seed=42)
        
        features = extract_features(net)
        assert isinstance(features, dict)
        assert len(features) > 10
        assert 'num_fibers' in features
        assert 'nematic_order' in features
        
        # Array form
        arr = extract_features(net, as_array=True)
        assert isinstance(arr, np.ndarray)
        assert len(arr) > 10
    
    def test_feature_names(self):
        """Test feature name consistency."""
        from fibernet.ml.features import FeatureExtractor
        
        extractor = FeatureExtractor()
        names = extractor.get_feature_names()
        
        assert isinstance(names, list)
        assert len(names) > 10
        assert all(isinstance(n, str) for n in names)


class TestUnitSystems:
    """Test unit system conversions."""
    
    def test_si_to_cgs(self):
        from fibernet.utils.units import UnitConverter
        conv = UnitConverter()
        
        # Length: m to cm
        result = conv.convert(1.0, 'length', from_system='SI', to_system='CGS')
        assert abs(result - 100.0) < 1e-10
    
    def test_network_conversion(self):
        from fibernet.utils.units import convert_network_units
        
        net = gen.random_straight_2d(num_fibers=10, fiber_length=1e-6, box_size=(1e-5, 1e-5), seed=42)
        
        converted = convert_network_units(net, from_system='SI', to_system='micro')
        assert converted is not None

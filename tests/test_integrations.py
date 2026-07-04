"""
Tests for integrations module.
"""

import pytest
import numpy as np
from fibernet import gen


class TestNetworkXIntegration:
    """Test NetworkX integration."""
    
    def test_import(self):
        from fibernet.integrations.networkx_integration import (
            NetworkXBridge, GraphAnalysisResult,
            analyze_network_topology
        )
    
    def test_to_networkx(self):
        from fibernet.integrations.networkx_integration import NetworkXBridge
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        bridge = NetworkXBridge(net)
        G = bridge.to_networkx(node_type='crosslink')
        assert G is not None
        assert G.number_of_nodes() >= 0
    
    def test_fiber_graph(self):
        from fibernet.integrations.networkx_integration import NetworkXBridge
        net = gen.random_straight_2d(num_fibers=20, seed=42)
        bridge = NetworkXBridge(net)
        G = bridge.to_networkx(node_type='fiber')
        assert G.number_of_nodes() == net.num_fibers
    
    def test_analyze(self):
        from fibernet.integrations.networkx_integration import NetworkXBridge
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        bridge = NetworkXBridge(net)
        result = bridge.analyze()
        assert result.num_nodes >= 0
        assert result.num_edges >= 0
    
    def test_centrality(self):
        from fibernet.integrations.networkx_integration import NetworkXBridge
        net = gen.random_straight_2d(num_fibers=20, seed=42)
        bridge = NetworkXBridge(net)
        centrality = bridge.centrality_analysis(method='degree')
        assert isinstance(centrality, dict)
    
    def test_community_detection(self):
        from fibernet.integrations.networkx_integration import NetworkXBridge
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        bridge = NetworkXBridge(net)
        communities = bridge.detect_communities(algorithm='greedy')
        assert isinstance(communities, list)
    
    def test_convenience_function(self):
        from fibernet.integrations.networkx_integration import analyze_network_topology
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        result = analyze_network_topology(net)
        assert result.num_nodes >= 0


class TestMDAnalysisIntegration:
    """Test MDAnalysis integration."""
    
    def test_import(self):
        from fibernet.integrations.mdanalysis_integration import (
            MDAnalysisBridge, MDAnalysisResult
        )
    
    def test_result_dataclass(self):
        from fibernet.integrations.mdanalysis_integration import MDAnalysisResult
        result = MDAnalysisResult(rmsd=1.5, radius_of_gyration=2.0)
        assert result.rmsd == 1.5
        data = result.to_dict()
        assert 'rmsd' in data


class TestLAMMPSIntegration:
    """Test LAMMPS integration."""
    
    def test_import(self):
        from fibernet.integrations.lammps_integration import (
            LAMMPSBridge, LAMMPSResult
        )
    
    def test_result_dataclass(self):
        from fibernet.integrations.lammps_integration import LAMMPSResult
        result = LAMMPSResult(simulation_time=10.0)
        assert result.simulation_time == 10.0
        data = result.to_dict()
        assert 'simulation_time' in data
    
    def test_write_data_file(self):
        from fibernet.integrations.lammps_integration import LAMMPSBridge
        import tempfile, os
        net = gen.random_straight_2d(num_fibers=10, seed=42)
        bridge = LAMMPSBridge(net)
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "test.data")
            bridge.write_data_file(data_file)
            assert os.path.exists(data_file)
            # Check file has content
            with open(data_file) as f:
                content = f.read()
            assert "atoms" in content
            assert "bonds" in content
    
    def test_write_input_script(self):
        from fibernet.integrations.lammps_integration import LAMMPSBridge
        import tempfile, os
        net = gen.random_straight_2d(num_fibers=10, seed=42)
        bridge = LAMMPSBridge(net)
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "test.data")
            script_file = os.path.join(tmpdir, "in.test")
            bridge.write_data_file(data_file)
            bridge.write_input_script(
                script_file, data_file,
                temperature=300.0, nsteps=100
            )
            assert os.path.exists(script_file)
            with open(script_file) as f:
                content = f.read()
            assert "run" in content


class TestOVITOIntegration:
    """Test OVITO integration."""
    
    def test_import(self):
        from fibernet.integrations.ovito_integration import (
            OVITOBridge, OVITOAnalysisResult
        )
    
    def test_result_dataclass(self):
        from fibernet.integrations.ovito_integration import OVITOAnalysisResult
        result = OVITOAnalysisResult()
        data = result.to_dict()
        assert 'mean_coordination' in data
    
    def test_to_data_collection(self):
        from fibernet.integrations.ovito_integration import OVITOBridge
        net = gen.random_straight_2d(num_fibers=20, seed=42)
        bridge = OVITOBridge(net)
        data = bridge.to_data_collection()
        assert data is not None
        # Should be dict if OVITO not available
        if isinstance(data, dict):
            assert 'positions' in data
    
    def test_write_xyz(self):
        from fibernet.integrations.ovito_integration import OVITOBridge
        import tempfile, os
        net = gen.random_straight_2d(num_fibers=10, seed=42)
        bridge = OVITOBridge(net)
        with tempfile.TemporaryDirectory() as tmpdir:
            xyz_file = os.path.join(tmpdir, "test.xyz")
            bridge.write_xyz(xyz_file)
            assert os.path.exists(xyz_file)
    
    def test_coordination_analysis(self):
        from fibernet.integrations.ovito_integration import OVITOBridge
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        bridge = OVITOBridge(net)
        coord = bridge.coordination_analysis()
        assert len(coord) == net.num_fibers

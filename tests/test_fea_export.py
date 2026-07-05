"""Tests for FEA export functionality."""

import pytest
import tempfile
from pathlib import Path
from fibernet import gen
from fibernet.io.fea_export import (
    export_to_abaqus,
    export_to_ansys,
    export_to_gmsh,
    export_to_lammps,
)


@pytest.fixture
def sample_network():
    """Create a sample network for testing."""
    return gen.random_straight_3d(
        num_fibers=10,
        box_size=(20, 20, 20),
        fiber_length=15.0,
        seed=42
    )


class TestAbaqusExport:
    """Test Abaqus export functionality."""
    
    def test_export_basic(self, sample_network):
        """Test basic Abaqus export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = Path(tmpdir) / "test.inp"
            export_to_abaqus(sample_network, filename)
            
            assert filename.exists()
            content = filename.read_text()
            
            # Check for key Abaqus sections
            assert "*HEADING" in content
            assert "*PART" in content
            assert "*NODE" in content
            assert "*ELEMENT" in content
            assert "*MATERIAL" in content
            assert "*STEP" in content
    
    def test_export_with_parameters(self, sample_network):
        """Test Abaqus export with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = Path(tmpdir) / "test.inp"
            export_to_abaqus(
                sample_network,
                filename,
                job_name="CustomJob",
                segments_per_fiber=10,
                beam_type="B32"
            )
            
            assert filename.exists()
            content = filename.read_text()
            assert "CustomJob" in content
            assert "B32" in content
    
    def test_export_node_count(self, sample_network):
        """Test that correct number of nodes are exported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = Path(tmpdir) / "test.inp"
            segments = 5
            export_to_abaqus(sample_network, filename, segments_per_fiber=segments)
            
            content = filename.read_text()
            # Count nodes (each fiber has segments+1 nodes)
            expected_nodes = len(sample_network.fibers) * (segments + 1)
            # Extract node block between *NODE and *ELEMENT keywords
            lines = content.split('\n')
            in_node_block = False
            node_lines = []
            for line in lines:
                if line.strip().startswith('*NODE'):
                    in_node_block = True
                    continue
                if in_node_block and line.strip().startswith('*'):
                    break
                if in_node_block and line.strip() and line[0].isdigit():
                    node_lines.append(line)
            assert len(node_lines) == expected_nodes


class TestAnsysExport:
    """Test ANSYS export functionality."""
    
    def test_export_basic(self, sample_network):
        """Test basic ANSYS export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = Path(tmpdir) / "test.dat"
            export_to_ansys(sample_network, filename)
            
            assert filename.exists()
            content = filename.read_text()
            
            # Check for key ANSYS commands
            assert "/PREP7" in content
            assert "ET," in content
            assert "MP," in content
            assert "N," in content  # Nodes
            assert "E," in content  # Elements
            assert "/SOLU" in content
    
    def test_export_with_parameters(self, sample_network):
        """Test ANSYS export with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = Path(tmpdir) / "test.dat"
            export_to_ansys(
                sample_network,
                filename,
                segments_per_fiber=8,
                element_type="BEAM189"
            )
            
            assert filename.exists()
            content = filename.read_text()
            assert "BEAM189" in content


class TestGmshExport:
    """Test Gmsh export functionality."""
    
    def test_export_basic(self, sample_network):
        """Test basic Gmsh export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = Path(tmpdir) / "test.msh"
            export_to_gmsh(sample_network, filename)
            
            assert filename.exists()
            content = filename.read_text()
            
            # Check for Gmsh format markers
            assert "$MeshFormat" in content
            assert "$Nodes" in content
            assert "$Elements" in content
            assert "$EndNodes" in content
            assert "$EndElements" in content
    
    def test_export_mesh_version(self, sample_network):
        """Test that correct mesh format version is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = Path(tmpdir) / "test.msh"
            export_to_gmsh(sample_network, filename)
            
            content = filename.read_text()
            assert "4.1 0 8" in content  # Gmsh 4.1 format


class TestLammpsExport:
    """Test LAMMPS export functionality."""
    
    def test_export_basic(self, sample_network):
        """Test basic LAMMPS export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = Path(tmpdir) / "test.lmp"
            export_to_lammps(sample_network, filename)
            
            assert filename.exists()
            content = filename.read_text()
            
            # Check for LAMMPS sections
            assert "atoms" in content
            assert "bonds" in content
            assert "Masses" in content
            assert "Atoms" in content
            assert "Bonds" in content
    
    def test_export_box_bounds(self, sample_network):
        """Test that box bounds are correctly calculated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = Path(tmpdir) / "test.lmp"
            export_to_lammps(sample_network, filename)
            
            content = filename.read_text()
            assert "xlo xhi" in content
            assert "ylo yhi" in content
            assert "zlo zhi" in content


class TestExportIntegration:
    """Integration tests for FEA export."""
    
    def test_export_all_formats(self, sample_network):
        """Test exporting to all formats from same network."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Export to all formats
            export_to_abaqus(sample_network, tmpdir / "test.inp")
            export_to_ansys(sample_network, tmpdir / "test.dat")
            export_to_gmsh(sample_network, tmpdir / "test.msh")
            export_to_lammps(sample_network, tmpdir / "test.lmp")
            
            # Verify all files exist
            assert (tmpdir / "test.inp").exists()
            assert (tmpdir / "test.dat").exists()
            assert (tmpdir / "test.msh").exists()
            assert (tmpdir / "test.lmp").exists()
    
    def test_export_empty_network(self):
        """Test exporting an empty network."""
        from fibernet.core.network import FiberNetwork
        empty_net = FiberNetwork(dimension=3)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Should not raise errors
            export_to_abaqus(empty_net, tmpdir / "empty.inp")
            export_to_ansys(empty_net, tmpdir / "empty.dat")
            export_to_gmsh(empty_net, tmpdir / "empty.msh")
            export_to_lammps(empty_net, tmpdir / "empty.lmp")
            
            # All files should be created
            assert (tmpdir / "empty.inp").exists()
            assert (tmpdir / "empty.dat").exists()
            assert (tmpdir / "empty.msh").exists()
            assert (tmpdir / "empty.lmp").exists()



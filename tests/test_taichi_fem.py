"""Tests for Taichi-accelerated FEM solver."""

import pytest
import numpy as np
from fibernet.sim import TaichiFEMSolver


class TestTaichiFEMSolver:
    """Test TaichiFEMSolver functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = TaichiFEMSolver(arch="cpu", num_threads=2)
    
    def test_solver_initialization(self):
        """Test solver initialization."""
        solver = TaichiFEMSolver(arch="cpu", num_threads=2)
        assert solver is not None
    
    def test_solve_beam_network_simple(self):
        """Test solving a simple beam network."""
        # Simple 2-element beam
        node_positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
        ])
        
        elements = np.array([
            [0, 1],
            [1, 2],
        ])
        
        radii = np.array([0.05, 0.05])
        youngs_modulus = 1e9
        fixed_nodes = [0]
        
        applied_forces = np.zeros((3, 3))
        applied_forces[2, 0] = 1e3  # Pull in x
        
        result = self.solver.solve_beam_network(
            node_positions=node_positions,
            elements=elements,
            youngs_modulus=youngs_modulus,
            radii=radii,
            fixed_nodes=fixed_nodes,
            applied_forces=applied_forces,
        )
        
        assert result.displacements is not None
        assert result.displacements.shape == (3, 3)
        # Fixed node should have zero displacement
        assert np.allclose(result.displacements[0], [0, 0, 0])
        # Other nodes should have some displacement
        assert not np.allclose(result.displacements[1], [0, 0, 0])
        assert not np.allclose(result.displacements[2], [0, 0, 0])
    
    def test_solve_beam_network_with_forces(self):
        """Test that forces are computed."""
        node_positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
        ])
        
        elements = np.array([[0, 1]])
        radii = np.array([0.05])
        youngs_modulus = 1e9
        fixed_nodes = [0]
        
        applied_forces = np.zeros((2, 3))
        applied_forces[1, 0] = 1e3
        
        result = self.solver.solve_beam_network(
            node_positions=node_positions,
            elements=elements,
            youngs_modulus=youngs_modulus,
            radii=radii,
            fixed_nodes=fixed_nodes,
            applied_forces=applied_forces,
        )
        
        assert result.forces is not None
        assert len(result.forces) == 1
    
    def test_contact_detection_basic(self):
        """Test basic contact detection."""
        fiber_positions = np.array([
            [0, 0, 0],
            [0.1, 0, 0],  # Overlapping
        ])
        
        fiber_directions = np.array([
            [1, 0, 0],
            [1, 0, 0],
        ])
        
        fiber_lengths = np.array([1.0, 1.0])
        radii = np.array([0.1, 0.1])
        
        contacts = self.solver.parallel_contact_detection(
            fiber_positions=fiber_positions,
            fiber_directions=fiber_directions,
            fiber_lengths=fiber_lengths,
            radii=radii,
            box_size=(10, 10, 10),
        )
        
        assert len(contacts) >= 1
        # Check overlap is positive
        i, j, overlap = contacts[0]
        assert overlap > 0
    
    def test_contact_detection_no_overlap(self):
        """Test contact detection with no overlaps."""
        fiber_positions = np.array([
            [0, 0, 0],
            [5, 5, 5],  # Far away
        ])
        
        fiber_directions = np.array([
            [1, 0, 0],
            [0, 1, 0],
        ])
        
        fiber_lengths = np.array([1.0, 1.0])
        radii = np.array([0.1, 0.1])
        
        contacts = self.solver.parallel_contact_detection(
            fiber_positions=fiber_positions,
            fiber_directions=fiber_directions,
            fiber_lengths=fiber_lengths,
            radii=radii,
            box_size=(10, 10, 10),
        )
        
        assert len(contacts) == 0
    
    def test_contact_detection_multiple(self):
        """Test contact detection with multiple contacts."""
        fiber_positions = np.array([
            [0, 0, 0],
            [0.1, 0, 0],  # Overlaps with 0
            [0.2, 0, 0],  # Overlaps with 1
            [10, 10, 10],  # Far away
        ])
        
        fiber_directions = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ])
        
        fiber_lengths = np.array([1.0, 1.0, 1.0, 1.0])
        radii = np.array([0.1, 0.1, 0.1, 0.1])
        
        contacts = self.solver.parallel_contact_detection(
            fiber_positions=fiber_positions,
            fiber_directions=fiber_directions,
            fiber_lengths=fiber_lengths,
            radii=radii,
            box_size=(20, 20, 20),
        )
        
        # Should detect at least 2 contacts (0-1 and 1-2)
        assert len(contacts) >= 2
    
    def test_segment_distance(self):
        """Test segment distance calculation."""
        # Parallel segments
        p1 = np.array([0, 0, 0])
        d1 = np.array([1, 0, 0])
        L1 = 2.0
        
        p2 = np.array([0, 1, 0])
        d2 = np.array([1, 0, 0])
        L2 = 2.0
        
        dist = self.solver._segment_distance(p1, d1, L1, p2, d2, L2)
        assert np.isclose(dist, 1.0)
        
        # Perpendicular segments
        p2 = np.array([0, 0, 0])
        d2 = np.array([0, 1, 0])
        L2 = 2.0
        
        dist = self.solver._segment_distance(p1, d1, L1, p2, d2, L2)
        assert np.isclose(dist, 0.0)
    
    def test_progressive_damage_basic(self):
        """Test basic progressive damage simulation."""
        node_positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
        ])
        
        elements = np.array([
            [0, 1],
            [1, 2],
        ])
        
        radii = np.array([0.05, 0.05])
        youngs_modulus = 1e9
        fixed_nodes = [0]
        
        # Low strength to trigger damage
        strength = np.array([1e6, 1e6])
        
        result = self.solver.progressive_damage(
            node_positions=node_positions,
            elements=elements,
            youngs_modulus=youngs_modulus,
            radii=radii,
            fixed_nodes=fixed_nodes,
            strain_range=(0, 0.1),
            num_steps=5,
            strength=strength,
            axis=0,
        )
        
        assert 'strain' in result
        assert 'stress' in result
        assert 'damage' in result
        assert 'broken_elements' in result
        assert len(result['strain']) > 0
        assert len(result['damage']) > 0


class TestTaichiFEMSolverIntegration:
    """Integration tests for TaichiFEMSolver."""
    
    def test_solve_larger_network(self):
        """Test solving a larger network."""
        # Create a 3x3 grid network
        node_positions = []
        for i in range(3):
            for j in range(3):
                node_positions.append([i, j, 0])
        node_positions = np.array(node_positions)
        
        # Create elements (horizontal and vertical connections)
        elements = []
        for i in range(3):
            for j in range(3):
                node_idx = i * 3 + j
                if j < 2:  # Horizontal
                    elements.append([node_idx, node_idx + 1])
                if i < 2:  # Vertical
                    elements.append([node_idx, node_idx + 3])
        elements = np.array(elements)
        
        radii = np.ones(len(elements)) * 0.05
        youngs_modulus = 1e9
        fixed_nodes = [0, 1, 2]  # Fix left edge
        
        applied_forces = np.zeros((9, 3))
        applied_forces[6, 0] = 1e3  # Pull right edge
        applied_forces[7, 0] = 1e3
        applied_forces[8, 0] = 1e3
        
        solver = TaichiFEMSolver(arch="cpu", num_threads=2)
        result = solver.solve_beam_network(
            node_positions=node_positions,
            elements=elements,
            youngs_modulus=youngs_modulus,
            radii=radii,
            fixed_nodes=fixed_nodes,
            applied_forces=applied_forces,
        )
        
        assert result.displacements is not None
        assert result.displacements.shape == (9, 3)

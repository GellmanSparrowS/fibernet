"""Tests for composite laminate generators."""

import pytest
import numpy as np
from fibernet.gen.laminates import (
    unidirectional_laminate,
    crossply_laminate,
    angle_ply_laminate,
    quasi_isotropic_laminate,
    custom_laminate,
    sandwich_laminate,
)


class TestUnidirectionalLaminate:
    """Test unidirectional laminate generator."""
    
    def test_basic_generation(self):
        """Test basic UD laminate."""
        net = unidirectional_laminate(
            num_layers=4,
            fibers_per_layer=10,
            seed=42
        )
        
        assert len(net.fibers) == 40  # 4 layers * 10 fibers
        assert net.dimension == 3
    
    def test_fiber_count(self):
        """Test fiber count."""
        net = unidirectional_laminate(
            num_layers=3,
            fibers_per_layer=15,
            seed=42
        )
        
        assert len(net.fibers) == 45
    
    def test_orientation(self):
        """Test fiber orientation."""
        net = unidirectional_laminate(
            num_layers=2,
            fibers_per_layer=10,
            orientation=np.pi / 4,  # 45 degrees
            seed=42
        )
        
        # Check first fiber direction
        fiber = net.fibers[0]
        direction = fiber.end_point - fiber.start_point
        direction = direction / np.linalg.norm(direction)
        
        # Should be approximately at 45 degrees
        angle = np.arctan2(direction[1], direction[0])
        assert abs(angle - np.pi / 4) < 0.1
    
    def test_reproducibility(self):
        """Test seed reproducibility."""
        net1 = unidirectional_laminate(num_layers=2, fibers_per_layer=10, seed=42)
        net2 = unidirectional_laminate(num_layers=2, fibers_per_layer=10, seed=42)
        
        for f1, f2 in zip(net1.fibers, net2.fibers):
            assert np.allclose(f1.start_point, f2.start_point)
            assert np.allclose(f1.end_point, f2.end_point)


class TestCrossplyLaminate:
    """Test cross-ply laminate generator."""
    
    def test_basic_generation(self):
        """Test basic cross-ply laminate."""
        net = crossply_laminate(
            num_layers=4,
            fibers_per_layer=10,
            seed=42
        )
        
        assert len(net.fibers) == 40
        assert net.dimension == 3
    
    def test_alternating_orientations(self):
        """Test alternating 0° and 90° orientations."""
        net = crossply_laminate(
            num_layers=4,
            fibers_per_layer=10,
            seed=42
        )
        
        # Check first fiber of each layer
        for layer in range(4):
            fiber = net.fibers[layer * 10]
            direction = fiber.end_point - fiber.start_point
            direction = direction / np.linalg.norm(direction)
            
            angle = np.arctan2(direction[1], direction[0])
            
            # Even layers: 0°, odd layers: 90°
            if layer % 2 == 0:
                assert abs(angle) < 0.1 or abs(angle - np.pi) < 0.1
            else:
                assert abs(abs(angle) - np.pi / 2) < 0.1


class TestAnglePlyLaminate:
    """Test angle-ply laminate generator."""
    
    def test_basic_generation(self):
        """Test basic angle-ply laminate."""
        net = angle_ply_laminate(
            num_layers=4,
            angle=np.pi / 4,  # ±45°
            fibers_per_layer=10,
            seed=42
        )
        
        assert len(net.fibers) == 40
        assert net.dimension == 3
    
    def test_angle_value(self):
        """Test specific angle."""
        net = angle_ply_laminate(
            num_layers=2,
            angle=np.pi / 6,  # ±30°
            fibers_per_layer=10,
            seed=42
        )
        
        # Check first fiber (should be +30°)
        fiber = net.fibers[0]
        direction = fiber.end_point - fiber.start_point
        direction = direction / np.linalg.norm(direction)
        
        angle = np.arctan2(direction[1], direction[0])
        assert abs(angle - np.pi / 6) < 0.1


class TestQuasiIsotropicLaminate:
    """Test quasi-isotropic laminate generator."""
    
    def test_basic_generation(self):
        """Test basic quasi-isotropic laminate."""
        net = quasi_isotropic_laminate(
            num_fibers_per_layer=10,
            seed=42
        )
        
        # Should have 4 layers: [0/+45/-45/90]
        assert len(net.fibers) == 40
        assert net.dimension == 3
    
    def test_four_layers(self):
        """Test that quasi-isotropic has 4 layers."""
        net = quasi_isotropic_laminate(
            num_fibers_per_layer=15,
            seed=42
        )
        
        assert len(net.fibers) == 60  # 4 layers * 15 fibers


class TestCustomLaminate:
    """Test custom laminate generator."""
    
    def test_basic_generation(self):
        """Test basic custom laminate."""
        angles = [0.0, np.pi / 4, np.pi / 2]
        
        net = custom_laminate(
            stacking_sequence=angles,
            fibers_per_layer=10,
            seed=42
        )
        
        assert len(net.fibers) == 30  # 3 layers * 10 fibers
        assert net.dimension == 3
    
    def test_arbitrary_angles(self):
        """Test arbitrary stacking angles."""
        angles = [0.0, np.pi / 6, np.pi / 3, np.pi / 2, -np.pi / 6]
        
        net = custom_laminate(
            stacking_sequence=angles,
            fibers_per_layer=10,
            seed=42
        )
        
        assert len(net.fibers) == 50  # 5 layers * 10 fibers
    
    def test_symmetric_laminate(self):
        """Test symmetric stacking sequence."""
        # Symmetric: [0/45/45/0]
        angles = [0.0, np.pi / 4, np.pi / 4, 0.0]
        
        net = custom_laminate(
            stacking_sequence=angles,
            fibers_per_layer=10,
            seed=42
        )
        
        assert len(net.fibers) == 40


class TestSandwichLaminate:
    """Test sandwich laminate generator."""
    
    def test_basic_generation(self):
        """Test basic sandwich structure."""
        net = sandwich_laminate(
            face_fibers_per_layer=10,
            num_face_layers=2,
            core_thickness=5.0,
            seed=42
        )
        
        # 2 face sheets * 2 layers * 10 fibers = 40 fibers
        assert len(net.fibers) == 40
        assert net.dimension == 3
    
    def test_face_sheet_structure(self):
        """Test that sandwich has two face sheets."""
        net = sandwich_laminate(
            face_fibers_per_layer=10,
            num_face_layers=2,
            seed=42
        )
        
        # Should have bottom and top face sheets
        assert len(net.fibers) == 40
    
    def test_core_thickness(self):
        """Test core thickness parameter."""
        net1 = sandwich_laminate(core_thickness=5.0, seed=42)
        net2 = sandwich_laminate(core_thickness=10.0, seed=42)
        
        # Both should generate valid networks
        assert len(net1.fibers) > 0
        assert len(net2.fibers) > 0


class TestLaminateIntegration:
    """Test laminate generators work together."""
    
    def test_all_laminate_types(self):
        """Test all laminate types can be generated."""
        laminates = [
            unidirectional_laminate(num_layers=2, fibers_per_layer=10, seed=42),
            crossply_laminate(num_layers=2, fibers_per_layer=10, seed=42),
            angle_ply_laminate(num_layers=2, fibers_per_layer=10, seed=42),
            quasi_isotropic_laminate(num_fibers_per_layer=10, seed=42),
            custom_laminate([0, np.pi/4], fibers_per_layer=10, seed=42),
            sandwich_laminate(face_fibers_per_layer=10, seed=42),
        ]
        
        # All should be valid 3D networks
        for net in laminates:
            assert len(net.fibers) > 0
            assert net.dimension == 3
    
    def test_with_materials(self):
        """Test laminates can use material properties."""
        from fibernet.core.material import Material
        
        mat = Material(
            name="carbon_fiber",
            youngs_modulus=230e9,
            density=1800.0,
            poissons_ratio=0.3
        )
        
        net = unidirectional_laminate(
            num_layers=2,
            fibers_per_layer=10,
            material=mat,
            seed=42
        )
        
        assert len(net.fibers) == 20



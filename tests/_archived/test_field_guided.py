"""Tests for field-guided network generator."""

import numpy as np
import pytest

from fibernet.gen.field_guided import (
    FieldGuidedConfig,
    OrientationField,
    field_guided_network,
    multi_scale_orientation_analysis,
)
from fibernet.core.material import Material


class TestOrientationField:
    def test_uniform_field(self):
        field = OrientationField(canvas_size=64, field_type="uniform", field_angle=np.pi/4)
        assert field.field.shape == (64, 64)
        assert np.allclose(field.field, np.pi/4)
    
    def test_radial_field(self):
        field = OrientationField(canvas_size=64, field_type="radial")
        assert field.field.shape == (64, 64)
    
    def test_vortex_field(self):
        field = OrientationField(canvas_size=64, field_type="vortex")
        assert field.field.shape == (64, 64)
    
    def test_gradient_field(self):
        field = OrientationField(canvas_size=64, field_type="gradient")
        assert field.field.shape == (64, 64)
    
    def test_random_smooth_field(self):
        field = OrientationField(canvas_size=64, field_type="random_smooth", smoothing_sigma=5.0)
        assert field.field.shape == (64, 64)
    
    def test_get_angle(self):
        field = OrientationField(canvas_size=64, field_type="uniform", field_angle=1.0)
        angle = field.get_angle(32, 32)
        assert np.isclose(angle, 1.0)
    
    def test_invalid_type(self):
        with pytest.raises(ValueError):
            OrientationField(canvas_size=64, field_type="invalid")


class TestFieldGuidedNetwork:
    def test_basic_generation(self):
        config = FieldGuidedConfig(fiber_count=50, canvas_size=128, seed=42)
        net = field_guided_network(config=config, box_size=(10, 10))
        assert net is not None
        assert len(net.fibers) > 0
    
    def test_with_field(self):
        field = OrientationField(canvas_size=128, field_type="radial")
        config = FieldGuidedConfig(fiber_count=30, canvas_size=128, seed=42)
        net = field_guided_network(config=config, field=field, box_size=(10, 10))
        assert net is not None
        assert len(net.fibers) > 0
    
    def test_metadata(self):
        config = FieldGuidedConfig(fiber_count=20, canvas_size=128, seed=42)
        net = field_guided_network(config=config, box_size=(10, 10))
        assert net.metadata.get('generator') == 'field_guided'
        assert 'field_type' in net.metadata


class TestMultiScaleAnalysis:
    def test_basic_analysis(self):
        config = FieldGuidedConfig(fiber_count=50, canvas_size=128, seed=42)
        net = field_guided_network(config=config, box_size=(10, 10))
        
        result = multi_scale_orientation_analysis(net)
        assert 'orientations' in result
        assert 'nematic_order' in result
        assert 'dominant_angle' in result
        assert result['nematic_order'] >= 0
    
    def test_empty_network(self):
        from fibernet.core.network import FiberNetwork
        net = FiberNetwork(fibers=[], box_size=np.array([10, 10, 1]))
        result = multi_scale_orientation_analysis(net)
        assert result['nematic_order'] == 0.0

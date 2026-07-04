"""Tests for stress-strain curve extraction."""

import pytest
import numpy as np
from fibernet import gen
from fibernet.analysis import extract_stress_strain, StressStrainCurve, compare_curves


class TestStressStrainCurve:
    def test_extract_basic(self):
        """Test basic stress-strain extraction."""
        net = gen.square_lattice_2d(spacing=10, grid_size=(3, 3))
        curve = extract_stress_strain(
            net,
            strain_range=(0.0, 0.02),
            num_steps=5,
            axis=0
        )
        
        assert isinstance(curve, StressStrainCurve)
        assert len(curve.strain) == 5
        assert len(curve.stress) == 5
        assert len(curve.energy) == 5
    
    def test_youngs_modulus(self):
        """Test Young's modulus calculation."""
        net = gen.square_lattice_2d(spacing=10, grid_size=(3, 3))
        curve = extract_stress_strain(
            net,
            strain_range=(0.0, 0.02),
            num_steps=5,
            axis=0
        )
        
        E = curve.youngs_modulus
        assert isinstance(E, float)
        assert E >= 0
    
    def test_ultimate_strength(self):
        """Test ultimate strength calculation."""
        net = gen.square_lattice_2d(spacing=10, grid_size=(3, 3))
        curve = extract_stress_strain(
            net,
            strain_range=(0.0, 0.02),
            num_steps=5,
            axis=0
        )
        
        sigma_u = curve.ultimate_strength
        assert isinstance(sigma_u, float)
        assert sigma_u >= 0
    
    def test_toughness(self):
        """Test toughness calculation."""
        net = gen.square_lattice_2d(spacing=10, grid_size=(3, 3))
        curve = extract_stress_strain(
            net,
            strain_range=(0.0, 0.02),
            num_steps=5,
            axis=0
        )
        
        toughness = curve.toughness
        assert isinstance(toughness, float)
        assert toughness >= 0
    
    def test_to_dataframe(self):
        """Test conversion to pandas DataFrame."""
        net = gen.square_lattice_2d(spacing=10, grid_size=(3, 3))
        curve = extract_stress_strain(
            net,
            strain_range=(0.0, 0.02),
            num_steps=5,
            axis=0
        )
        
        df = curve.to_dataframe()
        assert 'strain' in df.columns
        assert 'stress' in df.columns
        assert 'energy' in df.columns
        assert len(df) == 5
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        net = gen.square_lattice_2d(spacing=10, grid_size=(3, 3))
        curve = extract_stress_strain(
            net,
            strain_range=(0.0, 0.02),
            num_steps=5,
            axis=0
        )
        
        d = curve.to_dict()
        assert 'strain' in d
        assert 'stress' in d
        assert 'youngs_modulus' in d
        assert 'ultimate_strength' in d
    
    def test_metadata(self):
        """Test metadata is stored correctly."""
        net = gen.square_lattice_2d(spacing=10, grid_size=(3, 3))
        curve = extract_stress_strain(
            net,
            strain_range=(0.0, 0.02),
            num_steps=5,
            axis=0,
            segments_per_fiber=5
        )
        
        assert 'axis' in curve.metadata
        assert curve.metadata['axis'] == 0
        assert 'segments_per_fiber' in curve.metadata
        assert curve.metadata['segments_per_fiber'] == 5
    
    def test_compare_curves(self):
        """Test comparing multiple curves."""
        net = gen.square_lattice_2d(spacing=10, grid_size=(3, 3))
        
        curve1 = extract_stress_strain(net, strain_range=(0.0, 0.01), num_steps=3, axis=0)
        curve2 = extract_stress_strain(net, strain_range=(0.0, 0.02), num_steps=3, axis=1)
        
        # This should not raise an error
        import matplotlib
        matplotlib.use('Agg')
        ax = compare_curves([curve1, curve2], ['Axis 0', 'Axis 1'])
        assert ax is not None

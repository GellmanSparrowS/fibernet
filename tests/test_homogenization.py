"""Tests for homogenization and effective property computation."""

import pytest
import numpy as np
from fibernet import gen
from fibernet.analysis.homogenization import (
    EffectiveElasticProperties, EffectiveThermalProperties,
    EffectiveElectricalProperties, compute_effective_properties
)


class TestEffectiveElasticProperties:
    """Test effective elastic property computation."""
    
    def test_effective_modulus_2d_x(self):
        """Test effective modulus in x-direction."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        elastic = EffectiveElasticProperties(net)
        E = elastic.effective_modulus_2d(direction='x')
        assert E > 0
    
    def test_effective_modulus_2d_y(self):
        """Test effective modulus in y-direction."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        elastic = EffectiveElasticProperties(net)
        E = elastic.effective_modulus_2d(direction='y')
        assert E > 0
    
    def test_effective_poisson_ratio(self):
        """Test effective Poisson's ratio."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        elastic = EffectiveElasticProperties(net)
        nu = elastic.effective_poisson_ratio()
        assert 0.0 <= nu <= 0.5
    
    def test_effective_shear_modulus(self):
        """Test effective shear modulus."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        elastic = EffectiveElasticProperties(net)
        G = elastic.effective_shear_modulus()
        assert G > 0
    
    def test_compute_all_2d(self):
        """Test computing all elastic properties for 2D."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        elastic = EffectiveElasticProperties(net)
        props = elastic.compute_all()
        assert 'E_x' in props
        assert 'E_y' in props
        assert 'nu' in props
        assert 'G' in props
        assert props['E_x'] > 0
        assert props['E_y'] > 0
    
    def test_compute_all_3d(self):
        """Test computing all elastic properties for 3D."""
        net = gen.random_straight_3d(num_fibers=50, seed=42)
        elastic = EffectiveElasticProperties(net)
        props = elastic.compute_all()
        assert 'E' in props
        assert 'nu' in props
        assert 'G' in props
    
    def test_empty_network(self):
        """Test with empty network."""
        from fibernet.core.network import FiberNetwork
        net = FiberNetwork(dimension=2)
        elastic = EffectiveElasticProperties(net)
        E = elastic.effective_modulus_2d()
        assert E == 0.0


class TestEffectiveThermalProperties:
    """Test effective thermal property computation."""
    
    def test_thermal_conductivity(self):
        """Test effective thermal conductivity."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        thermal = EffectiveThermalProperties(net)
        k = thermal.effective_thermal_conductivity()
        assert k >= 0
    
    def test_thermal_expansion(self):
        """Test effective thermal expansion coefficient."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        thermal = EffectiveThermalProperties(net)
        alpha = thermal.effective_thermal_expansion()
        assert isinstance(alpha, float)


class TestEffectiveElectricalProperties:
    """Test effective electrical property computation."""
    
    def test_electrical_conductivity(self):
        """Test effective electrical conductivity."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        electrical = EffectiveElectricalProperties(net)
        sigma = electrical.effective_electrical_conductivity()
        assert sigma >= 0


class TestComputeEffectiveProperties:
    """Test comprehensive property computation."""
    
    def test_compute_all_2d(self):
        """Test computing all properties for 2D network."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        props = compute_effective_properties(net)
        assert 'elastic' in props
        assert 'thermal' in props
        assert 'electrical' in props
        assert 'n_fibers' in props
        assert 'total_length' in props
    
    def test_compute_all_3d(self):
        """Test computing all properties for 3D network."""
        net = gen.random_straight_3d(num_fibers=50, seed=42)
        props = compute_effective_properties(net)
        assert 'elastic' in props
        assert 'thermal' in props
        assert 'electrical' in props
    
    def test_different_networks(self):
        """Test with different network types."""
        networks = [
            gen.random_straight_2d(num_fibers=50, seed=42),
            gen.square_lattice_2d(spacing=2.0, grid_size=(5, 5)),
            gen.honeycomb_lattice_2d(cell_size=2.0, grid_size=(5, 5)),
        ]
        
        for net in networks:
            props = compute_effective_properties(net)
            assert 'elastic' in props
            assert props['n_fibers'] > 0

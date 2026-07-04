"""Tests for unit system module."""

import numpy as np
import pytest
from fibernet.utils.units import (
    SI, CGS, MICRO, NANO, MOLECULAR,
    UnitConverter, convert_network_units,
)
from fibernet.gen import square_lattice_2d


class TestUnitSystems:
    def test_si_system(self):
        assert SI.length == "m"
        assert SI.mass == "kg"
        assert SI.time == "s"
        assert SI.length_to_si == 1.0
    
    def test_cgs_system(self):
        assert CGS.length == "cm"
        assert CGS.length_to_si == pytest.approx(1e-2, rel=1e-10)
    
    def test_micro_system(self):
        assert MICRO.length == "µm"
        assert MICRO.length_to_si == pytest.approx(1e-6, rel=1e-10)
    
    def test_nano_system(self):
        assert NANO.length == "nm"
        assert NANO.length_to_si == pytest.approx(1e-9, rel=1e-10)
    
    def test_molecular_system(self):
        assert MOLECULAR.length == "Å"
        assert MOLECULAR.length_to_si == pytest.approx(1e-10, rel=1e-10)


class TestUnitConverter:
    def test_length_conversion(self):
        # 1 m = 100 cm
        result = UnitConverter.convert(1.0, 'length', 'SI', 'CGS')
        assert result == pytest.approx(100.0, rel=1e-6)
    
    def test_mass_conversion(self):
        # 1 kg = 1000 g
        result = UnitConverter.convert(1.0, 'mass', 'SI', 'CGS')
        assert result == pytest.approx(1000.0, rel=1e-6)
    
    def test_stress_conversion(self):
        # 1 Pa = 10 dyne/cm²
        result = UnitConverter.convert(1.0, 'stress', 'SI', 'CGS')
        assert result == pytest.approx(10.0, rel=1e-6)
    
    def test_force_conversion(self):
        # 1 N = 1e5 dyne
        result = UnitConverter.convert(1.0, 'force', 'SI', 'CGS')
        assert result == pytest.approx(1e5, rel=1e-6)
    
    def test_to_si(self):
        result = UnitConverter.to_si(100.0, 'length', 'CGS')
        assert result == pytest.approx(1.0, rel=1e-6)
    
    def test_from_si(self):
        result = UnitConverter.from_si(1.0, 'length', 'CGS')
        assert result == pytest.approx(100.0, rel=1e-6)


class TestNetworkConversion:
    def test_convert_to_cgs(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        
        # Assume original is in meters
        original_spacing = 5.0
        
        converted = convert_network_units(net, 'SI', 'CGS')
        
        # Spacing should be 500 cm
        bb_min, bb_max = converted.bounding_box()
        new_spacing = (bb_max[0] - bb_min[0]) / 2
        assert new_spacing > original_spacing  # Should be larger in CGS
    
    def test_convert_to_micro(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        
        converted = convert_network_units(net, 'SI', 'MICRO')
        
        bb_min, bb_max = converted.bounding_box()
        new_spacing = (bb_max[0] - bb_min[0]) / 2
        assert new_spacing > 5.0  # Should be larger in micrometers
    
    def test_preserve_structure(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        original_num_fibers = net.num_fibers
        
        converted = convert_network_units(net, 'SI', 'CGS')
        
        assert converted.num_fibers == original_num_fibers

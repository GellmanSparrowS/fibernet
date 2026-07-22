"""Tests for unit conversion utilities."""

import pytest
import numpy as np
from fibernet.units import (
    convert_length, convert_force, convert_pressure,
    convert_temperature, convert_energy, parse_unit_string,
    scale_network_properties
)


class TestLengthConversions:
    """Test length unit conversions."""
    
    def test_mm_to_um(self):
        """Test millimeters to micrometers."""
        result = convert_length(1.0, 'mm', 'um')
        assert np.isclose(result, 1000.0)
    
    def test_nm_to_m(self):
        """Test nanometers to meters."""
        result = convert_length(100, 'nm', 'm')
        assert np.isclose(result, 1e-7)
    
    def test_m_to_mm(self):
        """Test meters to millimeters."""
        result = convert_length(1.0, 'm', 'mm')
        assert np.isclose(result, 1000.0)
    
    def test_um_to_nm(self):
        """Test micrometers to nanometers."""
        result = convert_length(1.0, 'um', 'nm')
        assert np.isclose(result, 1000.0)
    
    def test_same_unit(self):
        """Test conversion to same unit."""
        result = convert_length(5.0, 'mm', 'mm')
        assert np.isclose(result, 5.0)
    
    def test_invalid_from_unit(self):
        """Test invalid source unit."""
        with pytest.raises(ValueError):
            convert_length(1.0, 'invalid', 'mm')
    
    def test_invalid_to_unit(self):
        """Test invalid target unit."""
        with pytest.raises(ValueError):
            convert_length(1.0, 'mm', 'invalid')


class TestPressureConversions:
    """Test pressure/stress unit conversions."""
    
    def test_mpa_to_gpa(self):
        """Test MPa to GPa."""
        result = convert_pressure(100.0, 'MPa', 'GPa')
        assert np.isclose(result, 0.1)
    
    def test_gpa_to_pa(self):
        """Test GPa to Pa."""
        result = convert_pressure(1.0, 'GPa', 'Pa')
        assert np.isclose(result, 1e9)
    
    def test_kpa_to_mpa(self):
        """Test kPa to MPa."""
        result = convert_pressure(1000.0, 'kPa', 'MPa')
        assert np.isclose(result, 1.0)
    
    def test_atm_to_pa(self):
        """Test atmospheres to Pascals."""
        result = convert_pressure(1.0, 'atm', 'Pa')
        assert np.isclose(result, 101325.0)


class TestForceConversions:
    """Test force unit conversions."""
    
    def test_kn_to_n(self):
        """Test kilonewtons to newtons."""
        result = convert_force(1.0, 'kN', 'N')
        assert np.isclose(result, 1000.0)
    
    def test_n_to_mn(self):
        """Test newtons to millinewtons."""
        result = convert_force(1.0, 'N', 'mN')
        assert np.isclose(result, 1000.0)


class TestTemperatureConversions:
    """Test temperature unit conversions."""
    
    def test_celsius_to_kelvin(self):
        """Test Celsius to Kelvin."""
        result = convert_temperature(0.0, 'C', 'K')
        assert np.isclose(result, 273.15)
    
    def test_kelvin_to_celsius(self):
        """Test Kelvin to Celsius."""
        result = convert_temperature(273.15, 'K', 'C')
        assert np.isclose(result, 0.0)
    
    def test_celsius_to_fahrenheit(self):
        """Test Celsius to Fahrenheit."""
        result = convert_temperature(0.0, 'C', 'F')
        assert np.isclose(result, 32.0)
    
    def test_fahrenheit_to_celsius(self):
        """Test Fahrenheit to Celsius."""
        result = convert_temperature(212.0, 'F', 'C')
        assert np.isclose(result, 100.0)


class TestEnergyConversions:
    """Test energy unit conversions."""
    
    def test_kj_to_j(self):
        """Test kilojoules to joules."""
        result = convert_energy(1.0, 'kJ', 'J')
        assert np.isclose(result, 1000.0)
    
    def test_cal_to_j(self):
        """Test calories to joules."""
        result = convert_energy(1.0, 'cal', 'J')
        assert np.isclose(result, 4.184)


class TestParseUnitString:
    """Test unit string parsing."""
    
    def test_parse_length_unit(self):
        """Test parsing length unit."""
        factor, utype = parse_unit_string('mm')
        assert np.isclose(factor, 1e-3)
        assert utype == 'length'
    
    def test_parse_pressure_unit(self):
        """Test parsing pressure unit."""
        factor, utype = parse_unit_string('MPa')
        assert np.isclose(factor, 1e6)
        assert utype == 'pressure'
    
    def test_parse_force_unit(self):
        """Test parsing force unit."""
        factor, utype = parse_unit_string('kN')
        assert np.isclose(factor, 1e3)
        assert utype == 'force'
    
    def test_parse_invalid_unit(self):
        """Test parsing invalid unit."""
        with pytest.raises(ValueError):
            parse_unit_string('invalid_unit')


class TestScaleNetworkProperties:
    """Test network property scaling."""
    
    def test_scale_to_mm(self):
        """Test scaling to millimeters."""
        factor = scale_network_properties(None, length_unit='mm')
        assert np.isclose(factor, 1e-3)
    
    def test_scale_to_um(self):
        """Test scaling to micrometers."""
        factor = scale_network_properties(None, length_unit='um')
        assert np.isclose(factor, 1e-6)
    
    def test_scale_invalid_unit(self):
        """Test scaling with invalid unit."""
        with pytest.raises(ValueError):
            scale_network_properties(None, length_unit='invalid')

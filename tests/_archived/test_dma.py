"""Tests for DMA simulations."""

import pytest
import numpy as np
from fibernet.sim.viscoelastic import GeneralizedMaxwell
from fibernet.sim.dma import frequency_sweep, temperature_sweep, master_curve, DMAResult


class TestFrequencySweep:
    def test_basic_sweep(self):
        """Test basic frequency sweep"""
        model = GeneralizedMaxwell(E_inf=1e9, E_i=[5e8, 3e8], tau_i=[0.1, 1.0])
        result = frequency_sweep(model, freq_range=(0.01, 100), num_points=20)
        
        assert isinstance(result, DMAResult)
        assert len(result.frequency) == 20
        assert len(result.storage_modulus) == 20
        assert len(result.loss_modulus) == 20
        assert len(result.loss_tangent) == 20
        assert len(result.complex_modulus) == 20
        assert len(result.phase_angle) == 20
    
    def test_storage_modulus_positive(self):
        """Test storage modulus is always positive"""
        model = GeneralizedMaxwell(E_inf=1e9, E_i=[5e8], tau_i=[0.1])
        result = frequency_sweep(model, freq_range=(0.01, 100), num_points=50)
        assert np.all(result.storage_modulus > 0)
    
    def test_loss_modulus_positive(self):
        """Test loss modulus is always positive"""
        model = GeneralizedMaxwell(E_inf=1e9, E_i=[5e8], tau_i=[0.1])
        result = frequency_sweep(model, freq_range=(0.01, 100), num_points=50)
        assert np.all(result.loss_modulus > 0)
    
    def test_tan_delta_positive(self):
        """Test tan delta is always positive"""
        model = GeneralizedMaxwell(E_inf=1e9, E_i=[5e8], tau_i=[0.1])
        result = frequency_sweep(model, freq_range=(0.01, 100), num_points=50)
        assert np.all(result.loss_tangent > 0)


class TestTemperatureSweep:
    def test_basic_sweep(self):
        """Test basic temperature sweep"""
        model = GeneralizedMaxwell(E_inf=1e9, E_i=[5e8, 3e8], tau_i=[0.1, 1.0])
        result = temperature_sweep(model, temp_range=(200, 400), num_points=20)
        
        assert isinstance(result, DMAResult)
        assert len(result.temperature) == 20
        assert len(result.storage_modulus) == 20
        assert len(result.loss_modulus) == 20
        assert len(result.loss_tangent) == 20
    
    def test_temperature_monotonic(self):
        """Test temperature is monotonically increasing"""
        model = GeneralizedMaxwell(E_inf=1e9, E_i=[5e8], tau_i=[0.1])
        result = temperature_sweep(model, temp_range=(200, 400), num_points=50)
        assert np.all(np.diff(result.temperature) > 0)


class TestMasterCurve:
    def test_basic_master_curve(self):
        """Test basic master curve construction"""
        model = GeneralizedMaxwell(E_inf=1e9, E_i=[5e8, 3e8], tau_i=[0.1, 1.0])
        curves = master_curve(
            model,
            reference_temp=298,
            temperatures=[280, 290, 300, 310],
            freq_range=(0.01, 100),
            num_points=20
        )
        
        assert isinstance(curves, dict)
        assert len(curves) == 4
        for T, curve in curves.items():
            assert isinstance(curve, DMAResult)
            assert len(curve.frequency) == 20
            assert len(curve.storage_modulus) == 20
    
    def test_shift_factor(self):
        """Test shift factors are reasonable"""
        model = GeneralizedMaxwell(E_inf=1e9, E_i=[5e8], tau_i=[0.1])
        curves = master_curve(
            model,
            reference_temp=298,
            temperatures=[280, 290, 300, 310],
            freq_range=(0.01, 100),
            num_points=20
        )
        
        # All curves should have shift_factor in metadata
        for T, curve in curves.items():
            assert 'shift_factor' in curve.metadata
            assert curve.metadata['shift_factor'] > 0


class TestDMAResult:
    def test_result_attributes(self):
        """Test DMA result has all required attributes"""
        model = GeneralizedMaxwell(E_inf=1e9, E_i=[5e8], tau_i=[0.1])
        result = frequency_sweep(model, freq_range=(0.01, 100), num_points=10)
        
        assert hasattr(result, 'frequency')
        assert hasattr(result, 'storage_modulus')
        assert hasattr(result, 'loss_modulus')
        assert hasattr(result, 'loss_tangent')
        assert hasattr(result, 'complex_modulus')
        assert hasattr(result, 'phase_angle')
    
    def test_complex_modulus_calculation(self):
        """Test complex modulus is calculated correctly"""
        model = GeneralizedMaxwell(E_inf=1e9, E_i=[5e8], tau_i=[0.1])
        result = frequency_sweep(model, freq_range=(0.01, 100), num_points=10)
        
        expected_complex = np.sqrt(result.storage_modulus**2 + result.loss_modulus**2)
        np.testing.assert_allclose(result.complex_modulus, expected_complex, rtol=1e-10)
    
    def test_phase_angle_calculation(self):
        """Test phase angle is calculated correctly"""
        model = GeneralizedMaxwell(E_inf=1e9, E_i=[5e8], tau_i=[0.1])
        result = frequency_sweep(model, freq_range=(0.01, 100), num_points=10)
        
        expected_phase = np.arctan2(result.loss_modulus, result.storage_modulus)
        np.testing.assert_allclose(result.phase_angle, expected_phase, rtol=1e-10)

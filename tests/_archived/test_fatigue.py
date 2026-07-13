"""
Tests for fatigue analysis module.
"""

import pytest
import numpy as np
from fibernet import gen
from fibernet.sim.fatigue import (
    FatigueAnalyzer, FatigueResult, CyclicLoadResult, SNPoint,
    analyze_fatigue
)


class TestSNPoint:
    """Test SNPoint dataclass."""
    
    def test_creation(self):
        point = SNPoint(stress_amplitude=1e7, cycles_to_failure=1000)
        assert point.stress_amplitude == 1e7
        assert point.cycles_to_failure == 1000
        assert point.stress_ratio == -1.0
    
    def test_to_dict(self):
        point = SNPoint(stress_amplitude=1e7, cycles_to_failure=1000, stress_ratio=0.1)
        data = point.to_dict()
        assert data['stress_amplitude'] == 1e7
        assert data['cycles_to_failure'] == 1000
        assert data['stress_ratio'] == 0.1


class TestFatigueResult:
    """Test FatigueResult dataclass."""
    
    def test_empty(self):
        result = FatigueResult()
        assert len(result.sn_curve) == 0
        assert result.fatigue_strength_coefficient == 0.0
    
    def test_to_dict(self):
        result = FatigueResult(
            fatigue_strength_coefficient=1e8,
            fatigue_strength_exponent=-0.12,
            endurance_limit=5e6,
        )
        data = result.to_dict()
        assert data['fatigue_strength_coefficient'] == 1e8
        assert data['endurance_limit'] == 5e6


class TestFatigueAnalyzer:
    """Test FatigueAnalyzer."""
    
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = FatigueAnalyzer(net)
        assert analyzer.network == net
        assert analyzer.uts > 0
        assert analyzer.sigma_f > 0
    
    def test_generate_sn_curve(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = FatigueAnalyzer(net)
        result = analyzer.generate_sn_curve(num_points=5)
        assert isinstance(result, FatigueResult)
        assert len(result.sn_curve) == 5
        assert result.fatigue_strength_coefficient > 0
    
    def test_predict_life(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = FatigueAnalyzer(net)
        Nf = analyzer.predict_life(stress_amplitude=1e7)
        assert isinstance(Nf, int)
        assert Nf > 0
    
    def test_cyclic_loading(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = FatigueAnalyzer(net)
        result = analyzer.cyclic_loading(
            stress_amplitude=1e7,
            num_cycles=100,
        )
        assert isinstance(result, CyclicLoadResult)
        assert result.num_cycles > 0
        assert result.residual_stiffness <= 1.0
    
    def test_variable_amplitude_loading(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = FatigueAnalyzer(net)
        load_spectrum = [
            (1e7, 100),
            (5e6, 500),
            (2e6, 1000),
        ]
        result = analyzer.variable_amplitude_loading(load_spectrum)
        assert isinstance(result, FatigueResult)
        assert result.damage_accumulation >= 0
    
    def test_goodman_diagram(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = FatigueAnalyzer(net)
        data = analyzer.goodman_diagram(
            stress_amplitudes=[1e7, 5e6, 2e6],
            mean_stresses=[0, 1e7, 2e7],
        )
        assert 'alternating_stress' in data
        assert 'safe_region' in data
        assert len(data['safe_region']) == 3


class TestAnalyzeFatigue:
    """Test convenience function."""
    
    def test_sn_curve(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        result = analyze_fatigue(net)
        assert isinstance(result, FatigueResult)
        assert len(result.sn_curve) > 0
    
    def test_cyclic_loading(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        result = analyze_fatigue(net, stress_amplitude=1e7, num_cycles=50)
        assert isinstance(result, CyclicLoadResult)
        assert result.num_cycles > 0

"""
Tests for creep analysis module.
"""

import pytest
import numpy as np
from fibernet import gen
from fibernet.sim.creep import (
    CreepAnalyzer, CreepResult, CreepModelParameters,
    analyze_creep
)


class TestCreepResult:
    """Test CreepResult dataclass."""
    
    def test_empty(self):
        result = CreepResult()
        assert len(result.time) == 0
        assert result.applied_stress == 0.0
    
    def test_to_dict(self):
        result = CreepResult(
            applied_stress=1e6,
            total_strain=0.01,
            steady_state_rate=1e-6,
        )
        data = result.to_dict()
        assert data['applied_stress'] == 1e6
        assert data['total_strain'] == 0.01


class TestCreepModelParameters:
    """Test CreepModelParameters dataclass."""
    
    def test_creation(self):
        params = CreepModelParameters(E1=1e9, E2=0.5e9, eta1=1e10, eta2=5e9)
        assert params.E1 == 1e9
        assert params.eta1 == 1e10
    
    def test_to_dict(self):
        params = CreepModelParameters(E1=1e9, E2=0.5e9, eta1=1e10, eta2=5e9)
        data = params.to_dict()
        assert data['E1'] == 1e9


class TestCreepAnalyzer:
    """Test CreepAnalyzer."""
    
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = CreepAnalyzer(net)
        assert analyzer.network == net
        assert analyzer.E0 > 0
        assert analyzer.eta > 0
    
    def test_creep_test(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = CreepAnalyzer(net)
        result = analyzer.creep_test(stress=1e6, duration=3600, num_points=50)
        assert isinstance(result, CreepResult)
        assert len(result.time) == 50
        assert len(result.strain) == 50
        assert result.applied_stress == 1e6
        assert result.total_strain > 0
    
    def test_creep_recovery(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = CreepAnalyzer(net)
        creep_result, recovery_result = analyzer.creep_recovery(
            stress=1e6,
            creep_duration=3600,
            recovery_duration=1800,
        )
        assert isinstance(creep_result, CreepResult)
        assert isinstance(recovery_result, CreepResult)
        assert creep_result.total_strain > 0
        assert recovery_result.recovery_strain >= 0
    
    def test_fit_burgers_model(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = CreepAnalyzer(net)
        result = analyzer.creep_test(stress=1e6, duration=3600)
        params = analyzer.fit_burgers_model(result.time, result.strain, stress=1e6)
        assert isinstance(params, CreepModelParameters)
        assert params.E1 > 0
        assert params.eta1 > 0
    
    def test_time_temperature_superposition(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = CreepAnalyzer(net)
        data = analyzer.time_temperature_superposition(
            reference_temperature=293.15,
            temperatures=[273.15, 293.15, 313.15, 333.15],
        )
        assert 'temperatures' in data
        assert 'shift_factors' in data
        assert len(data['shift_factors']) == 4
    
    def test_stress_relaxation(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = CreepAnalyzer(net)
        data = analyzer.stress_relaxation(
            initial_strain=0.01,
            duration=3600,
        )
        assert 'time' in data
        assert 'stress' in data
        assert data['initial_stress'] > 0
        assert data['final_stress'] <= data['initial_stress']


class TestAnalyzeCreep:
    """Test convenience function."""
    
    def test_basic_usage(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        result = analyze_creep(net, stress=1e6, duration=3600)
        assert isinstance(result, CreepResult)
        assert result.total_strain > 0

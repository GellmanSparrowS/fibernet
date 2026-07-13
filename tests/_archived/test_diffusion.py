"""
Tests for diffusion and transport module.
"""

import pytest
import numpy as np
from fibernet import gen
from fibernet.sim.diffusion import (
    DiffusionAnalyzer, DiffusionResult, FiltrationResult,
    analyze_diffusion
)


class TestDiffusionResult:
    """Test DiffusionResult dataclass."""
    
    def test_empty(self):
        result = DiffusionResult()
        assert result.effective_diffusion_coefficient == 0.0
        assert result.tortuosity == 0.0
    
    def test_to_dict(self):
        result = DiffusionResult(
            effective_diffusion_coefficient=5e-10,
            tortuosity=2.0,
            porosity=0.8,
        )
        data = result.to_dict()
        assert data['effective_diffusion_coefficient'] == 5e-10
        assert data['tortuosity'] == 2.0


class TestDiffusionAnalyzer:
    """Test DiffusionAnalyzer."""
    
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = DiffusionAnalyzer(net)
        assert analyzer.network == net
        assert analyzer.D_mol > 0
        assert 0 <= analyzer.porosity <= 1
    
    def test_compute_porosity(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = DiffusionAnalyzer(net)
        porosity = analyzer._compute_porosity()
        assert 0 <= porosity <= 1
    
    def test_compute_effective_diffusion(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = DiffusionAnalyzer(net)
        result = analyzer.compute_effective_diffusion()
        assert isinstance(result, DiffusionResult)
        assert result.effective_diffusion_coefficient > 0
        assert result.tortuosity >= 1.0
    
    def test_simulate_concentration_profile(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = DiffusionAnalyzer(net)
        result = analyzer.simulate_concentration_profile(
            initial_concentration=1.0,
            duration=3600,
            num_time_steps=50,
            num_space_steps=30,
        )
        assert isinstance(result, DiffusionResult)
        assert result.concentration_profile is not None
        assert result.concentration_profile.shape == (50, 30)
        assert result.breakthrough_time >= 0
    
    def test_compute_tortuosity(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = DiffusionAnalyzer(net)
        
        tau_bruggeman = analyzer.compute_tortuosity(method='bruggeman')
        assert tau_bruggeman >= 1.0
        
        tau_comiti = analyzer.compute_tortuosity(method='comiti')
        assert tau_comiti >= 1.0
        
        tau_weissberg = analyzer.compute_tortuosity(method='weissberg')
        assert tau_weissberg >= 1.0
    
    def test_filtration_analysis(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = DiffusionAnalyzer(net)
        result = analyzer.filtration_analysis(
            particle_size=1e-6,
            flow_velocity=0.01,
        )
        assert isinstance(result, FiltrationResult)
        assert 0 <= result.capture_efficiency <= 1
        assert result.pressure_drop >= 0


class TestAnalyzeDiffusion:
    """Test convenience function."""
    
    def test_basic_usage(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        result = analyze_diffusion(net)
        assert isinstance(result, DiffusionResult)
        assert result.effective_diffusion_coefficient > 0
        assert result.tortuosity >= 1.0

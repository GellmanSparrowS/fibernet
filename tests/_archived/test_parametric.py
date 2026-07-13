"""Tests for parametric study utilities."""

import numpy as np
import pytest
from fibernet import gen
from fibernet.analysis import MorphologyAnalyzer
from fibernet.utils.parametric import (
    parametric_sweep, sensitivity_analysis,
    monte_carlo_analysis, correlation_matrix
)


def analyze(net):
    morph = MorphologyAnalyzer(net)
    return {
        'nematic': morph.nematic_order_parameter(),
        'porosity': morph.porosity()
    }


class TestParametricSweep:
    def test_basic_sweep(self):
        params, metrics = parametric_sweep(
            {'num_fibers': [20, 40]},
            lambda **kw: gen.random_straight_2d(**kw, fiber_length=8, box_size=(25, 25), seed=42),
            analyze,
            show_progress=False
        )
        
        assert len(params['num_fibers']) == 2
        assert len(metrics['nematic']) == 2
        assert len(metrics['porosity']) == 2
    
    def test_multi_param_sweep(self):
        params, metrics = parametric_sweep(
            {'num_fibers': [20, 40], 'fiber_length': [5, 10]},
            lambda **kw: gen.random_straight_2d(**kw, box_size=(25, 25), seed=42),
            analyze,
            show_progress=False
        )
        
        assert len(params['num_fibers']) == 4  # 2 x 2 combinations


class TestSensitivityAnalysis:
    def test_basic_sensitivity(self):
        results = sensitivity_analysis(
            'num_fibers',
            [20, 40, 60],
            lambda **kw: gen.random_straight_2d(**kw, fiber_length=8, box_size=(25, 25)),
            analyze,
            num_samples=2,
            show_progress=False
        )
        
        assert len(results['param_values']) == 3
        assert len(results['metrics_mean']['nematic']) == 3
        assert len(results['metrics_std']['nematic']) == 3


class TestMonteCarlo:
    def test_basic_mc(self):
        results = monte_carlo_analysis(
            lambda seed: gen.random_straight_2d(num_fibers=30, fiber_length=8, box_size=(25, 25), seed=seed),
            analyze,
            num_samples=5,
            show_progress=False
        )
        
        assert 'nematic' in results['mean']
        assert 'nematic' in results['std']
        assert len(results['samples']) == 5


class TestCorrelationMatrix:
    def test_basic_correlation(self):
        params = {
            'num_fibers': np.array([20, 40, 60, 80]),
            'fiber_length': np.array([5, 10, 15, 20])
        }
        metrics = {
            'nematic': np.array([0.1, 0.2, 0.3, 0.4]),
            'porosity': np.array([0.9, 0.8, 0.7, 0.6])
        }
        
        corr, pvals = correlation_matrix(params, metrics)
        assert corr.shape == (2, 2)
        assert pvals.shape == (2, 2)

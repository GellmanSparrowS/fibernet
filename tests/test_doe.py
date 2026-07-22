"""
Tests for design of experiments module.
"""

import pytest
import numpy as np
from fibernet import gen
from fibernet.doe import (
    DesignOfExperiments, ExperimentDesign, ExperimentResult, SweepResult,
    run_parameter_sweep
)


class TestExperimentDesign:
    """Test ExperimentDesign dataclass."""
    
    def test_empty_design(self):
        design = ExperimentDesign()
        assert len(design.parameters) == 0
        assert design.num_points == 0
    
    def test_design_with_parameters(self):
        design = ExperimentDesign(
            parameters={'num_fibers': [50, 100], 'length': [5.0, 10.0]},
            parameter_names=['num_fibers', 'length'],
            num_points=4,
        )
        assert design.num_points == 4
        assert len(design.parameter_names) == 2
    
    def test_to_dict(self):
        design = ExperimentDesign(
            parameters={'x': [1, 2, 3]},
            parameter_names=['x'],
            num_points=3,
        )
        data = design.to_dict()
        assert isinstance(data, dict)
        assert 'num_points' in data


class TestExperimentResult:
    """Test ExperimentResult dataclass."""
    
    def test_empty_result(self):
        result = ExperimentResult()
        assert len(result.parameters) == 0
        assert len(result.outputs) == 0
    
    def test_result_with_data(self):
        result = ExperimentResult(
            parameters={'x': 5.0},
            outputs={'modulus': 1e9},
        )
        assert result.parameters['x'] == 5.0
        assert result.outputs['modulus'] == 1e9
    
    def test_to_dict(self):
        result = ExperimentResult(
            parameters={'x': 5.0},
            outputs={'modulus': 1e9},
        )
        data = result.to_dict()
        assert 'parameters' in data
        assert 'outputs' in data


class TestDesignOfExperiments:
    """Test DesignOfExperiments."""
    
    def test_initialization(self):
        doe = DesignOfExperiments(gen.random_straight_2d, {})
        assert doe.generator_func == gen.random_straight_2d
    
    def test_grid_search(self):
        params = {
            'num_fibers': [20, 40],
            'fiber_length': [5.0, 10.0],
        }
        doe = DesignOfExperiments(gen.random_straight_2d, {'seed': 42})
        result = doe.grid_search(params)
        
        assert isinstance(result, SweepResult)
        assert len(result.results) == 4  # 2x2 grid
        assert result.design.num_points == 4
    
    def test_grid_search_with_output(self):
        def compute_length(net):
            return {'total_length': net.total_length}
        
        params = {'num_fibers': [20, 40]}
        doe = DesignOfExperiments(gen.random_straight_2d, {'seed': 42}, compute_length)
        result = doe.grid_search(params)
        
        assert len(result.results) == 2
        assert 'total_length' in result.results[0].outputs
        assert result.results[0].outputs['total_length'] > 0
    
    def test_latin_hypercube(self):
        params = {
            'num_fibers': (20.0, 100.0),
            'fiber_length': (5.0, 15.0),
        }
        doe = DesignOfExperiments(gen.random_straight_2d, {'seed': 42})
        result = doe.latin_hypercube(params, num_samples=10, seed=42)
        
        assert isinstance(result, SweepResult)
        assert len(result.results) == 10
        assert result.design.design_points.shape == (10, 2)
    
    def test_random_search(self):
        params = {
            'num_fibers': (20.0, 100.0),
            'fiber_length': (5.0, 15.0),
        }
        doe = DesignOfExperiments(gen.random_straight_2d, {'seed': 42})
        result = doe.random_search(params, num_samples=10, seed=42)
        
        assert isinstance(result, SweepResult)
        assert len(result.results) == 10
    
    def test_sensitivity_analysis(self):
        params = {
            'num_fibers': [20, 40, 60],
            'fiber_length': [5.0, 10.0],
        }
        
        def compute_length(net):
            return {'total_length': net.total_length}
        
        doe = DesignOfExperiments(gen.random_straight_2d, {'seed': 42}, compute_length)
        result = doe.grid_search(params)
        
        sensitivities = doe.sensitivity_analysis(result, 'total_length')
        assert isinstance(sensitivities, dict)
        assert 'num_fibers' in sensitivities
        assert 'fiber_length' in sensitivities


class TestSweepResult:
    """Test SweepResult."""
    
    def test_empty_result(self):
        result = SweepResult()
        assert len(result.results) == 0
        assert result.design is None
    
    def test_to_dataframe(self):
        pandas = pytest.importorskip("pandas")
        
        params = {'num_fibers': [20, 40]}
        doe = DesignOfExperiments(gen.random_straight_2d, {'seed': 42})
        result = doe.grid_search(params)
        
        df = result.to_dataframe()
        assert df is not None
        assert len(df) == 2
        assert 'num_fibers' in df.columns
    
    def test_to_dict(self):
        params = {'num_fibers': [20, 40]}
        doe = DesignOfExperiments(gen.random_straight_2d, {'seed': 42})
        result = doe.grid_search(params)
        
        data = result.to_dict()
        assert 'num_results' in data
        assert data['num_results'] == 2


class TestRunParameterSweep:
    """Test convenience function."""
    
    def test_grid_sweep(self):
        params = {'num_fibers': [20, 40]}
        result = run_parameter_sweep(
            gen.random_straight_2d,
            params,
            method='grid',
            seed=42,
        )
        assert len(result.results) == 2
    
    def test_lhs_sweep(self):
        params = {'num_fibers': (20.0, 100.0)}
        result = run_parameter_sweep(
            gen.random_straight_2d,
            params,
            method='lhs',
            num_samples=5,
            seed=42,
        )
        assert len(result.results) == 5
    
    def test_random_sweep(self):
        params = {'num_fibers': (20.0, 100.0)}
        result = run_parameter_sweep(
            gen.random_straight_2d,
            params,
            method='random',
            num_samples=5,
            seed=42,
        )
        assert len(result.results) == 5
    
    def test_invalid_method(self):
        params = {'num_fibers': [20, 40]}
        with pytest.raises(ValueError):
            run_parameter_sweep(gen.random_straight_2d, params, method='invalid')

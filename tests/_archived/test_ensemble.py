"""Tests for statistical ensemble generation."""

import pytest
import numpy as np
from fibernet import gen
from fibernet.utils.ensemble import generate_ensemble, ensemble_analysis
from fibernet.analysis.morphology import MorphologyAnalyzer


class TestGenerateEnsemble:
    def test_basic_generation(self):
        """Test basic ensemble generation"""
        ensemble = generate_ensemble(
            gen.random_straight_2d,
            num_networks=5,
            base_seed=42,
            num_fibers=50,
            fiber_length=10.0,
            box_size=(30, 30),
            show_progress=False
        )
        assert len(ensemble) == 5
        assert len(ensemble.seeds) == 5
    
    def test_different_seeds(self):
        """Test different seeds produce different networks"""
        ensemble = generate_ensemble(
            gen.random_straight_2d,
            num_networks=3,
            base_seed=42,
            num_fibers=50,
            fiber_length=10.0,
            box_size=(30, 30),
            show_progress=False
        )
        assert len(set(ensemble.seeds)) == 3
    
    def test_reproducibility(self):
        """Test same seed produces same ensemble"""
        ensemble1 = generate_ensemble(
            gen.random_straight_2d,
            num_networks=3,
            base_seed=42,
            num_fibers=50,
            fiber_length=10.0,
            box_size=(30, 30),
            show_progress=False
        )
        ensemble2 = generate_ensemble(
            gen.random_straight_2d,
            num_networks=3,
            base_seed=42,
            num_fibers=50,
            fiber_length=10.0,
            box_size=(30, 30),
            show_progress=False
        )
        assert ensemble1.seeds == ensemble2.seeds
    
    def test_base_seed_offset(self):
        """Test base_seed creates offset sequence"""
        ensemble = generate_ensemble(
            gen.random_straight_2d,
            num_networks=5,
            base_seed=100,
            num_fibers=50,
            fiber_length=10.0,
            box_size=(30, 30),
            show_progress=False
        )
        expected_seeds = [100, 101, 102, 103, 104]
        assert ensemble.seeds == expected_seeds


class TestEnsembleAnalysis:
    def test_basic_analysis(self):
        """Test basic ensemble analysis"""
        ensemble = generate_ensemble(
            gen.random_straight_2d,
            num_networks=5,
            base_seed=42,
            num_fibers=50,
            fiber_length=10.0,
            box_size=(30, 30),
            show_progress=False
        )
        
        def analyze(net):
            morph = MorphologyAnalyzer(net)
            return {
                'nematic_order': morph.nematic_order_parameter(),
                'porosity': morph.porosity(),
            }
        
        results = ensemble_analysis(ensemble, analyze, show_progress=False)
        
        assert 'nematic_order' in results
        assert 'porosity' in results
        assert 'mean' in results['nematic_order']
        assert 'std' in results['nematic_order']
        assert 'values' in results['nematic_order']
    
    def test_analysis_values(self):
        """Test analysis produces reasonable values"""
        ensemble = generate_ensemble(
            gen.random_straight_2d,
            num_networks=10,
            base_seed=42,
            num_fibers=50,
            fiber_length=10.0,
            box_size=(30, 30),
            show_progress=False
        )
        
        def analyze(net):
            morph = MorphologyAnalyzer(net)
            return {'nematic_order': morph.nematic_order_parameter()}
        
        results = ensemble_analysis(ensemble, analyze, show_progress=False)
        mean_nematic = results['nematic_order']['mean']
        assert 0 <= mean_nematic <= 1


class TestEnsembleStatistics:
    def test_basic_statistics(self):
        """Test basic ensemble statistics"""
        ensemble = generate_ensemble(
            gen.random_straight_2d,
            num_networks=10,
            base_seed=42,
            num_fibers=50,
            fiber_length=10.0,
            box_size=(30, 30),
            show_progress=False
        )
        
        stats = ensemble.compute_statistics()
        
        assert 'num_fibers' in stats
        assert 'num_crosslinks' in stats
        assert 'total_length' in stats
    
    def test_statistics_values(self):
        """Test statistics produce reasonable values"""
        ensemble = generate_ensemble(
            gen.random_straight_2d,
            num_networks=10,
            base_seed=42,
            num_fibers=50,
            fiber_length=10.0,
            box_size=(30, 30),
            show_progress=False
        )
        
        stats = ensemble.compute_statistics()
        
        assert abs(stats['num_fibers']['mean'] - 50) < 1
        assert abs(stats['total_length']['mean'] - 500) < 10
    
    def test_statistics_std(self):
        """Test standard deviation is calculated"""
        ensemble = generate_ensemble(
            gen.random_straight_2d,
            num_networks=10,
            base_seed=42,
            num_fibers=50,
            fiber_length=10.0,
            box_size=(30, 30),
            show_progress=False
        )
        
        stats = ensemble.compute_statistics()
        
        assert 'std' in stats['num_crosslinks']
        assert stats['num_crosslinks']['std'] >= 0
    
    def test_statistics_min_max(self):
        """Test min/max are calculated"""
        ensemble = generate_ensemble(
            gen.random_straight_2d,
            num_networks=10,
            base_seed=42,
            num_fibers=50,
            fiber_length=10.0,
            box_size=(30, 30),
            show_progress=False
        )
        
        stats = ensemble.compute_statistics()
        
        assert 'min' in stats['num_crosslinks']
        assert 'max' in stats['num_crosslinks']
        assert stats['num_crosslinks']['min'] <= stats['num_crosslinks']['max']


class TestEnsembleWithGenerators:
    def test_with_oriented_generator(self):
        """Test ensemble with oriented generator"""
        ensemble = generate_ensemble(
            gen.oriented_random_2d,
            num_networks=3,
            base_seed=42,
            num_fibers=30,
            fiber_length=10.0,
            box_size=(30, 30),
            show_progress=False
        )
        assert len(ensemble) == 3
    
    def test_with_random_generator(self):
        """Test ensemble with another random generator"""
        ensemble = generate_ensemble(
            gen.random_straight_3d,
            num_networks=3,
            base_seed=42,
            num_fibers=20,
            fiber_length=10.0,
            box_size=(30, 30, 30),
            show_progress=False
        )
        assert len(ensemble) == 3

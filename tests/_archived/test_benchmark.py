"""Benchmark tests for performance tracking."""

import os
import numpy as np
import pytest
import time
from fibernet import gen
from fibernet.sim.mechanical import FiberFEM
from fibernet.analysis import MorphologyAnalyzer

IS_CI = os.environ.get("CI", "false").lower() == "true"


class TestBenchmarks:
    @pytest.mark.skipif(IS_CI, reason="Performance benchmarks unreliable in CI")
    def test_generation_performance(self):
        """Test network generation speed."""
        start = time.time()
        net = gen.random_straight_2d(num_fibers=1000, fiber_length=10, box_size=(100, 100), seed=42)
        elapsed = time.time() - start
        
        assert net.num_fibers == 1000
        assert elapsed < 120.0

    @pytest.mark.skipif(IS_CI, reason="Performance benchmarks unreliable in CI")
    def test_analysis_performance(self):
        """Test analysis speed."""
        net = gen.random_straight_2d(num_fibers=500, fiber_length=10, box_size=(50, 50), seed=42)
        
        start = time.time()
        morph = MorphologyAnalyzer(net)
        report = morph.full_report()
        elapsed = time.time() - start
        
        assert elapsed < 60.0
        assert 'nematic_order' in report

    @pytest.mark.skipif(IS_CI, reason="Performance benchmarks unreliable in CI")
    def test_fem_performance(self):
        """Test FEM simulation speed."""
        net = gen.square_lattice_2d(spacing=5, grid_size=(3, 3))
        
        start = time.time()
        fem = FiberFEM(net, segments_per_fiber=3)
        result = fem.apply_uniaxial_strain(strain=0.001, axis=0)
        elapsed = time.time() - start
        
        assert elapsed < 120.0
        assert not np.isnan(result.energy)

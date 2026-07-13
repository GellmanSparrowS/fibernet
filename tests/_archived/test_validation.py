"""Tests for validation module."""

import numpy as np
import pytest

from fibernet.sim.validation import (
    gibson_ashby_honeycomb,
    gibson_ashby_foam_3d,
    gibson_ashby_closed_cell,
    euler_beam_cantilever,
    validate_cantilever_beam,
    validate_patch_test,
    run_all_validations,
    print_validation_report,
)


class TestGibsonAshby:
    def test_honeycomb_scaling(self):
        result1 = gibson_ashby_honeycomb(relative_density=0.1, E_solid=1e9)
        result2 = gibson_ashby_honeycomb(relative_density=0.2, E_solid=1e9)
        assert result2['E1'] > result1['E1']
    
    def test_honeycomb_properties(self):
        result = gibson_ashby_honeycomb(relative_density=0.1, E_solid=1e9)
        assert 'E1' in result
        assert 'E2' in result
        assert 'nu12' in result
        assert 'G12' in result
        assert result['E1'] > 0
        assert result['E2'] > 0
    
    def test_foam_3d(self):
        result = gibson_ashby_foam_3d(relative_density=0.1, E_solid=1e9)
        assert result['E_star'] > 0
        assert result['sigma_star'] > 0
    
    def test_closed_cell(self):
        result = gibson_ashby_closed_cell(relative_density=0.1, E_solid=1e9, phi=0.8)
        assert result['E_star'] > 0
        assert result['phi'] == 0.8


class TestEulerBeam:
    def test_cantilever(self):
        result = euler_beam_cantilever(L=1.0, E=1e9, I=1e-12, P=1.0)
        assert 'delta_tip' in result
        assert 'theta_tip' in result
        assert 'M_max' in result
        assert result['delta_tip'] > 0
        assert result['M_max'] == 1.0
    
    def test_scaling(self):
        r1 = euler_beam_cantilever(L=1.0, E=1e9, I=1e-12, P=1.0)
        r2 = euler_beam_cantilever(L=2.0, E=1e9, I=1e-12, P=1.0)
        assert r2['delta_tip'] == pytest.approx(r1['delta_tip'] * 8, rel=1e-10)


class TestValidation:
    def test_cantilever_fem(self):
        result = validate_cantilever_beam(segments=20, tolerance=0.2)
        assert result.test_name == "Cantilever Beam (Euler-Bernoulli)"
        assert result.analytical_value > 0
        assert result.numerical_value > 0
    
    def test_patch_test(self):
        result = validate_patch_test()
        assert result.test_name == "Patch Test (uniform strain)"
    
    def test_run_all(self):
        results = run_all_validations(verbose=False)
        assert len(results) >= 3
    
    def test_report(self):
        results = run_all_validations(verbose=False)
        report = print_validation_report(results)
        assert "Validation Report" in report
        assert "Summary" in report

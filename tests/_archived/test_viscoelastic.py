"""Tests for viscoelastic material models."""

import pytest
import numpy as np
from fibernet.sim import (
    MaxwellModel, KelvinVoigtModel, StandardLinearSolid, GeneralizedMaxwell
)


class TestMaxwellModel:
    def test_init(self):
        """Test Maxwell model initialization."""
        model = MaxwellModel(E=1e9, eta=1e8)
        assert model.E == 1e9
        assert model.eta == 1e8
        assert model.tau == 0.1
    
    def test_stress_relaxation(self):
        """Test stress relaxation."""
        model = MaxwellModel(E=1e9, eta=1e8)
        result = model.stress_relaxation(strain=0.01, time_range=(0, 1), num_steps=50)
        
        assert len(result.time) == 50
        assert len(result.stress) == 50
        
        # Initial stress should be E*strain
        assert result.stress[0] == pytest.approx(1e9 * 0.01, rel=1e-6)
        
        # Stress should decay
        assert result.stress[-1] < result.stress[0]
    
    def test_creep(self):
        """Test creep behavior."""
        model = MaxwellModel(E=1e9, eta=1e8)
        result = model.creep(stress=10e6, time_range=(0, 1), num_steps=50)
        
        assert len(result.time) == 50
        assert len(result.strain) == 50
        
        # Strain should increase over time
        assert result.strain[-1] > result.strain[0]


class TestKelvinVoigtModel:
    def test_init(self):
        """Test Kelvin-Voigt model initialization."""
        model = KelvinVoigtModel(E=1e9, eta=1e8)
        assert model.E == 1e9
        assert model.eta == 1e8
        assert model.tau == 0.1
    
    def test_creep(self):
        """Test creep behavior."""
        model = KelvinVoigtModel(E=1e9, eta=1e8)
        result = model.creep(stress=10e6, time_range=(0, 1), num_steps=50)
        
        assert len(result.time) == 50
        assert len(result.strain) == 50
        
        # Initial strain should be 0
        assert result.strain[0] == 0.0
        
        # Strain should approach steady state
        steady_state = 10e6 / 1e9
        assert result.strain[-1] == pytest.approx(steady_state, rel=0.1)


class TestStandardLinearSolid:
    def test_init(self):
        """Test SLS model initialization."""
        model = StandardLinearSolid(E1=0.5e9, E2=0.5e9, eta=1e8)
        assert model.E1 == 0.5e9
        assert model.E2 == 0.5e9
        assert model.eta == 1e8
        assert model.E_instant == 1e9
        assert model.E_equilibrium == 0.5e9
    
    def test_stress_relaxation(self):
        """Test stress relaxation."""
        model = StandardLinearSolid(E1=0.5e9, E2=0.5e9, eta=1e8)
        result = model.stress_relaxation(strain=0.01, time_range=(0, 1), num_steps=50)
        
        # Initial stress should be (E1+E2)*strain
        expected_initial = (0.5e9 + 0.5e9) * 0.01
        assert result.stress[0] == pytest.approx(expected_initial, rel=1e-6)
        
        # Final stress should approach E1*strain
        expected_final = 0.5e9 * 0.01
        assert result.stress[-1] == pytest.approx(expected_final, rel=0.1)


class TestGeneralizedMaxwell:
    def test_init(self):
        """Test Generalized Maxwell model initialization."""
        model = GeneralizedMaxwell(E_inf=0.5e9, E_i=[0.3e9, 0.2e9], tau_i=[0.5, 2.0])
        assert model.E_inf == 0.5e9
        assert len(model.E_i) == 2
        assert len(model.tau_i) == 2
        assert model.E_instant == pytest.approx(1e9, rel=1e-6)
    
    def test_init_length_mismatch(self):
        """Test that mismatched lengths raise error."""
        with pytest.raises(ValueError, match="same length"):
            GeneralizedMaxwell(E_inf=0.5e9, E_i=[0.3e9, 0.2e9], tau_i=[0.5])
    
    def test_stress_relaxation(self):
        """Test stress relaxation."""
        model = GeneralizedMaxwell(E_inf=0.5e9, E_i=[0.3e9, 0.2e9], tau_i=[0.5, 2.0])
        result = model.stress_relaxation(strain=0.01, time_range=(0, 10), num_steps=50)
        
        # Initial stress should be E_instant*strain
        expected_initial = 1e9 * 0.01
        assert result.stress[0] == pytest.approx(expected_initial, rel=1e-6)
        
        # Final stress should approach E_inf*strain
        expected_final = 0.5e9 * 0.01
        assert result.stress[-1] == pytest.approx(expected_final, rel=0.1)
    
    def test_single_element(self):
        """Test with single Maxwell element."""
        model = GeneralizedMaxwell(E_inf=0.5e9, E_i=[0.5e9], tau_i=[1.0])
        result = model.stress_relaxation(strain=0.01, time_range=(0, 5), num_steps=50)
        
        assert len(result.time) == 50
        assert len(result.stress) == 50

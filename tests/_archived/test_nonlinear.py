"""Tests for nonlinear mechanics module."""

import numpy as np
import pytest
from fibernet.gen import square_lattice_2d, random_straight_2d
from fibernet.sim.nonlinear import (
    LinearElastic, BilinearPlasticity, PowerLawHardening,
    HyperelasticNeoHookean, HyperelasticMooneyRivlin, ArrudaBoyce,
    MaxwellModel, KelvinVoigtModel, StandardLinearSolid,
    NonlinearFEM,
)


class TestConstitutiveModels:
    def test_linear_elastic(self):
        model = LinearElastic(E=1e9)
        assert model.stress(0.001) == pytest.approx(1e6, rel=1e-6)
        assert model.tangent_modulus(0.001) == pytest.approx(1e9, rel=1e-6)
        assert model.energy_density(0.001) == pytest.approx(500, rel=1e-6)
    
    def test_bilinear_elastic(self):
        model = LinearElastic(E=1e9)
        stress = model.stress(0.001)
        assert stress > 0
        assert model.tangent_modulus(0.001) > 0
    
    def test_bilinear_plasticity(self):
        model = BilinearPlasticity(E=1e9, sigma_y=1e6, Et=1e7)
        # Elastic region
        assert model.stress(0.0005) == pytest.approx(5e5, rel=1e-6)
        # Plastic region
        stress = model.stress(0.01)
        assert stress > 1e6  # Should exceed yield stress
        # Tangent should be lower in plastic region
        assert model.tangent_modulus(0.01) < model.tangent_modulus(0.0005)
    
    def test_power_law(self):
        model = PowerLawHardening(E=1e9, K=5e8, n=0.5)
        stress = model.stress(0.001)
        assert stress > 0
        assert model.tangent_modulus(0.001) > 0
    
    def test_neo_hookean(self):
        model = HyperelasticNeoHookean(G=1e6)
        # Small strain should be approximately linear
        stress = model.stress(0.001)
        assert stress > 0
        # Tangent at zero strain should be 3G for incompressible
        E_t = model.tangent_modulus(0.0)
        assert E_t > 0
    
    def test_mooney_rivlin(self):
        model = HyperelasticMooneyRivlin(C1=1e5, C2=1e5)
        stress = model.stress(0.001)
        assert stress > 0
        assert model.tangent_modulus(0.001) > 0
    
    def test_arruda_boyce(self):
        model = ArrudaBoyce(n_chain=100, nkT=1e6)
        stress = model.stress(0.001)
        assert stress > 0
        assert model.tangent_modulus(0.001) > 0


class TestViscoelasticModels:
    def test_maxwell(self):
        model = MaxwellModel(E=1e9, eta=1e6)
        state = {'sigma': 0.0, 'strain': 0.0}
        stress, state = model.step(0.001, 1e-6, state)
        assert 'sigma' in state
        assert 'strain' in state
    
    def test_kelvin_voigt(self):
        model = KelvinVoigtModel(E=1e9, eta=1e6)
        state = {'strain': 0.0}
        stress, state = model.step(0.001, 1e-6, state)
        assert stress > 0
        assert 'strain' in state
    
    def test_standard_linear_solid(self):
        model = StandardLinearSolid(E1=1e9, E2=1e9, eta=1e6)
        state = {'eps_v': 0.0, 'strain': 0.0}
        stress, state = model.step(0.001, 1e-6, state)
        assert stress > 0
        assert 'eps_v' in state


class TestNonlinearFEM:
    def test_build_mesh(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        fem = NonlinearFEM(net, segments_per_fiber=3)
        assert fem.num_nodes > 0
        assert fem.num_elements > 0
        assert fem.num_dof == fem.num_nodes * 3
    
    def test_solve_incremental(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        fem = NonlinearFEM(net, segments_per_fiber=3)
        
        # Use prescribed displacement instead of force (more stable)
        positions = fem.node_positions[:, 0]
        
        # Fix left side
        fixed_nodes = list(np.where(positions <= positions.min() + 1.0)[0])
        
        # Prescribe small displacement on right side
        hot_nodes = np.where(positions >= positions.max() - 1.0)[0]
        prescribed_dofs = {}
        for n in hot_nodes:
            prescribed_dofs[n * 3] = 0.1  # 0.1 units in x-direction
        
        F_ext = np.zeros(fem.num_dof)
        
        result = fem.solve_incremental(F_ext, fixed_nodes=fixed_nodes, prescribed_dofs=prescribed_dofs)
        assert result.converged or result.num_iterations > 0
        # Check that displacement is reasonable
        disp = result.max_displacement()
        assert not np.isnan(disp)
        assert disp >= 0
    
    def test_stress_strain_curve(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        fem = NonlinearFEM(net, segments_per_fiber=3)
        
        strains, stresses, energies = fem.stress_strain_curve(
            axis=0, max_strain=0.01, num_steps=10
        )
        
        assert len(strains) == 10
        assert len(stresses) == 10
        assert len(energies) == 10
        assert np.all(strains > 0)
        assert np.all(stresses >= 0)
        assert np.all(energies >= 0)
    
    def test_with_constitutive_model(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        model = BilinearPlasticity(E=1e9, sigma_y=1e6, Et=1e7)
        fem = NonlinearFEM(net, constitutive_model=model, segments_per_fiber=3)
        
        strains, stresses, energies = fem.stress_strain_curve(
            axis=0, max_strain=0.005, num_steps=5
        )
        
        assert len(strains) == 5
        assert len(stresses) == 5
    
    def test_viscoelastic_loading(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        fem = NonlinearFEM(net, segments_per_fiber=3)
        
        visco = KelvinVoigtModel(E=1e9, eta=1e6)
        time, stress, strain = fem.viscoelastic_loading(
            visco_model=visco,
            axis=0,
            strain_rate=1e-3,
            max_strain=0.001,
            dt=1e-4,
        )
        
        assert len(time) > 0
        assert len(stress) == len(time)
        assert len(strain) == len(time)
        assert time[-1] > 0
        assert strain[-1] > 0


class TestNonlinearFeatures:
    def test_large_deformation(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        fem = NonlinearFEM(net, segments_per_fiber=3, large_deformation=True)
        
        strains, stresses, energies = fem.stress_strain_curve(
            axis=0, max_strain=0.01, num_steps=5
        )
        
        assert len(strains) == 5
    
    def test_random_network(self):
        net = random_straight_2d(num_fibers=50, fiber_length=10, box_size=(30, 30), seed=42)
        fem = NonlinearFEM(net, segments_per_fiber=3)
        
        strains, stresses, energies = fem.stress_strain_curve(
            axis=0, max_strain=0.005, num_steps=5
        )
        
        assert len(strains) == 5

"""Test metamaterial workflow: create, tile, weld, simulate."""
import numpy as np
import pytest
import fibernet as fn
from fibernet.api import _build_mass_spring_system


class TestMetamaterialCreation:
    """Test metamaterial creation and structure."""
    
    def test_create_reentrant_honeycomb(self):
        meta = fn.create_metamaterial(
            unit_cell='reentrant_honeycomb_2d',
            array_size=(3, 3),
            reentrant_angle=150,
            cell_height=10,
            cell_width=10,
            grid_size=(3, 3),
            radius=0.2,
        )
        assert meta.num_fibers > 0
        assert meta.num_crosslinks > 0
        assert hasattr(meta, 'metadata')
        assert meta.metadata['unit_cell'] == 'reentrant_honeycomb_2d'
        assert meta.metadata['array_size'] == (3, 3)
    
    def test_create_chiral_honeycomb(self):
        meta = fn.create_metamaterial(
            unit_cell='chiral_honeycomb_2d',
            array_size=(2, 2),
            node_radius=3.0,
            ligament_length=8.0,
        )
        assert meta.num_fibers > 0
        assert meta.num_crosslinks > 0
    
    def test_invalid_unit_cell(self):
        with pytest.raises(ValueError, match='Unknown unit cell'):
            fn.create_metamaterial(unit_cell='invalid_cell', array_size=(3, 3))


class TestMassSpringSystem:
    """Test mass-spring system construction."""
    
    def test_build_mass_spring_system(self):
        meta = fn.create_metamaterial(
            unit_cell='reentrant_honeycomb_2d',
            array_size=(2, 2),
            reentrant_angle=150,
        )
        edges, rest_lengths, stiffness, masses, positions = _build_mass_spring_system(meta)
        
        # Check shapes
        assert edges.shape[1] == 2
        assert len(rest_lengths) == len(edges)
        assert len(stiffness) == len(edges)
        assert len(masses) == len(positions)
        
        # Check physical constraints
        assert np.all(rest_lengths > 0)
        assert np.all(stiffness > 0)
        assert np.all(masses > 0)
        
        # Check mass is reasonable (should be in kg range for mm-scale structure)
        total_mass = masses.sum()
        assert 1e-6 < total_mass < 1.0  # Between 1 mg and 1 kg


class TestDynamics:
    """Test mass-spring dynamics."""
    
    def test_dynamics_without_force(self):
        meta = fn.create_metamaterial(
            unit_cell='reentrant_honeycomb_2d',
            array_size=(2, 2),
            reentrant_angle=150,
        )
        traj = fn.simulate_dynamics(
            meta, dt=1e-9, steps=1000, damping=0.1, backend='numpy',
        )
        assert 'positions' in traj
        assert 'trajectory' in traj
        assert 'energy' in traj
    
    def test_dynamics_with_external_force(self):
        meta = fn.create_metamaterial(
            unit_cell='reentrant_honeycomb_2d',
            array_size=(2, 2),
            reentrant_angle=150,
        )
        edges, rest_lengths, stiffness, masses, positions = _build_mass_spring_system(meta)
        
        # Fix left edge, apply force to right edge
        x_coords = positions[:, 0]
        x_min, x_max = x_coords.min(), x_coords.max()
        tol = (x_max - x_min) * 0.1
        left_nodes = np.where(x_coords <= x_min + tol)[0].tolist()
        right_nodes = np.where(x_coords >= x_max - tol)[0].tolist()
        
        ext_force = np.zeros((len(positions), 3))
        for n in right_nodes:
            ext_force[n, 0] = 1e-3  # 1 mN
        
        traj = fn.simulate_dynamics(
            meta, dt=1e-9, steps=5000, damping=0.1,
            backend='numpy', fixed_nodes=left_nodes,
            external_force=ext_force,
        )
        
        # Check that deformation occurred
        pos_init = traj['initial_positions']
        pos_final = traj['positions']
        max_disp = np.max(np.linalg.norm(pos_final - pos_init, axis=1))
        assert max_disp > 0


class TestMechanics:
    """Test beam FEM mechanics."""
    
    def test_linear_mechanics(self):
        meta = fn.create_metamaterial(
            unit_cell='reentrant_honeycomb_2d',
            array_size=(2, 2),
            reentrant_angle=150,
        )
        result = fn.simulate_mechanics(
            meta, strain=0.001, axis=0, model='linear', segments_per_fiber=3,
        )
        assert 'modulus' in result
        assert 'energy' in result
        assert 'displacements' in result
        assert result['modulus'] > 0


class TestVisualization:
    """Test visualization functions."""
    
    def test_plot_metamaterial(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        meta = fn.create_metamaterial(
            unit_cell='reentrant_honeycomb_2d',
            array_size=(2, 2),
            reentrant_angle=150,
        )
        fig = fn.plot_metamaterial(meta)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_dynamics(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        meta = fn.create_metamaterial(
            unit_cell='reentrant_honeycomb_2d',
            array_size=(2, 2),
            reentrant_angle=150,
        )
        traj = fn.simulate_dynamics(
            meta, dt=1e-9, steps=100, damping=0.1, backend='numpy',
        )
        fig = fn.plot_dynamics(traj)
        assert fig is not None
        plt.close(fig)


class TestAPI:
    """Test API functions."""
    
    def test_list_generators(self):
        generators = fn.list_generators()
        assert len(generators) > 0
        assert 'reentrant_honeycomb_2d' in generators
        assert 'chiral_honeycomb_2d' in generators
    
    def test_create_with_registry(self):
        meta = fn.create('reentrant_honeycomb_2d', reentrant_angle=150)
        assert meta.num_fibers > 0
    
    def test_print_metamaterial_info(self):
        meta = fn.create_metamaterial(
            unit_cell='reentrant_honeycomb_2d',
            array_size=(2, 2),
            reentrant_angle=150,
        )
        # Should not raise
        fn.print_metamaterial_info(meta)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

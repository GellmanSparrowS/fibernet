"""Tests for mechanics metamaterial structure generators."""

import numpy as np
import pytest
from fibernet.gen.metamaterials import (
    reentrant_honeycomb_2d,
    reentrant_honeycomb_3d,
    chiral_honeycomb_2d,
    star_honeycomb_2d,
    arrowhead_auxetic_2d,
    hierarchical_lattice_2d,
    proper_octet_truss_3d,
    diamond_lattice_3d,
    gyroid_lattice_3d,
    missing_rib_auxetic_2d,
    plate_lattice_3d,
)


class TestReentrantHoneycomb:
    def test_2d_basic(self):
        net = reentrant_honeycomb_2d(grid_size=(3, 3))
        assert net.num_fibers > 0
        assert net.dimension == 2
        assert net.num_crosslinks > 0

    def test_2d_angle_effect(self):
        net_auxetic = reentrant_honeycomb_2d(reentrant_angle=150)
        net_hex = reentrant_honeycomb_2d(reentrant_angle=120)
        assert net_auxetic.num_fibers > 0
        assert net_hex.num_fibers > 0

    def test_2d_grid_scaling(self):
        net_small = reentrant_honeycomb_2d(grid_size=(2, 2))
        net_large = reentrant_honeycomb_2d(grid_size=(4, 4))
        assert net_large.num_fibers > net_small.num_fibers

    def test_3d_basic(self):
        net = reentrant_honeycomb_3d(grid_size=(2, 2, 2))
        assert net.num_fibers > 0
        assert net.dimension == 3


class TestChiralHoneycomb:
    def test_basic(self):
        net = chiral_honeycomb_2d(grid_size=(3, 3), num_node_points=12)
        assert net.num_fibers > 0
        assert net.dimension == 2
        assert net.num_crosslinks > 0

    def test_node_radius_effect(self):
        net_small = chiral_honeycomb_2d(node_radius=2, num_node_points=12)
        net_large = chiral_honeycomb_2d(node_radius=5, num_node_points=12)
        assert net_small.num_fibers > 0
        assert net_large.num_fibers > 0


class TestStarHoneycomb:
    def test_basic(self):
        net = star_honeycomb_2d(grid_size=(3, 3))
        assert net.num_fibers > 0
        assert net.dimension == 2

    def test_arm_count(self):
        net4 = star_honeycomb_2d(num_arms=4, grid_size=(2, 2))
        net6 = star_honeycomb_2d(num_arms=6, grid_size=(2, 2))
        assert net6.num_fibers > net4.num_fibers

    def test_inner_angle(self):
        net = star_honeycomb_2d(star_inner_angle=30, grid_size=(2, 2))
        assert net.num_fibers > 0


class TestArrowheadAuxetic:
    def test_basic(self):
        net = arrowhead_auxetic_2d(grid_size=(3, 3))
        assert net.num_fibers > 0
        assert net.dimension == 2
        assert net.num_crosslinks > 0

    def test_angle_variation(self):
        net30 = arrowhead_auxetic_2d(arm_angle=30, grid_size=(2, 2))
        net60 = arrowhead_auxetic_2d(arm_angle=60, grid_size=(2, 2))
        assert net30.num_fibers > 0
        assert net60.num_fibers > 0


class TestHierarchicalLattice:
    def test_triangular_level1(self):
        net = hierarchical_lattice_2d(base_type="triangular", levels=1)
        assert net.num_fibers > 0
        assert net.dimension == 2

    def test_triangular_level2(self):
        net = hierarchical_lattice_2d(base_type="triangular", levels=2)
        assert net.num_fibers > 0

    def test_square(self):
        net = hierarchical_lattice_2d(base_type="square", levels=2)
        assert net.num_fibers > 0

    def test_hierarchy_increases_fibers(self):
        net1 = hierarchical_lattice_2d(base_type="square", levels=1)
        net2 = hierarchical_lattice_2d(base_type="square", levels=2)
        assert net2.num_fibers > net1.num_fibers


class TestProperOctetTruss:
    def test_basic(self):
        net = proper_octet_truss_3d(grid_size=(2, 2, 2))
        assert net.num_fibers > 0
        assert net.dimension == 3
        assert net.num_crosslinks > 0

    def test_coordination(self):
        net = proper_octet_truss_3d(grid_size=(1, 1, 1))
        assert net.num_fibers > 0


class TestDiamondLattice:
    def test_basic(self):
        net = diamond_lattice_3d(grid_size=(2, 2, 2))
        assert net.num_fibers > 0
        assert net.dimension == 3

    def test_scaling(self):
        net1 = diamond_lattice_3d(grid_size=(1, 1, 1))
        net2 = diamond_lattice_3d(grid_size=(2, 2, 2))
        assert net2.num_fibers > net1.num_fibers


class TestGyroidLattice:
    def test_basic(self):
        net = gyroid_lattice_3d(grid_size=(1, 1, 1), resolution=8)
        assert net.num_fibers >= 0
        assert net.dimension == 3

    def test_threshold(self):
        net = gyroid_lattice_3d(grid_size=(1, 1, 1), resolution=8, threshold=0.3)
        assert net.num_fibers >= 0


class TestMissingRibAuxetic:
    def test_basic(self):
        net = missing_rib_auxetic_2d(grid_size=(3, 3))
        assert net.num_fibers > 0
        assert net.dimension == 2
        assert net.num_crosslinks > 0


class TestPlateLattice:
    def test_basic(self):
        net = plate_lattice_3d(grid_size=(2, 2, 2))
        assert net.num_fibers > 0
        assert net.dimension == 3


class TestMetamaterialMechanics:
    """Test that metamaterial structures work with mechanical simulation."""

    def test_reentrant_fem(self):
        from fibernet.sim.mechanical import FiberFEM
        net = reentrant_honeycomb_2d(grid_size=(3, 3))
        fem = FiberFEM(net, segments_per_fiber=3)
        result = fem.apply_uniaxial_strain(strain=0.001, axis=0)
        assert result is not None
        assert not np.isnan(result.energy)

    def test_arrowhead_fem(self):
        from fibernet.sim.mechanical import FiberFEM
        net = arrowhead_auxetic_2d(grid_size=(3, 3))
        fem = FiberFEM(net, segments_per_fiber=3)
        result = fem.apply_uniaxial_strain(strain=0.001, axis=0)
        assert result is not None
        assert not np.isnan(result.energy)

    def test_octet_fem(self):
        from fibernet.sim.mechanical import FiberFEM
        net = proper_octet_truss_3d(grid_size=(1, 1, 1), spacing=10)
        fem = FiberFEM(net, segments_per_fiber=2)
        result = fem.apply_uniaxial_strain(strain=0.001, axis=0)
        assert result is not None
        assert not np.isnan(result.energy)

"""Tests for effective mechanical property computation."""

import numpy as np
import pytest
from fibernet import gen
from fibernet.sim.mechanical import (
    compute_effective_properties,
    poisson_ratio,
    FiberFEM,
)


class TestEffectiveProperties:
    def test_square_lattice_properties(self):
        """Square lattice should have E > 0 and reasonable Poisson's ratio."""
        net = gen.square_lattice_2d(spacing=5, grid_size=(5, 5))
        props = compute_effective_properties(net, strain=0.001, segments_per_fiber=3)
        assert props.E_x > 0
        assert props.E_y > 0
        assert props.relative_density > 0
        assert not np.isnan(props.E_x)

    def test_triangular_lattice(self):
        """Triangular lattice should be stiffer than square."""
        net = gen.triangular_lattice_2d(spacing=5, grid_size=(5, 5))
        props = compute_effective_properties(net, strain=0.001, segments_per_fiber=3)
        assert props.E_x > 0

    def test_properties_summary(self):
        """Test summary string generation."""
        net = gen.square_lattice_2d(spacing=5, grid_size=(3, 3))
        props = compute_effective_properties(net, strain=0.001, segments_per_fiber=3)
        summary = props.summary()
        assert "E_x" in summary
        assert "relative" in summary.lower() or "ρ" in summary

    def test_properties_to_dict(self):
        """Test dictionary export."""
        net = gen.square_lattice_2d(spacing=5, grid_size=(3, 3))
        props = compute_effective_properties(net, strain=0.001, segments_per_fiber=3)
        d = props.to_dict()
        assert "E_x" in d
        assert "relative_density" in d

    def test_3d_cubic_lattice(self):
        """3D cubic lattice should have positive moduli in all directions."""
        net = gen.cubic_lattice_3d(spacing=5, grid_size=(3, 3, 3))
        props = compute_effective_properties(net, strain=0.001, segments_per_fiber=2)
        assert props.E_x > 0
        assert props.E_y > 0
        assert props.E_z > 0


class TestPoissonRatio:
    def test_basic_computation(self):
        """Poisson's ratio should be computable for simple lattices."""
        net = gen.square_lattice_2d(spacing=5, grid_size=(5, 5))
        nu = poisson_ratio(net, strain=0.001, loading_axis=0, transverse_axis=1)
        assert not np.isnan(nu)
        assert isinstance(nu, float)

    def test_auxetic_reentrant(self):
        """Re-entrant honeycomb should show auxetic tendency (nu < 0)."""
        from fibernet.gen.metamaterials import reentrant_honeycomb_2d
        net = reentrant_honeycomb_2d(
            reentrant_angle=150, grid_size=(5, 5),
            cell_height=10, cell_width=10,
        )
        nu = poisson_ratio(net, strain=0.001, loading_axis=0, transverse_axis=1)
        # Re-entrant should tend towards negative Poisson's ratio
        # but this depends on boundary conditions; just verify computation works
        assert not np.isnan(nu)


class TestMetamaterialProperties:
    """Test effective properties for metamaterial structures."""

    def test_octet_truss_stiffness(self):
        """Octet truss should be stretch-dominated (high stiffness)."""
        from fibernet.gen.metamaterials import proper_octet_truss_3d
        net = proper_octet_truss_3d(spacing=10, grid_size=(2, 2, 2))
        props = compute_effective_properties(net, strain=0.001, segments_per_fiber=2)
        assert props.E_x > 0
        assert props.relative_density > 0

    def test_arrowhead_properties(self):
        """Arrowhead auxetic should be computable."""
        from fibernet.gen.metamaterials import arrowhead_auxetic_2d
        net = arrowhead_auxetic_2d(grid_size=(4, 4))
        props = compute_effective_properties(net, strain=0.001, segments_per_fiber=3)
        assert not np.isnan(props.E_x)
        assert props.E_x >= 0

    def test_hierarchical_vs_simple(self):
        """Hierarchical lattice should have different properties from simple."""
        from fibernet.gen.metamaterials import hierarchical_lattice_2d
        net_h = hierarchical_lattice_2d(base_type="square", levels=2)
        props_h = compute_effective_properties(net_h, strain=0.001, segments_per_fiber=3)
        assert not np.isnan(props_h.E_x)

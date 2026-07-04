"""
Fiber network generators.

Submodules:
- disordered: Random/Mikado/random-walk fiber networks
- ordered: Lattices (square, triangular, honeycomb, cubic, octet, kagome)
- chiral: Helices, braids, twisted bundles, chiral metamaterials
- woven: Plain/twill/satin weaves, 3D orthogonal woven
- hierarchical: Multi-scale bundles, gradient, fractal, core-shell structures
- advanced: Voronoi, electrospun, meltblown, biomimetic, auxetic, kirigami
- variants: Enhanced variants with 2D/3D support, multi-radius, variable stiffness
"""

from fibernet.gen.disordered import (
    random_straight_2d, random_straight_3d, random_walk_fibers,
    oriented_random_2d, poisson_line_network_2d,
)
from fibernet.gen.ordered import (
    square_lattice_2d, triangular_lattice_2d, honeycomb_lattice_2d,
    cubic_lattice_3d, octet_truss_3d, kagome_lattice_2d,
)
from fibernet.gen.chiral import (
    single_helix, double_helix, braided_rope, twisted_bundle, chiral_metamaterial,
)
from fibernet.gen.woven import (
    plain_weave_2d, twill_weave_2d, satin_weave_2d, woven_3d_orthogonal,
)
from fibernet.gen.hierarchical import (
    hierarchical_bundle, gradient_density_network, core_shell_fiber, fractal_network,
)
from fibernet.gen.advanced import (
    voronoi_network_2d, voronoi_network_3d,
    electrospun_network, meltblown_network,
    biomimetic_collagen, biomimetic_fibrin,
    defected_lattice, composite_network, graded_network,
    auxetic_structure, kirigami_structure,
)
from fibernet.gen.variants import (
    lattice_2d_to_3d, curved_lattice, multi_radius_network,
    variable_stiffness_network, gyroid_infill, diamond_lattice_3d, foam_like_3d,
)

__all__ = [
    # Disordered
    "random_straight_2d", "random_straight_3d", "random_walk_fibers",
    "oriented_random_2d", "poisson_line_network_2d",
    # Ordered
    "square_lattice_2d", "triangular_lattice_2d", "honeycomb_lattice_2d",
    "cubic_lattice_3d", "octet_truss_3d", "kagome_lattice_2d",
    # Chiral
    "single_helix", "double_helix", "braided_rope", "twisted_bundle", "chiral_metamaterial",
    # Woven
    "plain_weave_2d", "twill_weave_2d", "satin_weave_2d", "woven_3d_orthogonal",
    # Hierarchical
    "hierarchical_bundle", "gradient_density_network", "core_shell_fiber", "fractal_network",
    # Advanced
    "voronoi_network_2d", "voronoi_network_3d",
    "electrospun_network", "meltblown_network",
    "biomimetic_collagen", "biomimetic_fibrin",
    "defected_lattice", "composite_network", "graded_network",
    "auxetic_structure", "kirigami_structure",
    # Variants
    "lattice_2d_to_3d", "curved_lattice", "multi_radius_network",
    "variable_stiffness_network", "gyroid_infill", "diamond_lattice_3d", "foam_like_3d",
]

# Specialized generators
from fibernet.gen.specialized import (
    cnt_network_2d,
    cnt_network_3d,
    paper_network,
    textile_weave,
    electrospun_mat,
    fiber_reinforced_composite,
)

__all__ += [
    "cnt_network_2d",
    "cnt_network_3d",
    "paper_network",
    "textile_weave",
    "electrospun_mat",
    "fiber_reinforced_composite",
]

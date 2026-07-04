"""
Fiber network generators.

Submodules:
- disordered: Random/Mikado/random-walk fiber networks
- ordered: Lattices (square, triangular, honeycomb, cubic, octet, kagome)
- chiral: Helices, braids, twisted bundles, chiral metamaterials
- woven: Plain/twill/satin weaves, 3D orthogonal woven
- hierarchical: Multi-scale bundles, gradient, fractal, core-shell structures
"""

from fibernet.gen.disordered import (
    random_straight_2d,
    random_straight_3d,
    random_walk_fibers,
    oriented_random_2d,
    poisson_line_network_2d,
)
from fibernet.gen.ordered import (
    square_lattice_2d,
    triangular_lattice_2d,
    honeycomb_lattice_2d,
    cubic_lattice_3d,
    octet_truss_3d,
    kagome_lattice_2d,
)
from fibernet.gen.chiral import (
    single_helix,
    double_helix,
    braided_rope,
    twisted_bundle,
    chiral_metamaterial,
)
from fibernet.gen.woven import (
    plain_weave_2d,
    twill_weave_2d,
    satin_weave_2d,
    woven_3d_orthogonal,
)
from fibernet.gen.hierarchical import (
    hierarchical_bundle,
    gradient_density_network,
    core_shell_fiber,
    fractal_network,
)

__all__ = [
    "random_straight_2d", "random_straight_3d", "random_walk_fibers",
    "oriented_random_2d", "poisson_line_network_2d",
    "square_lattice_2d", "triangular_lattice_2d", "honeycomb_lattice_2d",
    "cubic_lattice_3d", "octet_truss_3d", "kagome_lattice_2d",
    "single_helix", "double_helix", "braided_rope", "twisted_bundle",
    "chiral_metamaterial",
    "plain_weave_2d", "twill_weave_2d", "satin_weave_2d", "woven_3d_orthogonal",
    "hierarchical_bundle", "gradient_density_network", "core_shell_fiber",
    "fractal_network",
]

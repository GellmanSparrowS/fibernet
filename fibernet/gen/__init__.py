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
    oriented_random_2d, oriented_random_3d,
    poisson_line_network_2d, random_curved_fibers_3d,
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
    "oriented_random_2d", "oriented_random_3d",
    "poisson_line_network_2d", "random_curved_fibers_3d",
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

# Fractal networks
from .fractal import (
    sierpinski_triangle,
    koch_curve,
    fractal_tree,
    hilbert_curve,
)

__all__.extend([
    "sierpinski_triangle",
    "koch_curve",
    "fractal_tree",
    "hilbert_curve",
])

# Gradient networks
from .gradient import (
    density_gradient_2d,
    property_gradient_2d,
    multi_zone_2d,
)

__all__.extend([
    "density_gradient_2d",
    "property_gradient_2d",
    "multi_zone_2d",
])

# Fiber bundles
from .bundles import (
    parallel_bundle_2d,
    twisted_bundle_2d,
    random_bundle_3d,
    braided_bundle_3d,
    tendon_like_bundle_3d,
)

__all__.extend([
    "parallel_bundle_2d",
    "twisted_bundle_2d",
    "random_bundle_3d",
    "braided_bundle_3d",
    "tendon_like_bundle_3d",
])

# Curved fibers
from .curved import (
    sinusoidal_fiber_2d,
    helical_fiber_3d,
    arc_fiber_2d,
    bezier_fiber_3d,
    random_curved_network_3d,
    crimped_network_2d,
)

__all__.extend([
    "sinusoidal_fiber_2d",
    "helical_fiber_3d",
    "arc_fiber_2d",
    "bezier_fiber_3d",
    "random_curved_network_3d",
    "crimped_network_2d",
])

# Composite laminates
from .laminates import (
    unidirectional_laminate,
    crossply_laminate,
    angle_ply_laminate,
    quasi_isotropic_laminate,
    custom_laminate,
    sandwich_laminate,
)

__all__.extend([
    "unidirectional_laminate",
    "crossply_laminate",
    "angle_ply_laminate",
    "quasi_isotropic_laminate",
    "custom_laminate",
    "sandwich_laminate",
])

# Metamaterial structures (mechanics design)
from .metamaterials import (
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

__all__.extend([
    "reentrant_honeycomb_2d",
    "reentrant_honeycomb_3d",
    "chiral_honeycomb_2d",
    "star_honeycomb_2d",
    "arrowhead_auxetic_2d",
    "hierarchical_lattice_2d",
    "proper_octet_truss_3d",
    "diamond_lattice_3d",
    "gyroid_lattice_3d",
    "missing_rib_auxetic_2d",
    "plate_lattice_3d",
])

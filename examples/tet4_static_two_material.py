"""Force-controlled two-material Tet4 cube with an analytical reference.

Run from an installed or editable FiberNet environment:
    python examples/tet4_static_two_material.py
"""
from itertools import combinations, permutations, product

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4 import solve_static_tet4


def build_example(layer_axis=0):
    """Return a fitted 12-Tet4 mesh, nodal forces (N) and Cartesian holds."""
    coordinates = [[0., 1.] for _ in range(3)]
    coordinates[layer_axis] = [0., .5, 1.]
    dims = tuple(map(len, coordinates))
    points = np.array(list(product(*coordinates)))
    paths = [np.vstack((np.zeros(3, dtype=int), np.cumsum(np.eye(3, dtype=int)[list(p)], axis=0)))
             for p in permutations(range(3))]
    soft, hard = 71, 2**64-1
    materials = {soft: {'young_modulus_pa': 1e6, 'poisson_ratio': 0.},
                 hard: {'young_modulus_pa': 2e6, 'poisson_ratio': 0.}}
    cells, phases = [], []
    for start in product(*(range(d-1) for d in dims)):
        for path in paths:
            cells.append(np.ravel_multi_index((path + start).T, dims))
            phases.append((soft, hard)[start[layer_axis]])
    cells = np.asarray(cells, dtype=np.int64)
    ids = np.arange(len(points), dtype=np.uint64)*73 + np.uint64(2**63 + 3)
    phases = np.array(phases, dtype=np.uint64)
    bundle = Tet4Bundle(points, cells, ids, np.arange(len(cells)) + 100, phases, materials, 'mm')

    # Constant traction 1000 Pa for series, proportional to E for parallel;
    # both integrate to 1 mN over the x=1 mm end cap.
    vector = np.zeros(points.shape)
    for cell, phase in zip(cells, phases):
        for face in combinations(cell, 3):
            if not np.all(points[list(face), 0] == 1.):
                continue
            a, b, c = points[list(face)] * 1e-3
            area = np.linalg.norm(np.cross(b - a, c - a)) / 2.
            pressure = 1000.
            if layer_axis == 1:
                pressure *= materials[int(phase)]['young_modulus_pa'] / 1.5e6
            vector[list(face), 0] += pressure * area / 3.
    loads = {int(k): f for k, f in zip(ids, vector) if np.any(f)}
    held = {}
    for nid, p in zip(ids, points):
        if tuple(p) == (0., 0., 0.):
            held[int(nid)] = ('x', 'y', 'z')
        elif tuple(p) == (0., 1., 0.):
            held[int(nid)] = ('x', 'z')
        elif p[0] == 0.:
            held[int(nid)] = ('x',)
    return bundle, loads, held


def main():
    bundle, loads, held = build_example()
    result = solve_static_tet4(bundle, forces_by_id=loads, fixed_dofs_by_id=held)
    right = bundle.points[:, 0] == 1.
    delta = float(np.mean(result.displacement_m[right, 0]))
    reaction = float(result.reaction_N[:, 0].sum())
    # Series: 0.5 mm of each material under 1000 Pa.
    expected = 1000. * (0.5e-3/1e6 + 0.5e-3/2e6)
    np.testing.assert_allclose([delta, reaction, result.strain_energy_J],
                               [expected, -1e-3, .5e-3*expected], rtol=2e-12)
    print(f'12 tetrahedra, 1 mN: displacement {delta*1e6:.3f} um, '
          f'reaction {reaction*1e3:.3f} mN, energy {result.strain_energy_J:.6e} J')


if __name__ == '__main__':
    main()

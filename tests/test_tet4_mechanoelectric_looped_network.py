# SPDX-License-Identifier: MIT
from collections import Counter, defaultdict, deque
from itertools import combinations

import numpy as np

from examples import tet4_mechanoelectric_looped_network as demo


def test_looped_solid_has_one_manifold_and_three_terminal_paths():
    points, cells = demo._lattice()
    assert (len(points), len(cells)) == (387, 1056)
    faces = defaultdict(list)
    for owner, cell in enumerate(cells):
        for face in combinations(sorted(cell), 3):
            faces[face].append(owner)
    neighbors = defaultdict(set)
    rim = Counter()
    for face, owners in faces.items():
        assert len(owners) in (1, 2)
        if len(owners) == 2:
            a, b = owners
            neighbors[a].add(b)
            neighbors[b].add(a)
        else:
            rim.update(combinations(face, 2))
    assert set(rim.values()) == {2}
    pending, seen = deque([0]), {0}
    while pending:
        new = neighbors[pending.popleft()] - seen
        seen.update(new)
        pending.extend(new)
    assert len(seen) == len(cells)
    # Each pair is an uninterrupted rail across all 13 axial planes.
    for y, z in ((.5, .5), (1.5, .5), (.5, 1.5)):
        np.testing.assert_array_equal(points[(points[:, 1] == y) & (points[:, 2] == z), 0],
                                      np.arange(13) * .25)


def test_stiffness_changes_force_not_unstrained_electrical_geometry():
    cases = demo.run()
    low, high, half = (cases[key] for key in demo.MODES)
    assert 0.8 < low.load_reaction_N < 0.9
    assert low.load_reaction_N < half.load_reaction_N < high.load_reaction_N
    np.testing.assert_allclose(high.load_reaction_N, 2 * low.load_reaction_N, rtol=1e-8)
    np.testing.assert_allclose(high.resistance_slope_per_strain,
                               low.resistance_slope_per_strain, rtol=1e-8)
    np.testing.assert_allclose(half.rest_resistance_ohm, low.rest_resistance_ohm,
                               rtol=1e-10)
    for record in cases.values():
        assert 3500 < record.rest_resistance_ohm < 3800
        assert 1.45 < record.resistance_slope_per_strain < 1.65


def test_centered_readout_converges_at_halved_strain_increment():
    points, cells = demo._lattice()
    bundle = demo._bundle(points, cells, "uniform")
    rest, _ = demo._scan(bundle, 0.)
    h = demo.STRAIN
    small_plus, _ = demo._scan(bundle, h / 2)
    small_minus, _ = demo._scan(bundle, -h / 2)
    large_plus, _ = demo._scan(bundle, h)
    large_minus, _ = demo._scan(bundle, -h)
    np.testing.assert_allclose((small_plus - small_minus) / (h * rest),
                               (large_plus - large_minus) / (2 * h * rest),
                               rtol=2e-6)

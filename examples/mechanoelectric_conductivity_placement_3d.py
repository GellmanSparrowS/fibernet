"""Compare conductivity placements at fixed mechanics on a 360-rod network."""
import argparse
from contextlib import redirect_stdout
from math import fsum
import json
import sys

import numpy as np

if __package__:
    from .mechanoelectric_network_3d import build_sample, _dc_state, _length
else:
    from mechanoelectric_network_3d import build_sample, _dc_state, _length


def run_example(phase=0):
    """Run one deterministic distorted BCC geometry at +0.5% grip extension."""
    import fibernet as fn

    if type(phase) is not int or phase not in range(4):
        raise ValueError("phase must be an integer from 0 through 3")
    graph, nids, eids, nominal, ends, electrodes, _, radii = build_sample()
    # Realize nanoscale-exact integer geometry first, then convert to SI.
    cells = np.rint(nominal * 1000).astype(np.int64)
    corners_nm = cells * 1_000_000
    internal = (cells[:, 0] > 0) & (cells[:, 0] < 6)
    corners_nm[internal, 1] += ((cells[internal, 0] + cells[internal, 2] + phase) % 4 * 2 - 3) * 20_000
    corners_nm[internal, 2] += ((cells[internal, 0] + cells[internal, 1] + 2 * phase) % 4 * 2 - 3) * 20_000
    x = corners_nm.astype(float) * 1.e-9
    graph.set_node_positions(dict(zip(nids, x)))
    blocks = {}
    for idx, v in enumerate(corners_nm[ends[:, 1]] - corners_nm[ends[:, 0]]):
        sign = 1 if next(k for k in v if k) > 0 else -1
        blocks.setdefault((tuple(sign*v), radii[idx]), []).append(idx)

    def put_two_values(order, low, high):
        result = np.full(len(eids), low)
        for block in blocks.values():
            ranked = sorted(block, key=lambda j: (order[j], eids[j]))
            result[ranked[:len(block)//2]] = high
        return result

    modulus = put_two_values(eids, 1.e9, 2.e9)
    for eid, stiffness in zip(eids, modulus):
        graph.edges[eid].material = fn.Material(
            youngs_modulus=float(stiffness), poissons_ratio=.3)
    strain = .005
    solution = fn.simulate(
        graph, backend="fem", nonlinear=False, material_mode="graph",
        length_unit="m", stretch_ratio=1.+strain, pct=16.5/len(nids))
    by_id = dict(zip(solution.metadata["node_ids"], solution.deformed_positions))
    if len(by_id) != len(nids) or set(by_id) != set(nids):
        raise RuntimeError("Mechanical node identities changed")
    y = np.asarray([by_id[j] for j in nids])
    movement = np.zeros_like(x)
    hi = corners_nm[:, 0] == 6_000_000
    lo = corners_nm[:, 0] == 0
    movement[hi, 0] = strain * .006
    np.testing.assert_allclose(y[lo|hi]-x[lo|hi], movement[lo|hi], rtol=0, atol=3e-14)
    L = _length(x, ends)
    area = np.pi*radii**2
    length_ratio = L / _length(y, ends)
    transverse = electrodes[1]

    def state(position, sigma, geometry_factor):
        dc_graph = graph.copy()
        dc_graph.set_node_positions(dict(zip(nids, position)))
        resistance, dc = _dc_state(dc_graph, eids, sigma*area/L*geometry_factor, transverse)
        row_by_id = {int(eid): j for j, eid in enumerate(dc.edge_ids)}
        rows = [row_by_id[eid] for eid in eids]
        expected = np.asarray([(nids[i], nids[j]) for i, j in ends])
        np.testing.assert_array_equal(dc.endpoints_by_id[rows], expected)
        # Normalize to unit terminal current and divide per-edge power by total.
        weight = resistance * dc.current_A[rows]**2 / dc.conductance_S[rows]
        np.testing.assert_allclose(fsum(weight), 1., rtol=2e-12)
        return resistance, weight

    uniform = np.full(len(eids), 1.5)
    r0, w0 = state(x, uniform, 1.)
    r1, w1 = state(y, uniform, length_ratio)
    gradient = (r1/r0)*(w0-w1)/strain
    np.testing.assert_allclose(fsum(gradient), 0., rtol=0., atol=2e-11)
    by_placement = {}
    for name, ordering in (("low", gradient), ("stable_id", eids), ("high", -gradient)):
        sigma = put_two_values(ordering, 1., 2.)
        reference, _ = state(x, sigma, 1.)
        geometries = {}
        for law, factor in (("fixed_area", length_ratio),
                            ("fixed_member_volume", length_ratio**2)):
            loaded, _ = state(y, sigma, factor)
            geometries[law] = {"loaded_ohm": loaded,
                               "change_percent": 100.*(loaded/reference-1.)}
        by_placement[name] = {
            "unloaded_ohm": reference, "conductive_member_count": int(sum(sigma==2.)),
            "higher_conductivity_nominal_volume_m3": float(np.dot(area*L, sigma==2.)),
            "geometry_laws": geometries,
        }

    for key in ("conductive_member_count", "higher_conductivity_nominal_volume_m3"):
        np.testing.assert_allclose([v[key] for v in by_placement.values()],
                                   by_placement["stable_id"][key], rtol=8e-14, atol=0.)
    return {"phase": phase, "member_count": len(eids), "matched_group_count": len(blocks),
            "input_strain": strain, "voltage_readout_axis": "y",
            "low_high_conductivity_S_per_m": [1., 2.], "placements": by_placement}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        report = run_example(args.phase)
    print(json.dumps(report, indent=2, allow_nan=False))

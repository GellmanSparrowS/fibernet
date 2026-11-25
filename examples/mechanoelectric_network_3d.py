"""Stretch a complete 3D beam network and recalculate its DC resistance."""
import argparse
import json
import sys
from contextlib import redirect_stdout
from math import fsum

import numpy as np


def build_sample():
    import fibernet as fn

    graph = fn.pattern_3d(unit="bcc", box=(.002,) * 3, grid=(3, 3, 3), radius=.0001)
    node_ids = tuple(graph.nodes)
    edge_ids = tuple(graph.edges)
    if (len(node_ids), len(edge_ids)) != (91, 360):
        raise RuntimeError("Unexpected BCC sample topology")
    original = np.asarray([graph.nodes[n].position for n in node_ids], dtype=float)
    x = original.copy()
    # Deterministic mild disorder; keep the two mechanical grip faces planar.
    inside = (x[:, 0] != 0.) & (x[:, 0] != .006)
    wave = np.sin(np.asarray(node_ids)[:, None] * [.81, .63, 1.2] + [.2, .4, .6])
    x[inside] += .00008 * wave[inside]
    graph.set_node_positions(dict(zip(node_ids, x)))
    row_by_id = {nid: i for i, nid in enumerate(node_ids)}
    endpoints = []
    sigma, radius = [], []
    for i, eid in enumerate(edge_ids):
        edge = graph.edges[eid]
        edge.material = fn.Material(youngs_modulus=(1e9, 2e9)[i % 2], poissons_ratio=.3)
        edge.radius = (.000125, .000075)[(i // 3) % 2]
        endpoints.append((row_by_id[edge.node_i], row_by_id[edge.node_j]))
        sigma.append((1., 2.)[i % 2])
        radius.append(edge.radius)
    # Electrodes are material/ID tags from the manufactured sample, so they
    # move with the specimen even when their nodes are no longer coplanar.
    electrodes = [{nid: float(original[row, axis] == .006)
                   for row, nid in enumerate(node_ids)
                   if original[row, axis] in (0., .006)} for axis in range(3)]
    if any(len(terminals) != 32 for terminals in electrodes):
        raise RuntimeError("Expected 16 electrode nodes per face")
    return graph, node_ids, edge_ids, x, np.asarray(endpoints), electrodes, np.asarray(sigma), np.asarray(radius)


def _length(points, endpoints):
    return np.linalg.norm(points[endpoints[:, 1]] - points[endpoints[:, 0]], axis=1)


def _resistance(graph, edge_ids, conductance, electrodes):
    from fibernet.sim.linear_transport import solve_linear_transport

    e = solve_linear_transport(graph, dict(zip(edge_ids, conductance)), electrodes, length_unit="m")
    injected = fsum(float(j) for n, j in zip(e.node_ids, e.node_injection_A)
                    if electrodes.get(int(n)) == 1.)
    if injected <= 0 or not np.isclose(injected, e.dissipated_power_W, rtol=1e-10, atol=0.):
        raise RuntimeError("One-volt contact current does not reconcile with Joule work")
    return 1. / injected


def run_example(strain=.005):
    import fibernet as fn

    if not np.isfinite(strain) or not -.01 <= strain <= .01:
        raise ValueError("the illustrative linear axial strain must be in [-0.01, 0.01]")
    graph, nids, eids, x, ends, electrodes, sigma, radius = build_sample()
    a0 = np.pi * radius**2
    l0 = _length(x, ends)
    reference = sigma * a0 / l0
    base = [_resistance(graph, eids, reference, el) for el in electrodes]
    s = fn.simulate(graph, backend="fem", nonlinear=False, material_mode="graph",
                    length_unit="m", stretch_ratio=1.+strain, pct=16.5/len(nids))
    positions = dict(zip(s.metadata["node_ids"], s.deformed_positions))
    if set(positions) != set(nids) or len(s.deformed_positions) != len(nids):
        raise RuntimeError("Displacement rows do not match the source node IDs")
    y = np.asarray([positions[nid] for nid in nids])
    shift = np.array([.006 * strain, 0., 0.])
    np.testing.assert_allclose(y[x[:, 0] == 0.], x[x[:, 0] == 0.], rtol=0., atol=3e-14)
    np.testing.assert_allclose(y[x[:, 0] == .006], x[x[:, 0] == .006] + shift, rtol=0., atol=3e-14)
    derived = graph.copy()
    derived.set_node_positions(positions)
    l1 = _length(y, ends)
    ratio = l0 / l1
    area_laws = {"fixed_area": ratio, "fixed_member_volume": ratio**2}
    directions = {}
    for j, label in enumerate("xyz"):
        cases = {}
        frozen_R = _resistance(derived, eids, reference, electrodes[j])
        np.testing.assert_allclose(frozen_R, base[j], rtol=1e-12)
        for law, factor in area_laws.items():
            dc_R = _resistance(derived, eids, reference * factor, electrodes[j])
            cases[law] = {"ohm": dc_R, "change_percent": 100.*(dc_R/base[j]-1.)}
        directions[label] = {"baseline_ohm": base[j], **cases}
    return {
        "model": "linear welded beam mechanics; DC geometric conductance update; fixed topology",
        "node_count": len(nids), "member_count": len(eids), "axial_strain": strain,
        "mechanical_energy_J": float(s.energy), "directions": directions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strain", type=float, default=.005)
    args = parser.parse_args()
    # Keep stdout as one JSON document even when an optional dependency emits
    # an import banner. Do not hide diagnostics or change library-wide logging.
    with redirect_stdout(sys.stderr):
        result = run_example(args.strain)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))

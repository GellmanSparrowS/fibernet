"""Build and split five synthetic beam responses without training a model."""

import argparse
from hashlib import sha256
import json
from math import fsum, pi
from pathlib import Path

import fibernet as fn
import numpy as np
from fibernet.ml.beam_frame_fem import BeamFrameFEM
from fibernet.ml.dataset import DatasetSample, FiberNetDataset


CASES = (
    ("A", (2, 1, 1), 2e9), ("A", (2, 1, 1), 4e9),
    ("B", (2, 2, 1), 2e9), ("B", (2, 2, 1), 4e9),
    ("C", (3, 1, 1), 4e9),
)
ROLES = ("train", "validation", "test")
LENGTH_M, RADIUS_M, STRETCH = .002, 8e-5, 1.001


def solve_case(grid, stiff_pa):
    """Build a cubic grid and independently check its active axial paths."""
    nx, ny, nz = grid
    paths = (ny + 1) * (nz + 1)
    g = fn.pattern_3d(unit="cubic", box=(LENGTH_M,) * 3, grid=grid, radius=RADIUS_M)
    order = list(g.nodes)
    if len(order) != (nx + 1) * paths:
        raise ValueError("unexpected cubic generator geometry")
    x = {i: g.nodes[i].position[0] for i in order}
    axial = {i for i, e in g.edges.items() if x[e.node_i] != x[e.node_j]}
    cut = {i for i in axial if min(x[g.edges[i].node_i], x[g.edges[i].node_j]) == 0.}
    if len(axial) != nx * paths or len(cut) != paths:
        raise ValueError("unexpected axial path topology")
    by_id = {i: 1e9 if i in cut else stiff_pa for i in g.edges}
    for i, edge in g.edges.items():
        edge.material = fn.Material(youngs_modulus=by_id[i], poissons_ratio=.30)

    # Ask for exactly one end-plate's row count and verify the chosen sets.
    pct = (paths + .25) / len(order)
    setup = BeamFrameFEM.graph_to_fem_input(g, dim=3, pct=pct)
    for side, cap_x in (("left", 0.), ("right", nx * LENGTH_M)):
        selected = {order[i] for i in setup["boundaries"][side]}
        if selected != {i for i in order if np.isclose(x[i], cap_x, rtol=0., atol=1e-14)}:
            raise ValueError("fractional grips did not select the expected caps")
    r = fn.simulate(
        g, backend="fem", stretch_ratio=STRETCH, nonlinear=False,
        material_mode="graph", length_unit="m", pct=pct,
    )
    ids = r.metadata["edge_ids"]
    material = r.metadata["material_contract"]
    if (len(ids) != len(set(ids)) or set(ids) != set(g.edges)
            or material["edge_ids"] != ids or dict(zip(ids, material["youngs_modulus"])) != by_id):
        raise ValueError("incompatible result edge or material IDs")
    stroke = nx * LENGTH_M * (STRETCH - 1.)
    area = pi * RADIUS_M**2
    expected_path = stroke / (LENGTH_M / (1e9 * area) + (nx - 1) * LENGTH_M / (stiff_pa * area))
    np.testing.assert_allclose(
        r.edge_forces, [expected_path if i in axial else 0. for i in ids], rtol=1e-7, atol=1e-12,
    )
    energy = r.metadata["beam_energy"]
    if energy["status"] != "available" or energy["unit_system"] != "SI":
        raise ValueError("expected available SI beam energy")
    np.testing.assert_allclose(energy["strain_energy"], .5 * paths * expected_path * stroke, rtol=1e-8)
    forces = dict(zip(ids, r.edge_forces))
    features = {"beam_count": float(len(g.edges)), "stiff_modulus_Pa": stiff_pa,
                "axial_spans": float(nx), "parallel_paths": float(paths),
                "radius_m": RADIUS_M, "stroke_m": stroke}
    labels = {"cut_force_N": fsum(float(forces[i]) for i in sorted(cut)),
              "strain_energy_J": energy["strain_energy"]}
    return g, r, features, labels


def audit_roles(dataset, splits):
    """Refuse missing, duplicated, or cross-role sample/source identities."""
    expected = {row.sample_id: row.metadata["source_group"] for row in dataset.samples}
    if len(expected) != len(dataset.samples):
        raise ValueError("original sample IDs must be unique")
    seen_sources = set()
    seen_rows = []
    for part in splits:
        if not part.samples:
            raise ValueError("every role must contain at least one sample")
        sources = set()
        for row in part.samples:
            source = row.metadata["source_group"]
            if row.sample_id not in expected or expected[row.sample_id] != source:
                raise ValueError("split row must retain its original source group")
            sources.add(source)
            seen_rows.append(row.sample_id)
        if sources & seen_sources:
            raise ValueError("a source group appears in more than one role")
        seen_sources.update(sources)
    if len(seen_rows) != len(set(seen_rows)) or set(seen_rows) != set(expected):
        raise ValueError("sample IDs must appear exactly once across roles")


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    try:
        import sklearn.model_selection  # noqa: F401 -- optional group splitter
    except ImportError as exc:
        raise ImportError("This example requires scikit-learn for group splitting") from exc
    output.mkdir()
    rows = []
    for group, grid, stiffness in CASES:
        g, r, features, labels = solve_case(grid, stiffness)
        name = f"{group}{int(stiffness / 1e9)}"
        target = output / name
        target.mkdir()
        gp, rp = target / "graph.json", target / "result.json"
        g.save_json(gp)
        r.save(rp, detailed=True)
        lineage = {"source_group": group, "graph_sha256": sha256(gp.read_bytes()).hexdigest(),
                   "result_sha256": sha256(rp.read_bytes()).hexdigest()}
        rows.append(DatasetSample(name, "cubic", grid[:2],
                                  {"grid_xyz": grid, "length_unit": "m", "stretch": STRETCH},
                                  features, labels, lineage))

    data = FiberNetDataset(rows, feature_names=list(rows[0].features), target_names=list(rows[0].labels))
    groups = [row.metadata["source_group"] for row in rows]
    parts = data.split(method="group", groups=groups, ratios=(.4, .3, .3), random_state=42)
    audit_roles(data, parts)
    for role, part in zip(ROLES, parts):
        part.save(output / role)
    return {role: [{"sample_id": row.sample_id, "source_group": row.metadata["source_group"]}
                   for row in part.samples] for role, part in zip(ROLES, parts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.output), indent=2, allow_nan=False))
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

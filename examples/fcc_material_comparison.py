"""Compare material layouts on one FCC beam graph (SI units).

Run from a checkout with FiberNet installed::
    python examples/fcc_material_comparison.py --output fcc-material-results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import fibernet as fn
from fibernet.gen.standard_lattice import octet_fcc_v1
from fibernet.ml.beam_frame_fem import BeamFrameFEM


def make_problem():
    """Return the fixed 123-node, 516-bar graph and force-controlled boundary."""
    graph = octet_fcc_v1(
        a=0.01, grid=(3, 3, 2), radius=0.00025,
        material=fn.Material(youngs_modulus=1e9, poissons_ratio=0.3),
    )
    points, edges, radii = graph.to_numpy()
    node_ids = np.array(sorted(graph.nodes))
    edge_ids = np.array(list(graph.edges))
    expected = np.array([[np.searchsorted(node_ids, graph.edges[int(i)].node_i),
                          np.searchsorted(node_ids, graph.edges[int(i)].node_j)]
                         for i in edge_ids])
    np.testing.assert_array_equal(edges, expected)
    fixed = np.flatnonzero(points[:, 0] == 0)
    loaded = np.flatnonzero(points[:, 0] == points[:, 0].max())
    forces = np.zeros_like(points)
    forces[loaded, 0] = 0.001  # N per node; all six DOFs clamped at x=0.
    return graph, points, edges, radii, node_ids, edge_ids, fixed, loaded, forces


def material_layouts(points, edges, edge_ids):
    """Deterministic initial contrast and a spatially balanced reversal control."""
    delta = points[edges[:, 1]] - points[edges[:, 0]]
    family = np.where(np.isclose(delta[:, 0], 0), "YZ",
                      np.where(np.isclose(delta[:, 1], 0), "XZ", "XY"))
    mid = points[edges].mean(axis=1)
    initial = np.zeros(len(edges), dtype=bool)
    for plane in ("XY", "XZ"):
        rows = np.flatnonzero(family == plane)
        order = np.lexsort((edge_ids[rows], *np.round(mid[rows], 12).T[::-1]))
        ranks = ((2 * np.arange(129) + 1) * len(rows)) // (2 * 129)
        initial[rows[order[ranks]]] = True

    # Quarter-cell x layers and three bands. Exact layers and y/z positions
    # remain different: these are controlled layout examples, not an isolation
    # of direction from position.
    q = np.rint(mid[:, 0] / 0.0025).astype(int)
    np.testing.assert_allclose(q * 0.0025, mid[:, 0], atol=1e-14, rtol=0)
    band = np.digitize(q, [4, 9])
    rows = np.arange(len(edges))

    def spread(selected, count):
        order = np.lexsort((edge_ids[selected],
                            *np.round(mid[selected][:, [0, 2, 1]], 12).T))
        ranks = ((2 * np.arange(count) + 1) * len(selected)) // (2 * count)
        return selected[order[ranks]]

    a, b = np.zeros(len(edges), bool), np.zeros(len(edges), bool)
    for layer, count in ((1, 12), (3, 12), (5, 36), (7, 36), (9, 12), (11, 12)):
        for plane in ("XY", "XZ"):
            a[spread(rows[(q == layer) & (family == plane)], count // 2)] = True
    b[(q % 2 == 0) & (q > 0) & (q < 12)] = True
    common = np.zeros(len(edges), bool)
    for i, count in enumerate((58, 22, 58)):
        common[spread(rows[(band == i) & ~(a | b)], count)] = True
    for i in (0, 2):
        for switch in (a, b):
            common[spread(rows[(band == i) & switch], 12)] = True
    masks = {"initial_A": initial, "initial_B": ~initial,
             "interior_A": common | (a & (band == 1)),
             "interior_B": common | (b & (band == 1))}
    return masks, family, mid, band, q


def run(output):
    """Solve seven layouts, check physical identities, and write compact results."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    graph, points, edges, radii, node_ids, edge_ids, fixed, loaded, forces = make_problem()
    original = graph.to_dict()
    masks, family, mid, band, q = material_layouts(points, edges, edge_ids)
    moduli = {"uniform_0p5": np.full(len(edges), 0.5e9),
              "uniform_1": np.full(len(edges), 1e9),
              "uniform_2": np.full(len(edges), 2e9)}
    moduli.update({name: np.where(mask, 2e9, 0.5e9) for name, mask in masks.items()})
    free = np.setdiff1d(np.arange(len(points)), fixed)
    results, displacements = {}, {}
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    axes = (points[edges[:, 1]] - points[edges[:, 0]]) / lengths[:, None]
    for name, E in moduli.items():
        materials = [fn.Material(youngs_modulus=float(e), poissons_ratio=0.3) for e in E]
        raw = BeamFrameFEM().solve_3d(
            edges.T, points, radii, forces=forces, fixed_nodes=fixed.tolist(),
            damping=0.0, edge_materials=materials,
        )
        u, reaction = raw["u"], raw["reactions"]
        energy = float(raw["beam_energy"]["strain_energy"])
        np.testing.assert_array_equal(raw["material_contract"]["youngs_modulus"], E)
        elongation = np.sum(axes * (u[edges[:, 1], :3] - u[edges[:, 0], :3]), axis=1)
        np.testing.assert_allclose(raw["edge_forces"][:, 0],
                                   E * np.pi * radii**2 * elongation / lengths,
                                   rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(reaction[free, :3], 0, atol=1e-11)
        np.testing.assert_allclose(reaction[free, 3:], 0, atol=1e-13)
        np.testing.assert_array_equal(u[fixed], 0)
        np.testing.assert_array_equal(raw["edge_list"], np.arange(len(edges)))
        np.testing.assert_allclose(reaction[fixed, :3].sum(axis=0) + forces.sum(axis=0),
                                   0, atol=1e-11)
        moment = (reaction[fixed, 3:].sum(axis=0)
                  + np.cross(points[fixed], reaction[fixed, :3]).sum(axis=0)
                  + np.cross(points, forces).sum(axis=0))
        np.testing.assert_allclose(moment, 0, atol=1e-13)
        np.testing.assert_allclose(energy, 0.5 * np.sum(forces * u[:, :3]), rtol=1e-10)
        np.testing.assert_allclose(energy, 0.5 * u.ravel() @ (raw["K"] @ u.ravel()), rtol=1e-10)
        mean_ux = float(u[loaded, 0].mean())
        results[name] = dict(mean_loaded_ux_m=mean_ux,
            stiffness_N_per_m=float(forces[:, 0].sum() / mean_ux), strain_energy_J=energy,
            max_free_force_N=float(np.max(np.abs(reaction[free, :3]))),
            max_free_moment_Nm=float(np.max(np.abs(reaction[free, 3:]))),
            support_force_N=reaction[fixed, :3].sum(axis=0).tolist(),
            moment_balance_Nm=moment.tolist())
        if name in masks:
            mask = masks[name]
            results[name].update(high_E_count=int(mask.sum()),
                high_E_band_counts=[int(np.sum(mask & (band == i))) for i in range(3)],
                high_E_band_x_moments_m=[float(mid[mask & (band == i), 0].sum()) for i in range(3)],
                high_E_orientation_counts={k: int(np.sum(mask & (family == k))) for k in ("XY", "XZ", "YZ")})
        displacements[name] = u
    for name, factor in (("uniform_0p5", 0.5), ("uniform_2", 2.0)):
        np.testing.assert_allclose(displacements[name] * factor, displacements["uniform_1"],
                                   rtol=2e-10, atol=1e-13)
        np.testing.assert_allclose(results[name]["strain_energy_J"] * factor,
                                   results["uniform_1"]["strain_energy_J"], rtol=2e-10)
    assert graph.to_dict() == original
    contrasts = {prefix: 100 * (results[prefix + "_A"]["stiffness_N_per_m"] /
                                  results[prefix + "_B"]["stiffness_N_per_m"] - 1)
                 for prefix in ("initial", "interior")}
    report = dict(fibernet_version=fn.__version__, nodes=len(points), bars=len(edges),
                  size_m=np.ptp(points, axis=0).tolist(), radius_m=0.00025,
                  poisson_ratio=0.3, total_force_N=forces.sum(axis=0).tolist(),
                  model="small-displacement isotropic linear 3D welded beam frame",
                  cases=results,
                  A_over_B_stiffness_change_percent=contrasts)
    np.savez_compressed(output / "layouts.npz", positions_m=points, edges=edges,
        radii_m=radii, node_ids=node_ids, edge_ids=edge_ids, forces_N=forces,
        fixed_rows=fixed, loaded_rows=loaded, case_names=list(moduli),
        youngs_modulus_Pa=np.array(list(moduli.values())),
        displacement_and_rotation=np.array(list(displacements.values())))
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New results directory (existing directories are not overwritten).")
    print(json.dumps(run(parser.parse_args().output), indent=2, allow_nan=False))

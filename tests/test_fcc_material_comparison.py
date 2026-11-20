"""Public FCC example: layout control, SI mechanics and reproducible outputs."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).parents[1] / "examples/fcc_material_comparison.py"
spec = importlib.util.spec_from_file_location("fcc_material_example", PATH)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_same_graph_layout_and_spatial_control():
    graph, points, edges, radii, node_ids, edge_ids, fixed, loaded, forces = example.make_problem()
    assert (len(node_ids), len(edge_ids), len(fixed), len(loaded)) == (123, 516, 18, 18)
    np.testing.assert_allclose(np.ptp(points, axis=0), [0.03, 0.03, 0.02])
    np.testing.assert_allclose(radii, 0.00025)
    np.testing.assert_allclose(forces.sum(axis=0), [0.018, 0, 0])
    masks, family, mid, band, q = example.material_layouts(points, edges, edge_ids)
    again = example.material_layouts(points, edges, edge_ids)[0]
    for name, mask in masks.items():
        assert mask.sum() == 258
        np.testing.assert_array_equal(mask, again[name])
    np.testing.assert_array_equal(masks["initial_A"], ~masks["initial_B"])
    a, b = masks["interior_A"], masks["interior_B"]
    np.testing.assert_array_equal(a[band != 1], b[band != 1])
    for i in range(3):
        assert np.sum(a & (band == i)) == np.sum(b & (band == i)) == [82, 94, 82][i]
        assert q[a & (band == i)].sum() == q[b & (band == i)].sum()
    # Equal counts imply equal nominal high-modulus volume: FCC bars have equal lengths/radii.
    np.testing.assert_allclose(np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1),
                               0.01 / np.sqrt(2))
    assert not np.array_equal(np.bincount(q[a], minlength=13), np.bincount(q[b], minlength=13))


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    output = tmp_path_factory.mktemp("fcc-material") / "run"
    return output, example.run(output)


def test_force_energy_units_and_homogeneous_inverse_modulus(solved):
    output, report = solved
    with np.load(output / "layouts.npz", allow_pickle=False) as data:
        u = data["displacement_and_rotation"]
        names = data["case_names"].tolist()
        for name, factor in (("uniform_0p5", 0.5), ("uniform_2", 2)):
            np.testing.assert_allclose(u[names.index(name)] * factor, u[names.index("uniform_1")],
                                       rtol=2e-10, atol=1e-13)
        for name in names:
            case = report["cases"][name]
            assert case["stiffness_N_per_m"] > 0
            np.testing.assert_allclose(case["strain_energy_J"],
                                       0.5 * 0.018 * case["mean_loaded_ux_m"], rtol=1e-10)
            np.testing.assert_allclose(case["support_force_N"], [-0.018, 0, 0], atol=1e-11)
            assert case["max_free_force_N"] < 1e-11
            assert case["max_free_moment_Nm"] < 1e-13
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == report


def test_named_layouts_reproduce_opposite_ordering(solved):
    _, report = solved
    # This checks this specified illustrative geometry, not a material design law.
    contrasts = report["A_over_B_stiffness_change_percent"]
    assert contrasts["initial"] > 0
    assert contrasts["interior"] < 0
    np.testing.assert_allclose([contrasts["initial"], contrasts["interior"]],
                               [7.55962865, -1.49097345], atol=1e-6, rtol=0)


def test_output_does_not_overwrite_existing_directory(solved):
    output, _ = solved
    before = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        example.run(output)
    assert (output / "summary.json").read_bytes() == before

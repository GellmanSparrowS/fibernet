"""Repository-only APP parity contracts, excluded from library distributions.

Public/golden tests remain independently collected in their original modules.
APP comparison inputs, random sequences, tolerances, and parameter cases match
those contracts; this file deliberately remains at the same directory depth.
"""
import ast
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pytest

from fibernet.gen import CustomCell, MappingConfig, load_obj, map_cells
from fibernet.gen.manufacturing import PlanarManufacturingConfig, compile_planar, compile_surface
from fibernet.gen.spectrum import fit_spectrum, resample_spectrum, rotated_displacements
from fibernet.sim import ReducedBeamConfig, ReducedBeamSolver
from fibernet.ml import PlanarPhysicalDataset, PhysicalRegressor, run_curve_inverse
from fibernet.ml.curve_inverse import fast_cfg as library_fast_cfg
from fibernet.analysis import ContactConfig, SnapshotFeatureExtractor, compute_percolation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FiberScope"))
from fslab.surface_mapping import map_cells as app_map_cells
from fslab.engine2 import Engine2, Engine2Config
from fslab.mlmodel import make_sample
from fslab.structure import StructureFactory, _cell_factory, valid_cell_spec
from fslab.learning import Regressor as AppRegressor
from fslab.features import compute_features as app_compute_features
from fslab.inverse import run_inverse as app_run_inverse
from fslab.inverse import fast_cfg as app_fast_cfg


SPEC = {"nodes": [[0, 0], [1, 0], [1, 1], [0, 1], [.5, .5]],
        "edges": [[0, 1], [1, 2], [2, 3], [3, 0],
                  [0, 4], [1, 4], [2, 4], [3, 4]],
        "zh": "自定义", "en": "Custom",
        "settings": {"expansion_rule": "translate"}}



def _digest(features):
    def ordinary(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value
    data = {key: ordinary(value) for key, value in features.items()}
    return hashlib.sha256(json.dumps(data, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()



def _builder(unit, pert, ld):
    return {"unit": unit, "pert": pert, "ld": ld}



def _evaluator(graph, target):
    values = np.asarray(graph["ld"] or [[0, 0], [0, 0]], float)
    return float(np.sum((values - 0.1) ** 2) + graph["pert"]), graph



@pytest.mark.parametrize("unit, expected", [
    ("square", ((2461, 3), (3408, 2), "d66dad1f8c534833", "3e9d2497fa8e1bde")),
    ("hexagon", ((3852, 3), (4704, 2), "34a30d33c3221c40", "db0bbae331ed9aa5")),
])
def test_surface_map_app_parity(unit, expected):
    path = ROOT / "FiberScope" / "assets" / "obj" / "demo_pyramid.obj"
    vertices, faces = load_obj(path)
    spectrum = [[0.02, -0.01], [0.01, 0.03]]
    config = MappingConfig(max_points=100000, max_segments=200000)
    points, edges = map_cells(vertices, faces, spectrum, unit, config)
    app_points, app_edges = app_map_cells(
        vertices, faces, spectrum, unit,
        MappingConfig(max_points=100000, max_segments=200000))
    assert np.array_equal(points, app_points)
    assert np.array_equal(edges, app_edges)



def test_recruitment_ui_describes_strain_not_force_flow():
    # Inspect the static translation table without importing Qt or the app.
    tree = ast.parse((ROOT / "FiberScope" / "studio" / "i18n.py").read_text(encoding="utf-8"))
    table = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "TEXT"
                         for target in node.targets))
    strings = {ast.literal_eval(key): ast.literal_eval(value)
               for key, value in zip(table.keys, table.values)
               if isinstance(value, ast.Tuple) and all(isinstance(item, ast.Constant)
                                                     for item in value.elts)}
    assert strings["quantile"][1] == "Strain threshold α"
    assert strings["view_perc"][1] == "Tensile recruitment"
    assert strings["legend_active"][1] == "recruited"
    assert "not a fixed 5% strain" in strings["quantile_hint"][1]
    assert "not complete force flow" in strings["explain_perc"][1]
    assert "force flows along continuous paths only" not in strings["explain_perc"][1]



def test_app_reexports_shared_profile_functions():
    import sys
    from pathlib import Path

    app = str(Path(__file__).resolve().parents[1] / 'FiberScope')
    sys.path.insert(0, app)
    try:
        from fslab.structure import (fit_spectrum as app_fit,
                                     resample_spectrum as app_resample,
                                     rotated_displacements as app_rotate)
        assert app_fit is fit_spectrum
        assert app_resample is resample_spectrum
        assert app_rotate is rotated_displacements
    finally:
        sys.path.remove(app)



def test_reduced_solver_app_aliases():
    assert ReducedBeamSolver is Engine2
    assert ReducedBeamConfig is Engine2Config



def test_physical_dataset_app_parity(tmp_path):
    rng = np.random.default_rng(5)
    factory = StructureFactory(
        unit="square", seed=5,
        line_displacements=rng.uniform(-0.2, 0.2, (5, 2)).tolist(),
        perturbation=0.1 * rng.uniform())
    expected_x, expected_y = make_sample(factory)
    path = tmp_path / "labels.npz"
    stream = PlanarPhysicalDataset(unit="square", grid=3, seed0=5, path=path)
    X, Y = stream.generate(1)
    np.testing.assert_array_equal(X[0], expected_x)
    np.testing.assert_array_equal(Y[0], expected_y)



def test_custom_cell_app_parity(tmp_path):
    assert valid_cell_spec(SPEC)
    cell = CustomCell.from_mapping(SPEC)
    graph = cell.to_graph()
    app_graph = _cell_factory(SPEC)(box=(10, 10), n_pts_per_side=0)
    np.testing.assert_array_equal(graph.node_positions(), app_graph.node_positions())
    np.testing.assert_array_equal(graph.edge_array(), app_graph.edge_array())



@pytest.mark.parametrize('unit', ['square', 'hexagon', 'ring', 'kagome', 'octagon'])
@pytest.mark.parametrize('rule', ['translate', 'checker_mirror'])
def test_planar_manufacturing_app_parity(unit, rule):
    settings = dict(unit=unit, grid_x=2, grid_y=2, n_pts_per_side=2,
                    line_displacements=[[0.02, -0.01], [0.01, 0.03]],
                    perturbation=0.2, seed=23, expansion_rule=rule)
    library = compile_planar(PlanarManufacturingConfig(**settings))
    desktop = compile_planar(StructureFactory(**settings))
    assert np.array_equal(library.edges, desktop.edges)
    assert np.array_equal(library.route_nodes, desktop.route_nodes)
    assert np.array_equal(library.route_edges, desktop.route_edges)
    assert np.allclose(library.reference, desktop.reference, atol=1e-12)
    assert np.allclose(library.positions, desktop.positions, atol=1e-12)



@pytest.mark.parametrize('unit', ['square', 'hexagon', 'kagome'])
def test_surface_manufacturing_app_parity(unit):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.2],
                         [0, 1, 0.1], [1, 1, 0.2], [2, 1, 0.4]])
    faces = [[0, 1, 4, 3], [1, 2, 5, 4]]
    settings = dict(unit=unit, n_pts_per_side=2, seed=23,
                    line_displacements=[[0.02, -0.01], [0.01, 0.03]])
    library = compile_surface(vertices, faces,
                              PlanarManufacturingConfig(**settings))
    desktop = compile_surface(vertices, faces, StructureFactory(**settings))
    assert np.array_equal(library.edges, desktop.edges)
    assert np.array_equal(library.route_nodes, desktop.route_nodes)
    assert np.allclose(library.positions, desktop.positions, atol=1e-12)



def test_physical_regressor_app_alias():
    assert PhysicalRegressor is AppRegressor



def test_snapshot_features_app_parity():
    positions = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
                          [.5, -.2], [.5, 1.2], [-.2, .5], [1.2, .5]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [6, 7]], dtype=int)
    config = ContactConfig(width=.16, resolution=64, max_pixels=200000)
    library = SnapshotFeatureExtractor(include_contact=True,
                                       contact_config=config)
    actual = library.extract(positions, edges)
    expected = app_compute_features(positions, edges, include_contact=True,
                                    contact_config=config)
    assert _digest(actual) == _digest(expected)



def test_curve_inverse_app_parity():
    kw = dict(budget=7, seed=31, fixed_unit="hexagon", evaluator=_evaluator)
    public = run_curve_inverse(_builder, "J", **kw)
    desktop = app_run_inverse(_builder, "J", **kw)
    assert public["evaluations"] == desktop["evaluations"] == 7
    assert public["best_dist"] == desktop["best_dist"]
    assert public["best_spec"] == desktop["best_spec"]
    assert [r.params for r in public["records"]] == [r.params for r in desktop["records"]]



def test_inverse_solver_configuration_matches_app():
    assert asdict(library_fast_cfg(1.7)) == asdict(app_fast_cfg(1.7).engine_cfg())



def test_app_import_uses_same_calculation():
    from pathlib import Path
    import sys

    app_root = Path(__file__).resolve().parents[1] / 'FiberScope'
    sys.path.insert(0, str(app_root))
    try:
        from fslab.percolation import compute_percolation as app_compute
        assert app_compute is compute_percolation
    finally:
        sys.path.remove(str(app_root))


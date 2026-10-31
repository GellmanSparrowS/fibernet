"""Source-only FEM grip admission checks; no project import or physical solve.

Run directly to avoid pytest conftest:
    python tests/test_stretch_boundary_fraction_source.py

Only selected function definitions are extracted. Backend imports are inert
doubles, and every raw solver route raises a capture sentinel immediately.
These checks do not qualify numerical results, units, or native runtimes.
"""
from __future__ import annotations

import ast
import builtins
import copy
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def extract(relative, name, namespace, owner=None):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    body = tree.body
    if owner is not None:
        body = next(node.body for node in body
                    if isinstance(node, ast.ClassDef) and node.name == owner)
    node = copy.deepcopy(next(node for node in body
                              if isinstance(node, ast.FunctionDef)
                              and node.name == name))
    node.decorator_list = []
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__",
                       names=[ast.alias(name="annotations")], level=0),
        node,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(relative), "exec"),
         namespace)
    return namespace[name]


def graph_fixture(count=5, dimension=2, tied=False):
    # Deliberately sparse, non-sorted IDs in a non-geometric insertion order.
    ids = [41, 9, 84, 2, 60][:count]
    xs = [2., 0., 4., 1., 3.][:count]
    if count > 5:
        ids.extend(range(101, 101 + count - 5))
        xs.extend(float(i) for i in range(5, count))
    if tied:
        xs = [float(i // 2) for i in range(count)]
    nodes = {nid: types.SimpleNamespace(
        position=np.array([x, float(row % 2), 0.]))
        for row, (nid, x) in enumerate(zip(ids, xs))}
    geometric_ids = sorted(ids, key=lambda nid: nodes[nid].position[0])
    edges = {200 + row: types.SimpleNamespace(
        node_i=left, node_j=right, internal_points=None, radius=.05)
        for row, (left, right) in enumerate(
            zip(geometric_ids[:-1], geometric_ids[1:]))}
    return types.SimpleNamespace(nodes=nodes, edges=edges, dimension=dimension)


def legacy_percentile_rows(positions, pct):
    # Independent expected contract: retain the original argsort and floor rule.
    count = max(int(len(positions) * pct), 1)
    output = {}
    for axis, low, high in ((0, "left", "right"),
                            (1, "bottom", "top"),
                            (2, "back", "front")):
        if axis >= positions.shape[1]:
            continue
        order = np.argsort(positions[:, axis])
        output[low] = list(order[:count])
        output[high] = list(order[-count:])
    return output


class SolverCapture(Exception):
    pass


class GripAdmission(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.import_calls = []
        self.captured = None
        namespace = {"np": np, "warnings": warnings}
        self.array_adapter = extract("fibernet/sim/accelerated.py",
                                     "_graph_to_arrays", namespace)
        self.selector = extract("fibernet/sim/accelerated.py",
                                "_get_boundary_indices", namespace)
        conversion = extract("fibernet/ml/beam_frame_fem.py",
                             "graph_to_fem_input", namespace, "BeamFrameFEM")
        stretch = extract("fibernet/ml/beam_frame_fem.py",
                          "stretch_test", namespace, "BeamFrameFEM")
        owner = self

        class FEM:
            graph_to_fem_input = staticmethod(conversion)
            stretch_test = stretch

            def __init__(self, E=1e9, nu=.3):
                owner.calls.append(("fem_constructor", E, nu))

            def solve_2d(self, *args, **kwargs):
                owner.captured = ("solve_2d", args, kwargs)
                raise SolverCapture("raw solve intentionally not executed")

            def solve_2d_nonlinear(self, *args, **kwargs):
                owner.captured = ("solve_2d_nonlinear", args, kwargs)
                raise SolverCapture("raw solve intentionally not executed")

            def solve_3d(self, *args, **kwargs):
                owner.captured = ("solve_3d", args, kwargs)
                raise SolverCapture("raw solve intentionally not executed")

            def to_sim_result(self, *args, **kwargs):
                raise AssertionError("result conversion must not execute")

        self.FEM = FEM
        self.simulate = extract("fibernet/easy.py", "simulate", {"np": np})
        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name == "fibernet.sim.accelerated":
                owner.import_calls.append(name)
                return types.SimpleNamespace(
                    _graph_to_arrays=owner.array_adapter,
                    _get_boundary_indices=owner.selector)
            if name == "fibernet.ml.beam_frame_fem":
                owner.import_calls.append(name)
                return types.SimpleNamespace(BeamFrameFEM=FEM)
            if name.split(".")[0] in ("fibernet", "taichi", "lammps"):
                raise AssertionError("forbidden project/native import: " + name)
            return real_import(name, *args, **kwargs)

        self.guard = patch("builtins.__import__", side_effect=guard)
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def test_invalid_fraction_fails_before_raw_solve(self):
        invalid = [-.01, .5001, .6, 1, 10, 10**100, -10**100, 10**400,
                   True, False, np.bool_(False),
                   np.timedelta64(0), np.timedelta64(0, "D"),
                   np.timedelta64("NaT"),
                   float("nan"), float("inf"), -float("inf"),
                   "0.1", "1", None, 0.1j, [], {}, np.array(.1)]
        for dimension in (2, 3):
            for pct in invalid:
                with self.subTest(dimension=dimension, pct=repr(pct)):
                    self.captured = None
                    with self.assertRaisesRegex(ValueError, "pct"):
                        self.simulate(graph_fixture(dimension=dimension),
                                      backend="fem", stretch_ratio=1.01,
                                      nonlinear=False, pct=pct)
                    self.assertIsNone(self.captured)

    def test_direct_conversion_uses_the_same_fraction_contract(self):
        for pct in (-1., .6, True, np.nan, np.inf, "0.1", 10**100,
                    np.timedelta64(0), np.timedelta64(0, "D"),
                    np.timedelta64("NaT")):
            with self.subTest(pct=repr(pct)):
                with self.assertRaisesRegex(ValueError, "pct"):
                    self.FEM.graph_to_fem_input(graph_fixture(), pct=pct)
                self.assertIsNone(self.captured)

    def test_empty_and_single_node_graphs_have_clear_admission_error(self):
        for count in (0, 1):
            for dimension in (2, 3):
                with self.subTest(count=count, dimension=dimension):
                    with self.assertRaisesRegex(ValueError, "at least two nodes"):
                        self.simulate(graph_fixture(count, dimension),
                                      backend="fem", stretch_ratio=1.01,
                                      nonlinear=False)
                    self.assertIsNone(self.captured)

    def test_legal_rows_match_legacy_floor_minimum_and_tie_order(self):
        fractions = (0, -0., 0.1, .25, .5,
                     np.int64(0), np.float32(.25), np.float64(.5))
        for count in (2, 3, 5, 20):
            for tied in (False, True):
                graph = graph_fixture(count, tied=tied)
                positions, _, ids, _ = self.array_adapter(graph)
                for pct in fractions:
                    with self.subTest(count=count, tied=tied, pct=pct):
                        actual = self.FEM.graph_to_fem_input(graph, pct=pct)
                        self.assertEqual(actual["boundaries"],
                                         legacy_percentile_rows(positions, pct))
                        left = actual["boundaries"]["left"]
                        right = actual["boundaries"]["right"]
                        self.assertTrue(left and right)
                        self.assertFalse(set(left) & set(right))
                        self.assertEqual(list(graph.nodes), ids)
                        self.assertTrue(all(0 <= int(row) < count
                                            for row in left + right))

        # Do not reuse the old scalar-multiplication oracle here: NumPy 2 can
        # narrow the Python node count to float16/int8 before multiplying.
        for count, pct, expected_count in ((2051, np.float16(.5), 1025),
                                           (128, np.int8(0), 1)):
            with self.subTest(count=count, pct=repr(pct), scalar_type=type(pct)):
                graph = graph_fixture(count)
                actual = self.FEM.graph_to_fem_input(graph, pct=pct)
                left, right = (actual["boundaries"][side]
                               for side in ("left", "right"))
                self.assertEqual(len(left), expected_count)
                self.assertEqual(len(right), expected_count)
                self.assertFalse(set(left) & set(right))

        # A value immediately below one-half exercises longdouble precision
        # without assuming that a platform's longdouble is wider than float64.
        precise = np.nextafter(np.longdouble(.5), np.longdouble(0))
        actual = self.FEM.graph_to_fem_input(graph_fixture(4), pct=precise)
        self.assertEqual(len(actual["boundaries"]["left"]), 1)
        self.assertEqual(len(actual["boundaries"]["right"]), 1)

    def test_default_is_exactly_explicit_point_one(self):
        graph = graph_fixture(20)
        implicit = self.FEM.graph_to_fem_input(graph)
        explicit = self.FEM.graph_to_fem_input(graph, pct=.1)
        self.assertEqual(implicit["boundaries"], explicit["boundaries"])
        for key in ("edge_index", "node_pos", "radii"):
            np.testing.assert_array_equal(implicit[key], explicit[key])

    def test_public_stretch_preserves_sparse_id_row_translation(self):
        for dimension in (2, 3):
            graph = graph_fixture(dimension=dimension)
            original_ids = list(graph.nodes)
            original_edges = [(eid, edge.node_i, edge.node_j)
                              for eid, edge in graph.edges.items()]
            with self.assertRaises(SolverCapture):
                self.simulate(graph, backend="fem", stretch_ratio=1.01,
                              nonlinear=False, pct=.4)
            method, args, kwargs = self.captured
            self.assertEqual(method, "solve_" + str(dimension) + "d")
            self.assertEqual(kwargs["fixed_nodes"], [1, 3])
            self.assertEqual(set(kwargs["prescribed_disp"]), {4, 2})
            self.assertEqual([original_ids[row]
                              for row in kwargs["fixed_nodes"]], [9, 2])
            self.assertEqual([original_ids[row]
                              for row in sorted(kwargs["prescribed_disp"])],
                             [84, 60])
            for displacement in kwargs["prescribed_disp"].values():
                self.assertAlmostEqual(displacement[0], .04)
                self.assertEqual(tuple(displacement[1:]),
                                 (0.,) * (dimension - 1))
            np.testing.assert_array_equal(
                args[0], np.array([[1, 3, 0, 4], [3, 0, 4, 2]]))
            self.assertEqual(list(graph.nodes), original_ids)
            self.assertEqual([(eid, edge.node_i, edge.node_j)
                              for eid, edge in graph.edges.items()],
                             original_edges)

    def test_coordinate_ties_are_not_reinterpreted_as_faces(self):
        graph = graph_fixture()
        for node in graph.nodes.values():
            node.position[0] = 2.
        positions, _, _, _ = self.array_adapter(graph)
        actual = self.FEM.graph_to_fem_input(graph, pct=.4)
        self.assertEqual(actual["boundaries"],
                         legacy_percentile_rows(positions, .4))
        self.assertEqual(actual["x_range"], 0.)
        # This guard only governs selected rows, not the loading model's validity.

    def test_fixed_tolerance_mode_does_not_consume_pct(self):
        positions = np.array([[0., 0., 0.], [1., 1., 1.]])
        expected = self.selector(positions, tol=.1)
        for pct in (-1., 10., None, True, np.nan, "unused"):
            with self.subTest(pct=repr(pct)):
                self.assertEqual(self.selector(positions, tol=.1, pct=pct),
                                 expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)


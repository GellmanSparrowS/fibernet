"""Geometry acceptance, ownership and error contracts using inert graph objects.

The import-free production helper is loaded by the local support module; no
installed FiberNet or numeric provider is required by this test module.
"""
import ast
import copy
import inspect
import json
from types import SimpleNamespace
import unittest

from .support import H, ROOT

POLICY = SimpleNamespace(**H)
HELPER = ROOT / 'fibernet/sim/_taichi_geometry.py'

def edge(edge_id, points=None, node_i=9, node_j=-4):
    return SimpleNamespace(
        edge_id=edge_id, node_i=node_i, node_j=node_j,
        internal_points=points, radius=0.125 + (edge_id % 3) / 100,
        material={'youngs_modulus': 100 + edge_id, 'extra': {'tag': [edge_id]}},
        segments=4, metadata={'fiber': [edge_id]},
    )


def graph(entries):
    return SimpleNamespace(
        edges={eid: edge(eid, points) for eid, points in entries},
        nodes={9: SimpleNamespace(position=[0., 0., 0.]),
               -4: SimpleNamespace(position=[2., 0., 0.])},
        metadata={'units': {'length': 'mm', 'force': 'N'}, 'tag': ['original']},
        dimension=3,
    )


class LengthOnlyPoints:
    """Stand-in for array semantics: inspection must only call len()."""
    def __init__(self, count):
        self.count = count
    def __len__(self):
        return self.count
    def __bool__(self):
        raise AssertionError('Do not use array truthiness')
    def __iter__(self):
        raise AssertionError('Do not inspect coordinates')
    def __getitem__(self, key):
        raise AssertionError('Do not inspect coordinates')


class GraphAccessSentinel:
    @property
    def edges(self):
        raise AssertionError('Invalid policy must fail before graph access')


class GeometryPreflightTests(unittest.TestCase):
    def test_01_none_and_empty_points_are_accepted(self):
        g = graph([(52, None), (-9, []), (7, ())])
        record = POLICY.preflight_taichi_geometry(g)
        self.assertEqual(record, {
            'schema': 'taichi_endpoint_chord/1',
            'policy': 'reject_internal_points',
            'representation': 'endpoint_chord',
            'ignored_internal_point_edge_ids': [],
        })

    def test_02_empty_graph_is_only_geometry_eligible_not_solver_validated(self):
        self.assertEqual(POLICY.preflight_taichi_geometry(graph([]))[
            'ignored_internal_point_edge_ids'], [])

    def test_03_strict_rejects_every_nonempty_geometry(self):
        fixtures = {
            'collinear': [[0.5, 0., 0.], [1.5, 0., 0.]],
            'off_chord': [[1., 0.25, 0.]],
            'duplicate': [[1., 0., 0.], [1., 0., 0.]],
            'backtracking': [[1.5, 0., 0.], [0.5, 0., 0.]],
            'endpoint_duplicate': [[0., 0., 0.]],
            'tiny_off_chord': [[1., 1e-300, 0.]],
            'tiny_coordinate_scale': [[1e-300, 0., 0.]],
            'large_coordinate_scale': [[1e300, 0., 0.]],
        }
        for name, points in fixtures.items():
            for kwargs in ({}, {'geometry_policy': 'reject_internal_points'}):
                with self.subTest(name=name, kwargs=kwargs):
                    with self.assertRaises(POLICY.UnsupportedInternalPointsError) as caught:
                        POLICY.preflight_taichi_geometry(graph([(61, points)]), **kwargs)
                    self.assertEqual(caught.exception.rejected_edge_ids, (61,))

    def test_04_explicit_chord_accepts_each_nonempty_geometry(self):
        fixtures = [
            [[0.5, 0., 0.], [1.5, 0., 0.]], [[1., 1., 0.]],
            [[1., 0., 0.], [1., 0., 0.]],
            [[1.5, 0., 0.], [0.5, 0., 0.]], [[0., 0., 0.]],
        ]
        for points in fixtures:
            with self.subTest(points=points):
                record = POLICY.preflight_taichi_geometry(
                    graph([(61, points)]), geometry_policy='endpoint_chord')
                self.assertEqual(record['ignored_internal_point_edge_ids'], [61])
                self.assertEqual(record['policy'], 'endpoint_chord')
                self.assertEqual(record['representation'], 'endpoint_chord')
                self.assertNotIn('geometry_preserved', record)

    def test_05_nonmonotonic_ids_parallel_edges_and_order_are_preserved(self):
        g = graph([(90, [[1., 0., 0.]]), (-3, [[1., 1., 0.]]),
                   (17, None), (2, [[1.5, 0., 0.], [0.5, 0., 0.]])])
        self.assertEqual(len({(e.node_i, e.node_j) for e in g.edges.values()}), 1)
        before = copy.deepcopy(g)
        record = POLICY.preflight_taichi_geometry(g, geometry_policy='endpoint_chord')
        self.assertEqual(record['ignored_internal_point_edge_ids'], [90, -3, 2])
        self.assertEqual(list(g.edges), [90, -3, 17, 2])
        self.assertEqual(g, before)
        with self.assertRaises(POLICY.UnsupportedInternalPointsError) as caught:
            POLICY.preflight_taichi_geometry(g)
        self.assertEqual(caught.exception.rejected_edge_ids, (90, -3, 2))

    def test_06_actual_segmented_nodes_are_not_edge_owned_points(self):
        g = graph([])
        g.nodes = {
            19: SimpleNamespace(position=[0., 0., 0.]),
            -5: SimpleNamespace(position=[1., 1., 0.]),
            4: SimpleNamespace(position=[2., 0., 0.]),
        }
        g.edges = {61: edge(61, None, 19, -5), 2: edge(2, [], -5, 4)}
        before = copy.deepcopy(g)
        record = POLICY.preflight_taichi_geometry(g)
        self.assertEqual(record['ignored_internal_point_edge_ids'], [])
        self.assertEqual(g, before)
        # A non-straight actual-node chain is eligible; no claim of equivalent
        # curved-beam mechanics, no node creation, and no graph flattening.
        self.assertEqual(list(g.nodes), [19, -5, 4])

    def test_07_no_mutation_on_default_rejection(self):
        g = graph([(23, [[1., 1., 0.]])])
        before = copy.deepcopy(g)
        points = g.edges[23].internal_points
        material = g.edges[23].material
        with self.assertRaises(POLICY.UnsupportedInternalPointsError):
            POLICY.preflight_taichi_geometry(g)
        self.assertEqual(g, before)
        self.assertIs(g.edges[23].internal_points, points)
        self.assertIs(g.edges[23].material, material)

    def test_08_no_mutation_or_reference_replacement_on_opt_in(self):
        g = graph([(90, [[1., 1., 0.]]), (-3, [[1., 0., 0.]])])
        before = copy.deepcopy(g)
        owned = [(e, e.internal_points, e.material, e.metadata) for e in g.edges.values()]
        nodes = g.nodes
        units = g.metadata['units']
        POLICY.preflight_taichi_geometry(g, geometry_policy='endpoint_chord')
        self.assertEqual(g, before)
        self.assertIs(g.nodes, nodes)
        self.assertIs(g.metadata['units'], units)
        for current, refs in zip(g.edges.values(), owned):
            self.assertIs(current, refs[0])
            self.assertIs(current.internal_points, refs[1])
            self.assertIs(current.material, refs[2])
            self.assertIs(current.metadata, refs[3])

    def test_09_metadata_is_detached_across_calls_and_from_graph(self):
        g = graph([(90, [[1., 1., 0.]]), (-3, [[1., 0., 0.]])])
        one = POLICY.preflight_taichi_geometry(g, geometry_policy='endpoint_chord')
        two = POLICY.preflight_taichi_geometry(g, geometry_policy='endpoint_chord')
        self.assertIsNot(one, two)
        self.assertIsNot(one['ignored_internal_point_edge_ids'], two['ignored_internal_point_edge_ids'])
        one['ignored_internal_point_edge_ids'].clear()
        one['representation'] = 'tampered'
        self.assertEqual(two['ignored_internal_point_edge_ids'], [90, -3])
        self.assertEqual(two['representation'], 'endpoint_chord')
        self.assertEqual(list(g.edges), [90, -3])
        g.edges.clear()
        self.assertEqual(two['ignored_internal_point_edge_ids'], [90, -3])

    def test_10_straight_opt_in_records_choice_without_inventing_ignored_edges(self):
        record = POLICY.preflight_taichi_geometry(
            graph([(90, None), (-3, [])]), geometry_policy='endpoint_chord')
        self.assertEqual(record['policy'], 'endpoint_chord')
        self.assertEqual(record['ignored_internal_point_edge_ids'], [])

    def test_11_json_native_record_only_not_production_result_round_trip(self):
        record = POLICY.preflight_taichi_geometry(
            graph([(90, [[1., 1., 0.]])]), geometry_policy='endpoint_chord')
        self.assertEqual(json.loads(json.dumps(record, allow_nan=False)), record)
        self.assertEqual(set(record), {'schema', 'policy', 'representation',
                                      'ignored_internal_point_edge_ids'})

    def test_12_invalid_policies_raise_typed_errors_before_graph_access(self):
        class StringSubclass(str):
            pass
        invalid = [None, True, False, 0, 1, 0., [], {}, (), b'endpoint_chord',
                   '', 'chord', 'ENDPOINT_CHORD', 'endpoint_chord ',
                   ' reject_internal_points', StringSubclass('endpoint_chord'), object()]
        for value in invalid:
            with self.subTest(type=type(value).__name__, value=repr(value)):
                with self.assertRaises(POLICY.InvalidGeometryPolicyError):
                    POLICY.preflight_taichi_geometry(GraphAccessSentinel(), geometry_policy=value)
                with self.assertRaises(POLICY.InvalidGeometryPolicyError):
                    POLICY.validate_geometry_policy(value)

    def test_13_valid_policy_validator_needs_no_graph(self):
        for value in ('reject_internal_points', 'endpoint_chord'):
            self.assertEqual(POLICY.validate_geometry_policy(value), value)

    def test_14_typed_policy_errors_are_distinct_and_value_errors(self):
        for kind in (POLICY.InvalidGeometryPolicyError, POLICY.UnsupportedInternalPointsError):
            self.assertTrue(issubclass(kind, POLICY.GeometryPolicyError))
            self.assertTrue(issubclass(kind, ValueError))
        self.assertIsNot(POLICY.InvalidGeometryPolicyError, POLICY.UnsupportedInternalPointsError)

    def test_15_rejection_reports_all_ids_and_two_distinct_choices(self):
        g = graph([(90, [[1., 0., 0.]]), (-3, [[1., 1., 0.]])])
        with self.assertRaises(POLICY.UnsupportedInternalPointsError) as caught:
            POLICY.preflight_taichi_geometry(g)
        message = str(caught.exception)
        self.assertIn('(90, -3)', message)
        self.assertIn("geometry_policy='endpoint_chord'", message)
        self.assertIn('actual-node straight-edge StructureGraph', message)
        self.assertIn('Neither choice establishes curved-beam mechanical equivalence', message)
        g.edges.clear()
        self.assertEqual(caught.exception.rejected_edge_ids, (90, -3))

    def test_16_point_coordinates_and_array_truthiness_are_never_used(self):
        g = graph([(5, LengthOnlyPoints(0)), (2, LengthOnlyPoints(3))])
        with self.assertRaises(POLICY.UnsupportedInternalPointsError) as caught:
            POLICY.preflight_taichi_geometry(g)
        self.assertEqual(caught.exception.rejected_edge_ids, (2,))
        self.assertEqual(POLICY.preflight_taichi_geometry(
            g, geometry_policy='endpoint_chord')['ignored_internal_point_edge_ids'], [2])

    def test_17_only_existing_edge_read_surface_is_needed(self):
        g = SimpleNamespace(edges={81: SimpleNamespace(internal_points=None),
                                   4: SimpleNamespace(internal_points=[])})
        self.assertEqual(POLICY.preflight_taichi_geometry(g)['ignored_internal_point_edge_ids'], [])

    def test_18_policy_is_keyword_only(self):
        signature = inspect.signature(POLICY.preflight_taichi_geometry)
        self.assertEqual(signature.parameters['geometry_policy'].kind,
                         inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(signature.parameters['geometry_policy'].default, 'reject_internal_points')
        with self.assertRaises(TypeError):
            POLICY.preflight_taichi_geometry(graph([]), 'endpoint_chord')

    def test_19_module_has_no_imports_or_parallel_graph_definition(self):
        tree = ast.parse(HELPER.read_text(encoding='utf-8'))
        self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)))
        self.assertEqual({n.name for n in tree.body if isinstance(n, ast.ClassDef)}, {
            'GeometryPolicyError', 'InvalidGeometryPolicyError', 'UnsupportedInternalPointsError'})

    def test_20_malformed_graph_is_not_silently_normalized(self):
        with self.assertRaises(AttributeError):
            POLICY.preflight_taichi_geometry(SimpleNamespace(edges={1: object()}))
        with self.assertRaises(TypeError):
            POLICY.preflight_taichi_geometry(graph([(1, 3)]))


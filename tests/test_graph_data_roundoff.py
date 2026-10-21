"""Stored-value guards for graph features; no backend or FP-mode changes."""
import math
import struct
import unittest
from unittest.mock import patch

import numpy as np
from fibernet.core.structure_graph import StructureGraph
from fibernet.ml import graph_data


def floating(bits):
    return struct.unpack('>d', struct.pack('>Q', bits))[0]


def bits(value):
    return struct.unpack('>Q', struct.pack('>d', value))[0]


def position(payload):
    return np.frombuffer(struct.pack('>QQQ', payload, 0, 0), dtype='>f8').astype(np.float64)


def graph(left=0, right=0, interior=None):
    g = StructureGraph(3)
    a, b = g.add_node([0, 0, 0]), g.add_node([1, 0, 0])
    edge = g.add_edge(a, b)
    g.nodes[a].position = position(left)
    g.nodes[b].position = position(right)
    if interior is not None:
        g.edges[edge].internal_points = position(interior).reshape(1, 3)
    return g


def batch(g, name):
    return graph_data.graph_batch([g], graph_ids=['g'], length_unit='unit',
                                 coordinate_frame='test', edge_features=[name])


class TestGraphDataRoundoff(unittest.TestCase):
    def test_stored_identity_ignores_only_zero_sign(self):
        self.assertEqual(graph_data._float64_identity(floating(0)), 0)
        self.assertEqual(graph_data._float64_identity(floating(1 << 63)), 0)
        for payload in (1, (1 << 63)|1, (1 << 52)-1, 1 << 52,
                        0x3ff0000000000000, 0x7fefffffffffffff):
            self.assertEqual(graph_data._float64_identity(floating(payload)), payload)

    def test_injected_zero_distance_rejects_distinct_stored_endpoints(self):
        cases = [(0, 1), (0, (1 << 63)|1), (0, 0x123456789abcd),
                 (0, (1 << 52)-1), (1 << 52, (1 << 52)+1),
                 (0, 0x3ff0000000000000), (0, 0x7fefffffffffffff)]
        for left, right in cases:
            for name in ('chord_length', 'polyline_length'):
                g = graph(left, right);before = [n.position.tobytes() for n in g.nodes.values()]
                with patch.object(graph_data.math, 'dist', return_value=0.):
                    with self.assertRaisesRegex(ValueError, name + ' lost a nonzero endpoint distance'):
                        batch(g, name)
                self.assertEqual([n.position.tobytes() for n in g.nodes.values()], before)

    def test_injected_zero_sum_rejects_nonzero_stored_segments(self):
        for segment in (1, 0x3ff0000000000000):
            with patch.object(graph_data.math, 'dist', return_value=floating(segment)):
                with patch.object(graph_data.math, 'fsum', return_value=0.):
                    with self.assertRaisesRegex(ValueError, 'polyline_length lost a nonzero segment sum'):
                        batch(graph(0, 0x3ff0000000000000), 'polyline_length')

    def test_zero_and_signed_zero_paths_remain_available(self):
        for left, right, inside in ((0, 0, None), (1 << 63, 0, 1 << 63),
                                    (0x3ff0000000000000, 0x3ff0000000000000, None)):
            for name in ('chord_length', 'polyline_length'):
                out = batch(graph(left, right, inside), name)
                self.assertEqual(bits(out.edge_features[0, 0]), 0)
                self.assertTrue(out.edge_feature_available[0, 0])

    def test_closed_polyline_checks_each_internal_segment(self):
        g = graph(0, 0, 1)
        with patch.object(graph_data.math, 'dist', return_value=0.):
            self.assertEqual(bits(batch(g, 'chord_length').edge_features[0, 0]), 0)
            with self.assertRaisesRegex(ValueError, 'polyline_length lost a nonzero endpoint distance'):
                batch(g, 'polyline_length')

    def test_actual_subnormal_and_adjacent_normal_operations(self):
        cases = [(0, 1), (0, (1 << 63)|1), (0, 0x123456789abcd),
                 (0, (1 << 52)-1), (1 << 52, (1 << 52)+1)]
        for left, right in cases:
            g = graph(left, right);a, b = [n.position for n in g.nodes.values()]
            segment = math.dist(a, b)
            expected = (right & ((1 << 63)-1)) if not left else 1
            for name in ('chord_length', 'polyline_length'):
                value = segment if name == 'chord_length' else math.fsum([segment])
                if not bits(value) & ((1 << 63)-1):
                    with self.assertRaisesRegex(ValueError, name + ' lost a nonzero'):
                        batch(g, name)
                else:
                    out = batch(g, name)
                    self.assertEqual(bits(out.edge_features[0, 0]), expected)
                    self.assertTrue(out.edge_feature_available[0, 0])
                    self.assertEqual([bits(n.position[0]) for n in g.nodes.values()], [left, right])

    def test_normal_large_and_overflow_controls(self):
        for payload in (1 << 52, 0x3ff0000000000000, 0x6974e718d7d7625a):
            for name in ('chord_length', 'polyline_length'):
                self.assertEqual(bits(batch(graph(0, payload), name).edge_features[0, 0]), payload)
        for name in ('chord_length', 'polyline_length'):
            with self.assertRaisesRegex(ValueError, 'finite'):
                batch(graph(0xffefffffffffffff, 0x7fefffffffffffff), name)

    def test_geometry_only_does_not_require_distance_arithmetic(self):
        g = graph(0, 1)
        with patch.object(graph_data.math, 'dist', side_effect=AssertionError('unexpected distance')):
            out = graph_data.graph_batch([g], graph_ids=['g'], length_unit='unit', coordinate_frame='test')
        self.assertEqual(bits(out.positions[1, 0]), 1)
        self.assertEqual(out.edge_features.shape, (1, 0))


if __name__ == '__main__':
    unittest.main()

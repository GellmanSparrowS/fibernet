"""Facade compatibility and one bounded, real FEM save/reload journey.

No Taichi/LAMMPS engine construction, training, or native provider execution.
"""
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import fibernet as fn
from fibernet import easy


class StretchRatioTests(unittest.TestCase):
    def test_signature_preserves_positional_contract(self):
        signature = inspect.signature(easy.simulate)
        positional = [name for name, p in signature.parameters.items()
                      if p.kind == p.POSITIONAL_OR_KEYWORD]
        self.assertEqual(positional, ['graph', 'mode', 'strain', 'backend', 'save_path'])
        self.assertEqual(signature.parameters['stretch_ratio'].kind,
                         inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(signature.parameters['stretch_ratio'].default)

    def test_spring_alias_forwards_without_constructing_native_engine(self):
        graph = fn.StructureGraph(dimension=3)
        calls = []
        class Engine:
            def __init__(self, **kwargs):
                pass
            def stretch_test(self, graph, target_stretch, *, geometry_policy):
                if geometry_policy != "reject_internal_points":
                    raise AssertionError("unexpected geometry policy")
                calls.append(target_stretch)
                return target_stretch
        with patch.object(easy, 'TaichiEngine', Engine):
            for backend in ('spring', 'taichi'):
                for value in (1, 1.01, np.float64(.99)):
                    self.assertEqual(easy.simulate(graph, backend=backend,
                                                  stretch_ratio=value), value)
            self.assertEqual(easy.simulate(graph), 2.0)
            self.assertEqual(easy.simulate(graph, stretch_ratio=None), 2.0)
            self.assertEqual(easy.simulate(graph, strain=1.02, stretch_ratio=None), 1.02)
            self.assertEqual(easy.simulate(graph, 'stretch', 1.03, 'spring', None), 1.03)
        self.assertEqual(len(calls), 10)

    def test_invalid_explicit_ratio_fails_before_engine_creation(self):
        bad = (0, -1, float('nan'), float('inf'), -float('inf'),
               True, np.bool_(False), '1.01', [1.01], 1+0j, np.array(1.01), 10**1000)
        with patch.object(easy, 'TaichiEngine') as engine:
            for value in bad:
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite positive real ratio'):
                    easy.simulate(None, stretch_ratio=value)
            engine.assert_not_called()

    def test_conflicting_names_fail_even_when_equal(self):
        with patch.object(easy, 'TaichiEngine') as engine:
            for old, new in ((1.01, 1.01), (2., 1.01), (float('nan'), 1.01)):
                with self.subTest(old=old), self.assertRaisesRegex(ValueError, 'not both'):
                    easy.simulate(None, strain=old, stretch_ratio=new)
            engine.assert_not_called()

    def test_nonstretch_routes_reject_ratio_before_dispatch(self):
        with patch.object(easy, 'TaichiEngine') as engine:
            for backend, mode in (('spring', 'dynamics'), ('taichi', 'dynamics'),
                                  ('fem', 'history'), ('lammps', 'dynamics')):
                with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, 'only supported'):
                    easy.simulate(None, backend=backend, mode=mode, stretch_ratio=1.01)
                with self.subTest(legacy=backend), self.assertRaisesRegex(ValueError, 'strain is only supported'):
                    easy.simulate(None, backend=backend, mode=mode, strain=2.)
            engine.assert_not_called()

    def test_solver_keyword_remains_rejected(self):
        with patch.object(easy, 'TaichiEngine') as engine:
            with self.assertRaisesRegex(TypeError, 'not target_stretch'):
                easy.simulate(None, stretch_ratio=1.01, target_stretch=1.01)
            engine.assert_not_called()

    def test_real_fem_new_and_legacy_paths_agree_after_reload(self):
        # Exactly two tiny linear FEM solves, 18 nodes / 22 edges each.
        graph = fn.pattern_2d(unit='honeycomb', box=(10, 10), grid=(2, 2),
                              radius=.05, seed=23)
        before = graph.to_dict()
        options = dict(backend='fem', mode='stretch', dim=2, nonlinear=False)
        new = fn.simulate(graph, stretch_ratio=1.01, **options)
        old = fn.simulate(graph, strain=1.01, **options)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'detailed.json'
            new.save(path, detailed=True)
            restored = fn.SimResult.load(path)
            legacy_path = Path(directory) / 'legacy_detailed.json'
            old.save(legacy_path, detailed=True)
            restored_old = fn.SimResult.load(legacy_path)
        self.assertEqual(before, graph.to_dict())
        self.assertEqual(restored.metadata['node_ids'], list(graph.nodes))
        self.assertEqual(restored.metadata['edge_ids'], list(graph.edges))
        for field in ('deformed_positions', 'displacements', 'edge_forces', 'edge_stretches'):
            np.testing.assert_allclose(getattr(restored, field), getattr(restored_old, field))
        self.assertGreater(restored.max_displacement, 0)
        self.assertAlmostEqual(restored.max_force, restored_old.max_force)
        self.assertEqual(restored.mode, restored_old.mode)


if __name__ == '__main__':
    unittest.main()

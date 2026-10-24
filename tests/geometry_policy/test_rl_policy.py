"""RL geometry-policy contracts using inert collaborators.

Only selected production definitions execute, with inert graph, array, Gym,
factory and provider collaborators. This checks source-level dispatch and
exception handling, not installed-package, real-generator or Gym acceptance.
No numerical solve or training occurs. Actual StructureGraph segmentation is
covered separately by tests/test_taichi_geometry_package.py.
"""
from __future__ import annotations

import ast
import copy
import inspect
import types
import unittest
from unittest.mock import patch

from .support import ROOT, H, graph, isolated_builtins, load_functions


SOURCES = {
    'FiberStructureEnv': 'fibernet/rl/env.py',
    'FiberDesignEnv': 'fibernet/rl/env.py',
    'ParametricStructureEnv': 'fibernet/rl/parametric.py',
}


def load_environment(name, namespace):
    """Keep a real class body so the Gym methods retain their super() cell."""
    relative = SOURCES[name]
    tree = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
    node = copy.deepcopy(next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == name))
    selected = {'__init__', 'step', '_get_n_sides', '_compute_reward'}
    node.body = [method for method in node.body
                 if isinstance(method, ast.FunctionDef) and method.name in selected]
    if any(isinstance(item, (ast.Import, ast.ImportFrom)) for item in ast.walk(node)):
        raise AssertionError('An extracted RL method added an unreviewed import')
    module = ast.Module(body=[ast.ImportFrom(
        module='__future__', names=[ast.alias(name='annotations')], level=0), node],
        type_ignores=[])
    ast.fix_missing_locations(module)
    namespace['__builtins__'] = isolated_builtins()
    exec(compile(module, str(ROOT / relative), 'exec'), namespace)
    cls = namespace[name]
    # Observations are intentionally outside these dispatch tests.
    cls._get_obs = lambda self: ('inert observation',)
    return cls


def load_factory(namespace):
    """Replace only make_env's explicit legacy-class import, not providers."""
    relative = 'fibernet/rl/env.py'
    tree = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
    node = copy.deepcopy(next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == 'make_env'))

    class LegacyClassSeam(ast.NodeTransformer):
        def visit_ImportFrom(self, item):
            if (item.level, item.module, [(a.name, a.asname) for a in item.names]) != (
                    0, 'fibernet.rl.parametric', [('ParametricStructureEnv', None)]):
                raise AssertionError('Unreviewed factory import: ' + ast.dump(item))
            return ast.copy_location(ast.Pass(), item)

        def visit_Import(self, item):
            raise AssertionError('Unreviewed factory import: ' + ast.dump(item))

    module = ast.Module(body=[ast.ImportFrom(
        module='__future__', names=[ast.alias(name='annotations')], level=0),
        LegacyClassSeam().visit(node)], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace['__builtins__'] = isolated_builtins()
    exec(compile(module, str(ROOT / relative), 'exec'), namespace)
    return namespace['make_env']


class Vector(list):
    def flatten(self):
        return self


class RLPolicy(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.current_graph = graph()
        self.result = types.SimpleNamespace(
            max_force=2., max_stretch=1.5, mean_stretch=1.2, std_stretch=.1,
            metadata={'existing': {'retained': True}})
        owner = self

        class InertGymEnv:
            def __init__(self):
                self.np_random = types.SimpleNamespace(integers=lambda low, high: 17)

        class Engine:
            def __init__(self):
                owner.calls.append(('construct',))

            def stretch_test(self, value, **kwargs):
                owner.calls.append(('solve', value, kwargs))
                owner.result.metadata['taichi_geometry'] = H['preflight_taichi_geometry'](
                    value, geometry_policy=kwargs['geometry_policy'])
                return owner.result

        def factory(**kwargs):
            owner.calls.append(('graph', kwargs))
            return owner.current_graph

        self.factory = factory
        self.Engine = Engine
        self.namespace = {
            **H, 'gym': types.SimpleNamespace(Env=InertGymEnv), 'HAS_GYM': True,
            '_require_gym': lambda: None, 'TaichiEngine': Engine, 'pattern_2d': factory,
            'spaces': types.SimpleNamespace(Box=lambda **kw: kw, Dict=dict,
                                            Discrete=lambda bound: ('discrete', bound)),
            'np': types.SimpleNamespace(float32=float,
                                        asarray=lambda value, dtype: Vector(value),
                                        clip=lambda value, low, high: max(low, min(high, value))),
        }
        self.classes = {name: load_environment(name, self.namespace) for name in SOURCES}

    def action(self, env, n_internal=0):
        if type(env).__name__ == 'FiberDesignEnv':
            return {'unit_idx': 0, 'grid_x': 0, 'grid_y': 0,
                    'radius': [.03], 'n_internal': n_internal}
        return [0.] * env.n_actions

    def create(self, name, **kwargs):
        if name == 'FiberDesignEnv':
            return self.classes[name](available_units=['square'], **kwargs)
        return self.classes[name](n_pts_per_side=2, **kwargs)

    def test_new_keyword_only_option_preserves_positional_constructor_contracts(self):
        positional = {
            'FiberStructureEnv': ['self', 'unit', 'box', 'grid', 'n_pts_per_side',
                'target_stretch', 'stiffness', 'damping', 'num_steps',
                'displacement_range', 'reward_mode', 'target_force', 'reward_scale',
                'unit_kwargs', 'render_mode'],
            'FiberDesignEnv': ['self', 'available_units', 'max_grid', 'box',
                'target_properties', 'max_steps', 'reward_scale', 'render_mode'],
            'ParametricStructureEnv': ['self', 'unit', 'box', 'grid', 'n_pts_per_side',
                'target_stretch', 'stiffness', 'damping', 'num_steps',
                'displacement_range', 'reward_mode', 'unit_kwargs'],
        }
        for name, cls in self.classes.items():
            signature = inspect.signature(cls.__init__)
            with self.subTest(name=name):
                self.assertEqual([key for key, value in signature.parameters.items()
                    if value.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD], positional[name])
                policy = signature.parameters['geometry_policy']
                self.assertEqual(policy.kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertEqual(policy.default, 'reject_internal_points')
                env = self.create(name)
                self.assertIsNone(env.engine)
                self.assertEqual(env.geometry_policy, 'reject_internal_points')
        self.assertEqual(self.calls, [])

    def test_invalid_constructor_policy_before_factory_and_provider(self):
        class StringSubclass(str):
            pass
        for name in SOURCES:
            for invalid in (None, True, 0, 'auto', 'ENDPOINT_CHORD',
                            StringSubclass('endpoint_chord')):
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaises(H['InvalidGeometryPolicyError']):
                        self.create(name, geometry_policy=invalid)
        self.assertEqual(self.calls, [])

    def test_strict_actual_graph_rejection_precedes_first_engine_init(self):
        self.current_graph = graph(True)
        for name in SOURCES:
            self.calls.clear()
            env = self.create(name)
            with self.subTest(name=name), self.assertRaises(
                    H['UnsupportedInternalPointsError']) as caught:
                env.step(self.action(env))
            self.assertEqual(caught.exception.rejected_edge_ids, (9,))
            self.assertEqual([call[0] for call in self.calls], ['graph'])
            self.assertIsNone(env.engine)
            self.assertIsNone(env._current_graph)
            if name == 'FiberDesignEnv':
                self.assertEqual(env._history, [])

    def test_actual_node_fixture_accepted_without_rewriting_sampling(self):
        # This duck fixture models actual intermediate graph nodes, not a real
        # generator invocation. n_pts_per_side is not treated as internal_points.
        for name in SOURCES:
            self.calls.clear()
            env = self.create(name)
            result = env.step(self.action(env, n_internal=4))
            with self.subTest(name=name):
                self.assertEqual([call[0] for call in self.calls],
                                 ['graph', 'construct', 'solve'])
                request = self.calls[0][1]
                if name == 'FiberDesignEnv':
                    self.assertEqual(request['n_internal'], 4)
                    self.assertEqual(env.action_space['n_internal'], ('discrete', 6))
                else:
                    self.assertEqual(request['n_pts_per_side'], 2)
                    self.assertEqual(len(request['point_displacements']), 8)
                    self.assertNotIn('n_internal', request)
                self.assertIs(self.calls[2][1], self.current_graph)
                self.assertEqual(self.calls[2][2]['geometry_policy'], 'reject_internal_points')
                self.assertEqual(result[-1]['taichi_geometry']['ignored_internal_point_edge_ids'], [])
                self.assertIs(env._current_graph, self.current_graph)

    def test_explicit_chord_dispatch_and_provenance_preserve_graph(self):
        self.current_graph = graph(True)
        edges = self.current_graph.edges
        point_refs = {key: value.internal_points for key, value in edges.items()}
        for name in SOURCES:
            self.calls.clear()
            env = self.create(name, geometry_policy='endpoint_chord')
            result = env.step(self.action(env, n_internal=3))
            with self.subTest(name=name):
                record = result[-1]['taichi_geometry']
                self.assertEqual(record['policy'], 'endpoint_chord')
                self.assertEqual(record['representation'], 'endpoint_chord')
                self.assertEqual(record['ignored_internal_point_edge_ids'], [9])
                self.assertEqual(record, self.result.metadata['taichi_geometry'])
                self.assertEqual(self.result.metadata['existing'], {'retained': True})
                self.assertEqual(self.calls[-1][2]['geometry_policy'], 'endpoint_chord')
                self.assertIs(self.calls[-1][1], self.current_graph)
                self.assertIs(self.current_graph.edges, edges)
                for key, points in point_refs.items():
                    self.assertIs(edges[key].internal_points, points)
                if name == 'FiberStructureEnv':
                    self.assertIs(env._last_result, self.result)
                elif name == 'ParametricStructureEnv':
                    self.assertIs(result[1], self.result)

    def test_preflight_checks_each_graph_even_after_engine_exists(self):
        for name in SOURCES:
            self.current_graph = graph()
            env = self.create(name)
            env.step(self.action(env))
            engine = env.engine
            self.current_graph = graph(True)
            self.calls.clear()
            with self.subTest(name=name), self.assertRaises(H['UnsupportedInternalPointsError']):
                env.step(self.action(env))
            self.assertEqual([call[0] for call in self.calls], ['graph'])
            self.assertIs(env.engine, engine)

    def test_discrete_geometry_rejection_not_solver_or_reward_penalty(self):
        env = self.create('FiberDesignEnv')
        self.current_graph = graph(True)
        with patch.object(env, '_compute_reward', side_effect=AssertionError('reward touched')):
            with self.assertRaises(H['UnsupportedInternalPointsError']):
                env.step(self.action(env, n_internal=2))
        self.assertEqual(env._history, [])
        self.assertEqual([call[0] for call in self.calls], ['graph'])
        # The action's request is preserved, even though actual storage rejected it.
        self.assertEqual(self.calls[0][1]['n_internal'], 2)

    def test_geometry_errors_from_factory_or_solver_propagate_and_seams_restore(self):
        def reject(*args, **kwargs):
            raise H['UnsupportedInternalPointsError']([91])
        for name in SOURCES:
            env = self.create(name)
            with patch.dict(self.namespace, pattern_2d=reject):
                with self.subTest(name=name, seam='factory'), self.assertRaises(
                        H['UnsupportedInternalPointsError']):
                    env.step(self.action(env))
            self.assertIs(self.namespace['pattern_2d'], self.factory)
            env.engine = types.SimpleNamespace(stretch_test=reject)
            with self.subTest(name=name, seam='solver'), self.assertRaises(
                    H['UnsupportedInternalPointsError']):
                env.step(self.action(env))
            # These are local collaborator seams, not mutations of installed modules.
            self.assertIs(self.namespace['TaichiEngine'], self.Engine)

    def test_ordinary_solver_failure_retains_existing_penalties(self):
        def fail(*args, **kwargs):
            raise RuntimeError('inert solver failure')
        for name in SOURCES:
            env = self.create(name)
            env.engine = types.SimpleNamespace(stretch_test=fail)
            output = env.step(self.action(env))
            with self.subTest(name=name):
                self.assertEqual(output[2] if name == 'ParametricStructureEnv' else output[1],
                                 -1e6 if name == 'ParametricStructureEnv' else -10.)
                self.assertEqual(output[-1]['error'], 'inert solver failure')
                self.assertNotIn('taichi_geometry', output[-1])

    def test_factory_signatures_forward_explicit_and_default_policy(self):
        calls = []
        def constructor(**kwargs):
            calls.append(kwargs)
            return kwargs
        namespace = {**H, **{name: constructor for name in SOURCES}}
        make = load_factory(namespace)
        legacy = load_functions('fibernet/rl/parametric.py', ['create_rl_environment'],
                                namespace)['create_rl_environment']
        for factory in (make, legacy):
            policy = inspect.signature(factory).parameters['geometry_policy']
            self.assertEqual(policy.kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertEqual(policy.default, 'reject_internal_points')
        for env_type in ('structure', 'design', 'parametric'):
            self.assertEqual(make(env_type)['geometry_policy'], 'reject_internal_points')
            self.assertEqual(make(env_type, geometry_policy='endpoint_chord')['geometry_policy'],
                             'endpoint_chord')
        self.assertEqual(legacy()['geometry_policy'], 'reject_internal_points')
        self.assertEqual(legacy(geometry_policy='endpoint_chord')['geometry_policy'], 'endpoint_chord')
        calls.clear()
        with self.assertRaises(H['InvalidGeometryPolicyError']):
            make(geometry_policy='auto')
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()

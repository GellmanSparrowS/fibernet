import contextlib
import io
import unittest
from unittest.mock import patch

from fibernet.rl.utils import evaluate_agent


class Env:
    def __init__(self, truncated=False):
        self.resets = 0
        self.actions = []
        self.renders = 0
        self.truncated = truncated

    def reset(self):
        self.resets += 1
        return 'observation', {'reset': True}

    def step(self, action):
        self.actions.append(action)
        return 'next', 2.5, not self.truncated, self.truncated, {'final': True}

    def render(self):
        self.renders += 1


class Plain:
    def __init__(self):
        self.calls = []

    def predict(self, obs):
        self.calls.append(obs)
        return 3


class ProtocolTests(unittest.TestCase):
    def test_plain_and_results(self):
        a, e = Plain(), Env()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r = evaluate_agent(a, e, n_episodes=2, render=True, verbose=False)
        self.assertEqual(out.getvalue(), '')
        self.assertEqual(a.calls, ['observation'] * 2)
        self.assertEqual(e.actions, [3, 3])
        self.assertEqual((e.resets, e.renders), (2, 2))
        self.assertEqual(r, dict(mean_reward=2.5, std_reward=0.0,
            max_reward=2.5, min_reward=2.5, rewards=[2.5, 2.5],
            episode_infos=[dict(episode=i, reward=2.5, steps=1,
                                info={'final': True}) for i in range(2)]))

    def test_sb3_tuple_and_keyword(self):
        class Agent:
            def predict(self, obs, state=None, episode_start=None, deterministic=False):
                self.args = obs, state, episode_start, deterministic
                return 5, 'hidden'
        a, e = Agent(), Env(truncated=True)
        evaluate_agent(a, e, n_episodes=1, verbose=False)
        self.assertEqual(a.args, ('observation', None, None, True))
        self.assertEqual(e.actions, [5])

    def test_internal_typeerror_exactly_once(self):
        for with_keyword in (False, True):
            calls = []
            failure = TypeError('internal model bug')
            def plain(obs):
                calls.append(obs)
                raise failure
            def keyword(obs, deterministic=False):
                calls.append(obs)
                raise failure
            a = type('Agent', (), {})()
            a.predict = keyword if with_keyword else plain
            e = Env()
            with self.assertRaises(TypeError) as raised:
                evaluate_agent(a, e, n_episodes=1, verbose=False)
            self.assertIs(raised.exception, failure)
            self.assertEqual(calls, ['observation'])
            self.assertEqual(e.actions, [])

    def test_dictionary_plain_and_sb3(self):
        for predict, expected in ((lambda obs: 7, 7),
                                   (lambda obs, deterministic: (8, None), 8)):
            model = type('Model', (), {})()
            model.predict = predict
            e = Env()
            evaluate_agent({'model': model}, e, n_episodes=1, verbose=False)
            self.assertEqual(e.actions, [expected])

    def test_act_precedence_and_no_tuple_unwrap(self):
        class Agent:
            def act(self, obs):
                return (1, 2)
        e = Env()
        evaluate_agent(Agent(), e, n_episodes=1, verbose=False)
        self.assertEqual(e.actions, [(1, 2)])
        a = Plain()
        a.act = lambda obs: self.fail('predict must take precedence')
        evaluate_agent(a, Env(), n_episodes=1, verbose=False)

    def test_unsupported_before_reset(self):
        for predict in (None, lambda: 1, lambda obs, required: 1):
            a = type('Agent', (), {})()
            a.predict = predict
            e = Env()
            with self.assertRaisesRegex(TypeError, 'predict\\(obs\\).*adapter'):
                evaluate_agent(a, e, n_episodes=1, verbose=False)
            self.assertEqual(e.resets, 0)
        with self.assertRaisesRegex(ValueError, 'predict.*act'):
            evaluate_agent(object(), Env(), n_episodes=1, verbose=False)
        with self.assertRaisesRegex(TypeError, 'predict'):
            evaluate_agent({'model': object()}, Env(), n_episodes=1, verbose=False)

    def test_uninspectable_does_not_call(self):
        a = Plain()
        with patch('fibernet.rl.utils._inspect.signature', side_effect=ValueError('opaque')):
            with self.assertRaisesRegex(TypeError, 'inspectable') as raised:
                evaluate_agent(a, Env(), n_episodes=1, verbose=False)
        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertEqual(a.calls, [])

    def test_keyword_variants(self):
        calls = []
        def kwonly(obs, *, deterministic):
            calls.append(deterministic)
            return 1
        def kwargs(obs, **kwargs):
            calls.append(kwargs['deterministic'])
            return 1
        def positional(obs, /):
            calls.append('plain')
            return 1
        for predict in (kwonly, kwargs, positional):
            a = type('Agent', (), {})()
            a.predict = predict
            evaluate_agent(a, Env(), n_episodes=1, verbose=False)
        self.assertEqual(calls, [True, True, 'plain'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

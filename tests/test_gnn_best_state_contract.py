"""Pure control-flow tests: actual train_gnn AST, no Torch/NumPy import or fit.

Run directly with Python, or collect with pytest. The doubles implement only the
trainer protocol; they do not validate numerical training or backend compatibility.
"""
import ast
from collections import OrderedDict
import copy
import json
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'fibernet' / 'ml' / 'gnn.py'


class Array(list):
    def __getitem__(self, key):
        if isinstance(key, list):
            return Array(list.__getitem__(self, i) for i in key)
        value = list.__getitem__(self, key)
        return Array(value) if isinstance(key, slice) else value


class FakeHistory:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.lr_history = []

    def update(self, epoch, train_loss, val_loss, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.lr_history.append(lr)


class FakeModel:
    def __init__(self):
        self.weight = [0]
        self.buffer = [0]
        self.extra = {'events': ['initial']}
        self.metadata = {'': {'step': 0}}
        self.training = True
        self.steps = 0
        self.snapshots = 0
        self.loads = 0
        self.calls = []
        self.fail_load = False

    def parameters(self):
        return self

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, graphs):
        self.calls.append((self.training, [g['id'] for g in graphs]))
        return types.SimpleNamespace(squeeze=lambda axis: self.training)

    def state_dict(self):
        self.snapshots += 1
        state = OrderedDict(weight=self.weight, buffer=self.buffer, _extra_state=self.extra)
        state._metadata = self.metadata
        return state

    def load_state_dict(self, state):
        self.loads += 1
        if self.fail_load:
            raise RuntimeError('deliberate restore failure')
        self.weight[:] = state['weight']
        self.buffer[:] = state['buffer']
        self.extra = copy.deepcopy(state['_extra_state'])
        self.metadata = copy.deepcopy(state._metadata)


class NoGrad:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        return False


class Harness:
    def __init__(self, losses, source=SOURCE):
        self.model = FakeModel()
        self.losses = iter(losses)
        self.optimizers = 0
        self.loss_calls = []
        self.preflight_calls = 0
        harness = self

        class Optimizer:
            def __init__(self, model, lr, weight_decay):
                harness.optimizers += 1
                self.model = model
                self.param_groups = [{'lr': lr}]

            def zero_grad(self):
                pass

            def step(self):
                model = self.model
                model.steps += 1
                model.weight[0] = model.steps
                model.buffer[0] = 10 * model.steps
                model.extra['events'].append(model.steps)
                model.metadata['']['step'] = model.steps

        def loss(training, labels):
            self.loss_calls.append((training, list(labels)))
            value = 10.0 if training else next(self.losses)
            return types.SimpleNamespace(item=lambda: value, backward=lambda: None)

        def preflight(model, graphs, labels, groups, epochs, *args):
            self.preflight_calls += 1
            if epochs <= 0:
                raise ValueError('group mode requires positive epochs')
            return Array(labels), [0, 1], [2, 3], {
                'training': {'best_state_restored': False},
                'split': {'validation_role': 'validation'},
            }

        np = types.SimpleNamespace(
            float32='float32', asarray=lambda value, **kwargs: Array(value),
            random=types.SimpleNamespace(RandomState=lambda seed: types.SimpleNamespace(
                permutation=lambda n: list(reversed(range(n))))))
        torch = types.SimpleNamespace(
            float32='float32', tensor=lambda value, **kwargs: value, no_grad=NoGrad,
            optim=types.SimpleNamespace(AdamW=Optimizer),
            nn=types.SimpleNamespace(utils=types.SimpleNamespace(clip_grad_norm_=lambda *a: None)))
        namespace = dict(copy=copy, math=math, np=np, torch=torch,
                         F=types.SimpleNamespace(mse_loss=loss), _gnn_group_preflight=preflight)
        tree = ast.parse(source.read_text())
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == 'train_gnn'
                        and any(arg.arg == 'model' for arg in node.args.args))
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        isolated = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
        exec(compile(isolated, str(source), 'exec'), namespace)
        self.train_gnn = namespace['train_gnn']

    def run(self, **kwargs):
        module = types.ModuleType('fibernet.ml.training')
        module.TrainingHistory = FakeHistory
        options = dict(epochs=3, val_split=.5, batch_size=32, early_stopping=0, verbose=False)
        options.update(kwargs)
        with patch.dict(sys.modules, {'fibernet.ml.training': module}):
            return self.train_gnn(self.model, [{'id': i} for i in range(4)],
                                  [10, 20, 30, 40], **options)


class BestStateContract(unittest.TestCase):
    def test_default_and_explicit_false_keep_legacy_shape_and_last_state(self):
        for option in ({}, {'restore_best': False}):
            h = Harness([3., 1., 2.])
            result = h.run(**option)
            self.assertEqual(set(result), {'history', 'best_val_loss'})
            self.assertEqual(result['history'].val_losses, [3., 1., 2.])
            self.assertEqual(result['history'].train_losses, [10.] * 3)
            self.assertEqual(result['best_val_loss'], 1.)
            self.assertEqual(h.model.weight, [3])
            self.assertEqual((h.model.snapshots, h.model.loads), (0, 0))

    def test_nonmonotonic_best_is_an_independent_complete_state(self):
        h = Harness([3., 1., 2.])
        result = h.run(restore_best=True)
        s = result['model_selection']
        self.assertEqual((s['best_epoch'], s['model_state_epoch'], s['best_val_loss']), (1, 1, 1.))
        self.assertEqual((h.model.weight, h.model.buffer), ([2], [20]))
        self.assertEqual(h.model.extra, {'events': ['initial', 1, 2]})
        self.assertEqual(h.model.metadata, {'': {'step': 2}})
        self.assertTrue(s['restore_best_requested'])
        self.assertTrue(s['best_state_restored'])
        self.assertIsNone(s['no_restore_reason'])
        self.assertFalse(s['optimizer_state_restored'])
        self.assertFalse(s['scheduler_state_restored'])
        self.assertEqual((h.model.steps, h.model.snapshots, h.model.loads), (3, 2, 1))
        self.assertFalse(h.model.training)  # Existing final validation mode retained.
        self.assertEqual(s['criterion'], 'minimum_finite_nonnegative_validation_mse')
        self.assertEqual(s['epoch_index_base'], 0)
        json.dumps(s, allow_nan=False)

    def test_first_tie_and_early_stop_keep_existing_loop_semantics(self):
        h = Harness([1., 1., 8.])
        result = h.run(restore_best=True, early_stopping=1)
        self.assertEqual(h.model.steps, 2)
        self.assertEqual(h.model.weight, [1])
        self.assertEqual(result['model_selection']['best_epoch'], 0)
        self.assertEqual(result['model_selection']['epochs_run'], 2)
        self.assertEqual(result['model_selection']['tie_break'], 'first')

    def test_no_validation_never_selects_synthetic_zero(self):
        for restore in (False, True):
            h = Harness([])
            result = h.run(restore_best=restore, val_split=0, early_stopping=1)
            self.assertEqual(result['history'].val_losses, [0., 0.])
            self.assertEqual(result['best_val_loss'], 0.)  # Preserved legacy value.
            self.assertEqual(h.model.steps, 2)
            self.assertEqual((h.model.snapshots, h.model.loads), (0, 0))
            if restore:
                s = result['model_selection']
                self.assertEqual(s['no_restore_reason'], 'no_validation')
                self.assertIsNone(s['best_epoch'])
                self.assertIsNone(s['best_val_loss'])
                self.assertEqual(s['model_state_epoch'], 1)
                self.assertFalse(s['best_state_restored'])
                self.assertEqual(s['eligible_validation_epochs'], 0)
                json.dumps(s, allow_nan=False)

    def test_zero_epochs_preserves_model_and_no_selection(self):
        for val_split in (0, .5):
            h = Harness([])
            result = h.run(restore_best=True, epochs=0, val_split=val_split)
            s = result['model_selection']
            self.assertEqual(s['no_restore_reason'], 'no_epochs')
            self.assertEqual(s['epochs_run'], 0)
            self.assertIsNone(s['best_epoch'])
            self.assertIsNone(s['model_state_epoch'])
            self.assertIsNone(s['best_val_loss'])
            self.assertEqual(h.model.weight, [0])
            self.assertEqual((h.model.snapshots, h.model.loads), (0, 0))
            self.assertEqual(result['best_val_loss'], float('inf'))
            json.dumps(s, allow_nan=False)

    def test_invalid_losses_do_not_select_a_state(self):
        h = Harness([float('nan'), float('inf'), -float('inf'), -1.])
        result = h.run(restore_best=True, epochs=4)
        s = result['model_selection']
        self.assertEqual(s['no_restore_reason'], 'no_finite_nonnegative_validation_loss')
        self.assertEqual(s['invalid_validation_epochs'], 4)
        self.assertEqual(s['eligible_validation_epochs'], 0)
        self.assertIsNone(s['best_epoch'])
        self.assertIsNone(s['best_val_loss'])
        self.assertFalse(s['best_state_restored'])
        self.assertEqual((h.model.weight, h.model.loads), ([4], 0))
        json.dumps(s, allow_nan=False)

    def test_later_usable_loss_survives_legacy_negative_infinity(self):
        h = Harness([-float('inf'), 2., 3.])
        result = h.run(restore_best=True)
        self.assertEqual(result['best_val_loss'], -float('inf'))
        self.assertEqual(result['model_selection']['best_val_loss'], 2.)
        self.assertEqual(result['model_selection']['best_epoch'], 1)
        self.assertEqual(result['model_selection']['invalid_validation_epochs'], 1)
        self.assertEqual(h.model.weight, [2])
        json.dumps(result['model_selection'], allow_nan=False)

    def test_zero_loss_is_eligible_and_an_invocation_cannot_reuse_old_state(self):
        h = Harness([0., 1., 2.])
        first = h.run(restore_best=True)
        self.assertEqual(first['model_selection']['best_epoch'], 0)
        self.assertEqual(h.model.weight, [1])
        second = h.run(restore_best=True, epochs=0)
        self.assertIsNone(second['model_selection']['best_epoch'])
        self.assertFalse(second['model_selection']['best_state_restored'])
        self.assertEqual(h.model.loads, 1)

    def test_group_record_matches_actual_restore_and_is_not_aliased(self):
        for restore in (False, True):
            h = Harness([3., 1., 2.])
            result = h.run(restore_best=restore, split='group', groups=['a', 'a', 'b', 'b'])
            record = result['training_data']
            self.assertEqual(record['training']['best_state_restored'], restore)
            self.assertEqual(record['training']['epochs_run'], 3)
            self.assertEqual(h.model.calls, [(True, [0, 1]), (False, [2, 3])] * 3)
            self.assertEqual(h.loss_calls, [(True, [10, 20]), (False, [30, 40])] * 3)
            self.assertEqual(h.preflight_calls, 1)
            if restore:
                self.assertEqual(record['model_selection'], result['model_selection'])
                result['model_selection']['best_epoch'] = 99
                self.assertEqual(record['model_selection']['best_epoch'], 1)
            else:
                self.assertNotIn('model_selection', result)
                self.assertNotIn('model_selection', record)
            json.dumps(record, allow_nan=False)

    def test_existing_group_preflight_remains_before_optimizer(self):
        h = Harness([])
        with self.assertRaisesRegex(ValueError, 'positive epochs'):
            h.run(restore_best=True, split='group', groups=['a', 'a', 'b', 'b'], epochs=0)
        self.assertEqual(h.optimizers, 0)

    def test_restore_option_requires_boolean_before_any_mutation(self):
        for value in (None, 0, 1, 'yes', [], {}):
            h = Harness([])
            with self.assertRaisesRegex(ValueError, 'restore_best must be a bool'):
                h.run(restore_best=value)
            self.assertEqual(h.optimizers, 0)
            self.assertEqual(h.model.steps, 0)

    def test_restore_failure_propagates_instead_of_claiming_success(self):
        h = Harness([3., 1., 2.])
        h.model.fail_load = True
        with self.assertRaisesRegex(RuntimeError, 'deliberate restore failure'):
            h.run(restore_best=True)
        self.assertEqual(h.model.loads, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)

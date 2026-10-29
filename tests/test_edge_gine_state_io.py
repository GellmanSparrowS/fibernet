"""Fresh synthetic own-state fidelity only; no training or physics execution."""
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torch_geometric')
from fibernet.ml import edge_gine as gine
from fibernet.ml import edge_gine_io as io
from fibernet.core.structure_graph import SNode, SEdge, StructureGraph
from fibernet.ml.graph_data import graph_batch

TARGET = dict(name='declared_force', unit='N', kind='graph_scalar', encoding='identity')
ORIGIN = dict(kind='caller_asserted', description='Fresh deterministic synthetic fixture, not trained', source_ref=None)


def config(pool='mean'):
    return gine.make_config(features=[dict(name='radius', source={'snapshot': 'radius'},
        unit='m', offset=0., scale=1., required=True, parity='even')], length_unit='m',
        coordinate_frame='laboratory', node_offset=[0., 0., 0., 0.],
        node_scale=[1., 1., 1., 1.], pooling=pool)


def model(pool='mean'):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(271)
        return gine.GINERegressor(config(pool))


def prepared(c):
    graph = StructureGraph(2)
    graph.nodes[0] = SNode(np.array([0., 0., 0.]), 0)
    graph.nodes[1] = SNode(np.array([1., 0., 0.]), 1)
    graph.edges[0] = SEdge(0, 1, 0, radius=.25)
    batch = graph_batch([graph], graph_ids=['synthetic'], length_unit='m',
                        coordinate_frame='laboratory', edge_features=['radius'])
    return gine.prepare_gine(batch, config=c, source_groups={'synthetic': 'synthetic-only'})


@pytest.mark.parametrize('pool', ['mean', 'sum'])
def test_roundtrip_outputs_ownership_rng_mode_gradients(tmp_path, pool):
    original = model(pool)
    original.train()
    for parameter in original.parameters():
        parameter.grad = torch.ones_like(parameter)
    state = {k: v.clone() for k, v in original.state_dict().items()}
    gradients = [p.grad.clone() for p in original.parameters()]
    modes = [m.training for m in original.modules()]
    target = copy.deepcopy(TARGET); origin = copy.deepcopy(ORIGIN)
    rng = torch.get_rng_state().clone()
    path = tmp_path / 'state.json'
    assert io.save_gine_state(original, path, target=target, origin=origin) is None
    loaded, info = io.load_gine_state(path)
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == [m.training for m in original.modules()]
    assert all(torch.equal(a, p.grad) for a, p in zip(gradients, original.parameters()))
    assert all(torch.equal(state[k], v) for k, v in original.state_dict().items())
    assert all(torch.equal(state[k], v) for k, v in loaded.state_dict().items())
    assert not any(m.training for m in loaded.modules())
    assert info['target'] == target and info['origin'] == origin and info['model_config'] == config(pool)
    fixture = prepared(config(pool))
    original.eval()
    with torch.no_grad():
        assert torch.equal(original(fixture), loaded(fixture))
    target['unit'] = 'changed'; origin['description'] = 'changed'
    assert info['target']['unit'] == 'N' and info['origin'] == ORIGIN
    info['model_config']['pooling'] = 'changed'
    with torch.no_grad():
        next(loaded.parameters()).add_(1.)
    assert not torch.equal(next(original.parameters()), next(loaded.parameters()))
    assert json.loads(loaded._config_json)['pooling'] == pool


@pytest.fixture
def saved(tmp_path):
    path = tmp_path / 'state.json'
    io.save_gine_state(model(), path, target=TARGET)
    return path


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(extra=True),
    lambda r: r.update(schema='fibernet.gine_artifact.v1'),
    lambda r: r['model_config'].update(version='fibernet.edge_gine.material_view.v1'),
    lambda r: r['runtime'].update(torch='changed'),
    lambda r: r['runtime']['sources'].update({'edge_gine.py': '0' * 64}),
    lambda r: r['target'].update(unit=''),
    lambda r: r['target'].update(encoding='log'),
    lambda r: r.update(origin={'kind': 'verified', 'description': 'x', 'source_ref': None}),
    lambda r: r['state'].pop('head.bias'),
    lambda r: r['state']['head.bias'].update(shape=[True]),
    lambda r: r['state']['head.bias'].update(values=[]),
    lambda r: r['state']['head.bias'].update(values=[0.1]),
    lambda r: r['state']['head.bias'].update(values=[True]),
    lambda r: r['state']['layers.0.eps'].update(values=[-0.0]),
])
def test_invalid_records_reject_before_construction(saved, mutation):
    record = json.loads(saved.read_bytes()); mutation(record)
    saved.write_text(json.dumps(record))
    with patch.object(io, '_new_model', side_effect=AssertionError('must validate first')):
        with pytest.raises(ValueError):
            io.load_gine_state(saved)


@pytest.mark.parametrize('raw', [b'{"schema":1,"schema":2}', b'{"a":NaN}',
    b'[' * 34 + b'0' + b']' * 34, b'"' + b'x' * 4097 + b'"', b' ' * (io.MAX_BYTES + 1)])
def test_bounded_json(saved, raw):
    saved.write_bytes(raw)
    with pytest.raises(ValueError):
        io.load_gine_state(saved)


@pytest.mark.parametrize('change', [
    lambda m: setattr(m, 'pooling', 'sum'),
    lambda m: setattr(m, 'head', torch.nn.Sequential(m.head)),
    lambda m: setattr(m.layers[0].nn, '1', torch.nn.Sigmoid()),
    lambda m: setattr(m.layers[0], 'flow', 'target_to_source'),
    lambda m: setattr(m, 'forward', lambda x: x),
    lambda m: m.register_forward_hook(lambda *args: None),
    lambda m: m.double(),
    lambda m: m.layers[0].eps.fill_(1.),
])
def test_modified_model_refused_without_replacement(saved, change):
    original = model(); change(original)
    before = saved.read_bytes()
    with pytest.raises(ValueError):
        io.save_gine_state(original, saved, target=TARGET)
    assert saved.read_bytes() == before


def test_subclass_rejected(saved):
    class Other(gine.GINERegressor):
        pass
    with pytest.raises(ValueError):
        io.save_gine_state(Other(config()), saved, target=TARGET)


@pytest.mark.parametrize('operation', ['replace', 'chmod', 'fdopen'])
def test_failure_safe_write(saved, operation):
    before = saved.read_bytes(); mode = saved.stat().st_mode
    with patch.object(io.os, operation, side_effect=OSError('injected ordinary failure')):
        with pytest.raises(OSError):
            io.save_gine_state(model(), saved, target=TARGET)
    assert saved.read_bytes() == before and saved.stat().st_mode == mode
    assert list(saved.parent.iterdir()) == [saved]


def test_destination_refusals(saved):
    link = saved.parent / 'link'; link.symlink_to(saved)
    with pytest.raises(ValueError):
        io.save_gine_state(model(), link, target=TARGET)
    with pytest.raises(ValueError):
        io.save_gine_state(model(), saved.parent, target=TARGET)
    if os.name == 'posix':
        saved.chmod(0o1600)
        with pytest.raises(ValueError):
            io.save_gine_state(model(), saved, target=TARGET)


def test_backend_error_is_explicit(saved):
    with patch.object(gine, '_BACKEND_ERROR', ImportError('deliberately missing')):
        with pytest.raises(ImportError, match='optional compatible torch'):
            io.load_gine_state(saved)
        with pytest.raises(ImportError, match='optional compatible torch'):
            io.save_gine_state(None, saved, target=TARGET)


def test_no_training_deserialization_or_network(saved):
    with patch.object(torch, 'load', side_effect=AssertionError('pickle loading forbidden')), \
         patch.object(torch.Tensor, 'backward', side_effect=AssertionError('training forbidden')):
        io.save_gine_state(model(), saved, target=TARGET)
        io.load_gine_state(saved)
    source = Path(io.__file__).read_text()
    assert 'import pickle' not in source and 'torch.load(' not in source


def test_failed_construction_and_bad_target_preserve_rng(saved):
    original = model(); before = saved.read_bytes(); rng = torch.get_rng_state().clone()
    original_init = gine.GINERegressor.__init__
    def failing_init(self, config):
        original_init(self, config)
        raise ValueError('injected construction failure')
    with patch.object(gine.GINERegressor, '__init__', failing_init):
        with pytest.raises(ValueError, match='construction'):
            io.load_gine_state(saved)
        with pytest.raises(ValueError, match='construction'):
            io.save_gine_state(original, saved, target=TARGET)
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError):
        io.save_gine_state(original, saved, target=dict(TARGET, unit=''))
    assert saved.read_bytes() == before and torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('method', ['propagate', 'message', 'aggregate', 'update'])
def test_pyg_method_override_refused(saved, method):
    original = model()
    setattr(original.layers[0], method, lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match='override'):
        io.save_gine_state(original, saved, target=TARGET)


def test_parameter_hooks_and_parametrizations_refused(saved):
    original = model()
    next(original.parameters()).register_hook(lambda grad: grad)
    with pytest.raises(ValueError, match='hooks'):
        io.save_gine_state(original, saved, target=TARGET)
    original = model()
    torch.nn.utils.parametrize.register_parametrization(original.head, 'weight', torch.nn.Identity())
    with pytest.raises(ValueError):
        io.save_gine_state(original, saved, target=TARGET)


def test_reader_special_files_reject_before_open(saved):
    link = saved.parent / 'read-link'; link.symlink_to(saved)
    with pytest.raises(ValueError, match='regular'):
        io.load_gine_state(link)
    if hasattr(os, 'mkfifo'):
        fifo = saved.parent / 'fifo'; os.mkfifo(fifo)
        with patch('builtins.open', side_effect=AssertionError('must not open FIFO')):
            with pytest.raises(ValueError, match='regular'):
                io.load_gine_state(fifo)


@pytest.mark.parametrize('operation', ['write', 'flush', 'close'])
def test_stream_failures_preserve_file_and_clean_stage(saved, operation):
    before = saved.read_bytes(); real_fdopen = os.fdopen
    class Stream:
        def __init__(self, fd, mode): self.handle = real_fdopen(fd, mode)
        def __enter__(self): return self
        def write(self, raw):
            if operation == 'write': raise OSError('injected write')
            return self.handle.write(raw)
        def flush(self):
            if operation == 'flush': raise OSError('injected flush')
            self.handle.flush()
        def __exit__(self, *args):
            self.handle.close()
            if operation == 'close': raise OSError('injected close')
    with patch.object(io.os, 'fdopen', Stream):
        with pytest.raises(OSError):
            io.save_gine_state(model(), saved, target=TARGET)
    assert saved.read_bytes() == before and list(saved.parent.iterdir()) == [saved]


def test_unknown_state_nonfinite_item_limit_and_missing_destination(saved):
    record = json.loads(saved.read_bytes())
    record['state']['head.bias']['values'] = [float('inf')]
    saved.write_text(json.dumps(record))
    with pytest.raises(ValueError): io.load_gine_state(saved)
    saved.write_text(json.dumps([0] * 300001))
    with pytest.raises(ValueError): io.load_gine_state(saved)
    missing = saved.parent / 'new.json'
    with patch.object(io.os, 'replace', side_effect=OSError('injected')):
        with pytest.raises(OSError): io.save_gine_state(model(), missing, target=TARGET)
    assert not missing.exists()

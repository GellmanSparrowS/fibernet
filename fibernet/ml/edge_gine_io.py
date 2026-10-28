"""Bounded, exact-runtime JSON state IO for unmodified current CPU GINE models.

This is inference-state fidelity, not training provenance or quality validation.
No historical artifact conversion, unit conversion, or compatibility override.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import struct
import tempfile
import warnings

import numpy as np
from . import edge_gine as _gine
from .graph_training_contract import MAX_BYTES, canonical_bytes, parse_json

_SCHEMA = 'fibernet.gine_inference_state.v1'
_FIELDS = {'schema', 'model_config', 'target', 'state', 'runtime', 'origin'}
_SOURCES = ('edge_gine.py', 'graph_data.py', 'graph_training_contract.py', 'edge_gine_io.py')


def _json(value):
    canonical_bytes(value)
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(raw) > MAX_BYTES:
        raise ValueError('inference state JSON byte limit')
    return raw


def _keys(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(label + ' fields mismatch')


def _text(value, label):
    if type(value) is not str or not value.strip() or len(value.encode('utf-8')) > 4096:
        raise ValueError(label + ' must be bounded nonempty text')


def _metadata(target, origin):
    _keys(target, ('name', 'unit', 'kind', 'encoding'), 'target')
    _text(target['name'], 'target name'); _text(target['unit'], 'target unit')
    if target['kind'] != 'graph_scalar' or target['encoding'] != 'identity':
        raise ValueError('only identity graph_scalar target supported')
    if origin is not None:
        _keys(origin, ('kind', 'description', 'source_ref'), 'origin')
        if origin['kind'] != 'caller_asserted':
            raise ValueError('origin must be caller_asserted, not authenticated provenance')
        _text(origin['description'], 'origin description')
        if origin['source_ref'] is not None:
            _text(origin['source_ref'], 'origin source_ref')


def _runtime():
    _gine._require()
    import torch_geometric
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'torch': str(_gine.torch.__version__), 'pyg': torch_geometric.__version__,
            'device': 'cpu', 'dtype': 'float32',
            'sources': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in _SOURCES}}


def _config(value):
    if type(value) is not dict or value.get('version') != 'fibernet.edge_gine.v1':
        raise ValueError('only current edge_gine.v1 supported; material_view and legacy artifacts reject')
    return _gine._config(value)


def _shapes(config):
    shapes = {'input_proj.weight': [16, 4], 'input_proj.bias': [16],
              'head.weight': [1, 16], 'head.bias': [1]}
    for i in range(2):
        prefix = 'layers.' + str(i) + '.'
        shapes[prefix + 'eps'] = [1]
        for j in (0, 2):
            shapes[prefix + 'nn.' + str(j) + '.weight'] = [16, 16]
            shapes[prefix + 'nn.' + str(j) + '.bias'] = [16]
        shapes[prefix + 'lin.weight'] = [16, 2 * len(config['features'])]
        shapes[prefix + 'lin.bias'] = [16]
    return shapes


def _state_arrays(state, config):
    shapes = _shapes(config)
    _keys(state, shapes, 'state')
    arrays = {}
    for name, shape in shapes.items():
        record = state[name]
        _keys(record, ('shape', 'values'), 'state tensor')
        if (type(record['shape']) is not list or any(type(n) is not int for n in record['shape'])
                or record['shape'] != shape):
            raise ValueError('state shape mismatch: ' + name)
        values = record['values']
        if type(values) is not list or len(values) != math.prod(shape):
            raise ValueError('state scalar count mismatch: ' + name)
        packed = []
        for value in values:
            if type(value) is not float or not math.isfinite(value):
                raise ValueError('state values must be finite canonical float32 numbers')
            try:
                raw = struct.pack('<f', value)
            except (OverflowError, struct.error) as exc:
                raise ValueError('state value outside float32') from exc
            restored = struct.unpack('<f', raw)[0]
            if struct.pack('<d', restored) != struct.pack('<d', value):
                raise ValueError('state value is not canonical float32')
            packed.append(raw)
        raw = b''.join(packed)
        if name.endswith('.eps') and raw != b'\0' * 4:
            raise ValueError('fixed eps must be positive zero')
        arrays[name] = np.frombuffer(raw, dtype='<f4').copy().reshape(shape)
    return arrays


def _state_hash(arrays):
    digest = hashlib.sha256(b'fibernet.gine_inference_state.state.v1\0')
    for name in sorted(arrays):
        raw = name.encode('utf-8')
        digest.update(len(raw).to_bytes(4, 'big')); digest.update(raw)
        digest.update(canonical_bytes(list(arrays[name].shape)))
        digest.update(arrays[name].astype('<f4', copy=False).tobytes(order='C'))
    return digest.hexdigest()


def _new_model(config):
    torch = _gine.torch
    # Explicit CPU construction, preserving CPU RNG even on construction failure.
    with torch.random.fork_rng(devices=[]):
        with torch.device('cpu'):
            return _gine.GINERegressor(config).to(dtype=torch.float32)


def _standard_model(model, config):
    if type(model) is not _gine.GINERegressor:
        raise ValueError('exact unmodified GINERegressor required')
    reference = _new_model(config)
    actual = dict(model.named_modules()); expected = dict(reference.named_modules())
    if set(actual) != set(expected) or model.pooling != config['pooling']:
        raise ValueError('model module structure or pooling changed')
    for name, module in actual.items():
        baseline = expected[name]
        if type(module) is not type(baseline):
            raise ValueError('unsupported module replacement: ' + name)
        if 'forward' in vars(module) or hasattr(module, 'parametrizations'):
            raise ValueError('instance forward override or parametrization unsupported')
        if any(value for key, value in vars(module).items() if key.endswith('_hooks')):
            raise ValueError('custom module hooks unsupported')
        # Computation-affecting ordinary module settings, including PyG message flow.
        for key in ('in_features', 'out_features', 'in_channels', 'out_channels',
                    'inplace', 'aggr', 'flow', 'node_dim', 'decomposed_layers',
                    'initial_eps', 'explain', 'fuse'):
            if hasattr(baseline, key) and getattr(module, key) != getattr(baseline, key):
                raise ValueError('unsupported module setting: ' + name + '.' + key)
        for key in ('message', 'aggregate', 'message_and_aggregate', 'update', 'propagate'):
            if key in vars(module):
                current = vars(module)[key]; standard = vars(baseline).get(key)
                if getattr(current, '__func__', current) is not getattr(standard, '__func__', standard):
                    raise ValueError('instance computation override unsupported')
    for parameter in model.parameters():
        if parameter._backward_hooks or getattr(parameter, '_post_accumulate_grad_hooks', None):
            raise ValueError('custom parameter hooks unsupported')
    return reference


def _model_state(model, config):
    torch = _gine.torch
    state = model.state_dict()
    _keys(dict(state), _shapes(config), 'model state')
    result = {}
    for name, tensor in state.items():
        if (type(tensor) is not torch.Tensor or tensor.device.type != 'cpu'
                or tensor.dtype != torch.float32 or tensor.layout != torch.strided):
            raise ValueError('state must be ordinary CPU float32 tensors')
        # Copy only; do not touch gradients, training mode, dtype or storage.
        values = tensor.detach().contiguous().numpy().copy()
        result[name] = {'shape': list(values.shape), 'values': list(struct.unpack('<' + str(values.size) + 'f', values.astype('<f4', copy=False).tobytes()))}
    return result


def _validate(record):
    _keys(record, _FIELDS, 'inference state')
    if record['schema'] != _SCHEMA:
        raise ValueError('unsupported inference state schema; no legacy conversion')
    config = _config(record['model_config'])
    _metadata(record['target'], record['origin'])
    if record['runtime'] != _runtime():
        raise ValueError('exact runtime/source identity mismatch')
    arrays = _state_arrays(record['state'], config)
    return config, arrays


def _write(raw, path):
    out = Path(os.fsdecode(path))
    try:
        info = out.lstat()
    except FileNotFoundError:
        mode = 0o600
    else:
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('destination must be regular, not symlink or special')
        mode = stat.S_IMODE(info.st_mode)
        if mode & 0o7000:
            raise ValueError('destination has unsupported special mode bits')
    fd, temporary = tempfile.mkstemp(prefix='.' + out.name + '.save-', dir=out.parent)
    try:
        handle = os.fdopen(fd, 'wb'); fd = None
        with handle:
            handle.write(raw)
            handle.flush()
        os.chmod(temporary, mode)
        os.replace(temporary, out)
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        except OSError:
            try:
                warnings.warn('Could not remove owned inference-state temporary file: ' + temporary, RuntimeWarning)
            except Exception:
                pass


def save_gine_state(model, path, *, target, origin=None):
    """Save supported unmodified current state; target units are caller-declared.

    origin is None or {kind: caller_asserted, description: text, source_ref: text
    or None}. No reference is fetched and no claim is authenticated. Returns None.
    """
    _gine._require()
    if type(model) is not _gine.GINERegressor:
        raise ValueError('exact unmodified GINERegressor required')
    config = _config(parse_json(model._config_json))
    _metadata(target, origin)
    _standard_model(model, config)
    record = dict(schema=_SCHEMA, model_config=config, target=target,
                  state=_model_state(model, config), runtime=_runtime(), origin=origin)
    raw = _json(record)
    _validate(parse_json(raw))
    _write(raw, path)


def load_gine_state(path):
    """Return a fresh owned eval model and detached info, preserving CPU RNG.

    Exact runtime/source matching is intentionally conservative, including source
    comments. It is not hardware-independent output or model-quality assurance.
    """
    _gine._require()
    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise ValueError('inference state input must be regular, not symlink or special')
    with open(path, 'rb') as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError('inference state input must be a regular file')
        raw = handle.read(MAX_BYTES + 1)
    record = parse_json(raw)
    config, arrays = _validate(record)
    model = _new_model(config)
    model.load_state_dict({name: _gine.torch.from_numpy(value.copy()) for name, value in arrays.items()}, strict=True)
    expected_hash = _state_hash(arrays)
    actual_hash = _state_hash(_state_arrays(_model_state(model, config), config))
    if actual_hash != expected_hash:
        raise ValueError('post-load state hash mismatch')
    model.eval()
    info = {key: record[key] for key in ('schema', 'model_config', 'target', 'runtime', 'origin')}
    info.update(state_sha256=actual_hash, artifact_sha256=hashlib.sha256(raw).hexdigest())
    return model, info

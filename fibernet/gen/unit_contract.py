"""Non-executing declarations for registered units; never resume certification."""
from __future__ import annotations

import hashlib
import json
import math
import re

from . import pattern

SCHEMA_VERSION = 'fibernet.unit-descriptor/1'
_TYPES = {'object', 'array', 'string', 'boolean', 'integer', 'number'}


def canonical_json(value):
    """Strict JSON builtins only, finite numbers, string keys; no coercion/hooks."""
    active = set()

    def check(v, depth=0):
        if depth > 100:
            raise ValueError('JSON nesting exceeds 100 levels')
        if v is None or type(v) in (str, bool, int):
            return
        if type(v) is float and math.isfinite(v):
            return
        if type(v) in (list, dict):
            if id(v) in active:
                raise ValueError('cyclic JSON container')
            if type(v) is dict and any(type(k) is not str for k in v):
                raise ValueError('JSON object keys must be strings')
            active.add(id(v))
            try:
                for item in (v.values() if type(v) is dict else v):
                    check(item, depth + 1)
            finally:
                active.remove(id(v))
            return
        raise ValueError('expected finite strict JSON builtins with string keys')
    check(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False)


def _detach(value):
    return json.loads(canonical_json(value))


def _text(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError(label + ' must be a nonempty string')


def _schema(schema):
    if (type(schema) is not dict or type(schema.get('type')) is not str
            or schema['type'] not in _TYPES):
        raise ValueError('schema requires a supported type')
    kind = schema['type']
    allowed = {'type', 'default', 'enum', 'units'}
    if kind in ('integer', 'number'):
        allowed |= {'minimum', 'maximum'}
    elif kind == 'object':
        allowed |= {'properties', 'required', 'additionalProperties'}
    elif kind == 'array':
        allowed |= {'items', 'minItems', 'maxItems'}
    if set(schema) - allowed:
        raise ValueError('unsupported schema keys')
    if 'units' in schema:
        _text(schema['units'], 'units')
    if kind == 'object':
        props = schema.get('properties')
        required = schema.get('required', [])
        if type(props) is not dict or type(required) is not list:
            raise ValueError('object requires properties and list required')
        if any(type(k) is not str for k in required) or len(set(required)) != len(required):
            raise ValueError('required names must be unique strings')
        if set(required) - set(props):
            raise ValueError('required name absent from properties')
        if type(schema.get('additionalProperties')) is not bool:
            raise ValueError('object requires explicit boolean additionalProperties')
        for child in props.values():
            _schema(child)
    if kind == 'array':
        _schema(schema.get('items'))
        for key in ('minItems', 'maxItems'):
            if key in schema and (type(schema[key]) is not int or schema[key] < 0):
                raise ValueError('array bounds must be nonnegative integers')
        if schema.get('minItems', 0) > schema.get('maxItems', float('inf')):
            raise ValueError('inverted array bounds')
    if kind in ('integer', 'number'):
        for key in ('minimum', 'maximum'):
            if key in schema and type(schema[key]) not in (int, float):
                raise ValueError('numeric bounds cannot be bool')
        if schema.get('minimum', -float('inf')) > schema.get('maximum', float('inf')):
            raise ValueError('inverted numeric bounds')
    if 'enum' in schema:
        if type(schema['enum']) is not list or not schema['enum']:
            raise ValueError('enum must be a nonempty list')
        bare = {k: v for k, v in schema.items() if k not in ('enum', 'default')}
        for item in schema['enum']:
            _value(bare, item)
    if 'default' in schema:
        _value(schema, schema['default'])


def _value(schema, value):
    kind = schema['type']
    matches = {'object': type(value) is dict, 'array': type(value) is list,
               'string': type(value) is str, 'boolean': type(value) is bool,
               'integer': type(value) is int, 'number': type(value) in (int, float)}
    if not matches[kind]:
        raise ValueError('parameter type mismatch: ' + kind)
    if 'enum' in schema and canonical_json(value) not in [canonical_json(v) for v in schema['enum']]:
        raise ValueError('parameter outside enum')
    if kind in ('number', 'integer'):
        if value < schema.get('minimum', -float('inf')) or value > schema.get('maximum', float('inf')):
            raise ValueError('parameter outside bounds')
    if kind == 'array':
        if not schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', float('inf')):
            raise ValueError('array length outside bounds')
        return [_value(schema['items'], v) for v in value]
    if kind == 'object':
        props = schema['properties']
        if not schema['additionalProperties'] and set(value) - set(props):
            raise ValueError('undeclared parameter')
        out = dict(value)
        for key, child in props.items():
            if key in value:
                out[key] = _value(child, value[key])
            elif 'default' in child:
                out[key] = _value(child, child['default'])
            elif key in schema.get('required', []):
                raise ValueError('missing required parameter: ' + key)
        return out
    return value


def verify_source_bytes(source_bytes, expected_sha256):
    """Check supplied bytes only; does not link bytes to a callable or read paths."""
    if type(source_bytes) is not bytes:
        raise TypeError('source_bytes must be bytes')
    if type(expected_sha256) is not str or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('expected_sha256 must be lowercase SHA256')
    actual = hashlib.sha256(source_bytes).hexdigest()
    if actual != expected_sha256:
        raise ValueError('source bytes SHA256 mismatch')
    return {'evidence': 'supplied_bytes_match', 'sha256': actual,
            'callable_link_verified': False, 'resume_eligible': False}


def _lookup(name, dimension):
    _text(name, 'name')
    if type(dimension) is not int or dimension not in (2, 3):
        raise ValueError('dimension must be integer 2 or 3')
    key = name.lower()
    registry = pattern._UNIT_FACTORIES if dimension == 2 else pattern._UNIT_FACTORIES_3D
    if key not in registry:
        raise KeyError('unregistered unit: ' + name)
    factory = registry[key]
    if not callable(factory):
        raise TypeError('registered unit factory is not callable')
    return key, factory


class UnitContracts:
    """Explicit in-memory declaration catalog; lookup never executes a factory.

    A replaced callable is stale. Restoring the exact original object restores its
    association. This is a snapshot check, not registration-history tracking or a
    check of mutable closure/global state. No global catalog or loader is created.
    """

    def __init__(self):
        self._records = {}

    def declare(self, name, dimension, descriptor):
        key, factory = _lookup(name, dimension)
        record = _detach(descriptor)
        fields = {'descriptor_schema', 'plugin_id', 'plugin_version', 'kind',
                  'dimensions', 'representation', 'parameter_schema_version',
                  'parameters', 'capabilities', 'randomness', 'id_policy'}
        if type(record) is not dict or set(record) != fields:
            raise ValueError('descriptor requires exactly the documented fields')
        if record['descriptor_schema'] != SCHEMA_VERSION or record['kind'] != 'unit_generator':
            raise ValueError('unsupported descriptor schema or kind')
        if record['representation'] != 'StructureGraph':
            raise ValueError('only StructureGraph declarations supported')
        for field in ('plugin_id', 'plugin_version', 'parameter_schema_version'):
            _text(record[field], field)
        dims = record['dimensions']
        if (type(dims) is not list or not dims or any(type(d) is not int or d not in (2, 3) for d in dims)
                or len(set(dims)) != len(dims) or dimension not in dims):
            raise ValueError('invalid or mismatched dimensions')
        if record['parameters'] is not None:
            _schema(record['parameters'])
            if record['parameters']['type'] != 'object':
                raise ValueError('parameters must be an object schema or null')
        caps = record['capabilities']
        if caps != 'unknown':
            if type(caps) is not dict or any(v not in ('supported', 'unsupported', 'unknown') for v in caps.values()):
                raise ValueError('capabilities require explicit three-state declarations')
        for field in ('randomness', 'id_policy'):
            if record[field] != 'unknown' and type(record[field]) is not dict:
                raise ValueError(field + ' must be unknown or a JSON declaration object')
        self._records[(dimension, key)] = (factory, record)
        return self.describe(name, dimension)

    def describe(self, name, dimension):
        key, factory = _lookup(name, dimension)
        item = self._records.get((dimension, key))
        if item is not None and item[0] is not factory:
            raise ValueError('stale descriptor: registered callable was replaced')
        return _detach({'requested_name': name, 'registry_key': key, 'dimension': dimension,
                        'status': 'undeclared' if item is None else 'declared',
                        'descriptor': None if item is None else item[1],
                        'parameters': None if item is None else item[1]['parameters'],
                        'capabilities': 'unknown' if item is None else item[1]['capabilities'],
                        'resume_eligible': False})

    def resolve_parameters(self, name, dimension, parameters):
        description = self.describe(name, dimension)
        raw = _detach(parameters)
        if type(raw) is not dict:
            raise ValueError('parameters must be a JSON object')
        schema = description['parameters']
        if schema is None:
            raise ValueError('parameter schema unknown; cannot validate or resolve defaults')
        return {'raw': raw, 'effective': _detach(_value(schema, raw)),
                'resume_eligible': False}

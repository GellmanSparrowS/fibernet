"""Source-bound graph inputs, explicit group splits, and degree statistics.

Uses the existing GraphBatch representation. This module provides metadata
validation and lineage records, not a model, trainer, or backend adapter.
"""
from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import json
import math
import struct

import numpy as np
from .graph_data import GraphBatch, GraphBatchProvenance, _distance

MAX_BYTES = 8 * 1024 * 1024
MAX_ITEMS = 300000
MAX_DEPTH = 32
MAX_TEXT = 4096
MAX_SAMPLES = 10000
MAX_NODES = 100000
MAX_EDGES = 200000
I64 = 2**63 - 1
_PREFIX = b'fibernet.graph_contract.v1\x00'
_DIRICHLET = 'dirichlet_voltage_mask.v1'
_RESERVED = {'id', 'node_id', 'edge_id', 'graph_id', 'source_group',
             'parent_id', 'parent_row', 'message_orientation', 'target', 'label',
             'prediction', 'solved_voltage', 'solver_result', 'model'}
_MAT = {'density', 'youngs_modulus', 'poissons_ratio', 'shear_modulus',
        'yield_strength', 'tensile_strength', 'fracture_toughness',
        'thermal_conductivity', 'specific_heat', 'thermal_expansion',
        'electrical_conductivity', 'permittivity', 'permeability'}


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(name + ' fields mismatch')


def _text(value, name):
    if type(value) is not str or not value.strip() or len(value.encode('utf-8')) > MAX_TEXT:
        raise ValueError(name + ' must be bounded nonempty text')
    return value


def _integer(value, name, minimum=0, maximum=I64):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(name + ' must be a bounded nonbool integer')
    return value


def _number(value, name):
    if type(value) not in (int, float) or (type(value) is int and not -I64-1 <= value <= I64):
        raise ValueError(name + ' must be a finite bounded number')
    if not math.isfinite(value):
        raise ValueError(name + ' must be finite')
    return value


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('invalid SHA256')
    return value


def _frame(tag, payload):
    if len(payload) > MAX_BYTES:
        raise ValueError('canonical byte limit')
    return tag + len(payload).to_bytes(8, 'big') + payload


def _concat(parts):
    out=[]; size=0
    for part in parts:
        size+=len(part)
        if size>MAX_BYTES: raise ValueError('canonical aggregate byte limit')
        out.append(part)
    return b''.join(out)


def canonical_bytes(value):
    """Versioned exact encoding; arrays preserve signed zeros, canonical LE/C order."""
    active = set()
    count = [0]

    def enc(v, depth):
        count[0] += 1
        if depth > MAX_DEPTH or count[0] > MAX_ITEMS:
            raise ValueError('canonical depth/item limit')
        if v is None:
            return _frame(b'n', b'')
        if type(v) is bool:
            return _frame(b'b', b'1' if v else b'0')
        if type(v) is int:
            _integer(v, 'integer', -I64-1)
            return _frame(b'i', str(v).encode('ascii'))
        if type(v) is float:
            _number(v, 'float')
            return _frame(b'f', struct.pack('>d', v))
        if type(v) is str:
            raw = v.encode('utf-8')
            if len(raw) > MAX_TEXT:
                raise ValueError('text byte limit')
            return _frame(b's', raw)
        if type(v) is bytes:
            return _frame(b'y', v)
        if type(v) is np.ndarray:
            if v.dtype.kind + str(v.dtype.itemsize) not in {'f8', 'f4', 'i8', 'i1', 'b1'}:
                raise ValueError('unsupported canonical array dtype')
            if v.nbytes > MAX_BYTES or v.ndim > 4 or not np.isfinite(v).all():
                raise ValueError('invalid canonical array')
            a = np.array(v, dtype=v.dtype.newbyteorder('<'), order='C', copy=True)
            return _frame(b'a', enc(a.dtype.str, depth+1) + enc(list(a.shape), depth+1)
                          + _frame(b'y', a.tobytes(order='C')))
        if type(v) not in (dict, MappingProxyType, list, tuple):
            raise ValueError('unsupported canonical value')
        if id(v) in active:
            raise ValueError('canonical cycle')
        active.add(id(v))
        try:
            if type(v) in (dict, MappingProxyType):
                if any(type(k) is not str for k in v):
                    raise ValueError('canonical keys must be strings')
                parts = (enc(k, depth+1) + enc(v[k], depth+1)
                         for k in sorted(v, key=lambda x: x.encode('utf-8')))
                return _frame(b'd', _concat(parts))
            return _frame(b'l' if type(v) is list else b't',
                          _concat(enc(x, depth+1) for x in v))
        finally:
            active.remove(id(v))
    return _PREFIX + enc(value, 0)


def _hash(domain, value):
    return hashlib.sha256(_frame(b'D', _text(domain, 'hash domain').encode())
                          + canonical_bytes(value)).hexdigest()


def parse_json(raw):
    if type(raw) not in (bytes, str):
        raise ValueError('JSON text required')
    if len(raw if type(raw) is bytes else raw.encode('utf-8')) > MAX_BYTES:
        raise ValueError('JSON byte limit')
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('duplicate JSON key')
            out[k] = v
        return out
    def invalid(value):
        raise ValueError('nonfinite JSON constant')
    try:
        out = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
        canonical_bytes(out)
    except (RecursionError, UnicodeError, OverflowError, TypeError) as exc:
        raise ValueError('invalid JSON') from exc
    return out


@dataclass(frozen=True)
class FrozenRecord:
    """Immutable metadata only; as_dict returns a newly detached copy."""
    kind: str
    payload_json: str
    digest: str

    def as_dict(self):
        return _read(self, self.kind)


def _record(kind, payload):
    canonical_bytes(payload)
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if len(raw.encode()) > MAX_BYTES:
        raise ValueError('record byte limit')
    return FrozenRecord(kind, raw, _hash('record/' + kind, payload))


def _read(record, kind):
    if type(record) is not FrozenRecord or record.kind != kind:
        raise ValueError('record kind mismatch')
    payload = parse_json(record.payload_json)
    if record.digest != _hash('record/' + kind, payload):
        raise ValueError('record digest mismatch')
    return payload


def _array(value, code, shape, name):
    if type(value) is not np.ndarray or value.dtype.kind + str(value.dtype.itemsize) != code:
        raise ValueError(name + ' dtype/type mismatch')
    if value.shape != shape or value.nbytes > MAX_BYTES or not np.isfinite(value).all():
        raise ValueError(name + ' shape/finite/size mismatch')
    return np.array(value, dtype=value.dtype.newbyteorder('<'), order='C', copy=True)


def _snapshot(batch):
    if type(batch) is not GraphBatch or batch.schema_version not in ('fibernet.graph_batch.v1', 'fibernet.graph_batch.v2'):
        raise ValueError('existing exact GraphBatch required')
    if type(batch.graph_ids) is not tuple or len(batch.graph_ids) != 1:
        raise ValueError('one graph per experiment sample required')
    gid = _text(batch.graph_ids[0], 'graph ID')
    _text(batch.length_unit, 'length unit'); _text(batch.coordinate_frame, 'coordinate frame')
    if type(batch.node_ids) is not np.ndarray or batch.node_ids.ndim != 1:
        raise ValueError('node IDs invalid')
    if type(batch.edge_ids) is not np.ndarray or batch.edge_ids.ndim != 1:
        raise ValueError('edge IDs invalid')
    n, e = len(batch.node_ids), len(batch.edge_ids)
    if not 1 <= n <= MAX_NODES or e > MAX_EDGES:
        raise ValueError('nonempty bounded graph required')
    a = {}
    for name, code, shape in [('node_ids','i8',(n,)), ('edge_ids','i8',(e,)),
            ('dimensions','i8',(1,)), ('node_ptr','i8',(2,)), ('edge_ptr','i8',(2,)),
            ('positions','f8',(n,3)), ('edge_index','i8',(2,e)),
            ('internal_ptr','i8',(e+1,)), ('internal_available','b1',(e,)),
            ('message_index','i8',(2,2*e)), ('message_parent_rows','i8',(2*e,)),
            ('message_parent_ids','i8',(2*e,)), ('message_orientation','i1',(2*e,))]:
        a[name] = _array(getattr(batch,name),code,shape,name)
    if (np.any(a['node_ids'] < 0) or np.any(a['edge_ids'] < 0)
            or np.any(np.diff(a['node_ids']) <= 0) or len(set(map(int,a['edge_ids']))) != e):
        raise ValueError('node/edge identity invalid')
    if not np.array_equal(a['node_ptr'],[0,n]) or not np.array_equal(a['edge_ptr'],[0,e]):
        raise ValueError('graph pointer invalid')
    if a['dimensions'][0] not in (2,3):
        raise ValueError('dimension invalid')
    p = batch.internal_points
    if type(p) is not np.ndarray or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError('internal points shape invalid')
    a['internal_points'] = _array(p,'f8',p.shape,'internal points')
    ptr = a['internal_ptr']
    if ptr[0] != 0 or ptr[-1] != len(p) or np.any(np.diff(ptr)<0):
        raise ValueError('internal pointer invalid')
    if np.any((~a['internal_available']) & (np.diff(ptr)!=0)):
        raise ValueError('internal availability mismatch')
    edges = a['edge_index']
    if np.any(edges<0) or np.any(edges>=n) or np.any(edges[0]==edges[1]):
        raise ValueError('endpoint or excluded self-loop invalid')
    if not (np.array_equal(a['message_index'][:,::2],edges)
            and np.array_equal(a['message_index'][:,1::2],edges[::-1])
            and np.array_equal(a['message_parent_rows'],np.repeat(np.arange(e),2))
            and np.array_equal(a['message_parent_ids'],np.repeat(a['edge_ids'],2))
            and np.array_equal(a['message_orientation'],np.tile([1,-1],e))):
        raise ValueError('message parent/direction mapping mismatch')
    names = batch.edge_feature_names
    if type(names) is not tuple or any(type(x) is not str for x in names) or len(set(names)) != len(names) or any(x not in ('radius','chord_length','polyline_length') for x in names):
        raise ValueError('snapshot feature names invalid')
    a['edge_features'] = _array(batch.edge_features,'f8',(e,len(names)),'edge features')
    a['edge_feature_available'] = _array(batch.edge_feature_available,'b1',(e,len(names)),'edge masks')
    if np.any(a['edge_features'][~a['edge_feature_available']] != 0):
        raise ValueError('missing snapshot placeholder invalid')
    for row, (i,j) in enumerate(edges.T):
        for col, name in enumerate(names):
            if name == 'radius':
                continue
            if not a['edge_feature_available'][row,col]:
                raise ValueError('derived length must be available')
            path = [a['positions'][i], a['positions'][j]]
            if name == 'polyline_length':
                path[1:1] = list(a['internal_points'][ptr[row]:ptr[row+1]])
            distances = [_distance(x,y,name) for x,y in zip(path[:-1],path[1:])]
            expected = math.fsum(distances)
            if not math.isfinite(expected) or a['edge_features'][row,col] != expected:
                raise ValueError('geometry-derived feature mismatch')
    # Arbitrary metadata is validated but deliberately excluded from identity.
    # Input-relevant information must enter the closed raw/condition schemas.
    for name, size in [('graph_metadata',1),('node_metadata',n),('edge_metadata',e)]:
        meta = getattr(batch,name)
        if type(meta) is not tuple or len(meta)!=size or any(type(x) not in (dict,MappingProxyType) for x in meta):
            raise ValueError('metadata alignment invalid')
        canonical_bytes(meta)
    provenance = None
    if batch.schema_version.endswith('.v1'):
        if batch.provenance is not None:
            raise ValueError('v1 provenance must be absent')
    else:
        q = batch.provenance
        if type(q) is not GraphBatchProvenance or q.schema_version != 'fibernet.graph_batch.provenance.v1':
            raise ValueError('v2 provenance invalid')
        provenance = {}
        for name, code, shape in [('node_boundary_face_flags','b1',(n,6)),('node_boundary_face_available','b1',(n,)),
                ('edge_material_available','b1',(e,)),('edge_segments','i8',(e,)),('edge_segments_available','b1',(e,))]:
            provenance[name] = _array(getattr(q,name),code,shape,name)
        for name, availability in [('node_boundary_face_origin','node_boundary_face_available'),
                ('edge_material_origin','edge_material_available'),('edge_segments_origin','edge_segments_available')]:
            origin = getattr(q,name); mask = provenance[availability]
            if type(origin) is not tuple or len(origin)!=len(mask) or any(x != ('stored_origin_unknown' if present else 'missing') for x,present in zip(origin,mask)):
                raise ValueError('provenance origin mismatch')
            provenance[name] = origin
        if np.any(provenance['node_boundary_face_flags'][~provenance['node_boundary_face_available']]):
            raise ValueError('missing boundary placeholder invalid')
        seg, available = provenance['edge_segments'], provenance['edge_segments_available']
        if np.any(seg[~available]!=0) or np.any(seg[available]<1):
            raise ValueError('segment provenance invalid')
        records = q.edge_material_records
        if type(records) is not tuple or len(records)!=e:
            raise ValueError('material record coverage invalid')
        materials = []
        for record, present in zip(records,provenance['edge_material_available']):
            if not present:
                if record is not None: raise ValueError('missing material record invalid')
                materials.append(None); continue
            if type(record) is not MappingProxyType or set(record)-(_MAT|{'name','extra','stiffness_tensor'}):
                raise ValueError('closed material record required')
            out = {}
            for key, value in record.items():
                if key == 'name':
                    if type(value) is not str: raise ValueError('material name invalid')
                    # Preserve the raw material identity, never a numeric predictor.
                    out[key] = value; continue
                if key == 'extra':
                    if type(value) is not MappingProxyType or value: raise ValueError('custom material extra unsupported')
                    out[key] = {}; continue
                if key == 'stiffness_tensor':
                    if value is not None: raise ValueError('anisotropic material unsupported by profile')
                elif value is None:
                    if key in ('density','youngs_modulus','poissons_ratio'): raise ValueError('required material null')
                else: _number(value,'stored material')
                out[key] = value
            materials.append(out)
        provenance['edge_material_records'] = materials
    return {'graph_id':gid,'schema':batch.schema_version,'length_unit':batch.length_unit,
            'coordinate_frame':batch.coordinate_frame,'arrays':a,'edge_feature_names':list(names),
            'stored_provenance':provenance}


def _feature_spec(spec):
    if type(spec) is not dict or spec.get('schema') not in ('fibernet.graph_feature_spec.v1','fibernet.graph_feature_spec.v2'):
        raise ValueError('unsupported feature profile')
    fields=('schema','length_unit','coordinate_frame','dtype','node_offset','node_scale','features')
    encoded=spec['schema']=='fibernet.graph_feature_spec.v2'
    _keys(spec,fields+(('condition_profile',) if encoded else ()),'feature spec')
    if spec['dtype']!='float32' or (encoded and (type(spec['condition_profile']) is not str or spec['condition_profile']!=_DIRICHLET)):
        raise ValueError('unsupported feature profile')
    for name in ('length_unit','coordinate_frame'): _text(spec[name],name)
    for name in ('node_offset','node_scale'):
        if type(spec[name]) is not list or len(spec[name])!=4: raise ValueError('four node normalizers required')
        for x in spec[name]:
            _number(x,name)
            if name=='node_scale' and x<=0: raise ValueError('positive node scale required')
    fs=spec['features']
    if type(fs) is not list or not 1<=len(fs)<=64: raise ValueError('bounded ordered features required')
    names=set();sources=set()
    for f in fs:
        _keys(f,('name','source','unit','offset','scale','required','parity'),'feature')
        name=_text(f['name'],'feature name'); unit=_text(f['unit'],'feature unit')
        if name.lower() in _RESERVED or name in names: raise ValueError('reserved/duplicate feature')
        names.add(name); src=f['source']
        if type(src) is not dict or len(src)!=1 or next(iter(src)) not in ('snapshot','external'):
            raise ValueError('explicit snapshot/external source required')
        kind,field=next(iter(src.items()));_text(field,'feature source')
        if field.lower() in _RESERVED or (kind,field) in sources: raise ValueError('reserved/duplicate source')
        sources.add((kind,field))
        if kind=='snapshot' and (field not in ('radius','chord_length','polyline_length') or unit!=spec['length_unit']):
            raise ValueError('snapshot selector/unit mismatch')
        _number(f['offset'],'offset');_number(f['scale'],'scale')
        if f['scale']<=0 or type(f['required']) is not bool or f['parity']!='even':
            raise ValueError('invalid scalar feature policy')
    return parse_json(json.dumps(spec,allow_nan=False))


def _external(external, snapshot, spec):
    columns={next(iter(f['source'].values())):f for f in spec['features'] if 'external' in f['source']}
    if not columns:
        if external is not None: raise ValueError('unexpected external input')
        return [], {}
    _keys(external,('units','rows'),'raw external')
    if type(external['units']) is not dict or external['units']!={k:f['unit'] for k,f in columns.items()}:
        raise ValueError('external unit mismatch')
    rows=external['rows'];ids=list(map(int,snapshot['arrays']['edge_ids']))
    if type(rows) is not dict or any(type(k) is not int or not 0<=k<=I64 for k in rows) or set(rows)!=set(ids):
        raise ValueError('exact external physical edge IDs required')
    owned=[]; lookup={}
    for eid in ids:
        _keys(rows[eid],columns,'external columns'); values={}
        for field in columns:
            cell=rows[eid][field];_keys(cell,('value','available'),'external cell')
            if type(cell['available']) is not bool: raise ValueError('external mask must be bool')
            if cell['available']: _number(cell['value'],'external value')
            elif cell['value'] is not None: raise ValueError('missing external value must be None')
            values[field]=dict(cell)
        lookup[eid]=values;owned.append({'edge_id':eid,'values':values})
    return owned,lookup


def _conditions(value, snapshot):
    _keys(value,('schema','task_rule_id','conditioning','boundary_voltages','nodal_currents'),'conditions')
    if value['schema']!='fibernet.electrical_conditions.v1' or value['conditioning']!='metadata_only_fixed_rule':
        raise ValueError('variable/unencoded or unknown condition profile unsupported')
    _text(value['task_rule_id'],'task rule'); ids=set(map(int,snapshot['arrays']['node_ids']))
    for name,unit in [('boundary_voltages','V'),('nodal_currents','A')]:
        entries=value[name]
        if type(entries) is not list or len(entries)>len(ids): raise ValueError('condition records invalid')
        seen=set()
        for row in entries:
            _keys(row,('node_id','value','unit'),'condition row')
            nid=_integer(row['node_id'],'condition node ID')
            if nid not in ids or nid in seen: raise ValueError('condition node coverage invalid')
            seen.add(nid);_number(row['value'],'condition value')
            if row['unit']!=unit: raise ValueError('condition unit mismatch')
    return parse_json(json.dumps(value,allow_nan=False))


def _fixed_condition_record(condition):
    # List order declares fixed role-slot order. IDs may be graph-local, but
    # numerical values, units and slot cardinality must be identical throughout.
    record={'schema':'fibernet.fixed_electrical_rule.v1',
            'task_rule_id':condition['task_rule_id'],'conditioning':condition['conditioning']}
    for name in ('boundary_voltages','nodal_currents'):
        record[name]=[{key:row[key] for key in ('value','unit')} for row in condition[name]]
    _fixed_condition_bytes(record)
    return record


def _fixed_condition_bytes(record):
    _keys(record,('schema','task_rule_id','conditioning','boundary_voltages','nodal_currents'),'fixed condition record')
    if record['schema']!='fibernet.fixed_electrical_rule.v1' or record['conditioning']!='metadata_only_fixed_rule':
        raise ValueError('fixed condition schema/profile unsupported')
    _text(record['task_rule_id'],'fixed task rule')
    for name,unit in [('boundary_voltages','V'),('nodal_currents','A')]:
        rows=record[name]
        if type(rows) is not list or len(rows)>MAX_NODES: raise ValueError('fixed condition slots invalid')
        for row in rows:
            _keys(row,('value','unit'),'fixed condition slot')
            _number(row['value'],'fixed condition value')
            if row['unit']!=unit: raise ValueError('fixed condition unit mismatch')
    return canonical_bytes(record)


def _condition_for_spec(value, snapshot, spec):
    if spec['schema']=='fibernet.graph_feature_spec.v1':
        return _conditions(value,snapshot)
    _keys(value,('schema','task_rule_id','conditioning','boundary_voltages','nodal_currents','current_declaration'),'encoded conditions')
    if value['schema']!='fibernet.electrical_conditions.v2' or value['conditioning']!=_DIRICHLET:
        raise ValueError('encoded condition schema/profile mismatch')
    declaration=value['current_declaration']
    _keys(declaration,('kind','source_ref','source_sha256'),'explicit current declaration')
    if declaration['kind']!='no_nodal_current_injections.v1':
        raise ValueError('unknown/unsupported current injection semantics')
    _text(declaration['source_ref'],'current declaration source');_sha(declaration['source_sha256'])
    if type(value['nodal_currents']) is not list or value['nodal_currents']:
        raise ValueError('encoded profile requires explicit no nodal current injections')
    # Reuse exact legacy row validation, never infer a condition from geometry,
    # metadata, a trial field, or a solved field. The caller supplies raw authority.
    legacy={k:v for k,v in value.items() if k!='current_declaration'}
    legacy['schema']='fibernet.electrical_conditions.v1'
    legacy['conditioning']='metadata_only_fixed_rule'
    _conditions(legacy,snapshot)
    if not value['boundary_voltages']:
        raise ValueError('encoded profile requires nonempty Dirichlet conditions')
    return parse_json(json.dumps(value,allow_nan=False))


def _condition_record(condition, spec):
    if spec['schema']=='fibernet.graph_feature_spec.v1':
        return _fixed_condition_record(condition)
    return {'schema':'fibernet.encoded_electrical_rule.v1',
            'task_rule_id':condition['task_rule_id'],'conditioning':_DIRICHLET,
            'voltage_unit':'V','voltage_scale':1.0,'voltage_offset':0.0,
            'zero_policy':'positive_zero_projection_raw_signed_zero_preserved.v1',
            'current_policy':'no_nodal_current_injections.v1'}


def _condition_record_bytes(record, spec):
    if spec['schema']=='fibernet.graph_feature_spec.v1':
        return _fixed_condition_bytes(record)
    if type(record) is not dict:
        raise ValueError('encoded condition record required')
    _text(record.get('task_rule_id'),'encoded task rule')
    expected=_condition_record({'task_rule_id':record['task_rule_id']},spec)
    if canonical_bytes(record)!=canonical_bytes(expected):
        raise ValueError('encoded condition rule/profile mismatch')
    return canonical_bytes(record)


def _normalize(values, mask, offset, scale):
    raw=np.array(values,dtype=np.float64,copy=True)
    with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
        difference=raw-np.asarray(offset,dtype=np.float64)
        norm=difference/np.asarray(scale,dtype=np.float64)
        if np.any(mask & ((raw!=offset)&(difference==0))) or np.any(mask & ((difference!=0)&(norm==0))):
            raise ValueError('normalization loses nonzero value')
        norm=np.where(mask,norm,0.0);out=norm.astype(np.float32)
    if not np.isfinite(norm).all() or not np.isfinite(out).all() or np.any((norm!=0)&(out==0)):
        raise ValueError('projection exceeds finite float32 range')
    return out


def _project(snapshot, spec, lookup, supplied, condition=None):
    a=snapshot['arrays'];n=len(a['node_ids']);e=len(a['edge_ids']);fs=spec['features']
    raw=np.zeros((e,len(fs)),dtype=np.float64);mask=np.zeros(raw.shape,dtype=bool)
    for j,f in enumerate(fs):
        if 'snapshot' in f['source']:
            field=f['source']['snapshot']
            if field not in snapshot['edge_feature_names']: raise ValueError('selected snapshot column absent')
            k=snapshot['edge_feature_names'].index(field);raw[:,j]=a['edge_features'][:,k];mask[:,j]=a['edge_feature_available'][:,k]
        else:
            field=f['source']['external']
            for i,eid in enumerate(a['edge_ids']):
                cell=lookup[int(eid)][field];mask[i,j]=cell['available']
                if cell['available']:
                    value=cell['value']
                    if type(value) is int and int(float(value))!=value:
                        raise ValueError('raw integer loses precision before projection')
                    raw[i,j]=value
        if f['required'] and not mask[:,j].all(): raise ValueError('required feature missing')
    edge=_normalize(raw,mask,[f['offset'] for f in fs],[f['scale'] for f in fs])
    nodes=np.column_stack((a['positions'],np.full(n,a['dimensions'][0],dtype=np.float64)))
    output={'node_features':_normalize(nodes,np.ones(nodes.shape,dtype=bool),spec['node_offset'],spec['node_scale']),
            'edge_features':np.concatenate((edge,mask.astype(np.float32)),axis=1)[a['message_parent_rows']],
            'edge_index':a['message_index'].copy(),'graph_index':np.zeros(n,dtype=np.int64)}
    if spec['schema']=='fibernet.graph_feature_spec.v2':
        condition=_condition_for_spec(condition,snapshot,spec)
        voltage=np.zeros((n,1),dtype=np.float64);fixed=np.zeros((n,1),dtype=bool)
        node_rows={int(nid):i for i,nid in enumerate(a['node_ids'])}
        for row in condition['boundary_voltages']:
            value=row['value']
            if type(value) is int and int(float(value))!=value:
                raise ValueError('raw integer loses precision before projection')
            i=node_rows[row['node_id']];fixed[i,0]=True
            voltage[i,0]=0.0 if value==0 else value
        normalized=_normalize(voltage,fixed,[0.0],[1.0])
        output['node_features']=np.concatenate((output['node_features'],normalized,fixed.astype(np.float32)),axis=1)
    if supplied is not None:
        _keys(supplied,output,'derived NumPy view')
        for name,expected in output.items():
            actual=_array(supplied[name],expected.dtype.kind+str(expected.dtype.itemsize),expected.shape,name)
            if canonical_bytes(actual)!=canonical_bytes(expected): raise ValueError('derived view differs from authoritative recomputation')
    return output


def _prepare_owned(s, *, group, spec, external, condition, refs, supplied=None):
    gid=s['graph_id'];group=_text(group,'mother group')
    if s['length_unit']!=spec['length_unit'] or s['coordinate_frame']!=spec['coordinate_frame']:
        raise ValueError('geometry unit/frame mismatch')
    raw,lookup=_external(external,s,spec);condition=_condition_for_spec(condition,s,spec)
    _keys(refs,('graph','external','conditions'),'source references')
    for value in refs.values(): _text(value,'source reference')
    projected=_project(s,spec,lookup,supplied,condition);fs_hash=_hash('feature_spec/v1',spec)
    # Only closed pre-solve schemas enter data identity. Arbitrary graph/node/
    # edge metadata and opaque file references are not model inputs.
    ext={'units':dict(external['units']),'rows':raw} if external is not None else None
    src=_hash('source/v1',{'graph':s,'external':ext,'conditions':condition})
    projection=_hash('projection/v1',{'feature_spec_hash':fs_hash,'arrays':projected})
    inp=_hash('input/v1',{'source_hash':src,'feature_spec_hash':fs_hash,'projection_hash':projection})
    a=s['arrays'];topology=_hash('topology/v1',{k:a[k] for k in ('node_ids','edge_ids','edge_index','message_index','message_parent_rows','message_parent_ids','message_orientation')})
    return {'graph_id':gid,'source_group':group,'source_hash':src,'projection_hash':projection,
            'input_hash':inp,'topology_hash':topology,'node_count':len(a['node_ids']),
            'physical_edge_count':len(a['edge_ids']),'condition_rule':condition['task_rule_id'],
            'condition_record':_condition_record(condition,spec),
            'source_refs':dict(refs)},projected


def _bind_owned(s, *, group, spec, external, condition, refs, supplied=None):
    return _prepare_owned(s,group=group,spec=spec,external=external,condition=condition,refs=refs,supplied=supplied)[0]


def _input_schema(spec):
    return ('fibernet.graph_inputs.v2' if spec['schema']=='fibernet.graph_feature_spec.v1'
            else 'fibernet.graph_inputs.v3')


def bind_graph_inputs(batches, *, source_groups, feature_spec, external, conditions, source_refs, derived_views=None):
    """Bind raw authority; metadata-only result, no model-specific input identity."""
    if type(batches) not in (list,tuple) or not 1<=len(batches)<=MAX_SAMPLES:
        raise ValueError('bounded nonempty batch sequence required')
    spec=_feature_spec(feature_spec);fs_hash=_hash('feature_spec/v1',spec)
    snapshots=[_snapshot(b) for b in batches];ids=[s['graph_id'] for s in snapshots]
    if len(set(ids))!=len(ids): raise ValueError('duplicate graph IDs')
    for value,name in [(source_groups,'source groups'),(external,'external'),(conditions,'conditions'),(source_refs,'source refs')]:
        if type(value) is not dict or set(value)!=set(ids): raise ValueError('exact graph-ID '+name+' required')
    if derived_views is not None and (type(derived_views) is not dict or set(derived_views)!=set(ids)):
        raise ValueError('derived view coverage mismatch')
    rows=[]
    for s in snapshots:
        gid=s['graph_id']
        rows.append(_bind_owned(s,group=source_groups[gid],spec=spec,external=external[gid],
                    condition=conditions[gid],refs=source_refs[gid],
                    supplied=None if derived_views is None else derived_views[gid]))
    if len({_condition_record_bytes(r['condition_record'],spec) for r in rows})!=1:
        raise ValueError('mixed/unencoded numerical condition rules unsupported by this profile')
    return _record('inputs',{'schema':_input_schema(spec),'feature_spec':spec,'feature_spec_hash':fs_hash,'samples':rows})


def _inputs(record):
    p=_read(record,'inputs');_keys(p,('schema','feature_spec','feature_spec_hash','samples'),'inputs')
    spec=_feature_spec(p['feature_spec'])
    if p['schema']!=_input_schema(spec): raise ValueError('input schema/profile mismatch')
    if p['feature_spec_hash']!=_hash('feature_spec/v1',spec): raise ValueError('feature spec binding mismatch')
    if type(p['samples']) is not list or not 1<=len(p['samples'])<=MAX_SAMPLES: raise ValueError('sample records invalid')
    ids=[];condition_records=set()
    for row in p['samples']:
        _keys(row,('graph_id','source_group','source_hash','projection_hash','input_hash','topology_hash','node_count','physical_edge_count','condition_rule','condition_record','source_refs'),'sample binding')
        ids.append(_text(row['graph_id'],'graph ID'));_text(row['source_group'],'mother');_text(row['condition_rule'],'condition rule')
        condition_records.add(_condition_record_bytes(row['condition_record'],spec))
        if row['condition_rule']!=row['condition_record']['task_rule_id']: raise ValueError('condition rule linkage mismatch')
        for k in ('source_hash','projection_hash','input_hash','topology_hash'): _sha(row[k])
        _integer(row['node_count'],'node count',1,MAX_NODES);_integer(row['physical_edge_count'],'edge count',0,MAX_EDGES)
        if spec['schema']=='fibernet.graph_feature_spec.v1' and any(len(row['condition_record'][key])>row['node_count'] for key in ('boundary_voltages','nodal_currents')):
            raise ValueError('fixed condition slot count exceeds sample nodes')
        _keys(row['source_refs'],('graph','external','conditions'),'source references')
        for v in row['source_refs'].values(): _text(v,'source reference')
        expected=_hash('input/v1',{'source_hash':row['source_hash'],'feature_spec_hash':p['feature_spec_hash'],'projection_hash':row['projection_hash']})
        if expected!=row['input_hash']: raise ValueError('input digest linkage mismatch')
    if len(set(ids))!=len(ids) or len(condition_records)!=1: raise ValueError('sample identity/numerical rule mismatch')
    return p


def project_graph_inputs(inputs, *, sources):
    """Revalidate raw sources and return detached, call-local NumPy projections.

    Sources have the same four fields as training_sources. No labels, fitted
    transforms, optional backend or persistent graph representation is created.
    """
    p=_inputs(inputs);rows={r['graph_id']:r for r in p['samples']}
    if type(sources) is not dict or set(sources)!=set(rows):
        raise ValueError('exact bound graph-ID sources required')
    output={}
    for gid,row in rows.items():
        item=sources[gid];_keys(item,('batch','external','conditions','source_refs'),'projection source')
        snapshot=_snapshot(item['batch'])
        if snapshot['graph_id']!=gid: raise ValueError('projection graph ID mismatch')
        rebound,projected=_prepare_owned(snapshot,group=row['source_group'],spec=p['feature_spec'],
            external=item['external'],condition=item['conditions'],refs=item['source_refs'])
        if canonical_bytes(rebound)!=canonical_bytes(row):
            raise ValueError('projection source differs from bound raw authority')
        output[gid]=projected
    return output


def bind_explicit_group_split(inputs, *, train_ids, validation_ids, final_ids, split_seed, algorithm='explicit_replay.v1'):
    p=_inputs(inputs);_integer(split_seed,'split seed',0,2**31-1)
    if algorithm!='explicit_replay.v1': raise ValueError('only explicit replay membership supported; no shuffle')
    roles={'train':train_ids,'validation':validation_ids,'final':final_ids}
    rows={r['graph_id']:r for r in p['samples']};flat=[];groups={}
    for role,ids in roles.items():
        if type(ids) not in (list,tuple) or not ids or any(type(g) is not str or g not in rows for g in ids):
            raise ValueError('nonempty known role IDs required')
        flat.extend(ids);groups[role]={rows[g]['source_group'] for g in ids}
    if len(flat)!=len(set(flat)) or set(flat)!=set(rows): raise ValueError('split must exactly partition samples')
    if any(groups[x]&groups[y] for x,y in [('train','validation'),('train','final'),('validation','final')]):
        raise ValueError('mother groups overlap')
    entries=[{k:r[k] for k in ('graph_id','source_group','input_hash')} for r in p['samples']]
    return _record('split',{'schema':'fibernet.graph_split.v1','samples':entries,'roles':{k:list(v) for k,v in roles.items()},
                            'algorithm':algorithm,'split_seed':split_seed,'seed_semantics':'recorded_only_not_regenerated'})


def _split(inputs, split):
    p=_read(split,'split');_keys(p,('schema','samples','roles','algorithm','split_seed','seed_semantics'),'split')
    if p['schema']!='fibernet.graph_split.v1' or p['seed_semantics']!='recorded_only_not_regenerated': raise ValueError('split schema unsupported')
    _keys(p['roles'],('train','validation','final'),'split roles')
    expected=bind_explicit_group_split(inputs,train_ids=p['roles']['train'],validation_ids=p['roles']['validation'],final_ids=p['roles']['final'],split_seed=p['split_seed'],algorithm=p['algorithm'])
    if expected!=split: raise ValueError('split does not bind current inputs')
    return p


def _histogram_summary(counts):
    if type(counts) is not list or not 1<=len(counts)<=2*MAX_EDGES+1: raise ValueError('bounded histogram required')
    for x in counts: _integer(x,'histogram count')
    n=sum(counts);_integer(n,'population')
    incoming=sum(i*c for i,c in enumerate(counts));_integer(incoming,'incoming messages')
    if len(counts)>1 and counts[-1]==0: raise ValueError('noncanonical trailing histogram zeros')
    return {'histogram':list(counts),'node_population':n,'incoming_messages':incoming,
            'status':'nonempty' if n else 'empty_population',
            'mean_degree':incoming/n if n else None,
            'mean_log_degree':math.fsum(math.log1p(i)*c for i,c in enumerate(counts))/n if n else None}


def training_degree_histogram(inputs, split, *, training_sources):
    p=_inputs(inputs);s=_split(inputs,split);ids=s['roles']['train'];rows={r['graph_id']:r for r in p['samples']}
    if type(training_sources) is not dict or set(training_sources)!=set(ids): raise ValueError('exact training sources only required')
    counts=[0];nodes=0;edges=0
    for gid in ids:
        item=training_sources[gid];_keys(item,('batch','external','conditions','source_refs'),'training source')
        snapshot=_snapshot(item['batch'])
        if snapshot['graph_id']!=gid: raise ValueError('training graph ID mismatch')
        rebound=_bind_owned(snapshot,group=rows[gid]['source_group'],spec=p['feature_spec'],external=item['external'],condition=item['conditions'],refs=item['source_refs'])
        if any(canonical_bytes(rebound[k])!=canonical_bytes(rows[gid][k]) for k in ('source_hash','projection_hash','input_hash','topology_hash','node_count','physical_edge_count','condition_rule','condition_record')):
            raise ValueError('training source mutated or coalesced relative to bound source')
        a=snapshot['arrays'];n=len(a['node_ids'])
        degree=np.bincount(a['message_index'][1],minlength=n)
        hist=np.bincount(degree,minlength=1).tolist()
        counts.extend([0]*max(0,len(hist)-len(counts)))
        for i,c in enumerate(hist): counts[i]+=int(c)
        nodes+=n;edges+=len(a['edge_ids'])
    summary=_histogram_summary(counts)
    if summary['node_population']!=nodes or summary['incoming_messages']!=2*edges: raise ValueError('population/message accounting mismatch')
    return _record('degree',{'schema':'fibernet.training_degree.v1','split_hash':split.digest,
          'training':[ {k:rows[g][k] for k in ('graph_id','source_group','input_hash','topology_hash','node_count','physical_edge_count')} for g in ids],
          'counting':'incoming_message_index_1_all_nodes.v1','weighting':'one_per_selected_training_node_instance.v1',
          'physical_edges':edges,'summary':summary})


def validate_pna_degree_admission(statistics, *, scalers):
    p=_read(statistics,'degree');_keys(p,('schema','split_hash','training','counting','weighting','physical_edges','summary'),'degree record')
    if p['schema']!='fibernet.training_degree.v1' or p['counting']!='incoming_message_index_1_all_nodes.v1' or p['weighting']!='one_per_selected_training_node_instance.v1':
        raise ValueError('degree schema/counting unsupported')
    _sha(p['split_hash']);_integer(p['physical_edges'],'physical edges')
    summary=p['summary'];_keys(summary,('histogram','node_population','incoming_messages','status','mean_degree','mean_log_degree'),'histogram summary')
    _integer(summary['node_population'],'summary node population')
    _integer(summary['incoming_messages'],'summary incoming messages')
    if canonical_bytes(summary)!=canonical_bytes(_histogram_summary(summary['histogram'])) or summary['incoming_messages']!=2*p['physical_edges']:
        raise ValueError('forged histogram population/moments')
    if type(p['training']) is not list or not p['training']: raise ValueError('nonempty training metadata required')
    seen=set();n=e=0
    for r in p['training']:
        _keys(r,('graph_id','source_group','input_hash','topology_hash','node_count','physical_edge_count'),'training statistics reference')
        gid=_text(r['graph_id'],'graph ID');_text(r['source_group'],'mother');_sha(r['input_hash']);_sha(r['topology_hash'])
        if gid in seen: raise ValueError('duplicate training reference')
        seen.add(gid);n+=_integer(r['node_count'],'training nodes',1,MAX_NODES);e+=_integer(r['physical_edge_count'],'training edges',0,MAX_EDGES)
    if n!=summary['node_population'] or e!=p['physical_edges']: raise ValueError('training counts mismatch')
    if type(scalers) not in (list,tuple) or not scalers or any(type(x) is not str for x in scalers) or len(set(scalers))!=len(scalers) or any(x not in ('identity','amplification','attenuation','linear','inverse_linear') for x in scalers):
        raise ValueError('unsupported PNA scaler list')
    if not summary['node_population']: raise ValueError('PNA moment construction needs N > 0')
    if 'amplification' in scalers and summary['mean_log_degree']==0: raise ValueError('PNA amplification has zero denominator')
    if 'linear' in scalers and summary['mean_degree']==0: raise ValueError('PNA linear has zero denominator')
    return _record('pna_admission',{'schema':'fibernet.pna_denominator_admission.v1','pyg_version':'2.7.0',
               'statistics_hash':statistics.digest,'scalers':list(scalers),'status':'denominator_checks_only_not_backend_validation'})


def _legacy_development_membership(manifest):
    # Validate old two-way development membership; never reinterpret it as the
    # new three-way experiment split or silently fill missing legacy indices.
    if type(manifest) is not dict or not {'split','input_digests','graph_ids','source_groups'}<=set(manifest):
        raise ValueError('legacy membership missing')
    ids=manifest['graph_ids']
    if type(ids) is not list or not 2<=len(ids)<=MAX_SAMPLES:
        raise ValueError('legacy development IDs invalid')
    for gid in ids: _text(gid,'legacy graph ID')
    if len(ids)!=len(set(ids)): raise ValueError('duplicate legacy development ID')
    for key in ('source_groups','input_digests'):
        if type(manifest[key]) is not dict or set(manifest[key])!=set(ids):
            raise ValueError('legacy '+key+' coverage mismatch')
    for group in manifest['source_groups'].values(): _text(group,'legacy mother group')
    for digest in manifest['input_digests'].values(): _sha(digest)
    split=manifest['split']
    if type(split) is not dict or not {'graph_ids','train_indices','validation_indices'}<=set(split):
        raise ValueError('legacy development split missing')
    if type(split['graph_ids']) is not list or canonical_bytes(split['graph_ids'])!=canonical_bytes(ids):
        raise ValueError('legacy development split ID order mismatch')
    roles=[]
    for key in ('train_indices','validation_indices'):
        indices=split[key]
        if type(indices) is not list or not indices or len(indices)>len(ids):
            raise ValueError('legacy nonempty development index list required')
        for index in indices: _integer(index,'legacy development index',0,len(ids)-1)
        if len(set(indices))!=len(indices): raise ValueError('duplicate legacy development index')
        roles.append(set(indices))
    if roles[0]&roles[1] or roles[0]|roles[1]!=set(range(len(ids))):
        raise ValueError('legacy development indices must be disjoint and exactly cover original IDs')
    groups=manifest['source_groups']
    if {groups[ids[i]] for i in roles[0]} & {groups[ids[i]] for i in roles[1]}:
        raise ValueError('legacy development mother groups overlap')
    for split_key,manifest_key in [('groups','source_groups'),('input_digests','input_digests')]:
        if split_key in split and canonical_bytes(split[split_key])!=canonical_bytes(manifest[manifest_key]):
            raise ValueError('legacy split '+split_key+' linkage mismatch')
    return ids


def make_lineage_sidecar(artifact_bytes, *, result_bytes, reconstructed_inputs=None, raw_sources=None):
    """Metadata-only v1 lineage. Original bytes are never written or modified."""
    if type(artifact_bytes) is not bytes or type(result_bytes) is not bytes: raise ValueError('original immutable bytes required')
    old=parse_json(artifact_bytes);parse_json(result_bytes)
    _keys(old,('schema','kind','model_config','target','training_manifest','state','runtime'),'legacy artifact')
    if old['schema']!='fibernet.gine_artifact.v1' or old['kind']!='inference_best_validation': raise ValueError('unsupported legacy schema/kind')
    manifest=old['training_manifest']
    _legacy_development_membership(manifest)
    # These nested old records are preserved, not reinterpreted as new strong data.
    identity={'artifact_bytes_sha256':hashlib.sha256(artifact_bytes).hexdigest(),
              'results_bytes_sha256':hashlib.sha256(result_bytes).hexdigest(),
              'model_config':old['model_config'],'input_digests':manifest['input_digests'],
              'split':manifest['split'],'graph_ids':manifest['graph_ids'],'source_groups':manifest['source_groups'],
              'runtime':old['runtime'],'target':old['target'],
              'state_canonical_json_sha256':hashlib.sha256(json.dumps(old['state'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()}
    new=None;status='blocked_missing_raw_authority';strength='legacy_prepared_only'
    if reconstructed_inputs is not None and raw_sources is not None:
        newp=_inputs(reconstructed_inputs);rows=newp['samples'];ids=[r['graph_id'] for r in rows]
        if type(raw_sources) is not dict or set(raw_sources)!=set(ids): raise ValueError('complete authoritative reconstruction required')
        batches=[];external={};conditions={};refs={};groups={}
        for r in rows:
            gid=r['graph_id'];v=raw_sources[gid];_keys(v,('batch','external','conditions','source_refs'),'raw reconstruction')
            batches.append(v['batch']);external[gid]=v['external'];conditions[gid]=v['conditions'];refs[gid]=v['source_refs'];groups[gid]=r['source_group']
        rebuilt=bind_graph_inputs(batches,source_groups=groups,feature_spec=newp['feature_spec'],external=external,conditions=conditions,source_refs=refs)
        if rebuilt!=reconstructed_inputs: raise ValueError('reconstructed binding mismatch')
        if set(manifest['graph_ids'])-set(ids) or any(groups[g]!=manifest['source_groups'][g] for g in manifest['graph_ids']):
            raise ValueError('legacy development membership mismatch')
        new={'binding_digest':rebuilt.digest,'samples':rows};status='new_contract_lineage_only';strength='raw_reconstructed_separate_contract'
    return _record('lineage',{'schema':'fibernet.graph_lineage_sidecar.v1','legacy_identity':identity,
             'status':status,'authority_profile':strength,'new_contract':new,
             'weight_compatibility':'not_implemented','old_loader_migration':'not_implemented'})

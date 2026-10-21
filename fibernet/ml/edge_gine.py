"""Optional, no-fit edge-aware adapter. StructureGraph/GraphBatch remain authoritative.

Requires PyTorch and PyG only when this optional module is used. No root export,
trainer integration, material inference, unit conversion or target loading.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import json
import math
import numbers
import struct
from types import MappingProxyType
import numpy as np

try:
    import torch
    from torch import nn
    from torch_geometric.nn import GINEConv
    _BACKEND_ERROR = None
except ImportError as exc:
    torch = nn = GINEConv = None
    _BACKEND_ERROR = exc

_VERSION = 'fibernet.edge_gine.v1'
_MATERIAL_VERSION = 'fibernet.edge_gine.material_view.v1'
_RESERVED = {'id', 'node_id', 'edge_id', 'graph_id', 'source_group',
             'parent_id', 'parent_row', 'message_orientation'}

def _require():
    if _BACKEND_ERROR is not None:
        raise ImportError('edge_gine requires optional compatible torch and torch-geometric; no topology-only fallback') from _BACKEND_ERROR


def _plain(v, active=None, depth=0):
    if depth > 100:
        raise ValueError('metadata nesting exceeds 100')
    if v is None or type(v) in (str, bool, int):
        return v
    if type(v) is float and math.isfinite(v):
        return v
    if not isinstance(v, (Mapping, tuple, list)):
        raise ValueError('metadata must be finite JSON-like values')
    active = set() if active is None else active
    if id(v) in active:
        raise ValueError('metadata cycle')
    active.add(id(v))
    try:
        if isinstance(v, Mapping):
            if any(type(k) is not str for k in v):
                raise ValueError('metadata keys must be strings')
            return {k: _plain(x, active, depth+1) for k, x in v.items()}
        return [_plain(x, active, depth+1) for x in v]
    finally:
        active.remove(id(v))


def _json(v):
    return json.dumps(_plain(v), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _real(v, label):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) or not math.isfinite(v):
        raise ValueError(label + ' must be a finite real scalar')
    f = float(v)
    if v != 0 and f == 0:
        raise ValueError(label + ' underflows float64')
    return f


def _text(v, label):
    if type(v) is not str or not v.strip():
        raise ValueError(label + ' must be a nonempty string')
    return v


def make_config(*, features, length_unit, coordinate_frame, node_offset,
                node_scale, pooling, material_profile=None):
    """Create closed scalar-even schema. Explicit scaling is never fitted here."""
    if material_profile is not None:
        return _material_config(features, length_unit, coordinate_frame, node_offset, node_scale, pooling, material_profile)
    if pooling not in ('sum', 'mean'):
        raise ValueError('pooling must be sum or mean')
    _text(length_unit, 'length_unit'); _text(coordinate_frame, 'coordinate_frame')
    if type(features) not in (list, tuple) or not features:
        raise ValueError('features must be a nonempty ordered list')
    names, sources, cols = set(), set(), []
    for f in features:
        if type(f) is not dict or set(f) != {'name','source','unit','offset','scale','required','parity'}:
            raise ValueError('feature must have exactly the declared schema keys')
        name = _text(f['name'], 'feature name')
        if name in names or name.lower() in _RESERVED:
            raise ValueError('duplicate or identity feature name')
        names.add(name)
        src = f['source']
        if type(src) is not dict or len(src) != 1 or next(iter(src)) not in ('snapshot', 'external'):
            raise ValueError('feature needs one exclusive source selector')
        kind, field = next(iter(src.items())); _text(field, 'source field')
        if field.lower() in _RESERVED or (kind, field) in sources:
            raise ValueError('duplicate or identity feature source')
        sources.add((kind, field))
        if kind == 'snapshot' and field not in ('radius','chord_length','polyline_length'):
            raise ValueError('unknown snapshot feature')
        unit = _text(f['unit'], 'unit')
        if kind == 'snapshot' and unit != length_unit:
            raise ValueError('snapshot feature unit must equal declared geometry unit')
        offset, scale = _real(f['offset'], 'offset'), _real(f['scale'], 'scale')
        if scale <= 0 or type(f['required']) is not bool or f['parity'] != 'even':
            raise ValueError('positive scale, boolean required and even scalar parity required')
        cols.append(dict(name=name, source={kind:field}, unit=unit, offset=offset,
                         scale=scale, required=f['required'], parity='even'))
    def four(v, positive):
        if type(v) not in (list, tuple) or len(v) != 4:
            raise ValueError('node normalization must have four scalars')
        a = [_real(x, 'node normalization') for x in v]
        if positive and any(x <= 0 for x in a):
            raise ValueError('node scales must be positive')
        return a
    return {'version':_VERSION,'node_schema':['x','y','z','dimension'],
            'dtype':'float32','device':'cpu','features':cols,
            'length_unit':length_unit,'coordinate_frame':coordinate_frame,
            'node_offset':four(node_offset, False),'node_scale':four(node_scale, True),
            'pooling':pooling,'hidden':16,'layers':2,'eps':0.0}


def _config(c):
    if type(c) is not dict:
        raise ValueError('config must be a closed dictionary')
    try:
        options = {k:c[k] for k in ('features','length_unit','coordinate_frame','node_offset','node_scale','pooling')}
        if c.get('version') == _MATERIAL_VERSION:
            options['material_profile'] = c['material_profile']
        rebuilt = make_config(**options)
    except KeyError as exc:
        raise ValueError('incomplete config') from exc
    if _json(c) != _json(rebuilt):
        raise ValueError('unknown or changed config fields')
    return rebuilt


def _snapshot(b, *, _material=False):
    from fibernet.ml.graph_data import GraphBatch
    version = 'fibernet.graph_batch.v2' if _material else 'fibernet.graph_batch.v1'
    if not isinstance(b, GraphBatch) or b.schema_version != version:
        raise ValueError('GraphBatch v2 required' if _material else 'GraphBatch v1 required')
    gids = b.graph_ids
    if type(gids) is not tuple or not gids or any(type(x) is not str or not x.strip() for x in gids) or len(set(gids)) != len(gids):
        raise ValueError('unique nonempty graph IDs required')
    _text(b.length_unit,'length_unit'); _text(b.coordinate_frame,'coordinate_frame')
    def arr(name, kinds, shape=None):
        a = getattr(b,name)
        if np.ma.isMaskedArray(a) or not isinstance(a,np.ndarray) or a.dtype.kind not in kinds:
            raise ValueError(name+' invalid array type')
        if shape is not None and a.shape != shape:
            raise ValueError(name+' invalid shape')
        if not np.isfinite(a).all():
            raise ValueError(name+' nonfinite')
        if name != 'message_orientation' and a.dtype.kind in 'iu' and (np.any(a<0) or (a.size and int(a.max())>np.iinfo(np.int64).max)):
            raise ValueError(name+' invalid int64 range')
        if a.dtype.kind in 'iuf':
            with np.errstate(over='ignore',under='ignore'):
                converted=np.array(a,dtype=np.float64)
            if not np.isfinite(converted).all() or np.any((a!=0)&(converted==0)):
                raise ValueError(name+' exceeds float64 range')
        return a
    ni=arr('node_ids','iu');ei=arr('edge_ids','iu')
    if ni.ndim!=1 or ei.ndim!=1:raise ValueError('IDs must be one dimensional')
    n,e,B=len(ni),len(ei),len(gids)
    pos=arr('positions','iuf',(n,3));dims=arr('dimensions','iu',(B,))
    if not np.isin(dims,[2,3]).all():raise ValueError('invalid dimensions')
    def ptr(name,size,end):
        a=arr(name,'iu',(size+1,))
        if a[0]!=0 or a[-1]!=end or np.any(a[1:]<a[:-1]):raise ValueError(name+' invalid pointer')
        return a
    nptr=ptr('node_ptr',B,n);eptr=ptr('edge_ptr',B,e)
    ip=arr('internal_points','iuf')
    if ip.ndim!=2 or ip.shape[1]!=3:raise ValueError('invalid internal points')
    iptr=ptr('internal_ptr',e,len(ip));avail=arr('internal_available','b',(e,))
    if np.any((~avail)&(iptr[1:]!=iptr[:-1])):raise ValueError('internal availability mismatch')
    physical=arr('edge_index','iu',(2,e));msg=arr('message_index','iu',(2,2*e))
    parents=arr('message_parent_rows','iu',(2*e,));mids=arr('message_parent_ids','iu',(2*e,));orient=arr('message_orientation','i',(2*e,))
    # Signed orientation is not an ID; validate separately below.
    if not (np.array_equal(parents,np.repeat(np.arange(e),2)) and np.array_equal(mids,np.repeat(ei,2)) and np.array_equal(orient,np.tile([1,-1],e)) and np.array_equal(msg[:,::2],physical) and np.array_equal(msg[:,1::2],physical[::-1])):
        raise ValueError('message parent/direction mapping mismatch')
    if np.any(physical[0]==physical[1]):raise ValueError('self loops unsupported')
    names=b.edge_feature_names
    if type(names) is not tuple or len(set(names))!=len(names) or any(type(x) is not str or x not in ('radius','chord_length','polyline_length') for x in names):raise ValueError('invalid snapshot feature names')
    values=arr('edge_features','iuf',(e,len(names)));mask=arr('edge_feature_available','b',values.shape)
    if np.any(values[~mask]!=0):raise ValueError('missing snapshot values require zero placeholder')
    for field,size in [('graph_metadata',B),('node_metadata',n),('edge_metadata',e)]:
        v=getattr(b,field)
        if type(v) is not tuple or len(v)!=size or any(not isinstance(x,Mapping) for x in v):raise ValueError('metadata alignment mismatch')
        _plain(v)
    for i in range(B):
        lo,hi,elo,ehi=map(int,(nptr[i],nptr[i+1],eptr[i],eptr[i+1]))
        if lo==hi or np.any(ni[lo+1:hi]<=ni[lo:hi-1]) or len(set(ei[elo:ehi]))!=ehi-elo:raise ValueError('empty graph or invalid local IDs')
        if np.any(physical[:,elo:ehi]<lo) or np.any(physical[:,elo:ehi]>=hi):raise ValueError('cross-graph/out-of-bounds endpoints')
    return n,e,B


def _normalized(raw, mask, offset, scale):
    raw=np.array(raw,dtype=np.float64,copy=True);mask=np.array(mask,dtype=bool,copy=True)
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        difference=raw-offset; norm=difference/scale
        if np.any(mask & ((raw!=offset)&(difference==0))) or np.any(mask & ((difference!=0)&(norm==0))):
            raise ValueError('normalization loses nonzero value')
        norm=np.where(mask,norm,0.0);out=norm.astype(np.float32)
    if not np.isfinite(out).all() or not np.isfinite(norm).all() or np.any((norm!=0)&(out==0)):
        raise ValueError('normalized values exceed finite float32 range')
    return out


@dataclass(frozen=True)
class PreparedGINE:
    """Owned mutable tensors; immutable serialized identity/provenance, no source alias."""
    node_features: object
    edge_features: object
    edge_index: object
    graph_index: object
    config_json: str
    provenance_json: str


def prepare_gine(batch, *, config, source_groups, external=None, material_view=None):
    """Detach NumPy/JSON snapshot into CPU tensors. External torch tensors reject."""
    _require();c=_config(config)
    if c['version'] == _MATERIAL_VERSION:
        if external is not None:raise ValueError('external features unsupported by material profile')
        nodes,edge,indices,membership,cj,pj = _material_inputs(batch, config=c, source_groups=source_groups, material_view=material_view)
        prepared=PreparedGINE(torch.tensor(nodes),torch.tensor(edge),torch.tensor(indices,dtype=torch.long),torch.tensor(membership,dtype=torch.long),cj,pj)
        if prepared.node_features.numpy().tobytes()!=nodes.tobytes() or prepared.edge_features.numpy().tobytes()!=edge.tobytes():
            raise ValueError('material tensor conversion changed stored float32 bits')
        return prepared
    if material_view is not None:raise ValueError('material_view requires its explicit profile')
    n,e,B=_snapshot(batch)
    if batch.length_unit!=c['length_unit'] or batch.coordinate_frame!=c['coordinate_frame']:raise ValueError('unit/frame schema mismatch')
    if type(source_groups) is not dict or set(source_groups)!=set(batch.graph_ids) or any(type(v) is not str or not v.strip() for v in source_groups.values()):raise ValueError('exact graph-ID source_groups required')
    excols={f['source']['external']:f for f in c['features'] if 'external' in f['source']}
    if excols:
        if type(external) is not dict or set(external)!={'units','rows'} or type(external['units']) is not dict or external['units']!={k:f['unit'] for k,f in excols.items()} or type(external['rows']) is not dict or set(external['rows'])!=set(batch.graph_ids):raise ValueError('external units/graph coverage mismatch')
        for i,gid in enumerate(batch.graph_ids):
            rows=external['rows'][gid];expected=set(map(int,batch.edge_ids[batch.edge_ptr[i]:batch.edge_ptr[i+1]]))
            if type(rows) is not dict or any(type(k) is not int or k<0 for k in rows) or set(rows)!=expected:raise ValueError('external exact physical edge-ID coverage required')
            for row in rows.values():
                if type(row) is not dict or set(row)!=set(excols):raise ValueError('external field coverage mismatch')
                for v in row.values():
                    if type(v) is not dict or set(v)!={'value','available'} or type(v['available']) is not bool:raise ValueError('external value/mask schema invalid')
                    if v['available']:_real(v['value'],'external value')
                    elif v['value'] is not None:raise ValueError('missing external value must be None')
    elif external is not None:raise ValueError('unexpected external features')
    raw=np.zeros((e,len(c['features'])));mask=np.zeros(raw.shape,dtype=bool)
    for j,f in enumerate(c['features']):
        if 'snapshot' in f['source']:
            field=f['source']['snapshot']
            if field not in batch.edge_feature_names:raise ValueError('selected snapshot column absent')
            k=batch.edge_feature_names.index(field);raw[:,j]=batch.edge_features[:,k];mask[:,j]=batch.edge_feature_available[:,k]
        else:
            field=f['source']['external']
            for i,gid in enumerate(batch.graph_ids):
                for r in range(int(batch.edge_ptr[i]),int(batch.edge_ptr[i+1])):
                    v=external['rows'][gid][int(batch.edge_ids[r])][field];mask[r,j]=v['available'];raw[r,j]=_real(v['value'],'external value') if v['available'] else 0
        if f['required'] and not mask[:,j].all():raise ValueError('required feature missing')
    norm=_normalized(raw,mask,np.array([f['offset'] for f in c['features']]),np.array([f['scale'] for f in c['features']]))
    edge=np.concatenate([norm,mask.astype(np.float32)],axis=1)[batch.message_parent_rows]
    membership=np.repeat(np.arange(B),np.diff(batch.node_ptr).astype(np.int64))
    nodes=np.column_stack([batch.positions,batch.dimensions[membership]])
    nodes=_normalized(nodes,np.ones(nodes.shape,dtype=bool),np.array(c['node_offset']),np.array(c['node_scale']))
    prov={'graph_ids':list(batch.graph_ids),'source_groups':[source_groups[g] for g in batch.graph_ids],
          'node_ids':batch.node_ids.tolist(),'physical_edge_ids':batch.edge_ids.tolist(),
          'node_ptr':batch.node_ptr.tolist(),'edge_ptr':batch.edge_ptr.tolist(),
          'physical_edge_index':batch.edge_index.tolist(),'message_parent_rows':batch.message_parent_rows.tolist(),
          'message_parent_ids':batch.message_parent_ids.tolist(),'message_orientation':batch.message_orientation.tolist(),
          'positions':batch.positions.tolist(),'internal_points':batch.internal_points.tolist(),
          'internal_ptr':batch.internal_ptr.tolist(),'internal_available':batch.internal_available.tolist(),
          'graph_metadata':_plain(batch.graph_metadata),'node_metadata':_plain(batch.node_metadata),'edge_metadata':_plain(batch.edge_metadata)}
    return PreparedGINE(torch.tensor(nodes),torch.tensor(edge),torch.tensor(np.array(batch.message_index,copy=True),dtype=torch.long),torch.tensor(membership,dtype=torch.long),_json(c),_json(prov))


def _provenance(text, *, config=None):
    if type(text) is not str:
        raise ValueError('prepared provenance must be JSON text')
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('duplicate provenance key')
            result[k]=v
        return result
    try:p=json.loads(text,object_pairs_hook=unique)
    except (json.JSONDecodeError,RecursionError) as exc:raise ValueError('invalid provenance JSON') from exc
    _plain(p)
    expected={'graph_ids','source_groups','node_ids','physical_edge_ids','node_ptr','edge_ptr',
              'physical_edge_index','message_parent_rows','message_parent_ids','message_orientation',
              'positions','internal_points','internal_ptr','internal_available',
              'graph_metadata','node_metadata','edge_metadata'}
    material = config is not None and config.get('version') == _MATERIAL_VERSION
    if material:expected = expected | {'material_profile'}
    if type(p) is not dict or set(p)!=expected:raise ValueError('prepared provenance fields mismatch')
    def texts(name,size=None):
        a=p[name]
        if type(a) is not list or (size is not None and len(a)!=size) or any(type(x) is not str or not x.strip() for x in a):raise ValueError('invalid provenance '+name)
        return a
    gids=texts('graph_ids');B=len(gids)
    if not B or len(set(gids))!=B:raise ValueError('invalid provenance graph IDs')
    texts('source_groups',B)
    def ints(name,size=None,signed=False):
        a=p[name]
        if type(a) is not list or (size is not None and len(a)!=size) or any(type(x) is not int or (not signed and x<0) or abs(x)>np.iinfo(np.int64).max for x in a):raise ValueError('invalid provenance '+name)
        return a
    ni=ints('node_ids');ei=ints('physical_edge_ids');n,e=len(ni),len(ei)
    def ptr(name,size,end):
        a=ints(name,size+1)
        if a[0]!=0 or a[-1]!=end or any(x>y for x,y in zip(a[:-1],a[1:])):raise ValueError('invalid provenance '+name)
        return a
    nptr=ptr('node_ptr',B,n);eptr=ptr('edge_ptr',B,e)
    def coords(name,size=None):
        a=p[name]
        if type(a) is not list or (size is not None and len(a)!=size) or any(type(row) is not list or len(row)!=3 for row in a):raise ValueError('invalid provenance '+name)
        for row in a:
            for v in row:_real(v,'provenance coordinate')
        return a
    coords('positions',n);points=coords('internal_points');iptr=ptr('internal_ptr',e,len(points));available=p['internal_available']
    if type(available) is not list or len(available)!=e or any(type(x) is not bool for x in available):raise ValueError('invalid provenance geometry mask')
    if any(not available[i] and iptr[i]!=iptr[i+1] for i in range(e)):raise ValueError('provenance geometry availability mismatch')
    ph=p['physical_edge_index']
    if type(ph) is not list or len(ph)!=2 or any(type(row) is not list or len(row)!=e or any(type(v) is not int or v<0 or v>=n for v in row) for row in ph):raise ValueError('invalid provenance physical endpoints')
    if any(a==b for a,b in zip(*ph)):raise ValueError('provenance self loops unsupported')
    for i in range(B):
        lo,hi,elo,ehi=nptr[i],nptr[i+1],eptr[i],eptr[i+1]
        if lo==hi or any(x>=y for x,y in zip(ni[lo:hi-1],ni[lo+1:hi])) or len(set(ei[elo:ehi]))!=ehi-elo:raise ValueError('provenance local IDs invalid')
        if any(v<lo or v>=hi for row in ph for v in row[elo:ehi]):raise ValueError('prepared cross-graph edge')
    parents=ints('message_parent_rows',2*e);ids=ints('message_parent_ids',2*e);directions=ints('message_orientation',2*e,True)
    if parents!=[i for i in range(e) for _ in range(2)] or ids!=[i for i in ei for _ in range(2)] or directions!=[1,-1]*e:raise ValueError('provenance message parent mapping mismatch')
    for name,size in [('graph_metadata',B),('node_metadata',n),('edge_metadata',e)]:
        a=p[name]
        if type(a) is not list or len(a)!=size or any(type(row) is not dict for row in a):raise ValueError('provenance metadata alignment mismatch')
    if material:_validate_material_provenance(p['material_profile'],config,n,e)
    return p


# This opt-in bridge is pure NumPy until prepare_gine constructs owned tensors.
# Canonical trusted snapshots are a precondition, not hostile-object isolation.
def _material_profile_spec(value):
    from fibernet.ml.graph_data import _MATERIAL_UNITS, _material_number
    fixed = dict(view_schema='fibernet.material_feature_view.v1',
                 batch_schema='fibernet.graph_batch.v2',
                 provenance_schema='fibernet.graph_batch.provenance.v1',
                 material_assumption='isotropic_scalar',
                 origin_policy='retain_stored_origin_unknown')
    if type(value) is not dict or any(type(k) is not str for k in value) or set(value) != set(fixed) | {'fields','units','scales'}:
        raise ValueError('material_profile requires exactly its declared keys')
    if any(type(value[k]) is not str or value[k] != v for k,v in fixed.items()):
        raise ValueError('unsupported material profile schema/assumption/origin')
    fields=value['fields']
    if (type(fields) not in (list,tuple) or not fields or
            any(type(x) is not str or x not in _MATERIAL_UNITS for x in fields) or len(set(fields))!=len(fields)):
        raise ValueError('material fields must be ordered unique supported names')
    for key in ('units','scales'):
        if type(value[key]) is not dict or any(type(k) is not str for k in value[key]) or set(value[key])!=set(fields):
            raise ValueError('material '+key+' must exactly cover fields')
    if any(type(value['units'][k]) is not str or value['units'][k]!=_MATERIAL_UNITS[k] for k in fields):
        raise ValueError('material units must match explicit native SI units')
    scales={k:_material_number(value['scales'][k],k+' scale',positive=True) for k in fields}
    return dict(fixed,fields=list(fields),units={k:value['units'][k] for k in fields},scales=scales)


def _material_config(features, length_unit, coordinate_frame, node_offset, node_scale, pooling, profile):
    from fibernet.ml.graph_data import _material_number, _material_bits
    p=_material_profile_spec(profile)
    if type(length_unit) is not str or length_unit!='m':raise ValueError('material profile requires explicit geometry unit m')
    _text(coordinate_frame,'coordinate_frame')
    if type(pooling) is not str or pooling not in ('sum','mean'):raise ValueError('pooling must be sum or mean')
    if type(features) not in (list,tuple) or not features:raise ValueError('features must be a nonempty ordered list')
    names,sources,cols,selected=set(),set(),[],[]
    for f in features:
        if type(f) is not dict or any(type(k) is not str for k in f) or set(f)!={'name','source','unit','offset','scale','required','parity'}:
            raise ValueError('feature must have exactly the declared schema keys')
        name=_text(f['name'],'feature name');src=f['source']
        if name in names or name.lower() in _RESERVED:raise ValueError('duplicate or identity feature name')
        if type(src) is not dict or len(src)!=1 or any(type(k) is not str for k in src) or next(iter(src)) not in ('snapshot','material_view'):
            raise ValueError('material profile permits snapshot or material_view source only')
        kind,field=next(iter(src.items()));_text(field,'source field')
        if (kind,field) in sources or field.lower() in _RESERVED:raise ValueError('duplicate or identity feature source')
        unit=_text(f['unit'],'unit');offset=_material_number(f['offset'],'offset');scale=_material_number(f['scale'],'scale',positive=True)
        if type(f['required']) is not bool or type(f['parity']) is not str or f['parity']!='even':
            raise ValueError('boolean required and even scalar parity required')
        if kind=='snapshot':
            if field not in ('radius','chord_length','polyline_length') or unit!='m':raise ValueError('unsupported geometry feature/unit')
        else:
            if field not in p['fields'] or unit!=p['units'][field] or _material_bits(scale)!=_material_bits(p['scales'][field]) or _material_bits(offset)&((1<<63)-1):
                raise ValueError('material feature units/scales/zero offset must match profile')
            selected.append(field)
        names.add(name);sources.add((kind,field));cols.append(dict(name=name,source={kind:field},unit=unit,offset=offset,scale=scale,required=f['required'],parity='even'))
    if selected!=p['fields']:raise ValueError('material fields must appear exactly once in profile order')
    def four(values,positive):
        if type(values) not in (list,tuple) or len(values)!=4:raise ValueError('node normalization requires four scalars')
        return [_material_number(x,'node normalization',positive=positive) for x in values]
    return dict(version=_MATERIAL_VERSION,node_schema=['x','y','z','dimension'],dtype='float32',device='cpu',
                features=cols,length_unit='m',coordinate_frame=coordinate_frame,node_offset=four(node_offset,False),
                node_scale=four(node_scale,True),pooling=pooling,hidden=16,layers=2,eps=0.0,material_profile=p)


def _profile_array(value,dtype,shape,label):
    if type(value) is not np.ndarray or value.dtype!=np.dtype(dtype) or value.shape!=shape or value.flags.writeable:
        raise ValueError(label+' must be a canonical read-only array with exact shape/dtype')
    return value


def _profile_float32(value):
    from fibernet.ml.graph_data import _material_bits
    before=_material_bits(value)
    try:raw=struct.pack('=f',value)
    except (OverflowError,struct.error) as exc:raise ValueError('profile conversion exceeds finite float32 range') from exc
    after=struct.unpack('=I',raw)[0]
    if ((after>>23)&255==255 or (before&((1<<63)-1) and not after&((1<<31)-1)) or before>>63!=after>>31):
        raise ValueError('profile float32 conversion lost finite range, nonzero or sign')
    return raw


def _profile_normalized(value,offset,scale):
    from fibernet.ml.graph_data import _material_bits, _material_normalize
    vb,ob=_material_bits(value),_material_bits(offset);magnitude=(1<<63)-1
    # Nonzero raw == offset legitimately centers to zero. Only unequal stored
    # values losing their nonzero difference fail; signed zeros are equal here.
    same=vb==ob or not (vb&magnitude or ob&magnitude)
    difference=value-offset;db=_material_bits(difference)
    if (db>>52)&2047==2047 or (not same and not db&magnitude):
        raise ValueError('profile centering lost finite range or a nonzero difference')
    return _profile_float32(_material_normalize(difference,scale))


def _profile_record(record,kind):
    from fibernet.ml.graph_data import _MATERIAL_UNITS,_MATERIAL_OPTIONAL_SCALARS,_material_number
    if record is None:return
    if type(record) is not kind:raise ValueError('material record container mismatch')
    allowed=set(_MATERIAL_UNITS)|set(_MATERIAL_OPTIONAL_SCALARS)|{'name','extra','stiffness_tensor'}
    for key,value in record.items():
        if type(key) is not str or key not in allowed:raise ValueError('unsupported material record field')
        if key=='name':
            if type(value) is not str:raise ValueError('material name must be text')
        elif key=='extra':
            if type(value) is not kind or len(value):raise ValueError('custom material extra unsupported')
        elif key=='stiffness_tensor':
            if value is not None:raise ValueError('material tensors unsupported')
        elif value is None:
            if key in ('youngs_modulus','density','poissons_ratio'):raise ValueError('required stored property cannot be null')
        else:_material_number(value,key)


def _validate_material_provenance(p,c,n,e):
    fields=c['material_profile']['fields'];F=len(fields)
    keys={'config','property_status','physical_available','node_boundary_face_flags','node_boundary_face_available','node_boundary_face_origin',
          'edge_material_records','edge_material_available','edge_material_origin','edge_segments','edge_segments_available','edge_segments_origin'}
    if type(p) is not dict or set(p)!=keys or _json(p['config'])!=_json(c['material_profile']):raise ValueError('material provenance profile mismatch')
    def rows(key,count,width=None):
        a=p[key]
        if type(a) is not list or len(a)!=count or (width is not None and any(type(x) is not list or len(x)!=width for x in a)):
            raise ValueError('material provenance '+key+' shape mismatch')
        return a
    states=rows('property_status',e,F);available=rows('physical_available',e,F);records=rows('edge_material_records',e)
    ma=rows('edge_material_available',e);mo=rows('edge_material_origin',e)
    for i,record in enumerate(records):
        _profile_record(record,dict);present=record is not None
        if type(ma[i]) is not bool or ma[i]!=present or type(mo[i]) is not str or mo[i]!=('stored_origin_unknown' if present else 'missing'):
            raise ValueError('material provenance availability/origin mismatch')
        for j,field in enumerate(fields):
            state='material_missing' if record is None else 'property_missing' if field not in record else 'property_null' if record[field] is None else 'stored_origin_unknown'
            if type(states[i][j]) is not str or states[i][j]!=state or type(available[i][j]) is not bool or available[i][j]!=(state=='stored_origin_unknown'):
                raise ValueError('material provenance status/mask mismatch')
    faces=rows('node_boundary_face_flags',n,6);fa=rows('node_boundary_face_available',n);fo=rows('node_boundary_face_origin',n)
    for flags,present,origin in zip(faces,fa,fo):
        if any(type(x) is not bool for x in flags) or type(present) is not bool or type(origin) is not str or origin!=('stored_origin_unknown' if present else 'missing') or (not present and any(flags)):
            raise ValueError('cell-face provenance mismatch')
    segments=rows('edge_segments',e);sa=rows('edge_segments_available',e);so=rows('edge_segments_origin',e)
    for count,present,origin in zip(segments,sa,so):
        if type(count) is not int or type(present) is not bool or type(origin) is not str or origin!=('stored_origin_unknown' if present else 'missing') or (present and not 1<=count<=np.iinfo(np.int64).max) or (not present and count!=0):
            raise ValueError('segment provenance mismatch')


def _material_inputs(batch,*,config,source_groups,material_view):
    """Pure preparation contract; not a model call or a public graph interface."""
    from fibernet.ml.graph_data import MaterialFeatureView,_material_identity,_material_bits,_material_number,_material_normalize
    c=_config(config)
    if c['version']!=_MATERIAL_VERSION:raise ValueError('explicit material profile required')
    if type(material_view) is not MaterialFeatureView or material_view.snapshot is not batch:
        raise ValueError('material view must reference the supplied snapshot')
    e=_material_identity(batch);n=len(batch.node_ids)
    if type(batch.edge_feature_names) is not tuple or any(type(x) is not str for x in batch.edge_feature_names):
        raise ValueError('snapshot feature names must be canonical strings')
    # Check exact base arrays before legacy snapshot validation can convert them.
    for name,dtype,shape in [('positions',np.float64,(n,3)),('edge_features',np.float64,(e,len(batch.edge_feature_names))),
                             ('edge_feature_available',bool,(e,len(batch.edge_feature_names))),('internal_ptr',np.int64,(e+1,)),('internal_available',bool,(e,))]:
        _profile_array(getattr(batch,name),dtype,shape,name)
    if type(batch.internal_points) is not np.ndarray or batch.internal_points.ndim!=2 or batch.internal_points.shape[1]!=3:
        raise ValueError('internal_points require a canonical array')
    _profile_array(batch.internal_points,np.float64,batch.internal_points.shape,'internal_points')
    n,e2,B=_snapshot(batch,_material=True);assert e==e2
    if batch.length_unit!=c['length_unit'] or batch.coordinate_frame!=c['coordinate_frame']:raise ValueError('unit/frame schema mismatch')
    if type(source_groups) is not dict or set(source_groups)!=set(batch.graph_ids) or any(type(v) is not str or not v.strip() for v in source_groups.values()):raise ValueError('exact graph-ID source_groups required')
    p=c['material_profile'];v=material_view;F=len(p['fields'])
    if (type(v.schema_version) is not str or v.schema_version!=p['view_schema'] or type(v.feature_names) is not tuple or any(type(x) is not str for x in v.feature_names) or v.feature_names!=tuple(p['fields']) or
            type(v.property_units) is not tuple or any(type(x) is not str for x in v.property_units) or v.property_units!=tuple(p['units'][f] for f in p['fields']) or type(v.scales) is not tuple or len(v.scales)!=F or
            any(type(x) is not float or _material_bits(x)!=_material_bits(p['scales'][f]) for f,x in zip(p['fields'],v.scales)) or
            type(v.material_assumption) is not str or v.material_assumption!=p['material_assumption'] or type(v.origin_policy) is not str or v.origin_policy!=p['origin_policy']):
        raise ValueError('material view/profile schema, field, unit or scale mismatch')
    values=_profile_array(v.values,np.float64,(e,F),'material values');available=_profile_array(v.available,bool,(e,F),'material availability')
    if type(v.property_status) is not tuple or len(v.property_status)!=e or any(type(row) is not tuple or len(row)!=F for row in v.property_status):raise ValueError('material status shape mismatch')
    q=batch.provenance
    for record in q.edge_material_records:_profile_record(record,MappingProxyType)
    payload=dict(config=p,property_status=_plain(v.property_status),physical_available=available.tolist())
    for name,dtype,shape in [('node_boundary_face_flags',bool,(n,6)),('node_boundary_face_available',bool,(n,)),('edge_material_available',bool,(e,)),('edge_segments',np.int64,(e,)),('edge_segments_available',bool,(e,))]:
        payload[name]=_profile_array(getattr(q,name),dtype,shape,name).tolist()
    for name in ('node_boundary_face_origin','edge_material_records','edge_material_origin','edge_segments_origin'):
        if type(getattr(q,name)) is not tuple:raise ValueError('material provenance tuple required')
        payload[name]=_plain(getattr(q,name))
    _validate_material_provenance(payload,c,n,e)
    _profile_array(batch.positions,np.float64,(n,3),'positions');_profile_array(batch.edge_features,np.float64,(e,len(batch.edge_feature_names)),'edge features')
    _profile_array(batch.edge_feature_available,bool,batch.edge_features.shape,'edge availability')
    material_scalars=struct.unpack('='+str(e*F)+'d',values.tobytes()) if e else ()
    geometry_scalars=struct.unpack('='+str(batch.edge_features.size)+'d',batch.edge_features.tobytes()) if batch.edge_features.size else ()
    packed=[];masks=[]
    for r in range(e):
        row=[];mask=[]
        for f in c['features']:
            if 'material_view' in f['source']:
                k=p['fields'].index(f['source']['material_view']);present=bool(available[r,k]);value=material_scalars[r*F+k]
                if not present and _material_bits(value)!=0:raise ValueError('missing material requires positive zero placeholder')
                if present:
                    field=f['source']['material_view']
                    # Validate against the raw source record, without dividing
                    # the already-normalized view value a second time.
                    reference=_material_normalize(_material_number(q.edge_material_records[r][field],field),p['scales'][field])
                    if _material_bits(value)!=_material_bits(reference):
                        raise ValueError('material view value disagrees with stored record/scale')
                raw=_profile_float32(value) if present else bytes(4)
            else:
                field=f['source']['snapshot']
                if field not in batch.edge_feature_names:raise ValueError('selected snapshot column absent')
                k=batch.edge_feature_names.index(field);present=bool(batch.edge_feature_available[r,k]);value=geometry_scalars[r*len(batch.edge_feature_names)+k]
                if not present and _material_bits(value)!=0:raise ValueError('missing geometry requires positive zero placeholder')
                raw=_profile_normalized(value,f['offset'],f['scale']) if present else bytes(4)
            if f['required'] and not present:raise ValueError('required feature missing')
            row.append(raw);mask.append(present)
        packed.extend(row);masks.append(mask)
    width=len(c['features']);physical=np.frombuffer(b''.join(packed),dtype=np.float32).copy().reshape(e,width)
    edge=np.concatenate([physical,np.array(masks,dtype=np.float32).reshape(e,width)],axis=1)[batch.message_parent_rows]
    membership=np.repeat(np.arange(B,dtype=np.int64),np.diff(batch.node_ptr))
    positions=struct.unpack('='+str(n*3)+'d',batch.positions.tobytes());node_bytes=[]
    for r in range(n):
        raw=positions[3*r:3*r+3]+(float(batch.dimensions[membership[r]]),)
        node_bytes.extend(_profile_normalized(value,c['node_offset'][j],c['node_scale'][j]) for j,value in enumerate(raw))
    nodes=np.frombuffer(b''.join(node_bytes),dtype=np.float32).copy().reshape(n,4)
    prov=dict(graph_ids=list(batch.graph_ids),source_groups=[source_groups[g] for g in batch.graph_ids],node_ids=batch.node_ids.tolist(),physical_edge_ids=batch.edge_ids.tolist(),
              node_ptr=batch.node_ptr.tolist(),edge_ptr=batch.edge_ptr.tolist(),physical_edge_index=batch.edge_index.tolist(),message_parent_rows=batch.message_parent_rows.tolist(),
              message_parent_ids=batch.message_parent_ids.tolist(),message_orientation=batch.message_orientation.tolist(),positions=batch.positions.tolist(),internal_points=batch.internal_points.tolist(),
              internal_ptr=batch.internal_ptr.tolist(),internal_available=batch.internal_available.tolist(),graph_metadata=_plain(batch.graph_metadata),node_metadata=_plain(batch.node_metadata),edge_metadata=_plain(batch.edge_metadata),material_profile=payload)
    pj=_json(prov);_provenance(pj,config=c)
    if physical.tobytes()!=b''.join(packed) or nodes.tobytes()!=b''.join(node_bytes):raise ValueError('profile array conversion changed stored bits')
    return nodes,edge,np.array(batch.message_index,dtype=np.int64,copy=True),membership,_json(c),pj


class GINERegressor(nn.Module if nn is not None else object):
    """Two optional upstream GINE layers; forward is differentiable, no fit API."""
    def __init__(self,config):
        _require();super().__init__();c=_config(config);self._config_json=_json(c)
        self.input_proj=nn.Linear(4,16)
        self.layers=nn.ModuleList([GINEConv(nn.Sequential(nn.Linear(16,16),nn.ReLU(),nn.Linear(16,16)),eps=0,train_eps=False,edge_dim=2*len(c['features']),aggr='add') for _ in range(2)])
        self.head=nn.Linear(16,1);self.pooling=c['pooling']

    def forward(self,prepared):
        if not isinstance(prepared,PreparedGINE) or prepared.config_json!=self._config_json:raise ValueError('prepared model schema mismatch')
        c=json.loads(self._config_json);p=_provenance(prepared.provenance_json,config=c);B=len(p['graph_ids']);x=prepared.node_features;a=prepared.edge_features;idx=prepared.edge_index;group=prepared.graph_index
        n=len(p['node_ids']);m=2*len(p['physical_edge_ids'])
        for t,shape,dtype in [(x,(n,4),torch.float32),(a,(m,2*len(c['features'])),torch.float32),(idx,(2,m),torch.long),(group,(n,),torch.long)]:
            if not isinstance(t,torch.Tensor) or t.device.type!='cpu' or t.dtype!=dtype or tuple(t.shape)!=shape or not torch.isfinite(t).all():raise ValueError('prepared tensor shape/dtype/device/finite mismatch')
        if any(q.device.type!='cpu' or q.dtype!=torch.float32 or not torch.isfinite(q).all() for q in self.parameters()):raise ValueError('model requires finite CPU float32 parameters')
        expected=torch.tensor(p['physical_edge_index'],dtype=torch.long).reshape(2,-1)
        if not torch.equal(idx[:,::2],expected) or not torch.equal(idx[:,1::2],expected.flip(0)):raise ValueError('prepared connectivity mutated')
        expected_group=torch.repeat_interleave(torch.arange(B),torch.tensor(np.diff(p['node_ptr']),dtype=torch.long))
        if not torch.equal(group,expected_group):raise ValueError('prepared membership mutated')
        if (m and (idx.min()<0 or idx.max()>=n)) or (n and (group.min()<0 or group.max()>=B)):raise ValueError('prepared index out of range')
        width=len(c['features']);mask=a[:,width:]
        if not torch.all((mask==0)|(mask==1)) or torch.any(a[:,:width][mask==0]!=0):raise ValueError('prepared missingness mask invalid')
        if not torch.equal(a[::2],a[1::2]):raise ValueError('even scalar reverse features differ')
        for j,f in enumerate(c['features']):
            if f['required'] and not torch.all(mask[:,j]==1):raise ValueError('prepared required feature missing')
        if c['version'] == _MATERIAL_VERSION:
            availability = p['material_profile']['physical_available']
            fields = c['material_profile']['fields']
            for j,f in enumerate(c['features']):
                if 'material_view' in f['source']:
                    k=fields.index(f['source']['material_view'])
                    expected_mask=torch.tensor([availability[r][k] for r in p['message_parent_rows']],dtype=torch.float32)
                    if not torch.equal(mask[:,j],expected_mask):raise ValueError('material provenance/mask mismatch')
        x=self.input_proj(x)
        for layer in self.layers:x=layer(x,idx,a)
        pooled=x.new_zeros((B,16)).index_add(0,group,x)
        if self.pooling=='mean':pooled=pooled/torch.bincount(group,minlength=B).to(x.dtype).unsqueeze(1)
        prediction=self.head(pooled).squeeze(-1)
        if tuple(prediction.shape)!=(B,) or not torch.isfinite(prediction).all():raise ValueError('nonfinite or invalid prediction')
        return prediction

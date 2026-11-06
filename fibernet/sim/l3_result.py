"""Bounded saved-frame result family using StructureGraph and an L3 binding.

This is a saved-data projection, never a SimResult or a solver. Caller-supplied
hashes must come from a trusted source, not be invented from untrusted payloads.
No provider, raw archive reader, runtime admission, or physics acceptance is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

from fibernet.core.structure_graph import StructureGraph, SNode, SEdge
from fibernet.core.material import Material
from fibernet.sim.l3 import L3Binding, L3InputError, bind_l3

__all__ = ["L3ProjectionError", "L3ResultProjection", "L3RenderInputs",
           "project_l3_result", "load_l3_projection", "l3_render_inputs"]

_FRAME_SCHEMAS = ("fibernet.l3-frame-view/1", "private.l3.saved-frame-view/1")

MAX_BYTES = 8 * 1024**2
MAX_MEMBERS = 100000
MAX_DEPTH = 32
MAX_NODES = 4096
MAX_EDGES = 8192
UNITS = {'displacements_m':'m', 'positions_m':'m', 'rotations_rad':'rad', 'active':'boolean',
    'axial_strain':'1', 'axial_force_N':'N', 'end_moments_Nm':'N m', 'max_tensile_strain':'1', 'element_energy_J':'J'}
NODE_FIELDS = {'displacements_m', 'positions_m', 'rotations_rad'}
SCHEMA = 'fibernet.l3-result-projection/1'
POLICIES = {'projection':'saved L3 fields; not SimResult; not restart or physical validation',
    'omitted_native_fields':['gradient'],'rotation_policy':'retained separately in radians',
    'force_policy':'retained native axial force and endpoint moments; no spring recomputation'}

class L3ProjectionError(L3InputError):
    """Saved frame bytes do not satisfy this bounded result contract."""
    pass


def _require(ok, message):
    if not ok:
        raise L3ProjectionError(message)


def _raw(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hash(value):
    _require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value), 'Explicit lowercase SHA256 required')
    return value


def _bounded(value):
    stack=[(value,0)]; seen=set(); count=0; size=0
    while stack:
        item,depth=stack.pop();count+=1
        _require(count<=MAX_MEMBERS and depth<=MAX_DEPTH, 'JSON member/depth ceiling exceeded')
        if type(item) in (dict,list):
            _require(id(item) not in seen, 'Cyclic/aliased input container');seen.add(id(item))
            if type(item) is dict:
                for key,child in item.items():
                    _require(type(key) is str, 'JSON keys must be strings');size+=len(key.encode());stack.append((child,depth+1))
            else:stack.extend((x,depth+1) for x in item)
        elif type(item) is str:size+=len(item.encode())
        elif type(item) in (int,float):
            _require(type(item) is not int or -(2**63)<=item<=2**63-1, 'Integer exceeds int64')
            _require(math.isfinite(item),'Nonfinite scalar');size+=24
        else:_require(item is None or type(item) is bool,'Expected literal JSON');size+=5
        _require(size<=MAX_BYTES,'JSON byte ceiling exceeded')


def _decode(raw, expected):
    _require(type(raw) is bytes and len(raw)<=MAX_BYTES,'Bounded bytes required')
    _require(_sha(raw)==_hash(expected),'Source identity mismatch')
    def unique(pairs):
        result={}
        for k,v in pairs:
            _require(k not in result,'Duplicate JSON key');result[k]=v
        return result
    try:
        value=json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(L3ProjectionError('Nonfinite JSON')))
        _bounded(value)
    except (ValueError,UnicodeError,RecursionError) as exc:raise L3ProjectionError('Malformed bounded JSON') from exc
    return value


def _ids(value, count=None):
    _require(type(value) is list and all(type(x) is int and 0<=x<=2**63-1 for x in value),'Literal nonnegative ID list required')
    _require(len(set(value))==len(value) and (count is None or len(value)==count),'Duplicate or wrong-length IDs')
    return value


def _vector(value, size, label, boolean=False):
    _require(type(value) is list and len(value)==size,'Wrong '+label+' shape')
    if boolean:_require(all(type(x) is bool for x in value),'Literal boolean active mask required')
    else:_require(all(type(x) in (int,float) and math.isfinite(x) for x in value),'Finite numeric '+label+' required')


def _matrix(value, n, width, label):
    _require(type(value) is list and len(value)==n,'Wrong '+label+' row count')
    for row in value:_vector(row,width,label)


def _binding(graph, binding):
    _require(type(graph) is StructureGraph and type(graph.nodes) is dict and type(graph.edges) is dict,'Existing authoritative StructureGraph required')
    _require(1<=len(graph.nodes)<=MAX_NODES and 1<=len(graph.edges)<=MAX_EDGES,'Graph exceeds finite projection profile')
    _require(all(type(node) is SNode for node in graph.nodes.values()) and all(type(edge) is SEdge and type(edge.material) is Material for edge in graph.edges.values()),'Concrete graph/material records required')
    _require(type(binding) is L3Binding,'Explicit L3Binding required')
    b=_decode(binding.canonical,binding.sha256)
    _require(type(b) is dict and b.get('schema')=='fibernet.l3.binding/1','Unsupported binding')
    # Bound public graph serialization fields not covered by bind_l3's profile.
    box=graph.box_size
    if box is not None:
        _require(type(box).__module__=='numpy' and type(box).__name__=='ndarray'
            and box.shape in ((2,),(3,)) and box.dtype.kind in 'iuf','Small numeric graph box required')
        _bounded(box.tolist())
    _bounded([graph.tolerance,graph._next_node_id,graph._next_edge_id])
    # Bound provenance before public bind_l3 performs graph serialization.
    provenance=[graph.metadata]+[node.metadata for node in graph.nodes.values()]
    for edge in graph.edges.values():provenance.extend((edge.metadata,edge.material.extra,edge.material.name))
    size=0
    for item in provenance:
        _bounded(item);size+=len(_raw(item));_require(size<=MAX_BYTES,'Aggregate graph provenance ceiling exceeded')
    rebuilt=bind_l3(graph,coordinate_unit=b['coordinate_unit'],radius_unit=b['radius_unit'],
        edge_data={x['edge_id']:x['data'] for x in b['edge_data']},fixed_node_ids=b['fixed_node_ids'],
        loaded_node_ids=b['loaded_node_ids'],model=b['model'])
    _require(len(rebuilt.canonical)<=MAX_BYTES and rebuilt.canonical==binding.canonical and rebuilt.sha256==binding.sha256,'Graph/material/unit binding changed')
    return b


def _payload(value, b):
    _require(type(value) is dict,'Frame projection must be an object')
    nodes=_ids(value['node_ids']);edges=_ids(value['edge_ids'])
    _require(set(nodes)=={n['id'] for n in b['record']['nodes']} and set(edges)=={e['id'] for e in b['record']['segments']},'Stable IDs must cover the bound physical graph')
    n,m=len(nodes),len(edges)
    fields=value['fields'];_require(type(fields) is dict and set(fields)==set(UNITS),'Exact retained L3 fields required')
    _require(_raw(value['field_units'])==_raw(UNITS),'Explicit SI field units required')
    rows={key:'node_ids' if key in NODE_FIELDS else 'edge_ids' for key in UNITS}
    _require(_raw(value['field_rows'])==_raw(rows),'Field row axes differ from stable IDs')
    components={'displacements_m':['x','y'],'positions_m':['x','y'],'rotations_rad':['theta'],'end_moments_Nm':['node_i','node_j']}
    _require(_raw(value['field_components'])==_raw(components),'Oriented field components differ')
    for key in ('displacements_m','positions_m'):_matrix(fields[key],n,2,key)
    _vector(fields['rotations_rad'],n,'rotations_rad')
    _vector(fields['active'],m,'active',True)
    _matrix(fields['end_moments_Nm'],m,2,'end_moments_Nm')
    for key in ('axial_strain','axial_force_N','max_tensile_strain','element_energy_J'):_vector(fields[key],m,key)
    _matrix(value['reference_positions_m'],n,2,'reference_positions_m')
    _require(type(value['edge_node_ids']) is list and len(value['edge_node_ids'])==m,'Endpoint rows required')
    reference={x['id']:x['position'] for x in b['record']['nodes']}
    topology={x['id']:[x['node_i'],x['node_j']] for x in b['record']['segments']}
    _require(_raw(value['reference_positions_m'])==_raw([reference[i] for i in nodes]),'Reference coordinates differ from binding')
    _require(_raw(value['edge_node_ids'])==_raw([topology[i] for i in edges]),'Oriented endpoint IDs differ from binding')
    for ref,u,p in zip(value['reference_positions_m'],fields['displacements_m'],fields['positions_m']):
        _require([ref[j]+u[j] for j in range(2)]==p,'Position/displacement identity differs')
    original={key:b[key] for key in ('coordinate_unit','radius_unit','coordinate_scale_to_m','radius_scale_to_m')}
    _require(_raw(value['original_units'])==_raw(original),'Original graph units differ from binding')
    inactive=[x['edge_id'] for x in b['edge_data'] if not x['data']['initially_active']]
    _require(_raw(value['initial_inactive_ids'])==_raw(inactive),'Initial inactive IDs differ from binding')
    active=[i for i,a in zip(edges,fields['active']) if a]
    failed=[i for i,a in zip(edges,fields['active']) if not a and i not in set(inactive)]
    _require(_raw(value['active_edge_ids'])==_raw(active) and _raw(value['failed_edge_ids'])==_raw(failed),'Active/failed ID maps differ from retained mask')
    _require(not set(active)&set(inactive),'Initially inactive edge reactivated')
    new=_ids(value['newly_failed_ids']);_require(set(new)<=set(failed),'New failures are not in cumulative failures')
    _require(type(value['frame']) is int and 0<=value['frame']<=2**63-1,'Literal nonnegative frame required')
    observation=value['observation'];_require(type(observation) is dict,'Observation object required')
    required={'step','phase','u_m','reaction_N','internal_energy_J','external_work_J','equilibrium_jump_energy_J','energy_balance_residual_J'}
    _require(set(observation)==required,'Exact observation channels required')
    _require(type(observation['step']) is int and observation['step']>=0,'Literal nonnegative step required')
    _require(type(observation['phase']) is str and re.fullmatch('[a-z][a-z0-9_]{0,31}',observation['phase']) is not None,'Bounded phase identifier required')
    for key in required-{'step','phase'}:_require(type(observation[key]) in (int,float) and math.isfinite(observation[key]),'Finite observation required')
    _require(value['new_execution'] is False and value['provider_license_verified'] is False,'Projection cannot certify new execution or provider license')
    _require(value['historical_native_state']=='succeeded' and value['historical_outer_status'] in ('FAILED_WORKER','SUCCEEDED'),'Unsupported historical status profile')
    # Strict allowlist: no arbitrary provider/path/research metadata is exported.
    keys=('frame','node_ids','edge_ids','reference_positions_m','edge_node_ids','fields','field_units','field_rows','field_components',
        'active_edge_ids','failed_edge_ids','initial_inactive_ids','newly_failed_ids','observation','original_units',
        'new_execution','provider_license_verified','historical_native_state','historical_outer_status')
    return json.loads(_raw({key:value[key] for key in keys}))


@dataclass(frozen=True)
class L3ResultProjection:
    """Detached result-family record. Manual construction is not authentication."""
    canonical: bytes
    sha256: str

    def to_dict(self):
        return json.loads(self.canonical)

    def compute_detailed(self, *args, **kwargs):
        raise L3ProjectionError('L3 projection cannot run geometric-rest spring postprocessing')


def project_l3_result(graph, binding, *, frame_view_bytes, expected_view_sha256):
    """Import an explicitly pinned saved-frame view using the unchanged binding.

    Hash anchoring authenticates bytes only when the caller trusts its source.
    No solver result or physical acceptance is invented from a synthetic view.
    """
    try:
        b=_binding(graph,binding);v=_decode(frame_view_bytes,expected_view_sha256)
        _require(type(v) is dict and v.get('schema') in _FRAME_SCHEMAS,'Expected saved L3 frame view')
        _require(v['provenance']['binding_sha256']==binding.sha256,'View/binding identity mismatch')
        payload=_payload(v,b)
        record={'schema':SCHEMA,'source_view_sha256':_hash(expected_view_sha256),'binding_sha256':binding.sha256,
            'graph_sha256':b['record']['graph_sha256'],'frame_data':payload,**POLICIES}
        _bounded(record);raw=_raw(record);_require(len(raw)<=MAX_BYTES,'Projection byte ceiling exceeded')
        return L3ResultProjection(raw,_sha(raw))
    except L3ProjectionError:
        raise
    except L3InputError as exc:
        raise L3ProjectionError(str(exc)) from exc
    except (KeyError,TypeError,IndexError,OverflowError,RecursionError,UnicodeError,AttributeError) as exc:raise L3ProjectionError('Malformed L3 projection input') from exc


def load_l3_projection(graph, binding, *, projection_bytes, expected_projection_sha256):
    """Load this explicit result family, never a schema-less mechanical SimResult."""
    try:
        b=_binding(graph,binding);record=_decode(projection_bytes,expected_projection_sha256)
        _require(type(record) is dict and record.get('schema')==SCHEMA,'Unsupported projection schema')
        _require(record['binding_sha256']==binding.sha256 and record['graph_sha256']==b['record']['graph_sha256'],'Projection graph/binding identity mismatch')
        _hash(record['source_view_sha256']);payload=_payload(record['frame_data'],b)
        _require(_raw(record['frame_data'])==_raw(payload),'Unexpected exported frame metadata')
        expected_keys={'schema','source_view_sha256','binding_sha256','graph_sha256','projection','frame_data','omitted_native_fields','rotation_policy','force_policy'}
        _require(set(record)==expected_keys,'Unexpected projection export metadata')
        for key,value in POLICIES.items():_require(_raw(record[key])==_raw(value),'Unsupported projection policy')
        return L3ResultProjection(projection_bytes,_sha(projection_bytes))
    except L3ProjectionError:
        raise
    except L3InputError as exc:
        raise L3ProjectionError(str(exc)) from exc
    except (KeyError,TypeError,IndexError,OverflowError,RecursionError,UnicodeError,AttributeError) as exc:raise L3ProjectionError('Malformed L3 projection export') from exc


@dataclass(frozen=True)
class L3RenderInputs:
    """Detached visualization copies only; original remains authoritative."""
    original: object
    deformed: object
    displacement_data: object
    node_ids: tuple
    active_edge_ids: tuple
    axial_force_N: object
    coordinate_unit: str

    def deformation_kwargs(self):
        # The existing renderer reduces color_data and cannot color empty edges.
        return dict(original=self.original,deformed=self.deformed,
            displacement_data=self.displacement_data if self.active_edge_ids else None,
            node_ids=list(self.node_ids),title='Physical reference and active saved L3 frame ('+self.coordinate_unit+')')

    def force_kwargs(self):
        return dict(graph=self.deformed,color_by='custom' if self.active_edge_ids else 'uniform',
            color_data=self.axial_force_N if self.active_edge_ids else None,
            title='Saved active-edge axial force (N)')


def l3_render_inputs(graph, binding, projection):
    """Use only explicit-array 2D renderer seams, not generic SimResult metadata.

    Copies show the full physical reference and active deformed edges. Angular
    and force fields are not inserted into renderer metadata with mutable axes.
    Translation conversion is only for display in the bound graph's units.
    """
    _require(type(projection) is L3ResultProjection,'L3ResultProjection required')
    checked=load_l3_projection(graph,binding,projection_bytes=projection.canonical,expected_projection_sha256=projection.sha256)
    data=checked.to_dict()['frame_data'];scale=data['original_units']['coordinate_scale_to_m']
    import numpy as np
    original=graph.copy();deformed=graph.copy()
    positions={}
    for nid,position in zip(data['node_ids'],data['fields']['positions_m']):
        converted=np.array([position[0]/scale,position[1]/scale,0.0],dtype=float)
        _require(np.isfinite(converted).all(),'Display unit conversion overflow')
        positions[nid]=converted
    # Public batch update preserves the detached graph's spatial lookup index.
    deformed.set_node_positions(positions)
    active=set(data['active_edge_ids'])
    for eid in list(deformed.edges):
        if eid not in active:deformed.remove_edge(eid)
    order=list(deformed.edges);row={eid:i for i,eid in enumerate(data['edge_ids'])}
    displacement=np.asarray(data['fields']['displacements_m'],dtype=float)/scale
    _require(np.isfinite(displacement).all(),'Display displacement conversion overflow')
    displacement=np.frombuffer(displacement.tobytes(),dtype=float).reshape(displacement.shape)
    force=np.asarray([data['fields']['axial_force_N'][row[eid]] for eid in order],dtype=float)
    force=np.frombuffer(force.tobytes(),dtype=float)
    return L3RenderInputs(original,deformed,displacement,tuple(data['node_ids']),tuple(order),force,data['original_units']['coordinate_unit'])

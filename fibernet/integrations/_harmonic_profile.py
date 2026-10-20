"""Private compilation bounds, never permission to execute native software."""
import json
import math
from numbers import Real
from types import MappingProxyType

NAME = 'harmonic_large_v1'
PROFILE = MappingProxyType(dict(name=NAME, max_nodes=1024, max_edges=6144,
    steps=32, sample_every=1, frames=33, max_mass_types=2, max_loaded_nodes=20,
    prepared_bytes=8*1024*1024, raw_bytes=16*1024*1024,
    export_bytes=40*1024*1024))

SAMPLED_NAME = 'harmonic_sampled_v1'
SAMPLED_PROFILE = MappingProxyType(dict(name=SAMPLED_NAME, max_nodes=256,
    max_edges=1536, max_steps=1024, sample_every=1, max_output_frames=129,
    max_mass_types=2, max_loaded_nodes=20, prepared_bytes=8*1024*1024,
    raw_bytes=64*1024*1024, export_bytes=40*1024*1024))

def sampled(p):
    return p is not None and p['name'] == SAMPLED_NAME

def schedule(p, steps, every, stride=None):
    if sampled(p):
        if type(steps) is not int or not 1 <= steps <= p['max_steps'] or type(every) is not int or every != 1:
            raise ValueError('harmonic_sampled_v1 requires 1..1024 steps and sample_every=1')
        if type(stride) is not int or stride < 1 or steps % stride or steps//stride+1 > p['max_output_frames']:
            raise ValueError('output_stride must be a positive integer dividing steps with at most 129 output frames')
        return steps//stride+1
    if stride is not None:
        raise ValueError('output_stride requires harmonic_sampled_v1')
    if p and (type(steps) is not int or steps != 32 or type(every) is not int or every != 1):
        raise ValueError('harmonic_large_v1 requires exactly 32 steps and 33 frames')
    return steps+1

def frame_counts(c):
    p=contract_profile(c)
    output=schedule(p,c['steps'],c['sample_every'],c.get('output_stride'))
    if sampled(p) and (type(c.get('native_frames')) is not int or c['native_frames'] != c['steps']+1 or
                       type(c.get('output_frames')) is not int or c['output_frames'] != output):
        raise ValueError('native/output frame counts mismatch')
    return c['steps']+1,output

def select(value):
    if value is None: return None
    if type(value) is not str or value not in (NAME,SAMPLED_NAME): raise ValueError('unknown harmonic profile')
    return dict(PROFILE if value == NAME else SAMPLED_PROFILE)

def record(value):
    # JSON equality is type-sensitive here: True must not impersonate 1.
    expected = select(value.get('name')) if type(value) is dict else None
    if expected is None or set(value) != set(expected) or any(
        type(value[k]) is not type(v) or value[k] != v for k,v in expected.items()):
        raise ValueError('missing, malformed or tampered harmonic profile')
    return expected

def contract_profile(contract):
    if type(contract) is not dict:raise ValueError('harmonic model contract must be a dict')
    if contract.get('model')=='conservative-ball-cosine-strand-cg/1' and 'profile' in contract:
        from ._cg_network_profile import contract as check_network
        return check_network(contract)
    return record(contract['profile']) if 'profile' in contract else None

def counts(n,e,p):
    if type(n) is not int or type(e) is not int or not 2<=n<=(p['max_nodes'] if p else 256) or not 1<=e<=(p['max_edges'] if p else 1024):
        raise ValueError('harmonic profile node/edge counts exceeded or invalid')

def string_bytes(strings, limit):
    total=0
    for s in strings:
        if type(s) is not str or len(s)>limit-total: raise ValueError('harmonic byte budget exceeded')
        # Count Unicode without making a second whole UTF-8 string.
        for i in range(0,len(s),8192):
            total+=len(s[i:i+8192].encode('utf-8'))
            if total>limit: raise ValueError('harmonic byte budget exceeded')
    return total

def json_bytes(value, limit, *, indent=None):
    # Prewalk limits depth, leaves, and strings before encoder can allocate an
    # escaped giant string. Walk containers without copying them.
    budget=[limit]
    def walk(x,depth=0):
        if depth>64: raise ValueError('harmonic JSON nesting exceeds bound')
        budget[0]-=1
        if budget[0]<0: raise ValueError('harmonic JSON budget exceeded')
        if isinstance(x,str):
            budget[0]-=len(x)
        elif isinstance(x,dict):
            for k,v in x.items(): walk(k,depth+1);walk(v,depth+1)
        elif isinstance(x,(list,tuple)):
            if len(x)>budget[0]: raise ValueError('harmonic JSON budget exceeded')
            for v in x:walk(v,depth+1)
        if budget[0]<0: raise ValueError('harmonic JSON budget exceeded')
    walk(value)
    total=0
    for chunk in json.JSONEncoder(allow_nan=False,indent=indent).iterencode(value):
        total+=len(chunk.encode('utf-8'))
        if total>limit:raise ValueError('harmonic JSON byte budget exceeded')
    return total

def pairs(value,label,expected=None):
    if not isinstance(value,(list,tuple)) or (expected is not None and len(value)!=expected):
        raise ValueError(label+' count mismatch')
    seen=set()
    for row in value:
        if not isinstance(row,(list,tuple)) or len(row)!=2 or type(row[0]) is not int or row[0]<0 or row[0] in seen:
            raise ValueError(label+' duplicate or invalid IDs')
        seen.add(row[0])
    return seen

def check_contract(c,n,e):
    p=contract_profile(c);counts(n,e,p)
    if p:
        frame_counts(c)
        if sampled(p) and (type(c.get('timestep_s')) not in (int,float) or not math.isfinite(c['timestep_s']*c['steps']) or c['timestep_s']*c['steps'] <= 0):
            raise ValueError('sampled duration must be positive finite')
        pairs(c.get('masses_kg'),'masses',n);pairs(c.get('bonds'),'bonds',e)
        for key in ('forces_N','velocities_m_s','atom_to_type_id','type_to_mass_kg'):pairs(c.get(key),key)
        if len(c['forces_N'])>20 or not 1<=len(c['type_to_mass_kg'])<=2:raise ValueError('harmonic profile mass/load bound exceeded')
        masses=[]
        for _,v in c['masses_kg']:
            if type(v) not in (int,float) or not 0<float(v)<float('inf'):raise ValueError('invalid profile mass')
            masses.append(float(v).hex())
        if len(set(masses))>2:raise ValueError('harmonic profile exact-mass bound exceeded')
        node_ids=[row[0] for row in c['masses_kg']]
        edge_ids=[row[0] for row in c['bonds']]
        if node_ids!=sorted(node_ids) or edge_ids!=sorted(edge_ids):raise ValueError('profile contract IDs must be sorted')
        if pairs(c['atom_to_type_id'],'atom types',n)!=set(range(1,n+1)):raise ValueError('profile atom type coverage mismatch')
        expected_masses=sorted({float(v) for _,v in c['masses_kg']})
        expected_types=[[i+1,v] for i,v in enumerate(expected_masses)]
        if len(c['type_to_mass_kg'])!=len(expected_types) or any(type(row[1]) not in (float,int) or isinstance(row[1],bool) for row in c['type_to_mass_kg']) or list(map(list,c['type_to_mass_kg']))!=expected_types:
            raise ValueError('profile type masses mismatch')
        expected_atoms=[[i+1,expected_masses.index(float(v))+1] for i,(_,v) in enumerate(c['masses_kg'])]
        if any(type(row[1]) is not int for row in c['atom_to_type_id']) or list(map(list,c['atom_to_type_id']))!=expected_atoms:raise ValueError('profile atom mass mapping mismatch')
        fixed=c.get('fixed_node_ids')
        if not isinstance(fixed,(list,tuple)) or any(type(v)is not int for v in fixed) or len(set(fixed))!=len(fixed) or not set(fixed)<set(node_ids):raise ValueError('profile fixed IDs invalid')
        for key in ('forces_N','velocities_m_s'):
            if not pairs(c[key],key)<=set(node_ids):raise ValueError('profile load/velocity IDs unknown')
            for _,v in c[key]:shape(v,(3,),key)
        if pairs(c['forces_N'],'forces') & set(fixed):raise ValueError('fixed node cannot carry force')
        if any(k in fixed and any(v) for k,v in c['velocities_m_s']):raise ValueError('fixed node cannot carry velocity')
        for _,coeff in c['bonds']:
            shape(coeff,(2,),'bond coefficients')
            if min(coeff)<=0:raise ValueError('profile bond coefficients must be positive')
    return p

def shape(value,dimensions,label):
    def scalar(x):
        if isinstance(x,bool) or not isinstance(x,Real) or not math.isfinite(float(x)):
            raise ValueError(label+' requires finite real scalar leaves')
    if hasattr(value,'shape'):
        if tuple(value.shape)!=tuple(dimensions):raise ValueError(label+' shape mismatch')
        for x in value.flat:scalar(x)
        return
    if not isinstance(value,(list,tuple)) or len(value)!=dimensions[0]:raise ValueError(label+' shape mismatch')
    if len(dimensions)>1:
        for row in value:shape(row,dimensions[1:],label)
    else:
        for x in value:scalar(x)

def result_guard(metadata,n,e,state=None,arrays=None):
    c=metadata.get('model_contract',{})
    if c.get('model') == 'conservative-ball-cosine-strand-cg/1':
        from ._lammps_cg import guard_result
        return guard_result(metadata,n,e,state,arrays)
    if state is not None and state.get('schema') == 'lammps_cg_state/1':
        raise ValueError('CG state/model mismatch')
    p=check_contract(c,n,e)
    if ('profile' in metadata)!=bool(p) or (p and record(metadata['profile'])!=p):raise ValueError('result/profile mismatch')
    if state is not None and ('profile' in state)!=bool(p):raise ValueError('state/profile presence mismatch')
    if not p:return None
    native_count,output_count=frame_counts(c)
    identity=metadata.get('source_identity')
    if type(identity) is not dict or set(identity)!={'prepared_sha256','graph_sha256'} or any(type(v)is not str or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in identity.values()):
        raise ValueError('invalid profile source identity')
    array_keys={'dynamics_state','positions_trajectory','displacements','deformed_positions','edge_forces','edge_stretches','history'}
    json_bytes({k:v for k,v in metadata.items() if k not in array_keys},PROFILE['prepared_bytes'],indent=2 if sampled(p) else None)
    for key,count in [('node_ids',n),('edge_ids',e)]:
        ids=metadata.get(key)
        if not isinstance(ids,(list,tuple)) or len(ids)!=count or any(type(x)is not int or not 0<=x<2**31 for x in ids) or len(set(ids))!=count:
            raise ValueError('result stable IDs invalid')
    if list(metadata['node_ids'])!=[row[0] for row in c['masses_kg']] or list(metadata['edge_ids'])!=[row[0] for row in c['bonds']]:raise ValueError('result model ID order mismatch')
    if state is not None:
        if record(state.get('profile'))!=p:raise ValueError('state/profile mismatch')
        if state.get('final',{}).get('step')!=c['steps'] or type(state.get('final',{}).get('step')) is not int:raise ValueError('profile final step mismatch')
        if sampled(p) and any(state.get(k)!=c[k] or type(state.get(k)) is not int for k in ('output_stride','native_frames','output_frames')):
            raise ValueError('state/model sampled schedule mismatch')
        if state.get('representation')=='sampled':
            for key in ('steps','times','potential_energy','kinetic_energy','total_energy'):shape(state.get(key),(output_count,),key)
            for key in ('positions','velocities','nodal_forces'):shape(state.get(key),(output_count,n,3),key)
            shape(state.get('pressure_tensor'),(output_count,6),'pressure_tensor')
            shape(state.get('node_ids'),(n,),'node_ids')
        json_bytes(state,PROFILE['export_bytes'])
    if arrays:
        for key,dims in [('displacements',(n,3)),('deformed_positions',(n,3)),('edge_forces',(e,)),('edge_stretches',(e,))]:
            if arrays.get(key) is not None:shape(arrays[key],dims,key)
        trajectory=arrays.get('positions_trajectory')
        if trajectory is not None and len(trajectory):shape(trajectory,(output_count,n,3),'positions_trajectory')
        if len(arrays.get('history',[]))>output_count:raise ValueError('profile history exceeds frame bound')
    return p


def resource_forecast(n, e, steps, stride, prepared_bytes, setup_bytes):
    """Conservative planned payload sizes, not a bound on native diagnostics.

    %.17g finite binary64 tokens need at most 24 ASCII bytes. Atom rows
    allocate 233 bytes (nine tokens, two IDs and separators); frame headers
    allocate 1024 and thermo rows 256. A separate 1 MiB diagnostic reserve
    is an estimate; aggregate and per-file hard limits remain authoritative.
    JSON scalar slots allocate 64 bytes for a <=24-byte scalar, separators,
    and schema indentation. Container/key overhead has a separate 1 MiB
    allowance; bounded metadata has 8 MiB. This includes duplicated geometry.
    RAM is conservative accounting of simultaneous raw text/split lines,
    Python parsed state, selected lists/arrays and serialization copies;
    it is not a streaming-memory guarantee or a general interpreter bound.
    """
    counts(n,e,SAMPLED_PROFILE)
    out=schedule(SAMPLED_PROFILE,steps,1,stride)
    native=steps+1
    if type(prepared_bytes) is not int or not 0 <= prepared_bytes <= SAMPLED_PROFILE['prepared_bytes']:
        raise ValueError('prepared byte forecast invalid')
    dump=native*(233*n+1024)+32
    thermo=native*256
    if type(setup_bytes) is not int or not 0 <= setup_bytes <= SAMPLED_PROFILE['raw_bytes']:
        raise ValueError('exact setup artifact byte count invalid')
    setup=setup_bytes
    diagnostic_reserve=1024*1024
    aggregate=dump+2*thermo+setup+diagnostic_reserve
    # Detailed state, repeated final state, trajectory, final geometry, edge
    # arrays, times/energies/pressures, IDs, scalar summary and bounded history.
    slots=12*n*out+16*n+2*e+16*out+64
    export=64*slots+SAMPLED_PROFILE['prepared_bytes']+1024*1024
    raw_parse=4*SAMPLED_PROFILE['raw_bytes']
    parsed=native*(9*n+16)*64
    selected=4*slots*64
    serialization=3*SAMPLED_PROFILE['export_bytes']
    memory=raw_parse+parsed+selected+serialization+4*prepared_bytes
    return dict(native_frames=native,output_frames=out,node_frames=n*native,
        trajectory_bytes=dump,thermo_bytes=thermo,prepared_artifact_bytes=setup,
        diagnostic_reserve_bytes=diagnostic_reserve,native_aggregate_forecast_bytes=aggregate,
        detailed_export_forecast_bytes=export,raw_parse_memory_bytes=raw_parse,
        parsed_state_memory_bytes=parsed,selected_copy_memory_bytes=selected,
        serialization_memory_bytes=serialization,planned_memory_bytes=memory)


def export_metadata(prepared, runtime):
    """One exact metadata builder shared by admission planning and the adapter."""
    import hashlib
    c=json.loads(prepared.contract_json)
    pins=dict(runtime.file_sha256)
    unit=c['length_unit']
    value={'backend':'lammps','node_ids':list(prepared.node_ids),'edge_ids':list(prepared.edge_ids),
        'source_identity':{'prepared_sha256':prepared.identity,'graph_sha256':hashlib.sha256(prepared.graph_json.encode()).hexdigest()},
        'model_contract':c,
        'runtime_receipt':{'mode':runtime.mode,'expected_version':runtime.expected_version,
            'expected_banner':runtime.expected_banner,'python_sha256':pins.get(runtime.python_executable),
            'lammps_sha256':pins.get(runtime.lammps_executable),'inventory_sha256':sorted(pins.values())},
        'field_units':{'displacements':unit,'deformed_positions':unit,'positions_trajectory':unit,
            'max_displacement':unit,'edge_forces':'N','edge_stretches':'1','energy':'J','time_seconds':'s wall time'},
        'original_graph_length_unit':unit,'original_graph_length_to_m':c['length_to_m'],
        'edge_force_origin':'derived k*(length-equilibrium_length); tension-positive',
        'stress':None,'reactions':None,'work':None}
    profile=contract_profile(c)
    if profile:value['profile']=profile
    return value

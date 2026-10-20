"""Explicit source capacity bounds, not native or trajectory qualification."""
from types import MappingProxyType
import math
NAME = 'cg_network_sampled_v1'
MIXED_NAME = 'cg_network_mixed_sampled_v1'
NAMES = (NAME,MIXED_NAME)
SCOPE = 'sampled states only; no all-step geometry validation'
PROFILE = MappingProxyType(dict(name=NAME,max_nodes=1024,max_edges=6144,max_angles=8192,
    max_pairs=1023,max_steps=1600000,max_output_frames=129,max_thermo_frames=8193,
    max_node_frames=132096,prepared_bytes=4*1024**2,raw_bytes=64*1024**2,
    export_bytes=160*1024**2,max_cpu_seconds=60,max_wall_seconds=120,
    max_memory_bytes=2*1024**3,observation_scope=SCOPE,
    qualification='input bounds only; network native performance and physics unqualified',
    physical_scope='attractive selected sites only; radial repulsion absent'))
MIXED_PROFILE = MappingProxyType(dict(PROFILE,name=MIXED_NAME,
    physical_scope='explicit attractive and repulsive nodal sites; finite penetrable core; no continuous fiber contact'))
def select(value):
    if value is None:return None
    if type(value)is not str or value not in NAMES:raise ValueError('unknown CG network profile')
    return dict(MIXED_PROFILE if value==MIXED_NAME else PROFILE)
def record(value):
    if type(value)is not dict:raise ValueError('tampered CG network profile')
    expected=select(value.get('name'))
    if expected is None or set(value)!=set(expected) or any(type(value[k])is not type(v) or value[k]!=v for k,v in expected.items()):
        raise ValueError('tampered CG network profile')
    return expected
def schedule(steps,every,thermo_every):
    if type(steps)is not int or not 0<=steps<=PROFILE['max_steps']:raise ValueError('CG network step bound exceeded')
    for x in (every,thermo_every):
        if type(x)is not int or not 1<=x<=max(1,steps) or steps%x:raise ValueError('network strides must divide steps')
    if every%thermo_every:raise ValueError('retained states must coincide with thermo observations')
    out,thermo=steps//every+1,steps//thermo_every+1
    if out>PROFILE['max_output_frames'] or thermo>PROFILE['max_thermo_frames']:raise ValueError('network observation count exceeded')
    return out,thermo
def contract(c):
    if 'profile' not in c:return None
    p=record(c['profile'])
    if c.get('pair_encoding',{}).get('name')!='static-groups-zero-cutoff/1':raise ValueError('network requires reviewed group encoding')
    out,thermo=schedule(c['steps'],c['sample_every'],c.get('thermo_every'))
    for k,v in dict(output_frames=out,thermo_frames=thermo,integration_steps=c['steps']).items():
        if type(c.get(k))is not int or c[k]!=v:raise ValueError('network schedule declaration mismatch')
    if c.get('observation_scope')!=SCOPE:raise ValueError('network observation scope mismatch')
    if len(c['masses_kg'])*out>p['max_node_frames']:raise ValueError('network node-frame bound exceeded')
    return p
def forecast(n,e,steps,every,thermo_every,prepared_bytes,setup_bytes):
    out,thermo=schedule(steps,every,thermo_every)
    if type(n)is not int or not 2<=n<=PROFILE['max_nodes'] or type(e)is not int or not 0<=e<=PROFILE['max_edges'] or n*out>PROFILE['max_node_frames']:raise ValueError('network forecast counts invalid')
    for size,cap in ((prepared_bytes,PROFILE['prepared_bytes']),(setup_bytes,PROFILE['raw_bytes'])):
        if type(size)is not int or not 0<=size<=cap:raise ValueError('network forecast bytes invalid')
    dump=out*(239*n+1024)+32;log=thermo*256
    slots=12*n*out+16*n+2*e+16*out+64
    export=64*slots+PROFILE['prepared_bytes']+1024**2
    memory=4*PROFILE['raw_bytes']+out*(9*n+16)*64+4*slots*64+3*PROFILE['export_bytes']+4*prepared_bytes+68*(n+1)**2
    return dict(integration_steps=steps,output_frames=out,thermo_frames=thermo,node_frames=n*out,
        trajectory_bytes=dump,thermo_bytes=log,native_aggregate_forecast_bytes=dump+2*log+2*setup_bytes+1024**2,
        detailed_export_forecast_bytes=export,planned_memory_bytes=memory,
        native_morse_matrix_payload_lower_bound_bytes=68*(n+1)**2,observation_scope=SCOPE)

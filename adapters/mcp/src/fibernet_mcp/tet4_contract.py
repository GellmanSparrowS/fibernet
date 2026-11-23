"""Private protocol adapter: validated JSON, one canonical native solve."""
from hashlib import sha256
import math
import re
from fibernet_mcp.safety import ContractError, canonical, strict_json

MAX_SOURCE=2*1024**2
MAX_REQUEST=512*1024
MAX_ARTIFACT=4*1024**2
KEYS={'mesh_id','conductivity_S_per_m_by_material','electrode_voltage_V_by_id',
      'conductivity_W_per_m_K_by_material','prescribed_K_by_id',
      'faces_by_node_id','convection_W_per_m2_K','ambient_K'}

def require(ok,text):
    if not ok: raise ContractError(text)

def finite(v,lo,hi,scope=None):
    require(type(v) in (int,float) and math.isfinite(v) and lo<=v<=hi,'invalid finite SI value' if scope is None else f'invalid finite SI value for {scope}')
    return float(v)

def ident(v):
    require(isinstance(v,str) and re.fullmatch(r'0|[1-9][0-9]{0,19}',v) is not None,'invalid decimal ID')
    n=int(v)
    require(n<=2**64-1,'invalid uint64 ID')
    return n

def ids(items,n,unique):
    require(isinstance(items,list) and len(items)==n,'wrong ID count')
    asint=[ident(v) for v in items]
    require(not unique or len(set(asint))==n,'duplicate ID')
    return asint

def parse_mesh(raw,sha):
    require(len(raw)<=MAX_SOURCE and sha256(raw).hexdigest()==sha,'mesh bytes or digest mismatch')
    m=strict_json(raw)
    require(isinstance(m,dict) and set(m)=={'schema','unit','points','tetrahedra','node_ids','cell_ids','material_ids','materials'},'invalid mesh fields')
    require(m['schema']=='fibernet.tet4_mesh.v1' and m['unit'] in ('m','cm','mm','um','nm'),'invalid schema/length unit')
    p,t,properties=m['points'],m['tetrahedra'],m['materials']
    require(isinstance(p,list) and 4<=len(p)<=1024 and isinstance(t,list) and 1<=len(t)<=8000,'mesh size limit')
    n,c=ids(m['node_ids'],len(p),True),ids(m['cell_ids'],len(t),True)
    tags=ids(m['material_ids'],len(t),False)
    require(isinstance(properties,dict) and 1<=len(properties)<=128,'invalid material table')
    mat={}
    for k,q in properties.items():
        require(isinstance(q,dict) and set(q)=={'young_modulus_pa','poisson_ratio'},'invalid material fields')
        mat[ident(k)]={'young_modulus_pa':finite(q['young_modulus_pa'],1e3,1e13),'poisson_ratio':finite(q['poisson_ratio'],-.5,.45)}
    require(set(tags)==set(mat),'unassigned or missing mesh material')
    for row in p:
        require(isinstance(row,list) and len(row)==3,'invalid point shape')
        for v in row: finite(v,-1e9,1e9)
    for row in t:
        require(isinstance(row,list) and len(row)==4 and all(type(i) is int and 0<=i<len(p) for i in row) and len(set(row))==4,'invalid Tet4 row')
    require(len(set(tuple(sorted(row)) for row in t))==len(t),'duplicate Tet4')
    return p,t,n,c,tags,mat,m['unit']

def parse_req(raw,mesh,expected_handle):
    require(len(raw)<=MAX_REQUEST,'request byte budget exceeded')
    a=strict_json(raw)
    require(isinstance(a,dict) and set(a)==KEYS,'invalid or unsupported solve controls')
    require(a.pop('mesh_id')==expected_handle,'unknown solid mesh handle')
    _,_,n,_,_,mat,_=mesh
    for field,allowed,complete,lo,hi in (
        ('electrode_voltage_V_by_id',n,False,-100,100),
        ('prescribed_K_by_id',n,False,1,2000),
        ('conductivity_S_per_m_by_material',mat,True,1e-6,1e6),
        ('conductivity_W_per_m_K_by_material',mat,True,1e-6,1e6)):
        q=a[field]
        require(isinstance(q,dict) and 1<=len(q)<=len(allowed),'invalid control map length')
        row={ident(k):finite(v,lo,hi,field) for k,v in q.items()}
        require(set(row)<=set(allowed) and (not complete or set(row)==set(allowed)),'unknown or missing control IDs')
        a[field]=row
    faces=a['faces_by_node_id']
    require(isinstance(faces,list) and 1<=len(faces)<=4096,'invalid convection-face count')
    asint=[ids(f,3,True) for f in faces]
    allowed=set(n)
    require(all(set(f)<=allowed for f in asint),'unknown face node')
    require(len({tuple(sorted(f)) for f in asint})==len(asint),'duplicate convection face')
    a['faces_by_node_id']=asint
    coeff=finite(a['convection_W_per_m2_K'],0,1e8,'convection_W_per_m2_K')
    require(coeff==0 or coeff>=1e-8,'unsupported convection coefficient')
    a['convection_W_per_m2_K']=coeff
    a['ambient_K']=finite(a['ambient_K'],1,2000,'ambient_K')
    return a

def native(mesh,args,source_sha,controls_sha):
    from fibernet_mcp.safety import _verify_core_result
    provenance=_verify_core_result().provenance()
    from fibernet.io.tet4_meshio import Tet4Bundle
    from fibernet.sim.tet4_dc_thermal import solve_dc_thermal_tet4_with_convection
    import numpy as np
    p,t,n,c,tags,mat,unit=mesh
    own=Tet4Bundle(np.asarray(p,dtype=np.float64),np.asarray(t,dtype=np.int64),
                   np.asarray(n,dtype=np.uint64),np.asarray(c,dtype=np.uint64),
                   np.asarray(tags,dtype=np.uint64),mat,unit)
    result=solve_dc_thermal_tet4_with_convection(own,**args)
    e,h=result.electrical,result.thermal
    require(np.array_equal(e.node_ids,h.node_ids) and np.array_equal(e.cell_ids,h.cell_ids)
            and np.array_equal(e.joule_power_W_by_cell,h.integrated_source_W_by_cell),'invalid native handoff')
    fields={'voltage_V':e.voltage_V.tolist(),'temperature_K':h.temperature_K.tolist(),
            'joule_power_W_by_cell':e.joule_power_W_by_cell.tolist(),
            'convective_heat_into_solid_W_by_node':h.convective_heat_into_solid_W_by_node.tolist(),
            'prescribed_heat_into_solid_W_by_node':h.heat_into_solid_W.tolist()}
    out=dict(kind='tet4_dc_convection_result',source_sha256=source_sha,controls_sha256=controls_sha,
        provenance=provenance,node_ids=list(map(str,e.node_ids)),cell_ids=list(map(str,e.cell_ids)),fields=fields,
        summary={'total_joule_power_W':e.total_joule_power_W,'total_convective_heat_into_solid_W':h.total_convective_heat_into_solid_W,
                 'total_prescribed_heat_into_solid_W':math.fsum(h.heat_into_solid_W),'balance_W':h.balance_W})
    raw=canonical(out)
    require(len(raw)<=MAX_ARTIFACT,'result budget exceeded')
    return raw

def validate_result(v,mesh,source_sha,controls_sha,provenance):
    keys={'kind','source_sha256','controls_sha256','provenance','node_ids','cell_ids','fields','summary'}
    require(isinstance(v,dict) and set(v)==keys,'bad worker result fields')
    require(v['kind']=='tet4_dc_convection_result' and v['source_sha256']==source_sha and
            v['controls_sha256']==controls_sha and v['provenance']==provenance,'bad worker result identity')
    _,_,nodes,cells,*_=mesh
    require(v['node_ids']==list(map(str,nodes)) and v['cell_ids']==list(map(str,cells)),'bad worker result ID order')
    f=v['fields']
    counts={'voltage_V':len(nodes),'temperature_K':len(nodes),'joule_power_W_by_cell':len(cells),
            'convective_heat_into_solid_W_by_node':len(nodes),'prescribed_heat_into_solid_W_by_node':len(nodes)}
    require(isinstance(f,dict) and set(f)==set(counts),'bad worker field names')
    for k,n in counts.items():
        require(isinstance(f[k],list) and len(f[k])==n and all(type(a) in (float,int) and math.isfinite(a) for a in f[k]),'bad worker field shape/value')
    summary=v['summary']
    require(isinstance(summary,dict) and set(summary)=={'balance_W','total_convective_heat_into_solid_W',
        'total_joule_power_W','total_prescribed_heat_into_solid_W'} and
        all(type(a) in (float,int) and math.isfinite(a) for a in summary.values()),'bad worker summary')
    require(all(1<=x<=2000 for x in f['temperature_K']),'bad worker temperature range')
    checked_powers(f,summary)

# Whitelist just payload number values; do not rewrite IDs, indices,
# method names, unknown members, bools, strings, or transport metadata.
TET4_TOOL='fibernet_solve_tet4_dc_convection'
CONTROL_SCALARS={'ambient_K','convection_W_per_m2_K'}
CONTROL_MAPS={'conductivity_S_per_m_by_material','electrode_voltage_V_by_id',
              'conductivity_W_per_m_K_by_material','prescribed_K_by_id'}
FAIL_KIND='tet4_dc_convection_error'
MESSAGES={'SOURCE_ENVELOPE':'Joule density outside thermal source envelope',
          'NATIVE_INPUT_REJECTED':'native input rejected'}

def normalize_tet4_rpc(message):
    if isinstance(message,dict) and message.get('method')=='tools/call':
        p=message.get('params')
        a=p.get('arguments') if isinstance(p,dict) and p.get('name')==TET4_TOOL else None
        if isinstance(a,dict):
            def si(v):
                if type(v) not in (int,float): return v
                try: f=float(v)
                except OverflowError: return None
                return f if math.isfinite(f) else None
            for k in CONTROL_SCALARS & a.keys():
                a[k]=si(a[k])
            for k in CONTROL_MAPS & a.keys():
                if isinstance(a[k],dict):
                    a[k]={id_:si(v) for id_,v in a[k].items()}

def check_error(v,source_sha,controls_sha,provenance):
    require(isinstance(v,dict) and set(v)=={'kind','source_sha256','controls_sha256','provenance','code'},'bad worker error envelope')
    require(v['kind']==FAIL_KIND and v['source_sha256']==source_sha and v['controls_sha256']==controls_sha
            and v['provenance']==provenance,'bad worker error identity')
    code=v['code']
    require(type(code) is str and code in MESSAGES,'bad worker error code')
    return {'code':code,'error':MESSAGES[code]}

def checked_powers(f,s):
    import sys
    j=f['joule_power_W_by_cell']
    h=f['convective_heat_into_solid_W_by_node']
    c=f['prescribed_heat_into_solid_W_by_node']
    # A nonzero absolute floor is needed only for subnormal arithmetic, not
    # for application-scale powers. Native pilot cell volume <= 1 m^3 and
    # source density <= 1e8 W/m^3. The generous boundary cap (1e40 W) only
    # guards finite summation: it exceeds the native worst-case estimate
    # from 8000 cells, max k, max shape gradient and temperature bounds.
    require(all(0<=x<=1e8 for x in j),'bad Joule cell power')
    require(all(abs(x)<=1e40 for x in (*h,*c)),'bad boundary power')
    vals=[math.fsum(x) for x in (j,h,c)]
    scales=[math.fsum(map(abs,x)) for x in (j,h,c)]
    total=math.fsum(scales)
    roundoff=lambda scale: math.fsum((512*sys.float_info.epsilon*scale,math.ulp(0.0)*512))
    # Native published convection is accumulated per face, while the
    # transport fields have nodal integrals. Compare their equivalent
    # sums with a tolerance scaled to each ledger, not to unit-scale watts.
    for key,got,scale in zip(('total_joule_power_W','total_convective_heat_into_solid_W','total_prescribed_heat_into_solid_W'),vals,scales):
        require(abs(s[key]-got)<=roundoff(scale),'worker power summary does not match fields')
    balance=math.fsum((*j,*h,*c))
    require(abs(s['balance_W']-balance)<=roundoff(total),'worker balance summary does not match fields')
    require(abs(balance)<=max(math.ulp(0.0)*512,1e-9*total),'worker powers violate bounded heat balance')
    return vals,balance

TET4_KIND='tet4_dc_convection_result'
TET4_OP='tet4_dc_convection'
_FIXTURES={
    'simple':('tet4-simple-v1.json','9621fdca77a525d8c4c2dc86d7a261b4324b5e21640df17814b10c910738b349'),
    'welded':('tet4-welded-v1.json','e48b19822da4d52e9ddac27ccc9a32ccfa9a28799d1a53e0795e1114fb3fa8c4'),
}

def load_fixture(name):
    from pathlib import Path
    from fibernet_mcp.safety import read_file
    require(type(name) is str and name in _FIXTURES,'unknown bundled Tet4 fixture')
    filename,digest=_FIXTURES[name]
    raw=read_file(Path(__file__).with_name(filename),MAX_SOURCE)
    return raw,digest,parse_mesh(raw,digest)

def run_worker(raw,source_digest,request_path,request_digest,mesh_id,provenance):
    from fibernet_mcp.safety import read_file
    request=read_file(request_path,MAX_REQUEST)
    require(sha256(request).hexdigest()==request_digest,'unverified Tet4 controls')
    mesh=parse_mesh(raw,source_digest)
    a=parse_req(request,mesh,mesh_id)
    try:
        return strict_json(native(mesh,a,source_digest,request_digest))
    except ValueError as exc:
        code='SOURCE_ENVELOPE' if str(exc)==MESSAGES['SOURCE_ENVELOPE'] else 'NATIVE_INPUT_REJECTED'
        return dict(kind=FAIL_KIND,source_sha256=source_digest,controls_sha256=request_digest,
                    provenance=provenance,code=code)

def enable_tet4_tools(tools,*,operator_sources=False):
    """Augment schemas only in sessions that opted in to a bundled mesh."""
    import copy
    catalog=next(x for x in tools if x['name']=='fibernet_catalog')
    catalog_ok=catalog['outputSchema']['oneOf'][0]
    idref={'$ref':'#/$defs/Id'}
    href={'$ref':'#/$defs/Hash'}
    schema=lambda p:dict(type='object',properties=p,required=sorted(p),additionalProperties=False)
    props=catalog_ok['properties']
    props['capabilities']['const'].append('solve_tet4_dc_convection')
    mesh=schema(dict(mesh_id=idref,source_sha256=href,fixture={'enum':sorted(_FIXTURES)},
                    unit={'enum':['m','cm','mm','um','nm']},nodes={'type':'integer','minimum':4,'maximum':1024},
                    cells={'type':'integer','minimum':1,'maximum':8000}))
    if operator_sources:
        operator={k:v for k,v in mesh['properties'].items() if k!='fixture'}
        operator.update(source_kind={'const':'operator'},name={'type':'string','pattern':'^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$'})
        mesh={'oneOf':[mesh,schema(operator)]}
    props['tet4_meshes']=dict(type='array',items=mesh,minItems=1,maxItems=2)
    catalog_ok['required'].append('tet4_meshes')
    for tool in tools:
        kind=tool['outputSchema']['$defs']['Artifact']['properties']['kind']
        if 'enum' in kind: kind['enum'].append(TET4_KIND)
    exemplar=next(x for x in tools if x['name']=='fibernet_solve_linear_transport')
    uts={'type':'string','pattern':'^(0|[1-9][0-9]{0,19})$'}
    imap=lambda n:dict(type='object',propertyNames=uts,additionalProperties={'type':'number'},minProperties=1,maxProperties=n)
    number={'type':'number'}
    fs={'type':'array','items':{'type':'array','items':uts,'minItems':3,'maxItems':3},'minItems':1,'maxItems':4096}
    controls=dict(mesh_id=idref,conductivity_S_per_m_by_material=imap(128),
                  electrode_voltage_V_by_id=imap(1024),conductivity_W_per_m_K_by_material=imap(128),
                  prescribed_K_by_id=imap(1024),faces_by_node_id=fs,convection_W_per_m2_K=number,ambient_K=number)
    i=schema(controls);i['$defs']={'Id':copy.deepcopy(exemplar['inputSchema']['$defs']['Id'])}
    out_defs=copy.deepcopy(exemplar['outputSchema']['$defs'])
    summary=schema(dict(total_joule_power_W=number,total_convective_heat_into_solid_W=number,
                        total_prescribed_heat_into_solid_W=number,balance_W=number))
    o=dict(oneOf=[schema(dict(schema_version={'const':'fibernet.mcp/1'},mesh_id=idref,source_sha256=href,
                     controls_sha256=href,status={'const':'solved'},summary=summary,artifact={'$ref':'#/$defs/Artifact'})),
                  {'$ref':'#/$defs/Failure'}],
           **{'$schema':'https://json-schema.org/draft/2020-12/schema','$defs':out_defs})
    tools.append(dict(name=TET4_TOOL, description='Solve one-way steady DC heating with surface convection on a bundled Tet4 solid. Use stable decimal node IDs.',
                      inputSchema=i,outputSchema=o,annotations=copy.deepcopy(exemplar['annotations']),
                      execution=copy.deepcopy(exemplar['execution'])))

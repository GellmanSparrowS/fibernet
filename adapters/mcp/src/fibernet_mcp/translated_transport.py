"""Private composed native operation on explicitly bound bytes. Not an SDK tool."""
import re

from .safety import ContractError,canonical, graph_preflight, integer, keys, number, sha, strict_json, vector

OP = 'translated_transport_3d'
TOOL = 'fibernet_solve_translated_transport_3d'
KIND = 'translated_transport_3d_result'
SCHEMA = 'fibernet.translated_transport_3d/1'
PREPARATION_SCHEMA = 'fibernet.deformed_transport_preparation_3d/1'
POLICIES = {'geometry_policy':'endpoint_chord', 'area_policy':'fixed_effective_area',
            'conductivity_policy':'frozen_reference_sigma_A'}
FRAME_ERROR = 'TRANSLATED_TRANSPORT_FRAME_MISMATCH'
ERROR = 'TRANSLATED_TRANSPORT_INVALID'
FAILURES = {
    ERROR: 'contract',
    FRAME_ERROR: 'contract',
    'TRANSLATED_TRANSPORT_INVALID_GEOMETRY': 'domain',
    'TRANSLATED_TRANSPORT_FLOATING_COMPONENT': 'domain',
    'TRANSLATED_TRANSPORT_NUMERICAL_FAILURE': 'domain',
    'TRANSLATED_TRANSPORT_RESULT_LIMIT': 'resource',
}

class OperationError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)

LIMITS=dict(nodes=256,edges=1024,source=4*1024**2,request=256*1024,result=1024**2)
def stored_frame(graph_data, manifest_frame):
    # The graph has already passed the bounded JSON walk. Match the native
    # frame declaration scope, not arbitrary labels in application metadata.
    def frames(value, root=True, geometry=False):
        if type(value) is dict:
            for key, item in value.items():
                if key == 'coordinate_frame' and (root or geometry):
                    yield item
                yield from frames(item, False, key == 'geometry' and type(item) is dict)
        elif type(value) is list:
            for item in value:
                yield from frames(item, False, geometry)

    result = None
    for record in [graph_data, *graph_data['nodes'], *graph_data['edges']]:
        for label in frames(record.get('metadata', {})):
            if type(label) is not str or label != manifest_frame:
                raise OperationError(FRAME_ERROR)
            result = label
    return result

def reject(reason):
    raise ContractError(reason)
def exact_groups(items,tag,ids,type_check,complete):
    if type(items) is not list or len(items)>len(ids) or (complete and len(items)!=len(ids)) or not items:
        reject('list count')
    out={}
    for row in items:
        field='edge_id' if tag=='value_S' else 'node_id'
        keys(row,{field,tag},{field,tag})
        k=row[field];integer(k)
        if k not in ids or k in out: reject('unknown or duplicate ID')
        x=row[tag]; type_check(x)
        out[k]=x
    return out

def admit(source,request,*,source_digest,request_digest,manifest_unit,manifest_frame):
    if len(source)>LIMITS['source'] or len(request)>LIMITS['request']: reject('byte budget')
    if sha(source)!=source_digest or sha(request)!=request_digest:reject('content integrity')
    graph_data=graph_preflight(strict_json(source))
    source_node_ids={n['id'] for n in graph_data['nodes']}; source_edge_ids={e['id'] for e in graph_data['edges']}
    if not (0<len(source_node_ids)<=LIMITS['nodes'] and 0<len(source_edge_ids)<=LIMITS['edges']):
        reject('graph size budget')
    if graph_data['dimension']!=3:reject('dimension')
    q=strict_json(request)
    expected={'length_unit','translations_m','reference_conductance_S','dirichlet_voltage_V'}
    keys(q,expected,expected)
    if type(q['length_unit']) is not str or q['length_unit'] not in ('m','mm','um') or q['length_unit']!=manifest_unit:
        reject('unit')
    if manifest_frame in ('unknown','') or type(manifest_frame) is not str or len(manifest_frame)>64:reject('frame label')
    stored_frame(graph_data, manifest_frame)
    u=exact_groups(q['translations_m'],'value_m',source_node_ids,lambda x:vector(x,3),True)
    gs=exact_groups(q['reference_conductance_S'],'value_S',source_edge_ids,lambda x:number(x,positive=True),True)
    pin=exact_groups(q['dirichlet_voltage_V'],'value_V',source_node_ids,number,False)
    return graph_data, u, gs, pin

def candidate(source,request,*,source_digest,request_digest,manifest_unit,manifest_frame):
    graph_data, u, gs, pin = admit(source,request,source_digest=source_digest,request_digest=request_digest,
        manifest_unit=manifest_unit,manifest_frame=manifest_frame)
    # No scientific imports or graph construction until scoped wire admission.
    from fibernet.core.structure_graph import StructureGraph
    from fibernet.sim.deformed_transport import prepare_deformed_transport_3d
    from fibernet.sim.linear_transport import FloatingComponentError, TransportSolveError, solve_linear_transport
    try:
        original=StructureGraph.from_dict(graph_data)
        graph, conductance, binding=prepare_deformed_transport_3d(original,translations_m_by_id=u,
            reference_conductance_S=gs,length_unit=manifest_unit,max_nodes=LIMITS['nodes'],max_edges=LIMITS['edges'])
    except ValueError as exc:
        raise OperationError('TRANSLATED_TRANSPORT_INVALID_GEOMETRY') from exc
    if binding['coordinate_frame'] is not None and binding['coordinate_frame']!=manifest_frame:
        raise OperationError(FRAME_ERROR)
    try:
        result=solve_linear_transport(graph,conductance,pin,length_unit=manifest_unit)
    except FloatingComponentError as exc:
        raise OperationError('TRANSLATED_TRANSPORT_FLOATING_COMPONENT') from exc
    except (TransportSolveError, ValueError) as exc:
        raise OperationError('TRANSLATED_TRANSPORT_NUMERICAL_FAILURE') from exc
    encoded=canonical(dict(schema=SCHEMA,source_sha256=source_digest,
        request_sha256=request_digest,frame=manifest_frame,preparation=binding,transport=result.to_dict()))
    if len(encoded)>LIMITS['result']: raise OperationError('TRANSLATED_TRANSPORT_RESULT_LIMIT')
    strict_json(encoded)
    return encoded

def run_worker(source, expected_source_sha, request_path, expected_request_sha):
    from .safety import read_file
    raw = read_file(request_path, LIMITS['request'])
    if sha(raw) != expected_request_sha:
        raise ContractError('request integrity')
    bound = strict_json(raw)
    keys(bound, {'source_unit', 'source_frame', 'controls'}, {'source_unit', 'source_frame', 'controls'})
    controls = canonical(bound['controls'])
    try:
        result = strict_json(candidate(source, controls, source_digest=expected_source_sha,
            request_digest=sha(controls), manifest_unit=bound['source_unit'], manifest_frame=bound['source_frame']))
    except (OperationError,ContractError) as exc:
        code=exc.code if isinstance(exc,OperationError) else ERROR
        return {'source_sha256':expected_source_sha,'worker_request_sha256':expected_request_sha,
                'error_code':code,'error_category':FAILURES[code]}
    return {'source_sha256':expected_source_sha,'worker_request_sha256':expected_request_sha,
            'translated_transport':result}

def parent_request(graph_id, manifest, source, arguments):
    """Recheck operator-bound graph identity before any scientific child."""
    fields={'graph_id','frame_id','length_unit','translations_m','reference_conductance_S','dirichlet_voltage_V'}
    keys(arguments,fields,fields)
    if arguments['graph_id'] != graph_id or arguments['frame_id'] != manifest['frame_id']:
        reject('catalog graph or frame mismatch')
    controls={k:v for k,v in arguments.items() if k not in ('graph_id','frame_id')}
    control_bytes=canonical(controls)
    bound={'source_unit':manifest['length_unit'],'source_frame':manifest['frame_id'],'controls':controls}
    if len(canonical(bound))>LIMITS['request']: reject('worker request byte budget')
    admit(source,control_bytes,source_digest=manifest['source_sha256'],request_digest=sha(control_bytes),
        manifest_unit=bound['source_unit'],manifest_frame=bound['source_frame'])
    return bound

def parent_result(result, manifest, source, request, core_provenance):
    """Verify the worker binding before a result becomes a managed artifact."""
    source_sha = manifest['source_sha256']
    verified = {'source_sha256','worker_request_sha256','_core_provenance'}
    if type(result) is not dict or not verified <= result.keys(): reject('missing worker binding')
    if (result['source_sha256'] != source_sha or result['worker_request_sha256'] != sha(canonical(request))
            or canonical(result['_core_provenance']) != canonical(core_provenance)): reject('worker binding mismatch')
    if set(result) == verified | {'error_code','error_category'}:
        code=result['error_code']
        if type(code) is not str or code not in FAILURES or FAILURES[code]!=result['error_category']: reject('unknown worker error')
        return None
    keys(result,verified|{'translated_transport'},verified|{'translated_transport'})
    value = result['translated_transport']
    fields={'schema','source_sha256','request_sha256','frame','preparation','transport'}
    keys(value,fields,fields)
    if (value['schema'] != SCHEMA or value['source_sha256'] != source_sha
            or value['request_sha256'] != sha(canonical(request['controls']))
            or value['frame'] != request['source_frame']): reject('artifact binding mismatch')
    graph_data = graph_preflight(strict_json(source))
    if sha(source) != source_sha: reject('result source integrity')
    controls=request['controls']
    translations={row['node_id']:row['value_m'] for row in controls['translations_m']}
    reference_g={row['edge_id']:row['value_S'] for row in controls['reference_conductance_S']}
    ordered_nodes=sorted(translations)
    ordered_edges=sorted(reference_g)
    endpoints={row['id']:[row['node_i'],row['node_j']] for row in graph_data['edges']}
    binding=value['preparation']
    stored=stored_frame(graph_data,request['source_frame'])
    expected={
        'schema':PREPARATION_SCHEMA, 'geometry_dimension':3,
        'length_unit':request['source_unit'], 'translation_unit':'m', 'conductance_unit':'S',
        'coordinate_frame':stored, 'translation_frame':'source_graph_xyz_axes',
        'node_ids':ordered_nodes, 'edge_ids':ordered_edges,
        'endpoints_by_id':[endpoints[e] for e in ordered_edges],
        'translations_m':[translations[n] for n in ordered_nodes],
        'reference_conductance_S':[reference_g[e] for e in ordered_edges],
        **POLICIES,
    }
    if type(binding) is not dict or any(binding.get(k)!=val for k,val in expected.items()):
        reject('preparation binding mismatch')
    for field in ('node_ids','edge_ids'):
        if type(binding[field]) is not list: reject('native ID list')
        for item in binding[field]: integer(item)
    for pair in binding['endpoints_by_id']:
        for item in pair: integer(item)
    for row in binding['translations_m']: vector(row,3)
    for item in binding['reference_conductance_S']: number(item,positive=True)
    for prefix in ('source_full_graph','derived_full_graph','source_graph_order_envelope','derived_graph_order_envelope','input'):
        digest=binding.get(prefix+'_sha256')
        if type(digest) is not str or not re.fullmatch(r'[a-f0-9]{64}',digest) or digest=='0'*64:
            reject('native preparation hash')
    bound_input={k:binding[k] for k in ('schema','source_graph_order_envelope_sha256','node_ids','edge_ids',
        'translations_m','reference_conductance_S','length_unit','coordinate_frame')}
    if binding['input_sha256'] != sha(canonical(bound_input | {'policies':POLICIES})): reject('native input digest')
    for field in ('reference_lengths_m','derived_lengths_m','conductance_S'):
        rows=binding.get(field)
        if type(rows) is not list or len(rows)!=len(ordered_edges): reject('native edge field width')
        for item in rows: number(item,positive=True)
    transport=value['transport']
    if type(transport) is not dict or len(canonical(value)) > LIMITS['result']: reject('artifact budget or type')
    if (transport.get('schema')!='fibernet-linear-conductance-1' or transport.get('status')!='converged'
            or transport.get('prescribed_voltage_V') != {str(row['node_id']):row['value_V'] for row in request['controls']['dirichlet_voltage_V']}):
        reject('native transport contract')
    for field in ('node_ids','edge_ids','endpoints_by_id','conductance_S'):
        if type(transport.get(field)) is not list or transport[field] != binding.get(field):
            reject('native transport ordering or binding')
    for field, id_field in (('voltage_V','node_ids'),('node_injection_A','node_ids'),
                            ('current_A','edge_ids'),('edge_power_W','edge_ids')):
        values=transport.get(field)
        if type(values) is not list or len(values)!=len(transport[id_field]): reject('native field width')
        for value_item in values: number(value_item)
    for field in ('free_node_residual_A','global_current_balance_A','dissipated_power_W','boundary_power_W'):
        number(transport.get(field))
    return value

def enable_tool(definitions):
    """Add the opt-in tool and permit its artifact in opt-in metadata reads."""
    from copy import deepcopy
    electrical=next(d for d in definitions if d['name']=='fibernet_solve_linear_transport')
    tool=deepcopy(electrical)
    tool['name']=TOOL
    tool['description']=('Solve steady electrical conduction after explicit SI XYZ translations on a catalog 3D graph. '
        'Supply one translation per stable node, one positive reference conductance per stable edge, and fixed voltages. '
        'Preserves source topology and uses straight chords and fixed effective area. No mechanical solve or force feedback.')
    i=tool['inputSchema']
    wire_id={'type':'integer','minimum':0,'maximum':2**53-1}
    def row_group(id_key,value_key,value,max_items):
        return {'type':'array','minItems':1,'maxItems':max_items,'items':{
            'type':'object','properties':{id_key:wire_id,value_key:value},
            'required':[id_key,value_key],'additionalProperties':False}}
    i['properties']={
        'graph_id':{'$ref':'#/$defs/Id'},'frame_id':{'$ref':'#/$defs/Frame'},
        'length_unit':{'enum':['m','mm','um']},
        'translations_m':row_group('node_id','value_m',{'type':'array','minItems':3,'maxItems':3,'items':{'type':'number'}},LIMITS['nodes']),
        'reference_conductance_S':row_group('edge_id','value_S',{'type':'number','exclusiveMinimum':0},LIMITS['edges']),
        'dirichlet_voltage_V':row_group('node_id','value_V',{'type':'number'},LIMITS['nodes'])}
    i['required']=list(i['properties'])
    o=tool['outputSchema']
    o['$defs']['Artifact']['properties']['kind']={'const':KIND}
    success=o['oneOf'][0]
    success['properties']['controls_sha256']={'$ref':'#/$defs/Hash'}
    success['required'].append('controls_sha256')
    info=next(d for d in definitions if d['name']=='fibernet_artifact_info')
    info['outputSchema']['$defs']['Artifact']['properties']['kind']['enum'].append(KIND)
    definitions.append(tool)

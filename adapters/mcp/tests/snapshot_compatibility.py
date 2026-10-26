"""Exact additive inverse for this reviewed eighth-tool contract only."""
import copy
import hashlib
import json

NEW_TOOL_HASHES = ('af8179edf8a1106379001a3371fce6121675def59510945bb838688845ae0432', '4eb4499ff742c04e0c49fedade8cdbcf74aebc708849664c51573b83cd302e11')
NAME = 'fibernet_solve_thermoelastic_network_snapshot'
KIND = 'thermoelastic_network_snapshot'
CAPABILITY = 'solve_thermoelastic_network_snapshot'
PHYSICS = 'prescribed_temperature_linear_planar_welded_snapshot'
LIMITS = {'max_nodes':192,'max_edges':256,'max_source_bytes':1048576,
          'max_request_bytes':65536,'max_result_bytes':1048576}
NAMES = ('fibernet_catalog','fibernet_validate','fibernet_analyze_topology',
         'fibernet_artifact_info','fibernet_artifact_read','fibernet_solve_linear_transport',
         'fibernet_prepare_deformed_transport',NAME)


def legacy_seven(definitions):
    result=copy.deepcopy(definitions)
    assert tuple(t['name'] for t in result)==NAMES
    new=result.pop()
    raw=json.dumps(new,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    assert hashlib.sha256(raw).hexdigest() in NEW_TOOL_HASHES
    for tool in result:
        artifact=tool['outputSchema'].get('$defs',{}).get('Artifact')
        if artifact and 'enum' in artifact['properties']['kind']:
            assert artifact['properties']['kind']['enum'].pop()==KIND
    catalog=result[0]['outputSchema']['oneOf'][0]['properties']
    assert catalog['capabilities']['const'].pop()==CAPABILITY
    limits=catalog['limits']['properties'].pop('thermoelastic')
    assert limits=={'type':'object','properties':{k:{'const':v} for k,v in LIMITS.items()},
                    'required':list(LIMITS),'additionalProperties':False}
    assert catalog['limits']['required'].pop()=='thermoelastic'
    return result

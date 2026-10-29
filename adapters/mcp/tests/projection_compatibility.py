"""Exact additive inverse for the checked projection-report tool."""
import copy
import hashlib
import json

NAME = 'fibernet_project_fiber_network_report'
KIND = 'fiber_network_projection_report'
CAPABILITY = 'project_fiber_network_report'
NEW_TOOL_HASHES = ('159c14fa00a201030a56051a6dec0266b30a3d0aa7f891b5d29073479e109589', 'd2e333f738ab0bbe9be46ff7c010b47ca71ce498ff423e8f04b10a379761bcec')

def legacy_eight(definitions):
    result=copy.deepcopy(definitions)
    assert len(result)==9 and result[-1]['name']==NAME
    tool=result.pop()
    raw=json.dumps(tool,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    assert hashlib.sha256(raw).hexdigest() in NEW_TOOL_HASHES
    for old in result:
        for direction in ('inputSchema','outputSchema'):
            artifact=old[direction].get('$defs',{}).get('Artifact')
            if artifact and KIND in artifact['properties']['kind'].get('enum',[]):
                assert artifact['properties']['kind']['enum'].pop()==KIND
    assert result[0]['outputSchema']['oneOf'][0]['properties']['capabilities']['const'].pop()==CAPABILITY
    return result

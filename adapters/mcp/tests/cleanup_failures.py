"""Supplemental owned-file failure tests; transport proof remains separate."""
import asyncio,hashlib,json,os,subprocess,sys,uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from acceptance import ROOT,ENV
from fibernet_mcp.safety import write_at,open_directory
from fibernet_mcp.server import ManagedStore,serve
OUT=ROOT/'evidence'/('cleanup-'+uuid.uuid4().hex[:8]);OUT.mkdir()
fd=open_directory(str(OUT))
try:
 with patch('os.fsync',side_effect=OSError('injected fsync failure')):
  try:write_at(fd,'partial',b'abc')
  except OSError:pass
  else:raise AssertionError('injection absent')
 assert not (OUT/'partial').exists()
 (OUT/'preexisting').write_bytes(b'keep')
 try:write_at(fd,'preexisting',b'replace')
 except FileExistsError:pass
 else:raise AssertionError('overwrite accepted')
 assert (OUT/'preexisting').read_bytes()==b'keep'
finally:os.close(fd)
store=ManagedStore(str(ROOT/'evidence/manifest.json'),str(OUT));store.put_file('extra',b'x')
fds=[store.fd,store.root];original=os.unlink;failed=next(iter(store.files));calls=[]
def unlink(name,*a,**kw):
 calls.append(name)
 if name==failed:raise PermissionError('injected unlink failure')
 return original(name,*a,**kw)
with patch('os.unlink',side_effect=unlink):store.close()
assert set(calls)==store.files
for fd in fds:
 try:os.fstat(fd)
 except OSError:pass
 else:raise AssertionError('fd leaked')
# Clean the deliberate injected failure using the known test-owned filename.
(Path(store.path)/failed).unlink();Path(store.path).rmdir()
(OUT/'audit.json').write_bytes(b'preexisting audit')
p=subprocess.run([sys.executable,'-m','fibernet_mcp.server','--manifest',str(ROOT/'evidence/manifest.json'),'--output-root',str(OUT),'--audit','audit.json'],input=b'',stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV,timeout=10)
assert p.returncode==2 and (OUT/'audit.json').read_bytes()==b'preexisting audit'
assert not list(OUT.glob('fibernet-mcp-*'))
# Real CLI admission must precede protocol import; a valid admission must still
# release its owned files when an optional protocol import fails.
guard='''import json,runpy,sys
from pathlib import Path
output=Path(sys.argv.pop(1)); marker=Path(sys.argv.pop(1))
class Guard:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('mcp','jsonschema'):
            sessions=list(output.glob('fibernet-mcp-*'))
            marker.write_text(json.dumps({'module':fullname,'sessions':len(sessions),
                'owned_files':sum(len(list(s.iterdir())) for s in sessions)}))
            raise ImportError('controlled protocol setup failure')
sys.meta_path.insert(0,Guard())
sys.argv[0]='fibernet_mcp.server'
runpy.run_module('fibernet_mcp.server',run_name='__main__')
'''
valid=ROOT/'evidence/manifest.json'
invalid=OUT/'invalid.manifest.json';invalid.write_bytes(b'{}')
source=OUT/'malformed-source.json';source.write_bytes(b'{}')
source_config=json.loads(valid.read_text())
source_config['graphs'][0].update(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
invalid_source=OUT/'invalid-source.manifest.json';invalid_source.write_text(json.dumps(source_config))
for label,manifest in [('valid',valid),('invalid_manifest',invalid),('invalid_source',invalid_source)]:
 marker=OUT/(label+'-import.json')
 p=subprocess.run([sys.executable,'-c',guard,str(OUT),str(marker),
     '--manifest',str(manifest),'--output-root',str(OUT)],input=b'',
     stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV,timeout=10)
 assert p.returncode==2 and p.stdout==b''
 assert p.stderr==b'FiberNet MCP startup or transport rejected\n'
 assert not list(OUT.glob('fibernet-mcp-*'))
 if label=='valid':
  admission=json.loads(marker.read_text())
  assert admission['sessions']==1 and admission['owned_files']==1
 else:assert not marker.exists(), 'invalid manifest/source reached protocol imports'
# Supplemental source-level exception controls prove the ownership finally also
# covers BaseException/cancellation and closes descriptors in the same process.
args=SimpleNamespace(manifest=str(valid),output_root=str(OUT),timeout=10,audit=None)
for error in (RuntimeError('setup'),KeyboardInterrupt(),asyncio.CancelledError()):
 stores=[]
 def admitted(*a,**kw):
  value=ManagedStore(*a,**kw);stores.append(value);return value
 async def fail_setup(*a,**kw):raise error
 with patch('fibernet_mcp.server.ManagedStore',side_effect=admitted),patch('fibernet_mcp.server._serve_admitted',side_effect=fail_setup):
  try:asyncio.run(serve(args))
  except type(error):pass
  else:raise AssertionError('controlled setup failure absent')
 assert len(stores)==1 and not Path(stores[0].path).exists()
 for fd in (stores[0].fd,stores[0].root):
  try:os.fstat(fd)
  except OSError:pass
  else:raise AssertionError('setup failure leaked descriptor')
report={'partial_write_owned_file_removed':True,'preexisting_file_preserved':True,'unlink_failure_other_files_and_fds_closed':True,'audit_collision_session_cleaned':True,'operator_manifest_and_source_refused_before_protocol_import':True,'real_protocol_import_failure_session_cleaned':True,'setup_baseexception_and_cancellation_files_and_fds_closed':True}
(OUT/'report.json').write_text(json.dumps(report,indent=2));print(report)

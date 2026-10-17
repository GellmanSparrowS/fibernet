"""Supplemental owned-file failure tests; transport proof remains separate."""
import json,os,subprocess,sys,uuid
from pathlib import Path
from unittest.mock import patch
from acceptance import ROOT,ENV
from fibernet_mcp.safety import write_at,open_directory
from fibernet_mcp.server import ManagedStore
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
report={'partial_write_owned_file_removed':True,'preexisting_file_preserved':True,'unlink_failure_other_files_and_fds_closed':True,'audit_collision_session_cleaned':True}
(OUT/'report.json').write_text(json.dumps(report,indent=2));print(report)

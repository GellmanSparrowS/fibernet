"""Raw protocol boundary checks supplement the independent SDK acceptance."""
import json,os,select,subprocess,sys,time,uuid
from pathlib import Path
import psutil
from acceptance import ROOT,WORK,ENV
OUT=ROOT/'evidence'/('transport-'+uuid.uuid4().hex[:8]);OUT.mkdir()

def get(p,deadline=5):
 if not select.select([p.stdout],[],[],deadline)[0]:raise AssertionError('response timeout')
 raw=p.stdout.readline();assert len(raw)<=65536
 return json.loads(raw)
def send(p,obj):p.stdin.write(json.dumps(obj).encode()+b'\n');p.stdin.flush()
def launch(manifest):
 p=subprocess.Popen([sys.executable,'-m','fibernet_mcp.server','--manifest',str(manifest),'--output-root',str(OUT),'--audit','eof.json'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
 send(p,{'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2025-11-25','capabilities':{},'clientInfo':{'name':'raw-boundary-client','version':'1'}}});assert 'result' in get(p)
 send(p,{'jsonrpc':'2.0','method':'notifications/initialized'})
 return p

p=launch(ROOT/'evidence/manifest.json')
send(p,{'jsonrpc':'2.0','id':2,'method':'tools/call','params':{'name':'fibernet_catalog','arguments':{}}});gid=get(p)['result']['structuredContent']['graphs'][0]['graph_id']
send(p,{'jsonrpc':'2.0','id':3,'method':'tools/call','params':{'name':'not_a_tool','arguments':{}}});assert get(p)['error']['code']==-32602
send(p,{'jsonrpc':'2.0','id':4,'method':'tools/call','params':{'name':'fibernet_analyze_topology','arguments':{'graph_id':gid}}})
deadline=time.monotonic()+3
while True:
 workers=psutil.Process(p.pid).children(recursive=True)
 if workers:break
 assert time.monotonic()<deadline;time.sleep(.01)
child=workers[0].pid;p.stdin.close();p.wait(timeout=5)
assert not psutil.pid_exists(child)
audit=json.loads((OUT/'eof.json').read_text());assert audit['workers'][0]['reaped']
assert not list(OUT.glob('fibernet-mcp-*'))
# Schema-valid but invalid topology remains addressable for an invalid report.
data=json.loads((ROOT/'fixtures/voronoi_4340.json').read_text());data['edges'][0]['node_j']=999999
source=OUT/'invalid-topology.json';source.write_text(json.dumps(data))
import hashlib
manifest=json.loads((ROOT/'evidence/manifest.json').read_text());manifest['graphs'][0].update(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
m=OUT/'invalid.manifest.json';m.write_text(json.dumps(manifest))
# Retain original EOF evidence under another filename before launching second session.
(OUT/'eof.json').rename(OUT/'eof-first.json')
p=launch(m)
send(p,{'jsonrpc':'2.0','id':2,'method':'tools/call','params':{'name':'fibernet_catalog','arguments':{}}});gid=get(p)['result']['structuredContent']['graphs'][0]['graph_id']
send(p,{'jsonrpc':'2.0','id':3,'method':'tools/call','params':{'name':'fibernet_validate','arguments':{'graph_id':gid}}});d=get(p)['result'];assert not d['isError'] and d['structuredContent']['validation_status']=='invalid'
send(p,{'jsonrpc':'2.0','id':4,'method':'tools/call','params':{'name':'fibernet_analyze_topology','arguments':{'graph_id':gid}}});d=get(p)['result'];assert d['isError'] and d['structuredContent']['error']['category']=='domain'
p.stdin.close();p.wait(timeout=5)
report={'unknown_tool_protocol_error':True,'eof_worker_reaped':True,'eof_reason':audit['workers'][0]['reason'],'invalid_report_reachable':True,'invalid_analysis_domain_error':True}
(OUT/'report.json').write_text(json.dumps(report,indent=2));print(report)

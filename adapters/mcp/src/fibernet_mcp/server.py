"""Linux-only stdio server. Five bounded operations; no model-supplied paths."""
import argparse
import asyncio
import base64
import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import uuid

import anyio
import jsonschema
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
import mcp.types as mt
from .safety import (ContractError, MAX_BYTES, canonical, graph_preflight,
                     open_directory, read_at, read_file, sha, strict_json, write_at, verify_core)

VERSION = 'fibernet.mcp/1'
TREE = '3b1c2e98599ddea3923abba6a762e8cd5651bfb1'
MAX_SESSION = 48 * 1024 * 1024
ID = re.compile(r'^[a-f0-9]{32}$')

def envelope(**kw): return {'schema_version': VERSION, **kw}
def failure(code, category='contract', retryable=False):
    return envelope(error={'category':category,'code':code,'message':code.replace('_',' ').lower(),'retryable':retryable})

class ManagedStore:
    def __init__(self, manifest, output_root):
        self.root = open_directory(output_root)
        self.session = 'fibernet-mcp-' + uuid.uuid4().hex
        created=False
        try:
            os.mkdir(self.session, mode=0o700, dir_fd=self.root);created=True
            self.fd = os.open(self.session, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW, dir_fd=self.root)
        except BaseException:
            if created:
                with contextlib.suppress(OSError): os.rmdir(self.session,dir_fd=self.root)
            os.close(self.root)
            raise
        self.path = os.path.join(os.path.abspath(output_root), self.session)
        self.files = set(); self.graphs = {}; self.artifacts = {}; self.total = 0
        self.producer = {'fibernet_version':'4.3.0','source_tree':TREE,'adapter_version':'0.0.0-private',
                         'sdk_version':importlib.metadata.version('mcp')}
        try:
            cfg = strict_json(read_file(manifest))
            if not isinstance(cfg, dict) or set(cfg) != {'source_tree','graphs'} or cfg['source_tree'] != TREE:
                raise ContractError('invalid operator manifest')
            if not isinstance(cfg['graphs'],list) or not 1 <= len(cfg['graphs']) <= 32: raise ContractError('manifest graph limit')
            for item in cfg['graphs']:
                if not isinstance(item,dict) or set(item) != {'path','sha256','length_unit','frame_id'}: raise ContractError('invalid graph manifest')
                if item['length_unit'] not in ('m','cm','mm','um','nm','unspecified'): raise ContractError('invalid declared unit')
                if not isinstance(item['frame_id'],str) or not re.fullmatch('[A-Za-z0-9_-]{1,64}', item['frame_id']): raise ContractError('invalid frame')
                raw = read_file(item['path']); digest = sha(raw)
                if digest != item['sha256']: raise ContractError('source digest mismatch')
                data = graph_preflight(strict_json(raw))
                gid = uuid.uuid4().hex
                self.put_file(gid, raw)
                self.graphs[gid] = {'graph_id':gid,'source_sha256':digest,'graph_schema':'3.1',
                    'node_count':len(data['nodes']),'edge_count':len(data['edges']),
                    'length_unit':item['length_unit'],'frame_id':item['frame_id'],
                    'unit_provenance':'unknown' if item['length_unit']=='unspecified' else 'operator_asserted',
                    'frame_provenance':'unknown' if item['frame_id']=='unknown' else 'operator_asserted'}
        except BaseException:
            self.close(); raise

    def put_file(self, name, raw):
        if self.total + len(raw) > MAX_SESSION: raise ContractError('session storage limit')
        write_at(self.fd,name,raw); self.files.add(name); self.total += len(raw)

    def source(self,gid):
        if gid not in self.graphs: raise ContractError('unknown handle')
        raw = read_at(self.fd,gid)
        if sha(raw) != self.graphs[gid]['source_sha256']: raise ContractError('source integrity failure')
        return self.graphs[gid]

    def artifact(self, gid, metrics):
        raw = canonical(metrics); digest = sha(raw)
        for a in self.artifacts.values():
            if a['sha256']==digest and a['source_sha256']==self.graphs[gid]['source_sha256']: return a
        if len(self.artifacts)>=64: raise ContractError('artifact limit')
        aid=uuid.uuid4().hex; self.put_file(aid,raw)
        a={'artifact_id':aid,'uri':'fibernet://artifacts/'+aid,'mime_type':'application/json',
           'byte_size':len(raw),'sha256':digest,'source_sha256':self.graphs[gid]['source_sha256'],
           'producer':self.producer,'kind':'topology_metrics'}
        self.artifacts[aid]=a;return a

    def info(self,aid):
        if aid not in self.artifacts: raise ContractError('unknown handle')
        return self.artifacts[aid]

    def close(self):
        try:
            for name in self.files:
                with contextlib.suppress(OSError): os.unlink(name,dir_fd=self.fd)
        finally:
            try: os.close(self.fd)
            finally:
                try:
                    with contextlib.suppress(OSError): os.rmdir(self.session,dir_fd=self.root)
                finally: os.close(self.root)

class Supervisor:
    def __init__(self,store,timeout=10):
        self.store=store;self.timeout=timeout;self.active=None;self.audit=[]
    async def run(self,operation,gid):
        if len(self.audit)>=128: return failure('WORKER_LIMIT','resource')
        if self.active is not None: return failure('BUSY','resource',True)
        self.store.source(gid)
        out='worker-'+uuid.uuid4().hex
        env={k:os.environ[k] for k in ('PATH','PYTHONPATH','MPLCONFIGDIR','XDG_CACHE_HOME') if k in os.environ}
        env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        p=subprocess.Popen([sys.executable,'-m','fibernet_mcp.worker',operation,
                            os.path.join(self.store.path,gid),os.path.join(self.store.path,out),
                            self.store.graphs[gid]['source_sha256']],
                           stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                           env=env,start_new_session=True)
        self.active=p;start=time.monotonic();reason='completed'
        try:
            while p.poll() is None:
                if time.monotonic()-start >= self.timeout:
                    reason='timed_out';return failure('WORKER_TIMEOUT','resource',True)
                await anyio.sleep(.02)
            if p.returncode != 0: reason='failed';return failure('WORKER_FAILED','internal')
            raw=read_at(self.store.fd,out)
            return strict_json(raw)
        except BaseException:
            reason='cancelled';raise
        finally:
            # Shield actual process reaping from cancellation, with hard kill escalation.
            with anyio.CancelScope(shield=True):
                if p.poll() is None:
                    with contextlib.suppress(ProcessLookupError): os.killpg(p.pid,signal.SIGTERM)
                    deadline=time.monotonic()+.25
                    while p.poll() is None and time.monotonic()<deadline: await anyio.sleep(.01)
                    if p.poll() is None:
                        with contextlib.suppress(ProcessLookupError): os.killpg(p.pid,signal.SIGKILL)
                p.wait()
                with contextlib.suppress(FileNotFoundError): os.unlink(out,dir_fd=self.store.fd)
                self.audit.append({'pid':p.pid,'reason':reason,'returncode':p.returncode,
                                   'reaped':True,'elapsed_s':round(time.monotonic()-start,4)})
                self.active=None

class BoundedInput:
    def __aiter__(self): return self.lines()
    async def lines(self):
        while True:
            raw=await anyio.to_thread.run_sync(lambda: sys.stdin.buffer.readline(20*1024*1024+1),abandon_on_cancel=True)
            if not raw: return
            if len(raw)>20*1024*1024 or not raw.endswith(b'\n'): raise ContractError('transport line budget')
            # Strict JSON before SDK decoding; JSON RPC string arguments bounded too.
            strict_json(raw)
            yield raw.decode('utf-8')

class BoundedOutput:
    async def write(self,s):
        if len(s.encode('utf-8'))>65536: raise ContractError('reply budget exceeded')
        sys.stdout.write(s)
    async def flush(self): sys.stdout.flush()

async def serve(args):
    if sys.platform!='linux' or not hasattr(os,'O_NOFOLLOW'): raise ContractError('Linux confinement required')
    if importlib.metadata.version('mcp')!='1.29.0': raise ContractError('untested SDK version')
    verify_core()
    store=ManagedStore(args.manifest,args.output_root);sup=Supervisor(store,args.timeout)
    definitions=json.loads(Path(__file__).with_name('tool_schemas.json').read_text())['tools']
    # Remove unused local definitions from each advertised schema to fit bounded list output.
    def prune(schema):
        needed=set()
        def walk(x):
            if isinstance(x,dict):
                if '$ref' in x: needed.add(x['$ref'].split('/')[-1])
                for k,v in x.items():
                    if k!='$defs': walk(v)
            elif isinstance(x,list):
                for v in x: walk(v)
        walk(schema)
        previous=None
        while previous!=needed:
            previous=set(needed)
            for key in list(needed): walk(schema['$defs'][key])
        schema['$defs']={k:v for k,v in schema['$defs'].items() if k in needed}
    for definition in definitions:
        prune(definition['inputSchema']);prune(definition['outputSchema'])
    indexed={d['name']:d for d in definitions}
    server=Server('fibernet-private',version='0.0.0')
    @server.list_tools()
    async def tools(): return [mt.Tool(**d) for d in definitions]
    @server.list_resources()
    async def resources(): return [mt.Resource(uri='fibernet://capabilities',name='capabilities',mimeType='application/json',annotations=mt.Annotations(audience=['user','assistant'],priority=.5))]
    @server.list_resource_templates()
    async def templates(): return [mt.ResourceTemplate(uriTemplate='fibernet://artifacts/{artifact_id}',name='artifact manifest',mimeType='application/json')]
    @server.read_resource()
    async def resource(uri):
        u=str(uri)
        if u=='fibernet://capabilities': data={'schema_version':VERSION,'protocol':'2025-11-25','source_tree':TREE,'physics':[],'transport':'stdio','platform':'linux'}
        elif re.fullmatch(r'fibernet://artifacts/[a-f0-9]{32}',u):
            try: data=store.info(u.rsplit('/',1)[1])
            except ContractError: raise McpError(mt.ErrorData(code=-32002,message='Resource not found'))
        else: raise McpError(mt.ErrorData(code=-32002,message='Resource not found'))
        return [ReadResourceContents(content=canonical(data).decode(),mime_type='application/json')]
    @server.call_tool(validate_input=False)
    async def call(name,arguments):
        data=None
        try:
            if name not in indexed: raise ContractError('unknown tool')
            jsonschema.Draft202012Validator(indexed[name]['inputSchema']).validate(arguments)
            if name=='fibernet_catalog':
                data=envelope(graphs=list(store.graphs.values()),limits={'max_nodes':10000,'max_edges':30000,'max_graph_bytes':MAX_BYTES,'max_reply_bytes':65536},capabilities=['validate_structure','analyze_topology','read_artifact'])
            elif name in ('fibernet_validate','fibernet_analyze_topology'):
                gid=arguments['graph_id'];g=store.source(gid)
                result=await sup.run('validate' if name=='fibernet_validate' else 'topology',gid)
                if 'error' in result: data=result
                elif name=='fibernet_validate':
                    data=envelope(graph_id=gid,source_sha256=g['source_sha256'],validation_status='valid' if result['valid'] else 'invalid',scope='serialization_and_topology',node_count=g['node_count'],edge_count=g['edge_count'],issues=[] if result['valid'] else [{'code':'INVALID_GRAPH','path':'graph','message':'native graph or topology validation failed'}],issues_truncated=False)
                elif not result['valid']: data=failure('INVALID_GRAPH','domain')
                else:
                    metrics=result['metrics'];a=store.artifact(gid,metrics)
                    data=envelope(graph_id=gid,source_sha256=g['source_sha256'],metrics={k:metrics[k] for k in ['schema','n_nodes','n_physical_edges','n_unique_connections','n_components','n_isolates','cycle_rank_physical','largest_component_fraction']},artifact=a)
            elif name=='fibernet_artifact_info': data=envelope(artifact=store.info(arguments['artifact_id']))
            elif name=='fibernet_artifact_read':
                aid=arguments['artifact_id'];a=store.info(aid);raw=read_at(store.fd,aid)
                if sha(raw)!=a['sha256']: raise ContractError('artifact integrity failure')
                offset=arguments['offset']
                if offset>len(raw): raise ContractError('offset past EOF')
                chunk=raw[offset:offset+arguments.get('length',32768)];n=offset+len(chunk)
                data=envelope(artifact_id=aid,offset=offset,next_offset=n,eof=n==len(raw),encoding='base64',data=base64.b64encode(chunk).decode(),sha256=a['sha256'],total_bytes=len(raw))
        except (ContractError,jsonschema.ValidationError,TypeError,KeyError,ValueError,OSError): data=failure('INVALID_ARGUMENT_OR_HANDLE')
        except Exception: data=failure('INTERNAL_FAILURE','internal')
        if name in indexed:
            try: jsonschema.Draft202012Validator(indexed[name]['outputSchema']).validate(data)
            except jsonschema.ValidationError: data=failure('OUTPUT_CONTRACT_FAILURE','internal')
        # Avoid duplicating potentially44KiB base64 in textual fallback.
        text='FiberNet result: '+('error '+data['error']['code'] if 'error' in data else 'see structuredContent')
        return mt.CallToolResult(content=[mt.TextContent(type='text',text=text)],structuredContent=data,isError='error' in data)
    sdk_call = server.request_handlers[mt.CallToolRequest]
    async def protocol_call(request):
        if request.params.name not in indexed:
            raise McpError(mt.ErrorData(code=-32602,message='Unknown tool'))
        return await sdk_call(request)
    server.request_handlers[mt.CallToolRequest] = protocol_call
    try:
        async with stdio_server(stdin=BoundedInput(),stdout=BoundedOutput()) as streams:
            await server.run(*streams,server.create_initialization_options())
    finally:
        # SDK task group cancels in-flight work before this scope ends.
        try:
            if args.audit:
                fd=open_directory(args.output_root)
                try: write_at(fd,args.audit,canonical({'workers':sup.audit,'artifacts':len(store.artifacts)}))
                finally: os.close(fd)
        finally: store.close()

def main():
    if sys.platform!='linux': raise SystemExit('Linux confinement required')
    resource.setrlimit(resource.RLIMIT_AS,(384*1024*1024,512*1024*1024))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True);parser.add_argument('--output-root',required=True)
    parser.add_argument('--timeout',type=float,default=10)
    parser.add_argument('--audit',help='operator-only basename for bounded shutdown evidence')
    args=parser.parse_args()
    if not .001<=args.timeout<=10: parser.error('timeout must be .001..10 seconds')
    if args.audit and not re.fullmatch('[A-Za-z0-9_-]{1,64}\\.json',args.audit): parser.error('invalid audit basename')
    try: anyio.run(serve,args)
    except Exception:
        sys.stderr.write('FiberNet MCP startup or transport rejected\n');sys.exit(2)
if __name__=='__main__':main()

"""Exercise real same-session Tet4/graph BUSY and cancellation through MCP."""
import json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
from hashlib import sha256
from tet4_stdio import cfg,fixture_case,canonical,TET4_TOOL

def child_workers(server_pid):
    found=[]
    for f in Path('/proc').glob('[0-9]*/stat'):
        try:
            stat=f.read_text().rsplit(')',1)[1].split()
            if int(stat[1])==server_pid: found.append(int(f.parent.name))
        except (FileNotFoundError,PermissionError,OSError,IndexError,ValueError):continue
    return found
def wait(fn,seconds=5):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        value=fn()
        if value: return value
        time.sleep(.01)
    raise AssertionError('condition timed out')
def wire(id_,method,params):
    j=dict(jsonrpc='2.0',method=method,params=params)
    if id_ is not None:j['id']=id_
    return canonical(j)+b'\n'
def digest_paths(root):
    return {p.name:sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}

def test():
    with tempfile.TemporaryDirectory(prefix='fiber-mixed-lifecycle-') as q:
        root=Path(q);(root/'manifest').write_bytes(canonical(cfg(True)))
        cmd=[sys.executable,'-B','-m','fibernet_mcp.server','--manifest',str(root/'manifest'),
             '--output-root',str(root),'--audit','audit.json']
        with subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as p:
            def send(id_,meth,args):p.stdin.write(wire(id_,meth,args));p.stdin.flush()
            def read(i,skip=()):
                while True:
                    x=json.loads(p.stdout.readline())
                    if x.get('id') in skip:continue
                    assert x.get('id')==i,x
                    assert x.get('result'),x
                    return x['result']
            def call(i,tool,args,skip=()):
                send(i,'tools/call',dict(name=tool,arguments=args));return read(i,skip)
            def result(r):return r['structuredContent']
            send(1,'initialize',dict(protocolVersion='2025-11-25',capabilities={},clientInfo=dict(name='tet4-mixed-lifecycle-test',version='1')))
            assert read(1)['protocolVersion']=='2025-11-25'
            send(None,'notifications/initialized',{})
            cat=result(call(2,'fibernet_catalog',{})); gid=cat['graphs'][0]['graph_id']
            mids={x['fixture']:x['mesh_id'] for x in cat['tet4_meshes']}
            assert len(mids)==2
            session,=root.glob('fibernet-mcp-*')
            before=digest_paths(session); assert set(before)=={gid,*mids.values()}
            _,_,_,a=fixture_case('welded',mids['welded'],.5,True)
            overflow=wire(31,'tools/call',dict(name=TET4_TOOL,arguments=a|{'ambient_K':'EXACT_NUM_TOKEN'})).replace(b'"EXACT_NUM_TOKEN"',b'1e999')
            p.stdin.write(overflow);p.stdin.flush()
            failed=read(31); assert failed['isError'] and result(failed)['error']['code']=='SOLID_INVALID_ARGUMENT_OR_HANDLE'
            send(3,'tools/call',dict(name=TET4_TOOL,arguments=a))
            worker,=wait(lambda:child_workers(p.pid))
            worker_cmd=Path('/proc')/str(worker)/'cmdline'
            wait(lambda:b'fibernet_mcp.worker\0tet4_dc_convection\0' in worker_cmd.read_bytes())
            # Suspend only the verified child so BUSY and hard kill can be
            # tested deterministically; no altered production worker.
            os.kill(worker,signal.SIGSTOP)
            busy=call(4,'fibernet_validate',{'graph_id':gid})
            assert busy['isError'] and result(busy)['error']['code']=='BUSY'
            blocked=call(5,TET4_TOOL,a)
            assert blocked['isError'] and result(blocked)['error']['code']=='BUSY'
            send(None,'notifications/cancelled',dict(requestId=3,reason='test verified worker reaping'))
            wait(lambda:not Path('/proc',str(worker)).exists())
            wait(lambda:digest_paths(session)==before)
            # Same session and unmodified source files after the cancel.
            assert result(call(6,'fibernet_catalog',{},skip=(3,)))['tet4_meshes']==cat['tet4_meshes']
            healthy=call(7,TET4_TOOL,a); assert not healthy.get('isError')
            art=result(healthy)['artifact']
            check=call(8,TET4_TOOL,a); assert result(check)['artifact']==art
            assert set(digest_paths(session))==set(before)|{art['artifact_id']}
            assert {k:digest_paths(session)[k] for k in before}==before
            p.stdin.close()
            assert p.wait(timeout=7)==0,p.stderr.read().decode()
        assert not list(root.glob('fibernet-mcp-*'))
        audit=json.loads((root/'audit.json').read_bytes()); assert audit['artifacts']==1
        workers=audit['workers']; assert len(workers)==3 and all(x['reaped'] for x in workers)
        assert workers[0]['reason']=='cancelled' and workers[0]['pid']==worker and workers[0]['returncode']==-signal.SIGKILL
        assert {x['reason'] for x in workers[1:]}=={'completed'}
        print(json.dumps(dict(scoped_numeric_rpc_continued=True,graph_and_solid_busy=True,cancelled_worker_hard_killed_and_reaped=True,post_cancel_same_session=True,
                              dedupe_published_once=True,managed_session_cleaned=True,audit=audit),indent=2))
if __name__=='__main__':test()

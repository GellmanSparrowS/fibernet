"""Internal worker: reconstruct the graph, call the simulation API, serialize results."""
import hashlib,json,pathlib,sys,time

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from _isolated_taichi_io import read_json,write_json,write_progress

def main():
    root=pathlib.Path(sys.argv[1]).resolve();config=read_json(root/'config.json')
    with (root/'request.json').open('rb') as stream:raw=stream.read(8*1024**2+1)
    if len(raw)>8*1024**2 or hashlib.sha256(raw).hexdigest()!=config['request_sha256']:raise ValueError('request integrity/size failure')
    request=json.loads(raw)
    if request['schema']!='isolated-taichi-cpu/0':raise ValueError('unknown worker schema')
    site=pathlib.Path(config['runtime']['provider_site'])
    native=site/'taichi/_lib/core/taichi_python.cpython-312-x86_64-linux-gnu.so'
    with native.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=config['runtime']['native_sha256']:raise ValueError('native digest mismatch')
    sys.path[:0]=[str(pathlib.Path(__file__).resolve().parents[2]),str(site)]
    import numpy as np
    import fibernet as fn
    from fibernet.sim.isolated_taichi import prepare_request
    graph=fn.StructureGraph.from_dict(request['graph'])
    if list(graph.nodes)!=request['node_ids'] or list(graph.edges)!=request['edge_ids']:raise ValueError('worker row alignment mismatch')
    # Same contract admission in child; preserve all original row controls.
    if prepare_request(graph,**request['controls'])!=raw:raise ValueError('request round-trip mismatch')
    import taichi as ti
    if ti.__version__!=(1,7,4) or pathlib.Path(ti._lib.core.__file__).resolve()!=native.resolve():raise ValueError('unqualified native origin/version')
    if ti.lang.impl.get_runtime().prog is not None:raise ValueError('worker runtime initialized before dispatch')
    events=[dict(sequence=1,phase='provider_validated',monotonic_ns=time.monotonic_ns())]
    write_progress(root/'progress.json',config['request_sha256'],events)
    controls=dict(request['controls']);controls['external_force']=np.asarray(controls['external_force'],dtype=np.float64)
    result=fn.simulate(graph,mode='dynamics',backend='taichi',arch='cpu',num_threads=1,**controls)
    if result.metadata.get('node_ids')!=request['node_ids'] or result.metadata.get('edge_ids')!=request['edge_ids']:raise ValueError('native result row identity mismatch')
    events.append(dict(sequence=2,phase='result_ready',monotonic_ns=time.monotonic_ns()))
    write_progress(root/'progress.json',config['request_sha256'],events)
    result.save(root/'result.json',detailed=True)
    write_json(root/'runtime.json',dict(python=sys.version,numpy=np.__version__,taichi=list(ti.__version__),provider=str(ti._lib.core.__file__),arch=str(ti.cfg.arch),threads=int(ti.cfg.cpu_max_num_threads),default_fp=str(ti.cfg.default_fp),fast_math=bool(ti.cfg.fast_math)))
    # Runtime dies with this process: no caller/global reset or mode patch.

if __name__=='__main__':main()

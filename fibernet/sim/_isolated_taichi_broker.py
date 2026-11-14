"""Internal stdlib-only lease broker. No SDK or physics imports."""
import hashlib,json,os,pathlib,resource,select,signal,subprocess,sys,time

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from _isolated_taichi_io import read_json,write_json

def _bounded_limit(kind, requested):
    """Apply requested ceiling without raising an inherited hard limit."""
    _, inherited_hard = resource.getrlimit(kind)
    ceiling = requested if inherited_hard == resource.RLIM_INFINITY else min(requested, inherited_hard)
    resource.setrlimit(kind, (ceiling, ceiling))

def main():
    root=pathlib.Path(sys.argv[1]).resolve();config=read_json(root/'config.json')
    stop=[None];proc=None;status='error';error=None;reaped=False
    def cancelled(sig,frame): stop[0]='signal-cancelled'
    # Broker-local: do not inherit caller SIGCHLD=SIG_IGN auto-reaping.
    signal.signal(signal.SIGCHLD,signal.SIG_DFL)
    signal.signal(signal.SIGTERM,cancelled);signal.signal(signal.SIGINT,cancelled)
    start=time.monotonic()
    # Broker remains independent of caller; EOF on dedicated stdin is the lease.
    # The native child receives DEVNULL, so cannot accidentally retain that lease.
    def limits():
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
        _bounded_limit(resource.RLIMIT_AS,2*1024**3)
        _bounded_limit(resource.RLIMIT_CPU,120)
        _bounded_limit(resource.RLIMIT_FSIZE,64*1024**2)
        _bounded_limit(resource.RLIMIT_CORE,0)
    try:
        with (root/'worker.log').open('xb') as log:
            proc=subprocess.Popen([config['runtime']['python_executable'],'-I','-B',str(pathlib.Path(__file__).with_name('_isolated_taichi_worker.py')),str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ),cwd=root,start_new_session=True,close_fds=True,preexec_fn=limits)
            while True:
                # Keep the leader unreaped until group teardown, preventing PGID reuse.
                exited=os.waitid(os.P_PID,proc.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
                if exited is not None:
                    status='completed' if exited.si_code==os.CLD_EXITED and exited.si_status==0 else 'worker-error'
                    break
                if stop[0]:status='cancelled';break
                ready,_,_=select.select([sys.stdin.buffer],[],[],.05)
                if ready and os.read(sys.stdin.fileno(),1)==b'':status='cancelled';break
                if time.monotonic()-start>config['wall_seconds']:status='timeout';break
                if sum(f.stat().st_size for f in root.rglob('*') if f.is_file())>80*1024**2:
                    status='output-limit';break
                if (root/'worker.log').stat().st_size>1024**2:status='log-limit';break
    except BaseException as exc:error=repr(exc);status='broker-error'
    finally:
        if proc is not None:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            try:proc.wait(timeout=3);reaped=True
            except BaseException as exc:error=repr(exc);status='cleanup-error'
        result=root/'result.json'
        digest=None
        if result.is_file() and result.stat().st_size<=64*1024**2:
            with result.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        receipt=dict(status=status,error=error,worker_pid=None if proc is None else proc.pid,worker_exit=None if proc is None else proc.returncode,worker_reaped=reaped,wall_seconds=time.monotonic()-start,result_sha256=digest)
        write_json(root/'status.json',receipt)
    return 0 if status=='completed' and reaped else 1

if __name__=='__main__':sys.exit(main())

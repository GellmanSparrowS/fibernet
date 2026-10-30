"""Lease lifecycle tests with a sleeping stdlib-only worker substitute."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

linux_only=pytest.mark.skipif(sys.platform != "linux", reason="isolated broker lifecycle requires Linux")

@linux_only
def test_lease_eof_cancels_and_reaps(tmp_path):
    from fibernet.sim import _isolated_taichi_broker
    broker=tmp_path/'_isolated_taichi_broker.py'
    shutil.copyfile(_isolated_taichi_broker.__file__,broker)
    shutil.copyfile(Path(_isolated_taichi_broker.__file__).with_name('_isolated_taichi_io.py'),tmp_path/'_isolated_taichi_io.py')
    # Same sibling filename, no actual FiberNet or Taichi worker involved.
    (tmp_path/'_isolated_taichi_worker.py').write_text('import time\ntime.sleep(30)\n')
    (tmp_path/'config.json').write_text(json.dumps(dict(runtime=dict(python_executable=sys.executable),wall_seconds=10)))
    with subprocess.Popen([sys.executable,'-I','-B',str(broker),str(tmp_path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True) as proc:
        proc.stdin.close() # parent cancellation/exit lease edge
        proc.wait(timeout=5)
    result=json.loads((tmp_path/'status.json').read_text())
    assert result['status']=='cancelled' and result['worker_reaped'] is True
    assert result['worker_exit']!=0

import pytest

@pytest.mark.parametrize('body,wall,expected',[
    ('raise SystemExit(7)\n',5,'worker-error'),
    ('import time\ntime.sleep(30)\n',1,'timeout'),
])
@linux_only
def test_worker_failure_and_timeout(tmp_path,body,wall,expected):
    from fibernet.sim import _isolated_taichi_broker
    broker=tmp_path/'_isolated_taichi_broker.py';shutil.copyfile(_isolated_taichi_broker.__file__,broker)
    shutil.copyfile(Path(_isolated_taichi_broker.__file__).with_name('_isolated_taichi_io.py'),tmp_path/'_isolated_taichi_io.py')
    (tmp_path/'_isolated_taichi_worker.py').write_text(body)
    (tmp_path/'config.json').write_text(json.dumps(dict(runtime=dict(python_executable=sys.executable),wall_seconds=wall)))
    with subprocess.Popen([sys.executable,'-I','-B',str(broker),str(tmp_path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        proc.wait(timeout=8)
    receipt=json.loads((tmp_path/'status.json').read_text())
    assert receipt['status']==expected and receipt['worker_reaped'] is True

@linux_only
def test_popen_interruption_revokes_already_owned_lease(monkeypatch):
    import os,select
    from fibernet.sim.isolated_taichi import IsolatedTaichiTask
    readers=[]
    def interrupted(*args,**kwargs):
        readers.append(os.dup(kwargs['stdin']))
        raise KeyboardInterrupt('simulated interruption before Popen return')
    monkeypatch.setattr('fibernet.sim.isolated_taichi.subprocess.Popen',interrupted)
    with pytest.raises(KeyboardInterrupt):IsolatedTaichiTask(b'{}',dict(python_executable=sys.executable),5)
    try:
        assert select.select(readers,[],[],.1)[0]
        assert os.read(readers[0],1)==b''
    finally:
        for fd in readers:os.close(fd)

def test_malformed_status_is_domain_error(tmp_path):
    from fibernet.sim.isolated_taichi import IsolatedTaichiTask,IsolatedTaichiError
    class Finished:
        def poll(self):return 1
    task=IsolatedTaichiTask.__new__(IsolatedTaichiTask);task._broker=Finished();task.workdir=tmp_path
    (tmp_path/'status.json').write_text('{bad json')
    with pytest.raises(IsolatedTaichiError):_=task.status

@linux_only
def test_worker_startup_error_recorded(tmp_path):
    from fibernet.sim import _isolated_taichi_broker
    broker=tmp_path/'_isolated_taichi_broker.py';shutil.copyfile(_isolated_taichi_broker.__file__,broker)
    shutil.copyfile(Path(_isolated_taichi_broker.__file__).with_name('_isolated_taichi_io.py'),tmp_path/'_isolated_taichi_io.py')
    (tmp_path/'config.json').write_text(json.dumps(dict(runtime=dict(python_executable=str(tmp_path/'absent-python')),wall_seconds=1)))
    proc=subprocess.run([sys.executable,'-I','-B',str(broker),str(tmp_path)],stdin=subprocess.DEVNULL,capture_output=True,timeout=5)
    result=json.loads((tmp_path/'status.json').read_text())
    assert proc.returncode!=0 and result['status']=='broker-error' and result['worker_exit'] is None

@linux_only
def test_ignored_sigchld_is_reset_in_broker(tmp_path):
    import signal
    from fibernet.sim import _isolated_taichi_broker
    broker=tmp_path/'_isolated_taichi_broker.py';shutil.copyfile(_isolated_taichi_broker.__file__,broker)
    shutil.copyfile(Path(_isolated_taichi_broker.__file__).with_name('_isolated_taichi_io.py'),tmp_path/'_isolated_taichi_io.py')
    (tmp_path/'_isolated_taichi_worker.py').write_text('raise SystemExit(7)\n')
    (tmp_path/'config.json').write_text(json.dumps(dict(runtime=dict(python_executable=sys.executable),wall_seconds=3)))
    with subprocess.Popen([sys.executable,'-I','-B',str(broker),str(tmp_path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,preexec_fn=lambda:signal.signal(signal.SIGCHLD,signal.SIG_IGN)) as proc:
        proc.wait(timeout=6)
    receipt=json.loads((tmp_path/'status.json').read_text())
    assert receipt['status']=='worker-error' and receipt['worker_exit']==7 and receipt['worker_reaped']

@linux_only
def test_group_kill_precedes_reap(tmp_path,monkeypatch):
    import os,types
    from fibernet.sim import _isolated_taichi_broker as broker
    (tmp_path/'config.json').write_text(json.dumps(dict(runtime=dict(python_executable=sys.executable),wall_seconds=3)))
    order=[]
    class Fake:
        pid=987654;returncode=None
        def wait(self,timeout):order.append('reap');self.returncode=0
    def observed(*args):
        assert args[2] & os.WNOWAIT
        order.append('observe-unreaped');return types.SimpleNamespace(si_code=os.CLD_EXITED,si_status=0)
    monkeypatch.setattr(broker.sys,'argv',['broker',str(tmp_path)])
    monkeypatch.setattr(broker.signal,'signal',lambda *args:None)
    monkeypatch.setattr(broker.subprocess,'Popen',lambda *args,**kwargs:Fake())
    monkeypatch.setattr(broker.os,'waitid',observed)
    monkeypatch.setattr(broker.os,'killpg',lambda *args:order.append('kill-owned-group'))
    assert broker.main()==0
    assert order==['observe-unreaped','kill-owned-group','reap']

@linux_only
def test_actual_parent_exit_revokes_lease(tmp_path):
    import os,time
    from fibernet.sim import _isolated_taichi_broker
    broker=tmp_path/'_isolated_taichi_broker.py';shutil.copyfile(_isolated_taichi_broker.__file__,broker)
    shutil.copyfile(Path(_isolated_taichi_broker.__file__).with_name('_isolated_taichi_io.py'),tmp_path/'_isolated_taichi_io.py')
    (tmp_path/'_isolated_taichi_worker.py').write_text('import time\ntime.sleep(30)\n')
    (tmp_path/'config.json').write_text(json.dumps(dict(runtime=dict(python_executable=sys.executable),wall_seconds=10)))
    launcher=tmp_path/'launcher.py'
    launcher.write_text('import subprocess,sys,os\np=subprocess.Popen([sys.executable,"-I","-B",sys.argv[1],sys.argv[2]],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)\nos._exit(0)\n')
    subprocess.run([sys.executable,'-I','-B',str(launcher),str(broker),str(tmp_path)],check=True,timeout=3)
    deadline=time.monotonic()+5
    while not (tmp_path/'status.json').exists() and time.monotonic()<deadline:time.sleep(.05)
    receipt=json.loads((tmp_path/'status.json').read_text())
    assert receipt['status']=='cancelled' and receipt['worker_reaped']
    with pytest.raises(ProcessLookupError):os.killpg(receipt['worker_pid'],0)


@pytest.mark.parametrize("hard,requested,expected",[(0,64,0),(16,64,16),(128,64,64),(-1,64,64),(-1,0,0)])
@linux_only
def test_child_limits_respect_inherited_hard_cap(monkeypatch,hard,requested,expected):
    from fibernet.sim import _isolated_taichi_broker as broker
    import resource
    if hard == -1: hard = resource.RLIM_INFINITY
    calls=[]
    monkeypatch.setattr(broker.resource,"getrlimit",lambda kind:(0,hard))
    monkeypatch.setattr(broker.resource,"setrlimit",lambda kind,limits:calls.append((kind,limits)))
    for kind in (resource.RLIMIT_AS,resource.RLIMIT_CPU,resource.RLIMIT_FSIZE,resource.RLIMIT_CORE):
        broker._bounded_limit(kind,requested)
    assert calls == [(kind,(expected,expected)) for kind in (resource.RLIMIT_AS,resource.RLIMIT_CPU,resource.RLIMIT_FSIZE,resource.RLIMIT_CORE)]

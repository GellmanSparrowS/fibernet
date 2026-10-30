"""Progress contract tests: no native provider, worker launch or simulation."""
import copy
import json
import sys
import pytest
from fibernet.sim.isolated_taichi import IsolatedTaichiTask,IsolatedTaichiError
from fibernet.sim._isolated_taichi_io import write_progress,write_json,validate_progress

pytestmark=pytest.mark.skipif(sys.platform != 'linux', reason='isolated Taichi progress is a Linux-only profile')

DIGEST='a'*64

def task(tmp_path):
    value=IsolatedTaichiTask.__new__(IsolatedTaichiTask)
    value.workdir=tmp_path;value._request_sha256=DIGEST
    return value

def events():
    return [dict(sequence=1,phase='provider_validated',monotonic_ns=10),
            dict(sequence=2,phase='result_ready',monotonic_ns=20)]

def test_missing_then_retained_two_events_detached(tmp_path):
    t=task(tmp_path);before=set(sys.modules)
    assert t.progress is None
    write_progress(tmp_path/'progress.json',DIGEST,events()[:1])
    first=t.progress
    write_progress(tmp_path/'progress.json',DIGEST,events())
    final=t.progress
    assert first['events']==events()[:1]
    assert final['events']==events()
    final['events'].clear()
    assert t.progress['events']==events()
    assert not any(n=='taichi' or n.startswith('taichi.') for n in set(sys.modules)-before)
    assert not (tmp_path/'status.json').exists() # progress never invents terminal state

@pytest.mark.parametrize('bad',[
    {'schema':'wrong'}, {'request_schema':'wrong'}, {'request_sha256':'b'*64},
    {'events':[]}, {'events':events()[1:]}, {'events':events()*2},
    {'events':[dict(sequence=True,phase='provider_validated',monotonic_ns=10)]},
    {'events':[dict(sequence=1,phase='provider_validated',monotonic_ns=True)]},
    {'events':[events()[0],dict(sequence=2,phase='result_ready',monotonic_ns=9)]},
    {'events':[dict(sequence=1,phase='result_ready',monotonic_ns=10)]},
    {'events':[dict(sequence=1,phase='provider_validated',monotonic_ns=2**63)]},
    {'extra':0},
])
def test_malformed_or_wrong_request_rejected(tmp_path,bad):
    t=task(tmp_path);write_progress(tmp_path/'progress.json',DIGEST,events())
    value=json.loads((tmp_path/'progress.json').read_text());value.update(copy.deepcopy(bad))
    write_json(tmp_path/'progress.json',value)
    with pytest.raises(IsolatedTaichiError):_ = t.progress

@pytest.mark.parametrize('raw',[b'{',b'[]',b' '*2049,b'['*1000+b']'*1000])
def test_bounded_decode(tmp_path,raw):
    t=task(tmp_path);(tmp_path/'progress.json').write_bytes(raw)
    with pytest.raises(IsolatedTaichiError):_ = t.progress

def test_atomic_replace_failure_preserves_first_receipt(tmp_path,monkeypatch):
    from fibernet.sim import _isolated_taichi_io as io
    t=task(tmp_path);write_progress(tmp_path/'progress.json',DIGEST,events()[:1])
    def fail(*args):raise OSError('injected replace failure')
    monkeypatch.setattr(io.os,'replace',fail)
    with pytest.raises(OSError):write_progress(tmp_path/'progress.json',DIGEST,events())
    assert t.progress['events']==events()[:1]
    assert not list(tmp_path.glob('.control-*'))

def test_directory_receipt_is_protocol_error(tmp_path):
    t=task(tmp_path);(tmp_path/'progress.json').mkdir()
    with pytest.raises(IsolatedTaichiError):_ = t.progress

@pytest.mark.parametrize('kind',['fifo','symlink','broken-symlink'])
def test_nonregular_progress_rejected_without_pipe_read(tmp_path,kind,monkeypatch):
    import os
    from fibernet.sim import _isolated_taichi_io as io
    t=task(tmp_path);path=tmp_path/'progress.json'
    if kind=='fifo':os.mkfifo(path)
    else:
        target=tmp_path/'other.json'
        if kind=='symlink':target.write_text('{}')
        path.symlink_to(target)
    original=io.os.open;flags_seen=[]
    def checked_open(name,flags,*args,**kwargs):
        assert flags & os.O_NONBLOCK and flags & os.O_NOFOLLOW and flags & os.O_CLOEXEC
        flags_seen.append(flags)
        return original(name,flags,*args,**kwargs)
    monkeypatch.setattr(io.os,'open',checked_open)
    with pytest.raises(IsolatedTaichiError):_ = t.progress
    assert len(flags_seen)==1

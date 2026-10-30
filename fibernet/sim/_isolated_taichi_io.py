"""Bounded task-owned control JSON, without provider imports."""
import json,os,tempfile,stat
from pathlib import Path

def read_json(path,limit=65536):
    with Path(path).open('rb') as stream:raw=stream.read(limit+1)
    if len(raw)>limit:raise ValueError('control JSON exceeds byte limit')
    value=json.loads(raw)
    if type(value) is not dict:raise ValueError('control JSON must be object')
    return value

def write_json(path,value):
    path=Path(path);raw=json.dumps(value,allow_nan=False).encode()
    if len(raw)>65536:raise ValueError('control JSON exceeds byte limit')
    fd,name=tempfile.mkstemp(prefix='.control-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(raw)
        os.replace(name,path)
    finally:
        try:os.unlink(name)
        except FileNotFoundError:pass

PROGRESS_SCHEMA = 'isolated-taichi-progress/1'
PROGRESS_LIMIT = 2048
PROGRESS_PHASES = ('provider_validated', 'result_ready')

def validate_progress(value, request_sha256):
    """Validate a small historical receipt, never a live execution guarantee."""
    if type(value) is not dict or set(value) != {'schema','request_schema','request_sha256','events'}:
        raise ValueError('invalid progress fields')
    if value['schema'] != PROGRESS_SCHEMA or value['request_schema'] != 'isolated-taichi-cpu/0' or value['request_sha256'] != request_sha256:
        raise ValueError('progress identity mismatch')
    events=value['events']
    if type(events) is not list or not 1 <= len(events) <= 2:
        raise ValueError('invalid progress event count')
    previous=-1
    for i,event in enumerate(events):
        if type(event) is not dict or set(event) != {'sequence','phase','monotonic_ns'}:
            raise ValueError('invalid progress event')
        stamp=event['monotonic_ns']
        if type(event['sequence']) is not int or event['sequence'] != i+1 or event['phase'] != PROGRESS_PHASES[i]:
            raise ValueError('invalid progress sequence')
        if type(stamp) is not int or not 0 <= stamp <= 2**63-1 or stamp < previous:
            raise ValueError('invalid progress timestamp')
        previous=stamp
    return value

def write_progress(path, request_sha256, events):
    value=dict(schema=PROGRESS_SCHEMA, request_schema='isolated-taichi-cpu/0',
               request_sha256=request_sha256, events=events)
    validate_progress(value, request_sha256)
    if len(json.dumps(value, allow_nan=False).encode()) > PROGRESS_LIMIT:
        raise ValueError('progress byte limit exceeded')
    write_json(path,value)

def read_progress(path, request_sha256):
    """Nonblocking regular-file admission for the Linux progress receipt."""
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > PROGRESS_LIMIT:
            raise ValueError('progress must be a bounded regular file')
        with os.fdopen(fd,'rb') as stream:
            fd=None
            raw=stream.read(PROGRESS_LIMIT+1)
        if len(raw)>PROGRESS_LIMIT:
            raise ValueError('progress byte limit exceeded')
        return validate_progress(json.loads(raw),request_sha256)
    finally:
        if fd is not None:os.close(fd)

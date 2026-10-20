"""Strict original-byte validation and Linux directory-relative confinement."""
import hashlib
import json
import math
import os
import stat

MAX_BYTES = 16 * 1024 * 1024
MAX_NODES = 10000
MAX_EDGES = 30000

class ContractError(ValueError):
    pass

def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ContractError('duplicate JSON key')
        out[key] = value
    return out

def strict_json(raw):
    if len(raw) > MAX_BYTES:
        raise ContractError('JSON byte budget exceeded')
    try:
        result = json.loads(raw, object_pairs_hook=unique,
                            parse_constant=lambda _: (_ for _ in ()).throw(ContractError('nonfinite JSON')))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ContractError('invalid strict JSON') from exc
    check_tree(result)
    return result

def check_tree(value, depth=0):
    if depth > 16:
        raise ContractError('JSON depth budget exceeded')
    if isinstance(value, dict):
        for k, v in value.items():
            check_tree(k, depth + 1)
            check_tree(v, depth + 1)
    elif isinstance(value, list):
        for v in value:
            check_tree(v, depth + 1)
    elif isinstance(value, str):
        if len(value.encode('utf-8')) > 4096:
            raise ContractError('string budget exceeded')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ContractError('nonfinite number')
    elif isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53 - 1:
        raise ContractError('integer exceeds exact interoperable range')

def keys(x, allowed, required):
    if not isinstance(x, dict) or set(x) - set(allowed) or set(required) - set(x):
        raise ContractError('invalid record fields')

def integer(x, minimum=0):
    if type(x) is not int or not minimum <= x <= 2**53 - 1:
        raise ContractError('invalid integer')

def number(x, positive=False):
    if type(x) not in (int, float) or not math.isfinite(x) or (positive and x <= 0):
        raise ContractError('invalid finite number')

def vector(x, size):
    if not isinstance(x, list) or len(x) != size:
        raise ContractError('invalid vector shape')
    for v in x:
        number(v)

MATERIAL_NUMBERS = {'density','youngs_modulus','poissons_ratio','shear_modulus','yield_strength',
 'tensile_strength','fracture_toughness','thermal_conductivity','specific_heat','thermal_expansion',
 'electrical_conductivity','permittivity','permeability'}

def graph_preflight(d):
    keys(d, {'version','dimension','tolerance','next_node_id','next_edge_id','box_size','metadata','nodes','edges'},
         {'version','dimension','tolerance','nodes','edges'})
    if d['version'] != '3.1' or type(d['dimension']) is not int or d['dimension'] not in (2,3):
        raise ContractError('unsupported graph version or dimension')
    number(d['tolerance'], True)
    if d.get('box_size') is not None:
        if not isinstance(d['box_size'], list) or len(d['box_size']) not in (d['dimension'],3):
            raise ContractError('invalid box shape')
        for index,v in enumerate(d['box_size']):
            number(v, index < d['dimension'])
            if v < 0: raise ContractError('negative box size')
    if not isinstance(d.get('metadata', {}), dict): raise ContractError('invalid metadata')
    for kind, cap in [('nodes', MAX_NODES), ('edges', MAX_EDGES)]:
        if not isinstance(d[kind], list) or len(d[kind]) > cap: raise ContractError('entity budget exceeded')
    node_ids = set()
    for n in d['nodes']:
        keys(n, {'id','position','boundary','metadata'}, {'id','position'})
        integer(n['id'])
        if n['id'] in node_ids: raise ContractError('duplicate node ID')
        node_ids.add(n['id']); vector(n['position'], 3)
        if 'boundary' in n and (not isinstance(n['boundary'], list) or len(n['boundary']) != 6 or any(type(b) is not bool for b in n['boundary'])):
            raise ContractError('invalid boundary flags')
        if not isinstance(n.get('metadata', {}), dict): raise ContractError('invalid node metadata')
    edge_ids = set()
    for e in d['edges']:
        keys(e, {'id','node_i','node_j','radius','material','segments','internal_points','metadata'},
             {'id','node_i','node_j','radius','material'})
        for k in ('id','node_i','node_j'): integer(e[k])
        if e['id'] in edge_ids: raise ContractError('duplicate edge ID')
        edge_ids.add(e['id']); number(e['radius'], True)
        if 'segments' in e: integer(e['segments'], 1)
        ip = e.get('internal_points')
        if ip is not None:
            if not isinstance(ip, list) or len(ip) > 256: raise ContractError('internal point budget exceeded')
            for p in ip: vector(p, 3)
        if not isinstance(e.get('metadata', {}), dict): raise ContractError('invalid edge metadata')
        m = e['material']; keys(m, MATERIAL_NUMBERS | {'name','extra','stiffness_tensor'}, set())
        for k,v in m.items():
            if k == 'name':
                if not isinstance(v, str): raise ContractError('invalid material name')
            elif k == 'extra':
                if not isinstance(v, dict): raise ContractError('invalid material extra')
            elif k == 'stiffness_tensor':
                if v is not None:
                    if not isinstance(v, list) or len(v) != 6: raise ContractError('invalid stiffness tensor')
                    for row in v: vector(row, 6)
            elif v is None and k not in {'density','youngs_modulus','poissons_ratio'}: pass
            else: number(v)
    for key, ids in [('next_node_id',node_ids),('next_edge_id',edge_ids)]:
        if key in d:
            integer(d[key])
            if d[key] <= max(ids, default=-1): raise ContractError('invalid next ID')
    return d

def canonical(d):
    return json.dumps(d, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

def sha(raw): return hashlib.sha256(raw).hexdigest()

def open_directory(path):
    """Reject symlinks in every absolute path component, not just the leaf."""
    path = os.path.abspath(path)
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.split('/')[1:]:
            if not part: continue
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = nxt
        return fd
    except BaseException:
        os.close(fd); raise

def read_file(path, cap=MAX_BYTES):
    parent, name = os.path.split(os.path.abspath(path))
    root = open_directory(parent)
    try: return read_at(root, name, cap)
    finally: os.close(root)

def read_at(root, name, cap=MAX_BYTES):
    if '/' in name or name in ('', '.', '..'): raise ContractError('invalid internal name')
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > cap: raise ContractError('invalid file or byte limit')
        with os.fdopen(fd, 'rb', closefd=False) as f: raw = f.read(cap + 1)
        if len(raw) > cap: raise ContractError('byte limit')
        return raw
    finally: os.close(fd)

def write_at(root, name, raw):
    if len(raw) > MAX_BYTES or '/' in name: raise ContractError('invalid managed output')
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=root)
    try:
        with os.fdopen(fd, 'wb', closefd=False) as f: f.write(raw); f.flush(); os.fsync(fd)
    except BaseException:
        # Only this successful O_EXCL open grants ownership of the partial file.
        try: os.unlink(name, dir_fd=root)
        except OSError: pass
        raise
    finally: os.close(fd)


def verify_core():
    """Resolve but do not import FiberNet, then verify every shipped Python file."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.find_spec('fibernet')
    if spec is None or spec.origin is None: raise ContractError('missing native core')
    root = os.path.dirname(spec.origin)
    manifest = strict_json(read_file(str(Path(__file__).with_name('core_manifest.json'))))
    # Reject extra and missing source, and fail closed if traversal is incomplete.
    actual = set()
    def traversal_error(error):
        raise ContractError('native core inventory unavailable') from error
    for directory, directories, files in os.walk(root, followlinks=False, onerror=traversal_error):
        if any(os.path.islink(os.path.join(directory, name)) for name in directories):
            raise ContractError('native core inventory unavailable')
        actual.update(Path(directory, name).relative_to(root).as_posix()
                      for name in files if name.endswith('.py'))
    if actual != set(manifest):
        raise ContractError('native core file inventory differs from frozen source')
    for relative, expected in manifest.items():
        if sha(read_file(os.path.join(root, relative))) != expected:
            raise ContractError('native core differs from frozen source')
    return root

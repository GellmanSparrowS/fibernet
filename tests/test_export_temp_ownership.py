"""Small geometry only: persistence ownership, not a manufacturing calculation."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import zipfile
import warnings
import numpy as np
import pytest
import fibernet.gen.solid_export as m

@pytest.fixture
def fixture():
 v=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
 f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
 return m.FiberSolid(v,f,v.copy(),0.,m.PrintSettings()),SimpleNamespace(positions=v[:3],edges=np.array([[0,1],[1,2],[2,0]]),route_nodes=np.array([0,1,2,0]),route_edges=np.array([0,1,2]))
def export(path,fixture):
 solid,net=fixture
 if path.suffix=='.csv':m.export_route(path,net,net.positions)
 else:m.export_solid(path,solid)
def seed(path,existing=True):
 if existing:path.write_bytes(b'old destination');path.chmod(0o640)
 sentinel=path.with_name(path.name+'.tmp');sentinel.write_bytes(b'untouched sentinel');return sentinel

@pytest.mark.parametrize('suffix',['.stl','.3mf','.csv'])
@pytest.mark.parametrize('existing',[False,True])
def test_success_owns_unique_private_temp(tmp_path,fixture,monkeypatch,suffix,existing):
 dst=tmp_path/('mesh'+suffix);sentinel=seed(dst,existing);original=m.atomic_replace;seen=[]
 def inspect(src,target):
  src=Path(src);assert src.parent==dst.parent and src!=sentinel and src!=dst
  assert src.stat().st_mode&0o777==(0o640 if existing else 0o600)
  seen.append(src);original(src,target)
 monkeypatch.setattr(m,'atomic_replace',inspect);export(dst,fixture)
 assert sentinel.read_bytes()==b'untouched sentinel';assert len(seen)==1 and not seen[0].exists()
 assert dst.stat().st_mode&0o777==(0o640 if existing else 0o600)
 assert sorted(p.name for p in tmp_path.iterdir())==sorted([dst.name,sentinel.name])

@pytest.mark.parametrize('suffix',['.stl','.3mf','.csv'])
@pytest.mark.parametrize('existing',[False,True])
def test_replace_failure_preserves_all_originals(tmp_path,fixture,monkeypatch,suffix,existing):
 dst=tmp_path/('mesh'+suffix);sentinel=seed(dst,existing)
 def fail(*a):raise OSError('replace injection')
 monkeypatch.setattr(m,'atomic_replace',fail)
 with pytest.raises(OSError,match='replace injection'):export(dst,fixture)
 assert sentinel.read_bytes()==b'untouched sentinel'
 assert dst.read_bytes()==b'old destination' if existing else not dst.exists()
 assert not list(tmp_path.glob('.*.export-*'))

@pytest.mark.parametrize('kind',['unsupported','solid','route'])
def test_invalid_before_temp_creation(tmp_path,fixture,monkeypatch,kind):
 dst=tmp_path/('mesh.xyz' if kind=='unsupported' else 'mesh.stl' if kind=='solid' else 'mesh.csv');sentinel=seed(dst)
 if kind=='solid':fixture[0].faces=np.array([[0,0,1]])
 if kind=='route':fixture[1].route_edges=np.array([0,0,1])
 def forbidden(*a,**k):pytest.fail('temporary created before validation')
 monkeypatch.setattr(m.tempfile,'mkstemp',forbidden)
 with pytest.raises(ValueError):export(dst,fixture)
 assert dst.read_bytes()==b'old destination' and sentinel.read_bytes()==b'untouched sentinel'

@pytest.mark.parametrize('suffix',['.stl','.3mf','.csv'])
@pytest.mark.parametrize('kind',['symlink','directory','special'])
def test_unsafe_target_rejected(tmp_path,fixture,suffix,kind):
 dst=tmp_path/('mesh'+suffix);sentinel=seed(dst,False)
 if kind=='symlink':
  real=tmp_path/'referent';real.write_bytes(b'original');dst.symlink_to(real)
 elif kind=='directory':dst.mkdir()
 else:dst.write_bytes(b'original');dst.chmod(0o1640)
 mode=dst.lstat().st_mode
 with pytest.raises(ValueError):export(dst,fixture)
 assert dst.lstat().st_mode==mode and sentinel.read_bytes()==b'untouched sentinel'
 if kind=='symlink':assert real.read_bytes()==b'original'
 if kind=='special':assert dst.read_bytes()==b'original'
 assert not list(tmp_path.glob('.*.export-*'))

@pytest.mark.parametrize('suffix',['.stl','.3mf','.csv'])
def test_write_failure_preserves_target(tmp_path,fixture,monkeypatch,suffix):
 dst=tmp_path/('mesh'+suffix);sentinel=seed(dst)
 if suffix=='.3mf':
  def fail(*a,**k):raise OSError('write injection')
  monkeypatch.setattr(m.zipfile.ZipFile,'writestr',fail)
 else:
  old=Path.open
  def fail(path,*a,**k):
   if '.export-' in path.name:raise OSError('write injection')
   return old(path,*a,**k)
  monkeypatch.setattr(Path,'open',fail)
 with pytest.raises(OSError,match='write injection'):export(dst,fixture)
 assert dst.read_bytes()==b'old destination' and sentinel.read_bytes()==b'untouched sentinel'
 assert not list(tmp_path.glob('.*.export-*'))

@pytest.mark.parametrize('commit',[False,True])
def test_cleanup_failure_cannot_override_outcome(tmp_path,fixture,monkeypatch,commit):
 dst=tmp_path/'mesh.stl';sentinel=seed(dst);old=Path.unlink
 def fail(path,*a,**k):
  if '.export-' in path.name:raise OSError('cleanup injection')
  return old(path,*a,**k)
 monkeypatch.setattr(Path,'unlink',fail)
 if not commit:
  def replacefail(*a):raise OSError('original replace failure')
  monkeypatch.setattr(m,'atomic_replace',replacefail)
 with warnings.catch_warnings():
  warnings.simplefilter('error')
  if commit:export(dst,fixture);assert dst.read_bytes()!=b'old destination'
  else:
   with pytest.raises(OSError,match='original replace failure'):export(dst,fixture)
   assert dst.read_bytes()==b'old destination'
 assert sentinel.read_bytes()==b'untouched sentinel'

@pytest.mark.parametrize('suffix,expected',[
 ('.stl','0244da7c08a749f73027ec4be0daaa1bb215d3c49cea648fb66af35dbfe29531'),
 ('.csv','1b6f2fdc8eba8e974eadcd06b0af67abd40d3249d6608cdbd419edf707c8b650')])
def test_original_format_bytes(tmp_path,fixture,suffix,expected):
 import hashlib
 dst=tmp_path/('mesh'+suffix);export(dst,fixture)
 assert hashlib.sha256(dst.read_bytes()).hexdigest()==expected

def test_original_3mf_member_bytes(tmp_path,fixture):
 import hashlib
 expected={'[Content_Types].xml':'b64b4f60185ce8fed5251ed59ebb225bdc1ad22bcd23a580656d60cb0cdf004d',
 '_rels/.rels':'16320c5e9ae02d96c5be0452dda9c06b74425de791265c66dd4e8618cb2a43d7',
 '3D/3dmodel.model':'f73538c8878efcf8756c1d8457739743d5f4e6980e90dda303cea9eb7dfc81d0'}
 dst=tmp_path/'mesh.3mf';export(dst,fixture)
 with zipfile.ZipFile(dst) as z:
  assert {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}==expected

def test_missing_parent_is_not_created(tmp_path,fixture):
 dst=tmp_path/'missing'/'mesh.stl'
 with pytest.raises(FileNotFoundError):export(dst,fixture)
 assert not dst.parent.exists()

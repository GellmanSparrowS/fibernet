"""One-way Tet4 VTU diagnostics: alignment, analytical recovery, atomic staging."""
import errno
from hashlib import sha256
from itertools import combinations, permutations, product
import json
import subprocess
import sys
from types import SimpleNamespace
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pytest

from fibernet.io import tet4_results as io
from fibernet.io.tet4_meshio import Tet4Bundle, read_tet4_vtu_bundle


def fitted_patch():
    """Split 10-mm cube at z=5 mm; 12 tetrahedra and compatible interface traction."""
    xr = list(product(range(2), range(2), range(3)))
    rows = {key:i for i,key in enumerate(xr)}
    xyz = np.array(xr, dtype=float)*[10.,10.,5.]
    tets = []
    for iz in (0,1):
        for axes in permutations(range(3)):
            point = [0,0,iz]
            tet = [rows[tuple(point)]]
            for axis in axes:
                point[axis] += 1
                tet.append(rows[tuple(point)])
            tets.append(tet)
    # Fixed, compatible piecewise-affine field, nu=.25, E=1/2 GPa.
    # sigma*z is continuous, with signed xz and yz shear. Other stresses may jump.
    stress = 1e6*np.array([[[4/3,0,.4],[0,8/15,-.2],[.4,-.2,.8]],
                          [[2.4,0,.4],[0,.8,-.2],[.4,-.2,.8]]])
    grad = np.array([[[.001,0,.001],[0,0,-.0005],[0,0,1/3000]],
                     [[.001,0,.0005],[0,0,-.00025],[0,0,0]]])
    strain = .5*(grad+grad.transpose(0,2,1))
    phase = np.repeat(np.arange(2),6)
    faces = {}
    for tet, k in zip(tets,phase):
        for face in combinations(tet,3):
            opposite = next(i for i in tet if i not in face)
            faces.setdefault(tuple(sorted(face)),[]).append((k,opposite))
    xyz_m = xyz*1e-3
    f = np.zeros_like(xyz_m)
    for face, owners in faces.items():
        if len(owners) != 1:
            continue
        k,opp = owners[0]
        a,b,c = xyz_m[list(face)]
        area = .5*np.cross(b-a,c-a)
        if area@(xyz_m[opp]-a)>0:
            area=-area
        f[list(face)] += stress[k]@area/3
    x,y,z = xyz.T
    lo,hi = np.minimum(z,5), np.maximum(z-5,0)
    u = 1e-3*np.column_stack([.001*x+.001*lo+.0005*hi,-.0005*lo-.00025*hi,lo/3000])
    node_ids = np.uint64(2**64-1)-np.arange(len(xr),dtype=np.uint64)*71
    cell_ids = np.uint64(2**64-1)-np.arange(len(tets),dtype=np.uint64)*39
    mids = np.array([9,2**64-1],dtype=np.uint64)
    order = np.arange(len(xr))[::-1]
    cell_order = np.r_[np.arange(0,12,2),np.arange(1,12,2)]
    bundle = Tet4Bundle(xyz[order], np.argsort(order)[np.asarray(tets)[cell_order,::-1]],
                       node_ids[order], cell_ids[cell_order], mids[phase[cell_order]],
                       {int(p):{'young_modulus_pa':v,'poisson_ratio':.25}
                        for p,v in zip(mids,[1e9,2e9])}, 'mm')
    held = {int(node_ids[rows[key]]):axes for key,axes in (
        ((0,0,0),('x','y','z')), ((1,0,0),('y','z')), ((0,1,0),('z',)))}
    forces = {int(k):v for k,v in zip(node_ids,f) if np.any(v)}
    return bundle, forces, held, u[order], strain[phase[cell_order]], stress[phase[cell_order]]


def test_meshio_real_fitted_normal_shear_roundtrip_and_v1_refusal(tmp_path, monkeypatch):
    meshio = pytest.importorskip('meshio')
    b, forces, holds, u, eps, stress = fitted_patch()
    points = b.points*1e-3
    conn, ids, cids, mids = (a.copy() for a in
        (b.tetrahedra,b.node_ids,b.cell_ids,b.material_ids))
    properties = {k:v.copy() for k,v in b.materials.items()}
    original_validate = io._validate_static_mesh
    calls = []
    def validate_once_and_mangle_caller(bundle):
        snapshot = original_validate(bundle)
        calls.append(snapshot)
        # Prove that post-validation geometry, ordering, and material data
        # cannot be sourced from the caller again, either for solve or output.
        bundle.points[:] = 17.
        bundle.tetrahedra[:] = 0
        bundle.node_ids[:] = bundle.cell_ids[:] = bundle.material_ids[:] = 33
        bundle.materials.clear()
        return snapshot
    monkeypatch.setattr(io,'_validate_static_mesh',validate_once_and_mangle_caller)
    path = tmp_path/'response.vtu'
    result = io.solve_static_tet4_to_vtu(path,b,forces_by_id=forces,fixed_dofs_by_id=holds)
    assert len(calls)==1
    f = result.cell_fields
    assert path.is_file() and sorted(tmp_path.iterdir())==[path]
    np.testing.assert_allclose(result.displacement_m,u,atol=8e-18)
    np.testing.assert_allclose(f.strain,eps,atol=2e-14)
    np.testing.assert_allclose(f.stress_Pa,stress,atol=4e-5)
    np.testing.assert_allclose(result.strain_energy_J,.0011875,rtol=2e-11)
    reread = meshio.read(path)
    np.testing.assert_array_equal(reread.points,points)
    assert len(reread.cells)==1 and reread.cells[0].type=='tetra'
    np.testing.assert_array_equal(reread.cells[0].data,conn)
    expected_points = {'node_id':ids,'displacement_m':result.displacement_m}
    expected_cells = {'cell_id':cids,'material_id':mids,'strain':f.strain.reshape(-1,9),
                      'stress_Pa':f.stress_Pa.reshape(-1,9),'volume_m3':f.volume_m3}
    assert set(reread.point_data)==set(expected_points)
    assert set(reread.cell_data)==set(expected_cells)
    for name, expected in expected_points.items():
        data = reread.point_data[name]
        assert data.dtype==expected.dtype
        np.testing.assert_array_equal(data,expected)
    for name, expected in expected_cells.items():
        data, = reread.cell_data[name]
        assert data.dtype==expected.dtype
        np.testing.assert_array_equal(data,expected)
    tree = ET.parse(path)
    for name, kind, width in (('node_id','UInt64',1),('cell_id','UInt64',1),
                              ('material_id','UInt64',1),('displacement_m','Float64',3),
                              ('strain','Float64',9),('stress_Pa','Float64',9),
                              ('volume_m3','Float64',1)):
        da = tree.find(f'.//DataArray[@Name="{name}"]')
        assert da.attrib['type']==kind
        assert int(da.get('NumberOfComponents','1'))==width
    data=path.read_bytes()
    deliberately_not_exchange = tmp_path/'still-not-exchange.zip'
    with ZipFile(deliberately_not_exchange,'w') as archive:
        archive.writestr('mesh.vtu',data)
        archive.writestr('manifest.json',json.dumps({
            'schema':'fibernet-tet4-vtu-1','length_unit':'m','materials':properties,
            'mesh_sha256':sha256(data).hexdigest(),
        }))
    with pytest.raises(ValueError,match='unsupported VTU fields/blocks'):
        read_tet4_vtu_bundle(deliberately_not_exchange)


def test_existing_target_including_dangling_symlink_not_touched(tmp_path, monkeypatch):
    path=tmp_path/'result.vtu'
    path.write_bytes(b'existing')
    def never():
        pytest.fail('must refuse existing output before loading provider')
    monkeypatch.setattr(io,'_meshio',never)
    with pytest.raises(FileExistsError):
        io.solve_static_tet4_to_vtu(path,None,forces_by_id={},fixed_dofs_by_id={})
    assert path.read_bytes()==b'existing'
    try:
        (tmp_path/'dangling.vtu').symlink_to(tmp_path/'not-created')
    except OSError:  # Windows may not permit unprivileged symlinks.
        return
    with pytest.raises(FileExistsError):
        io.solve_static_tet4_to_vtu(tmp_path/'dangling.vtu',None,
                                  forces_by_id={},fixed_dofs_by_id={})
    assert not (tmp_path/'not-created').exists()


@pytest.mark.parametrize('failure', ['write','collision','unsupported','oversized'])
def test_staged_write_no_partial_and_no_clobber(tmp_path, monkeypatch, failure):
    # A synthetic provider drives filesystem failures without needing Meshio.
    b,forces,held,*_ = fitted_patch()
    target = tmp_path/'result.vtu'
    def write(path,*args,**kwargs):
        assert path.parent.parent==target.parent
        path.write_bytes(b'incomplete')
        if failure=='write':
            raise RuntimeError('synthetic provider failure after partial staging')
        if failure=='oversized':
            with path.open('wb') as handle:
                handle.truncate(io._MAX_RESULT_VTU_BYTES+1)
    fake = SimpleNamespace(Mesh=lambda *a,**kw:object(),vtu=SimpleNamespace(write=write))
    monkeypatch.setattr(io,'_meshio',lambda:fake)
    def link(src,dest):
        if failure=='collision':
            dest.write_bytes(b'competing writer')
            raise FileExistsError(errno.EEXIST,'competing writer',str(dest))
        if failure=='unsupported':
            raise OSError(errno.ENOTSUP,'unsupported',str(dest))
        pytest.fail('incomplete or oversized stage must not be published')
    monkeypatch.setattr(io.os,'link',link)
    kind, message = {
        'write':(RuntimeError,'partial staging'),
        'collision':(FileExistsError,'competing writer'),
        'unsupported':(OSError,'supporting hard links'),
        'oversized':(ValueError,'exceeds its 4 MiB'),
    }[failure]
    with pytest.raises(kind,match=message):
        io.solve_static_tet4_to_vtu(target,b,forces_by_id=forces,fixed_dofs_by_id=held)
    if failure=='collision':
        assert target.read_bytes()==b'competing writer'
        assert sorted(tmp_path.iterdir())==[target]
    else:
        assert not list(tmp_path.iterdir())


def test_import_and_request_without_meshio(tmp_path):
    code = r"""
import importlib.abc, sys
class ForbidMeshio(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*_):
        if fullname=='meshio':
            raise ImportError('Meshio absent')
sys.meta_path.insert(0,ForbidMeshio())
from fibernet.io.tet4_results import solve_static_tet4_to_vtu
from fibernet.sim.tet4 import solve_static_tet4
assert 'meshio' not in sys.modules
try:
    solve_static_tet4_to_vtu(sys.argv[1],None,forces_by_id={},fixed_dofs_by_id={})
except ImportError as error:
    assert 'requires optional meshio' in str(error)
else:
    raise AssertionError('did not explain the missing optional dependency')
"""
    target=tmp_path/'unused.vtu'
    subprocess.run([sys.executable,'-B','-c',code,str(target)],check=True)
    assert not list(tmp_path.iterdir())


def test_relative_target_stays_in_resolved_parent_when_cwd_changes(tmp_path, monkeypatch):
    original, other = tmp_path/'original', tmp_path/'other'
    original.mkdir()
    other.mkdir()
    monkeypatch.chdir(original)
    b, forces, held, *_ = fitted_patch()
    def write(path, *args, **kwargs):
        assert path.is_absolute() and path.parent.parent == original.resolve()
        path.write_bytes(b'provider finished')
        monkeypatch.chdir(other)
    fake = SimpleNamespace(Mesh=lambda *a, **kw: object(), vtu=SimpleNamespace(write=write))
    monkeypatch.setattr(io, '_meshio', lambda: fake)
    io.solve_static_tet4_to_vtu('result.vtu', b, forces_by_id=forces, fixed_dofs_by_id=held)
    assert (original/'result.vtu').read_bytes() == b'provider finished'
    assert not list(other.iterdir())
    assert len(list(original.iterdir())) == 1

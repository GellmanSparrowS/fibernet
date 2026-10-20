"""Geometry and format-boundary regressions, not manufacturing qualification.

A binary64 solid need not survive STL's binary32 coordinate welding. The known
Lung500, Vans500, demo pyramid and three exact dense garment configurations
must fail closed at STL serialization. The exact dense Paris configuration is
refused by the contact-normal guard during construction. Explicit coarse
configurations below are separate positive controls, not repairs to those
fixtures. 3MF preserves stored coordinates and indices, not manufacturing fitness.
"""
import sys
import time
import traceback
import hashlib
import struct
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fslab.structure import StructureFactory, all_unit_keys
from fslab.manufacturing import compile_planar, compile_surface
from fslab.surface_mapping import load_obj
from fslab.print_export import build_solid, export_solid, PrintSettings
from fslab.tube_solid import tube_mesh


def check_stl_bytes(path, solid):
    """Exercise the public writer and compare the actual binary triangle soup."""
    export_solid(path, solid)
    data = path.read_bytes()
    assert len(data) == 84 + 50 * len(solid.faces)
    assert struct.unpack('<I', data[80:84])[0] == len(solid.faces)
    records = np.frombuffer(data[84:], dtype=[('normal', '<f4', (3,)),
                            ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    expected = solid.vertices[solid.faces].astype('<f4')
    assert records['vertices'].tobytes() == expected.tobytes()
    assert np.isfinite(records['normal']).all()
    assert np.all(records['attribute'] == 0)


def check_stl_refusal(path, solid):
    """A serialization refusal must not create or replace the destination."""
    sentinel = b'existing destination must survive an unsafe STL export'
    for existing in (True, False):
        if existing:
            path.write_bytes(sentinel)
        else:
            path.unlink()
        before = set(path.parent.iterdir())
        try:
            export_solid(path, solid)
        except ValueError as exc:
            assert str(exc).startswith('binary STL serialization is invalid:'), str(exc)
        else:
            raise AssertionError('unsafe binary32 STL export was accepted')
        assert set(path.parent.iterdir()) == before, 'refusal left an output or temporary file'
        if existing:
            assert path.read_bytes() == sentinel
        else:
            assert not path.exists()


def check_3mf_preservation(path, solid):
    """Read XML as binary64; preserve exact coordinates and face indices.

    Equal indices preserve the source's indexed connectivity, including any
    disconnected components. This does not prove absence of self-intersections
    or qualify a slicer or a printed part.
    """
    export_solid(path, solid)
    vertices = np.empty(solid.vertices.shape, dtype=np.float64)
    faces = np.empty(solid.faces.shape, dtype=np.int64)
    vertex_count = face_count = 0
    ns = '{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}'
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        with archive.open('3D/3dmodel.model') as stream:
            for _, element in ET.iterparse(stream, events=('end',)):
                if element.tag == ns + 'vertex':
                    assert vertex_count < len(vertices)
                    vertices[vertex_count] = [float(element.get(k)) for k in ('x', 'y', 'z')]
                    vertex_count += 1
                elif element.tag == ns + 'triangle':
                    assert face_count < len(faces)
                    faces[face_count] = [int(element.get(k)) for k in ('v1', 'v2', 'v3')]
                    face_count += 1
                elif element.tag == ns + 'model':
                    assert element.get('unit') == 'millimeter'
                element.clear()
    assert vertex_count == len(vertices) and face_count == len(faces)
    assert np.array_equal(vertices.view(np.uint64),
                          solid.vertices.astype(np.float64).view(np.uint64))
    assert np.array_equal(faces, solid.faces)


# These are exact regression inputs, not claims that an asset is always invalid.
# Different geometry or factory settings require fresh classification. Keep the
# source assets intact and exercise usable alternatives through the public loader.
ORIGINAL_REFUSAL_INPUTS = {
    'demo_sleeve.obj': ('7b1065566c8aa5685c600d8923f30c72e9702a6dcb685d3022878f6b102b755e', 360, 'stl'),
    'demo_vest.obj': ('82b54c3bb6bba7ea4fca3751e5404f0fb15339b418a7044cb424c8ffd671eeb6', 272, 'stl'),
    'reference_shirt.obj': ('21d9248b45de8c194cb71b784979c08ba9b401243ebc73ee3de0a40731ceaabb', 248, 'stl'),
    'reference_paris.obj': ('101fe6372673319996e3c5ed4966a7641ba9d251c61d90ed282de8a0900e3fb7', 576, 'contact'),
}

# Requested mesh count is a target: the public loader returns 45 quads for
# vest/sleeve/Paris and 48 for shirt in the qualified runtime. Do not silently
# substitute these configurations for the user's current structure.
COARSE_EXPORT_CONFIGS = {
    'demo_sleeve.obj': (2, .35),
    'demo_vest.obj': (1, 0.),
    'reference_shirt.obj': (1, 0.),
    'reference_paris.obj': (1, 0.),
}


def check_asset(path, minimum_spans, target_faces=None,
                n_pts_per_side=1, perturbation=0.):
    started = time.monotonic()
    refusal = ORIGINAL_REFUSAL_INPUTS.get(path.name) if target_faces is None else None
    if refusal is not None:
        assert hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == refusal[0], path.name+' fixture changed'
    v, f = load_obj(path, target_faces=target_faces)
    if refusal is not None:
        assert len(f) == refusal[1], path.name+' original quad count changed'
    assert np.isfinite(v).all() and all(len(q)==4 for q in f)
    if target_faces is None:
        if path.name in minimum_spans:
            assert np.all(np.ptp(v,axis=0)>minimum_spans[path.name]), path.name
    else:
        original, _ = load_obj(path)
        original_span, coarse_span = np.ptp(original,axis=0), np.ptp(v,axis=0)
        if path.name in minimum_spans:
            assert np.all(original_span>minimum_spans[path.name]), path.name
        assert np.isfinite(coarse_span).all() and np.all(coarse_span>0), path.name
        assert np.isfinite(original_span).all() and np.all(original_span>0), path.name
        print('[geometry27]',path.name,'coarse/source span ratios',
              (coarse_span/original_span).tolist(),'not contour-fidelity validation',flush=True)
    network = compile_surface(v,f,StructureFactory(n_pts_per_side=n_pts_per_side,
                                                  perturbation=perturbation,seed=7))
    settings = PrintSettings(width=250,depth=250,curved=True)
    if refusal is not None and refusal[2] == 'contact':
        # The current mesher cannot assign a reliable separation direction to
        # every coincident contact in this exact configuration. This does not
        # establish that every Paris model is invalid, nor permit guard bypass.
        positions, edges = network.positions.copy(), network.edges.copy()
        try:
            build_solid(network, settings)
        except ValueError as exc:
            assert str(exc) == 'ambiguous zero-thickness solid contact', str(exc)
        else:
            raise AssertionError('dense Paris contact refusal changed; investigate before reclassification')
        assert np.array_equal(network.positions, positions)
        assert np.array_equal(network.edges, edges)
        print('[geometry27]',path.name,'exact dense contact refusal; no solid exported PASS',flush=True)
        return
    solid = build_solid(network, settings)
    assert solid.validate() > 0, path.name
    packed=solid.vertices.astype(np.float32)
    _,weld=np.unique(packed,axis=0,return_inverse=True)
    triangles=weld[solid.faces]
    edges=np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1)
    _,incidence=np.unique(edges,axis=0,return_counts=True)
    t=packed[solid.faces].astype(float)
    areas=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
    if target_faces is None and (path.name in ('3_Lung_quad_500.obj', '6_vans_500.obj', 'demo_pyramid.obj') or (refusal is not None and refusal[2] == 'stl')):
        # Keep the unsafe fixture as a negative serialization regression.
        # A failed expectation here requires investigation, not a mesh edit.
        assert np.any(incidence != 2), path.name+' binary32 welding defect changed'
        if path.name == '3_Lung_quad_500.obj':
            assert np.any(areas == 0), 'Lung500 binary32 collapse changed'
        with tempfile.TemporaryDirectory() as folder:
            check_stl_refusal(Path(folder)/(path.stem+'.stl'), solid)
            check_3mf_preservation(Path(folder)/(path.stem+'.3mf'), solid)
        serialization = 'STL refused; 3MF coordinates/indices preserved; not manufacturing-qualified'
    else:
        assert np.all(incidence==2), 'STL coordinate welding broke '+path.name
        assert np.all(areas>0),path.name
        if path.name == '3_Lung_quad_200.obj' or target_faces is not None:
            # Positive control retained alongside the negative Lung500 case.
            with tempfile.TemporaryDirectory() as folder:
                check_stl_bytes(Path(folder)/(path.stem+'.stl'), solid)
                if target_faces is not None:
                    check_3mf_preservation(Path(folder)/(path.stem+'.3mf'), solid)
        serialization = 'binary32 geometry checks passed'
    assert np.all(np.ptp(solid.vertices,axis=0)<=250.001), path.name
    delta=solid.centers-settings.coordinates(network)
    assert np.allclose(delta,delta[0]), 'export changed mapped centerlines'
    assert network.health['odd']==0 and network.health['components']==1
    print('[geometry27]',path.name,'source' if target_faces is None else f'coarse target={target_faces} actual={len(f)}',len(solid.faces),'triangles',round(time.monotonic()-started,2),'s',serialization,'PASS',flush=True)


def check_assets(paths, minimum_spans):
    failures = []
    for path in paths:
        try:
            check_asset(path, minimum_spans)
        except Exception as exc:
            failure = f'{path.name}: {type(exc).__name__}: {exc}'
            failures.append(failure)
            print('[geometry27] FAIL', failure, flush=True)
            traceback.print_exc()
    if failures:
        raise AssertionError('geometry27 failed assets:\n'+'\n'.join(failures))


def main():
    for unit in all_unit_keys():
        factory = StructureFactory(unit=unit, grid_x=2, grid_y=2, n_pts_per_side=3, line_displacements=[[.02,.04]]*3)
        base = compile_planar(factory)
        perturbed = compile_planar(replace(factory, perturbation=.5))
        repeated = compile_planar(replace(factory, perturbation=.5))
        assert np.array_equal(base.edges, perturbed.edges), unit
        assert np.array_equal(base.route_nodes, perturbed.route_nodes), unit
        assert np.array_equal(perturbed.positions, repeated.positions), unit
        displacement = np.linalg.norm(base.positions-perturbed.positions, axis=1)
        assert np.any(displacement > 1e-6), unit
        assert np.any(displacement < 1e-12), unit
        spectrum = factory.effective_spectrum()
        changed = replace(factory, perturbation=.5).effective_spectrum()
        assert np.all(np.abs(changed/spectrum-1)<=.5), unit
    large = compile_planar(StructureFactory(grid_x=10, grid_y=10, n_pts_per_side=8))
    assert large.health['odd'] == 0
    vertices, faces = tube_mesh(np.array([[0.,0.,0.],[0.,0.,20.]]), [[0,1]], 1.)
    middle = vertices[(vertices[:,2] >= 0) & (vertices[:,2] <= 20)]
    assert np.all(np.linalg.norm(middle[:,:2], axis=1) <= 1.0001)
    assert np.allclose(np.ptp(vertices,axis=0), [2,2,22],atol=1e-5)
    assets = Path(__file__).resolve().parents[1]/'assets'/'obj'
    minimum_spans = {'reference_shirt.obj':[100,65,20],
                     'reference_shoe.obj':[.1,.14,.3],
                     'reference_paris.obj':[2.,6.,2.]}
    check_assets(sorted(assets.glob('*.obj')), minimum_spans)
    # Original failures remain observable above; these explicit alternate
    # configurations must actually export, rather than merely avoid an exception.
    for name, (points, perturbation) in COARSE_EXPORT_CONFIGS.items():
        check_asset(assets/name, minimum_spans, target_faces=48,
                    n_pts_per_side=points, perturbation=perturbation)
    print('[geometry27] shared parameter perturbation, fixed paths, larger grids, round fibers, named format/contact refusals, explicit coarse export controls and 250mm limits PASS')


if __name__ == '__main__':
    main()

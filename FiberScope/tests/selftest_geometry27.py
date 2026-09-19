"""Regression checks for perturbation, whole demo geometry and printed tubes."""
import sys
import time
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fslab.structure import StructureFactory, all_unit_keys
from fslab.manufacturing import compile_planar, compile_surface
from fslab.surface_mapping import load_obj
from fslab.print_export import build_solid, PrintSettings
from fslab.tube_solid import tube_mesh


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
    for path in sorted(assets.glob('*.obj')):
        started = time.monotonic()
        v, f = load_obj(path)
        assert np.isfinite(v).all() and all(len(q)==4 for q in f)
        if path.name in minimum_spans:
            assert np.all(np.ptp(v,axis=0)>minimum_spans[path.name]), path.name
        network = compile_surface(v,f,StructureFactory(n_pts_per_side=1))
        settings = PrintSettings(width=250,depth=250,curved=True)
        solid = build_solid(network, settings)
        assert solid.validate() > 0, path.name
        packed=solid.vertices.astype(np.float32)
        _,weld=np.unique(packed,axis=0,return_inverse=True)
        triangles=weld[solid.faces]
        edges=np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1)
        _,incidence=np.unique(edges,axis=0,return_counts=True)
        assert np.all(incidence==2), 'STL coordinate welding broke '+path.name
        t=packed[solid.faces].astype(float)
        assert np.all(np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)>0),path.name
        assert np.all(np.ptp(solid.vertices,axis=0)<=250.001), path.name
        delta=solid.centers-settings.coordinates(network)
        assert np.allclose(delta,delta[0]), 'export changed mapped centerlines'
        assert network.health['odd']==0 and network.health['components']==1
        print('[geometry27]',path.name,len(solid.faces),'triangles',round(time.monotonic()-started,2),'s PASS',flush=True)
    print('[geometry27] shared parameter perturbation, fixed paths, larger grids, round fibers, all bundled curved solids and 250mm limits PASS')


if __name__ == '__main__':
    main()

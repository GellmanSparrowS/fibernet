"""OBJ concavity, mixed faces, compact defaults and stable curved identities."""
import json
import sys
import tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fslab.obj_polygons import triangulate
from fslab.surface_mapping import load_obj
from fslab.manufacturing import compile_surface
from fslab.structure import StructureFactory


def main():
    points=np.array([[0,0,0],[2,0,0],[2,1,0],[1,1,0],[1,2,0],[0,2,0]],float)
    triangles=triangulate(points,[list(range(6))])
    area=np.linalg.norm(np.cross(points[triangles[:,1]]-points[triangles[:,0]],
                                points[triangles[:,2]]-points[triangles[:,0]]),axis=1).sum()/2
    assert len(triangles)==4 and np.isclose(area,3)
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'large_polygon.obj'
        angle=np.linspace(0,2*np.pi,100,endpoint=False)
        path.write_text(''.join('v %.8f %.8f 0\n'%(np.cos(a),np.sin(a)) for a in angle)
                        +'f '+' '.join(str(i+1) for i in range(100))+'\n',encoding='utf-8')
        v,f,info=load_obj(path,return_info=True,target_faces=90)
        assert 0<len(f)<=90 and all(len(face)==4 for face in f)
    root=Path(__file__).resolve().parents[1]/'assets'/'obj'
    manifest=json.loads((root/'reference26.json').read_text(encoding='utf-8'))
    for name in manifest:
        v,f=load_obj(root/name)
        assert len(f)<=1200 and (root/name).stat().st_size<100000
        factory=StructureFactory(unit='triangle',n_pts_per_side=1)
        a=compile_surface(v,f,factory)
        b=compile_surface(v,f,replace(factory,line_displacements=[[.4,-.4]]))
        assert a.topology_id==b.topology_id and np.array_equal(a.route_edges,b.route_edges)
        assert a.health['components']==1 and a.health['odd']==0
    print('[obj26] concave triangulation, 100-corner OBJ, bounded quads, three lightweight assets and curved fixed routes PASS')


if __name__=='__main__':
    main()

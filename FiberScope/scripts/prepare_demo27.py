"""Prepare regular quad demos from preserved references; resume each asset."""
from pathlib import Path
import sys
import subprocess
import json
import hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fslab.surface_mapping import corner_quads,load_obj


class DemoPreparation:
    def __init__(self):
        self.root=Path(__file__).resolve().parents[1]
        self.work=self.root/'_cache'/'demo27'
        self.work.mkdir(parents=True,exist_ok=True)
        self.output=self.root/'assets'/'obj'

    def run(self):
        from scripts.wrap_reference27 import ReferenceWrap
        import shutil
        ReferenceWrap().run()
        for name in ('reference_shirt','reference_shoe'):
            shutil.copy2(self.work/(name+'_wrapped.obj'),self.output/(name+'.obj'))
        from scripts.tower_envelope27 import TowerEnvelope
        TowerEnvelope().build(self.root/'参考'/'OBJ'/'paris.obj',self.output/'reference_paris.obj')
        # A square-based pyramid with a flat build-plate boundary, in millimetres.
        vertices=[]; triangles=[]; lookup={}
        corners=np.array([[-60,-60,0],[60,-60,0],[60,60,0],[-60,60,0]],float)
        apex=np.array([0,0,90.])
        for side in range(4):
            ids={}
            for i in range(4):
                for j in range(4-i):
                    p=corners[side]+(corners[(side+1)%4]-corners[side])*i/3+(apex-corners[side])*j/3
                    key=tuple(np.round(p,8))
                    if key not in lookup:
                        lookup[key]=len(vertices); vertices.append(p)
                    ids[i,j]=lookup[key]
            for i in range(3):
                for j in range(3-i):
                    triangles.append([ids[i,j],ids[i+1,j],ids[i,j+1]])
                    if i+j<2:
                        triangles.append([ids[i+1,j],ids[i+1,j+1],ids[i,j+1]])
        v,f=corner_quads(vertices,triangles)
        with (self.output/'demo_pyramid.obj').open('w',encoding='utf-8') as out:
            for point in v: out.write('v %.8g %.8g %.8g\n'%tuple(point))
            for face in f: out.write('f '+' '.join(str(int(x)+1) for x in face)+'\n')
        print('[demo27] pyramid',len(f),'quads',flush=True)
        manifest={}
        for source,name in [('Shirt OBJ.obj','reference_shirt.obj'),('Shoe.obj','reference_shoe.obj'),('paris.obj','reference_paris.obj')]:
            target=self.output/name
            vertices,faces=load_obj(target)
            manifest[name]=dict(source='参考/OBJ/'+source, source_sha256=hashlib.sha256((self.root/'参考'/'OBJ'/source).read_bytes()).hexdigest(), sha256=hashlib.sha256(target.read_bytes()).hexdigest(), bytes=target.stat().st_size, quad_faces=len(faces), span=np.ptp(vertices,axis=0).tolist(), source_license='not supplied', conversion='bounded silhouette wrap and offline QuadriFlow; envelope only')
        manifest['reference_paris.obj']['conversion']='regular quad envelope fitted to reference height and silhouette; four arch openings; details omitted'
        (self.output/'reference26.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    DemoPreparation().run()

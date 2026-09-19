"""Resumable real-asset manufacturing matrix: python scripts/validate_manufacturing26.py."""
from pathlib import Path
from dataclasses import replace
import hashlib
import json
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fslab.structure import StructureFactory, all_unit_keys
from fslab.surface_mapping import load_obj
from fslab.manufacturing import compile_surface
from fslab.print_export import PrintSettings, build_solid
from fslab.search_algorithms import atomic_json


class ManufacturingValidation:
    def run(self):
        root=Path(__file__).resolve().parents[1]
        report=root/'docs/validation/manufacturing26.json'
        sources=['fslab/manufacturing.py','fslab/print_export.py','fslab/structure.py','fslab/surface_mapping.py','fslab/cell_cycles.py','fslab/obj_polygons.py','assets/obj/reference26.json']
        signature=hashlib.sha256(b''.join((root/name).read_bytes() for name in sources)).hexdigest()
        previous=json.loads(report.read_text(encoding='utf-8')) if report.exists() else {}
        results=previous.get('results',{}) if previous.get('source_hash')==signature else {}
        for name in ('3_Lung_quad_500.obj','4_Heart_quad_500.obj','6_vans_500.obj','reference_shirt.obj','reference_shoe.obj','reference_paris.obj'):
            vertices,faces=load_obj(root/'assets/obj'/name)
            for unit in all_unit_keys():
                key=name+':'+unit
                if results.get(key,{}).get('pass'):
                    continue
                start=time.monotonic()
                factory=StructureFactory(unit=unit,n_pts_per_side=2)
                reference=compile_surface(vertices,faces,factory)
                deformed=compile_surface(vertices,faces,replace(factory,line_displacements=[[.6,-.4],[-.2,.5]]))
                assert reference.topology_id==deformed.topology_id
                assert np.array_equal(reference.route_edges,deformed.route_edges)
                assert reference.health['odd']==0 and reference.health['components']==1
                assert reference.health['max_degree']<=8
                results[key]=dict(pass_=True,nodes=len(reference.positions),edges=len(reference.edges),mapped_faces=reference.mapped_faces,seconds=round(time.monotonic()-start,3))
                results[key]['pass']=results[key].pop('pass_')
                atomic_json(report,dict(source_hash=signature,results=results))
                print('[manufacturing26]',key,'PASS',flush=True)
            key=name+':solid'
            if not results.get(key,{}).get('pass'):
                start=time.monotonic()
                network=compile_surface(vertices,faces,StructureFactory(unit='hexagon',n_pts_per_side=2))
                solid=build_solid(network,PrintSettings(curved=True))
                volume=solid.validate()
                assert np.ptp(solid.vertices,axis=0)[2]>2
                results[key]={'pass':True,'faces':len(solid.faces),'volume_mm3':volume,'bounds_mm':np.ptp(solid.vertices,axis=0).tolist(),'resolution_mm':solid.resolution,'seconds':round(time.monotonic()-start,3)}
                atomic_json(report,dict(source_hash=signature,results=results))
                print('[manufacturing26]',key,'PASS',flush=True)
        print('[manufacturing26]',len(results),'complete cases')


if __name__=='__main__':
    ManufacturingValidation().run()

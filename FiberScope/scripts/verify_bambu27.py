"""Verify installed Bambu reads STL/3MF and preserves geometry; never slice/send."""
from pathlib import Path
import sys
import subprocess
import json
import zipfile
import xml.etree.ElementTree as ET
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fslab.bambu import find_bambu
from fslab.surface_mapping import load_obj
from fslab.manufacturing import compile_surface
from fslab.structure import StructureFactory
from fslab.print_export import build_solid, PrintSettings, export_solid


class BambuCheck:
    def run(self, model='demo_pyramid.obj'):
        import trimesh
        root=Path(__file__).resolve().parents[1]
        executable=find_bambu()
        if not executable: raise RuntimeError('Bambu Studio not installed')
        v,f=load_obj(root/'assets'/'obj'/model)
        network=compile_surface(v,f,StructureFactory(n_pts_per_side=1))
        solid=build_solid(network,PrintSettings(width=200,depth=200,curved=True,up_axis='z' if model=='demo_pyramid.obj' else 'y'))
        records=[]
        with tempfile.TemporaryDirectory(prefix='fs_bambu27_') as temporary:
            folder=Path(temporary)
            for suffix in ('stl','3mf'):
                source=folder/('source.'+suffix)
                output=folder/('bambu_'+suffix+'.3mf')
                export_solid(source,solid)
                process=subprocess.run([executable,'--export-3mf',str(output),str(source)],
                    cwd=str(folder),capture_output=True,timeout=90)
                assert process.returncode==0 and output.is_file(),process.stderr
                result=json.loads((folder/'result.json').read_text())
                assert result['return_code']==0,result
                with zipfile.ZipFile(output) as archive:
                    meshes=[]
                    for name in archive.namelist():
                        if not name.endswith('.model'): continue
                        model_xml=ET.fromstring(archive.read(name))
                        for item in model_xml.findall('.//{*}mesh'):
                            vertices=[[float(v.attrib[k]) for k in ('x','y','z')] for v in item.findall('{*}vertices/{*}vertex')]
                            faces=[[int(t.attrib[k]) for k in ('v1','v2','v3')] for t in item.findall('{*}triangles/{*}triangle')]
                            meshes.append(trimesh.Trimesh(vertices,faces,process=True))
                    assert len(meshes)==1
                    mesh=meshes[0]
                if not mesh.is_watertight:
                    import shutil
                    shutil.copy2(output,root/'_tmp'/('bambu_failed_'+suffix+'.3mf'))
                    shutil.copy2(source,root/'_tmp'/('bambu_source_'+suffix+'.'+suffix))
                    print('mesh diagnostics',suffix,len(mesh.vertices),len(mesh.faces),mesh.area_faces.min(),flush=True)
                assert mesh.is_watertight and mesh.is_winding_consistent
                assert np.allclose(mesh.extents,np.ptp(solid.vertices,axis=0),atol=1e-3)
                records.append(dict(format=suffix,result=result,extents_mm=mesh.extents.tolist(),watertight=True))
        report=dict(model=model,executable=executable,checks=records,physical_print=False,
            cli_reference='https://github.com/bambulab/BambuStudio/wiki/Command-Line-Usage')
        (root/'docs'/'validation'/('bambu_2.7.0_'+Path(model).stem+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('[bambu27]',model,'STL and 3MF round-trip: dimensions, closed mesh and orientation PASS',flush=True)
        return report


if __name__=='__main__':
    reports=[BambuCheck().run(name) for name in ('demo_pyramid.obj','3_Lung_quad_500.obj','4_Heart_quad_500.obj','6_vans_500.obj')]
    (Path(__file__).resolve().parents[1]/'docs'/'validation'/'bambu_2.7.0.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')

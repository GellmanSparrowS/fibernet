"""Independent development-only STL/3MF reader check; requires trimesh."""
from pathlib import Path
import sys
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fslab.manufacturing import compile_planar
from fslab.structure import StructureFactory
from fslab.print_export import build_solid, export_solid


class PrintFormatCheck:
    def run(self):
        import trimesh
        network=compile_planar(StructureFactory(unit='square',grid_x=2,grid_y=2,n_pts_per_side=2,line_displacements=[[0.,.15],[0.,.15]]))
        solid=build_solid(network)
        with tempfile.TemporaryDirectory(prefix='fs_formats25_') as folder:
            for suffix in ('stl','3mf'):
                path=Path(folder)/('network.'+suffix)
                export_solid(path,solid)
                scene=trimesh.load(str(path),force='scene')
                assert len(scene.geometry)==1
                mesh=next(iter(scene.geometry.values()))
                assert mesh.is_watertight and mesh.is_winding_consistent
                assert mesh.volume>0
                assert np.allclose(mesh.extents,[100.,100.,2.],atol=1e-4)
                print('[formats25]',suffix,'independent reader: closed, oriented, one object, 100x100x2 mm PASS')


if __name__=='__main__':
    PrintFormatCheck().run()

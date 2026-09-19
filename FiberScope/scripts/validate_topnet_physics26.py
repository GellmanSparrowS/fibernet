"""Resumable finite-physics matrix for the new, explicitly versioned topology."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fslab.structure import StructureFactory,all_unit_keys
from fslab.engine2 import Engine2,Engine2Config
from fslab.search_algorithms import atomic_json


class PhysicsValidation:
    def run(self):
        root=Path(__file__).resolve().parents[1]
        source=['fslab/engine2.py','fslab/structure.py','fslab/manufacturing.py','fslab/cell_cycles.py',
                'scripts/validate_topnet_physics26.py']
        digest=hashlib.sha256(b''.join((root/name).read_bytes() for name in source)).hexdigest()
        path=root/'docs/validation/physics26.json'
        previous=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        results=previous.get('results',{}) if previous.get('source_hash')==digest else {}
        for unit in all_unit_keys():
            for amplitude in (0.,.3,.6):
                key=unit+':'+str(amplitude)
                if results.get(key,{}).get('pass'):
                    continue
                start=time.monotonic()
                graph=StructureFactory(unit=unit,grid_x=2,grid_y=2,n_pts_per_side=2,
                    line_displacements=[[amplitude,-amplitude],[-amplitude,amplitude]]).build()
                result=Engine2(graph,Engine2Config(target_stretch=1.3,num_steps=600,n_increments=12)).run()
                assert np.isfinite(result.frames_xy).all(),key
                assert np.isfinite(result.force_curve).all(),key
                assert all(np.isfinite(v).all() for v in result.energies.values()),key
                assert result.strain_levels[-1]>result.strain_levels[0],key
                results[key]={'pass':True,'nodes':graph.num_nodes,'edges':graph.num_edges,
                    'peak_force':float(result.force_curve.max()),'seconds':round(time.monotonic()-start,3)}
                atomic_json(path,dict(source_hash=digest,scope='numerical finiteness, not convergence or material calibration',results=results))
                print('[physics26]',key,'PASS',flush=True)
        print('[physics26]',len(results),'cases complete')


if __name__=='__main__':
    PhysicsValidation().run()

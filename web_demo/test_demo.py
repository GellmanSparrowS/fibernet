"""Actual CPU workflow smoke; no credentials, network services, or printer required."""
import sys,shutil,struct,csv
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import StudioDemo,create_app,MODEL_SPECS
import gradio as gr
from fslab.print_export import FiberSolid,PrintSettings

def main():
    demo=StudioDemo();state,figure,status=demo.generate('square',2,.35,7)
    other,_,_=demo.generate('ring',1,.2,9)
    assert state['directory']!=other['directory']
    smoke=None
    try:
        assert len(figure.data)==1
        state,_,curve,_=demo.simulate(state,2,True)
        assert len(curve.data[0].x)>5
        rows=demo.features(state);assert len(rows)>10
        for key in MODEL_SPECS:
            state,plot,scores,data,_=demo.train(state,12,key)
            assert len(scores)==3 and Path(data).exists()
        state,_,_,status=demo.inverse(state,4,'J')
        assert '4' in status
        state,plot,_=demo.surface(state,'金字塔')
        assert plot.data[0].type=='scatter3d'
        try:
            demo.manufacture(state,'曲面结构',2,100)
        except gr.Error as exc:
            assert isinstance(exc.__cause__,ValueError)
            assert str(exc.__cause__)=='ambiguous zero-thickness solid contact'
            assert '制造校验未通过' in str(exc)
        else:
            raise AssertionError('the unsafe inverse-result pyramid unexpectedly exported')
        assert not any((Path(state['directory'])/name).exists() for name in
                       ('FiberScope.stl','FiberScope-route.csv'))
        smoke,_,_=demo.generate('square',3,.35,7)
        smoke,_,_=demo.map_obj(smoke,'肺',None,48,2)
        assert smoke['surface_faces']==48 and smoke['up_axis']=='y'
        network=smoke['network']
        assert network.health['odd']==0 and network.health['components']==1
        stl,route,_=demo.manufacture(smoke,'曲面结构',2,250)
        data=Path(stl).read_bytes();count=struct.unpack('<I',data[80:84])[0]
        assert count>0 and len(data)==84+50*count
        records=np.frombuffer(data[84:],dtype=[('normal','<f4',(3,)),
                             ('vertices','<f4',(3,3)),('attribute','<u2')])
        assert np.isfinite(records['normal']).all() and np.all(records['attribute']==0)
        vertices,inverse=np.unique(records['vertices'].reshape(-1,3),axis=0,return_inverse=True)
        serialized=FiberSolid(vertices.astype(float),inverse.reshape(-1,3),np.zeros((0,3)),
                              .1,PrintSettings(curved=True))
        assert serialized.validate()>0
        with Path(route).open(encoding='utf-8',newline='') as stream:
            rows=list(csv.DictReader(stream))
        assert [int(row['node_id']) for row in rows]==network.route_nodes.tolist()
        assert [int(row['incoming_edge_id']) for row in rows]==[-1]+network.route_edges.tolist()
        coordinates=np.array([[float(row[key]) for key in ('x_mm','y_mm','z_mm')] for row in rows])
        assert np.isfinite(coordinates).all() and np.array_equal(coordinates[0],coordinates[-1])
        app=create_app();assert len(app.config['dependencies'])==9
        print('[web] isolated sessions, generation, stretch, features, six real physical-label models, J inverse, unsafe pyramid refusal, curved lung mesh STL/route readback and Gradio config PASS')
    finally:
        shutil.rmtree(state['directory'],ignore_errors=True);shutil.rmtree(other['directory'],ignore_errors=True)
        if smoke is not None:shutil.rmtree(smoke['directory'],ignore_errors=True)

if __name__=='__main__':main()

"""Actual CPU workflow smoke; no credentials, network services, or printer required."""
import sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import Demo,create_app,MODEL_SPECS

def main():
    demo=Demo();state,figure,status=demo.generate('square',2,.35,7)
    other,_,_=demo.generate('ring',1,.2,9)
    assert state['directory']!=other['directory']
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
        stl,route,_=demo.manufacture(state,'曲面结构',2,100)
        assert Path(stl).stat().st_size>1000 and Path(route).stat().st_size>100
        app=create_app();assert len(app.config['dependencies'])==7
        print('[web] isolated sessions, generation, stretch, features, six real physical-label models, J inverse, pyramid, STL and Gradio config PASS')
    finally:
        shutil.rmtree(state['directory'],ignore_errors=True);shutil.rmtree(other['directory'],ignore_errors=True)

if __name__=='__main__':main()

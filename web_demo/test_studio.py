"""Enhanced Studio regression: python web_demo/test_studio.py."""
from app import StudioDemo,SURFACE_MODELS,cleanup_session
import numpy as np

def main():
    demo=StudioDemo();state,_,_=demo.generate('square',2,.2,7)
    try:
        state,_,_=demo.edit(state,'mirror',[[.1,-.2,.3,0,0,.2,.1,0,0,0]])
        state,_,_,_=demo.simulate(state,2,True)
        assert demo.replay(state,50).data
        rows,plot=demo.feature_plot(state);assert len(rows)>10 and plot.data
        for model in SURFACE_MODELS:
            state,plot,status=demo.map_obj(state,model,None,48,3)
            assert np.isfinite(state['network'].positions).all()
            assert len(state['network'].route_edges)==len(state['network'].edges)
            print('[surface]',model,len(state['network'].edges),flush=True)
        for acquisition in ('diversity','committee','hybrid'):
            state,_,scores,_,_=demo.train(state,12,'ridge',acquisition,'{"alpha":1}')
            assert len(scores)==3
        print('[studio] parameter editing, replay, features, seven original models and active strategies PASS')
    finally:cleanup_session(state)

if __name__=='__main__':main()

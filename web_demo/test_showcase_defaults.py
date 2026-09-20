"""Verify showcase defaults and dense geometry. Run: python web_demo/test_showcase_defaults.py."""
from app import StudioDemo,create_app,cleanup_session
from pathlib import Path
import time,json,numpy as np

def main():
    demo=StudioDemo();state,_,_=demo.generate('square',3,.35,7)
    try:
        app=create_app();components=[x.get('props',{}) for x in app.config['components']]
        get=lambda label:next(x for x in components if x.get('label')==label)
        assert get('桌面版曲面模型')['value']=='金字塔'
        assert get('目标曲面网格数')['value']==1000 and get('目标曲面网格数')['maximum']==10000
        assert get('每段采样点数')['value']==5
        assert get('纤维直径 / mm')['value']==2 and get('纤维直径 / mm')['maximum']==8
        assert get('网络数量（每方向）')['value']==3 and get('网络数量（每方向）')['maximum']==12
        text=' '.join(str(x.get('value','')) for x in components)
        assert '桌面版为完整版本' in text and '集群仍在排队或计算' in text
        sizes=[]
        for density in (300,1000,10000):
            start=time.monotonic();state,_,_=demo.map_obj(state,'心脏',None,density,5)
            assert np.isfinite(state['network'].positions).all()
            assert len(state['network'].route_edges)==len(state['network'].edges)
            sizes.append(state['surface_faces'])
            print('[heart]',density,state['surface_faces'],len(state['network'].edges),round(time.monotonic()-start,2),flush=True)
        assert sizes[0]<sizes[1]<sizes[2]
        state,_,_=demo.map_obj(state,'金字塔',None,1000,5)
        print('[pyramid]',state['surface_faces'],len(state['network'].edges),flush=True)
        start=time.monotonic();stl,route,status,preview=demo.preview(state,'曲面结构',2,250)
        assert stl==preview and Path(stl).stat().st_size>1000
        print('[solid]',Path(stl).stat().st_size,round(time.monotonic()-start,2),flush=True)
        print('[defaults] UI, true subdivision, dense heart and default pyramid STL PASS',flush=True)
    finally:cleanup_session(state)

if __name__=='__main__':main()

"""New unit, initial welded intersections and adjustable quad resolution."""
import sys
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fslab.structure import StructureFactory,all_unit_keys
from fslab.manufacturing import compile_planar
from fslab.welding import IntersectionWelder
from fslab.surface_mapping import corner_quads,subdivide_quads
from fslab.fibernet_bridge import ensure_fibernet
from fslab.simcache import RunConfig,run_stretch,cache_tag


def main():
    assert 'ring' in all_unit_keys() and 'voronoi' not in all_unit_keys()
    factory=StructureFactory(unit='ring',grid_x=2,grid_y=2,n_pts_per_side=2)
    a=compile_planar(factory); b=compile_planar(replace(factory,perturbation=.3))
    assert a.health['odd']==0 and a.health['components']==1
    assert np.array_equal(a.route_nodes,b.route_nodes)
    ensure_fibernet()
    from fibernet.core.structure_graph import StructureGraph
    graph=StructureGraph(dimension=2)
    for p in ((0,0),(2,2),(0,2),(2,0)): graph.add_node(p,merge=False)
    graph.add_edge(0,1); graph.add_edge(2,3)
    welded=IntersectionWelder().apply(graph)
    assert len(welded.node_positions())==5
    edges=np.asarray(welded.edge_array())[:,:2].astype(int)
    assert len(edges)==4 and np.bincount(edges.ravel()).max()==4
    assert len(graph.node_positions())==4
    p=welded.node_positions(); assert np.isclose(np.linalg.norm(p[edges[:,0]]-p[edges[:,1]],axis=1).sum(),4*np.sqrt(2))
    class CrossingFactory:
        def build(self): return graph
    run=run_stretch(CrossingFactory(),RunConfig(weld_intersections=True,target_stretch=1.1,num_steps=1000,n_increments=8),use_cache=False)
    assert run.frames_xy.shape[1]==5 and np.isfinite(run.frames_xy).all()
    assert run.metadata['weld_intersections']
    assert cache_tag(factory,RunConfig())!=cache_tag(factory,RunConfig(weld_intersections=True))
    for unit in all_unit_keys():
        sample=StructureFactory(unit=unit,grid_x=2,grid_y=2,n_pts_per_side=2,perturbation=.15)
        result=run_stretch(sample,RunConfig(weld_intersections=True,target_stretch=1.1,num_steps=800,n_increments=8),use_cache=False)
        assert np.isfinite(result.frames_xy).all() and np.isfinite(result.force_curve).all(),unit
    v,f=corner_quads([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]])
    v2,f2=subdivide_quads(v,f,2)
    assert len(f2)==48 and np.allclose(np.ptp(v,axis=0),np.ptp(v2,axis=0))
    try:
        subdivide_quads(v,f,3,max_faces=100)
        raise AssertionError('subdivision budget ignored')
    except MemoryError: pass
    print('[upgrade28] ring fixed Euler path, crossing weld, finite physics, cache isolation and quad refinement PASS')


if __name__=='__main__': main()

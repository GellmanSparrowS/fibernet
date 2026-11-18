"""Subprocess-only curve-rendering regressions; no imports or writes at collection.

Input fixtures are hand-authored geometry/result records, never simulations.
The actual profile requires a genuinely absent optional Taichi provider.
"""
import argparse
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

def graph(curved=True):
    g=StructureGraph(dimension=3)
    a=g.add_node([0,0,0]);b=g.add_node([1,0,0]);c=g.add_node([2,0,0])
    ip=np.array([[.2,3,1],[.8,-2,2.]]) if curved else None
    g.add_edge(a,b,internal_points=ip)
    g.add_edge(a,b,allow_parallel=True,internal_points=None)
    g.add_edge(b,c,internal_points=np.empty((0,3)))
    d=g.to_dict(); nm={a:7,b:42,c:90};em={0:31,1:9,2:77}
    for n in d['nodes']:n['id']=nm[n['id']]
    for e in d['edges']:
        e['id']=em[e['id']];e['node_i']=nm[e['node_i']];e['node_j']=nm[e['node_j']]
    d['nodes'].reverse();d['edges'].reverse();d.pop('next_node_id');d.pop('next_edge_id')
    out=StructureGraph.from_dict(d)
    out.edges[77].internal_points=np.empty((0,3))
    return out

def result(g, moving=False):
    ids=[90,7,42]; disp=np.zeros((3,3))
    if moving:disp[2]=[.01,0,0]
    return Result(n_nodes=g.num_nodes,n_edges=g.num_edges,metadata={'node_ids':ids,'edge_ids':[9,31,77]},
        displacements=disp,deformed_positions=np.array([g.nodes[n].position for n in ids])+disp,
        positions_trajectory=[],edge_forces=np.array([2.,8.,5.]),edge_stretches=np.array([1.1,1.8,1.5]))

def oracle(g):
    out=[];owners=[]
    for eid,e in g.edges.items():
        pts=[g.nodes[e.node_i].position]
        if e.internal_points is not None:pts.extend(e.internal_points)
        pts.append(g.nodes[e.node_j].position)
        out.extend(zip(pts[:-1],pts[1:]));owners.extend([eid]*(len(pts)-1))
    return np.array(out),owners

def drawn(ax):
    return np.array([s for c in ax.collections if hasattr(c,'_segments3d') for s in c._segments3d])

class Curves(unittest.TestCase):
    def tearDown(self):
        # Close only figures created by this fixture, never parent state.
        for number in plt.get_fignums():plt.close(number)
    def assert_geometry(self,ax,g):
        expected,_=oracle(g);np.testing.assert_array_equal(drawn(ax),expected)
        for dim,limits in enumerate((ax.get_xlim(),ax.get_ylim(),ax.get_zlim())):
            self.assertLess(limits[0],expected[:,:,dim].min());self.assertGreater(limits[1],expected[:,:,dim].max())
    def test_zero_public_paths(self):
        g=graph();r=result(g);before=g.to_dict()
        for call,axes in [
            (lambda:render.render_deformation_3d(g,r),[0,1]),
            (lambda:render.render_trajectory_3d(g,[g.node_positions()]),[0]),
            (lambda:render.render_stress_3d(g,r),[0]),
            (lambda:render.render_comparison_3d(g,r),[0,1,2]),
            (lambda:render.render_multi_angle_3d(g,r,angles=[(20,30)]),[0])]:
            fig=call()
            for index in axes:self.assert_geometry([ax for ax in fig.axes if ax.name == "3d"][index],g)
            plt.close(fig)
        self.assertEqual(before,g.to_dict())
    def test_scale_zero_and_no_result(self):
        g=graph();r=result(g,True)
        fig=render.render_deformation_3d(g,r,displacement_scale=0);self.assert_geometry(fig.axes[1],g);plt.close(fig)
        r.edge_forces=None
        fig=render.render_comparison_3d(g,r,displacement_scale=0);self.assert_geometry(fig.axes[1],g);plt.close(fig)
        fig=render.render_deformation_3d(g);self.assert_geometry(fig.axes[1],g)
    def test_reject_before_figure_and_output(self):
        from unittest.mock import patch
        g=graph();r=result(g,True);dest=report.parent/'must_not_exist'/'x.png'
        calls=[lambda:render.render_deformation_3d(g,r,save_path=dest),
            lambda:render.render_trajectory_3d(g,[g.node_positions(),g.node_positions()+[.01,0,0]],save_dir=dest.parent),
            lambda:render.render_stress_3d(g,r,save_path=dest),
            lambda:render.render_comparison_3d(g,r,save_path=dest),
            lambda:render.render_comparison_3d(g,r,displacement_scale=0,save_path=dest),
            lambda:render.render_multi_angle_3d(g,r,save_path=dest)]
        with patch.object(plt,'figure',side_effect=AssertionError('figure allocated before rejection')):
            for call in calls:
                with self.assertRaisesRegex(ValueError,'interior'):call()
        self.assertFalse(dest.parent.exists())
    def test_helper_owners_and_node_colors_N_equals_E(self):
        g=graph();expected,edge_ids=oracle(g);segments,owners=render._polyline_segments(g)
        np.testing.assert_array_equal(segments,expected)
        self.assertEqual([list(g.edges)[i] for i in owners],edge_ids)
        values=np.array([0.,2.,10.]);node_row={n:i for i,n in enumerate(sorted(g.nodes))}
        for global_scale in [False,True]:
            fig=plt.figure();ax=fig.add_subplot(projection='3d');norm=Normalize(0,10)
            if global_scale:render._render_3d_colored_global(ax,g,values,norm,render.THEMES['dark'],1,20,30,'')
            else:
                render._render_3d_colored(ax,g,values,render.THEMES['dark'],1,20,30,'');norm=Normalize(.5,10)
            expected_colors=[plt.cm.viridis(norm((values[node_row[g.edges[e].node_i]]+values[node_row[g.edges[e].node_j]])/2)) for e in edge_ids]
            np.testing.assert_array_equal(ax.collections[0].get_colors(),expected_colors);self.assert_geometry(ax,g);plt.close(fig)
    def test_edge_field_order_parallel_N_equals_E(self):
        g=graph();r=result(g);_,edge_ids=oracle(g);norm=Normalize(2,8);vals={9:2,31:8,77:5}
        expected=[plt.get_cmap('RdYlBu_r')(norm(vals[e])) for e in edge_ids]
        for call,index in [(lambda:render.render_stress_3d(g,r),0),(lambda:render.render_comparison_3d(g,r),2)]:
            fig=call();colors=np.concatenate([c.get_colors() for c in [ax for ax in fig.axes if ax.name == "3d"][index].collections])
            np.testing.assert_array_equal(colors,expected);plt.close(fig)
    def test_straight_motion_and_empty_points(self):
        g=graph(False);r=result(g,True);fig=render.render_deformation_3d(g,r)
        deformed=g.copy();deformed.nodes[42].position+=np.array([.01,0,0]);np.testing.assert_array_equal(drawn(fig.axes[1]),oracle(deformed)[0]);plt.close(fig)
        fig=render.render_comparison_3d(g,r);self.assertEqual(len(drawn([ax for ax in fig.axes if ax.name == "3d"][2])),3)
    def test_motion_elsewhere_does_not_invent_curve_motion(self):
        g=graph();r=result(g);r.displacements[0]=[.1,0,0];r.deformed_positions[0]+=[.1,0,0]
        fig=render.render_deformation_3d(g,r);self.assertEqual(len(drawn(fig.axes[1])),5)

    def test_actual_class_detailed_roundtrip(self):
        if MODE != 'actual':
            self.skipTest('actual class roundtrip belongs to absent-provider profile')
        g=graph();r=result(g);r.positions_trajectory=[r.deformed_positions.copy()]
        path=report.parent/'detailed.json';r.save(str(path),detailed=True)
        restored=Result.load(str(path));self.assertIs(type(restored),Result)
        self.assertEqual(restored.metadata,r.metadata)
        for field in ('displacements','deformed_positions','edge_forces','edge_stretches'):
            np.testing.assert_array_equal(getattr(restored,field),getattr(r,field))
        np.testing.assert_array_equal(restored.positions_trajectory[0],r.positions_trajectory[0])
        fig=render.render_deformation_3d(g,restored);self.assert_geometry(fig.axes[1],g);plt.close(fig)
        fig=render.render_trajectory_3d(g,sim_result=restored);self.assert_geometry(fig.axes[0],g);plt.close(fig)

def main():
    global np,plt,StructureGraph,render,Normalize,Result,MODE,report
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('protocol','actual'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();MODE=args.mode
    args.output.mkdir(parents=True,exist_ok=True);report=args.output/'result.json'
    if MODE=='actual' and importlib.util.find_spec('taichi') is not None:
        print('SKIP: actual missing-provider profile requires genuine Taichi absence; provider not imported')
        return 77
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from fibernet.core.structure_graph import StructureGraph
    from fibernet.viz import render
    Result=SimpleNamespace
    accelerated=None
    if MODE=='actual':
        from fibernet.sim.accelerated import SimResult
        import fibernet.sim.accelerated as accelerated
        Result=SimResult
        assert accelerated.HAS_TAICHI is False
        assert accelerated._taichi_runtime is accelerated._taichi_program is accelerated._taichi_owner is None
        assert 'taichi' not in sys.modules
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Curves)
    res=unittest.TextTestRunner(verbosity=2).run(suite)
    if accelerated is not None:
        assert accelerated.HAS_TAICHI is False
        assert accelerated._taichi_runtime is accelerated._taichi_program is accelerated._taichi_owner is None
        assert 'taichi' not in sys.modules
    out={'profile':MODE,'tests':res.testsRun,'failures':len(res.failures),
         'errors':len(res.errors),'skipped':len(res.skipped),'success':res.wasSuccessful(),
         'render_origin':str(Path(render.__file__).resolve()),
         'actual_result_class':str(Result) if MODE=='actual' else None,
         'taichi_imported':'taichi' in sys.modules}
    report.write_text(json.dumps(out,indent=2))
    return 0 if res.wasSuccessful() else 1

if __name__=='__main__':
    raise SystemExit(main())

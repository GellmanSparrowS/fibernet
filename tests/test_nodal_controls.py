import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim import prepare_nodal_controls
from fibernet.ml.beam_frame_fem import BeamFrameFEM
from fibernet.sim.accelerated import SimResult


def graph(reverse=False):
    base = StructureGraph(dimension=2)
    for y in (0., 1.):
        for x in (0., 1., 2., 3.):
            base.add_node([x, y], merge=False)
    for a,b in [(0,1),(1,2),(2,3),(4,5),(5,6),(6,7),(0,4),(1,5),(2,6),(3,7),(0,5),(1,6),(2,7),(0,1)]:
        base.add_edge(a,b,radius=.03,allow_parallel=True)
    data=base.to_dict(); ids=[101,7,9001,44,8,63,300,20]; mapping=dict(enumerate(ids))
    for n in data['nodes']: n['id']=mapping[n['id']]
    for i,e in enumerate(data['edges']):
        e.update(id=100+7*i,node_i=mapping[e['node_i']],node_j=mapping[e['node_j']])
    if reverse: data['nodes'].reverse(); data['edges'].reverse()
    data['next_node_id']=10000; data['next_edge_id']=1000
    return StructureGraph.from_dict(data)

class PreparationTests(unittest.TestCase):
    def test_reorder_and_mixed_integer_ids(self):
        for reverse in (False, True):
            g=graph(reverse); before=g.to_dict()
            p=prepare_nodal_controls(g,fixed_node_ids=[np.int64(101),8],forces_by_id={np.int32(44):[1.,0.],20:[0.,-1.]},dimension=2)
            kw=p.fem_kwargs(g)
            self.assertEqual([p.node_ids[i] for i in kw['fixed_nodes']],[101,8])
            self.assertEqual(kw['forces'][p.node_ids.index(44)].tolist(),[1.,0.])
            self.assertEqual(kw['forces'][p.node_ids.index(20)].tolist(),[0.,-1.])
            self.assertEqual(g.to_dict(),before); self.assertEqual(len(p.edge_ids),14)
            json.dumps(p.to_dict(),allow_nan=False)

    def test_unknown_and_mixed_bad_ids(self):
        for ids in ([101,True],[101,'8'],[101,8.0],[101,101],[-1],[999]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                prepare_nodal_controls(graph(),fixed_node_ids=ids)
        for key in (True,'44',44.,999):
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare_nodal_controls(graph(),forces_by_id={key:[0,0,1]})

    def test_conflicts_even_empty(self):
        for kw in ({'fixed_node_ids':[],'fixed_nodes':[]},{'forces_by_id':{},'external_force':np.zeros((8,3))}):
            with self.assertRaisesRegex(ValueError,'choose'): prepare_nodal_controls(graph(),**kw)

    def test_invalid_force_vectors(self):
        for force in ([1,2],[True,False,True],[True,1.,0.],[1,np.nan,2],[1,np.inf,2],['1','2','3'],np.ma.array([1,2,3],mask=[0,1,0])):
            with self.subTest(force=str(force)), self.assertRaises(ValueError):
                prepare_nodal_controls(graph(),forces_by_id={44:force})

    def test_legacy_rows_and_detachment(self):
        g=graph(); forces=np.arange(24.).reshape(8,3); fixed=[1,3]
        p=prepare_nodal_controls(g,fixed_nodes=fixed,external_force=forces)
        kw=p.taichi_kwargs(g); np.testing.assert_array_equal(kw['external_force'],forces)
        self.assertEqual(kw['fixed_nodes'],fixed)
        fixed.append(6); forces[:]=0; kw['external_force'][:]=99; kw['fixed_nodes'].clear()
        self.assertEqual(p.fixed_rows,(1,3)); self.assertEqual(p.forces[0],(0.,1.,2.))

    def test_graph_changes_invalidate(self):
        def reorder(g): g.nodes[101]=g.nodes.pop(101)
        def material(g): next(iter(g.edges.values())).material.youngs_modulus*=2
        def geom(g): g.nodes[101].position[0]+=1e-12
        def radius(g): next(iter(g.edges.values())).radius+=1e-12
        for mutate in (reorder,material,geom,radius):
            g=graph(); p=prepare_nodal_controls(g); mutate(g)
            with self.subTest(mutate=mutate.__name__), self.assertRaisesRegex(ValueError,'prepare_nodal_controls again'): p.taichi_kwargs(g)

    def test_geometry_material_and_multiedges_preserved(self):
        g=graph(); first=next(iter(g.edges.values())); first.internal_points=np.array([[.4,.1,0.]])
        before=g.to_dict(); p=prepare_nodal_controls(g)
        self.assertEqual(before,g.to_dict()); self.assertEqual(len(p.edge_ids),14)
        first.internal_points[0,1]+=.1
        with self.assertRaisesRegex(ValueError,'graph changed'): p.taichi_kwargs(g)

    def test_invalid_rows_and_empty_graph(self):
        for rows in ([True],[-1],[8],[1.0]):
            with self.assertRaises(ValueError): prepare_nodal_controls(graph(),fixed_nodes=rows)
        with self.assertRaisesRegex(ValueError,'at least one'):
            prepare_nodal_controls(StructureGraph())
        with self.assertRaises(ValueError):
            prepare_nodal_controls(graph(),external_force=np.zeros((8,2)))

    def test_dimension(self):
        for dim in (True,2.,'3',1,4):
            with self.assertRaises(ValueError): prepare_nodal_controls(graph(),dimension=dim)
        g=graph(); p=prepare_nodal_controls(g,dimension=2)
        with self.assertRaisesRegex(ValueError,'dimension=3'): p.taichi_kwargs(g)

    def test_cpu_fem_prepare_solve_save_reload_reordered(self):
        witnesses=[]
        temporary = tempfile.TemporaryDirectory(prefix='fibernet-nodal-controls-')
        self.addCleanup(temporary.cleanup)
        out = Path(temporary.name)
        for reverse in (False,True):
            g=graph(reverse); p=prepare_nodal_controls(g,fixed_node_ids=[101,8],forces_by_id={44:[.001,0.],20:[0.,-.001]},dimension=2)
            fem=BeamFrameFEM(E=1e7,nu=.3); inputs=fem.graph_to_fem_input(g,dim=2)
            raw=fem.solve_2d(inputs['edge_index'],inputs['node_pos'],inputs['radii'],**p.fem_kwargs(g),damping=0.,deduplicate=False)
            result=fem.to_sim_result(raw,g,node_ids=p.node_ids,edge_ids=p.edge_ids)
            result.metadata['nodal_controls']=p.to_dict()
            result.metadata['example_units']={'length':'m','force':'N','youngs_modulus':'Pa','model':'linear straight 2D frame; global E=1e7; no material conversion'}
            path=out/('reversed' if reverse else 'original');path.mkdir(exist_ok=True)
            result.save(str(path/'result.json'),detailed=True); g.save_json(str(path/'graph.json'))
            loaded=SimResult.load(str(path/'result.json')); restored=StructureGraph.load_json(str(path/'graph.json'))
            self.assertEqual(set(loaded.metadata['node_ids']),set(restored.nodes))
            self.assertEqual(set(loaded.metadata['edge_ids']),set(restored.edges));self.assertEqual(len(loaded.edge_forces),14)
            self.assertGreater(loaded.max_displacement,0)
            witnesses.append({nid:loaded.displacements[i].copy() for i,nid in enumerate(loaded.metadata['node_ids'])})
        for nid in witnesses[0]: np.testing.assert_allclose(witnesses[0][nid],witnesses[1][nid],rtol=1e-10,atol=1e-13)

if __name__=='__main__':unittest.main(verbosity=2)

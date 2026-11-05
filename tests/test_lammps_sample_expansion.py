import copy
import unittest
import numpy as np
from fibernet.core.structure_graph import StructureGraph
from fibernet.core.material import Material
from fibernet.integrations.lammps_backend import (
    HarmonicBond, RunSpec, LAMMPSContractError, prepare_harmonic_beads, validate_prepared)
from fibernet.integrations.lammps_geometry import expand_edge_samples


def source():
    g = StructureGraph(dimension=3, box_size=[2, 2, 2])
    a=g.add_node([0., 0., 0.]); b=g.add_node([1., 0., 0.])
    g.add_edge(a, b, radius=.1, material=Material(name='A'),
               internal_points=np.array([[.5, 0., 0.]]), segments=8, label='source')
    g.metadata['owner']=['original']
    return g

class ExpansionTests(unittest.TestCase):
    def test_geometry_mapping_then_explicit_lammps_compilation(self):
        graph=source(); before=copy.deepcopy(graph.to_dict())
        with self.assertRaises(LAMMPSContractError):
            prepare_harmonic_beads(graph,length_unit='m',maximum_bond_length_m=1.1,
                masses_kg={0:1.,1:1.},bonds={0:HarmonicBond(2.,1.)},
                run=RunSpec(.01,1,1,((-1.,2.),(-1.,1.),(-1.,1.))))
        expanded, report=expand_edge_samples(graph,edge_ids=[0])
        self.assertEqual(report['source_edge_to_new_ids'][0], {'new_node_ids':[2], 'new_edge_ids':[1,2]})
        self.assertEqual((expanded.num_nodes,expanded.num_edges),(3,2))
        np.testing.assert_array_equal(expanded.nodes[2].position,[.5,0,0])
        self.assertEqual([e.segments for e in expanded.edges.values()],[1,1])
        self.assertTrue(all(e.internal_points is None for e in expanded.edges.values()))
        self.assertTrue(all(e.metadata == {'label':'source'} for e in expanded.edges.values()))
        self.assertIsNot(expanded.edges[1].material, graph.edges[0].material)
        expanded.metadata['owner'].append('changed')
        self.assertEqual(graph.to_dict(),before)
        prepared=prepare_harmonic_beads(expanded,length_unit='m',maximum_bond_length_m=.75,
            masses_kg={0:1.,1:1.,2:1.},bonds={1:HarmonicBond(4.,.5),2:HarmonicBond(4.,.5)},
            run=RunSpec(.01,1,1,((-1.,2.),(-1.,1.),(-1.,1.))))
        self.assertIs(validate_prepared(prepared),prepared)
        self.assertEqual(prepared.node_ids,(0,1,2)); self.assertEqual(prepared.edge_ids,(1,2))

    def test_sparse_ids_and_unselected_edge_remain_explicit(self):
        data = source().to_dict()
        data['nodes'][0]['id'], data['nodes'][1]['id'] = 10, 40
        data['edges'][0].update(id=99, node_i=10, node_j=40,
                                internal_points=[[.25, 0., 0.], [.75, 0., 0.]])
        data['next_node_id'], data['next_edge_id'] = 41, 100
        g = StructureGraph.from_dict(data)
        extra = g.add_node([0., 1., 0.]); untouched = g.add_edge(10, extra, internal_points=np.array([[0., .5, 0.]]))
        before = copy.deepcopy(g.to_dict())
        out, report = expand_edge_samples(g, edge_ids=[99])
        self.assertEqual(report['source_edge_to_new_ids'][99], {'new_node_ids':[42,43], 'new_edge_ids':[101,102,103]})
        self.assertIn(untouched,out.edges)
        np.testing.assert_array_equal(out.edges[untouched].internal_points, [[0., .5, 0.]])
        self.assertEqual(g.to_dict(),before)

    def test_shared_endpoint_reverse_selection_and_detached_copies(self):
        g=StructureGraph(dimension=3)
        n0=g.add_node([0.,0.,0.]); n1=g.add_node([1.,0.,0.]); n2=g.add_node([0.,1.,0.])
        material=Material(name='source',extra={'labels':['original']})
        e0=g.add_edge(n0,n1,radius=.1,material=material,internal_points=np.array([[.5,.25,0.]]),
                      source_info={'old_edge_id':0, 'tags':['first']})
        e1=g.add_edge(n0,n2,radius=.2,material=material,internal_points=np.array([[-.25,.5,0.]]),
                      source_info={'old_edge_id':1, 'tags':['second']})
        before=copy.deepcopy(g.to_dict())
        out, report=expand_edge_samples(g,edge_ids=[e1,e0])
        self.assertEqual(report['source_edge_to_new_ids'][e1], {'new_node_ids':[3], 'new_edge_ids':[2,3]})
        self.assertEqual(report['source_edge_to_new_ids'][e0], {'new_node_ids':[4], 'new_edge_ids':[4,5]})
        self.assertEqual([(out.edges[i].node_i,out.edges[i].node_j) for i in (2,3,4,5)],
                         [(0,3),(3,2),(0,4),(4,1)])
        np.testing.assert_array_equal(out.nodes[3].position,[-.25,.5,0.])
        np.testing.assert_array_equal(out.nodes[4].position,[.5,.25,0.])
        self.assertEqual(out.edges[2].metadata['source_info']['old_edge_id'],1)
        self.assertEqual(out.edges[4].metadata['source_info']['old_edge_id'],0)
        out.edges[2].metadata['source_info']['tags'].append('changed')
        out.edges[2].material.extra['labels'].append('changed')
        self.assertEqual(out.edges[3].metadata['source_info']['tags'],['second'])
        self.assertEqual(out.edges[3].material.extra['labels'],['original'])
        self.assertEqual(g.to_dict(),before)

    def test_rejects_ambiguous_or_invalid_without_mutation(self):
        for name in ('endpoint','masked','complex','box_masked','duplicate_selection','missing','empty','id_range'):
            with self.subTest(name=name):
                g=source(); selection=[0]
                if name=='endpoint': g.edges[0].internal_points=np.array([[0.,0.,0.]])
                if name=='masked': g.edges[0].internal_points=np.ma.array([[.5,0.,0.]],mask=False)
                if name=='complex': g.edges[0].internal_points=np.array([[.5+0j,0,0]])
                if name=='box_masked': g._box_size=np.ma.array([2.,2.,2.],mask=False)
                if name=='duplicate_selection': selection=[0,0]
                if name=='missing': selection=[5]
                if name=='empty': g.edges[0].internal_points=np.empty((0,3))
                if name=='id_range': g._next_node_id=2**31
                before_nodes=set(g.nodes); before_edges=set(g.edges)
                with self.assertRaises((LAMMPSContractError, ValueError)):
                    expand_edge_samples(g,edge_ids=selection)
                self.assertEqual(set(g.nodes),before_nodes); self.assertEqual(set(g.edges),before_edges)

if __name__ == '__main__': unittest.main()

"""Synthetic result records only; no LAMMPS provider or solver."""
import copy,hashlib,json,unittest
from types import SimpleNamespace
import numpy as np
from fibernet.core.structure_graph import StructureGraph
from fibernet.analysis.geometry_paths import compare_lammps_node_displacements as compare

def fixture():
    initial=StructureGraph(dimension=3);a=initial.add_node([1000.,0.,0.]);b=initial.add_node([2000.,0.,0.]);initial.add_edge(a,b)
    d=initial.to_dict();mapping={a:42,b:7}
    for n in d['nodes']:n['id']=mapping[n['id']]
    for e in d['edges']:e['node_i']=mapping[e['node_i']];e['node_j']=mapping[e['node_j']]
    d['nodes'].reverse();d.pop('next_node_id');initial=StructureGraph.from_dict(d)
    reference=initial.copy();reference.nodes[42].position=np.array([.5,0.,0.]);reference.nodes[7].position=np.array([1.5,0.,0.])
    digest=hashlib.sha256(json.dumps(initial.to_dict(),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    result=SimpleNamespace(metadata={'backend':'lammps','node_ids':[42,7], 'original_graph_length_unit':'um','original_graph_length_to_m':1e-6,'field_units':{'deformed_positions':'um'},'source_identity':{'graph_sha256':digest}},deformed_positions=np.array([[1100.,0.,0.],[2300.,0.,0.]]))
    return reference,initial,result

class DisplacementComparison(unittest.TestCase):
    def test_alignment_units_and_detachment(self):
        r,g,s=fixture();out=compare(r,g,s,reference_length_unit='mm',initial_length_unit='um')
        self.assertEqual(out['node_ids'],(7,42));self.assertEqual(out['units'],'m')
        np.testing.assert_allclose(out['from_initial_m'][:,0],[.0003,.0001],atol=1e-18)
        np.testing.assert_allclose(out['from_reference_m'][:,0],[.0008,.0006],atol=1e-18)
        out['from_initial_m'][0,0]=8;self.assertEqual(s.deformed_positions[1,0],2300.)
    def test_fails_closed_for_maps_units_identity_and_positions(self):
        cases=['missing','duplicate','foreign','unit','digest','masked','nonfinite','shape','record','mixed_bool']
        for case in cases:
            with self.subTest(case=case):
                r,g,s=fixture()
                if case=='missing':del s.metadata['node_ids']
                elif case=='duplicate':s.metadata['node_ids']=[42,42]
                elif case=='foreign':s.metadata['node_ids']=[42,88]
                elif case=='unit':s.metadata['field_units']['deformed_positions']='m'
                elif case=='digest':g.nodes[42].position[0]+=1
                elif case=='masked':s.deformed_positions=np.ma.array(s.deformed_positions,mask=False)
                elif case=='nonfinite':s.deformed_positions[0,0]=np.nan
                elif case=='shape':s.deformed_positions=np.ones((2,2))
                elif case=='record':r.nodes[42].node_id=9
                elif case=='mixed_bool':s.deformed_positions=[np.array([True,False,False]),[2300.,0.,0.]]
                with self.assertRaises((ValueError,TypeError)):compare(r,g,s,reference_length_unit='mm',initial_length_unit='um')
    def test_all_units_and_equal_reference_consistency(self):
        from fibernet.units import LENGTH_FACTORS
        for unit,scale in LENGTH_FACTORS.items():
            with self.subTest(unit=unit):
                _,g,s=fixture();s.metadata['original_graph_length_unit']=unit
                s.metadata['original_graph_length_to_m']=scale;s.metadata['field_units']['deformed_positions']=unit
                out=compare(g.copy(),g,s,reference_length_unit=unit,initial_length_unit=unit)
                np.testing.assert_array_equal(out['from_initial_m'],out['from_reference_m'])

    def test_elementwise_large_difference_fallback(self):
        _,g,s=fixture();g.nodes[42].position[0]=1e308;s.deformed_positions[0,0]=-1e308
        s.metadata['original_graph_length_unit']='mm';s.metadata['original_graph_length_to_m']=1e-3
        s.metadata['field_units']['deformed_positions']='mm'
        s.metadata['source_identity']['graph_sha256']=hashlib.sha256(json.dumps(g.to_dict(),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        out=compare(g,g,s,reference_length_unit='mm',initial_length_unit='mm')
        self.assertTrue(np.isfinite(out['from_initial_m']).all())
        self.assertAlmostEqual(out['from_initial_m'][1,0]/1e305,-2.)
        np.testing.assert_array_equal(out['from_initial_m'],out['from_reference_m'])

    def test_real_adapter_projection_from_synthetic_record(self):
        from fibernet.integrations import lammps_backend as b
        from fibernet.sim import _lammps_result as adapter
        r,g,_=fixture()
        p=b.prepare_harmonic_beads(g,length_unit='um',maximum_bond_length_m=.01,
            masses_kg={7:1.,42:1.},bonds={0:b.HarmonicBond(2.,.001)},
            run=b.RunSpec(1e-6,0,1,((-.01,.01),)*3))
        positions=tuple(tuple(g.nodes[i].position*1e-6) for i in p.node_ids)
        zeros=((0.,0.,0.),)*2
        frame=b.TrajectoryFrame(0,0.,p.node_ids,positions,zeros,zeros)
        thermo=b.ThermoFrame(0,0.,0.,0.,(0.,)*6)
        raw=b.LAMMPSRunResult(p.identity,(frame,),(thermo,),'')
        result=adapter.to_sim_result(p,raw,b.RuntimeSpec('cli_serial_static','/mock/python','/mock/lmp'),0.,False)
        out=compare(r,g,result,reference_length_unit='mm',initial_length_unit='um')
        np.testing.assert_allclose(out['from_initial_m'],0,atol=1e-18)
        np.testing.assert_allclose(out['from_reference_m'][:,0],[.0005,.0005],atol=1e-18)

    def test_cg_synthetic_detailed_save_load(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from fibernet.integrations import lammps_backend as b
        from fibernet.integrations._lammps_cg import CGRequest, CGAngle, MorsePair, prepare_cg
        from fibernet.sim import _lammps_result as adapter
        from fibernet.sim.accelerated import SimResult
        g=StructureGraph(dimension=3)
        for p in ((0,0,0),(1,0,0),(1,1,0),(0,0,2)):g.add_node(p)
        g.add_edge(0,1);g.add_edge(1,2)
        q=CGRequest('m',3,{n:1. for n in g.nodes},{e:b.HarmonicBond(2.,1.) for e in g.edges},
                    {9:CGAngle(0,1,2,0,1,3.)},3,{0:MorsePair(1.,2.,1.,3.)},b.RunSpec(.001,0,1,((-5.,5.),)*3))
        prepared=prepare_cg(g,q);points=tuple(tuple(g.nodes[i].position) for i in prepared.node_ids)
        zeros=((0.,0.,0.),)*4;frame=b.TrajectoryFrame(0,0.,prepared.node_ids,points,zeros,zeros)
        thermo=b.ThermoFrame(0,0.,0.,0.,(0.,)*6);raw=b.LAMMPSRunResult(prepared.identity,(frame,),(thermo,),'')
        result=adapter.to_sim_result(prepared,raw,b.RuntimeSpec('cli_serial_static','/mock/python','/mock/lmp'),0.,True)
        with TemporaryDirectory() as td:
            path=Path(td)/'detailed.json';result.save(str(path),detailed=True);loaded=SimResult.load(str(path))
            out=compare(g,g,loaded,reference_length_unit='m',initial_length_unit='m')
        self.assertEqual(out['node_ids'],(0,1,2,3));np.testing.assert_array_equal(out['from_initial_m'],np.zeros((4,3)))
        np.testing.assert_array_equal(out['from_initial_m'],out['from_reference_m'])

    def test_overflow_and_reference_id_mismatch(self):
        r,g,s=fixture();r.nodes[42].position[0]=-1e308;s.deformed_positions[0,0]=1e308
        # m-unit source produces overflow on subtraction, without touching input.
        s.metadata['field_units']['deformed_positions']='m';s.metadata['original_graph_length_unit']='m';s.metadata['original_graph_length_to_m']=1.
        with self.assertRaises(OverflowError):compare(r,g,s,reference_length_unit='m',initial_length_unit='m')
        r,g,s=fixture();r.nodes.pop(7)
        with self.assertRaises(ValueError):compare(r,g,s,reference_length_unit='mm',initial_length_unit='um')
if __name__=='__main__':unittest.main()

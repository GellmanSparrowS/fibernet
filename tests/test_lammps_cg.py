"""Source and mocked lifecycle checks; never import or execute native LAMMPS."""
from dataclasses import replace
import copy
import json
import math
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations import lammps_backend as b
from fibernet.integrations._lammps_cg import CGRequest,CGAngle,MorsePair,prepare_cg,morse_energy,MODEL
from fibernet.sim import _lammps_result as adapter
from fibernet.sim.accelerated import SimResult


def fixture(edges=True,steps=0):
    g=StructureGraph(dimension=3)
    for p in ((0,0,0),(1,0,0),(1,1,0),(0,0,2)):g.add_node(p)
    if edges:g.add_edge(0,1,n_internal=0);g.add_edge(1,2,n_internal=0)
    q=CGRequest('m',3,{n:1. for n in g.nodes},{e:b.HarmonicBond(2.,1.) for e in g.edges},
        {9:CGAngle(0,1,2,0,1,3.)} if edges else {},3,{0:MorsePair(1.,2.,1.,3.)},
        b.RunSpec(.001,steps,1,((-5.,5.),)*3))
    return g,q


def raw(p,pe=-.25):
    g=json.loads(p.graph_json);points=tuple(tuple(n['position']) for n in g['nodes'])
    frames=tuple(b.TrajectoryFrame(i,i*p.timestep_s,p.node_ids,points,((0,0,0),)*4,((1,2,3),)*4) for i in range(p.steps+1))
    thermo=tuple(b.ThermoFrame(i,pe,0,pe,(0,)*6) for i in range(p.steps+1))
    return b.LAMMPSRunResult(p.identity,frames,thermo,'')


RUNTIME=b.RuntimeSpec('cli_serial_static','/mock/python','/mock/lmp')


class CGTests(unittest.TestCase):
    def test_deterministic_regeneration_and_prefactors(self):
        g,q=fixture();p=prepare_cg(g,q)
        self.assertEqual(p,prepare_cg(g,q));self.assertEqual(p,b.validate_prepared(p))
        self.assertIn('bond_coeff 1 1 1',p.input_text)
        self.assertIn('angle_coeff 1 3',p.input_text)
        self.assertEqual(json.loads(p.contract_json)['initial_bending_energy_J'],3)
        self.assertEqual(p.manifest()['schema'],'fibernet.lammps.conservative-cg/1')
        for bad in (replace(p,data_text=p.data_text+'\n'),replace(p,steps=1)):
            with self.assertRaises(b.LAMMPSContractError):b.validate_prepared(bad)

    def test_exact_equal_masses_still_unique_types_and_pair_policy(self):
        g,q=fixture();p=prepare_cg(g,q);c=json.loads(p.contract_json)
        self.assertEqual(c['atom_to_type_id'],[[1,1],[2,2],[3,3],[4,4]])
        lines=p.input_text.splitlines()
        self.assertEqual(sum(x.startswith('pair_coeff') and x.endswith(' zero') for x in lines),10)
        self.assertEqual(sum(x.startswith('neigh_modify exclude type') for x in lines),9)
        self.assertNotIn('neigh_modify exclude type 1 4',lines)
        self.assertIn('special_bonds lj/coul 1 1 1',lines)
        self.assertNotIn('*',p.input_text)

    def test_angle_orientation_and_duplicate(self):
        g,q=fixture();p=prepare_cg(g,q)
        self.assertEqual(p,prepare_cg(g,replace(q,angles={9:CGAngle(2,1,0,1,0,3.)})))
        with self.assertRaises(b.LAMMPSContractError):prepare_cg(g,replace(q,angles={**q.angles,10:CGAngle(2,1,0,1,0,3.)}))

    def test_negative_signed_potential_and_zero_edges_roundtrip(self):
        g,q=fixture(False);p=prepare_cg(g,q)
        r=adapter.to_sim_result(p,raw(p),RUNTIME,0,True)
        self.assertEqual(r.dynamics_state['schema'],'lammps_cg_state/1');self.assertEqual(r.energy,-.25)
        self.assertEqual((r.max_force,r.max_stretch,r.mean_stretch,r.std_stretch),(0,1,1,0))
        self.assertEqual(r.edge_forces.tolist(),[])
        self.assertAlmostEqual(r.metadata['final_max_net_nodal_force_N'],math.sqrt(14))
        self.assertEqual(r.metadata['force_roles']['max_force'],'maximum_absolute_harmonic_bond_axial')
        with tempfile.TemporaryDirectory() as d:
            for detailed in (True,False):
                path=Path(d)/'state.json';r.save(path,detailed=detailed)
                self.assertEqual(SimResult.load(path).to_dict(detailed),r.to_dict(detailed))
        with self.assertRaises(ValueError):r.compute_detailed(g)
        state=r.dynamics_state.copy();state['schema']='lammps_harmonic_state/1'
        with self.assertRaises(ValueError):adapter.validate_state(state)

    def test_static_geometry_metadata_with_and_without_recorded_state(self):
        # This is one synthetic step-zero frame, not a static native evaluation.
        g, q = fixture(False, steps=0)
        p = prepare_cg(g, q)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            for record in (False, True):
                for detailed in (False, True):
                    with self.subTest(record_state=record, detailed=detailed):
                        result = adapter.to_sim_result(p, raw(p), RUNTIME, 0., record)
                        valid = result.to_dict(detailed)
                        result.save(path, detailed=detailed)
                        self.assertEqual(SimResult.load(path).to_dict(detailed), valid)
                        self.assertEqual(result.energy, -.25)
                        self.assertEqual('dynamics_state' in valid, record)
                        for mutation in ('scale', 'field_unit'):
                            changed = copy.deepcopy(result)
                            payload = copy.deepcopy(valid)
                            if mutation == 'scale':
                                changed.metadata['original_graph_length_to_m'] = .001
                                payload['original_graph_length_to_m'] = .001
                            else:
                                changed.metadata['field_units']['displacements'] = 'mm'
                                payload['field_units']['displacements'] = 'mm'
                            with self.assertRaisesRegex(ValueError, 'inconsistent LAMMPS geometry units'):
                                changed.to_dict(detailed)
                            path.write_text(json.dumps(payload), encoding='utf-8')
                            with self.assertRaisesRegex(ValueError, 'inconsistent LAMMPS geometry units'):
                                SimResult.load(path)

    def test_text_parser_preserves_signed_cg_potential(self):
        from test_lammps_backend_contract import fixture as harmonic_texts
        g, q = fixture(False, steps=2)
        p = prepare_cg(g, q)
        dump, log = harmonic_texts(p)
        lines = log.splitlines()
        for step in (0, 1, 2):
            old = f'{step} 4 1 2 3 4 5 6 7 8 9'
            self.assertEqual(lines.count(old), 1)
            lines[lines.index(old)] = f'{step} 4 -0.25 2 1.75 4 5 6 7 8 9'
        parsed = b.parse_outputs(p, dump, '\n'.join(lines)+'\n')
        self.assertEqual([frame.potential_J for frame in parsed.thermo], [-.25]*3)
        for record in (False, True):
            with self.subTest(record_state=record):
                result = adapter.to_sim_result(p, parsed, RUNTIME, 0., record)
                self.assertEqual(result.energy, -.25)

    def test_morse_shifted_energy(self):
        c=(1.,2.,1.,3.);rc=c[3]
        self.assertEqual(morse_energy(rc,c),0)
        self.assertAlmostEqual(morse_energy(rc-1e-7,c),0,places=12)
        zc=math.exp(-c[1]*(rc-c[2]));minimum=c[2]-math.log1p(-zc)/c[1]
        self.assertGreater(minimum,c[2]);self.assertLess(morse_energy(minimum,c),0)
        self.assertNotEqual(morse_energy(minimum,c),-c[0])

    def test_preparse_bounds_and_escaped_discriminator(self):
        g,q=fixture();p=prepare_cg(g,q)
        escaped=p.contract_json.replace('conservative','\\u0063onservative')
        with self.assertRaises(b.LAMMPSContractError):b.validate_prepared(replace(p,contract_json=escaped,graph_json=' '*1048577))
        with patch.object(b.json,'loads',side_effect=AssertionError('must not parse oversized contract')):
            with self.assertRaises(b.LAMMPSContractError):b.validate_prepared(replace(p,contract_json=' '*(8*1024*1024+1)))

    def test_attraction_guard(self):
        g,q=fixture()
        for c in (MorsePair(1,1,1,1.5),MorsePair(1,1,1,1+math.log(2)),MorsePair(1,1,1,1000)):
            with self.assertRaises(b.LAMMPSContractError):prepare_cg(g,replace(q,projectile_pairs={0:c}))

    def test_result_spoof_rejected(self):
        import copy
        g,q=fixture(False);p=prepare_cg(g,q);r=adapter.to_sim_result(p,raw(p),RUNTIME,0,True)
        for kind in ('model','schema','counts','missing_pairs','unknown_contract'):
            rr=copy.deepcopy(r)
            if kind == 'model':rr.metadata['model_contract']['model']='unknown'
            if kind == 'schema':rr.dynamics_state['schema']='lammps_harmonic_state/1'
            if kind == 'counts':rr.n_edges=1
            if kind == 'missing_pairs':del rr.metadata['model_contract']['projectile_pairs']
            if kind == 'unknown_contract':rr.metadata['model_contract']['unknown']=1
            with self.assertRaises(ValueError):rr.to_dict(True)

    def test_invalid_requests(self):
        g,q=fixture()
        bad=[replace(q,projectile_node_id=True),replace(q,projectile_node_id=0),
             replace(q,masses_kg={**q.masses_kg,0:True}),replace(q,masses_kg={**q.masses_kg,0:float('inf')}),
             replace(q,angles={9:CGAngle(0,1,2,1,0,3.)}),
             replace(q,projectile_pairs={3:MorsePair(1,2,1,3)}),
             replace(q,projectile_pairs={0:MorsePair(1,1e308,1,3)}),
             replace(q,run=replace(q.run,steps=1001))]
        for request in bad:
            with self.subTest(request=request),self.assertRaises((b.LAMMPSContractError,OverflowError)):prepare_cg(g,request)
        with self.assertRaises(TypeError):CGAngle(0,1,2,0,1,1,theta0=3.14)
        with self.assertRaises(TypeError):MorsePair(1,2,1,3,unknown=0)

    def test_failed_lifecycle_cancels_and_waits(self):
        g,q=fixture();events=[]
        class Task:
            result=None;error='mock failed'
            def __init__(self,*args):pass
            def start(self):events.append('start')
            def wait(self):events.append('wait');return b.TaskState.FAILED
            def cancel(self):events.append('cancel')
        with patch.object(b,'LAMMPSTask',Task),self.assertRaises(b.LAMMPSContractError):
            adapter.simulate_lammps(g,'dynamics',2.,cg_request=q,runtime=RUNTIME,workdir='unused',limits=b.ResourceLimits())
        self.assertEqual(events,['start','wait','cancel','wait'])

    def test_unified_dispatch_mock_and_conflicts(self):
        g,q=fixture();p=prepare_cg(g,q)
        class Task:
            def __init__(self,prepared,*args):self.result=raw(prepared);self.error=None
            def start(self):pass
            def wait(self):return b.TaskState.SUCCEEDED
            def cancel(self):pass
        with patch.object(b,'LAMMPSTask',Task):
            from fibernet import simulate
            r=simulate(g,backend='lammps',mode='dynamics',cg_request=q,runtime=RUNTIME,workdir='unused',limits=b.ResourceLimits(),record_state=True)
        self.assertEqual(r.dynamics_state['schema'],'lammps_cg_state/1')
        with self.assertRaises(TypeError):adapter.simulate_lammps(g,'dynamics',2.,cg_request=q,runtime=RUNTIME,workdir='unused',limits=b.ResourceLimits(),bonds=q.bonds)

if __name__ == '__main__':unittest.main()

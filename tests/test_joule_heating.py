"""Pure source/contract checks, including reuse of already accepted JSON fields.

No NumPy/SciPy imports or solvers. Normal package integration and actual thermal
acceptance are a separate bounded numerical stage. Directly runnable with
--base-root for a source overlay; installed source trees use their own root.
"""
import argparse
import ast
import copy
from dataclasses import FrozenInstanceError
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
import unittest
from _source_check_support import run_source_check

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT
TORUS = ROOT / "tests/data/joule_torus"


def load_source():
    package = types.ModuleType('_joule_source_contract')
    package.__path__ = []
    sys.modules[package.__name__] = package
    def load(name, path):
        spec = importlib.util.spec_from_file_location(package.__name__+'.'+name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    load('_roundoff', BASE/'fibernet/sim/_roundoff.py')
    return load('joule_heating', ROOT/'fibernet/sim/joule_heating.py')


def graph_from_dict(data):
    return types.SimpleNamespace(
        nodes={n['id']: types.SimpleNamespace(position=list(n['position'])) for n in data['nodes']},
        edges={e['id']: types.SimpleNamespace(node_i=e['node_i'], node_j=e['node_j']) for e in data['edges']})


def fixture(voltage=2., conductances=(1., 3., 0.)):
    graph = graph_from_dict({'nodes':[{'id':101,'position':[0,0,0]},{'id':907,'position':[1,0,0]}],
        'edges':[{'id':11,'node_i':101,'node_j':907},{'id':31,'node_i':907,'node_j':101},{'id':67,'node_i':101,'node_j':907}]})
    ids=[101,907];eids=[11,31,67];ends=[[101,907],[907,101],[101,907]]
    volts=[float(voltage),0.];g=list(map(float,conductances));bc={'101':float(voltage),'907':0.}
    differences=[voltage,-voltage,voltage];current=[a*b if a else 0. for a,b in zip(g,differences)]
    power=[a*b for a,b in zip(current,differences)];injection=[math.fsum([current[0],-current[1],current[2]])]
    injection.append(-injection[0]);total=math.fsum(power)
    binding={'node_ids':ids,'edge_ids':eids,'endpoints':ends,'conductance_S':g,'prescribed_voltage_V':[(101,float(voltage)),(907,0.)],
        'coordinates':[[0.,0.,0.],[1.,0.,0.]],'length_unit':'m'}
    data={'schema':'fibernet-linear-conductance-1','status':'converged','node_ids':ids,'edge_ids':eids,'endpoints_by_id':ends,
        'conductance_S':g,'voltage_V':volts,'current_A':current,'node_injection_A':injection,'edge_power_W':power,
        'prescribed_voltage_V':bc,'free_node_residual_A':0.,'global_current_balance_A':0.,'dissipated_power_W':total,'boundary_power_W':total,
        'provenance':{'validation_contract':'transport-roundoff-2','model_kind':'steady-linear-electrical-conductance',
        'geometry_conversion':'none','length_unit':'m','units':{'conductance':'S','voltage':'V','current':'A','power':'W'},
        'input_sha256':hashlib.sha256(json.dumps(binding,sort_keys=True,allow_nan=False).encode()).hexdigest(),
        'tolerances':{'rtol':1e-10,'current_atol_A':0.,'power_atol_W':0.},'achieved_limits':{'node_current_A':[1e-12,1e-12]}}}
    weights={e:{101:.25,907:.75} for e in eids}
    return graph,data,weights


class JouleSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api=load_source()

    def run_load(self, graph=None, data=None, weights=None, external=None, length='m'):
        original=fixture()
        return self.api._assemble_payload(original[0] if graph is None else graph,
            original[1] if data is None else data,endpoint_weights=original[2] if weights is None else weights,
            external_source_W=external,length_unit=length)

    def test_16W_fixture_and_ledgers(self):
        result=self.run_load(external={101:.5,907:2.})
        self.assertEqual(result.edge_power_W,(4.,12.,0.))
        self.assertEqual(result.endpoint_heat_W,((1.,3.),(9.,3.),(0.,0.)))
        self.assertEqual(result.joule_source_W,{101:4.,907:12.})
        self.assertEqual(result.total_source_W,{101:4.5,907:14.})
        self.assertEqual(result.total_joule_power_W,16.)
        self.assertEqual(result.global_partition_residual_W,0.)
        self.assertEqual(result.total_source_balance_residual_W,0.)
        ledger=json.loads(result.component_ledger_json)
        self.assertEqual(ledger[0]['component_kind'],'positive-electrical-conductance')
        self.assertEqual(ledger[0]['assigned_node_heat_W'],16.)

    def test_detached_immutable_and_JSON_copies(self):
        graph,data,weights=fixture();external={101:.5,907:2.}
        result=self.run_load(graph,data,weights,external)
        weights[11][101]=0.;data['edge_power_W'][0]=99.;external[101]=9.
        result.total_source_W[101]=999.
        exported=result.to_dict();exported['provenance']['units']['node_source']='wrong'
        self.assertEqual(result.total_source_W,{101:4.5,907:14.})
        self.assertEqual(result.endpoint_weights[0],(.25,.75))
        self.assertEqual(result.to_dict()['provenance']['units']['node_source'],'W')
        with self.assertRaises(FrozenInstanceError):result.total_joule_power_W=5.
        json.dumps(result.to_dict(),allow_nan=False)

    def test_record_permutation_by_identity(self):
        graph,data,weights=fixture();original=self.run_load(graph,data,weights)
        graph.nodes=dict(reversed(list(graph.nodes.items())))
        graph.edges=dict(reversed(list(graph.edges.items())))
        reordered=self.run_load(graph,data,dict(reversed(list(weights.items()))))
        self.assertEqual(original,reordered)

    def test_explicit_equal_lumping_and_inactive_edges(self):
        graph,data,weights=fixture(conductances=(0.,0.,0.))
        weights={e:{101:.5,907:.5} for e in weights}
        result=self.run_load(graph,data,weights,{101:1.,907:2.})
        self.assertEqual(result.joule_source_W,{101:0.,907:0.})
        self.assertEqual(result.total_source_W,{101:1.,907:2.})
        self.assertEqual(len(json.loads(result.component_ledger_json)),2)
        self.assertTrue(all(v==0. for pair in result.endpoint_heat_W for v in pair))

    def test_weight_tolerance_is_visible_without_normalization(self):
        graph,data,weights=fixture()
        for value in weights.values():value[907]+=(8*sys.float_info.epsilon)
        result=self.run_load(graph,data,weights)
        self.assertEqual(result.endpoint_weights[0][1],weights[11][907])
        self.assertGreater(result.edge_weight_sum_error[0],0.)
        self.assertLessEqual(abs(result.global_partition_residual_W),result.global_partition_bound_W)
        self.assertFalse(json.loads(result.provenance_json)['weights_normalized'])

    def test_weight_refusals(self):
        for kind in ('missing_edge','unknown_edge','missing_endpoint','unknown_endpoint','negative','over_one','bad_sum','bool','nan','string_id'):
            with self.subTest(kind=kind):
                graph,data,w=fixture()
                if kind=='missing_edge':del w[11]
                if kind=='unknown_edge':w[999]={101:.25,907:.75}
                if kind=='missing_endpoint':del w[11][907]
                if kind=='unknown_endpoint':w[11]={101:.25,999:.75}
                if kind=='negative':w[11]={101:-.25,907:1.25}
                if kind=='over_one':w[11]={101:1.+sys.float_info.epsilon,907:0.}
                if kind=='bad_sum':w[11]={101:.25,907:.75+9*sys.float_info.epsilon}
                if kind=='bool':w[11]={101:True,907:0.}
                if kind=='nan':w[11]={101:float('nan'),907:.75}
                if kind=='string_id':w['11']=w.pop(11)
                with self.assertRaises(ValueError):self.run_load(graph,data,w)

    def test_external_source_refusals(self):
        for external in ({101:1.},{101:1.,907:-1.},{101:1.,907:float('inf')},{'101':1.,907:2.}):
            with self.subTest(external=external):
                with self.assertRaises(ValueError):self.run_load(external=external)
        with self.assertRaisesRegex(ArithmeticError,'lost'):
            self.run_load(external={101:1e300,907:0.})

    def test_fingerprint_and_field_refusals(self):
        for kind in ('geometry','endpoint','conductance','boundary','fingerprint','unit','current','power','negative_power','injection','status','boundary_total'):
            with self.subTest(kind=kind):
                graph,data,weights=fixture();length='m'
                if kind=='geometry':graph.nodes[101].position[0]=.1
                if kind=='endpoint':graph.edges[11].node_i=907
                if kind=='conductance':data['conductance_S'][0]=2.
                if kind=='boundary':data['prescribed_voltage_V']['101']=3.
                if kind=='fingerprint':data['provenance']['input_sha256']='0'*64
                if kind=='unit':length='mm'
                if kind=='current':data['current_A'][0]=3.
                if kind=='power':data['edge_power_W'][0]=9.
                if kind=='negative_power':data['edge_power_W'][0]=-4.
                if kind=='injection':data['node_injection_A'][0]=7.
                if kind=='status':data['status']='failed'
                if kind=='boundary_total':data['boundary_power_W']=99.
                with self.assertRaises(ValueError):self.run_load(graph,data,weights,length=length)

    def test_underflow_allocation_refuses(self):
        graph,data,weights=fixture(voltage=1e-50,conductances=(1e-200,3e-200,0.))
        for w in weights.values():w.update({101:1e-100,907:1.})
        with self.assertRaisesRegex(ArithmeticError,'Joule allocation'):
            self.run_load(graph,data,weights)

    def test_real_conversion_underflow_refuses(self):
        for value in (Fraction(1,10**400), Fraction(-1,10**400)):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.run_load(external={101:value,907:0.})
                graph,data,weights=fixture()
                weights[11]={101:value,907:1.}
                with self.assertRaises(ValueError):self.run_load(graph,data,weights)

    def test_public_workflow_one_thermal_delegation(self):
        graph,data,weights=fixture()
        package=self.api.__package__
        linear=types.ModuleType(package+'.linear_transport')
        class LinearTransportResult:
            def to_dict(self):return copy.deepcopy(data)
        linear.LinearTransportResult=LinearTransportResult
        thermal=types.ModuleType(package+'.transient_thermal');calls=[];marker=object()
        def thermal_stub(*args,**kwargs):calls.append((args,kwargs));return marker
        thermal.solve_transient_thermal=thermal_stub
        sys.modules[linear.__name__]=linear;sys.modules[thermal.__name__]=thermal
        C={101:2.,907:3.};G={11:.2,31:.3,67:0.};T0={101:300.,907:300.};times=[0.,.1];fixed={907:300.}
        result=self.api.run_joule_heating(graph,LinearTransportResult(),C,G,T0,times,
            endpoint_weights=weights,length_unit='m',fixed_temperature_K=fixed)
        self.assertIs(result.thermal,marker);self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][1]['source_W'],{101:4.,907:12.})
        self.assertIs(calls[0][1]['fixed_temperature_K'],fixed)
        self.assertIs(calls[0][0][2],G)
        with self.assertRaises(TypeError):self.api.assemble_joule_heating(graph,object(),endpoint_weights=weights,length_unit='m')
        self.assertEqual(len(calls),1)

    def test_reuse_accepted_torus_fields_without_solver(self):
        if TORUS is None:self.skipTest('optional existing accepted torus JSON not provided')
        graph=graph_from_dict(json.loads((TORUS/'run01/torus_graph.json').read_text(encoding='utf-8')))
        data=json.loads((TORUS/'run01/torus_electrical.json').read_text(encoding='utf-8'))
        weights={}
        for eid,e in graph.edges.items():
            a,b=sorted([e.node_i,e.node_j]);fraction=(eid%5)/4
            weights[eid]={a:fraction,b:1-fraction}
        load=self.run_load(graph,data,weights)
        self.assertEqual(len(load.node_ids),80);self.assertEqual(len(load.edge_ids),162)
        self.assertLessEqual(abs(load.global_partition_residual_W),load.global_partition_bound_W)
        self.assertAlmostEqual(sum(load.joule_source_W.values()),data['dissipated_power_W'],places=14)
        graph.edges=dict(reversed(list(graph.edges.items())));graph.nodes=dict(reversed(list(graph.nodes.items())))
        self.assertEqual(load,self.run_load(graph,data,weights))

    def test_no_scientific_imports_or_new_solver(self):
        if not getattr(sys, "_fibernet_source_check_child", False):
            run_source_check(__file__)
            return
        self.assertNotIn('numpy',sys.modules);self.assertNotIn('scipy',sys.modules)
        tree=ast.parse((ROOT/'fibernet/sim/joule_heating.py').read_text(encoding='utf-8'))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        forbidden={'spsolve','solve','inv','solve_linear_transport'}
        self.assertFalse(any(isinstance(n.func,ast.Name) and n.func.id in forbidden for n in calls))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-root',type=Path,default=ROOT)
    parser.add_argument('--accepted-torus',type=Path)
    args,rest=parser.parse_known_args();BASE=args.base_root.resolve();TORUS=args.accepted_torus.resolve() if args.accepted_torus else ROOT/'tests/data/joule_torus'
    unittest.main(argv=[sys.argv[0]]+rest,verbosity=2)

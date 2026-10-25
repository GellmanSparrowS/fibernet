"""AST-isolated event/state checks; never import FiberNet or Taichi.

The forced 3.9-signature route is a compatibility test double. Actual supported
Python 3.9 import/runtime coverage remains a separate CI requirement.
"""
from __future__ import annotations
import ast
import builtins
import copy
import dataclasses
import inspect
import hashlib
import json
from pathlib import Path
import pickle
import sys
import types
import unittest
import weakref
from _source_check_support import run_source_check
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'fibernet/sim/accelerated.py'
WANTED={'_native_axial_event_dataclass','NativeAxialObserverEvent','_emit_axial_observation',
 '_finite_real_array','_strict_state_ids','_state_units','_validated_fixed_rows',
 '_resolve_nodal_masses','_checked_kinetic_energy','_validate_mass_inputs',
 '_validate_dynamics_state','_validate_result_state','_dynamics_state_summary','SimResult',
 '_spring_coefficient_record','_spring_provenance','_validate_spring_provenance','_populate_spring_result'}

def isolated(legacy=False):
    tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
    selected=[n for n in tree.body if getattr(n,'name',None) in WANTED or
              (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and
               t.id=='_SPRING_PROVENANCE' for t in n.targets))]
    module=types.ModuleType('_taichi_event_compat_legacy' if legacy else '_taichi_event_compat_current')
    sys.modules[module.__name__]=module
    def dataclass39(*,frozen=False):
        # Deliberately rejects the post-3.9 slots keyword.
        return dataclasses.dataclass(frozen=frozen)
    module.__dict__.update(np=np,json=json,dataclass=dataclass39 if legacy else dataclasses.dataclass,
        field=dataclasses.field,sys=types.SimpleNamespace(version_info=(3,9)) if legacy else sys)
    # SimResult uses bare @dataclass, unlike the event compatibility decorator.
    if legacy:
        def dc39(cls=None,*,frozen=False):
            decorate=dataclass39(frozen=frozen)
            return decorate if cls is None else decorate(cls)
        module.dataclass=dc39
    selected.insert(0,ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0))
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(SOURCE),'exec'),module.__dict__)
    return module

class EventCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.variants=[isolated(False),isolated(True)]

    def event(self,m):
        events=[]
        x=np.array([[0.,0.,0.],[1.,0.,0.]])
        v=np.zeros_like(x)
        m._emit_axial_observation(events.append,0,.125,(17,83),(('length','m'),),x,v)
        return events[0],x,v

    def test_field_order_signature_and_slots(self):
        expected=('schema','step','time','node_ids','units','positions','velocities')
        signatures=[]
        for m in self.variants:
            c=m.NativeAxialObserverEvent;e,*_=self.event(m)
            self.assertEqual(tuple(c.__dataclass_fields__),expected)
            self.assertEqual(tuple(c.__slots__),expected)
            self.assertFalse(hasattr(e,'__dict__'))
            with self.assertRaises(TypeError):
                weakref.ref(e)
            signatures.append(str(inspect.signature(c)))
        self.assertEqual(signatures[0],signatures[1])

    def test_frozen_fields_and_forbidden_arbitrary_attributes(self):
        for m in self.variants:
            e,*_=self.event(m)
            for name in ('step','positions'):
                with self.assertRaises(dataclasses.FrozenInstanceError):setattr(e,name,90)
                with self.assertRaises(dataclasses.FrozenInstanceError):delattr(e,name)
            with self.assertRaises((dataclasses.FrozenInstanceError,TypeError,AttributeError)):e.extra=90

    def test_arrays_detached_and_immutable(self):
        for m in self.variants:
            e,x,v=self.event(m);x[:]=99;v[:]=99
            self.assertEqual(e.positions[1,0],1.)
            self.assertEqual(e.velocities.sum(),0.)
            for a in (e.positions,e.velocities):
                self.assertEqual(a.dtype,np.dtype('float64'))
                self.assertEqual(a.shape,(2,3))
                self.assertFalse(np.shares_memory(a,x));self.assertFalse(np.shares_memory(a,v))
                with self.assertRaises(ValueError):a.setflags(write=True)
                with self.assertRaises(ValueError):a[0,0]=99

    def test_repr_equality_hash_exclude_arrays(self):
        for m in self.variants:
            e,*_=self.event(m)
            other=m.NativeAxialObserverEvent(e.schema,e.step,e.time,e.node_ids,e.units,np.ones((2,3)),np.ones((2,3)))
            self.assertEqual(e,other);self.assertEqual(hash(e),hash(other))
            self.assertNotIn('positions=',repr(e));self.assertNotIn('velocities=',repr(e))
            self.assertNotEqual(e,dataclasses.replace(e,step=1))

    def test_copy_pickle_state_shape_matches_standard_slots(self):
        for m in self.variants:
            e,*_=self.event(m)
            self.assertEqual(len(e.__getstate__()),7)
            for other in (copy.copy(e),copy.deepcopy(e),pickle.loads(pickle.dumps(e))):
                self.assertEqual(e,other)
                np.testing.assert_array_equal(e.positions,other.positions)
                np.testing.assert_array_equal(e.velocities,other.velocities)
                self.assertFalse(hasattr(other,'__dict__'))
            # Reconstruction is normal dataclass behavior, not a security API;
            # copying/pickling does not promise immutable NumPy buffer flags.

    def test_callback_error_identity(self):
        for m in self.variants:
            failure=RuntimeError('callback sentinel')
            def fail(event):raise failure
            with self.assertRaises(RuntimeError) as caught:
                m._emit_axial_observation(fail,0,.125,(17,83),None,np.zeros((2,3)),np.zeros((2,3)))
            self.assertIs(caught.exception,failure)

    def test_native_imports_absent(self):
        if not getattr(sys, '_fibernet_source_check_child', False):
            run_source_check(__file__)
            return
        self.assertNotIn('taichi',sys.modules)
        self.assertFalse(any(n=='fibernet' or n.startswith('fibernet.') for n in sys.modules))

class StateContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=isolated(False)
    def state(self):
        x=[[[0.,0.,0.],[1.,0.,0.]]];v=[[[0.,0.,0.],[0.,0.,0.]]]
        final=dict(step=0,time=0.,kinetic_energy=0.,potential_energy=0.,total_energy=0.,near_cutoff=False,positions=x[0],velocities=v[0])
        return dict(schema='native_axial_state/1',representation='sampled',availability='available',
            units=dict(length='m',time='s',mass='kg',frame='world'),dt=.125,final=final,
            work_ledger=dict(status='unavailable',value=None,reason='not_collected'),cutoff_check='sampled_only',
            node_ids=[17,83],masses=[2.,3.],steps=[0],times=[0.],positions=x,velocities=v,
            kinetic_energy=[0.],potential_energy=[0.],total_energy=[0.],near_cutoff=[False])

    def test_mass_identity_units_and_fixed_row_validation(self):
        m=self.m
        np.testing.assert_array_equal(m._resolve_nodal_masses([17,83],None,{83:3.,17:2.}),[2.,3.])
        for ids,masses in [([17,83],{17:2.}),([17,83],{17:0.,83:3.}),([17,83],{17:2.,83:np.nan})]:
            with self.assertRaises(ValueError):m._resolve_nodal_masses(ids,None,masses)
        for units in ({'length':'m'},dict(length='',time='s',mass='kg',frame='world')):
            with self.assertRaises(ValueError):m._state_units(units)
        for rows in ([True],[2],[-1]):
            with self.assertRaises(ValueError):m._validated_fixed_rows(rows,2)
        self.assertEqual(m._validated_fixed_rows([1],2),[1])
        self.assertEqual(m._validated_fixed_rows([0,0],2),[0,0])

    def test_state_detachment_summary_and_actual_result(self):
        m=self.m;s=self.state();actual=m._validate_dynamics_state(s,2,[17,83])
        s['velocities'][0][0][0]=99
        self.assertEqual(actual['velocities'][0][0][0],0.)
        result=m.SimResult(n_nodes=2,energy=0.,metadata={'node_ids':[17,83]},dynamics_state=actual,
            deformed_positions=np.array(actual['final']['positions']),positions_trajectory=[np.array(actual['positions'][0])])
        detailed=result.to_dict(True);summary=result.to_dict(False)
        self.assertEqual(detailed['dynamics_state']['velocities'],actual['velocities'])
        self.assertEqual(summary['dynamics_state']['availability'],'summary_only')
        self.assertNotIn('velocities',summary['dynamics_state']['final'])
        self.assertNotIn('dynamics_state',m.SimResult().to_dict(True))

    def test_state_rejects_identity_mass_nonfinite_energy_mismatch(self):
        m=self.m
        for change in ('ids','mass','nonfinite','energy'):
            s=self.state()
            if change=='ids':s['node_ids']=[83,17]
            elif change=='mass':s['masses'][0]=0.
            elif change=='nonfinite':s['velocities'][0][0][0]=float('nan')
            else:s['total_energy'][0]=1.
            with self.assertRaises(ValueError):m._validate_dynamics_state(s,2,[17,83])

class LazyFacade(unittest.TestCase):
    def test_lazy_name_resolution_identity_and_failure(self):
        import importlib
        for relative,name in [('fibernet/__init__.py','render_graph'),('fibernet/easy.py','render_deformation'),('fibernet/viz/__init__.py','render_network_2d')]:
            tree=ast.parse((ROOT/relative).read_text(encoding='utf-8'));selected=[]
            for n in tree.body:
                if getattr(n,'name',None) in {'__getattr__','__dir__'}:selected.append(n)
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'_RENDER_EXPORTS','_LAZY_EXPORTS','_LAZY_MODULES'} for t in n.targets):selected.append(n)
            ns={'__name__':'source_facade'};exec(compile(ast.Module(body=selected,type_ignores=[]),relative,'exec'),ns)
            self.assertIn(name,ns['__dir__']())
            failure=ModuleNotFoundError('missing plot dependency')
            with patch.object(importlib,'import_module',side_effect=failure):
                with self.assertRaises(ModuleNotFoundError) as caught:ns['__getattr__'](name)
                self.assertIs(caught.exception,failure);self.assertNotIn(name,ns)
            sentinel=object()
            with patch.object(importlib,'import_module',return_value=types.SimpleNamespace(**{name:sentinel})):
                self.assertIs(ns['__getattr__'](name),sentinel);self.assertIs(ns[name],sentinel)
            with self.assertRaises(AttributeError):ns['__getattr__']('no_such_name')


# Deliberately usable on 3.9: empty newer AST bookkeeping fields are omitted.
def canonical_ast(value):
    if isinstance(value,ast.AST):
        return [type(value).__name__,[(key,canonical_ast(item)) for key,item in ast.iter_fields(value)
            if not (key=='type_params' and not item)]]
    if isinstance(value,list):return [canonical_ast(item) for item in value]
    if isinstance(value,bytes):return ['bytes',value.hex()]
    if isinstance(value,complex):return ['complex',value.real,value.imag]
    if value is Ellipsis:return ['ellipsis']
    return value

PUBLIC_MODULES=(
    'fibernet/__init__.py', 'fibernet/easy.py', 'fibernet/api.py',
    'fibernet/viz/__init__.py', 'fibernet/sim/accelerated.py',
    'fibernet/sim/_energy_accounting.py',
)

API_V2_MODULES = frozenset((PUBLIC_MODULES[0], PUBLIC_MODULES[1], PUBLIC_MODULES[2], PUBLIC_MODULES[4]))

def public_baseline(relative):
    if relative not in PUBLIC_MODULES:
        raise ValueError('unknown public source contract')
    directory = 'public_api_contract_v2' if relative in API_V2_MODULES else 'public_source_contract'
    return ROOT/'tests/fixtures'/directory/Path(relative).with_suffix('.py.txt')

# Whole-file reviewed facade versions only. No normalization of arbitrary regions.
FACADE_PINS = {
    'fibernet/__init__.py': ('root.py.txt', '5a4b85e0d6b5faa6c2008014ea766a98dd260f01ec9616fcca8856005c22028b'),
    'fibernet/sim/__init__.py': ('sim.py.txt', '7fa239798226e30c4541213bb079043b6fdedff3fbe7e26f53f84e2302c3caa8'),
}

def reviewed_facade(relative):
    name, digest = FACADE_PINS[relative]
    raw = (ROOT/'tests/fixtures/optional_facades_v2'/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, 'reviewed facade fixture changed'
    return raw


def prior_sim_facade():
    raw = (ROOT/'tests/fixtures/optional_facades_v2/sim_prior.py.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '0078e02e3492792774843f0b2f86a23a86d78553327191ff79dc5b2d2b0d7036', 'prior sim facade fixture changed'
    return raw


# Reviewed result-detail method only; historical API-v2 fixtures stay immutable.
RESULT_IDENTITY_METHOD_PIN = '705e63bd4dc84b86ed2cd2a85b4fc4db1050f84a69eabc34624488f324bc77e5'


def result_detail_method(raw):
    tree = ast.parse(raw.decode('utf-8'))
    results = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SimResult']
    assert len(results) == 1, 'expected one SimResult class'
    methods = [n for n in results[0].body if isinstance(n, ast.FunctionDef) and n.name == 'compute_detailed']
    assert len(methods) == 1, 'expected one compute_detailed method'
    return methods[0]


def reviewed_result_identity_method():
    raw = (ROOT/'tests/fixtures/result_identity_contract_v1/compute_detailed.py.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RESULT_IDENTITY_METHOD_PIN, 'reviewed result identity method fixture changed'
    return raw


def reviewed_result_identity_source():
    """Reconstruct only the approved method and two field-comment changes."""
    raw = public_baseline('fibernet/sim/accelerated.py').read_bytes()
    method = result_detail_method(raw)
    lines = raw.splitlines(keepends=True)
    approved = b''.join(lines[:method.lineno-1]) + reviewed_result_identity_method() + b''.join(lines[method.end_lineno:])
    for before, after in (
        (b'    edge_forces: np.ndarray = None       # per-edge axial force magnitude\n',
         b'    edge_forces: np.ndarray = None       # signed per-edge axial force\n'),
        (b'    max_force: float = 0.0              # maximum edge force\n',
         b'    max_force: float = 0.0              # maximum absolute axial edge force\n'),
    ):
        assert approved.count(before) == 1, 'historical result field comment changed'
        approved = approved.replace(before, after, 1)
    return approved


# Only the reviewed provenance delta is applied to the prior method-qualified
# source. Historical fixtures and all unrelated code retain their byte identity.
SPRING_PROVENANCE_DELTA_PIN = '3654c03ab5b13c5773b5fa8aa665e50819006ddc6a6093aad7389e230f2faf42'


def reviewed_spring_provenance_source():
    raw = (ROOT/'tests/fixtures/spring_coefficient_provenance_v1/accelerated_delta.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SPRING_PROVENANCE_DELTA_PIN, 'reviewed spring source delta changed'
    delta = json.loads(raw)
    assert delta['schema'] == 'reviewed_spring_source_delta/1', 'unsupported spring source delta'
    source = reviewed_result_identity_source()
    assert hashlib.sha256(source).hexdigest() == delta['base_sha256'], 'spring source delta base changed'
    for hunk in delta['hunks']:
        before, after = hunk['before'].encode('utf-8'), hunk['after'].encode('utf-8')
        assert before and source.count(before) == 1, 'spring source delta is not uniquely located'
        source = source.replace(before, after, 1)
    assert hashlib.sha256(source).hexdigest() == delta['result_sha256'], 'spring source delta result changed'
    return source


def normalize_result_identity_method(function):
    """Qualify the complete reviewed prefix; retain every appended statement."""
    historical = result_detail_method(public_baseline('fibernet/sim/accelerated.py').read_bytes())
    approved = result_detail_method(b'class SimResult:\n' + reviewed_result_identity_method())
    provenance = result_detail_method(reviewed_spring_provenance_source())
    for reference in (provenance, approved, historical):
        prefix = copy.deepcopy(function)
        suffix = prefix.body[len(reference.body):]
        prefix.body = prefix.body[:len(reference.body)]
        if canonical_ast(prefix) == canonical_ast(reference):
            normalized = copy.deepcopy(historical)
            normalized.body.extend(suffix)
            return normalized
    raise AssertionError('unreviewed result identity method change')


EASY_STRETCH_RATIO_PIN = 'fe9cde78f9618555c22cdac2e63c1998c2af781c970596fc192d362857b469ff'


def reviewed_easy_stretch_ratio_source():
    raw = (ROOT/'tests/fixtures/easy_stretch_ratio_v1/easy.py.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EASY_STRETCH_RATIO_PIN, 'reviewed stretch ratio source changed'
    return raw



EASY_RL_FAILURE_PIN = 'a2303068861cf198148c7cb1f6eb78f46aa35f3f8eb648fa1d72c85d430cd73e'


def reviewed_easy_rl_failure_source():
    raw = (ROOT/'tests/fixtures/easy_rl_failure_v1/easy.py.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EASY_RL_FAILURE_PIN, 'reviewed RL failure source changed'
    return raw


def reviewed_nodal_controls_facade():
    raw = (ROOT/'tests/fixtures/optional_facades_v2/sim_nodal_controls.py.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '268c34a2446a4ca46aab8c29c39053502f8d343f9b25a9f77ee2ddc66161ddc1', 'reviewed nodal controls facade changed'
    return raw


def reviewed_thermoelastic_facade():
    raw = (ROOT/'tests/fixtures/optional_facades_v2/sim_thermoelastic.py.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'a86c885e655b3833ba292bf89f76d481c849bc9c9f7a204d3d46896ac839c6aa', 'reviewed thermoelastic facade changed'
    return raw



def reviewed_fiber_network_profile_facade(reverse=None):
    """Admit only the pinned profile exports, with an exact historical inverse."""
    raw = (ROOT/'tests/fixtures/optional_facades_v2/root_fiber_network_profile_delta.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'c1d7e4c0825bad86bf19ed22d93ebefa3776f5682f0a8d1b71ba393c7b2c45be', 'reviewed FiberNetwork profile facade delta changed'
    delta = json.loads(raw)
    assert delta['schema'] == 'reviewed_fiber_network_profile_facade_delta/1'
    assert delta['source'] == 'fibernet/__init__.py'
    assert delta['baseline_main'] == 'e6b9bd31aab88f861879e79e906aea4387686e6f'
    base = reviewed_facade('fibernet/__init__.py')
    assert hashlib.sha256(base).hexdigest() == delta['base_sha256'], 'profile facade delta base changed'
    source = base if reverse is None else reverse
    if reverse is not None:
        assert hashlib.sha256(source).hexdigest() == delta['result_sha256'], 'profile facade inverse input changed'
    for hunk in (delta['hunks'] if reverse is None else reversed(delta['hunks'])):
        before, after = hunk['before'].encode('utf-8'), hunk['after'].encode('utf-8')
        if reverse is not None:
            before, after = after, before
        assert before and source.count(before) == 1, 'profile facade delta is not uniquely located'
        source = source.replace(before, after, 1)
    digest = delta['result_sha256'] if reverse is None else delta['base_sha256']
    assert hashlib.sha256(source).hexdigest() == digest, 'profile facade delta result changed'
    if reverse is not None:
        assert source == base, 'profile facade inverse changed historical bytes'
    return source


def reviewed_lammps_capability_facade(reverse=None):
    """Admit only the pinned lazy alias delta, with an exact historical inverse."""
    raw = (ROOT/'tests/fixtures/optional_facades_v2/sim_lammps_capabilities_delta.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'f0a9e038e15e8eec98c74096bb0d5262147ee71d27e9ed235fdff66537790ba7', 'reviewed LAMMPS facade delta changed'
    delta = json.loads(raw)
    assert delta['schema'] == 'reviewed_lammps_capability_facade_delta/1'
    assert delta['source'] == 'fibernet/sim/__init__.py'
    assert delta['baseline_main'] == 'cfb220026e906c34e68ec41f784a65abe3524291'
    base = reviewed_thermoelastic_facade()
    assert hashlib.sha256(base).hexdigest() == delta['base_sha256'], 'LAMMPS facade delta base changed'
    source = base if reverse is None else reverse
    if reverse is not None:
        assert hashlib.sha256(source).hexdigest() == delta['result_sha256'], 'LAMMPS facade inverse input changed'
    for hunk in (delta['hunks'] if reverse is None else reversed(delta['hunks'])):
        before, after = hunk['before'].encode('utf-8'), hunk['after'].encode('utf-8')
        if reverse is not None:
            before, after = after, before
        assert before and source.count(before) == 1, 'LAMMPS facade delta is not uniquely located'
        source = source.replace(before, after, 1)
    digest = delta['result_sha256'] if reverse is None else delta['base_sha256']
    assert hashlib.sha256(source).hexdigest() == digest, 'LAMMPS facade delta result changed'
    if reverse is not None:
        assert source == base, 'LAMMPS facade inverse changed historical bytes'
    return source


# Byte-pinned deltas from the immutable public-main source chain. Historical
# fixtures are retained verbatim; only these two exact candidate versions enter.
TAICHI_GEOMETRY_SOURCE_PINS = {
    'fibernet/easy.py': ('easy_delta.json', 'f29c4619d8ff2e5cd118aa59a21ab65ec0a43470f4032f1881136ddb4e07943c'),
    'fibernet/sim/accelerated.py': ('accelerated_delta.json', '17271606f8dfaa9deecef7446beb8b0b98cfe7d1b01af37718475c700d63afe1'),
}


def reviewed_taichi_geometry_source(relative):
    if relative not in TAICHI_GEOMETRY_SOURCE_PINS:
        raise ValueError('unknown geometry source contract')
    name, digest = TAICHI_GEOMETRY_SOURCE_PINS[relative]
    raw = (ROOT/'tests/fixtures/taichi_geometry_policy_v1'/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, 'reviewed geometry source delta changed'
    delta = json.loads(raw)
    assert delta['schema'] == 'reviewed_taichi_geometry_source_delta/1', 'unsupported geometry source delta'
    assert delta['source'] == relative, 'geometry source delta target changed'
    assert delta['baseline_main'] == '30f29eb4c574a25b7d15b8540ad2d98b9cbcef36', 'geometry source baseline changed'
    source = (reviewed_easy_rl_failure_source() if relative == 'fibernet/easy.py'
              else reviewed_spring_provenance_source())
    assert hashlib.sha256(source).hexdigest() == delta['base_sha256'], 'geometry source delta base changed'
    for hunk in delta['hunks']:
        before, after = hunk['before'].encode('utf-8'), hunk['after'].encode('utf-8')
        assert before and source.count(before) == 1, 'geometry source delta is not uniquely located'
        source = source.replace(before, after, 1)
    assert hashlib.sha256(source).hexdigest() == delta['result_sha256'], 'geometry source delta result changed'
    return source


# One exact additive saved-result method; no generic method exclusion.
SAVED_ENERGY_DIAGNOSTICS_METHOD = '    def viscoelastic_energy_diagnostics(self) -> Dict:\n        """Inspect cumulative energy residuals at saved material endpoints.\n\n        Returns detached lists/dictionaries in joules, without simulation or\n        material advancement. Residuals are observations, not a cumulative\n        consistency certificate or a solver-accuracy test. Sparse samples do\n        not reconstruct omitted intervals. Only native axial viscoelastic v1\n        results are supported; the separate elastic ledger is not interpreted.\n        """\n        from ._taichi_viscoelastic import energy_diagnostics\n        return energy_diagnostics(self)\n\n'


def reviewed_saved_energy_diagnostics_source():
    base = reviewed_taichi_geometry_source('fibernet/sim/accelerated.py')
    assert hashlib.sha256(base).hexdigest() == '12ea0357e026a6b4713e55bf4d19a786f318da00617cfb4645a95bf27d3846ba', 'saved diagnostics baseline changed'
    anchor = b'    def to_dict(self, detailed: bool = False) -> Dict:\n'
    assert base.count(anchor) == 1, 'saved diagnostics insertion is not unique'
    result = base.replace(anchor, SAVED_ENERGY_DIAGNOSTICS_METHOD.encode('utf-8') + anchor, 1)
    assert hashlib.sha256(result).hexdigest() == 'a5bdc54048c2012f4582c79e514b835c1967746276236d093b0d0ea353985e71', 'reviewed saved diagnostics source changed'
    return result


def remove_reviewed_saved_energy_diagnostics(result_node):
    node = copy.deepcopy(result_node)
    methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == 'viscoelastic_energy_diagnostics']
    assert len(methods) <= 1, 'duplicate saved diagnostics method'
    if methods:
        expected = ast.parse('class SimResult:\n' + SAVED_ENERGY_DIAGNOSTICS_METHOD).body[0].body[0]
        assert canonical_ast(methods[0]) == canonical_ast(expected), 'saved diagnostics method changed'
        node.body.remove(methods[0])
    return node


def require_public_source(relative, raw):
    if relative == 'fibernet/__init__.py':
        assert raw in (public_baseline(relative).read_bytes(), reviewed_facade(relative), reviewed_fiber_network_profile_facade()), 'public source bytes changed: '+relative
        return
    if relative == 'fibernet/sim/__init__.py':
        assert raw in (prior_sim_facade(), reviewed_facade(relative), reviewed_nodal_controls_facade(), reviewed_thermoelastic_facade(), reviewed_lammps_capability_facade()), 'public source bytes changed: '+relative
        return
    if relative == 'fibernet/easy.py':
        assert raw in (public_baseline(relative).read_bytes(), reviewed_easy_stretch_ratio_source(), reviewed_easy_rl_failure_source(), reviewed_taichi_geometry_source(relative)), 'public source bytes changed: '+relative
        return
    if relative in FACADE_PINS:
        prior = prior_sim_facade() if relative == 'fibernet/sim/__init__.py' else public_baseline(relative).read_bytes()
        assert raw in (prior, reviewed_facade(relative)), 'public source bytes changed: '+relative
        return
    expected=public_baseline(relative).read_bytes()
    if relative == 'fibernet/sim/accelerated.py':
        assert raw in (expected, reviewed_result_identity_source(), reviewed_spring_provenance_source(), reviewed_taichi_geometry_source(relative), reviewed_saved_energy_diagnostics_source()), 'public source bytes changed: '+relative
        return
    if relative in PUBLIC_MODULES[3:] or relative in API_V2_MODULES:
        assert raw==expected, 'public source bytes changed: '+relative
        return
    if raw==expected:
        return
    assert canonical_ast(ast.parse(raw.decode('utf-8')))==canonical_ast(
        ast.parse(expected.decode('utf-8'))), 'public source structure changed: '+relative

LAMMPS_PREFIXES={
    '_validate_result_state': """if isinstance(result.dynamics_state, dict) and result.dynamics_state.get("schema") in ("lammps_harmonic_state/1", "lammps_cg_state/1"):
    from ._lammps_result import validate_result
    return validate_result(result)
""",
    '_dynamics_state_summary': """if isinstance(value, dict) and value.get("schema") in ("lammps_harmonic_state/1", "lammps_cg_state/1"):
    from ._lammps_result import state_summary
    return state_summary(value)
""",
    'SimResult.compute_detailed': """if self.metadata.get('backend') == 'lammps':
    raise ValueError('LAMMPS results use explicit equilibrium lengths; geometric-rest recomputation is unsupported')
""",
}

def remove_public_lammps_branch(function, key):
    """Check the readable routing contract before removing its one prefix."""
    node=copy.deepcopy(function)
    if key == 'SimResult.compute_detailed':
        node = normalize_result_identity_method(node)
    index=1 if (node.body and isinstance(node.body[0],ast.Expr) and
                isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str)) else 0
    if key == 'SimResult.compute_detailed':
        reference=ast.parse(public_baseline('fibernet/sim/accelerated.py').read_text(encoding='utf-8'))
        result=next(n for n in reference.body if getattr(n,'name',None)=='SimResult')
        method=next(n for n in result.body if getattr(n,'name',None)=='compute_detailed')
        assert canonical_ast(node.body[index])==canonical_ast(method.body[index]), 'material result prefix changed'
        del node.body[index]
    expected=ast.parse(LAMMPS_PREFIXES[key]).body[0]
    assert canonical_ast(node.body[index])==canonical_ast(expected), key
    del node.body[index]
    return node


def restore_historical_save(result_node):
    """Validate the public result contract before comparing the old remainder."""
    node=remove_reviewed_saved_energy_diagnostics(result_node)
    expected=ast.parse(public_baseline('fibernet/sim/accelerated.py').read_text(encoding='utf-8'))
    expected_result=next(n for n in expected.body if getattr(n,'name',None)=='SimResult')
    save=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='save')
    expected_save=next(n for n in expected_result.body if isinstance(n,ast.FunctionDef) and n.name=='save')
    assert canonical_ast(save)==canonical_ast(expected_save), 'public result writer changed'
    baseline=(ROOT/'tests/fixtures/observer_prior71_accelerated.txt').read_text(encoding='utf-8')
    old_result=next(n for n in ast.parse(baseline).body if getattr(n,'name',None)=='SimResult')
    old_save=next(n for n in old_result.body if isinstance(n,ast.FunctionDef) and n.name=='save')
    node.body[node.body.index(save)]=old_save
    # Match full approved methods, including the independently reviewed spring
    # metadata delta, before comparing the historical engine. No loose removal.
    provenance_tree = ast.parse(reviewed_spring_provenance_source().decode('utf-8'))
    provenance_result = next(n for n in provenance_tree.body if getattr(n,'name',None)=='SimResult')
    for name in ('to_dict', 'load'):
        method=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name)
        expected_method=next(n for n in expected_result.body if isinstance(n,ast.FunctionDef) and n.name==name)
        provenance_method=next(n for n in provenance_result.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert canonical_ast(method) in (canonical_ast(expected_method), canonical_ast(provenance_method)), 'public result method changed: '+name
        old_method=next(n for n in old_result.body if isinstance(n,ast.FunctionDef) and n.name==name)
        node.body[node.body.index(method)]=old_method
    detailed=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='compute_detailed')
    node.body[node.body.index(detailed)]=remove_public_lammps_branch(detailed,'SimResult.compute_detailed')
    return node


class StretchRatioSourceAdmission(unittest.TestCase):
    def test_exact_old_and_new_versions_are_admitted(self):
        old = public_baseline('fibernet/easy.py').read_bytes()
        new = reviewed_easy_stretch_ratio_source()
        self.assertIn((ROOT/'fibernet/easy.py').read_bytes(),
                      (new, reviewed_easy_rl_failure_source(), reviewed_taichi_geometry_source('fibernet/easy.py')))
        self.assertNotEqual(old, new)
        require_public_source('fibernet/easy.py', old)
        require_public_source('fibernet/easy.py', new)

    def test_source_outside_simulate_is_unchanged(self):
        def without_simulate(raw):
            tree = ast.parse(raw)
            tree.body = [node for node in tree.body
                         if not (isinstance(node, ast.FunctionDef) and node.name == 'simulate')]
            return canonical_ast(tree)
        self.assertEqual(without_simulate(public_baseline('fibernet/easy.py').read_bytes()),
                         without_simulate(reviewed_easy_stretch_ratio_source()))

    def test_unreviewed_mutations_are_refused(self):
        raw = reviewed_easy_stretch_ratio_source()
        changes = (
            (b'if strain is not None:', b'if False:'),
            (b'strain = 2.0', b'strain = 1.02'),
            (b'stretch_ratio > 0', b'stretch_ratio >= 0'),
            (b'np.isfinite(stretch_ratio)', b'True'),
            (b'    *,\n    stretch_ratio:', b'    stretch_ratio:'),
            (b'strain = stretch_ratio', b'strain = stretch_ratio + 1'),
            (b'target: str = "max_force"', b'target: str = "energy"'),
        )
        for before, after in changes:
            with self.subTest(mutation=before):
                self.assertIn(before, raw)
                changed = raw.replace(before, after, 1)
                with self.assertRaises(AssertionError):
                    require_public_source('fibernet/easy.py', changed)
        for changed in (raw + b'\npass\n', raw + b'\n# unreviewed\n',
                        raw.replace(b'\n', b'\r\n')):
            with self.subTest(suffix=changed[-40:]), self.assertRaises(AssertionError):
                require_public_source('fibernet/easy.py', changed)

    def test_fixture_bytes_survive_crlf_checkout(self):
        import subprocess
        import tempfile
        name = 'tests/fixtures/easy_stretch_ratio_v1/easy.py.txt'
        raw = reviewed_easy_stretch_ratio_source()
        for autocrlf in ('false', 'true'):
            with self.subTest(autocrlf=autocrlf), tempfile.TemporaryDirectory() as directory:
                repo = Path(directory)
                def git(*args):
                    return subprocess.run(['git', '-c', 'core.autocrlf=' + autocrlf,
                                           '-c', 'core.safecrlf=false', *args],
                                          cwd=repo, check=True, capture_output=True, timeout=10)
                git('init', '-q')
                (repo/'.gitattributes').write_bytes((ROOT/'.gitattributes').read_bytes())
                target = repo/name
                target.parent.mkdir(parents=True)
                target.write_bytes(raw)
                git('add', '--', '.gitattributes', name)
                target.unlink()
                git('checkout-index', '--all', '--force')
                self.assertEqual(target.read_bytes(), raw)

    def test_fixture_digest_is_not_self_accepting(self):
        fixture = ROOT/'tests/fixtures/easy_stretch_ratio_v1/easy.py.txt'
        original = Path.read_bytes
        def changed_read(path):
            raw = original(path)
            return raw + b'\n# changed fixture\n' if path == fixture else raw
        with patch.object(Path, 'read_bytes', changed_read):
            with self.assertRaisesRegex(AssertionError, 'reviewed stretch ratio source changed'):
                reviewed_easy_stretch_ratio_source()



class RLFailureSourceAdmission(unittest.TestCase):
    def test_exact_historical_versions_and_current_are_admitted(self):
        old = public_baseline('fibernet/easy.py').read_bytes()
        stretch = reviewed_easy_stretch_ratio_source()
        failure = reviewed_easy_rl_failure_source()
        current = reviewed_taichi_geometry_source('fibernet/easy.py')
        self.assertEqual((ROOT/'fibernet/easy.py').read_bytes(), current)
        require_public_source('fibernet/easy.py', current)
        self.assertEqual(len({old, stretch, failure}), 3)
        for raw in (old, stretch, failure):
            require_public_source('fibernet/easy.py', raw)

    def test_source_outside_rollout_and_summary_is_unchanged(self):
        def without_rollout_and_summary(raw):
            tree = ast.parse(raw)
            self.assertIsInstance(tree.body[0], ast.Expr)
            self.assertIsInstance(tree.body[0].value, ast.Constant)
            self.assertIsInstance(tree.body[0].value.value, str)
            tree.body = [node for node in tree.body[1:]
                         if not (isinstance(node, ast.FunctionDef) and node.name == 'train_rl')]
            return canonical_ast(tree)
        self.assertEqual(without_rollout_and_summary(reviewed_easy_stretch_ratio_source()),
                         without_rollout_and_summary(reviewed_easy_rl_failure_source()))

    def test_unreviewed_mutations_are_refused(self):
        raw = reviewed_easy_rl_failure_source()
        changes = (
            (b'n_episodes <= 0', b'n_episodes < 0'),
            (b'not math.isfinite(reward)', b'False'),
            (b'if "error" in info:', b'if False:'),
            (b'env.close()', b'pass'),
            (b'allow_nan=False', b'allow_nan=True'),
            (b'os.replace(temporary, destination)', b'pass'),
            (b'if strain is not None:', b'if False:'),
            (b'strain = 2.0', b'strain = 1.02'),
            (b'target: str = "max_force"', b'target: str = "energy"'),
        )
        for before, after in changes:
            with self.subTest(mutation=before):
                self.assertIn(before, raw)
                with self.assertRaises(AssertionError):
                    require_public_source('fibernet/easy.py', raw.replace(before, after, 1))
        for changed in (raw + b'\npass\n', raw + b'\n# unreviewed\n',
                        raw.replace(b'\n', b'\r\n')):
            with self.subTest(suffix=changed[-40:]), self.assertRaises(AssertionError):
                require_public_source('fibernet/easy.py', changed)

    def test_fixture_digest_is_not_self_accepting(self):
        fixture = ROOT/'tests/fixtures/easy_rl_failure_v1/easy.py.txt'
        original = Path.read_bytes
        def changed_read(path):
            raw = original(path)
            return raw + b'\n# changed fixture\n' if path == fixture else raw
        with patch.object(Path, 'read_bytes', changed_read):
            with self.assertRaisesRegex(AssertionError, 'reviewed RL failure source changed'):
                reviewed_easy_rl_failure_source()


class GeometryPolicySourceAdmission(unittest.TestCase):
    def test_current_sources_match_only_the_exact_reviewed_deltas(self):
        for relative in TAICHI_GEOMETRY_SOURCE_PINS:
            with self.subTest(source=relative):
                expected = (reviewed_saved_energy_diagnostics_source() if relative == 'fibernet/sim/accelerated.py'
                            else reviewed_taichi_geometry_source(relative))
                self.assertEqual((ROOT/relative).read_bytes(), expected)
                require_public_source(relative, expected)
                for changed in (expected+b'\n# unreviewed\n',
                                expected.replace(b'\n', b'\r\n')):
                    with self.assertRaises(AssertionError):
                        require_public_source(relative, changed)

    def test_easy_delta_leaves_all_other_definitions_and_exports_unchanged(self):
        old = ast.parse(reviewed_easy_rl_failure_source())
        new = ast.parse(reviewed_taichi_geometry_source('fibernet/easy.py'))
        expected_import = ast.parse(
            'from fibernet.sim._taichi_geometry import '
            'preflight_taichi_geometry as _preflight_taichi_geometry, '
            'validate_geometry_policy as _validate_geometry_policy').body[0]
        imports = [node for node in new.body if isinstance(node, ast.ImportFrom)
                   and node.module == 'fibernet.sim._taichi_geometry']
        self.assertEqual([canonical_ast(node) for node in imports],
                         [canonical_ast(expected_import)])
        new.body.remove(imports[0])
        allowed = {'simulate', '_batch_parameters', 'batch_simulate_from_json'}
        old_functions = {node.name: node for node in old.body
                         if isinstance(node, ast.FunctionDef)}
        for index, node in enumerate(new.body):
            if isinstance(node, ast.FunctionDef) and node.name in allowed:
                new.body[index] = copy.deepcopy(old_functions[node.name])
        def sources(tree):
            return next(node for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and
                                target.id == '_BATCH_SOURCE_FILES' for target in node.targets))
        old_sources, new_sources = sources(old), sources(new)
        expected = list(ast.literal_eval(old_sources.value))
        expected.insert(2, 'fibernet/sim/_taichi_geometry.py')
        self.assertEqual(ast.literal_eval(new_sources.value), tuple(expected))
        new.body[new.body.index(new_sources)] = copy.deepcopy(old_sources)
        self.assertEqual(canonical_ast(old), canonical_ast(new))

    def test_accelerated_delta_leaves_serializers_and_native_kernels_unchanged(self):
        old = ast.parse(reviewed_spring_provenance_source())
        new = ast.parse(reviewed_taichi_geometry_source('fibernet/sim/accelerated.py'))
        expected_import = ast.parse(
            'from ._taichi_geometry import preflight_taichi_geometry').body[0]
        imports = [node for node in new.body if isinstance(node, ast.ImportFrom)
                   and node.module == '_taichi_geometry']
        self.assertEqual([canonical_ast(node) for node in imports],
                         [canonical_ast(expected_import)])
        new.body.remove(imports[0])
        old_engine = next(node for node in old.body if isinstance(node, ast.ClassDef)
                          and node.name == 'TaichiEngine')
        new_engine = next(node for node in new.body if isinstance(node, ast.ClassDef)
                          and node.name == 'TaichiEngine')
        old_methods = {node.name: node for node in old_engine.body
                       if isinstance(node, ast.FunctionDef)}
        for index, node in enumerate(new_engine.body):
            if isinstance(node, ast.FunctionDef) and node.name in {'dynamics', 'stretch_test'}:
                new_engine.body[index] = copy.deepcopy(old_methods[node.name])
        self.assertEqual(canonical_ast(old), canonical_ast(new))

    def test_policy_changes_cannot_be_silently_normalized(self):
        cases = {
            'fibernet/easy.py': (
                (b'kwargs.setdefault("geometry_policy", "reject_internal_points")',
                 b'kwargs.setdefault("geometry_policy", "endpoint_chord")'),
                (b'_preflight_taichi_geometry(graph, geometry_policy=kwargs["geometry_policy"])',
                 b'pass'),
                (b'"fibernet/sim/_taichi_geometry.py",', b''),
            ),
            'fibernet/sim/accelerated.py': (
                (b'geometry_policy: str = "reject_internal_points"',
                 b'geometry_policy: str = "endpoint_chord"'),
                (b'"taichi_geometry": geometry_record', b'"taichi_geometry": {}'),
                (b'geometry_record = preflight_taichi_geometry(',
                 b'geometry_record = ignored_preflight('),
            ),
        }
        for relative, replacements in cases.items():
            raw = reviewed_taichi_geometry_source(relative)
            for before, after in replacements:
                with self.subTest(source=relative, mutation=before):
                    self.assertIn(before, raw)
                    with self.assertRaises(AssertionError):
                        require_public_source(relative, raw.replace(before, after, 1))

    def test_delta_fixture_digest_is_not_self_accepting(self):
        original = Path.read_bytes
        for relative, (name, _) in TAICHI_GEOMETRY_SOURCE_PINS.items():
            fixture = ROOT/'tests/fixtures/taichi_geometry_policy_v1'/name
            def changed_read(path):
                raw = original(path)
                return raw+b'\n' if path == fixture else raw
            with self.subTest(source=relative), patch.object(Path, 'read_bytes', changed_read):
                with self.assertRaisesRegex(AssertionError, 'reviewed geometry source delta changed'):
                    reviewed_taichi_geometry_source(relative)

    def test_delta_source_path_is_allowlisted(self):
        for relative in ('../outside', '/etc/passwd', 'fibernet/api.py'):
            with self.assertRaises(ValueError):
                reviewed_taichi_geometry_source(relative)


class SavedEnergyDiagnosticsSourceAdmission(unittest.TestCase):
    def test_only_the_exact_additive_method_is_admitted(self):
        prior = reviewed_taichi_geometry_source('fibernet/sim/accelerated.py')
        current = reviewed_saved_energy_diagnostics_source()
        self.assertEqual(current, SOURCE.read_bytes())
        require_public_source('fibernet/sim/accelerated.py', prior)
        require_public_source('fibernet/sim/accelerated.py', current)
        old, new = ast.parse(prior), ast.parse(current)
        index = next(i for i, n in enumerate(new.body) if isinstance(n, ast.ClassDef) and n.name == 'SimResult')
        new.body[index] = remove_reviewed_saved_energy_diagnostics(new.body[index])
        self.assertEqual(canonical_ast(old), canonical_ast(new))

    def test_changed_method_and_unrelated_source_are_refused(self):
        raw = reviewed_saved_energy_diagnostics_source()
        for before, after in ((b'viscoelastic_energy_diagnostics(self)', b'viscoelastic_energy_diagnostics(self, tolerance=1.)'),
                              (b'from ._taichi_viscoelastic import energy_diagnostics', b'from ._taichi_viscoelastic import source_capabilities as energy_diagnostics'),
                              (b'return energy_diagnostics(self)', b'return energy_diagnostics(None)')):
            self.assertIn(before, raw)
            changed = raw.replace(before, after, 1)
            with self.subTest(mutation=before), self.assertRaises(AssertionError):
                require_public_source('fibernet/sim/accelerated.py', changed)
            result = next(n for n in ast.parse(changed).body if isinstance(n, ast.ClassDef) and n.name == 'SimResult')
            with self.subTest(normalization=before), self.assertRaises(AssertionError):
                restore_historical_save(result)
        for changed in (raw+b'\n# unreviewed\n', raw.replace(b'\n', b'\r\n')):
            with self.assertRaises(AssertionError):
                require_public_source('fibernet/sim/accelerated.py', changed)

    def test_duplicate_method_and_fixture_literal_tampering_are_refused(self):
        result = next(n for n in ast.parse(reviewed_saved_energy_diagnostics_source()).body
                      if isinstance(n, ast.ClassDef) and n.name == 'SimResult')
        method = next(n for n in result.body if isinstance(n, ast.FunctionDef) and n.name == 'viscoelastic_energy_diagnostics')
        result.body.append(copy.deepcopy(method))
        with self.assertRaisesRegex(AssertionError, 'duplicate saved diagnostics'):
            restore_historical_save(result)
        with patch.dict(globals(), SAVED_ENERGY_DIAGNOSTICS_METHOD=SAVED_ENERGY_DIAGNOSTICS_METHOD+'    # unreviewed\n'):
            with self.assertRaisesRegex(AssertionError, 'reviewed saved diagnostics source changed'):
                reviewed_saved_energy_diagnostics_source()


class SourceShape(unittest.TestCase):
    def test_spring_provenance_delta_is_scoped_and_bound(self):
        before = ast.parse(reviewed_result_identity_source().decode('utf-8'))
        after = ast.parse(reviewed_spring_provenance_source().decode('utf-8'))
        old = {n.name:n for n in before.body if hasattr(n, 'name')}
        new = {n.name:n for n in after.body if hasattr(n, 'name')}
        self.assertEqual(set(new)-set(old), {'_spring_coefficient_record', '_spring_provenance',
                                            '_validate_spring_provenance', '_populate_spring_result'})
        self.assertEqual(set(old)-set(new), set())
        for name in set(old)-{'SimResult', 'TaichiEngine'}:
            self.assertEqual(canonical_ast(old[name]), canonical_ast(new[name]), name)
        for cls, allowed in (('SimResult', {'compute_detailed', 'to_dict', 'load'}),
                             ('TaichiEngine', {'dynamics'})):
            old_methods = {n.name:n for n in old[cls].body if isinstance(n, ast.FunctionDef)}
            new_methods = {n.name:n for n in new[cls].body if isinstance(n, ast.FunctionDef)}
            self.assertEqual(set(old_methods), set(new_methods))
            for name in set(old_methods)-allowed:
                self.assertEqual(canonical_ast(old_methods[name]), canonical_ast(new_methods[name]), name)
        self.assertEqual(reviewed_saved_energy_diagnostics_source(), SOURCE.read_bytes())

    def test_spring_provenance_persistence_mutations_are_refused(self):
        raw = SOURCE.read_bytes()
        replacements = [
            (b"_SPRING_PROVENANCE = 'spring_coefficient_provenance'", b"_SPRING_PROVENANCE = 'changed'"),
            (b"quantity_kind='force'", b"quantity_kind='force_per_length'"),
            (b"reason='omitted_from_summary'", b"reason='inferred_from_stiffness'"),
            (b"self.metadata.get('edge_ids'), detailed=detailed", b"self.metadata.get('edge_ids'), detailed=True"),
            (b"array_mode='owned'", b"array_mode='retain'"),
            (b"result.compute_detailed(graph, stiffness=normalized)", b"result.compute_detailed(graph, stiffness=1.)"),
        ]
        for before, after in replacements:
            self.assertIn(before, raw)
            with self.subTest(before=before):
                with self.assertRaises(AssertionError):
                    require_public_source('fibernet/sim/accelerated.py', raw.replace(before, after, 1))

    def test_spring_provenance_normalization_rejects_unreviewed_result_edits(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        result = next(n for n in tree.body if getattr(n, 'name', None)=='SimResult')
        for name in ('to_dict', 'load'):
            altered = copy.deepcopy(result)
            method = next(n for n in altered.body if isinstance(n, ast.FunctionDef) and n.name==name)
            method.body.append(ast.Pass())
            with self.subTest(method=name):
                with self.assertRaises(AssertionError):
                    restore_historical_save(altered)
        altered = copy.deepcopy(result)
        method = next(n for n in altered.body if isinstance(n, ast.FunctionDef) and n.name=='compute_detailed')
        method.body.append(ast.Pass())
        self.assertNotEqual(canonical_ast(restore_historical_save(result)),
                            canonical_ast(restore_historical_save(altered)))

    def test_result_identity_reviewed_source_is_method_scoped(self):
        baseline = public_baseline('fibernet/sim/accelerated.py').read_bytes()
        self.assertEqual(reviewed_saved_energy_diagnostics_source(), SOURCE.read_bytes())
        require_public_source('fibernet/sim/accelerated.py', reviewed_result_identity_source())
        require_public_source('fibernet/sim/accelerated.py', baseline)
        require_public_source('fibernet/sim/accelerated.py', SOURCE.read_bytes())
        approved = result_detail_method(SOURCE.read_bytes())
        historical = result_detail_method(baseline)
        self.assertEqual(canonical_ast(normalize_result_identity_method(approved)), canonical_ast(historical))

    def test_result_identity_prefix_mutations_are_refused(self):
        method = result_detail_method(SOURCE.read_bytes())
        # FEM must still precede native-material and LAMMPS routing.
        for index in (1, 2, 3):
            for mutation in ('condition', 'body', 'missing', 'extra'):
                with self.subTest(index=index, mutation=mutation):
                    changed = copy.deepcopy(method)
                    if mutation == 'condition': changed.body[index].test = ast.Constant(value=True)
                    elif mutation == 'body': changed.body[index].body = [ast.Pass()]
                    elif mutation == 'extra': changed.body[index].body.append(ast.Pass())
                    else: del changed.body[index]
                    with self.assertRaises(AssertionError):
                        remove_public_lammps_branch(changed, 'SimResult.compute_detailed')
        changed = copy.deepcopy(method)
        changed.args.defaults = [ast.Constant(value=1.)]
        with self.assertRaises(AssertionError): normalize_result_identity_method(changed)
        changed = copy.deepcopy(method)
        call = next(n for n in ast.walk(changed) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == '_strict_state_ids')
        call.keywords.append(ast.keyword(arg='nonnegative', value=ast.Constant(value=True)))
        with self.assertRaises(AssertionError): normalize_result_identity_method(changed)

    def test_result_identity_arithmetic_mutations_are_refused(self):
        method = result_detail_method(SOURCE.read_bytes())
        changed = copy.deepcopy(method)
        force = next(n for n in ast.walk(changed) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'forces' for t in n.targets))
        force.value.op = ast.Add()
        with self.assertRaises(AssertionError): normalize_result_identity_method(changed)
        changed = copy.deepcopy(method)
        energy = next(n for n in ast.walk(changed) if isinstance(n, ast.keyword) and n.arg == 'energy')
        energy.value = ast.Constant(value=0.)
        with self.assertRaises(AssertionError): normalize_result_identity_method(changed)

    def test_result_identity_normalization_preserves_appended_statements(self):
        method = result_detail_method(SOURCE.read_bytes())
        changed = copy.deepcopy(method)
        tail = ast.parse('unrelated_appended_statement = 17').body[0]
        changed.body.append(tail)
        normalized = normalize_result_identity_method(changed)
        self.assertEqual(canonical_ast(normalized.body[-1]), canonical_ast(tail))
        self.assertNotEqual(canonical_ast(normalize_result_identity_method(method)), canonical_ast(normalized))
        raw = SOURCE.read_bytes()
        appended = raw.replace(b'\n    def to_dict(', b'\n        unrelated_appended_statement = 17\n\n    def to_dict(', 1)
        self.assertNotEqual(appended, raw)
        with self.assertRaises(AssertionError): require_public_source('fibernet/sim/accelerated.py', appended)

    def test_result_identity_unrelated_source_mutations_are_refused(self):
        raw = SOURCE.read_bytes()
        changes = [raw+b'\n# unrelated comment\n', raw.replace(b'\n', b'\r\n'),
                   raw.replace(b'max_displacement: float = 0.0', b'max_displacement: float = 1.0', 1),
                   raw.replace(b'def to_dict(self, detailed: bool = False)', b'def to_dict(self, detailed: bool = True)', 1),
                   raw.replace(b'forces = spring_coefficients * strain', b'forces = spring_coefficients + strain', 1)]
        for changed in changes:
            self.assertNotEqual(changed, raw)
            with self.subTest(changed=changed[-80:]):
                with self.assertRaises(AssertionError): require_public_source('fibernet/sim/accelerated.py', changed)

    def test_versioned_source_provenance_and_unchanged_kernels(self):
        manifest=json.loads((ROOT/'tests/fixtures/public_api_contract_v2/manifest.json').read_text())
        self.assertEqual(set(manifest['sources']), set(API_V2_MODULES))
        for relative, hashes in manifest['sources'].items():
            old=ROOT/'tests/fixtures/public_source_contract'/Path(relative).with_suffix('.py.txt')
            self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(), hashes['prior_sha256'])
            self.assertEqual(hashlib.sha256(public_baseline(relative).read_bytes()).hexdigest(), hashes['sha256'])
        def kernels(raw):
            return [canonical_ast(n) for n in ast.walk(ast.parse(raw))
                    if isinstance(n, ast.FunctionDef) and any(
                        isinstance(d, ast.Attribute) and isinstance(d.value, ast.Name)
                        and d.value.id == 'ti' and d.attr in ('kernel', 'func')
                        for d in n.decorator_list)]
        old=ROOT/'tests/fixtures/public_source_contract/fibernet/sim/accelerated.py.txt'
        self.assertEqual(kernels(old.read_text()), kernels(SOURCE.read_text()))

    def test_engine_matches_checked_in_public_source(self):
        require_public_source('fibernet/sim/accelerated.py',SOURCE.read_bytes())

    def test_lammps_prefix_mutations_are_refused(self):
        tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
        targets={n.name:n for n in tree.body if getattr(n,'name',None) in ('_validate_result_state','_dynamics_state_summary')}
        result=next(n for n in tree.body if getattr(n,'name',None)=='SimResult')
        targets['SimResult.compute_detailed']=next(n for n in result.body if getattr(n,'name',None)=='compute_detailed')
        for key,function in targets.items():
            index=1 if key.startswith('SimResult.') else 0
            remove_public_lammps_branch(function,key)
            for mutation in ('condition','body','missing','extra'):
                with self.subTest(key=key,mutation=mutation):
                    node=copy.deepcopy(function)
                    if mutation=='condition':node.body[index].test=ast.Constant(value=True)
                    elif mutation=='body':node.body[index].body=[ast.Pass()]
                    elif mutation=='extra':node.body[index].body.append(ast.Pass())
                    else:del node.body[index]
                    with self.assertRaises(AssertionError):remove_public_lammps_branch(node,key)

    def test_lammps_normalization_preserves_unrelated_statements(self):
        tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
        for name in ('_validate_result_state','_dynamics_state_summary'):
            function=next(n for n in tree.body if getattr(n,'name',None)==name)
            changed=copy.deepcopy(function);changed.body.append(ast.Pass())
            clean=remove_public_lammps_branch(function,name)
            remainder=remove_public_lammps_branch(changed,name)
            self.assertNotEqual(canonical_ast(clean),canonical_ast(remainder))
        result=next(n for n in tree.body if getattr(n,'name',None)=='SimResult')
        changed=copy.deepcopy(result)
        method=next(n for n in changed.body if getattr(n,'name',None)=='compute_detailed')
        method.body.append(ast.Pass())
        self.assertNotEqual(canonical_ast(restore_historical_save(result)),canonical_ast(restore_historical_save(changed)))

    def test_public_facades_match_checked_in_sources(self):
        for path in PUBLIC_MODULES[:3]:
            with self.subTest(path=path):require_public_source(path,(ROOT/path).read_bytes())

    def test_reviewed_optional_facades_are_exact(self):
        for relative in FACADE_PINS:
            raw = reviewed_facade(relative)
            require_public_source(relative, raw)
            require_public_source(relative, (ROOT/relative).read_bytes())
            if relative == 'fibernet/sim/__init__.py':
                require_public_source(relative, prior_sim_facade())
            # Both unrelated edits and edits inside lazily extracted declarations
            # or functions require a fresh source review, including whitespace.
            mutations = [raw+b'\npass\n', raw.replace(b'\n', b'\r\n'),
                         raw.replace(b'def __getattr__(name):', b'def __getattr__(name): # changed', 1),
                         raw.replace(b'_LAZY_EXPORTS = {', b'_LAZY_EXPORTS = { # changed', 1)]
            for changed in mutations:
                self.assertNotEqual(changed, raw)
                with self.subTest(relative=relative, changed=changed[:40]):
                    with self.assertRaises(AssertionError):require_public_source(relative, changed)
        root = reviewed_facade('fibernet/__init__.py')
        changed = root.replace(b'_LAZY_MODULES = {', b'_LAZY_MODULES = { # changed', 1)
        self.assertNotEqual(changed, root)
        with self.assertRaises(AssertionError):require_public_source('fibernet/__init__.py', changed)

    def test_reviewed_nodal_controls_facade_is_exact(self):
        relative = 'fibernet/sim/__init__.py'
        raw = reviewed_nodal_controls_facade()
        require_public_source(relative, raw)
        for changed in (raw+b'\npass\n', raw.replace(b'\n', b'\r\n'),
                        raw.replace(b'prepare_nodal_controls', b'wrong_controls', 1),
                        raw.replace(b'PreparedNodalControls', b'WrongControls', 1),
                        raw.replace(b'def __getattr__(name):', b'def __getattr__(name): # changed', 1)):
            self.assertNotEqual(changed, raw)
            with self.assertRaises(AssertionError):
                require_public_source(relative, changed)
        original = Path.read_bytes
        fixture = ROOT/'tests/fixtures/optional_facades_v2/sim_nodal_controls.py.txt'
        def changed_read(path):
            value = original(path)
            return value+b'\n' if path == fixture else value
        with patch.object(Path, 'read_bytes', changed_read):
            with self.assertRaisesRegex(AssertionError, 'reviewed nodal controls facade changed'):
                reviewed_nodal_controls_facade()

    def test_reviewed_thermoelastic_facade_is_exact(self):
        relative = 'fibernet/sim/__init__.py'
        raw = reviewed_thermoelastic_facade()
        self.assertEqual(raw, reviewed_lammps_capability_facade(reverse=(ROOT/relative).read_bytes()))
        require_public_source(relative, raw)
        for changed in (raw+b'\npass\n', raw.replace(b'\n', b'\r\n'),
                        raw.replace(b'ThermoelasticSizeLimits', b'WrongLimits', 1),
                        raw.replace(b'solve_thermoelastic_network_snapshot', b'wrong_owner', 1),
                        raw.replace(b'prepare_nodal_controls', b'wrong_controls', 1),
                        raw.replace(b'def __getattr__(name):', b'def __getattr__(name): # changed', 1)):
            self.assertNotEqual(changed, raw)
            with self.assertRaises(AssertionError):
                require_public_source(relative, changed)
        original = Path.read_bytes
        fixture = ROOT/'tests/fixtures/optional_facades_v2/sim_thermoelastic.py.txt'
        def changed_read(path):
            value = original(path)
            return value+b'\n' if path == fixture else value
        with patch.object(Path, 'read_bytes', changed_read):
            with self.assertRaisesRegex(AssertionError, 'reviewed thermoelastic facade changed'):
                reviewed_thermoelastic_facade()

    def test_reviewed_fiber_network_profile_facade_is_exact_and_reversible(self):
        relative = 'fibernet/__init__.py'
        raw = reviewed_fiber_network_profile_facade()
        self.assertEqual(raw, (ROOT/relative).read_bytes())
        self.assertEqual(reviewed_fiber_network_profile_facade(reverse=raw),
                         reviewed_facade(relative))
        for admitted in (public_baseline(relative).read_bytes(), reviewed_facade(relative), raw):
            require_public_source(relative, admitted)
        for changed in (raw+b'\npass\n', raw.replace(b'\n', b'\r\n'),
                        raw.replace(b'FiberNetworkProfile', b'WrongProfile', 1),
                        raw.replace(b'fibernet.core.fiber_network_profile', b'fibernet.core.network', 1),
                        raw.replace(b'"FiberNetworkProjectionLimits",', b'"WrongLimits",', 1),
                        raw.replace(b'def __getattr__(name):', b'def __getattr__(name): # changed', 1)):
            self.assertNotEqual(changed, raw)
            with self.assertRaises(AssertionError):
                require_public_source(relative, changed)
            with self.assertRaises(AssertionError):
                reviewed_fiber_network_profile_facade(reverse=changed)
        original = Path.read_bytes
        fixture = ROOT/'tests/fixtures/optional_facades_v2/root_fiber_network_profile_delta.json'
        for mutation in (lambda value: value+b'\n',
                         lambda value: value.replace(b'FiberNetworkProfile', b'WrongProfile', 1)):
            def changed_read(path):
                value = original(path)
                return mutation(value) if path == fixture else value
            with patch.object(Path, 'read_bytes', changed_read):
                with self.assertRaisesRegex(AssertionError, 'reviewed FiberNetwork profile facade delta changed'):
                    reviewed_fiber_network_profile_facade()

    def test_reviewed_lammps_capability_facade_is_exact_and_reversible(self):
        relative = 'fibernet/sim/__init__.py'
        raw = reviewed_lammps_capability_facade()
        self.assertEqual(raw, (ROOT/relative).read_bytes())
        self.assertEqual(reviewed_lammps_capability_facade(reverse=raw),
                         reviewed_thermoelastic_facade())
        require_public_source(relative, raw)
        for changed in (raw+b'\npass\n', raw.replace(b'\n', b'\r\n'),
                        raw.replace(b'lammps_source_capabilities', b'wrong_capabilities', 1),
                        raw.replace(b'fibernet.sim._lammps_result', b'fibernet.sim.accelerated', 1),
                        raw.replace(b'ThermoelasticSizeLimits', b'WrongLimits', 1),
                        raw.replace(b'prepare_nodal_controls', b'wrong_controls', 1),
                        raw.replace(b'def __getattr__(name):', b'def __getattr__(name): # changed', 1)):
            self.assertNotEqual(changed, raw)
            with self.assertRaises(AssertionError):
                require_public_source(relative, changed)
            with self.assertRaises(AssertionError):
                reviewed_lammps_capability_facade(reverse=changed)
        original = Path.read_bytes
        fixture = ROOT/'tests/fixtures/optional_facades_v2/sim_lammps_capabilities_delta.json'
        for mutation in (lambda value: value+b'\n',
                         lambda value: value.replace(b'lammps_source_capabilities', b'wrong_capabilities', 1)):
            def changed_read(path):
                value = original(path)
                return mutation(value) if path == fixture else value
            with patch.object(Path, 'read_bytes', changed_read):
                with self.assertRaisesRegex(AssertionError, 'reviewed LAMMPS facade delta changed'):
                    reviewed_lammps_capability_facade()

    def test_lazy_easy_all_extension(self):
        tree=ast.parse((ROOT/'fibernet/easy.py').read_text(encoding='utf-8'))
        extra=[n for n in tree.body if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='__all__']
        self.assertEqual(len(extra),1)
        self.assertEqual(ast.literal_eval(extra[0].value),['render_graph','render_deformation'])

    def test_energy_helper_and_lazy_viz_match_public_sources(self):
        for path in ('fibernet/viz/__init__.py','fibernet/sim/_energy_accounting.py'):
            with self.subTest(path=path):require_public_source(path,(ROOT/path).read_bytes())

    def test_public_source_mutations_are_refused(self):
        for path in PUBLIC_MODULES:
            raw=public_baseline(path).read_bytes()
            require_public_source(path,raw)
            with self.subTest(path=path):
                with self.assertRaises(AssertionError):require_public_source(path,raw+b'\npass\n')

    def test_checkout_line_endings_follow_source_contract(self):
        for path in PUBLIC_MODULES:
            raw=public_baseline(path).read_bytes()
            for changed in (raw+b'\n# changed\n',raw.replace(b'\n',b'\r\n')):
                with self.subTest(path=path):
                    if path in PUBLIC_MODULES[3:] or path in API_V2_MODULES:
                        with self.assertRaises(AssertionError):require_public_source(path,changed)
                    else:
                        require_public_source(path,changed)

    def test_writer_mutation_is_refused(self):
        tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
        result=next(n for n in tree.body if getattr(n,'name',None)=='SimResult')
        writer=next(n for n in result.body if isinstance(n,ast.FunctionDef) and n.name=='save')
        writer.body.append(ast.Pass())
        with self.assertRaises(AssertionError):restore_historical_save(result)

    def test_public_baseline_path_is_allowlisted(self):
        for path in ('../outside','fibernet/not_a_contract.py','/etc/passwd'):
            with self.assertRaises(ValueError):public_baseline(path)

    def test_python39_syntax(self):
        for path in PUBLIC_MODULES:
            ast.parse((ROOT/path).read_text(encoding='utf-8'),filename=path,feature_version=(3,9))

if __name__=='__main__':unittest.main()

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


def reviewed_octet_fcc_facade(reverse=None):
    """Admit only the pinned additive octet export; retain its exact inverse."""
    raw = (ROOT/'tests/fixtures/standard_octet_v1/root_facade_delta.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '56aa9fe12ce9fc46fc0f9b6a8243802e4aa0d79423034b20da4b7f21773d23e0', 'reviewed octet FCC facade delta changed'
    delta = json.loads(raw)
    assert delta['schema'] == 'reviewed_octet_fcc_facade_delta/1'
    assert delta['source'] == 'fibernet/__init__.py'
    assert delta['baseline_main'] == '1a5e69c031810126a56e2a87a25a70748ba4c4bf'
    base = reviewed_fiber_network_profile_facade()
    assert hashlib.sha256(base).hexdigest() == delta['base_sha256'], 'octet facade delta base changed'
    source = base if reverse is None else reverse
    if reverse is not None:
        assert hashlib.sha256(source).hexdigest() == delta['result_sha256'], 'octet facade inverse input changed'
    for hunk in (delta['hunks'] if reverse is None else reversed(delta['hunks'])):
        before, after = hunk['before'].encode('utf-8'), hunk['after'].encode('utf-8')
        if reverse is not None:
            before, after = after, before
        assert before and source.count(before) == 1, 'octet facade delta is not uniquely located'
        source = source.replace(before, after, 1)
    digest = delta['result_sha256'] if reverse is None else delta['base_sha256']
    assert hashlib.sha256(source).hexdigest() == digest, 'octet facade delta result changed'
    if reverse is not None:
        assert source == base, 'octet facade inverse changed historical bytes'
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


def reviewed_archived_ge_facade(reverse=None):
    """Admit only the pinned lazy alias delta, with an exact historical inverse."""
    raw = (ROOT/'tests/fixtures/optional_facades_v2/sim_archived_ge_delta.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '2fa6da528f3d52363d6ce94708f54d8788bfc2bcba45d964a599437216e51adf', 'reviewed archived GE facade delta changed'
    delta = json.loads(raw)
    assert delta['schema'] == 'reviewed_archived_ge_facade_delta/1'
    assert delta['source'] == 'fibernet/sim/__init__.py'
    assert delta['baseline_main'] == 'ef6a83ddb72021295c126610d21f96113de4a218'
    base = reviewed_lammps_capability_facade()
    assert hashlib.sha256(base).hexdigest() == delta['base_sha256'], 'archived GE facade delta base changed'
    source = base if reverse is None else reverse
    if reverse is not None:
        assert hashlib.sha256(source).hexdigest() == delta['result_sha256'], 'archived GE facade inverse input changed'
    for hunk in (delta['hunks'] if reverse is None else reversed(delta['hunks'])):
        before, after = hunk['before'].encode('utf-8'), hunk['after'].encode('utf-8')
        if reverse is not None:
            before, after = after, before
        assert before and source.count(before) == 1, 'archived GE facade delta is not uniquely located'
        source = source.replace(before, after, 1)
    digest = delta['result_sha256'] if reverse is None else delta['base_sha256']
    assert hashlib.sha256(source).hexdigest() == digest, 'archived GE facade delta result changed'
    if reverse is not None:
        assert source == base, 'archived GE facade inverse changed historical bytes'
    return source


def reviewed_easy_show_source(reverse=None):
    """Admit only the pinned show/import delta, with an exact historical inverse."""
    raw = (ROOT/'tests/fixtures/easy_show_dimension_v1/easy_delta.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'f7f619fb7982682a6a2998db83e909a0531f2ea7d48ac3e1c450dbd3f2bcb2c8', 'reviewed dimension-aware show source delta changed'
    delta = json.loads(raw)
    assert delta['schema'] == 'reviewed_easy_show_source_delta/1'
    assert delta['source'] == 'fibernet/easy.py'
    assert delta['baseline_main'] == 'a39cc3b0806abe2978f3c7e6ad6ad87aff87c2d3'
    base = reviewed_taichi_geometry_source('fibernet/easy.py')
    assert hashlib.sha256(base).hexdigest() == delta['base_sha256'], 'dimension-aware show source delta base changed'
    source = base if reverse is None else reverse
    if reverse is not None:
        assert hashlib.sha256(source).hexdigest() == delta['result_sha256'], 'dimension-aware show source inverse input changed'
    for hunk in (delta['hunks'] if reverse is None else reversed(delta['hunks'])):
        before, after = hunk['before'].encode('utf-8'), hunk['after'].encode('utf-8')
        if reverse is not None:
            before, after = after, before
        assert before and source.count(before) == 1, 'dimension-aware show source delta is not uniquely located'
        source = source.replace(before, after, 1)
    digest = delta['result_sha256'] if reverse is None else delta['base_sha256']
    assert hashlib.sha256(source).hexdigest() == digest, 'dimension-aware show source delta result changed'
    if reverse is not None:
        assert source == base, 'dimension-aware show source inverse changed historical bytes'
    return source


def reviewed_easy_batch_report_source(reverse=None):
    """Admit only the pinned batch report delta, with an exact historical inverse."""
    raw = (ROOT/'tests/fixtures/easy_batch_report_v1/easy_delta.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '21479533442333ce88a5a4a19bbcb55627f747bce7828fd12714e0d765bb2383', 'reviewed batch report source delta changed'
    delta = json.loads(raw)
    assert delta['schema'] == 'reviewed_easy_batch_report_source_delta/1'
    assert delta['source'] == 'fibernet/easy.py'
    assert delta['baseline_main'] == '427c98bed8ce98bb78bd2bf05d63cd5d7585d373'
    base = reviewed_easy_show_source()
    assert hashlib.sha256(base).hexdigest() == delta['base_sha256'], 'batch report source delta base changed'
    source = base if reverse is None else reverse
    if reverse is not None:
        assert hashlib.sha256(source).hexdigest() == delta['result_sha256'], 'batch report source inverse input changed'
    for hunk in (delta['hunks'] if reverse is None else reversed(delta['hunks'])):
        before, after = hunk['before'].encode('utf-8'), hunk['after'].encode('utf-8')
        if reverse is not None:
            before, after = after, before
        assert before and source.count(before) == 1, 'batch report source delta is not uniquely located'
        source = source.replace(before, after, 1)
    digest = delta['result_sha256'] if reverse is None else delta['base_sha256']
    assert hashlib.sha256(source).hexdigest() == digest, 'batch report source delta result changed'
    if reverse is not None:
        assert source == base, 'batch report source inverse changed historical bytes'
    return source


def reviewed_native_io_source(reverse=None):
    """Admit only the pinned native IO/import delta, with an exact historical inverse."""
    raw = (ROOT/'tests/fixtures/native_io_dispatch_v1/api_delta.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '68455aeb2e1f3a0b3f55c123e7c8ca192f5503444bd98e9a1b5851654b485f9d', 'reviewed native IO source delta changed'
    delta = json.loads(raw)
    assert delta['schema'] == 'reviewed_native_io_source_delta/1'
    assert delta['source'] == 'fibernet/api.py'
    assert delta['baseline_main'] == 'a39cc3b0806abe2978f3c7e6ad6ad87aff87c2d3'
    base = public_baseline('fibernet/api.py').read_bytes()
    assert hashlib.sha256(base).hexdigest() == delta['base_sha256'], 'native IO source delta base changed'
    source = base if reverse is None else reverse
    if reverse is not None:
        assert hashlib.sha256(source).hexdigest() == delta['result_sha256'], 'native IO source inverse input changed'
    for hunk in (delta['hunks'] if reverse is None else reversed(delta['hunks'])):
        before, after = hunk['before'].encode('utf-8'), hunk['after'].encode('utf-8')
        if reverse is not None:
            before, after = after, before
        assert before and source.count(before) == 1, 'native IO source delta is not uniquely located'
        source = source.replace(before, after, 1)
    digest = delta['result_sha256'] if reverse is None else delta['base_sha256']
    assert hashlib.sha256(source).hexdigest() == digest, 'native IO source delta result changed'
    if reverse is not None:
        assert source == base, 'native IO source inverse changed historical bytes'
    return source


def reviewed_legacy_analysis_source(reverse=None):
    """Admit only the pinned legacy summary delta, with an exact historical inverse."""
    raw = (ROOT/'tests/fixtures/legacy_analysis_v1/api_delta.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'b15343961605399630fb5acc5f8b988b32f1d4449684d87ef78fb68de3384662', 'reviewed legacy summary source delta changed'
    delta = json.loads(raw)
    assert delta['schema'] == 'reviewed_legacy_analysis_source_delta/1'
    assert delta['source'] == 'fibernet/api.py'
    assert delta['baseline_main'] == '59a160f7959144656dd8b2b249faf80cf5395c5f'
    base = reviewed_native_io_source()
    assert hashlib.sha256(base).hexdigest() == delta['base_sha256'], 'legacy summary source delta base changed'
    source = base if reverse is None else reverse
    if reverse is not None:
        assert hashlib.sha256(source).hexdigest() == delta['result_sha256'], 'legacy summary source inverse input changed'
    for hunk in (delta['hunks'] if reverse is None else reversed(delta['hunks'])):
        before, after = hunk['before'].encode('utf-8'), hunk['after'].encode('utf-8')
        if reverse is not None:
            before, after = after, before
        assert before and source.count(before) == 1, 'legacy summary source delta is not uniquely located'
        source = source.replace(before, after, 1)
    digest = delta['result_sha256'] if reverse is None else delta['base_sha256']
    assert hashlib.sha256(source).hexdigest() == digest, 'legacy summary source delta result changed'
    if reverse is not None:
        assert source == base, 'legacy summary source inverse changed historical bytes'
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


def lammps_metadata_source_delta():
    raw = (ROOT/'tests/fixtures/lammps_metadata_persistence_v1/accelerated_delta.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '80c3393734c3cf0081bb820c2a07ea6bbe61111ed02db6d713b394c4d9d46bca', 'reviewed LAMMPS metadata delta changed'
    delta = json.loads(raw)
    assert delta['schema'] == 'reviewed_lammps_metadata_source_delta/1'
    assert delta['source'] == 'fibernet/sim/accelerated.py'
    assert delta['baseline_main'] == 'a3e77458415f741fa19af79913e2707cefd6b002'
    assert len(delta['hunks']) == 2, 'LAMMPS metadata delta requires two insertions'
    return delta


def reviewed_lammps_metadata_source():
    delta = lammps_metadata_source_delta()
    source = reviewed_saved_energy_diagnostics_source()
    assert hashlib.sha256(source).hexdigest() == delta['base_sha256'], 'LAMMPS metadata delta base changed'
    for hunk in delta['hunks']:
        before, after = hunk['before'].encode('utf-8'), hunk['after'].encode('utf-8')
        assert before and source.count(before) == 1, 'LAMMPS metadata insertion is not unique'
        source = source.replace(before, after, 1)
    assert hashlib.sha256(source).hexdigest() == delta['result_sha256'], 'reviewed LAMMPS metadata source changed'
    return source


def restore_reviewed_lammps_metadata_source(source):
    delta = lammps_metadata_source_delta()
    assert hashlib.sha256(source).hexdigest() == delta['result_sha256'], 'unreviewed LAMMPS metadata source'
    for hunk in reversed(delta['hunks']):
        before, after = hunk['before'].encode('utf-8'), hunk['after'].encode('utf-8')
        assert after and source.count(after) == 1, 'LAMMPS metadata removal is not unique'
        source = source.replace(after, before, 1)
    assert source == reviewed_saved_energy_diagnostics_source(), 'LAMMPS metadata inverse changed prior source'
    return source


# Exact host-control validation change; no native kernel or result normalization.
_TAICHI_CONTROLS_BEFORE = "        if explicit_mass or record_state or units is not None:\n            fixed_nodes = _validated_fixed_rows(fixed_nodes, num_nodes)\n            _strict_state_ids(list(node_ids))\n            for name, value in (('dt', dt), ('dashpot', dashpot), ('drag', drag)):\n                scalar = _finite_real_array(value, name, ())\n                if (name == 'dt' and scalar <= 0) or (name != 'dt' and scalar < 0):\n                    raise ValueError(f'{name} is outside the supported mass/state range')\n            if external_force is not None:\n                external_force = _finite_real_array(external_force, 'external_force', (num_nodes, dim))\n"
_TAICHI_CONTROLS_AFTER = "        if explicit_mass or record_state or units is not None:\n            _strict_state_ids(list(node_ids))\n\n        # Caller controls have the same contract with or without mass/state output.\n        fixed_nodes = _validated_fixed_rows(fixed_nodes, num_nodes)\n        for name, value in (('dt', dt), ('dashpot', dashpot), ('drag', drag)):\n            scalar = _finite_real_array(value, name, ())\n            if (name == 'dt' and scalar <= 0) or (name != 'dt' and scalar < 0):\n                raise ValueError(f'{name} is outside the supported range')\n        if external_force is not None:\n            external_force = _finite_real_array(external_force, 'external_force', (num_nodes, dim))\n"


def reviewed_taichi_controls_source(reverse=None):
    base = reviewed_lammps_metadata_source()
    assert hashlib.sha256(base).hexdigest() == 'b77aaa7b37216174d97805b4529f1dbaec3d3099106f630221b60147124ccf82', 'controls baseline changed'
    before = _TAICHI_CONTROLS_BEFORE.encode('utf-8')
    after = _TAICHI_CONTROLS_AFTER.encode('utf-8')
    assert before and base.count(before) == 1, 'controls insertion is not unique'
    current = base.replace(before, after, 1)
    assert hashlib.sha256(current).hexdigest() == '59c8761ccef2840a4437aebbacece9e93c94f3f4f6a051d74c31c114381c4a65', 'reviewed controls source changed'
    if reverse is None:
        return current
    assert reverse == current, 'unreviewed controls source'
    assert after and reverse.count(after) == 1, 'controls removal is not unique'
    restored = reverse.replace(after, before, 1)
    assert restored == base, 'controls inverse changed prior source'
    return restored


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


# Exact PR101-controls then saved-reference lineage. The separately qualified
# main53-only reference fixture is not an alternative branch admitted here.
LAMMPS_REFERENCE_METHOD = '    def lammps_displacement_relationship(self) -> Dict:\n        """Inspect saved displacement/reference consistency, not native accuracy.\n\n        Missing historical/no-state references remain unavailable and are never\n        synthesized. Malformed declared references raise rather than downgrade.\n        """\n        from ._lammps_result import displacement_relationship_status\n        return displacement_relationship_status(self)\n\n'
LAMMPS_REFERENCE_PARENT_SHA256 = '59c8761ccef2840a4437aebbacece9e93c94f3f4f6a051d74c31c114381c4a65'
LAMMPS_REFERENCE_RESULT_SHA256 = 'e86c85d10d8e05057b1752df189ab8cb210c15a1269a2a1a21a70ae7a680f48f'
LAMMPS_REFERENCE_METHOD_SHA256 = '749ae4b05c3945e04f6e65a29f4572fe07bb22a35030e203c72090b63ee6451a'


def reviewed_lammps_reference_source(reverse=None):
    method = LAMMPS_REFERENCE_METHOD.encode('utf-8')
    assert hashlib.sha256(method).hexdigest() == LAMMPS_REFERENCE_METHOD_SHA256, 'reviewed reference method literal changed'
    anchor = b'    def viscoelastic_energy_diagnostics(self) -> Dict:'
    prior = reviewed_taichi_controls_source()
    assert hashlib.sha256(prior).hexdigest() == LAMMPS_REFERENCE_PARENT_SHA256, 'reference delta parent changed'
    assert prior.count(anchor) == 1 and method not in prior, 'reference insertion is not unique'
    current = prior.replace(anchor, method + anchor, 1)
    assert hashlib.sha256(current).hexdigest() == LAMMPS_REFERENCE_RESULT_SHA256, 'reviewed reference source changed'
    if reverse is None:
        return current
    assert reverse == current, 'unreviewed reference source'
    assert reverse.count(method + anchor) == 1, 'reference removal is not unique'
    restored = reverse.replace(method + anchor, anchor, 1)
    assert restored == prior, 'reference inverse changed prior source'
    return restored


def remove_reviewed_lammps_reference(result_node, *, require_present=False):
    # Absence is permitted only for historical ASTs. Current-source admission
    # separately requires the complete pinned new bytes, including this method.
    reviewed_lammps_reference_source()
    node = copy.deepcopy(result_node)
    methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == 'lammps_displacement_relationship']
    assert len(methods) <= 1, 'duplicate reference status method'
    assert methods or not require_present, 'missing reference status method'
    if methods:
        expected = ast.parse('class SimResult:\n' + LAMMPS_REFERENCE_METHOD).body[0].body[0]
        assert canonical_ast(methods[0]) == canonical_ast(expected), 'reference status method changed'
        node.body.remove(methods[0])
    return node


# Reviewed optional-import layer over the existing batch-report source.
EASY_OPTIONAL_IMPORT_BASE_SHA256 = 'fea8496a65d68b9ee706418b538a242fad22992fa41763beefb6f21df1736c85'
EASY_OPTIONAL_IMPORT_SHA256 = '55bb90fa65932457cd8b552522c17d459f33dae06e221373ca04707a0f94c762'
_EASY_OPTIONAL_IMPORT_HUNKS = (
    (
        (
            b'import numpy as np\n'
            b'\n'
            b'from fibernet.core.structure_graph import StructureGraph\n'
            b'from fibernet.gen.pattern import pattern_2d, pattern_3d\n'
            b'from fibernet.sim.accelerated import TaichiEngine, SimResult\n'
            b'from fibernet.sim._viscoelastic_fem import ViscoelasticResult as _ViscoelasticResult\n'
            b'from fibernet.sim._taichi_geometry import (\n'
            b'    preflight_taichi_geometry as _preflight_taichi_geometry,\n'
            b'    validate_geometry_policy as _validate_geometry_policy,\n'
            b')\n'
            b'\n'
            b'\n'
            b'def show(\n'
        ),
        (
            b'import numpy as np\n'
            b'\n'
            b'from fibernet.core.structure_graph import StructureGraph\n'
            b'import fibernet.sim as _sim\n'
            b'from fibernet.gen.pattern import pattern_2d, pattern_3d\n'
            b'from fibernet.sim._viscoelastic_fem import ViscoelasticResult as _ViscoelasticResult\n'
            b'from fibernet.sim._taichi_geometry import (\n'
            b'    preflight_taichi_geometry as _preflight_taichi_geometry,\n'
            b'    validate_geometry_policy as _validate_geometry_policy,\n'
            b')\n'
            b'\n'
            b'\n'
            b'# Keep provider-independent utilities usable without probing a native engine.\n'
            b'# Explicit alias access still returns the original implementation object.\n'
            b'_LAZY_EXPORTS = {\n'
            b'    "TaichiEngine": "fibernet.sim.accelerated",\n'
            b'    "SimResult": "fibernet.sim.accelerated",\n'
            b'    "render_graph": "fibernet.viz.render",\n'
            b'    "render_deformation": "fibernet.viz.render",\n'
            b'}\n'
            b'for _name in _LAZY_EXPORTS:\n'
            b'    globals().pop(_name, None)\n'
            b'del _name\n'
            b'\n'
            b'\n'
            b'def _taichi_engine_class():\n'
            b'    # Preserve caller/test overrides of the historical module-level alias.\n'
            b'    if "TaichiEngine" in globals():\n'
            b'        return globals()["TaichiEngine"]\n'
            b'    return __getattr__("TaichiEngine")\n'
            b'\n'
            b'\n'
            b'def show(\n'
        ),
    ),
    (
        (
            b'    *,\n'
            b'    stretch_ratio: Optional[float] = None,\n'
            b'    **kwargs,\n'
            b') -> Union[SimResult, _ViscoelasticResult]:\n'
            b'    """Run spring dynamics, beam-frame stretch or a viscoelastic frame history.\n'
            b'\n'
            b'    Parameters\n'
        ),
        (
            b'    *,\n'
            b'    stretch_ratio: Optional[float] = None,\n'
            b'    **kwargs,\n'
            b') -> Union[_sim.SimResult, _ViscoelasticResult]:\n'
            b'    """Run spring dynamics, beam-frame stretch or a viscoelastic frame history.\n'
            b'\n'
            b'    Parameters\n'
        ),
    ),
    (
        (
            b'        if (isinstance(num_threads, (bool, np.bool_)) or\n'
            b'                not isinstance(num_threads, (int, np.integer)) or num_threads <= 0):\n'
            b'            raise ValueError("num_threads must be a positive integer")\n'
            b'        method = TaichiEngine.stretch_test if mode == "stretch" else TaichiEngine.dynamics\n'
            b'        kwargs.setdefault("geometry_policy", "reject_internal_points")\n'
            b'        method_kwargs = dict(kwargs)\n'
            b'        if mode == "stretch":\n'
        ),
        (
            b'        if (isinstance(num_threads, (bool, np.bool_)) or\n'
            b'                not isinstance(num_threads, (int, np.integer)) or num_threads <= 0):\n'
            b'            raise ValueError("num_threads must be a positive integer")\n'
            b'        engine_class = _taichi_engine_class()\n'
            b'        method = engine_class.stretch_test if mode == "stretch" else engine_class.dynamics\n'
            b'        kwargs.setdefault("geometry_policy", "reject_internal_points")\n'
            b'        method_kwargs = dict(kwargs)\n'
            b'        if mode == "stretch":\n'
        ),
    ),
    (
        (
            b'            method_kwargs["target_stretch"] = strain\n'
            b'        signature(method).bind(None, graph, **method_kwargs)\n'
            b'        _preflight_taichi_geometry(graph, geometry_policy=kwargs["geometry_policy"])\n'
            b'        engine = TaichiEngine(arch=arch, num_threads=int(num_threads))\n'
            b'        if mode == "stretch":\n'
            b'            result = engine.stretch_test(graph, target_stretch=strain, **kwargs)\n'
            b'        else:\n'
        ),
        (
            b'            method_kwargs["target_stretch"] = strain\n'
            b'        signature(method).bind(None, graph, **method_kwargs)\n'
            b'        _preflight_taichi_geometry(graph, geometry_policy=kwargs["geometry_policy"])\n'
            b'        engine = engine_class(arch=arch, num_threads=int(num_threads))\n'
            b'        if mode == "stretch":\n'
            b'            result = engine.stretch_test(graph, target_stretch=strain, **kwargs)\n'
            b'        else:\n'
        ),
    ),
    (
        (
            b'    if not pending:\n'
            b'        return str(csv_file)\n'
            b'\n'
            b'    engine = TaichiEngine()\n'
            b'    output_path.mkdir(parents=True, exist_ok=True)\n'
            b'\n'
            b'    def write_checkpoint():\n'
        ),
        (
            b'    if not pending:\n'
            b'        return str(csv_file)\n'
            b'\n'
            b'    engine = _taichi_engine_class()()\n'
            b'    output_path.mkdir(parents=True, exist_ok=True)\n'
            b'\n'
            b'    def write_checkpoint():\n'
        ),
    ),
    (
        (
            b'\n'
            b'\n'
            b'def __getattr__(name):\n'
            b'    # Preserve the historical easy-module re-exports without loading plotting\n'
            b'    # for simulate(), training helpers, or importing the top-level package.\n'
            b'    if name in {"render_graph", "render_deformation"}:\n'
            b'        from importlib import import_module\n'
            b'        value = getattr(import_module("fibernet.viz.render"), name)\n'
            b'        globals()[name] = value\n'
            b'        return value\n'
            b'    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")\n'
            b'\n'
            b'\n'
            b'def __dir__():\n'
            b'    return sorted(set(globals()) | {"render_graph", "render_deformation"})\n'
            b'\n'
            b'\n'
            b'# Preserve the historical wildcard surface, including incidental public imports.\n'
            b'# A wildcard explicitly requests the renderers; importing this module does not.\n'
            b'__all__ = [name for name in globals() if not name.startswith("_")]\n'
            b'__all__ += ["render_graph", "render_deformation"]\n'
        ),
        (
            b'\n'
            b'\n'
            b'def __getattr__(name):\n'
            b"    # Preserve historical aliases; propagate the owning implementation's error.\n"
            b'    if name in _LAZY_EXPORTS:\n'
            b'        from importlib import import_module\n'
            b'        value = getattr(import_module(_LAZY_EXPORTS[name]), name)\n'
            b'        globals()[name] = value\n'
            b'        return value\n'
            b'    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")\n'
            b'\n'
            b'\n'
            b'def __dir__():\n'
            b'    return sorted(set(globals()) | set(_LAZY_EXPORTS))\n'
            b'\n'
            b'\n'
            b'# Preserve the historical wildcard surface, including incidental public imports.\n'
            b'# A wildcard explicitly requests the renderers; importing this module does not.\n'
            b'__all__ = [name for name in globals() if not name.startswith("_")]\n'
            b'# Retain the original wildcard order as well as its public names.\n'
            b'_index = __all__.index("pattern_3d") + 1\n'
            b'__all__[_index:_index] = ["TaichiEngine", "SimResult"]\n'
            b'del _index\n'
            b'__all__ += ["render_graph", "render_deformation"]\n'
        ),
    ),
)


def reviewed_easy_optional_import_source(reverse=None):
    """Admit one exact lazy-import delta and restore its exact historical base."""
    base = reviewed_easy_batch_report_source()
    assert hashlib.sha256(base).hexdigest() == EASY_OPTIONAL_IMPORT_BASE_SHA256, 'optional-import base changed'
    source = base if reverse is None else reverse
    if reverse is not None:
        assert hashlib.sha256(source).hexdigest() == EASY_OPTIONAL_IMPORT_SHA256, 'unreviewed optional-import inverse input'
    for before, after in (_EASY_OPTIONAL_IMPORT_HUNKS if reverse is None
                          else reversed(_EASY_OPTIONAL_IMPORT_HUNKS)):
        if reverse is not None:
            before, after = after, before
        assert before and source.count(before) == 1, 'optional-import delta is not uniquely located'
        source = source.replace(before, after, 1)
    expected = EASY_OPTIONAL_IMPORT_SHA256 if reverse is None else EASY_OPTIONAL_IMPORT_BASE_SHA256
    assert hashlib.sha256(source).hexdigest() == expected, 'optional-import delta result changed'
    if reverse is not None:
        assert source == base, 'optional-import inverse changed historical bytes'
    return source


def require_public_source(relative, raw):
    if relative == 'fibernet/__init__.py':
        assert raw in (public_baseline(relative).read_bytes(), reviewed_facade(relative), reviewed_fiber_network_profile_facade(), reviewed_octet_fcc_facade()), 'public source bytes changed: '+relative
        return
    if relative == 'fibernet/sim/__init__.py':
        assert raw in (prior_sim_facade(), reviewed_facade(relative), reviewed_nodal_controls_facade(), reviewed_thermoelastic_facade(), reviewed_lammps_capability_facade(), reviewed_archived_ge_facade()), 'public source bytes changed: '+relative
        return
    if relative == 'fibernet/easy.py':
        assert raw in (public_baseline(relative).read_bytes(), reviewed_easy_stretch_ratio_source(), reviewed_easy_rl_failure_source(), reviewed_taichi_geometry_source(relative), reviewed_easy_show_source(), reviewed_easy_batch_report_source(), reviewed_easy_optional_import_source()), 'public source bytes changed: '+relative
        return
    if relative == 'fibernet/api.py':
        assert raw in (public_baseline(relative).read_bytes(), reviewed_native_io_source(), reviewed_legacy_analysis_source()), 'public source bytes changed: '+relative
        return
    if relative in FACADE_PINS:
        prior = prior_sim_facade() if relative == 'fibernet/sim/__init__.py' else public_baseline(relative).read_bytes()
        assert raw in (prior, reviewed_facade(relative)), 'public source bytes changed: '+relative
        return
    expected=public_baseline(relative).read_bytes()
    if relative == 'fibernet/sim/accelerated.py':
        assert raw in (expected, reviewed_result_identity_source(), reviewed_spring_provenance_source(), reviewed_taichi_geometry_source(relative), reviewed_saved_energy_diagnostics_source(), reviewed_lammps_metadata_source(), reviewed_taichi_controls_source(), reviewed_lammps_reference_source()), 'public source bytes changed: '+relative
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
    node=remove_reviewed_saved_energy_diagnostics(remove_reviewed_lammps_reference(result_node))
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
    metadata_tree = ast.parse(reviewed_lammps_metadata_source().decode('utf-8'))
    metadata_result = next(n for n in metadata_tree.body if getattr(n,'name',None)=='SimResult')
    for name in ('to_dict', 'load'):
        method=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name)
        expected_method=next(n for n in expected_result.body if isinstance(n,ast.FunctionDef) and n.name==name)
        provenance_method=next(n for n in provenance_result.body if isinstance(n,ast.FunctionDef) and n.name==name)
        metadata_method=next(n for n in metadata_result.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert canonical_ast(method) in (canonical_ast(expected_method), canonical_ast(provenance_method), canonical_ast(metadata_method)), 'public result method changed: '+name
        old_method=next(n for n in old_result.body if isinstance(n,ast.FunctionDef) and n.name==name)
        node.body[node.body.index(method)]=old_method
    detailed=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='compute_detailed')
    node.body[node.body.index(detailed)]=remove_public_lammps_branch(detailed,'SimResult.compute_detailed')
    return node


class EasyOptionalImportSourceAdmission(unittest.TestCase):
    def test_exact_candidate_and_ordered_inverse(self):
        current = reviewed_easy_optional_import_source()
        self.assertEqual(current, (ROOT/'fibernet/easy.py').read_bytes())
        require_public_source('fibernet/easy.py', current)
        self.assertEqual(reviewed_easy_optional_import_source(reverse=current),
                         reviewed_easy_batch_report_source())
        with self.assertRaises(AssertionError):
            reviewed_easy_optional_import_source(reverse=reviewed_easy_batch_report_source())

    def test_unreviewed_lazy_layer_mutations_are_refused(self):
        raw = reviewed_easy_optional_import_source()
        changes = (
            (b'"TaichiEngine": "fibernet.sim.accelerated"', b'"TaichiEngine": "fibernet.sim"'),
            (b'"SimResult": "fibernet.sim.accelerated"', b'"SimResult": "fibernet.easy"'),
            (b'if "TaichiEngine" in globals():', b'if False:'),
            (b'return globals()["TaichiEngine"]', b'return None'),
            (b'return __getattr__("TaichiEngine")', b'return __getattr__("SimResult")'),
            (b'globals().pop(_name, None)', b'pass'),
            (b'engine = engine_class(arch=arch, num_threads=int(num_threads))',
             b'engine = engine_class(arch=arch, num_threads=1)'),
            (b'engine = _taichi_engine_class()()', b'engine = None'),
            (b'__all__[_index:_index] = ["TaichiEngine", "SimResult"]',
             b'__all__[_index:_index] = ["SimResult", "TaichiEngine"]'),
        )
        mutations = [raw + b'\npass\n', raw + b'\n# unreviewed\n',
                     raw.replace(b'\n', b'\r\n')]
        for before, after in changes:
            self.assertEqual(raw.count(before), 1)
            mutations.append(raw.replace(before, after, 1))
        for changed in mutations:
            with self.subTest(tail=changed[-40:]):
                with self.assertRaises(AssertionError):
                    require_public_source('fibernet/easy.py', changed)
                with self.assertRaises(AssertionError):
                    reviewed_easy_optional_import_source(reverse=changed)

    def test_inline_delta_is_not_self_accepting(self):
        before, after = _EASY_OPTIONAL_IMPORT_HUNKS[0]
        corrupted = ((before, after + b'\npass\n'), *_EASY_OPTIONAL_IMPORT_HUNKS[1:])
        with patch.dict(globals(), {'_EASY_OPTIONAL_IMPORT_HUNKS': corrupted}):
            with self.assertRaises(AssertionError):
                reviewed_easy_optional_import_source()


class StretchRatioSourceAdmission(unittest.TestCase):
    def test_exact_old_and_new_versions_are_admitted(self):
        old = public_baseline('fibernet/easy.py').read_bytes()
        new = reviewed_easy_stretch_ratio_source()
        self.assertIn((ROOT/'fibernet/easy.py').read_bytes(),
                      (new, reviewed_easy_rl_failure_source(), reviewed_taichi_geometry_source('fibernet/easy.py'), reviewed_easy_show_source(), reviewed_easy_batch_report_source(), reviewed_easy_optional_import_source()))
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
        self.assertEqual(reviewed_easy_show_source(reverse=reviewed_easy_batch_report_source(reverse=reviewed_easy_optional_import_source(reverse=(ROOT/'fibernet/easy.py').read_bytes()))), current)
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
                expected = (reviewed_taichi_controls_source() if relative == 'fibernet/sim/accelerated.py'
                            else reviewed_easy_optional_import_source())
                self.assertEqual(reviewed_lammps_reference_source(reverse=(ROOT/relative).read_bytes())
                                 if relative == 'fibernet/sim/accelerated.py' else (ROOT/relative).read_bytes(), expected)
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
        self.assertEqual(current, restore_reviewed_lammps_metadata_source(
            reviewed_taichi_controls_source(reverse=reviewed_lammps_reference_source(reverse=SOURCE.read_bytes()))))
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


class LammpsMetadataSourceAdmission(unittest.TestCase):
    def result_node(self, raw):
        return next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == 'SimResult')

    def test_exact_reversible_delta_changes_only_two_methods(self):
        prior = reviewed_saved_energy_diagnostics_source()
        current = reviewed_lammps_metadata_source()
        self.assertEqual(current, reviewed_taichi_controls_source(reverse=reviewed_lammps_reference_source(reverse=SOURCE.read_bytes())))
        self.assertEqual(restore_reviewed_lammps_metadata_source(current), prior)
        require_public_source('fibernet/sim/accelerated.py', prior)
        require_public_source('fibernet/sim/accelerated.py', current)
        old, new = ast.parse(prior), ast.parse(current)
        old_result, new_result = self.result_node(prior), self.result_node(current)
        old_methods = {n.name:n for n in old_result.body if isinstance(n, ast.FunctionDef)}
        new_methods = {n.name:n for n in new_result.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(set(old_methods), set(new_methods))
        self.assertEqual({name for name in old_methods if canonical_ast(old_methods[name]) != canonical_ast(new_methods[name])}, {'to_dict', 'load'})
        for name in ('to_dict', 'load'):
            new_result.body[new_result.body.index(new_methods[name])] = old_methods[name]
        new.body[next(i for i,n in enumerate(new.body) if isinstance(n,ast.ClassDef) and n.name == 'SimResult')] = new_result
        self.assertEqual(canonical_ast(old), canonical_ast(new))
        self.assertEqual(canonical_ast(restore_historical_save(self.result_node(prior))),
                         canonical_ast(restore_historical_save(self.result_node(current))))

    def test_guard_import_name_target_and_unrelated_statements_are_rejected(self):
        current = reviewed_lammps_metadata_source()
        for name, target in (('to_dict', 'self.metadata'), ('load', 'data')):
            method = next(n for n in self.result_node(current).body if isinstance(n,ast.FunctionDef) and n.name == name)
            start, end = method.lineno-1, method.end_lineno
            lines = current.decode('utf-8').splitlines(keepends=True)
            original = ''.join(lines[start:end])
            mutations = (
                (target + ".get('backend') == 'lammps'", target + ".get('backend') == 'spring'"),
                ('from ._lammps_result import validate_persisted_metadata', 'from ._other_result import validate_persisted_metadata'),
                ('from ._lammps_result import validate_persisted_metadata', 'from ._lammps_result import validate_other_metadata as validate_persisted_metadata'),
                ('validate_persisted_metadata(' + target + ')', 'validate_other_metadata(' + target + ')'),
                ('validate_persisted_metadata(' + target + ')', 'validate_persisted_metadata(None)'),
                ('validate_persisted_metadata(' + target + ')', 'pass'),
            )
            for before, after in mutations:
                with self.subTest(method=name, mutation=before, replacement=after):
                    self.assertEqual(original.count(before), 1)
                    changed = (''.join(lines[:start])+original.replace(before,after,1)+''.join(lines[end:])).encode('utf-8')
                    with self.assertRaises(AssertionError): require_public_source('fibernet/sim/accelerated.py', changed)
                    with self.assertRaises(AssertionError): restore_reviewed_lammps_metadata_source(changed)
                    with self.assertRaises(AssertionError): restore_historical_save(self.result_node(changed))
            changed = self.result_node(current)
            changed_method = next(n for n in changed.body if isinstance(n,ast.FunctionDef) and n.name == name)
            changed_method.body.append(ast.Pass())
            with self.subTest(method=name, mutation='unrelated appended statement'), self.assertRaises(AssertionError):
                restore_historical_save(changed)

    def test_fixture_and_source_bytes_are_not_self_accepting(self):
        fixture = ROOT/'tests/fixtures/lammps_metadata_persistence_v1/accelerated_delta.json'
        original = Path.read_bytes
        def changed_read(path):
            raw = original(path)
            return raw+b'\n' if path == fixture else raw
        with patch.object(Path, 'read_bytes', changed_read):
            with self.assertRaisesRegex(AssertionError, 'reviewed LAMMPS metadata delta changed'):
                reviewed_lammps_metadata_source()
        raw = reviewed_lammps_metadata_source()
        for changed in (raw+b'\n# unreviewed\n', raw.replace(b'\n',b'\r\n'),
                        raw.replace(b'def _interpolate_schedule(schedule, step, total_steps):',
                                    b'def _interpolate_schedule(schedule, step, total_steps, extra=None):',1)):
            self.assertNotEqual(changed, raw)
            with self.assertRaises(AssertionError): require_public_source('fibernet/sim/accelerated.py', changed)
            with self.assertRaises(AssertionError): restore_reviewed_lammps_metadata_source(changed)

    def test_new_fixture_has_an_exact_byte_preserving_attribute(self):
        rule = 'tests/fixtures/lammps_metadata_persistence_v1/accelerated_delta.json -text'
        self.assertEqual((ROOT/'.gitattributes').read_text(encoding='utf-8').splitlines().count(rule), 1)


class TaichiControlsSourceAdmission(unittest.TestCase):
    def test_exact_reversible_delta_changes_only_dynamics_body(self):
        prior = reviewed_lammps_metadata_source()
        current = reviewed_taichi_controls_source()
        self.assertEqual(current, reviewed_lammps_reference_source(reverse=SOURCE.read_bytes()))
        self.assertEqual(reviewed_taichi_controls_source(reverse=current), prior)
        require_public_source('fibernet/sim/accelerated.py', prior)
        require_public_source('fibernet/sim/accelerated.py', current)
        old, new = ast.parse(prior), ast.parse(current)
        def dynamics(tree):
            engine = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                          and n.name == 'TaichiEngine')
            return next(n for n in engine.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'dynamics')
        old_method, new_method = dynamics(old), dynamics(new)
        self.assertEqual(canonical_ast(old_method.args), canonical_ast(new_method.args))
        new_method.body = old_method.body
        self.assertEqual(canonical_ast(old), canonical_ast(new))

    def test_controls_and_unrelated_mutations_are_refused(self):
        raw = reviewed_taichi_controls_source()
        mutations = (
            (b'_validated_fixed_rows(fixed_nodes, num_nodes)', b'list(fixed_nodes or [])'),
            (b"name == 'dt' and scalar <= 0", b"name == 'dt' and scalar < 0"),
            (b"scalar = _finite_real_array(value, name, ())", b'scalar = value'),
            (b"_finite_real_array(external_force, 'external_force', (num_nodes, dim))",
             b'np.asarray(external_force)'),
            (b'fixed_arr[ni] = 1', b'fixed_arr[ni] = 0'),
            (b'def _interpolate_schedule(schedule, step, total_steps):',
             b'def _interpolate_schedule(schedule, step, total_steps, extra=None):'),
        )
        changed_sources = [raw + b'\npass\n', raw.replace(b'\n', b'\r\n')]
        for before, after in mutations:
            self.assertIn(before, raw)
            changed_sources.append(raw.replace(before, after, 1))
        for changed in changed_sources:
            with self.subTest(digest=hashlib.sha256(changed).hexdigest()):
                with self.assertRaises(AssertionError):
                    require_public_source('fibernet/sim/accelerated.py', changed)
                with self.assertRaises(AssertionError):
                    reviewed_taichi_controls_source(reverse=changed)

    def test_delta_literals_are_not_self_accepting(self):
        for name in ('_TAICHI_CONTROLS_BEFORE', '_TAICHI_CONTROLS_AFTER'):
            with self.subTest(name=name), patch.dict(globals(), {name: globals()[name] + '\npass\n'}):
                with self.assertRaises(AssertionError):
                    reviewed_taichi_controls_source()


class LammpsReferenceSourceAdmission(unittest.TestCase):
    @staticmethod
    def result_node(raw):
        return next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == 'SimResult')

    def test_exact_current_forward_inverse_and_historical_admission(self):
        prior = reviewed_taichi_controls_source()
        current = reviewed_lammps_reference_source()
        self.assertEqual(current, SOURCE.read_bytes())
        self.assertEqual(reviewed_lammps_reference_source(reverse=current), prior)
        for raw in (prior, current):
            require_public_source('fibernet/sim/accelerated.py', raw)
        with self.assertRaises(AssertionError):
            reviewed_lammps_reference_source(reverse=prior)

    def test_one_method_only_and_all_old_nodes_kernels_unchanged(self):
        old = ast.parse(reviewed_taichi_controls_source())
        new = ast.parse(reviewed_lammps_reference_source())
        old_result = self.result_node(reviewed_taichi_controls_source())
        result = self.result_node(reviewed_lammps_reference_source())
        old_names = [n.name for n in old_result.body if isinstance(n, ast.FunctionDef)]
        names = [n.name for n in result.body if isinstance(n, ast.FunctionDef)]
        self.assertEqual(set(names)-set(old_names), {'lammps_displacement_relationship'})
        self.assertEqual(len(names), len(old_names)+1)
        index = next(i for i,n in enumerate(new.body) if isinstance(n, ast.ClassDef) and n.name == 'SimResult')
        new.body[index] = remove_reviewed_lammps_reference(new.body[index], require_present=True)
        self.assertEqual(canonical_ast(old), canonical_ast(new))
        self.assertEqual(canonical_ast(restore_historical_save(old_result)),
                         canonical_ast(restore_historical_save(result)))

    def test_method_and_unrelated_byte_mutations_rejected(self):
        raw = reviewed_lammps_reference_source()
        changes = (
            ('lammps_displacement_relationship(self)', 'renamed_relationship(self)'),
            ('lammps_displacement_relationship(self)', 'lammps_displacement_relationship(self, extra=None)'),
            ('from ._lammps_result import displacement_relationship_status', 'from ._other_result import displacement_relationship_status'),
            ('from ._lammps_result import displacement_relationship_status', 'from ._lammps_result import other_status as displacement_relationship_status'),
            ('return displacement_relationship_status(self)', 'return displacement_relationship_status(None)'),
            ('return displacement_relationship_status(self)', 'return other_status(self)'),
            ('return displacement_relationship_status(self)', "return {'availability': 'consistent'}"),
            ('def _interpolate_schedule(schedule, step, total_steps):', 'def _interpolate_schedule(schedule, step, total_steps, extra=None):'),
        )
        variants = []
        for before, after in changes:
            self.assertEqual(raw.count(before.encode()), 1)
            variants.append(raw.replace(before.encode(), after.encode(), 1))
        variants.extend((raw+b'\n# unreviewed\n', raw.replace(b'\n', b'\r\n')))
        for changed in variants:
            with self.subTest(changed=hashlib.sha256(changed).hexdigest()):
                with self.assertRaises(AssertionError):
                    require_public_source('fibernet/sim/accelerated.py', changed)
                with self.assertRaises(AssertionError):
                    reviewed_lammps_reference_source(reverse=changed)

    def test_strict_ast_mutations_duplicates_and_missing_are_not_removed(self):
        raw = reviewed_lammps_reference_source()
        original = self.result_node(raw)
        for mutation in ('missing', 'duplicate', 'rename', 'signature', 'decorator', 'doc', 'import', 'return', 'append'):
            result = copy.deepcopy(original)
            method = next(n for n in result.body if isinstance(n, ast.FunctionDef) and n.name == 'lammps_displacement_relationship')
            if mutation == 'missing': result.body.remove(method)
            elif mutation == 'duplicate': result.body.append(copy.deepcopy(method))
            elif mutation == 'rename': method.name = 'other_status'
            elif mutation == 'signature': method.args.args.append(ast.arg(arg='extra'))
            elif mutation == 'decorator': method.decorator_list.append(ast.Name(id='staticmethod', ctx=ast.Load()))
            elif mutation == 'doc': method.body[0].value.value += ' changed'
            elif mutation == 'import': method.body[1].module = '_other_result'
            elif mutation == 'return': method.body[-1].value = ast.Constant(value=True)
            else: method.body.append(ast.Pass())
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                remove_reviewed_lammps_reference(result, require_present=True)
        # Historical absence is readable, but an unrecognized extra method is
        # retained for the surrounding full-AST comparison, never silently lost.
        renamed = copy.deepcopy(original)
        next(n for n in renamed.body if getattr(n, 'name', None) == 'lammps_displacement_relationship').name = 'other_status'
        normalized = remove_reviewed_lammps_reference(renamed)
        self.assertTrue(any(getattr(n, 'name', None) == 'other_status' for n in normalized.body))

    def test_literal_tampering_is_not_self_accepting(self):
        for changed in (LAMMPS_REFERENCE_METHOD+'    # unreviewed\n',
                        LAMMPS_REFERENCE_METHOD.replace('return displacement_relationship_status(self)', 'return True')):
            with self.subTest(literal=changed[-40:]), patch.dict(globals(), LAMMPS_REFERENCE_METHOD=changed):
                with self.assertRaisesRegex(AssertionError, 'reference method literal changed'):
                    reviewed_lammps_reference_source()


class LammpsReferenceControlsComposition(unittest.TestCase):
    def test_exact_ordered_inverse_reaches_main53(self):
        combined = reviewed_lammps_reference_source()
        controls = reviewed_lammps_reference_source(reverse=combined)
        baseline = reviewed_taichi_controls_source(reverse=controls)
        self.assertEqual(combined, SOURCE.read_bytes())
        self.assertEqual(baseline, reviewed_lammps_metadata_source())
        self.assertEqual(hashlib.sha256(baseline).hexdigest(),
                         'b77aaa7b37216174d97805b4529f1dbaec3d3099106f630221b60147124ccf82')
        with self.assertRaises(AssertionError):
            reviewed_taichi_controls_source(reverse=combined)
        with self.assertRaises(AssertionError):
            reviewed_lammps_reference_source(reverse=controls)

    def test_standalone_reference_is_not_an_alternate_admitted_lineage(self):
        baseline = reviewed_lammps_metadata_source()
        anchor = b'    def viscoelastic_energy_diagnostics(self) -> Dict:'
        standalone = baseline.replace(anchor, LAMMPS_REFERENCE_METHOD.encode('utf-8') + anchor, 1)
        self.assertEqual(hashlib.sha256(standalone).hexdigest(),
                         '13eaa179822bdbc8fc48f686f2bcee0b072eed1ead381a85763bc2e27910adbc')
        with self.assertRaises(AssertionError):
            require_public_source('fibernet/sim/accelerated.py', standalone)
        with self.assertRaises(AssertionError):
            reviewed_lammps_reference_source(reverse=standalone)

    def test_combined_source_rejects_control_mutation_and_parent_literal_tamper(self):
        combined = reviewed_lammps_reference_source()
        original = _TAICHI_CONTROLS_AFTER.encode('utf-8')
        self.assertEqual(combined.count(original), 1)
        for before, after in (
                ('fixed_nodes = _validated_fixed_rows(fixed_nodes, num_nodes)', 'fixed_nodes = list(fixed_nodes or [])'),
                ("name == 'dt' and scalar <= 0", "name == 'dt' and scalar < 0"),
                ("scalar = _finite_real_array(value, name, ())", 'scalar = value')):
            changed_block = original.replace(before.encode(), after.encode(), 1)
            self.assertNotEqual(changed_block, original)
            changed = combined.replace(original, changed_block, 1)
            with self.subTest(control=before):
                with self.assertRaises(AssertionError):
                    require_public_source('fibernet/sim/accelerated.py', changed)
                with self.assertRaises(AssertionError):
                    reviewed_lammps_reference_source(reverse=changed)
        for name in ('_TAICHI_CONTROLS_BEFORE', '_TAICHI_CONTROLS_AFTER'):
            with self.subTest(literal=name), patch.dict(globals(), {name: globals()[name] + '\npass\n'}):
                with self.assertRaises(AssertionError):
                    reviewed_lammps_reference_source()


class BackendLazySourceComposition(unittest.TestCase):
    def test_current_pair_preserves_both_ordered_lineages(self):
        accelerated = SOURCE.read_bytes()
        easy = (ROOT/'fibernet/easy.py').read_bytes()
        self.assertEqual(hashlib.sha256(accelerated).hexdigest(), LAMMPS_REFERENCE_RESULT_SHA256)
        self.assertEqual(hashlib.sha256(easy).hexdigest(), EASY_OPTIONAL_IMPORT_SHA256)
        self.assertEqual(accelerated, reviewed_lammps_reference_source())
        self.assertEqual(easy, reviewed_easy_optional_import_source())
        require_public_source('fibernet/sim/accelerated.py', accelerated)
        require_public_source('fibernet/easy.py', easy)
        self.assertEqual(reviewed_taichi_controls_source(
            reverse=reviewed_lammps_reference_source(reverse=accelerated)),
            reviewed_lammps_metadata_source())
        self.assertEqual(reviewed_easy_optional_import_source(reverse=easy),
                         reviewed_easy_batch_report_source())

    def test_mixed_owners_and_unreviewed_pair_members_are_rejected(self):
        accelerated = SOURCE.read_bytes()
        easy = (ROOT/'fibernet/easy.py').read_bytes()
        for relative, expected, other in (
                ('fibernet/sim/accelerated.py', accelerated, easy),
                ('fibernet/easy.py', easy, accelerated)):
            for changed in (other, expected+b'\npass\n', expected.replace(b'\n', b'\r\n')):
                with self.subTest(source=relative, digest=hashlib.sha256(changed).hexdigest()):
                    with self.assertRaises(AssertionError):
                        require_public_source(relative, changed)


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
        self.assertEqual(reviewed_taichi_controls_source(), reviewed_lammps_reference_source(reverse=SOURCE.read_bytes()))

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
        self.assertEqual(reviewed_taichi_controls_source(), reviewed_lammps_reference_source(reverse=SOURCE.read_bytes()))
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
        self.assertEqual(raw, reviewed_lammps_capability_facade(reverse=reviewed_archived_ge_facade(reverse=(ROOT/relative).read_bytes())))
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
        self.assertEqual(raw, reviewed_octet_fcc_facade(reverse=(ROOT/relative).read_bytes()))
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

    def test_reviewed_octet_fcc_facade_is_exact_and_reversible(self):
        relative = 'fibernet/__init__.py'
        raw = reviewed_octet_fcc_facade()
        base = reviewed_fiber_network_profile_facade()
        self.assertEqual(raw, (ROOT/relative).read_bytes())
        self.assertEqual(reviewed_octet_fcc_facade(reverse=raw), base)
        self.assertEqual(reviewed_fiber_network_profile_facade(reverse=base),
                         reviewed_facade(relative))
        for admitted in (public_baseline(relative).read_bytes(), reviewed_facade(relative), base, raw):
            require_public_source(relative, admitted)
        import_line = b'from fibernet.gen.standard_lattice import octet_fcc_v1\n'
        export_line = b'__all__.append("octet_fcc_v1")\n'
        for changed in (raw+b'\npass\n', raw.replace(b'\n', b'\r\n'),
                        raw.replace(import_line, b'', 1), raw.replace(export_line, b'', 1),
                        raw.replace(import_line, import_line+import_line, 1),
                        raw.replace(export_line, export_line+export_line, 1),
                        raw.replace(b'fibernet.gen.standard_lattice', b'fibernet.gen.pattern', 1),
                        raw.replace(b'octet_fcc_v1', b'wrong_octet', 1),
                        raw.replace(export_line, b'__all__.append("wrong_octet")\n', 1),
                        raw.replace(b'FiberNetworkProfile', b'WrongProfile', 1),
                        raw.replace(b'def __getattr__(name):', b'def __getattr__(name): # changed', 1),
                        raw.replace(b'_LAZY_EXPORTS = {', b'_LAZY_EXPORTS = { # changed', 1)):
            self.assertNotEqual(changed, raw)
            with self.assertRaises(AssertionError):
                require_public_source(relative, changed)
            with self.assertRaises(AssertionError):
                reviewed_octet_fcc_facade(reverse=changed)
        original = Path.read_bytes
        fixture = ROOT/'tests/fixtures/standard_octet_v1/root_facade_delta.json'
        for mutation in (lambda value: value+b'\n',
                         lambda value: value.replace(b'octet_fcc_v1', b'wrong_octet', 1)):
            def changed_read(path):
                value = original(path)
                return mutation(value) if path == fixture else value
            with patch.object(Path, 'read_bytes', changed_read):
                with self.assertRaisesRegex(AssertionError, 'reviewed octet FCC facade delta changed'):
                    reviewed_octet_fcc_facade()

    def test_reviewed_lammps_capability_facade_is_exact_and_reversible(self):
        relative = 'fibernet/sim/__init__.py'
        raw = reviewed_lammps_capability_facade()
        self.assertEqual(raw, reviewed_archived_ge_facade(reverse=(ROOT/relative).read_bytes()))
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

    def test_reviewed_archived_ge_facade_is_exact_and_reversible(self):
        relative = 'fibernet/sim/__init__.py'
        raw = reviewed_archived_ge_facade()
        self.assertEqual(raw, (ROOT/relative).read_bytes())
        self.assertEqual(reviewed_archived_ge_facade(reverse=raw),
                         reviewed_lammps_capability_facade())
        require_public_source(relative, raw)
        for changed in (raw+b'\npass\n', raw.replace(b'\n', b'\r\n'),
                        raw.replace(b'import_archived_ge_state', b'wrong_capabilities', 1),
                        raw.replace(b'ArchivedGEProjection', b'WrongProjection', 1),
                        raw.replace(b'fibernet.sim.archived_ge', b'fibernet.sim.accelerated', 1),
                        raw.replace(b'ThermoelasticSizeLimits', b'WrongLimits', 1),
                        raw.replace(b'prepare_nodal_controls', b'wrong_controls', 1),
                        raw.replace(b'def __getattr__(name):', b'def __getattr__(name): # changed', 1)):
            self.assertNotEqual(changed, raw)
            with self.assertRaises(AssertionError):
                require_public_source(relative, changed)
            with self.assertRaises(AssertionError):
                reviewed_archived_ge_facade(reverse=changed)
        original = Path.read_bytes
        fixture = ROOT/'tests/fixtures/optional_facades_v2/sim_archived_ge_delta.json'
        for mutation in (lambda value: value+b'\n',
                         lambda value: value.replace(b'import_archived_ge_state', b'wrong_capabilities', 1)):
            def changed_read(path):
                value = original(path)
                return mutation(value) if path == fixture else value
            with patch.object(Path, 'read_bytes', changed_read):
                with self.assertRaisesRegex(AssertionError, 'reviewed archived GE facade delta changed'):
                    reviewed_archived_ge_facade()

    def test_dimension_aware_show_is_exact_and_reversible(self):
        raw = reviewed_easy_show_source()
        self.assertEqual(raw, reviewed_easy_batch_report_source(reverse=reviewed_easy_optional_import_source(reverse=(ROOT/'fibernet/easy.py').read_bytes())))
        self.assertEqual(reviewed_easy_show_source(reverse=raw),
                         reviewed_taichi_geometry_source('fibernet/easy.py'))
        for changed in (raw+b'\npass\n', raw.replace(b'\n', b'\r\n'),
                        raw.replace(b'graph.dimension == 3', b'graph.dimension == 2', 1),
                        raw.replace(b'signature(renderer).bind', b'signature(renderer).bind_partial', 1),
                        raw.replace(b'kwargs.setdefault("dpi", 150)', b'kwargs.setdefault("dpi", 10)', 1),
                        raw.replace(b'def simulate(', b'def changed_simulate(', 1)):
            with self.assertRaises(AssertionError):
                require_public_source('fibernet/easy.py', changed)
            with self.assertRaises(AssertionError):
                reviewed_easy_show_source(reverse=changed)
        original = Path.read_bytes
        fixture = ROOT/'tests/fixtures/easy_show_dimension_v1/easy_delta.json'
        def mutated(path):
            raw = original(path)
            return raw+b'\n' if path == fixture else raw
        with patch.object(Path, 'read_bytes', mutated):
            with self.assertRaisesRegex(AssertionError, 'dimension-aware show source delta changed'):
                reviewed_easy_show_source()

    def test_batch_report_source_is_exact_and_reversible(self):
        raw = reviewed_easy_batch_report_source()
        self.assertEqual(raw, reviewed_easy_optional_import_source(reverse=(ROOT/'fibernet/easy.py').read_bytes()))
        self.assertEqual(reviewed_easy_batch_report_source(reverse=raw), reviewed_easy_show_source())
        for changed in (raw+b'\npass\n', raw.replace(b'\n', b'\r\n'),
                        raw.replace(b'report_path: Optional[str] = None', b'report_path: Optional[str] = "shared.json"', 1),
                        raw.replace(b'os.path.samefile(destination, csv_path)', b'False', 1),
                        raw.replace(b'allow_nan=False', b'allow_nan=True', 1),
                        raw.replace(b'def simulate(', b'def changed_simulate(', 1)):
            self.assertNotEqual(changed, raw)
            with self.assertRaises(AssertionError):
                require_public_source('fibernet/easy.py', changed)
            with self.assertRaises(AssertionError):
                reviewed_easy_batch_report_source(reverse=changed)
        original = Path.read_bytes
        fixture = ROOT/'tests/fixtures/easy_batch_report_v1/easy_delta.json'
        def mutated(path):
            raw = original(path)
            return raw+b'\n' if path == fixture else raw
        with patch.object(Path, 'read_bytes', mutated):
            with self.assertRaisesRegex(AssertionError, 'batch report source delta changed'):
                reviewed_easy_batch_report_source()

    def test_legacy_analysis_source_is_exact_and_reversible(self):
        raw = reviewed_legacy_analysis_source()
        self.assertEqual(raw, (ROOT/'fibernet/api.py').read_bytes())
        self.assertEqual(reviewed_legacy_analysis_source(reverse=raw), reviewed_native_io_source())
        for changed in (raw+b'\npass\n', raw.replace(b'\n', b'\r\n'),
                        raw.replace(b'return analyze_legacy_endpoints(network)', b'return {}', 1)):
            self.assertNotEqual(changed, raw)
            with self.assertRaises(AssertionError):
                require_public_source('fibernet/api.py', changed)
            with self.assertRaises(AssertionError):
                reviewed_legacy_analysis_source(reverse=changed)

    def test_native_io_source_is_exact_and_reversible(self):
        raw = reviewed_native_io_source()
        self.assertEqual(raw, reviewed_legacy_analysis_source(reverse=(ROOT/'fibernet/api.py').read_bytes()))
        self.assertEqual(reviewed_native_io_source(reverse=raw), public_baseline('fibernet/api.py').read_bytes())
        for changed in (raw+b'\npass\n', raw.replace(b'\n', b'\r\n'),
                        raw.replace(b'return _StructureGraph.from_dict(data)', b'return FiberNetwork.from_dict(data)', 1),
                        raw.replace(b'if graph_fields and network_fields:', b'if False:', 1),
                        raw.replace(b'def analyze(', b'def changed_analyze(', 1)):
            self.assertNotEqual(changed, raw)
            with self.assertRaises(AssertionError):
                require_public_source('fibernet/api.py', changed)
            with self.assertRaises(AssertionError):
                reviewed_native_io_source(reverse=changed)
        original = Path.read_bytes
        fixture = ROOT/'tests/fixtures/native_io_dispatch_v1/api_delta.json'
        def mutated(path):
            raw = original(path)
            return raw+b'\n' if path == fixture else raw
        with patch.object(Path, 'read_bytes', mutated):
            with self.assertRaisesRegex(AssertionError, 'native IO source delta changed'):
                reviewed_native_io_source()

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

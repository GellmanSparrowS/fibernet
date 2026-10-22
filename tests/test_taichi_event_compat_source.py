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


def require_public_source(relative, raw):
    if relative == 'fibernet/sim/__init__.py':
        assert raw in (prior_sim_facade(), reviewed_facade(relative), reviewed_nodal_controls_facade(), reviewed_thermoelastic_facade()), 'public source bytes changed: '+relative
        return
    if relative == 'fibernet/easy.py':
        assert raw in (public_baseline(relative).read_bytes(), reviewed_easy_stretch_ratio_source(), reviewed_easy_rl_failure_source()), 'public source bytes changed: '+relative
        return
    if relative in FACADE_PINS:
        prior = prior_sim_facade() if relative == 'fibernet/sim/__init__.py' else public_baseline(relative).read_bytes()
        assert raw in (prior, reviewed_facade(relative)), 'public source bytes changed: '+relative
        return
    expected=public_baseline(relative).read_bytes()
    if relative == 'fibernet/sim/accelerated.py':
        assert raw in (expected, reviewed_result_identity_source(), reviewed_spring_provenance_source()), 'public source bytes changed: '+relative
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
    node=copy.deepcopy(result_node)
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
                      (new, reviewed_easy_rl_failure_source()))
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
    def test_exact_three_versions_and_current_are_admitted(self):
        old = public_baseline('fibernet/easy.py').read_bytes()
        stretch = reviewed_easy_stretch_ratio_source()
        failure = reviewed_easy_rl_failure_source()
        self.assertEqual((ROOT/'fibernet/easy.py').read_bytes(), failure)
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
        self.assertEqual(reviewed_spring_provenance_source(), SOURCE.read_bytes())

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
        self.assertEqual(reviewed_spring_provenance_source(), SOURCE.read_bytes())
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
        self.assertEqual(raw, (ROOT/relative).read_bytes())
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

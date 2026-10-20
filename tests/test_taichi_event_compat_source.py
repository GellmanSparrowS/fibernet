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
 '_validate_dynamics_state','_validate_result_state','_dynamics_state_summary','SimResult'}

def isolated(legacy=False):
    tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
    selected=[n for n in tree.body if getattr(n,'name',None) in WANTED]
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
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'_RENDER_EXPORTS','_LAZY_EXPORTS'} for t in n.targets):selected.append(n)
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

def public_baseline(relative):
    if relative not in PUBLIC_MODULES:
        raise ValueError('unknown public source contract')
    return ROOT/'tests/fixtures/public_source_contract'/Path(relative).with_suffix('.py.txt')

def require_public_source(relative, raw):
    expected=public_baseline(relative).read_bytes()
    if relative in PUBLIC_MODULES[3:]:
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
    # Only the reviewed LAMMPS guards alter these public result methods.
    # Match their full public baselines before comparing the historical engine.
    for name in ('to_dict', 'load'):
        method=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name)
        expected_method=next(n for n in expected_result.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert canonical_ast(method)==canonical_ast(expected_method), 'public result method changed: '+name
        old_method=next(n for n in old_result.body if isinstance(n,ast.FunctionDef) and n.name==name)
        node.body[node.body.index(method)]=old_method
    detailed=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='compute_detailed')
    node.body[node.body.index(detailed)]=remove_public_lammps_branch(detailed,'SimResult.compute_detailed')
    return node


class SourceShape(unittest.TestCase):
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
                    if path in PUBLIC_MODULES[3:]:
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

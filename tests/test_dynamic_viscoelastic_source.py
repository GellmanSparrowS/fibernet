"""Source-inert checks: actual orchestration, prescribed fields, fake trial.

No FiberNet/Taichi imports, material-law calls, physical integration or native calls.
Run explicitly as a script for a bounded JSON report.
"""
import ast
import builtins
import copy
import dataclasses
import hashlib
import json
import pathlib
import sys
import time
import types
import warnings
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run_checks():
    start = time.process_time()
    checks = []
    def check(name, yes):
        assert yes, name
        checks.append(name)
    material_source = (ROOT/'fibernet/sim/_taichi_viscoelastic.py').read_text()
    native_source = (ROOT/'fibernet/sim/accelerated.py').read_text()
    tree = ast.parse(material_source)
    check('exactly one trial callsite in adapter', sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'trial' for n in ast.walk(tree)) == 1)
    check('adapter has no node integration or native import', 'import taichi' not in material_source and '@ti.kernel' not in material_source)

    class FakeMaterial:
        """Predetermined nonphysical sentinels, never a second material law."""
        parallel_viscosity = 0.
        branches = ((1., 1.),)
        equilibrium_modulus = 0.
        def __init__(self):
            self.calls = []
        def coefficients(self, dt):
            return np.array([1.]), 1.
        def to_dict(self):
            return dict(schema='FAKE_NONPHYSICAL_TRIAL_SENTINEL')
        def trial(self, strain, previous, viscous, dt):
            self.calls.append((strain.copy(), previous.copy(), viscous.copy(), dt))
            k = len(self.calls)
            return dict(stress=np.full_like(strain, 7.*k), viscous_strain=np.full_like(viscous, .01*k),
                stored_energy_density=np.full_like(strain, 2.*k),
                physical_dissipation_density=np.full_like(strain, .3),
                numerical_dissipation_density=np.full_like(strain, .4),
                endpoint_work_density=np.full_like(strain, .5))

    mod = types.ModuleType('_isolated_dynamic_material')
    sys.modules[mod.__name__] = mod
    stripped = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'fibernet.core.viscoelastic')]
    mod.LinearViscoelasticMaterial = FakeMaterial
    mod.__dict__['__builtins__'] = dict(vars(builtins))
    exec(compile(ast.Module(body=stripped, type_ignores=[]), '<actual source; material class replaced with inert sentinel>', 'exec'), mod.__dict__)
    base_import = builtins.__import__
    def imports(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ('_taichi_viscoelastic', 'fibernet.sim._taichi_viscoelastic'):
            return mod
        if name == 'accelerated':
            return types.SimpleNamespace(_checked_kinetic_energy=ns['_checked_kinetic_energy'])
        return base_import(name, globals, locals, fromlist, level)

    ntree = ast.parse(native_source)
    helpers = {'_legacy_damping', '_finite_real_array', '_strict_state_ids', '_state_units', '_native_axial_event_dataclass',
        'NativeAxialObserverEvent', '_emit_axial_observation', '_validated_fixed_rows',
        '_resolve_nodal_masses', '_resolve_initial_velocities', '_canonical_rest_lengths',
        '_resolve_rest_lengths', '_validate_mass_inputs', '_normalize_spring_stiffness', '_prepare_displacement_schedule', '_checked_kinetic_energy'}
    method = next(n for c in ntree.body if isinstance(c, ast.ClassDef) and c.name == 'TaichiEngine'
                  for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'dynamics')
    class Result:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.dynamics_state = self.algorithmic_energy_ledger = None
            self.n_nodes = self.n_edges = 0
        def compute_detailed(self, graph, stiffness=None):
            if 'native_axial_viscoelastic' in self.metadata:
                mod.populate_result(self, self.metadata['native_axial_viscoelastic'], graph, stiffness)
    mod.__dict__['__builtins__'] = dict(vars(builtins), __import__=imports)
    # Functions created above retain their original builtins mapping; replace
    # the importer in that private mapping before invoking any adapter methods.
    for cls in (mod.DynamicAxialMaterial,):
        for value in vars(cls).values():
            if hasattr(value, '__builtins__'):
                value.__builtins__['__import__'] = imports
    ns = dict(warnings=warnings, np=np, time=time, sys=sys, dataclass=dataclasses.dataclass, field=dataclasses.field,
        __builtins__=dict(vars(builtins), __import__=imports), SimResult=Result,
        _material_options=mod.material_options, _check_taichi_owner=lambda: None,
        _ensure_taichi=lambda *a: None, _prepare_force_feedback=lambda *a: None,
        _feedback_sampling=types.SimpleNamespace(observe_run=lambda f:f),
        _element_data=lambda x,e:(np.linalg.norm(x[e[:,1]]-x[e[:,0]], axis=1), None))
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    nodes += [n for n in ntree.body if getattr(n, 'name', None) in helpers
              or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'_DAMPING_UNSET', '_SPRING_PROVENANCE'} for t in n.targets))] + [method]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<actual dynamics AST, cached fake native fields>', 'exec'), ns)
    class Field:
        def __init__(self):
            self.value = None
            self.uploads = []
        def from_numpy(self, value):
            self.value = np.array(value, copy=True)
            self.uploads.append(self.value.copy())
        def to_numpy(self):
            return self.value.copy()
        def __setitem__(self, key, value):
            self.value = value

    x0 = np.array([[0.,0.,0.],[1.,0.,0.]])
    graph = types.SimpleNamespace(num_nodes=2,
        nodes={17:types.SimpleNamespace(position=x0[0]), 83:types.SimpleNamespace(position=x0[1])},
        edges={91:types.SimpleNamespace(node_i=17, node_j=83)})
    ns['_graph_to_arrays'] = lambda g:(x0.copy(), np.array([[0,1]]), [17,83], None)
    fields = None
    steps = None
    observed = None
    def run(**overrides):
        nonlocal fields, steps, observed
        fields = {key:Field() for key in ('ti_pos','ti_vel','ti_vel_old','ti_mass','ti_mass_mode',
            'ti_edges','ti_L0','ti_k','ti_fixed','ti_ext','ti_pos0','ti_sched','ti_sched_mask',
            'ti_dt','ti_dashpot','ti_drag','ti_ramp','ti_material_force','material_v_kick',
            'material_v_drag','material_x_drift')}
        steps = []; observed = []
        def step():
            steps.append(fields['ti_material_force'].value.copy())
            x = x0.copy(); x[1,0] += .01*len(steps)
            v = np.zeros_like(x); v[1,0] = .1*len(steps)
            for key in ('ti_pos','material_x_drift'):fields[key].value=x.copy()
            for key in ('ti_vel','material_v_kick','material_v_drag'):fields[key].value=v.copy()
        fields['substep'] = step
        ns['TaichiEngine'] = types.SimpleNamespace(_field_cache={('material-state-1',3,2,1):fields})
        material = FakeMaterial()
        def observe(event):
            observed.append(event)
            assert len(material.calls) == event.step
            assert np.array_equal(fields['ti_material_force'].value,
                np.array(event.material_state['stress_Pa'])*.25)
        kwargs = dict(material=material, reference_areas_by_id={91:.25}, max_abs_strain=.1,
            nodal_masses_by_id={83:3.,17:2.}, rest_lengths_by_id={91:1.}, state_units=dict(mod.SI),
            dashpot=0., drag=0., dt=.125, num_steps=3, save_interval=2, fixed_nodes=[0],
            initial_velocities_by_id={83:[.2,0.,0.]}, state_observer=observe)
        kwargs.update(overrides)
        result = ns['dynamics'](types.SimpleNamespace(arch='cpu',num_threads=1), graph, **kwargs)
        return result, material

    result, material = run()
    material_events = list(observed)
    check('exactly one fake trial for each actual mocked interval', len(material.calls) == 3)
    check('t0 has no fictitious trial or relaxation', observed[0].material_state['stress_Pa'] == (0.,) and observed[0].material_state['interval_dt'] == 0.)
    check('cached stress drives following native kick', [float(a[0]) for a in steps] == [0.,1.75,3.5])
    check('material committed and uploaded before every completed observer', [e.step for e in observed] == [0,1,2,3])
    check('actual interval dt is passed unchanged', all(c[3] == .125 for c in material.calls))
    check('state history is chained once, not by sample schedule', [float(c[2][0,0]) for c in material.calls] == [0.,.01,.02])
    check('result uses sentinel committed stress and U, not elastic stiffness', result.edge_forces.tolist() == [5.25] and result.energy == 1.5)
    record = result.metadata['native_axial_viscoelastic']
    check('sampled material state includes initial, sampled and final state', [s['step'] for s in record['samples']] == [0,2,3])
    check('stable IDs, geometry and final branch/x/v state retained', record['node_ids'] == [17,83] and record['edge_ids'] == [91] and record['final']['viscous_strain'] == [[.03]] and len(record['final_velocities_m_per_s']) == 2)
    check('physical and BE numerical loss use A0 L0, kept separate', np.allclose(record['final']['cumulative']['physical_dissipation_J'], [.225]) and np.allclose(record['final']['cumulative']['be_numerical_dissipation_J'], [.3]))
    check('no fake memoryless state or elastic-rest provenance', result.dynamics_state is None and 'native_axial_rest' not in result.metadata)
    before = len(material.calls)
    copy.deepcopy(record); json.dumps(record, allow_nan=False)
    check('serialization and observation do not advance history', len(material.calls) == before)
    try: observed[-1].material_state['stress_Pa'] = (99.,)
    except TypeError: check('material event mapping immutable', True)
    else: raise AssertionError('mutable event')
    try: observed[-1].positions.setflags(write=True)
    except ValueError: check('material event positions immutable', True)
    else: raise AssertionError('mutable positions')
    try: result.compute_detailed(graph, stiffness=1.)
    except ValueError: check('elastic recomputation explicitly refused', True)
    else: raise AssertionError('elastic result overwrite')
    parallel_material = FakeMaterial()
    parallel_material.parallel_viscosity = 1.
    for name, overrides in (
        ('spring conflict', dict(spring_k=1.)), ('explicit default stiffness conflict', dict(stiffness=1e5)),
        ('missing stable masses', dict(nodal_masses_by_id=None)), ('missing rest IDs', dict(rest_lengths_by_id=None)),
        ('missing areas', dict(reference_areas_by_id=None)), ('bad area IDs', dict(reference_areas_by_id={92:.25})),
        ('nonpositive bound', dict(max_abs_strain=0.)), ('nonfinite bound', dict(max_abs_strain=float('inf'))),
        ('wrong units', dict(state_units=dict(mod.SI, length='mm'))), ('legacy dashpot', dict(dashpot=1.)),
        ('elastic ledger', dict(record_energy_ledger=True)), ('memoryless restart', dict(initial_state={})),
        ('displacement schedule', dict(displacement_schedule={1:[(1,[0.,0.,0.])]})),
        ('nonzero parallel viscosity', dict(material=parallel_material))):
        try:run(**overrides)
        except (ValueError,TypeError): check('refuse '+name+' before native step', not steps)
        else:raise AssertionError(name+' accepted')
    try:run(max_abs_strain=.015)
    except ValueError:check('strain bound stops before completed observation and next kick', len(steps)==2 and [e.step for e in observed]==[0,1])
    else:raise AssertionError('strain bound not enforced')
    try:run(state_observer=lambda e: (_ for _ in ()).throw(RuntimeError('stop')) if e.step==1 else None)
    except RuntimeError:check('callback failure stops without another fake interval', len(steps)==1)
    else:raise AssertionError('observer did not stop')
    # Read-only serialization validator fixture. Branch state is deliberately
    # set equal to strain, with zero stress/U; no material update is executed.
    FakeMaterial.from_dict = classmethod(lambda cls, value: cls())
    fixture = copy.deepcopy(record)
    for state in fixture['samples']:
        state['viscous_strain'] = [list(state['strain'])]
        state['stress_Pa'] = [0.]
        state['stored_energy_J'] = [0.]
        state['increments'] = {k:[0.] for k in mod.ENERGY_FIELDS[1:]}
        state['cumulative'] = {k:[0.] for k in mod.ENERGY_FIELDS[1:]}
    fixture['final'] = copy.deepcopy(fixture['samples'][-1])
    mod.validate_record(fixture)
    check('read-only material record validator accepts aligned algebraic fixture', True)
    for label, mutate in (
        ('nonfinite dynamic interval', lambda f:f['samples'][1]['dynamic_interval'].__setitem__('kick_drift_residual_J', float('nan'))),
        ('nonfinite cumulative', lambda f:f['samples'][1]['dynamic_cumulative'].__setitem__('projection_force_work_J', float('inf'))),
        ('initial loss', lambda f:f['samples'][0]['increments'].__setitem__('physical_dissipation_J', [1.])),
        ('bad fixed ID', lambda f:f.__setitem__('fixed_node_ids', [999])),
        ('negative drag', lambda f:f.__setitem__('drag_per_s', -1.)),
        ('reordered material IDs', lambda f:f['samples'][1].__setitem__('edge_ids', [92]))):
        bad = copy.deepcopy(fixture); mutate(bad)
        try:mod.validate_record(bad)
        except ValueError:check('record refuses '+label, True)
        else:raise AssertionError(label+' record accepted')
    # Empty branches remain (0,M) after JSON [] roundtrip.
    check('empty branch history preserves declared shape', mod._array([], (0,3), 'branches').shape == (0,3))
    # Exercise actual SimResult summary serialization with the inert valid record.
    result_class = next(n for n in ntree.body if isinstance(n,ast.ClassDef) and n.name=='SimResult')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[nodes[0],result_class], type_ignores=[])),
        '<actual SimResult AST>', 'exec'), ns)
    saved = ns['SimResult'](mode='dynamics',n_nodes=2,n_edges=1,energy=0.,
        metadata=dict(node_ids=[17,83],edge_ids=[91],dt=.125,num_steps=3,
            native_axial_viscoelastic=fixture,
            native_axial_inputs=dict(schema='native_axial_inputs/1',node_ids=[17,83],masses=[2.,3.],units=dict(mod.SI))))
    summary = saved.to_dict(detailed=False)
    check('viscoelastic summary does not invent spring provenance',
        ns['_SPRING_PROVENANCE'] not in summary)
    check('actual summary export retains full material branch state and final x/v',
        summary['native_axial_viscoelastic']['final']==fixture['final'] and
        summary['native_axial_viscoelastic']['final_positions_m']==fixture['final_positions_m'])
    summary['native_axial_viscoelastic']['final']['stress_Pa'][0] = 99.
    check('actual summary export is detached from live material result', fixture['final']['stress_Pa']==[0.])
    check('no real solver/package imports', 'taichi' not in sys.modules and not any(k=='fibernet' or k.startswith('fibernet.') for k in sys.modules))
    return dict(status='PASS_SOURCE_AND_MOCK_ORCHESTRATION_ONLY', checks=checks, count=len(checks),
        CPU_seconds=time.process_time()-start, native_calls=0, real_material_trial_calls=0,
        physical_trajectories=0, limitations=['No constitutive or native physics execution',
            'Prescribed fake fields and nonphysical trial sentinels test sequencing, not numerical accuracy'])


def test_source_contract():
    import subprocess
    completed = subprocess.run([sys.executable, '-B', str(__file__)],
        capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)['status'] == 'PASS_SOURCE_AND_MOCK_ORCHESTRATION_ONLY'


if __name__ == '__main__':
    print(json.dumps(run_checks(), indent=2))

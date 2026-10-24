"""Source metadata only: no native launch, installation or qualification."""
import copy
import importlib
import json
import os
import subprocess
import sys
import textwrap
from unittest.mock import patch

from fibernet.integrations import _harmonic_profile as harmonic
from fibernet.integrations import _cg_network_profile as network
from fibernet.integrations import _lammps_cg as cg
from fibernet.sim import _lammps_result as adapter


def test_public_lazy_export_is_single_implementation():
    import fibernet.sim as sim
    assert 'lammps_source_capabilities' in sim.__all__
    assert 'lammps_source_capabilities' in dir(sim)
    assert sim.lammps_source_capabilities is adapter.source_capabilities
    importlib.reload(sim)
    assert 'lammps_source_capabilities' not in vars(sim)
    assert sim.lammps_source_capabilities is adapter.source_capabilities


def test_models_and_profile_limits_match_authoritative_contracts():
    c = adapter.source_capabilities()
    assert set(c['models']) == {'explicit-synthetic-or-coarse-grained-beads', cg.MODEL}
    assert c['models'][cg.MODEL]['contract_capabilities'] == cg.source_capabilities()
    assert c['cg_profiles'] == {p['name']: dict(p) for p in (network.PROFILE, network.MIXED_PROFILE)}
    keys = ('max_nodes', 'max_edges', 'steps', 'max_steps', 'sample_every',
            'output_stride', 'frames', 'max_output_frames', 'max_mass_types', 'max_loaded_nodes')
    assert c['harmonic_profiles'] == {p['name']: {k: p[k] for k in keys if k in p}
        for p in (harmonic.PROFILE, harmonic.SAMPLED_PROFILE, harmonic.FULL_HISTORY_PROFILE)}
    assert c['models'][cg.MODEL]['profiles'] == list(network.NAMES)
    assert 'unsupported' in c['models'][cg.MODEL]['loads']
    assert 'no network profile' in c['models'][cg.MODEL]['fiber_only']
    assert c['profile'] == 'explicit-harmonic-beads'
    assert 'legacy harmonic' in c['profile_scope']


def test_observation_claims_match_real_schedule_functions():
    c = adapter.source_capabilities()
    observations = c['observations']
    for name in ('harmonic_unprofiled', 'cg_unprofiled', harmonic.NAME,
                 harmonic.SAMPLED_NAME, harmonic.FULL_HISTORY_NAME):
        assert observations[name]['native_observations'] == 'every integration step'
        assert 'no continuous-time guarantee' in observations[name]['geometry_validation']
    assert harmonic.schedule(harmonic.SAMPLED_PROFILE, 256, 1, 4) == 65
    assert 'output_stride-selected' in observations[harmonic.SAMPLED_NAME]['exported_states']
    assert harmonic.schedule(harmonic.FULL_HISTORY_PROFILE, 256, 1, 1) == 257
    assert 'every integration step' in observations[harmonic.FULL_HISTORY_NAME]['exported_states']
    assert network.schedule(1000, 100, 10) == (11, 101)
    for name in network.NAMES:
        assert observations[name]['geometry_validation'] == network.SCOPE
        assert 'thermo_every' in observations[name]['native_observations']
        assert 'dump-sampled' in observations[name]['exported_states']
    assert 'omits dynamics_state' in c['record_state']


def test_metadata_is_fully_detached_and_json_serializable():
    baseline = adapter.source_capabilities()
    changed = adapter.source_capabilities()
    def corrupt(value):
        if isinstance(value, dict):
            for child in list(value.values()):
                corrupt(child)
            value['mutation_sentinel'] = True
        elif isinstance(value, list):
            for child in value:
                corrupt(child)
            value.clear()
    corrupt(changed)
    assert adapter.source_capabilities() == baseline
    assert cg.source_capabilities() == baseline['models'][cg.MODEL]['contract_capabilities']
    assert dict(network.PROFILE) == baseline['cg_profiles'][network.NAME]
    assert json.loads(json.dumps(baseline)) == baseline
    assert copy.deepcopy(baseline) == baseline


def test_no_runtime_admission_or_launch_and_no_readiness_claim():
    from fibernet.integrations import lammps_backend as backend
    with patch.object(backend.RuntimeSpec, 'checked', side_effect=AssertionError('admission')), \
         patch.object(backend.LAMMPSTask, 'start', side_effect=AssertionError('launch')), \
         patch('subprocess.Popen', side_effect=AssertionError('child')):
        c = adapter.source_capabilities()
    assert c['runtime_readiness'] == 'unknown'
    assert c['physics_qualification'] == 'not_established_by_dispatch'
    assert 'not runtime or physics qualification' in c['source_support']
    assert 'this bounded dispatcher' in c['unsupported_scope']
    assert 'continuous fiber contact' in c['unsupported']


def test_cold_public_accessor_with_native_providers_absent():
    script = textwrap.dedent('''
        import importlib
        import importlib.abc
        import json
        import sys
        blocked = {'lammps', 'taichi', 'torch', 'exudyn', 'mujoco', 'pybullet',
                   'gym', 'gymnasium', 'MDAnalysis', 'ovito', 'dolfin', 'fenics'}
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in blocked:
                    raise ModuleNotFoundError('native provider blocked', name=fullname)
        sys.meta_path.insert(0, Block())
        for name in blocked:
            try:
                importlib.import_module(name)
            except ModuleNotFoundError:
                pass
            else:
                raise AssertionError(name)
        import subprocess
        from unittest.mock import patch
        with patch.object(subprocess, 'Popen', side_effect=AssertionError('native launch')):
            import fibernet.sim as sim
            assert 'lammps_source_capabilities' not in vars(sim)
            from fibernet.sim import lammps_source_capabilities
            from fibernet.sim._lammps_result import source_capabilities
            assert lammps_source_capabilities is source_capabilities
            c = lammps_source_capabilities()
            assert c['runtime_readiness'] == 'unknown'
        assert not any(name.split('.')[0] in blocked for name in sys.modules)
        print(json.dumps({'models': sorted(c['models'])}))
    ''')
    result = subprocess.run([sys.executable, '-c', script], env=dict(os.environ),
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert cg.MODEL in result.stdout

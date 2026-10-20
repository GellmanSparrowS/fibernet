"""Public API and retained saved-output checks, without advancing material/native state."""
import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import fibernet as fn
from fibernet.sim.accelerated import SimResult, TaichiEngine

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / 'tests/fixtures/dynamic_viscoelastic_generalized_summary.json'


def example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'examples'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    with patch.object(fn, 'simulate', side_effect=AssertionError('unexpected simulation')):
        spec.loader.exec_module(module)
    return module


def test_retained_summary_public_save_load_without_advance(tmp_path):
    with patch.object(fn.LinearViscoelasticMaterial, 'trial', side_effect=AssertionError('unexpected advance')):
        result = SimResult.load(str(SAVED))
        expected = json.loads(SAVED.read_text())['native_axial_viscoelastic']
        assert result.metadata['native_axial_viscoelastic'] == expected
        detached = result.to_dict()
        detached['native_axial_viscoelastic']['final']['stress_Pa'][0] = 999.
        assert result.metadata['native_axial_viscoelastic'] == expected
        for detailed in (False, True):
            path = tmp_path/f'result-{detailed}.json'
            result.save(str(path), detailed=detailed)
            reloaded = SimResult.load(str(path))
            assert reloaded.metadata['native_axial_viscoelastic'] == expected
            np.testing.assert_array_equal(reloaded.edge_forces, result.edge_forces)
            assert reloaded.energy == result.energy
        with pytest.raises(ValueError, match='stiffness'):
            result.compute_detailed(None, stiffness=1.)


def test_saved_corruption_refused(tmp_path):
    payload = json.loads(SAVED.read_text())
    payload['native_axial_viscoelastic']['final']['stress_Pa'][0] += 1.
    path = tmp_path/'bad.json'; path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        SimResult.load(str(path))


def test_examples_import_and_frozen_configuration_without_run():
    assert callable(example('dynamic_viscoelastic_network').run)
    module = example('dynamic_viscoelastic_105')
    with patch.object(fn.LinearViscoelasticMaterial, 'trial', side_effect=AssertionError('unexpected advance')):
        for level, steps in [('coarse',1024),('medium',2048),('fine',4096)]:
            graph, kwargs = module.prepare(level)
            assert (graph.num_nodes,graph.num_edges) == (105,273)
            assert kwargs['num_steps'] == steps
            assert kwargs['dt']*steps == .5
            assert len(kwargs['reference_areas_by_id']) == 273
            assert kwargs['record_state'] is True


def test_public_dispatch_preserves_material_and_arguments():
    module = example('dynamic_viscoelastic_105')
    graph, kwargs = module.prepare('coarse')
    sentinel = object()
    with patch('fibernet.easy.TaichiEngine') as engine:
        engine.return_value.dynamics.return_value = sentinel
        assert fn.simulate(graph, **kwargs) is sentinel
        assert engine.return_value.dynamics.call_args.kwargs['material'] is kwargs['material']
    with patch('fibernet.easy.TaichiEngine') as engine:
        with pytest.raises(ValueError, match='mode=dynamics'):
            fn.simulate(graph, backend='taichi', mode='stretch', material=kwargs['material'])
        engine.assert_not_called()


def test_static_capability_is_honest():
    caps = TaichiEngine.dynamic_material_capabilities()
    assert caps['constitutive_source'] == 'LinearViscoelasticMaterial.trial'
    assert caps['contact'] == 'not exposed by this public dynamic material slice'
    assert 'finite_strain_polymer' in caps['unsupported']

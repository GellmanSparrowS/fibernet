"""Small public-call integration checks, separate from the accepted benchmarks."""
import copy
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import get_args, get_type_hints

import numpy as np
import pytest

import fibernet as fn
from fibernet.ml.beam_frame_fem import BeamFrameFEM


def test_public_history_ramp_hold_and_serialized_restart(tmp_path):
    graph = fn.StructureGraph(dimension=2)
    left = graph.add_node([0., 0.])
    unused = graph.add_node([99., 99.])
    graph.remove_node(unused)
    right = graph.add_node([10., 0.])
    edge = graph.add_edge(left, right, radius=.1)
    before = copy.deepcopy(graph.to_dict())
    law = fn.LinearViscoelasticMaterial.standard_linear_solid(1e6, 2e6, .2)
    controls = {left: [[0., 0., 0.]] * 2,
                right: [[0., None, None], [1e-5, None, None]]}
    destination = tmp_path / 'history.json'
    first = fn.simulate(graph, backend='fem', mode='history', material=law,
                        times=[0., .1], constraints=controls, length_unit='mm',
                        save_path=destination)
    assert isinstance(first, fn.ViscoelasticResult)
    assert first.node_ids == [left, right] and first.edge_ids == [edge]
    restored = fn.ViscoelasticResult.load(destination)
    state = restored.final_state
    untouched = copy.deepcopy(state)
    second = BeamFrameFEM().history(
        graph, material=law, times=[.1, .2], length_unit='mm',
        constraints={left: [[0., 0., 0.]] * 2,
                     right: [[1e-5, None, None]] * 2}, initial_state=state)
    # Independently evaluate one BE ramp interval followed by a held interval.
    epsilon = .001
    factor = .2 / (.2 + .1)
    q1 = 2e6 * factor * epsilon
    q2 = factor * q1
    volume = math.pi * (1e-4)**2 * .01
    np.testing.assert_allclose(first.axial_stress[-1, 0], 1e6 * epsilon + q1)
    np.testing.assert_allclose(second.axial_stress[-1, 0], 1e6 * epsilon + q2)
    np.testing.assert_allclose(second.energies['physical_dissipation'][-1],
                               volume * .1 * (q1*q1 + q2*q2) / (2e6 * .2))
    assert second.final_state['step'] == 2
    assert state == untouched and graph.to_dict() == before
    assert restored.to_dict() == first.to_dict()
    assert max(abs(x) for x in second.energies['balance_residual']) < 1e-20


def test_history_return_annotation_resolves():
    assert set(get_args(get_type_hints(fn.simulate)['return'])) == {
        fn.SimResult, fn.ViscoelasticResult}


def test_history_rejects_strain_shortcut_before_solving(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid history shortcut reached the solver')
    monkeypatch.setattr(BeamFrameFEM, 'history', forbidden)
    with pytest.raises(ValueError, match='explicit controls'):
        fn.simulate(fn.StructureGraph(dimension=2), backend='fem',
                    mode='history', strain=1.01)


def test_beam_import_does_not_attempt_torch():
    root = Path(__file__).resolve().parents[1]
    code = '''import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('base FEM attempted to import Torch')
sys.meta_path.insert(0, BlockTorch())
import fibernet
from fibernet.ml.beam_frame_fem import BeamFrameFEM
assert callable(BeamFrameFEM.history)
assert 'torch' not in sys.modules
'''
    environment = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-c', code], cwd=root,
                            env=environment, capture_output=True, text=True,
                            encoding='utf-8', timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('dimension', [2, 3])
def test_existing_static_tensor_conversion_after_module_import(monkeypatch, dimension):
    # A compatibility double checks the late-loaded module path without Torch.
    # This is not a real Torch runtime qualification.
    class Tensor:
        def __init__(self, array):
            self.array = np.asarray(array)
        def numpy(self):
            return self.array
    edge_index = np.array([[0], [1]])
    positions = np.zeros((2, dimension)); positions[1, 0] = .01
    radii = np.array([1e-4])
    forces = np.zeros((2, dimension)); forces[1, 0] = 1e-5
    solver = getattr(BeamFrameFEM(E=1e6), 'solve_' + str(dimension) + 'd')
    expected = solver(edge_index, positions, radii, forces=forces,
                      fixed_nodes=[0], damping=0.)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(Tensor=Tensor))
    actual = solver(Tensor(edge_index), Tensor(positions), Tensor(radii),
                    forces=Tensor(forces), fixed_nodes=[0], damping=0.)
    np.testing.assert_array_equal(actual['u'], expected['u'])
    np.testing.assert_array_equal(actual['reactions'], expected['reactions'])

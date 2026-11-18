"""Missing history samples must not be silently replaced by masked payloads."""
import copy
import math
import warnings

import numpy as np
import pytest

import fibernet as fn
from fibernet.ml.beam_frame_fem import BeamFrameFEM


def _case():
    graph = fn.StructureGraph(dimension=2)
    left = graph.add_node([0., 0.])
    right = graph.add_node([1., 0.])
    graph.add_edge(left, right, radius=.1)
    return graph, dict(
        backend='fem', mode='history', length_unit='m',
        material=fn.LinearViscoelasticMaterial.standard_linear_solid(1e6, 2e6, .2),
        times=np.array([0., .1]),
        constraints={left: np.zeros((2, 3))},
        loads={right: np.array([[0., 0., 0.], [1., 0., 0.]])},
    )


def _wrap(value, representation):
    if representation == 'array':
        return value
    if representation == 'list':
        return list(value)
    if representation == 'tuple':
        return tuple(value)
    # An object array can hold masked rows without carrying an outer mask.
    result = np.empty(len(value), dtype=object)
    for index, row in enumerate(value):
        result[index] = row
    return result


@pytest.mark.parametrize('field', ['times', 'loads', 'constraints'])
@pytest.mark.parametrize('representation', ['array', 'list', 'tuple', 'object'])
def test_public_history_rejects_missing_controls_before_assembly(monkeypatch, field, representation):
    graph, kwargs = _case()
    value = kwargs[field] if field == 'times' else next(iter(kwargs[field].values()))
    missing = np.ma.array(value, mask=np.zeros(value.shape, dtype=bool), copy=True)
    missing.mask[-1 if field == 'times' else (1, 0)] = True
    wrapped = _wrap(missing, representation)
    if field == 'times':
        kwargs[field] = wrapped
    else:
        kwargs[field][next(iter(kwargs[field]))] = wrapped
    original_graph = copy.deepcopy(graph.to_dict())
    original_data, original_mask = missing.data.copy(), np.ma.getmaskarray(missing).copy()

    def unexpected(*args, **kwargs):
        pytest.fail('missing controls reached stiffness assembly')

    monkeypatch.setattr(BeamFrameFEM, 'build_stiffness_2d', unexpected)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        with pytest.raises(ValueError, match='masked'):
            fn.simulate(graph, **kwargs)
    assert graph.to_dict() == original_graph
    np.testing.assert_array_equal(missing.data, original_data)
    np.testing.assert_array_equal(np.ma.getmaskarray(missing), original_mask)


def test_all_valid_masked_controls_preserve_public_solution_and_analytic_force():
    graph, kwargs = _case()
    expected = fn.simulate(graph, **kwargs)
    kwargs['times'] = np.ma.array(kwargs['times'], mask=False)
    for field in ('loads', 'constraints'):
        for nid, value in kwargs[field].items():
            kwargs[field][nid] = np.ma.array(value, mask=False)
    actual = fn.simulate(graph, **kwargs)
    assert actual.to_dict() == expected.to_dict()
    tangent = 1e6 + 2e6 * .2 / (.2 + .1)
    displacement = 1. / (math.pi * .1**2 * tangent)
    np.testing.assert_allclose(actual.displacements[-1, 1, 0], displacement, rtol=1e-12)
    np.testing.assert_allclose(actual.axial_force[-1, 0], 1., rtol=1e-12)


def test_all_valid_masked_constraints_preserve_none_as_free_dof():
    graph, kwargs = _case()
    kwargs['constraints'][1] = [[0., None, None], [1e-5, None, None]]
    kwargs['loads'] = None
    expected = fn.simulate(graph, **kwargs)
    kwargs['constraints'][1] = np.ma.array(kwargs['constraints'][1], dtype=object, mask=False)
    actual = fn.simulate(graph, **kwargs)
    assert actual.to_dict() == expected.to_dict()

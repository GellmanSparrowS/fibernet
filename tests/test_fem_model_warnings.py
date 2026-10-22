"""FEM applicability warnings follow the selected model, not the facade.

Dispatch checks use inert solve methods. State checks reuse a two-node beam;
none of these checks qualifies large-deformation equilibrium.
"""
import warnings

import numpy as np
import pytest

import fibernet as fn
from fibernet.ml.beam_frame_fem import BeamFrameFEM
from test_fem_linear_diagnostics import sparse_graph


@pytest.mark.parametrize('dim,ratio,nonlinear,branch,message', [
    (2, 1.01, None, 'solve_2d', None),
    (2, 1.01, False, 'solve_2d', None),
    (2, .99, np.bool_(False), 'solve_2d', None),
    (2, 1.01, True, 'solve_2d_nonlinear', 'equilibrium is unverified'),
    (2, 1.01, np.bool_(True), 'solve_2d_nonlinear', 'equilibrium is unverified'),
    (2, 2., None, 'solve_2d_nonlinear', 'equilibrium is unverified'),
    (2, .5, None, 'solve_2d_nonlinear', 'equilibrium is unverified'),
    (2, 2., True, 'solve_2d_nonlinear', 'equilibrium is unverified'),
    (2, 2., False, 'solve_2d', 'small-deformation linear solver'),
    (2, .5, False, 'solve_2d', 'small-deformation linear solver'),
    # Preserve the actual historical floating-point comparison at the switch.
    (2, np.nextafter(1.3, 1.), None, 'solve_2d', None),
    (2, 1.3, None, 'solve_2d_nonlinear', 'equilibrium is unverified'),
    (2, np.nextafter(1.3, 1.), False, 'solve_2d', None),
    (2, 1.3, False, 'solve_2d', 'small-deformation linear solver'),
    (3, 1.01, None, 'solve_3d', None),
    (3, 1.01, False, 'solve_3d', None),
    (3, 2., None, 'solve_3d', 'no 3-D nonlinear solver'),
    (3, .5, False, 'solve_3d', 'no 3-D nonlinear solver'),
])
def test_warning_matches_executed_branch(monkeypatch, dim, ratio, nonlinear, branch, message):
    graph = sparse_graph(dim)
    calls = []
    sentinel = {'u': np.zeros((graph.num_nodes, 3 if dim == 2 else 6))}
    for name in ('solve_2d', 'solve_2d_nonlinear', 'solve_3d'):
        def inert(*args, _name=name, **kwargs):
            calls.append((_name, kwargs))
            return sentinel
        monkeypatch.setattr(BeamFrameFEM, name, inert)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = BeamFrameFEM().stretch_test(
            graph, target_stretch=ratio, dim=dim, nonlinear=nonlinear)
    assert result is sentinel
    assert [name for name, kwargs in calls] == [branch]
    assert calls[0][1]['deduplicate'] is False
    if branch == 'solve_2d_nonlinear':
        assert calls[0][1]['n_steps'] == 10
    assert len(caught) == (0 if message is None else 1)
    if message is not None:
        assert caught[0].category is UserWarning
        assert message in str(caught[0].message)
        if branch == 'solve_2d_nonlinear':
            assert 'experimental incremental updated-geometry linear solves' in str(caught[0].message)
            assert 'final stiffness, reactions and nodal moments are unavailable' in str(caught[0].message)
        elif dim == 2:
            assert 'nonlinear=False' in str(caught[0].message)


@pytest.mark.parametrize('kwargs,error', [
    ({'dim': 3, 'nonlinear': True}, NotImplementedError),
    ({'dim': 2, 'nonlinear': 'yes'}, TypeError),
    ({'dim': 2, 'target_stretch': np.nan}, ValueError),
    ({'dim': 4}, ValueError),
])
def test_invalid_requests_do_not_issue_applicability_warning(monkeypatch, kwargs, error):
    def unexpected(*args, **kw):
        pytest.fail('invalid request reached graph preparation or a solver')
    monkeypatch.setattr(BeamFrameFEM, 'graph_to_fem_input', unexpected)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(error):
            BeamFrameFEM().stretch_test(sparse_graph(2), **kwargs)
    assert not caught


@pytest.mark.parametrize('nonlinear,ratio,incremental', [
    (None, None, True), (True, 1.01, True), (False, 2., False), (False, 1.01, False),
])
def test_public_route_warns_once_and_preserves_availability(tmp_path, nonlinear, ratio, incremental):
    graph = sparse_graph(2)
    before = graph.to_dict()
    options = dict(backend='fem', nonlinear=nonlinear, E=1000., nu=.3)
    if ratio is not None:
        options['stretch_ratio'] = ratio
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = fn.simulate(graph, **options)
    assert len(caught) == (0 if ratio == 1.01 and nonlinear is False else 1)
    assert graph.to_dict() == before
    path = tmp_path / 'result.json'
    result.save(path, detailed=True)
    restored = fn.SimResult.load(path)
    for current in (result, restored):
        assert current.metadata['energy_kind'] == 'axial_stress_proxy'
        assert current.metadata['numerical_diagnostics']['physical_equilibrium'] == 'not_evaluated'
        if incremental:
            assert current.metadata['solver_status']['equilibrium'] == 'unverified'
            assert current.metadata['solver_status']['update_criterion_is_equilibrium_check'] is False
            for field in ('K', 'reactions', 'node_moment'):
                assert current.metadata[field] is None
                assert current.metadata['field_status'][field]['status'] == 'not_supported'
            assert current.metadata['field_status']['edge_forces']['status'] == 'approximate'
        else:
            assert current.metadata['reactions'] is not None
            assert 'solver_status' not in current.metadata
    np.testing.assert_array_equal(result.displacements, restored.displacements)
    np.testing.assert_array_equal(result.edge_forces, restored.edge_forces)


@pytest.mark.parametrize('name', [
    'README.md', 'PYPI_README.md', 'docs/quickstart.rst',
    'docs/API_LAYERS.md', 'docs/METHODS_BEAM_FEM_CONTRACT.md',
])
def test_introductory_fem_calls_are_explicit_and_parse(name):
    import ast
    from pathlib import Path
    import re
    import textwrap

    source = (Path(__file__).parents[1] / name).read_text(encoding='utf-8')
    if name.endswith('.rst'):
        blocks = [textwrap.dedent(block) for block in re.findall(
            r'\.\. code-block:: python\n\n((?:(?:   [^\n]*|)\n)+)', source)]
    else:
        blocks = re.findall(r'```python\n(.*?)```', source, re.S)
    calls = []
    for block in blocks:
        for node in ast.walk(ast.parse(block)):
            if not isinstance(node, ast.Call):
                continue
            keywords = {key.arg: key.value for key in node.keywords}
            backend = keywords.get('backend')
            if not isinstance(backend, ast.Constant) or backend.value != 'fem':
                continue
            calls.append(node)
            assert ast.literal_eval(keywords['stretch_ratio']) == 1.01
            assert ast.literal_eval(keywords['nonlinear']) is False
            assert ast.literal_eval(keywords['E']) == 1e9
            assert ast.literal_eval(keywords['nu']) == .3
            assert 'strain' not in keywords
    assert calls, name
    assert 'coordinates and radius in m; E in Pa; forces in N' in source

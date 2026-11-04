"""APP/library provider isolation, lifecycle and serialization regressions.

Run with repository root on PYTHONPATH plus FiberScope; no cwd/data writes.
All writable files use tmp_path. No saved private oracle or old checkout required.
"""
import copy
import json
import importlib
from pathlib import Path
import pytest
app = pattern = StructureGraph = None  # imports deferred until selected test execution

TRI = {'nodes': [[0., 0.], [1., 0.], [.5, 1.]],
       'edges': [[0, 1], [1, 2], [2, 0]], 'zh': 'test', 'en': 'test'}
LINE = {'nodes': [[0., .5], [1., .5]], 'edges': [[0, 1]],
        'zh': 'line', 'en': 'line'}

@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    global app, pattern, StructureGraph
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo))
    monkeypatch.syspath_prepend(str(repo / 'FiberScope'))
    monkeypatch.setenv('FIBERNET_ROOT', str(repo))
    app = importlib.import_module('fslab.structure')
    pattern = importlib.import_module('fibernet.gen.pattern')
    core = importlib.import_module('fibernet.core.structure_graph')
    StructureGraph = core.StructureGraph
    for module in (app, pattern, core):
        assert Path(module.__file__).resolve().is_relative_to(repo)
    # No solver is needed by these geometry/provider contracts.
    reduced = importlib.import_module('fibernet.sim.reduced_beam')
    def no_solver(*args, **kwargs):
        raise AssertionError('solver execution forbidden in registry regression')
    monkeypatch.setattr(reduced.Engine2, '__init__', no_solver)
    # Hard per-graph allocation caps, including temporary/baseline graphs.
    add_node, add_edge = StructureGraph.add_node, StructureGraph.add_edge
    copy_graph = StructureGraph.copy
    def bounded_copy(graph):
        assert graph.num_nodes <= 2000 and graph.num_edges <= 4000
        result = copy_graph(graph)
        assert result.num_nodes <= 2000 and result.num_edges <= 4000
        return result
    monkeypatch.setattr(StructureGraph, "copy", bounded_copy)
    def bounded_node(graph, *args, **kwargs):
        if graph.num_nodes >= 2000:
            raise MemoryError("approval budget: 2000 nodes per graph")
        return add_node(graph, *args, **kwargs)
    def bounded_edge(graph, *args, **kwargs):
        if graph.num_edges >= 4000:
            raise MemoryError("approval budget: 4000 edges per graph")
        return add_edge(graph, *args, **kwargs)
    monkeypatch.setattr(StructureGraph, "add_node", bounded_node)
    monkeypatch.setattr(StructureGraph, "add_edge", bounded_edge)
    classic = dict(app._CLASSIC_BINDINGS)
    registry = dict(pattern._UNIT_FACTORIES)
    state = [(m, copy.deepcopy(m.CUSTOM_CELLS), set(m._CELLS_REGISTERED),
              dict(m._INTERMEDIATE_CACHE)) for m in (app,)]
    for m, *_ in state:
        m.CUSTOM_CELLS.clear()
        m._CELLS_REGISTERED.clear()
        m._INTERMEDIATE_CACHE.clear()
    yield
    app._CLASSIC_BINDINGS.clear(); app._CLASSIC_BINDINGS.update(classic)
    pattern._UNIT_FACTORIES.clear()
    pattern._UNIT_FACTORIES.update(registry)
    for m, cells, markers, cache in state:
        m.CUSTOM_CELLS.clear(); m.CUSTOM_CELLS.update(cells)
        m._CELLS_REGISTERED.clear(); m._CELLS_REGISTERED.update(markers)
        m._INTERMEDIATE_CACHE.clear(); m._INTERMEDIATE_CACHE.update(cache)


@pytest.mark.parametrize('order', ['helper_first', 'build_first'])
def test_public_registry_never_replaced(order):
    saved = {k: f for k, f in pattern._UNIT_FACTORIES.items()
             if not k.startswith('fiberscope.cell.v1:')}
    expected = pattern.pattern_2d(unit='kagome', grid=(1, 1), seed=0).to_dict()
    if order == 'helper_first':
        app._ensure_cells()
    app.StructureFactory(unit='kagome', topology='legacy', grid_x=1,
                         grid_y=1, n_pts_per_side=1).build()
    app._ensure_cells(); app._ensure_cells()
    assert all(pattern._UNIT_FACTORIES[k] is f for k, f in saved.items())
    assert pattern.pattern_2d(unit='kagome', grid=(1, 1), seed=0).to_dict() == expected


def test_edit_reload_delete_and_preview(tmp_path, monkeypatch):
    path = tmp_path / 'custom_units.json'
    monkeypatch.setattr(app, 'CUSTOM_CELL_FILE', str(path))
    app.CUSTOM_CELLS['sample'] = copy.deepcopy(TRI)
    first_key = app.StructureFactory(unit='sample').key()
    first_count = app.n_intermediate_for('sample', 2)
    app.CustomCellRegistry(path).replace_all({'sample': LINE})
    app.load_custom_cells(str(path))
    assert not app._INTERMEDIATE_CACHE
    assert app.StructureFactory(unit='sample').key() != first_key
    assert app.n_intermediate_for('sample', 2) != first_count
    assert app.delete_custom_cell('sample')
    assert not app._INTERMEDIATE_CACHE
    with pytest.raises(ValueError, match='unknown APP unit'):
        app.n_intermediate_for('sample', 2)
    app.CUSTOM_CELLS['__editor_preview__'] = copy.deepcopy(TRI)
    key, identity, *_ = app._resolve_pattern_unit('__editor_preview__')
    app.CUSTOM_CELLS['__editor_preview__'] = copy.deepcopy(LINE)
    key2, identity2, *_ = app._resolve_pattern_unit('__editor_preview__')
    assert key == key2
    assert identity['cell_geometry_sha256'] != identity2['cell_geometry_sha256']
    del app.CUSTOM_CELLS['__editor_preview__']
    with pytest.raises(ValueError, match='unknown APP unit'):
        app._resolve_pattern_unit('__editor_preview__')


def test_shadowing_p1_precedence_ambiguity_and_geometry_digest():
    app.CUSTOM_CELLS.update(kagome=copy.deepcopy(TRI), square=copy.deepcopy(TRI))
    _, identity, geometry, _ = app._resolve_pattern_unit('kagome')
    assert geometry['edges'] == TRI['edges']
    assert app._resolve_pattern_unit('square')[1]['provider'] == 'fiberscope.points'
    app.CUSTOM_CELLS['kagome']['en'] = 'new label'
    assert app._resolve_pattern_unit('kagome')[1] == identity
    app.CUSTOM_CELLS['kagome']['edges'].reverse()
    assert app._resolve_pattern_unit('kagome')[1] != identity
    app.CUSTOM_CELLS['Kagome'] = copy.deepcopy(TRI)
    with pytest.raises(ValueError, match='Kagome.*kagome'):
        app._resolve_pattern_unit('kagome')
    assert 'Kagome' in app.CUSTOM_CELLS


def test_explicit_library_override_cache_token_and_key():
    first = app._resolve_pattern_unit('hexagon')
    first_key = app.StructureFactory(unit='hexagon').key()
    first_count = app.n_intermediate_for('hexagon', 2)
    def alternate(**kwargs):
        return app._cell_factory(TRI)(**kwargs)
    pattern.register_unit('hexagon', alternate)
    second = app._resolve_pattern_unit('hexagon')
    assert second[3] != first[3]
    assert second[1] != first[1]
    assert app.StructureFactory(unit='hexagon').key() != first_key
    assert app.n_intermediate_for('hexagon', 2) != first_count
    app._ensure_cells()
    assert pattern._UNIT_FACTORIES['hexagon'] is alternate


@pytest.mark.parametrize('topology', ['legacy', 'topnet26'])
def test_graph_and_app_json_identity_roundtrip(tmp_path, topology):
    from fslab.exporter import export_json
    factory = app.StructureFactory(unit='kagome', topology=topology,
                                   grid_x=1, grid_y=1, n_pts_per_side=1)
    graph = factory.build()
    graph.save_json(str(tmp_path / 'graph.json'))
    restored = StructureGraph.load_json(str(tmp_path / 'graph.json'))
    assert restored.to_dict() == graph.to_dict()
    export_json(factory, str(tmp_path / 'app.json'))
    doc = json.loads((tmp_path / 'app.json').read_text())
    assert doc['spec']['unit'] == 'kagome'
    assert doc['version'] == 1
    assert doc['generator_identity'] == graph.metadata['generator_identity']
    if topology == 'legacy':
        assert graph.metadata['pattern']['unit'] == 'kagome'


def test_successive_same_source_closures_get_unique_cache_epochs():
    def make(spec):
        def factory(**kwargs):
            return app._cell_factory(spec)(**kwargs)
        return factory
    previous_epoch = -1
    counts = []
    for spec in [TRI, LINE, TRI, LINE]:
        factory = make(spec)
        pattern.register_unit('hexagon', factory)
        binding = app._resolve_pattern_unit('hexagon')
        assert binding[3] > previous_epoch
        assert app._CLASSIC_BINDINGS['hexagon'][0] is factory
        previous_epoch = binding[3]
        counts.append(app.n_intermediate_for('hexagon', 2))
    assert counts[0] == counts[2] and counts[1] == counts[3]
    assert counts[0] != counts[1]
    assert len(app._CLASSIC_BINDINGS['hexagon']) == 2


@pytest.mark.parametrize('topology', ['legacy', 'topnet26'])
@pytest.mark.parametrize('rule', ['translate', 'checker_mirror'])
def test_final_boundary_identity_and_library_registry(topology, rule):
    registry = dict(pattern._UNIT_FACTORIES)
    factory = app.StructureFactory(unit='kagome', grid_x=2, grid_y=1,
                                   n_pts_per_side=2, topology=topology,
                                   expansion_rule=rule,
                                   line_displacements=[[.01, -.02], [-.01, .03]])
    graph = factory.build()
    identity = graph.metadata['generator_identity']
    assert identity['provider'] == 'fiberscope.cell'
    assert identity['visible_unit'] == 'kagome'
    assert identity['requested_topology'] == topology
    assert graph.num_nodes <= 2000 and graph.num_edges <= 4000
    assert all(pattern._UNIT_FACTORIES[k] is f for k, f in registry.items()
               if not k.startswith('fiberscope.cell.v1:'))
    if topology == 'legacy':
        assert graph.metadata.get('pattern', {}).get('unit', 'kagome') == 'kagome'
    else:
        assert identity['topology_compiler'] == 'fibernet.gen.manufacturing:topnet26'

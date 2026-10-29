import json
from pathlib import Path
import numpy as np
import pytest
from fibernet import StructureGraph
from fibernet.sim import transient_thermal as th


def graph(ids=(101,907),edges=((11,101,907),(31,101,907))):
    return StructureGraph.from_dict({'dimension':2,'nodes':[{'id':v,'position':[i,0,0]} for i,v in enumerate(ids)],'edges':[{'id':e,'node_i':a,'node_j':b} for e,a,b in edges]})


def run(g=None,C=None,G=None,T=None,t=None,Q=None,bc=((907,310),)):
    return th.solve_transient_thermal(g or graph(),C or {101:2,907:1}, {11:.2,31:.3} if G is None else G,T or {101:300,907:310},[0,.01,.02] if t is None else t,source_W=Q,fixed_temperature_K=bc,length_unit='m')


def test_parallel_sparse_ids_reservoir_and_energy():
    g=graph();before=g.to_dict();r=run(g)
    np.testing.assert_array_equal(r.edge_ids,[11,31]);np.testing.assert_array_equal(r.node_ids,[101,907])
    np.testing.assert_allclose(r.edge_heat_rate_W[:,0]/r.edge_heat_rate_W[:,1],2/3)
    assert np.all(r.reservoir_injection_W[:,1]>0)
    assert not r.interval_diagnostics_available[0] and np.isnan(r.free_node_residual_J[0]).all()
    assert np.isnan(r.free_node_residual_J[:,1]).all()
    np.testing.assert_allclose(r.node_stored_heat_change_J[-1].sum(),np.sum(np.diff(r.times_s)*r.reservoir_injection_W[1:].sum(axis=1)),rtol=1e-10)
    assert g.to_dict()==before and not r.temperature_K.flags.writeable
    provenance = json.loads(r.provenance_json)
    assert not provenance['time_accuracy_verified']
    assert provenance['validation_contract'] == 'transport-roundoff-2'
    assert provenance['thermal_capacity_reconstruction_profile'] == 'binary64-rn-capacity-rho-1'


def test_unanchored_two_node_discrete_reference():
    r=run(bc=(),t=np.linspace(0,1,11))
    mean=(2*300+310)/3
    diff=-10*(1+.1*.5*(1/2+1))**(-np.arange(11))
    np.testing.assert_allclose(r.temperature_K[:,0],mean+diff/3,atol=1e-12)
    np.testing.assert_allclose(r.node_stored_heat_change_J.sum(axis=1),0,atol=2e-12)


def test_isolated_source_and_zero_source():
    g=graph(ids=(3,99),edges=())
    r=run(g,{3:2,99:7},{},{3:300,99:305},[0,1,2],{3:4,99:0},bc=())
    np.testing.assert_array_equal(r.temperature_K,[[300,305],[302,305],[304,305]])
    assert r.edge_heat_rate_W.shape==(3,0)


def test_zero_g_retains_edge_rows():
    r=run(G={11:0,31:0});assert np.all(r.edge_heat_rate_W==0)
    np.testing.assert_array_equal(r.temperature_K,[[300,310]]*3)


def test_bias_lost_increment_rejected():
    g=graph(ids=(7,),edges=())
    with pytest.raises(th.ThermalSolveError,match='lost') as e:
        run(g,{7:1},{},{7:1e16},[0,1],{7:1},bc=())
    assert e.value.partial_result.times_s.tolist()==[0]
    assert e.value.attempted_step==1 and not e.value.partial_result.completed


def test_tiny_flux_underflow_rejected_initial():
    with pytest.raises(th.ThermalSolveError,match='not representable') as e:
        run(G={11:1e-300,31:0},T={101:1e-100,907:0},bc=((907,0),))
    assert e.value.partial_result is None and e.value.attempted_step==0


def test_dt_conductance_underflow_rejected():
    with pytest.raises(th.ThermalSolveError,match='dt times conductance'):
        run(G={11:1e-300,31:0},t=[0,1e-100])


def test_mixed_scale_wrong_solver_rejected(monkeypatch):
    g=graph(ids=(1,2,3,4),edges=((5,1,2),(6,3,4)))
    original=th.spsolve
    def wrong(A,b):
        x=original(A,b);x[-1]+=.1;return x
    monkeypatch.setattr(th,'spsolve',wrong)
    with pytest.raises(th.ThermalSolveError) as e:
        run(g,{1:1e20,2:1e20,3:1,4:1},{5:1e20,6:1},{1:300,2:310,3:300,4:310},[0,1],bc=((2,310),(4,310)))
    assert e.value.partial_result.times_s.tolist()==[0]


@pytest.mark.parametrize('kwargs',[
 {'C':{101:0,907:1}}, {'G':{11:-1,31:1}}, {'G':{11:1}},
 {'T':{101:300,907:309}}, {'bc':[(907,310),(907,311)]},
 {'t':[0,1,3]}, {'t':[0,0]}, {'t':[1,2]}, {'t':[0,True]},
 {'Q':{101:-1,907:0}}, {'C':{'101':2,907:1}}, {'bc':{'907':310}},
])
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError):run(**kwargs)


def test_repeat_digest_and_fields():
    a,b=run(),run();assert a.input_sha256==b.input_sha256
    np.testing.assert_array_equal(a.temperature_K,b.temperature_K)


def test_initial_reservoir_overflow_has_no_accepted_frame():
    g=graph(ids=(1,2),edges=((7,1,2),))
    with pytest.raises(th.ThermalSolveError,match='reservoir overflow') as e:
        run(g,{1:1,2:1},{7:1e308},{1:0,2:1},[0,1],{1:1e308,2:0},bc=((1,0),(2,1)))
    assert e.value.partial_result is None
    assert e.value.attempted_step==0 and e.value.attempted_time_s==0


@pytest.mark.parametrize('metadata',[[],0,False])
def test_metadata_must_be_dictionary(metadata):
    with pytest.raises(ValueError,match='metadata'):
        th.solve_transient_thermal(graph(),{101:2,907:1},{11:.2,31:.3},{101:300,907:310},[0,1],length_unit='m',model_metadata=metadata)


def test_to_dict_strict_json_preserves_fields_availability_and_arrays():
    from dataclasses import fields
    r = run()
    arrays_before = {f.name: getattr(r, f.name).copy() for f in fields(r)
                     if isinstance(getattr(r, f.name), np.ndarray)}
    data = json.loads(json.dumps(r.to_dict(), allow_nan=False))
    assert set(data) == {f.name for f in fields(r)} - {'provenance_json'} | {'provenance'}
    assert data['completed'] is True
    assert data['input_sha256'] == r.input_sha256
    assert data['provenance'] == json.loads(r.provenance_json)
    assert data['node_ids'] == [101, 907]
    assert data['edge_ids'] == [11, 31]
    assert data['endpoints_by_id'] == [[101, 907], [101, 907]]
    assert data['fixed_node_ids'] == [907]
    assert data['interval_diagnostics_available'] == [False, True, True]
    for name in ('free_node_residual_J', 'node_acceptance_bound_J'):
        assert data[name][0] == [None, None]
        assert all(row[1] is None for row in data[name])
        assert all(np.isfinite(row[0]) for row in data[name][1:])
    for name in ('component_energy_residual_J', 'component_acceptance_bound_J'):
        assert data[name][0] == [None]
        np.testing.assert_array_equal(data[name][1:], getattr(r, name)[1:])
    assert data['node_stored_heat_change_J'][0] == [0., 0.]
    for name, before in arrays_before.items():
        np.testing.assert_array_equal(getattr(r, name), before)
        assert not getattr(r, name).flags.writeable
    # Detached nested fields cannot mutate the typed result or later exports.
    data['temperature_K'][0][0] = -1
    data['provenance']['units']['T'] = 'wrong'
    assert r.to_dict()['temperature_K'][0][0] == 300.
    assert r.to_dict()['provenance']['units']['T'] == 'K'


def test_to_dict_actual_failed_prefix_retains_initial_frame():
    g = graph(ids=(7,), edges=())
    with pytest.raises(th.ThermalSolveError, match='lost') as caught:
        run(g, {7: 1}, {}, {7: 1e16}, [0, 1], {7: 1}, bc=())
    error = caught.value
    data = json.loads(json.dumps(error.partial_result.to_dict(), allow_nan=False))
    assert error.attempted_step == 1 and error.attempted_time_s == 1.
    assert data['completed'] is False
    assert data['times_s'] == [0.]
    assert data['temperature_K'] == [[1e16]]
    assert data['interval_diagnostics_available'] == [False]
    assert data['free_node_residual_J'] == [[None]]
    assert data['component_energy_residual_J'] == [[None]]


@pytest.mark.parametrize('name', [
    'temperature_K', 'edge_heat_rate_W', 'reservoir_injection_W',
    'node_stored_heat_change_J', 'free_node_residual_J',
    'node_acceptance_bound_J', 'component_energy_residual_J',
    'component_acceptance_bound_J',
])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_to_dict_rejects_nonfinite_available_values(name, bad):
    from dataclasses import replace
    r = run()
    values = getattr(r, name).copy()
    values[1, 0] = bad
    with pytest.raises(ValueError, match=name):
        replace(r, **{name: values}).to_dict()


@pytest.mark.parametrize('bad', [0., np.inf, -np.inf])
@pytest.mark.parametrize('row_column', [(0, 0), (1, 1)])
def test_to_dict_rejects_inconsistent_unavailable_sentinels(bad, row_column):
    from dataclasses import replace
    r = run()
    values = r.free_node_residual_J.copy()
    values[row_column] = bad
    with pytest.raises(ValueError, match='unavailable diagnostics'):
        replace(r, free_node_residual_J=values).to_dict()


@pytest.mark.parametrize('mask', [[True, True, True], [False, False, True],
                                 [0, 1, 1], [False]])
def test_to_dict_rejects_inconsistent_interval_mask(mask):
    from dataclasses import replace
    with pytest.raises(ValueError, match='interval_diagnostics_available'):
        replace(run(), interval_diagnostics_available=np.asarray(mask)).to_dict()


def test_to_dict_rejects_nonfinite_provenance():
    from dataclasses import replace
    with pytest.raises(ValueError):
        replace(run(), provenance_json='{"metadata": {"bad": NaN}}').to_dict()


def test_to_dict_empty_edges_and_all_fixed_nodes():
    r = run(graph(ids=(3, 99), edges=()), {3: 2, 99: 7}, {},
            {3: 300, 99: 305}, [0, 1], {3: 4, 99: 0},
            bc=((3, 300), (99, 305)))
    data = r.to_dict()
    assert data['edge_heat_rate_W'] == [[], []]
    assert data['free_node_residual_J'] == [[None, None], [None, None]]
    assert data['node_acceptance_bound_J'] == [[None, None], [None, None]]
    np.testing.assert_array_equal(data['component_energy_residual_J'][1:],
                                  r.component_energy_residual_J[1:])

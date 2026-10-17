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
    assert not json.loads(r.provenance_json)['time_accuracy_verified']


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

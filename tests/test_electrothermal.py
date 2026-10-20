"""Focused contracts for the experimental explicit electrothermal module."""
import copy
from dataclasses import FrozenInstanceError
import json
import math

import pytest

from fibernet import StructureGraph
from fibernet.sim import electrothermal as et
from fibernet.sim._thermal_step import ThermalResidualRejection
from fibernet.sim.linear_transport import FloatingComponentError


def graph():
    return StructureGraph.from_dict({'dimension':2,
        'nodes':[{'id':101,'position':[0,0,0]},{'id':907,'position':[1,0,0]}],
        'edges':[{'id':11,'node_i':101,'node_j':907}]})


def law_inputs():
    return dict(G0_S={11:1.},alpha_per_K={11:.01},reference_temperature_K={11:300.},
        endpoint_temperature_weights={11:{101:.5,907:.5}},temperature_domain_K=[290.,350.])


def inputs():
    return dict(prescribed_voltage_V={101:0.,907:4.},capacity_J_per_K={101:2.,907:2.},
        conductance_W_per_K={11:0.},initial_temperature_K={101:300.,907:300.},
        times_s=[0.,.1,.2],endpoint_heat_weights={11:{101:.5,907:.5}},length_unit='m')


def run(**changes):
    data=inputs();data.update(changes)
    return et.solve_electrothermal(graph(),et.TemperatureConductanceLaw(**law_inputs()),**data)


@pytest.mark.parametrize('field,value',[
    ('G0_S',-1.),('G0_S',float('nan')),('G0_S',True),
    ('alpha_per_K',-.01),('alpha_per_K',float('inf')),('alpha_per_K',True),
    ('reference_temperature_K',-1.),('reference_temperature_K',351.),('reference_temperature_K',True),
])
def test_law_scalar_rejections(field,value):
    data=law_inputs();data[field]={11:value}
    with pytest.raises(ValueError):et.TemperatureConductanceLaw(**data)


@pytest.mark.parametrize('domain',[[350.,290.],[-1.,350.],[290.,float('inf')],[290.],[True,350.]])
def test_explicit_domain_rejections(domain):
    data=law_inputs();data['temperature_domain_K']=domain
    with pytest.raises(ValueError):et.TemperatureConductanceLaw(**data)


@pytest.mark.parametrize('mutation',['string_edge','missing_alpha','unknown_temperature_weight_edge','negative_weight','nan_weight','bad_sum','bool_endpoint'])
def test_law_identity_and_weight_rejections(mutation):
    data=law_inputs()
    if mutation=='string_edge':data['G0_S']={'11':1.}
    elif mutation=='missing_alpha':data['alpha_per_K']={}
    elif mutation=='unknown_temperature_weight_edge':data['endpoint_temperature_weights']={99:{101:.5,907:.5}}
    elif mutation=='negative_weight':data['endpoint_temperature_weights']={11:{101:-.1,907:1.1}}
    elif mutation=='nan_weight':data['endpoint_temperature_weights']={11:{101:float('nan'),907:.5}}
    elif mutation=='bad_sum':data['endpoint_temperature_weights']={11:{101:.3,907:.3}}
    elif mutation=='bool_endpoint':data['endpoint_temperature_weights']={11:{True:.5,907:.5}}
    with pytest.raises(ValueError):et.TemperatureConductanceLaw(**data)


@pytest.mark.parametrize('weights',[{11:{101:1.}},{11:{101:.5,999:.5}}])
def test_temperature_weights_bind_actual_endpoints(weights):
    data=law_inputs();data['endpoint_temperature_weights']=weights
    law=et.TemperatureConductanceLaw(**data)
    with pytest.raises(ValueError,match='endpoints'):et.solve_electrothermal(graph(),law,**inputs())


@pytest.mark.parametrize('weights',[{}, {11:{101:1.}}, {11:{101:.5,999:.5}},
    {11:{101:.4,907:.4}}, {11:{101:float('nan'),907:.5}}, {'11':{101:.5,907:.5}}])
def test_heat_partition_weights_are_validated_independently(weights):
    with pytest.raises(ValueError):run(endpoint_heat_weights=weights)


@pytest.mark.parametrize('changes',[
    {'capacity_J_per_K':{101:0.,907:2.}},
    {'capacity_J_per_K':{101:2.}},
    {'capacity_J_per_K':{101:2.,907:2.,999:2.}},
    {'capacity_J_per_K':{101:True,907:2.}},
    {'conductance_W_per_K':{11:-.1}},
    {'conductance_W_per_K':{}},
    {'initial_temperature_K':{101:300.}},
    {'initial_temperature_K':{101:289.,907:300.}},
    {'external_source_W':{101:-1.,907:0.}},
    {'external_source_W':{101:0.}},
    {'prescribed_voltage_V':[(101,0.),(101,1.),(907,4.)]},
    {'fixed_temperature_K':{907:301.}},
])
def test_physical_input_map_rejections(changes):
    with pytest.raises(ValueError):run(**changes)


@pytest.mark.parametrize('maximum',[0,26,True,1.5])
def test_iteration_limit_is_explicit_and_bounded(maximum):
    with pytest.raises(ValueError,match='max_iterations'):run(max_iterations=maximum)


@pytest.mark.parametrize('times',[[.1,.2],[0.,0.],[0.,.1,.3],[0.,True]])
def test_time_grid_rejections(times):
    with pytest.raises(ValueError):run(times_s=times)


def test_history_and_metadata_caps_fail_before_electrical_solve(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('unexpected electrical solve')
    monkeypatch.setattr(et,'solve_linear_transport',forbidden)
    with pytest.raises(ValueError,match='time grid'):run(times_s=list(range(2050)))
    with pytest.raises(ValueError,match='metadata'):run(model_metadata={'payload':'x'*17000})


def test_inactive_edge_still_requires_valid_domain():
    data=law_inputs();data['G0_S']={11:0.};data['temperature_domain_K']=[0.,350.]
    with pytest.raises(ValueError,match='denominator'):et.TemperatureConductanceLaw(**data)


def test_active_conductance_overflow_is_not_clipped():
    data=law_inputs();data.update(G0_S={11:1e308},alpha_per_K={11:.09})
    law=et.TemperatureConductanceLaw(**data)
    kwargs=inputs();kwargs['initial_temperature_K']={101:290.,907:290.}
    with pytest.raises(ArithmeticError,match='conductance'):et.solve_electrothermal(graph(),law,**kwargs)


def test_floating_electrical_component_is_not_silently_anchored():
    data=law_inputs();data['G0_S']={11:0.}
    kwargs=inputs();kwargs['prescribed_voltage_V']={101:0.}
    with pytest.raises(FloatingComponentError):et.solve_electrothermal(graph(),et.TemperatureConductanceLaw(**data),**kwargs)


def test_inactive_edge_rows_and_zero_flux_are_retained():
    data=law_inputs();data['G0_S']={11:0.}
    result=et.solve_electrothermal(graph(),et.TemperatureConductanceLaw(**data),**inputs())
    assert result.edge_ids==(11,) and result.completed
    for frame in result.frames:
        assert frame.conductance_S==(0.,)
        assert json.loads(frame.electrical_json)['current_A']==[0.]
        assert json.loads(frame.heat_load_json)['total_joule_power_W']==0.


def test_law_and_results_are_detached_and_immutable():
    data=law_inputs();law=et.TemperatureConductanceLaw(**data)
    data['G0_S'][11]=9.;data['endpoint_temperature_weights'][11][101]=0.;data['temperature_domain_K'][0]=0.
    assert law.G0_S[11]==1. and law.endpoint_temperature_weights[11][101]==.5
    assert law.temperature_domain_K==(290.,350.)
    with pytest.raises(TypeError):law.G0_S[11]=2.
    detached=law.to_dict();detached['endpoint_temperature_weights'][11][101]=0.
    assert law.endpoint_temperature_weights[11][101]==.5
    geometry=graph();kwargs=inputs();before_graph=copy.deepcopy(geometry.to_dict());before_inputs=copy.deepcopy(kwargs)
    metadata={'nested':{'name':'original'}}
    result=et.solve_electrothermal(geometry,law,model_metadata=metadata,**kwargs)
    metadata['nested']['name']='changed'
    provenance = json.loads(result.provenance_json)
    assert provenance['metadata']['nested']['name']=='original'
    assert provenance['validation_contract'] == 'transport-roundoff-2'
    assert provenance['thermal_capacity_reconstruction_profile'] == 'binary64-rn-capacity-rho-1'
    assert geometry.to_dict()==before_graph and kwargs==before_inputs
    with pytest.raises(FrozenInstanceError):result.completed=False
    with pytest.raises(TypeError):result.frames[-1].temperature_K[0]=0.
    parsed=json.loads(result.frames[-1].electrical_json)
    assert parsed['provenance']['validation_contract'] == 'transport-roundoff-2'
    assert 'thermal_capacity_reconstruction_profile' not in parsed['provenance']
    assert json.loads(result.frames[-1].heat_load_json)['provenance']['electrical_validation_contract'] == 'transport-roundoff-2'
    parsed['conductance_S'][0]=999.
    assert json.loads(result.frames[-1].electrical_json)['conductance_S'][0]!=999.


def test_final_fields_share_accepted_temperature_and_preserve_external_source():
    geometry=StructureGraph.from_dict({'dimension':2,
        'nodes':[{'id':101,'position':[0,0,0]},{'id':303,'position':[1,0,0]},{'id':907,'position':[2,0,0]}],
        'edges':[{'id':11,'node_i':101,'node_j':303},{'id':29,'node_i':303,'node_j':907}]})
    law=et.TemperatureConductanceLaw({11:1.,29:2.},{11:.01,29:.02},{11:300.,29:300.},
        {11:{101:.25,303:.75},29:{303:.5,907:.5}},(290.,350.))
    C={101:2.,303:3.,907:4.};g={11:.1,29:.2};external={101:.0,303:.1,907:.2}
    result=et.solve_electrothermal(geometry,law,{101:0.,907:4.},C,g,{101:300.,303:305.,907:310.},[0.,.01,.02],
        endpoint_heat_weights={11:{101:.8,303:.2},29:{303:.3,907:.7}},
        external_source_W=external,fixed_temperature_K={101:300.},length_unit='m')
    rows={n:i for i,n in enumerate(result.node_ids)}
    assert result.thermal_component_labels==(0,0,0)
    for index,frame in enumerate(result.frames):
        T=dict(zip(result.node_ids,frame.temperature_K));G={11:1./(1.+.01*(.25*T[101]+.75*T[303]-300.)),
            29:2./(1.+.02*(.5*T[303]+.5*T[907]-300.))}
        electric=json.loads(frame.electrical_json);heat=json.loads(frame.heat_load_json)
        assert frame.conductance_S==tuple(G[e] for e in result.edge_ids)
        assert electric['conductance_S']==list(frame.conductance_S)
        assert heat['external_node_power_W']==[external[n] for n in result.node_ids]
        assert frame.temperature_K[rows[101]]==300.
        for n,Q,J in zip(result.node_ids,heat['total_node_power_W'],heat['joule_node_power_W']):
            assert Q==math.fsum([J,external[n]])
        if index:
            previous=result.frames[index-1];dt=frame.time_s-previous.time_s
            q={e:g[e]*(T[geometry.edges[e].node_i]-T[geometry.edges[e].node_j]) for e in result.edge_ids}
            for n in (303,907):
                i=rows[n];terms=[C[n]*(T[n]-previous.temperature_K[i]),-dt*heat['total_node_power_W'][i]]
                terms += [dt*q[e] if geometry.edges[e].node_i==n else -dt*q[e]
                          for e in result.edge_ids if n in (geometry.edges[e].node_i,geometry.edges[e].node_j)]
                assert abs(math.fsum(terms))<=frame.node_acceptance_bound_J[i]
            assert all(abs(r)<=b for r,b in zip(frame.component_energy_residual_J,frame.component_acceptance_bound_J))
    assert json.loads(result.frames[-1].electrical_json)['voltage_V'][1]!=json.loads(result.frames[0].electrical_json)['voltage_V'][1]


def test_nonconvergence_and_domain_failure_keep_accepted_prefix_only():
    with pytest.raises(et.ElectrothermalSolveError) as caught:run(max_iterations=1)
    failure=caught.value
    assert failure.attempted_step==1 and not failure.partial_result.completed
    assert [f.time_s for f in failure.partial_result.frames]==[0.]
    assert len(failure.partial_result.iteration_history)==1
    data=law_inputs();data['temperature_domain_K']=[300.,300.5]
    with pytest.raises(et.ElectrothermalSolveError) as caught:
        et.solve_electrothermal(graph(),et.TemperatureConductanceLaw(**data),**inputs())
    failure=caught.value
    assert failure.attempted_step==2 and not failure.partial_result.completed
    assert [f.time_s for f in failure.partial_result.frames]==[0.,.1]


@pytest.mark.parametrize('kind',['precision','finite_residual'])
def test_only_typed_finite_residual_rejections_are_retried(monkeypatch,kind):
    calls=[]
    def reject(**kwargs):
        calls.append(kwargs)
        error=ArithmeticError if kind=='precision' else ThermalResidualRejection
        raise error('component energy balance failed')
    monkeypatch.setattr(et,'evaluate_thermal_step',reject)
    with pytest.raises(et.ElectrothermalSolveError) as caught:run(times_s=[0.,.1],max_iterations=2)
    failure=caught.value
    assert [f.time_s for f in failure.partial_result.frames]==[0.]
    if kind=='precision':
        assert len(calls)==1 and str(failure)=='component energy balance failed'
    else:
        assert len(calls)==2 and 'iteration limit' in str(failure)

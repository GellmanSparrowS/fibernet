"""Pure admission/delegation/error contracts. Physical assembly/solve forbidden."""
import ast
import builtins
import copy
from dataclasses import FrozenInstanceError
import importlib.util
import json
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from fibernet import StructureGraph
from fibernet.sim import _thermoelastic as owner
from fibernet.sim.thermoelastic import ThermoelasticSizeLimits, solve_thermoelastic_network_snapshot
from fibernet.sim.saved_temperature import prepare_saved_temperature


@pytest.fixture(autouse=True)
def no_physics():
    with patch.object(owner,'_assemble_snapshot_system',side_effect=AssertionError('physical assembly forbidden')), \
         patch.object(owner,'_solve_constrained_sparse',side_effect=AssertionError('physical solve forbidden')):
        yield


def compact():
    g=StructureGraph.from_dict({'dimension':2,'nodes':[
        {'id':91,'position':[.1,0.,0.]},{'id':7,'position':[0.,0.,0.]}],
        'edges':[{'id':83,'node_i':91,'node_j':7,'radius':.001},
                 {'id':11,'node_i':7,'node_j':91,'radius':.002}]})
    kw=dict(length_unit='m',E_Pa=1e9,expansion_alpha_per_K={11:1e-5,83:1e-5},
        stress_free_temperature_K=300.,temperature_K={7:300.,91:300.1},
        constraints={7:(0.,0.,0.)},loads={91:(1e-10,0.,0.)})
    return g,kw


def thermal_fixture(g):
    ids=list(g.nodes);eids=list(g.edges)
    return {'node_ids':ids,'edge_ids':eids,
        'endpoints_by_id':[[g.edges[e].node_i,g.edges[e].node_j] for e in eids],
        'positions':[g.nodes[n].position.tolist() for n in ids],
        'times_s':[0.,1.],'temperature_K':[[300.]*len(ids),[300.1]*len(ids)],
        'completed':True,'input_sha256':'a'*64,
        'provenance':{'model':'constant_lumped_thermal','length_unit':'m','units':{'T':'K'}}}


def limits():return ThermoelasticSizeLimits(max_nodes=200,max_edges=300)


def capture(g,kw,chosen=None):
    marker=object()
    with patch.object(owner,'_snapshot_core',return_value=marker) as core:
        assert solve_thermoelastic_network_snapshot(g,**kw,size_limits=chosen) is marker
        assert core.call_count==1
        assert core.call_args.args==(g,)
        return core.call_args.kwargs


def test_public_exports_are_existing_owner_functions():
    from fibernet import sim
    from fibernet.sim import thermoelastic as api
    assert api.solve_thermoelastic_network_snapshot is owner.solve_thermoelastic_network_snapshot
    assert sim.solve_thermoelastic_network_snapshot is owner.solve_thermoelastic_network_snapshot
    assert sim.ThermoelasticSizeLimits is ThermoelasticSizeLimits
    assert api.solve_thermoelastic_snapshot is owner.solve_thermoelastic_snapshot
    assert api.ThermoelasticSnapshot is owner.ThermoelasticSnapshot
    assert set(api.__all__)<=set(dir(api))
    with pytest.raises(AttributeError):getattr(api,'not_a_backend')


def test_limit_record_module_source_load_is_stdlib_only():
    path=Path(__file__).resolve().parents[1]/'fibernet/sim/thermoelastic.py'
    original=builtins.__import__
    def guard(name,*a,**k):
        assert not name.startswith(('numpy','scipy','torch','taichi','fibernet'))
        return original(name,*a,**k)
    spec=importlib.util.spec_from_file_location('_bounded_thermoelastic_contract_test',path)
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{spec.name:module}),patch.object(builtins,'__import__',side_effect=guard):
        spec.loader.exec_module(module)
        assert module.ThermoelasticSizeLimits(2,1).max_nodes==2


@pytest.mark.parametrize('name',['max_nodes','max_edges'])
@pytest.mark.parametrize('bad',[True,False,None,0,-1,1.5,'200',float('inf'),2**100,np.int64(200)])
def test_limits_strict_finite_platform_integer(name,bad):
    kw={'max_nodes':200,'max_edges':300};kw[name]=bad
    with pytest.raises(ValueError,match=name):ThermoelasticSizeLimits(**kw)


def test_limits_required_immutable_and_revalidated():
    with pytest.raises(TypeError):ThermoelasticSizeLimits()
    chosen=limits()
    with pytest.raises(FrozenInstanceError):chosen.max_nodes=1
    object.__setattr__(chosen,'max_nodes',True)
    g,kw=compact()
    with pytest.raises(ValueError,match='max_nodes'):capture(g,kw,chosen)


@pytest.mark.parametrize('bad',[True,{},300,'large'])
def test_no_untyped_or_implicit_optin(bad):
    g,kw=compact()
    with pytest.raises(TypeError,match='ThermoelasticSizeLimits'):capture(g,kw,bad)


def test_subclass_cannot_override_size_admission():
    class NotTheContract(ThermoelasticSizeLimits):
        def _admit(self,graph):return {}
    g,kw=compact()
    with pytest.raises(TypeError):capture(g,kw,NotTheContract(200,300))


def test_bounded_route_requires_authoritative_graph():
    with pytest.raises(TypeError,match='StructureGraph'):limits()._admit(SimpleNamespace(nodes={1:0,2:0},edges={1:0}))


@pytest.mark.parametrize('nodes,edges',[(193,1),(2,257),(1,1),(2,0)])
def test_legacy_caps_and_exact_error_precede_preflight(nodes,edges):
    fake=SimpleNamespace(nodes=range(nodes),edges=range(edges))
    with patch.object(owner,'_network_preflight') as preflight:
        with pytest.raises(ValueError) as error:
            solve_thermoelastic_network_snapshot(fake,length_unit='bad',E_Pa=0,
                expansion_alpha_per_K={},stress_free_temperature_K=0,temperature_K={},constraints={})
        assert str(error.value)=='network profile requires 2..192 nodes and 1..256 edges'
        preflight.assert_not_called()


def test_default_small_kwargs_and_optin_preparation_match():
    g,kw=compact();original=copy.deepcopy(g.to_dict());old=capture(g,kw);new=capture(g,kw,limits())
    assert old['profile']==owner.NETWORK_PROFILE
    assert new.pop('profile')==owner.BOUNDED_NETWORK_PROFILE
    old.pop('profile');admission=new.pop('_size_admission')
    assert admission['physical_edge_count']==2 and admission['assembly_entries_upper_bound']==72
    assert admission['factorization_memory_or_time_guaranteed'] is False
    for key in old:
        if isinstance(old[key],np.ndarray):np.testing.assert_array_equal(old[key],new[key])
        else:assert old[key]==new[key]
    assert g.to_dict()==original
    assert old['ids']==[7,91] and old['eids']==[11,83]
    assert old['edges'].tolist()==[[0,1],[1,0]]


def test_generic_193_node_chain_optin_not_exact_graph_fingerprint():
    ids=[7+5*i for i in range(193)]
    g=StructureGraph.from_dict({'dimension':2,'nodes':[{'id':n,'position':[.001*i,0.,0.]} for i,n in enumerate(ids)],
        'edges':[{'id':1000+3*i,'node_i':ids[i],'node_j':ids[i+1],'radius':1e-5} for i in range(192)]})
    kw=dict(length_unit='m',E_Pa=1e9,expansion_alpha_per_K={e:1e-5 for e in g.edges},
        stress_free_temperature_K=300.,temperature_K={n:300. for n in g.nodes},constraints={ids[0]:(0.,0.,0.)})
    with pytest.raises(ValueError,match='2..192'):capture(g,kw)
    prepared=capture(g,kw,ThermoelasticSizeLimits(193,192))
    assert prepared['_size_admission']['dof_count']==579
    assert len(prepared['eids'])==192


@pytest.mark.parametrize('chosen,pattern',[(ThermoelasticSizeLimits(2,1),'physical-edge')])
def test_bounds_precede_geometry_or_core(chosen,pattern):
    g,kw=compact()
    with patch.object(owner,'_network_preflight') as prepare:
        with pytest.raises(ValueError,match=pattern):capture(g,kw,chosen)
        prepare.assert_not_called()


def test_node_budget_checked_before_geometry():
    g,kw=compact();g.nodes[104]=copy.deepcopy(g.nodes[91])
    with patch.object(owner,'_network_preflight') as prepare:
        with pytest.raises(ValueError,match='node count'):capture(g,kw,ThermoelasticSizeLimits(2,2))
        prepare.assert_not_called()


@pytest.mark.parametrize('kind',['units','modulus','heterogeneous_E','alpha_missing','temperature_missing','thermal_bound',
    'unsupported_dimension','nonplanar','curved','short','radius','unsupported_support','wrong_load_id','isolated'])
def test_optin_never_relaxes_non_size_preflight(kind):
    g,kw=compact()
    if kind=='units':kw['length_unit']='cm'
    if kind=='modulus':kw['E_Pa']=0
    if kind=='heterogeneous_E':kw['E_Pa']={11:1e9,83:1e9}
    if kind=='alpha_missing':kw['expansion_alpha_per_K'].pop(11)
    if kind=='temperature_missing':kw['temperature_K'].pop(91)
    if kind=='thermal_bound':kw['temperature_K'][91]=302.
    if kind=='unsupported_dimension':
        record=g.to_dict();record['dimension']=3;g=StructureGraph.from_dict(record)
    if kind=='nonplanar':g.nodes[91].position[2]=1.
    if kind=='curved':g.edges[11].internal_points=np.array([[.05,.01,0.]])
    if kind=='short':g.nodes[91].position[0]=1e-13
    if kind=='radius':g.edges[11].radius=0.
    if kind=='unsupported_support':kw['constraints']={7:(0.,None,None)}
    if kind=='wrong_load_id':kw['loads']={101:(0.,0.,0.)}
    if kind=='isolated':
        record=g.to_dict();record['nodes'].append({'id':111,'position':[.2,0.,0.]});record['next_node_id']=112;g=StructureGraph.from_dict(record)
        kw['temperature_K'][111]=300.
    with patch.object(owner,'_snapshot_core') as core:
        with pytest.raises((ValueError,FloatingPointError)):
            solve_thermoelastic_network_snapshot(g,**kw,size_limits=limits())
        core.assert_not_called()


def test_reordering_and_mixed_support_mask_nonzero_prescriptions():
    g,kw=compact();kw['constraints']={7:(0.,None,1e-7),91:(None,0.,None)}
    a=capture(g,kw,limits());record=g.to_dict();record['nodes'].reverse();record['edges'].reverse();g=StructureGraph.from_dict(record)
    b=capture(g,kw,limits())
    assert a['ids']==b['ids'] and a['eids']==b['eids']
    assert a['mask'].tolist()==[True,False,True,False,True,False]
    np.testing.assert_array_equal(a['values'],b['values'])
    assert a['values'][0,2]==1e-7


def test_core_failure_propagates_once_without_fallback():
    g,kw=compact();failure=FloatingPointError('injected owner rejection')
    with patch.object(owner,'_snapshot_core',side_effect=failure) as core:
        with pytest.raises(FloatingPointError) as caught:solve_thermoelastic_network_snapshot(g,**kw,size_limits=limits())
        assert caught.value is failure and core.call_count==1


def test_saved_adapter_explicit_contract_identity_and_no_extra_schema():
    g,kw=compact();frame=prepare_saved_temperature(g,thermal_fixture(g),frame_index=1,length_unit='m')
    chosen=limits();control={k:v for k,v in kw.items() if k not in ('temperature_K','length_unit')}
    with patch.object(owner,'solve_thermoelastic_network_snapshot',return_value=SimpleNamespace(to_dict=lambda:{'stub':True})) as solve:
        result=frame.solve(g,profile='network',size_limits=chosen,**control)
        assert solve.call_args.kwargs['size_limits'] is chosen and solve.call_count==1
        assert result['mechanical']=={'stub':True} and result['temperature_source']['source_completed'] is True
        assert result['feedback_to_electrical_or_thermal'] is False
    with patch.object(owner,'solve_thermoelastic_network_snapshot') as solve:
        g.edges[11].radius*=2
        with pytest.raises(ValueError,match='graph changed'):frame.solve(g,profile='network',size_limits=chosen,**control)
        solve.assert_not_called()


def test_saved_elementary_rejects_size_limits_not_silently_ignored():
    g,kw=compact();frame=prepare_saved_temperature(g,thermal_fixture(g),frame_index=1,length_unit='m')
    with patch.object(owner,'solve_thermoelastic_snapshot') as solve:
        with pytest.raises(ValueError,match='only for profile=network'):
            frame.solve(g,profile='elementary',size_limits=limits(),E_Pa=1e9,
                expansion_alpha_per_K={11:1e-5,83:1e-5},stress_free_temperature_K=300.,constraints={7:(0.,0.,0.)})
        solve.assert_not_called()


def test_partial_saved_source_still_requires_explicit_consent():
    g,kw=compact();data=thermal_fixture(g);data['completed']=False
    with pytest.raises(ValueError):prepare_saved_temperature(g,data,frame_index=1,length_unit='m')
    frame=prepare_saved_temperature(g,data,frame_index=1,length_unit='m',allow_partial=True)
    assert frame.to_dict()['source_completed'] is False


def dummy_assembly(**kwargs):
    # Arbitrary identity storage and zero data, not a physical FEM assembly.
    from scipy.sparse import eye
    recovery=lambda *a,**k:(np.zeros(2),np.zeros(2),np.zeros(2),np.zeros((2,2)),np.zeros((2,3)))
    return SimpleNamespace(_compute_element_stress_2d=recovery),np.zeros(6),eye(6,format='csr'),[0,1]


def test_dummy_core_result_owns_size_metadata_and_retains_default_schema():
    g,kw=compact();kw['loads']=None;kw['temperature_K']={7:300.,91:300.}
    with patch.object(owner,'_assemble_snapshot_system',side_effect=dummy_assembly), \
         patch.object(owner,'_solve_constrained_sparse',return_value=np.zeros(6)) as delegate:
        plain=solve_thermoelastic_network_snapshot(g,**kw).to_dict()
        bounded=solve_thermoelastic_network_snapshot(g,**kw,size_limits=limits())
        result=bounded.to_dict()
        assert delegate.call_count==2
    assert 'size_admission' not in plain
    assert result['profile']==owner.BOUNDED_NETWORK_PROFILE
    assert set(result)==set(plain)|{'size_admission'}
    for key in set(plain)-{'profile','input_sha256'}:assert result[key]==plain[key]
    result['size_admission']['limits']['max_nodes']=0
    assert bounded.to_dict()['size_admission']['limits']['max_nodes']==200
    assert plain['input_sha256']!=result['input_sha256']


def test_legacy_upper_bound_admits_without_size_metadata():
    fake=SimpleNamespace(nodes=range(192),edges=range(256));prepared={'profile':owner.NETWORK_PROFILE}
    with patch.object(owner,'_network_preflight',return_value=prepared) as preflight, \
         patch.object(owner,'_snapshot_core',return_value='dummy') as core:
        assert solve_thermoelastic_network_snapshot(fake,length_unit='m',E_Pa=1e9,
            expansion_alpha_per_K={},stress_free_temperature_K=300.,temperature_K={},constraints={})=='dummy'
        assert preflight.call_count==core.call_count==1
        assert core.call_args.kwargs==prepared


def test_exact_conservative_budget_boundary():
    g,kw=compact();result=capture(g,kw,ThermoelasticSizeLimits(2,2))
    assert result['_size_admission']['assembly_entries_upper_bound']==72
    for args in ((1,1),(2,0),(2,sys.maxsize//36+1)):
        with pytest.raises(ValueError):ThermoelasticSizeLimits(*args)


@pytest.mark.parametrize('unsupported',[{'backend':'gpu'},{'density':1000},{'buckling':True}])
def test_new_entry_does_not_advertise_unsupported_backend_or_physics(unsupported):
    g,kw=compact()
    with patch.object(owner,'_network_preflight') as prepare:
        with pytest.raises(TypeError):solve_thermoelastic_network_snapshot(g,**kw,size_limits=limits(),**unsupported)
        prepare.assert_not_called()


def test_two_knob_constructor_and_read_only_derived_ceiling():
    import inspect
    from dataclasses import asdict, fields
    assert list(inspect.signature(ThermoelasticSizeLimits).parameters)==['max_nodes','max_edges']
    assert [field.name for field in fields(ThermoelasticSizeLimits)]==['max_nodes','max_edges']
    chosen=ThermoelasticSizeLimits(max_nodes=5000,max_edges=6000)
    assert asdict(chosen)=={'max_nodes':5000,'max_edges':6000}
    assert chosen.max_assembly_entries==216000
    with pytest.raises(FrozenInstanceError):chosen.max_assembly_entries=72
    with pytest.raises(AttributeError):object.__setattr__(chosen,'max_assembly_entries',72)
    with pytest.raises(TypeError):ThermoelasticSizeLimits(2,2,72)
    with pytest.raises(TypeError):ThermoelasticSizeLimits(max_nodes=2,max_edges=2,max_assembly_entries=72)


@pytest.mark.parametrize('name,upper',[('max_nodes',sys.maxsize//3),('max_edges',sys.maxsize//36)])
def test_exact_platform_overflow_boundaries(name,upper):
    args={'max_nodes':2,'max_edges':2};args[name]=upper
    chosen=ThermoelasticSizeLimits(**args)
    assert chosen.max_nodes*3<=sys.maxsize
    assert 36<=chosen.max_assembly_entries<=sys.maxsize
    args[name]=upper+1
    with pytest.raises(ValueError,match=name):ThermoelasticSizeLimits(**args)


@pytest.mark.parametrize('bad',[True,0,sys.maxsize//36+1])
def test_tampered_edge_ceiling_revalidated_before_preflight(bad):
    g,kw=compact();chosen=limits();object.__setattr__(chosen,'max_edges',bad)
    with patch.object(owner,'_network_preflight') as prepare:
        with pytest.raises(ValueError,match='max_edges'):capture(g,kw,chosen)
        prepare.assert_not_called()


def independent_identity_hash(result):
    identity_keys=('profile','node_ids','edge_ids','original_geometry','endpoints_by_id','positions_m','radii_m',
        'length_unit','E_Pa','stress_free_temperature_K','temperature_K','expansion_alpha_per_K',
        'constraint_mask','prescribed_values','loads','size_admission')
    identity={key:result[key] for key in identity_keys}
    encoded=json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_explicit_snapshot_roundtrip_admission_identity_and_applicability():
    g,kw=compact();kw['loads']=None;kw['temperature_K']={7:300.,91:300.}
    with patch.object(owner,'_assemble_snapshot_system',side_effect=dummy_assembly), \
         patch.object(owner,'_solve_constrained_sparse',return_value=np.zeros(6)), \
         patch.object(owner,'_network_applicability',wraps=owner._network_applicability) as applicability:
        first=solve_thermoelastic_network_snapshot(g,**kw,size_limits=ThermoelasticSizeLimits(2,2))
        second=solve_thermoelastic_network_snapshot(g,**kw,size_limits=ThermoelasticSizeLimits(2,3))
        assert applicability.call_count==2
    result=first.to_dict();other=second.to_dict()
    assert owner.ThermoelasticSnapshot(first.data_json).to_dict()==result
    assert json.loads(json.dumps(result,allow_nan=False))==result
    assert result['profile']=='thermoelastic_2d_network_bounded/1'
    assert result['size_admission']['contract']=='thermoelastic_network_size_limits/2'
    assert result['size_admission']['limits']=={'max_nodes':2,'max_edges':2}
    assert result['size_admission']['derived_limits']=={'max_assembly_entries':72,'assembly_entries_per_physical_edge':36}
    assert result['size_admission']['assembly_entries_upper_bound']==72
    assert other['size_admission']['derived_limits']['max_assembly_entries']==108
    assert result['input_sha256']==independent_identity_hash(result)
    assert other['input_sha256']==independent_identity_hash(other)
    assert result['input_sha256']!=other['input_sha256']
    assert result['applicability']==other['applicability']
    assert np.shape(result['moments'])==(2,2) and np.shape(result['edge_forces'])==(2,3)
    changed=copy.deepcopy(result);changed['size_admission']['derived_limits']['max_assembly_entries']+=36
    assert independent_identity_hash(changed)!=result['input_sha256']
    assert first.to_dict()==result


def test_bounded_profile_cannot_bypass_owner_applicability_rejection():
    g,kw=compact();kw['loads']=None;kw['temperature_K']={7:300.,91:300.}
    failure=FloatingPointError('injected ordinary network applicability failure')
    with patch.object(owner,'_assemble_snapshot_system',side_effect=dummy_assembly), \
         patch.object(owner,'_solve_constrained_sparse',return_value=np.zeros(6)), \
         patch.object(owner,'_network_applicability',side_effect=failure) as applicability, \
         patch.object(owner,'_accept_residual') as residual:
        with pytest.raises(FloatingPointError) as caught:
            solve_thermoelastic_network_snapshot(g,**kw,size_limits=limits())
        assert caught.value is failure and applicability.call_count==1
        residual.assert_not_called()


def test_saved_adapter_bounded_result_roundtrip():
    g,kw=compact();data=thermal_fixture(g);data['temperature_K']=[[300.,300.],[300.,300.]]
    frame=prepare_saved_temperature(g,data,frame_index=1,length_unit='m')
    control={k:v for k,v in kw.items() if k not in ('temperature_K','length_unit','loads')}
    with patch.object(owner,'_assemble_snapshot_system',side_effect=dummy_assembly), \
         patch.object(owner,'_solve_constrained_sparse',return_value=np.zeros(6)) as delegate:
        result=frame.solve(g,profile='network',size_limits=ThermoelasticSizeLimits(2,2),**control)
        assert delegate.call_count==1
    assert result==json.loads(json.dumps(result,allow_nan=False))
    assert result['schema']=='fibernet.saved_temperature_mechanics/1'
    assert result['mechanical']['input_sha256']==independent_identity_hash(result['mechanical'])
    assert result['temperature_source']['frame_index']==1
    assert result['feedback_to_electrical_or_thermal'] is False

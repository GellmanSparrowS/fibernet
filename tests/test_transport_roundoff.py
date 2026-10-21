"""Local representation contracts, not representative-network substitutes."""
import math
import struct
import sys
from fractions import Fraction
import numpy as np
import pytest
from fibernet import StructureGraph
from fibernet.sim import linear_transport as el, transient_thermal as th
from fibernet.sim._roundoff import upper_product, upper_sum, evaluation_ulp


def graph(edges, rename=None, reverse=False):
    rename=rename or {i:i for pair in edges for i in pair}
    nodes=sorted(rename)
    pairs=list(enumerate(edges))
    if reverse:pairs=pairs[::-1]
    return StructureGraph.from_dict({'dimension':3,'nodes':[{'id':rename[i],'position':[float(i),0,0]} for i in nodes[::-1]],
        'edges':[{'id':j,'node_i':rename[b if reverse else a],'node_j':rename[a if reverse else b]} for j,(a,b) in pairs]})


def test_upward_budget_operations_and_negative_ulp():
    # math.ulp(0) can itself become zero when a native runtime enables FTZ.
    tiny=struct.unpack('>d',bytes.fromhex('0000000000000001'))[0]
    gradual=tiny>0. and tiny+tiny>0.
    for a,b in [(.1,.2),(1e150,1e150)]:
        assert Fraction(upper_product(a,b))>=Fraction(a)*Fraction(b)
        assert Fraction(upper_sum([a,b]))>=Fraction(a)+Fraction(b)
    if gradual:
        assert upper_product(tiny,.5)>0
        for a,b in [(tiny,.5),(1e-300,1e-20)]:
            assert Fraction(upper_product(a,b))>=Fraction(a)*Fraction(b)
            assert Fraction(upper_sum([a,b]))>=Fraction(a)+Fraction(b)
        assert Fraction(evaluation_ulp(tiny))>=Fraction(tiny)
    else:
        for operation in [lambda: upper_product(tiny,.5),
                          lambda: upper_product(1e-300,1e-20),
                          lambda: upper_sum([tiny,tiny]),
                          lambda: evaluation_ulp(tiny),
                          lambda: evaluation_ulp(1e-300)]:
            with pytest.raises(ArithmeticError,match='^subnormal roundoff arithmetic is unavailable$'):
                operation()
    assert evaluation_ulp(-300.)==math.ulp(300.)
    assert upper_product(0.,1e308)==0 and upper_sum([])==0
    with pytest.raises(ArithmeticError):upper_product(1e308,1e308)


@pytest.mark.parametrize('value',[-1.,float('inf'),float('-inf'),float('nan'),
                                 struct.unpack('>d',bytes.fromhex('8000000000000001'))[0]])
def test_budget_rejects_invalid_values_before_arithmetic(value):
    with pytest.raises(ArithmeticError,match='^nonfinite roundoff budget$'):
        upper_product(value,1.)
    with pytest.raises(ArithmeticError,match='^nonfinite roundoff budget$'):
        upper_sum([value])


def test_budget_exact_signed_zero_and_normal_values():
    assert upper_product(-0.,1e308)==0.
    assert upper_sum([0.,-0.])==0.
    assert evaluation_ulp(-0.)==0.
    assert Fraction(upper_product(1e-150,1e-150))>=Fraction(1e-150)**2


@pytest.mark.parametrize('bits',['0000000000000001','8000000000000001',
                                '0000000000000000','8000000000000000'])
def test_subnormal_input_boundary_and_true_signed_zero(bits):
    value=struct.unpack('>d',bytes.fromhex(bits))[0]
    magnitude=int(bits,16)&0x7fffffffffffffff
    tiny=struct.unpack('>d',bytes.fromhex('0000000000000001'))[0]
    gradual=tiny>0. and tiny+tiny>0.
    g=graph([(0,1)])
    if magnitude and not gradual:
        with pytest.raises(ValueError,match='^conductance: nonzero subnormal input requires gradual underflow$'):
            el.solve_linear_transport(g,{0:value},{0:1.,1:0.},length_unit='m')
        with pytest.raises(ValueError,match='^nonzero subnormal physical input requires gradual underflow$'):
            th.solve_transient_thermal(g,{0:1.,1:1.},{0:value},{0:300.,1:310.},[0.,1.],
                                      fixed_temperature_K={0:300.,1:310.},length_unit='m')
    elif magnitude and bits.startswith('8'):
        with pytest.raises(ValueError,match='must be finite and nonnegative'):
            el._number(value,'conductance',True)
        with pytest.raises(ValueError,match='physical inputs must be finite nonnegative'):
            th._num(value)
    elif magnitude:
        assert struct.pack('>d',el._number(value,'conductance',True)).hex()==bits
        assert struct.pack('>d',th._num(value)).hex()==bits
    else:
        electrical=el.solve_linear_transport(g,{0:value},{0:1.,1:0.},length_unit='m')
        thermal=th.solve_transient_thermal(g,{0:1.,1:1.},{0:value},{0:300.,1:310.},[0.,1.],
                                          fixed_temperature_K={0:300.,1:310.},length_unit='m')
        assert electrical.status=='converged' and np.all(electrical.current_A==0.)
        assert thermal.completed and np.all(thermal.edge_heat_rate_W==0.)


def test_lost_nonzero_endpoint_difference_and_equal_endpoint_control():
    g=graph([(0,1)])
    a=sys.float_info.min
    b=math.nextafter(a,math.inf)
    tiny=struct.unpack('>d',bytes.fromhex('0000000000000001'))[0]
    gradual=tiny>0. and tiny+tiny>0.
    if not gradual:
        with pytest.raises(el.TransportSolveError,match='^nonzero voltage difference is not representable$'):
            el.solve_linear_transport(g,{0:1e308},{0:a,1:b},length_unit='m')
        with pytest.raises(th.ThermalSolveError,match='^nonzero temperature difference is not representable$') as err:
            th.solve_transient_thermal(g,{0:1.,1:1.},{0:1e308},{0:a,1:b},[0.,1.],
                                      fixed_temperature_K={0:a,1:b},length_unit='m')
        assert err.value.partial_result is None and err.value.attempted_step==0
    else:
        # In gradual mode the original strict power-underflow guard remains.
        with pytest.raises(el.TransportSolveError,match='^power underflow:'):
            el.solve_linear_transport(g,{0:1e308},{0:a,1:b},length_unit='m')
    electrical=el.solve_linear_transport(g,{0:1e308},{0:a,1:a},length_unit='m')
    thermal=th.solve_transient_thermal(g,{0:1.,1:1.},{0:1e308},{0:a,1:a},[0.,1.],
                                      fixed_temperature_K={0:a,1:a},length_unit='m')
    assert electrical.status=='converged' and np.all(electrical.current_A==0.)
    assert thermal.completed and np.all(thermal.edge_heat_rate_W==0.)


def test_stored_subnormal_solver_output_difference(monkeypatch):
    tiny=struct.unpack('>d',bytes.fromhex('0000000000000001'))[0]
    gradual=tiny>0. and tiny+tiny>0.
    # A native solver can return stored bits that DAZ compares equal to zero.
    monkeypatch.setattr(el,'spsolve',lambda *args,**kwargs: np.array([tiny]))
    reason='current underflow: nonzero driven current is not normally representable' if gradual else 'nonzero voltage difference is not representable'
    with pytest.raises(el.TransportSolveError,match='^'+reason+'$'):
        el.solve_linear_transport(graph([(0,1),(1,2)]),{0:1.,1:1.},{0:0.,2:1.},length_unit='m')


def test_true_signed_zero_endpoints_are_equal():
    g=graph([(0,1)])
    electrical=el.solve_linear_transport(g,{0:1.},{0:0.,1:-0.},length_unit='m')
    thermal=th.solve_transient_thermal(g,{0:1.,1:1.},{0:1.},{0:0.,1:-0.},[0.,1.],
                                      fixed_temperature_K={0:0.,1:-0.},length_unit='m')
    assert electrical.status=='converged' and np.all(electrical.current_A==0.)
    assert thermal.completed and np.all(thermal.edge_heat_rate_W==0.)


@pytest.mark.parametrize('offset',[-1e6,0.,.02,1e6,1e14])
@pytest.mark.parametrize('scale',[1e-150,1.,1e150])
def test_electrical_gauges_scales_and_resolved_bad_solver(monkeypatch,offset,scale):
    g=graph([(0,1),(1,2),(1,3)])
    base=el.spsolve
    # Existing component-power checks intentionally fail closed when the
    # absolute-coordinate solve loses the driven drop at extreme gauge/scale.
    expected_unresolved=(scale==1. and offset==1e14) or (scale==1e150 and offset in (1e6,1e14))
    if expected_unresolved:
        with pytest.raises(el.TransportSolveError,match='component-local'):
            el.solve_linear_transport(g,{0:scale,1:scale,2:scale},{0:offset+1,2:offset},length_unit='m')
    else:
        try:
            clean=el.solve_linear_transport(g,{0:scale,1:scale,2:scale},{0:offset+1,2:offset},length_unit='m')
        except el.TransportSolveError as exc:
            # These ill-resolved negative-gauge cases depend on the numeric
            # stack. Only the unchanged conservative component guard is valid.
            assert offset==-1e6 and scale in (1e-150,1e150)
            assert str(exc)=='component-local current conservation or power balance tolerance failed'
        else:
            assert clean.status=='converged'
            assert clean.provenance['achieved_limits']['roundoff_coordinates']=='absolute solved voltage'
    # Deliberately perturb above the stated engineering resolution at this
    # gauge. A fixed tiny perturbation is not resolvable at arbitrary offset.
    perturb=max(.01,512*np.finfo(float).eps*max(1,abs(offset)))
    def bad(*a,**kw):
        x=base(*a,**kw);x[-1]+=perturb;return x
    monkeypatch.setattr(el,'spsolve',bad)
    with pytest.raises(el.TransportSolveError):
        el.solve_linear_transport(g,{0:scale,1:scale,2:scale},{0:offset+1,2:offset},length_unit='m')


def test_electrical_components_ids_orientation():
    edges=[(0,1),(1,2),(1,3),(4,5)]
    ren={0:907,1:101,2:8001,3:27,4:9000,5:654}
    a=el.solve_linear_transport(graph(edges),{i:1. for i in range(4)},{0:1,2:0,4:3.14},length_unit='m')
    b=el.solve_linear_transport(graph(edges,ren,True),{i:1. for i in range(4)},{ren[0]:1,ren[2]:0,ren[4]:3.14},length_unit='m')
    va=dict(zip(a.node_ids,a.voltage_V));vb=dict(zip(b.node_ids,b.voltage_V))
    for i in va:assert va[i]==pytest.approx(vb[ren[i]],abs=1e-14)
    qa=dict(zip(a.edge_ids,a.current_A));qb=dict(zip(b.edge_ids,b.current_A))
    for i in qa:assert qa[i]==pytest.approx(-qb[i],abs=1e-14)


@pytest.mark.parametrize('offset',[0.,300.,1e6])
def test_thermal_deadend_components_and_bad_solver(monkeypatch,offset):
    g=graph([(0,1),(1,2),(1,3),(4,5)])
    C={i:1. for i in range(6)};G={i:10. for i in range(4)}
    T={i:offset+10 for i in range(6)};T[0]=offset+20
    bc={0:offset+20,4:offset+10}
    base=th.spsolve
    clean=th.solve_transient_thermal(g,C,G,T,[0,1,2],fixed_temperature_K=bc,length_unit='m')
    assert clean.completed
    def bad(*a,**kw):
        x=base(*a,**kw);x[0]+=.01;return x
    monkeypatch.setattr(th,'spsolve',bad)
    with pytest.raises(th.ThermalSolveError) as err:
        th.solve_transient_thermal(g,C,G,T,[0,1,2],fixed_temperature_K=bc,length_unit='m')
    assert not err.value.partial_result.completed
    assert err.value.partial_result.times_s.tolist()==[0.]


def test_thermal_all_fixed_component_and_zero_scale():
    g=graph([(0,1)])
    a=th.solve_transient_thermal(g,{0:1,1:1},{0:.13},{0:300,1:310},[0,1],fixed_temperature_K={0:300,1:310},length_unit='m')
    assert a.completed
    z=th.solve_transient_thermal(g,{0:1,1:1},{0:1},{0:300,1:300},[0,1],length_unit='m')
    assert z.completed and np.all(z.node_acceptance_bound_J[1]==0)


@pytest.mark.parametrize('scale',[1e-100,1.,1e100])
def test_thermal_scale_ids_orientation(scale):
    edges=[(0,1),(1,2),(1,3),(4,5)];ren={0:907,1:101,2:8001,3:27,4:9000,5:654}
    C={i:scale for i in range(6)};G={i:scale for i in range(4)}
    T={i:300. for i in range(6)};T[0]=310.
    a=th.solve_transient_thermal(graph(edges),C,G,T,[0,1,2],fixed_temperature_K={0:310.,4:300.},length_unit='m')
    b=th.solve_transient_thermal(graph(edges,ren,True),{ren[i]:v for i,v in C.items()},G,{ren[i]:v for i,v in T.items()},[0,1,2],fixed_temperature_K={ren[0]:310.,ren[4]:300.},length_unit='m')
    ar={int(k):i for i,k in enumerate(a.node_ids)};br={int(k):i for i,k in enumerate(b.node_ids)}
    for i in ar:np.testing.assert_allclose(a.temperature_K[:,ar[i]],b.temperature_K[:,br[ren[i]]],rtol=0,atol=1e-12)
    assert a.completed and b.completed


def test_thermal_unresolved_precision_cap_stays_closed():
    with pytest.raises(th.ThermalSolveError,match='precision budget'):
        th.solve_transient_thermal(graph([(0,1)]),{0:1.,1:1.},{0:1.},{0:1e12,1:1e12+10},[0,1],length_unit='m')

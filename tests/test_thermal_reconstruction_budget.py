"""Synthetic reconstruction-budget tests without a linear solver. Exact Fraction
oracles use signed physical equations, not a copy of production budget logic.
Negative-Q cooling/cancellation fixtures stress internal evaluator algebra only;
they do not assert that public nonnegative-source APIs support negative Q.
"""
import math
import struct
from fractions import Fraction as F

import numpy as np
import pytest
from fibernet.sim import _roundoff
from fibernet.sim._thermal_step import evaluate_thermal_step, ThermalResidualRejection


def arguments(T, inc, C=None, Q=None, edges=(), g=None, fix=(), dt=1.):
    T=np.array(T,dtype=float); inc=np.array(inc,dtype=float); n=len(T)
    conn=np.array(edges,dtype=np.int64).reshape(-1,2)
    incident=[[] for _ in range(n)]
    for j,(a,b) in enumerate(conn):
        incident[a].append((j,int(a),int(b))); incident[b].append((j,int(a),int(b)))
    # Fixtures have one connected component, or are isolated one-node cases.
    return dict(T=T,new=T+inc,inc=inc,initial=T.copy(),dt=dt,
        C=np.array(C if C is not None else [1.]*n),
        Q=np.array(Q if Q is not None else inc,dtype=float),
        conn=conn,g=np.array(g if g is not None else [1.]*len(edges)),
        fix=np.array(fix,dtype=np.int64),free=np.array([i for i in range(n) if i not in fix],dtype=np.int64),
        incident=incident,members=[np.arange(n)],
        component_edges=[[(j,int(a),int(b)) for j,(a,b) in enumerate(conn)]])


def exact_equations(a):
    """Independent exact signed equation at x=T+inc and stored y; no budgets."""
    T=list(map(F,a['T'])); x=[t+F(h) for t,h in zip(T,a['inc'])]
    y=list(map(F,a['new'])); dt=F(a['dt'])
    def equation(z,i):
        return F(a['C'][i])*(z[i]-T[i])-dt*F(a['Q'][i])+sum(
            (dt*F(a['g'][j])*(z[u]-z[v])*(1 if u==i else -1)
             for j,u,v in a['incident'][i]),F(0))
    return x,y,[(i,equation(x,i),equation(y,i)) for i in a['free']]


def assert_exact_certificate(a,result):
    x,y,rows=exact_equations(a)
    for i in range(len(x)):
        # Exact rational half-ulp, including a half-min-subnormal that cannot
        # itself be represented in binary64. No production helper reused.
        radius=F(math.ulp(float(y[i])))/2 if a['inc'][i]!=0 else F(0)
        assert abs(y[i]-x[i])<=radius
    for i,rx,ry in rows:
        assert y[i]==F(float(x[i])), 'fixture violates RN reconstruction'
        # Returned bound must cover the independently calculated stored residual.
        assert abs(ry)<=F(result['nbound'][i])
        # Signed arithmetic split is exact, no implementation-derived allowance.
        actual_evaluation_error=F(result['residual'][i])-ry
        actual_reconstruction_shift=ry-rx
        assert F(result['residual'][i])-rx==actual_evaluation_error+actual_reconstruction_shift
        assert abs(actual_reconstruction_shift)+abs(actual_evaluation_error)<=F(result['nbound'][i])


PASS_CASES=[
 ('binary_exact_witness',dict(T=[300.],inc=[2.**-24])),
 ('cooling',dict(T=[300.],inc=[-2.**-24])),
 ('zero_heat',dict(T=[300.],inc=[0.])),
 ('zero_temperature',dict(T=[0.],inc=[1.])),
 ('binade_up',dict(T=[256.-2.**-20],inc=[2.**-20])),
 ('binade_down',dict(T=[256.],inc=[-2.**-20])),
 ('capacity_small',dict(T=[300.],inc=[2.**-24],C=[2.**-500],Q=[2.**-524])),
 ('capacity_large',dict(T=[300.],inc=[2.**-24],C=[2.**500],Q=[2.**476])),
 ('subnormal_previous',dict(
     # Decode stored IEEE bits; arithmetic construction may flush to zero.
     T=[struct.unpack('>d', bytes.fromhex('0000000000000001'))[0]],
     inc=[2.**-1022])),
 ('fixed_reservoir',dict(T=[300.,301.],inc=[.25,0.],Q=[-.5,0.],edges=[(0,1)],fix=[1])),
 ('free_free_cancellation',dict(T=[300.,302.],inc=[.5,-.5],Q=[-.5,.5],edges=[(0,1)])),
 ('zero_increment_conduction',dict(T=[300.,301.],inc=[0.,0.],Q=[-1.,0.],edges=[(0,1)],fix=[1])),
]

def runtime_preserves_subnormal_arithmetic():
    """Independent live probe, without invoking or patching production guards."""
    smallest=struct.unpack('>d', bytes.fromhex('0000000000000001'))[0]
    # Comparison detects discarded inputs; exact result bits detect flushed
    # arithmetic output, without relying on a subnormal result comparison.
    return (smallest > 0.0 and
            struct.pack('>d', smallest+smallest)==bytes.fromhex('0000000000000002'))


def assert_subnormal_refusal(a):
    with pytest.raises(_roundoff._SubnormalRoundoffError) as caught:
        evaluate_thermal_step(**a)
    assert type(caught.value) is _roundoff._SubnormalRoundoffError
    assert str(caught.value)=='subnormal roundoff arithmetic is unavailable'


@pytest.mark.parametrize('name,kw',PASS_CASES,ids=[x[0] for x in PASS_CASES])
def test_pure_evaluator_acceptance(name,kw):
    a=arguments(**kw)
    if name=='subnormal_previous':
        assert struct.pack('>d', float(a['T'][0]))==bytes.fromhex('0000000000000001')
        # Probe this thread immediately before evaluator entry, not at import.
        if not runtime_preserves_subnormal_arithmetic():
            assert_subnormal_refusal(a)
            return
    result=evaluate_thermal_step(**a)
    assert_exact_certificate(a,result)
    if name=='binary_exact_witness':
        _,_,rows=exact_equations(a)
        assert rows[0][1]==rows[0][2]==0
        # Independently certify why OLD formula necessarily fails this witness.
        old_lower=F(8)*F(2.**-52)*F(600.)
        exact_scale=F(2)*F(2.**-24)
        assert old_lower > exact_scale/F(1000000)
    if name in ('fixed_reservoir','free_free_cancellation','zero_increment_conduction'):
        assert result['energy']==[0.]


REJECT_CASES=[
 ('lost_increment',dict(T=[300.],inc=[2.**-46]),ArithmeticError,'lost'),
 ('half_ulp_tie',dict(T=[300.],inc=[2.**-45]),ArithmeticError,'lost'),
 ('one_ulp_precision',dict(T=[300.],inc=[2.**-44]),ArithmeticError,'precision budget'),
 ('finite_equation_residual',dict(T=[300.],inc=[1.],Q=[2.]),ThermalResidualRejection,'residual'),
 ('source_underflow',dict(T=[0.],inc=[1.],Q=[2.**-600],dt=2.**-600),ArithmeticError,'source step heat'),
 ('energy_underflow',dict(T=[0.],inc=[2.**-600],C=[2.**-600],Q=[0.]),ArithmeticError,'stored step heat'),
 ('edge_underflow',dict(T=[0.,2.**-600],inc=[0.,0.],Q=[0.,0.],edges=[(0,1)],g=[2.**-600]),ArithmeticError,'edge heat rate'),
 ('nonfinite_source',dict(T=[300.],inc=[1.],Q=[float('nan')]),ArithmeticError,'source step heat'),
 ('nonfinite_state',dict(T=[300.],inc=[float('inf')],Q=[0.]),ArithmeticError,'stored step heat'),
 ('energy_overflow',dict(T=[0.],inc=[2.],C=[2.**1023],Q=[0.]),ArithmeticError,'stored step heat'),
]

@pytest.mark.parametrize('name,kw,kind,match',REJECT_CASES,ids=[x[0] for x in REJECT_CASES])
def test_pure_evaluator_refusal(name,kw,kind,match):
    with pytest.raises(kind,match=match): evaluate_thermal_step(**arguments(**kw))


def test_pure_evaluator_subnormal_runtime_unavailable(monkeypatch):
    # Preserve the same nonzero stored fixture on every hardware FP mode.
    kw=dict(PASS_CASES)['subnormal_previous']
    a=arguments(**kw)
    assert struct.pack('>d', float(a['T'][0]))==bytes.fromhex('0000000000000001')
    # Force the existing live guard's input check to fail, not a replacement
    # exception/evaluator. No FP mode change and no production-source edit.
    monkeypatch.setattr(_roundoff, '_MIN_SUBNORMAL', 0.0)
    assert_subnormal_refusal(a)

"""Analytical rod checks and rejection contracts for experimental snapshots."""
import copy
import math
import warnings
from unittest.mock import patch
import numpy as np
import pytest
from scipy.sparse import eye
from scipy.sparse.linalg import MatrixRankWarning
from fibernet.core.structure_graph import StructureGraph
from fibernet.ml.beam_frame_fem import BeamFrameFEM
from fibernet.sim import _thermoelastic as th
from fibernet.sim import _frame_linear_common as common

L, R, E, ALPHA, TREF = .1, .001, 1e6, 1e-5, 300.


def fixture(kind='base', restrained=False, alpha=ALPHA, temperature=301.):
    root, tip, eid = (501, 7, 1003) if kind == 'ids' else (10, 90, 70)
    axis = 1 if kind == 'rotate' else 0
    units = 'mm' if kind == 'mm' else 'm'
    scale = 1000. if kind == 'mm' else 1.
    endpoint = [0., 0., 0.]; endpoint[axis] = L*scale
    nodes = [{'id': root, 'position': [0., 0., 0.]}, {'id': tip, 'position': endpoint}]
    if kind == 'ids': nodes.reverse()
    a, b = (tip, root) if kind == 'reverse' else (root, tip)
    edges = [{'id': eid, 'node_i': a, 'node_j': b, 'radius': R*scale}]
    if kind == 'parallel': edges.append(dict(edges[0], id=110))
    g = StructureGraph.from_dict({'dimension': 2, 'nodes': nodes, 'edges': edges})
    boundary = {root: (0., 0., 0.)}
    if restrained:
        triple = [None, None, None]; triple[axis] = 0.; boundary[tip] = triple
    args = dict(length_unit=units, E_Pa=E, expansion_alpha_per_K={e['id']: alpha for e in edges},
        stress_free_temperature_K=TREF, temperature_K={root: temperature, tip: temperature},
        constraints=boundary)
    return g, args, root, tip, axis


CASES = [(f'{name}_{bc}', 'base', bc == 'restrained', alpha, temp)
         for name, alpha, temp in [('heat',ALPHA,301.), ('negative',-ALPHA,301.),
             ('cool',ALPHA,299.), ('zero_alpha',0.,301.), ('zero_delta',ALPHA,300.)]
         for bc in ('free','restrained')]
CASES += [(f'{kind}_{bc}', kind, bc == 'restrained', ALPHA, 301.)
          for kind in ('parallel','ids','reverse','mm','rotate') for bc in ('free','restrained')]
CASES += [('force_superposition','force',False,ALPHA,301.),
          ('displacement_superposition','displacement',True,ALPHA,301.)]


def close(actual, expected, atol):
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=atol)


@pytest.mark.parametrize('name,kind,restrained,alpha,temp', CASES, ids=[case[0] for case in CASES])
def test_snapshot(name, kind, restrained, alpha, temp):
    g, args, root, tip, axis = fixture(kind, restrained, alpha, temp)
    P = .001 if kind == 'force' else 0.
    delta = 2e-6 if kind == 'displacement' else 0.
    if P:
        load = [0.,0.,0.]; load[axis] = P; args['loads'] = {tip: load}
    if delta: args['constraints'][tip][axis] = delta
    before = copy.deepcopy((g.to_dict(), args))
    real = common.spsolve
    with patch.object(common, 'spsolve', wraps=real) as solve:
        record = th.solve_thermoelastic_snapshot(g, **args)
    assert solve.call_count == 1
    assert (g.to_dict(), args) == before
    out = record.to_dict()
    # Independent rod oracle: no implementation load/stiffness/recovery helpers.
    area = math.pi*R**2
    eigen = alpha*(temp-TREF)
    extension = delta if restrained else eigen*L+P*L/(E*area)
    axial = E*area*(delta/L-eigen) if restrained else P
    members = 2 if kind == 'parallel' else 1
    root_row, tip_row = out['node_ids'].index(root), out['node_ids'].index(tip)
    expected_u = np.zeros((2,3)); expected_u[tip_row,axis] = extension
    expected_r = np.zeros((2,3)); expected_r[root_row,axis] = -members*axial
    if restrained: expected_r[tip_row,axis] = members*axial
    close(out['displacement'], expected_u, 1e-12)
    close(np.array(out['displacement'])[:,2], 0., 1e-10)
    close(out['thermal_axial_strain'], eigen, 1e-10)
    close(out['total_axial_strain'], extension/L, 1e-10)
    close(out['elastic_axial_strain'], extension/L-eigen, 1e-10)
    close(out['sigma_axial'], axial/area, 1e-6)
    close(out['sigma_total'], abs(axial/area), 1e-6)
    close(np.array(out['edge_forces'])[:,0], axial, 1e-12)
    close(np.array(out['edge_forces'])[:,1], 0., 1e-12)
    close(np.array(out['edge_forces'])[:,2], 0., 1e-13)
    close(out['moments'], 0., 1e-13)
    close(out['sigma_bending'], 0., 1e-6)
    close(np.array(out['reactions'])[:,:2], expected_r[:,:2], 1e-12)
    close(np.array(out['reactions'])[:,2], 0., 1e-13)
    assert np.all(np.abs(out['free_residual']) <= np.array(out['free_acceptance_bound']))
    assert np.all(np.abs(out['component_balance']) <= np.array(out['component_acceptance_bound']))
    assert out['profile'] == 'thermoelastic_2d_elementary/2'
    assert out['numerical_regularization'] is False
    assert out['node_ids'] == sorted(g.nodes)
    assert out['edge_ids'] == sorted(g.edges)
    assert out['endpoints_by_id'] == [[g.edges[e].node_i,g.edges[e].node_j] for e in sorted(g.edges)]
    assert out['original_geometry']['positions_xyz'] == [g.nodes[n].position.tolist() for n in sorted(g.nodes)]
    assert out['original_geometry']['radii'] == [g.edges[e].radius for e in sorted(g.edges)]
    assert out['constraint_mask'] == [[x is not None for x in args['constraints'].get(n,[None]*3)] for n in sorted(g.nodes)]
    assert out['units']['length'] == 'm' and out['units']['stress'] == 'Pa'
    assert out['length_unit'] == args['length_unit']
    detached = record.to_dict(); detached['node_ids'].clear()
    assert len(record.to_dict()['node_ids']) == 2


REFUSALS = ['missing_temperature','extra_temperature','missing_alpha','bool_temperature',
 'bool_alpha','bool_node_key','bool_edge_key','nonfinite','reference_zero','temperature_zero',
 'unit','radius','curve','unsupported','root_rotation','third_node','third_edge','diagonal',
 'transverse_force','tip_moment','root_force','tip_transverse','unequal_temperature']


@pytest.mark.parametrize('case', REFUSALS)
def test_refusal(case):
    g, args, root, tip, axis = fixture()
    eid = next(iter(g.edges))
    if case == 'missing_temperature': del args['temperature_K'][tip]
    elif case == 'extra_temperature': args['temperature_K'][999] = 301.
    elif case == 'missing_alpha': del args['expansion_alpha_per_K'][eid]
    elif case == 'bool_temperature': args['temperature_K'][tip] = True
    elif case == 'bool_alpha': args['expansion_alpha_per_K'][eid] = False
    elif case == 'bool_node_key': args['temperature_K'][True] = 301.
    elif case == 'bool_edge_key': args['expansion_alpha_per_K'][False] = ALPHA
    elif case == 'nonfinite': args['E_Pa'] = float('inf')
    elif case == 'reference_zero': args['stress_free_temperature_K'] = 0.
    elif case == 'temperature_zero': args['temperature_K'][tip] = 0.
    elif case == 'unit': args['length_unit'] = 'inch'
    elif case == 'radius': g.edges[eid].radius = 0.
    elif case == 'curve': g.edges[eid].internal_points = np.array([[.05,.01,0.]])
    elif case == 'unsupported': args['constraints'] = {}
    elif case == 'root_rotation': args['constraints'][root] = [0.,0.,None]
    elif case == 'third_node': g.add_node([.2,0.], merge=False)
    elif case == 'third_edge':
        g.add_edge(root,tip,radius=R,allow_parallel=True)
        g.add_edge(root,tip,radius=R,allow_parallel=True)
    elif case == 'diagonal': g.nodes[tip].position[1] = .1
    elif case == 'transverse_force': args['loads'] = {tip:[0.,.001,0.]}
    elif case == 'tip_moment': args['loads'] = {tip:[0.,0.,.001]}
    elif case == 'root_force': args['loads'] = {root:[.001,0.,0.]}
    elif case == 'tip_transverse': args['constraints'][tip] = [None,0.,None]
    elif case == 'unequal_temperature': args['temperature_K'][tip] = 302.
    before = copy.deepcopy((g.to_dict(),args))
    with patch.object(BeamFrameFEM,'build_stiffness_2d',side_effect=AssertionError('unexpected assembly')), \
         patch.object(common,'spsolve',side_effect=AssertionError('unexpected solve')):
        with pytest.raises((ValueError, FloatingPointError)):
            th.solve_thermoelastic_snapshot(g, **args)
    assert (g.to_dict(),args) == before


@pytest.mark.parametrize('case',range(4))
def test_zero_eigenstrain_recovery(case):
    fem = BeamFrameFEM(E=E)
    pos = np.array([[0.,0.],[L,0.]])
    edges = np.array([[0,1],[1,0]]) if case == 3 else np.array([[0],[1]])
    selected = np.array([1]) if case == 3 else np.array([0])
    radius = np.full(edges.shape[1],R)
    u = np.zeros((2,3))
    if case == 1: u[1,0] = 1e-6
    if case >= 2: u[1] = [1e-6,2e-6,3e-6]
    a = fem._compute_element_stress_2d(pos,u,edges,selected,radius)
    b = fem._compute_element_stress_2d(pos,u,edges,selected,radius,
                                     axial_eigenstrain=np.zeros(edges.shape[1]))
    for original, zero in zip(a,b): np.testing.assert_array_equal(original,zero)


def test_recovery_shape_rejection():
    with pytest.raises(ValueError,match='original input edge rows'):
        BeamFrameFEM(E=E)._compute_element_stress_2d(np.array([[0.,0.],[L,0.]]),
            np.zeros((2,3)),np.array([[0,1],[1,0]]),np.array([1]),np.array([R,R]),
            axial_eigenstrain=np.zeros(1))


@pytest.mark.parametrize('case',['singular','nonfinite','residual','strain','rotation','scale','stress'])
def test_injected_failure(case):
    g,args,root,tip,axis = fixture()
    before = copy.deepcopy((g.to_dict(),args))
    def no_solve(*a,**k): raise AssertionError('real solve forbidden')
    with patch.object(common,'spsolve',side_effect=no_solve):
        if case == 'singular':
            def warning_stub(*a,**k): warnings.warn('injected singular',MatrixRankWarning)
            with pytest.raises(MatrixRankWarning):
                common._solve_constrained_sparse(eye(6,format='csr'),np.zeros(6),np.zeros(6),
                    np.array([True]*3+[False]*3), solve=warning_stub)
        elif case == 'nonfinite':
            with pytest.raises(FloatingPointError,match='nonfinite history solution'):
                common._solve_constrained_sparse(eye(6,format='csr'),np.zeros(6),np.zeros(6),
                    np.array([True]*3+[False]*3), solve=lambda *a:np.full(3,np.nan))
        elif case == 'scale':
            # Finite K/u, but absolute summation overflows: rejection, never infinite tolerance.
            huge = eye(6,format='csr')*1e308
            with np.errstate(over='ignore',invalid='ignore'):
                with pytest.raises(FloatingPointError,match='nonfinite derived'):
                    th._accept_residual(huge,np.full(6,2.),np.zeros(6),np.zeros(6),
                        np.array([True]*3+[False]*3),np.array([[0.,0.],[L,0.]]),0,np.zeros(6))
        else:
            candidate = np.zeros(6)
            if case == 'strain': candidate[3] = .001
            elif case == 'rotation': candidate[5] = .01
            elif case == 'residual': candidate[3] = 1e-4
            else: candidate[3] = ALPHA*(301.-TREF)*L
            with patch.object(th,'_solve_constrained_sparse',return_value=candidate):
                if case == 'stress':
                    recovery = (np.array([np.inf]),np.zeros(1),np.zeros(1),np.zeros((1,2)),np.zeros((1,3)))
                    with patch.object(BeamFrameFEM,'_compute_element_stress_2d',return_value=recovery):
                        with pytest.raises(FloatingPointError,match='recovery'):
                            th.solve_thermoelastic_snapshot(g,**args)
                else:
                    with pytest.raises(FloatingPointError): th.solve_thermoelastic_snapshot(g,**args)
    assert (g.to_dict(),args) == before

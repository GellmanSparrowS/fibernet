"""Independent energy assembly and API contracts for bounded welded frames."""
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from fibernet.core.structure_graph import StructureGraph
from fibernet.sim import _thermoelastic as th
from fibernet.sim import _frame_linear_common as common
from fibernet.ml.beam_frame_fem import BeamFrameFEM


def assert_components(actual, expected, atol):
    """Compare separately so NumPy's scalar tolerance formatter remains valid."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    if np.ndim(atol) == 0:
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=atol)
    else:
        assert actual.shape[-1] == len(atol)
        for component, tolerance in enumerate(atol):
            np.testing.assert_allclose(actual[...,component], expected[...,component],
                                       rtol=1e-8, atol=tolerance)


def fixture():
    record = json.loads((Path(__file__).parent / 'fixtures' /
                         'thermoelastic_welded_frame.json').read_text())
    args = record['arguments']
    for name in ('temperature_K', 'expansion_alpha_per_K', 'constraints', 'loads'):
        args[name] = {int(key): value for key, value in args[name].items()}
    return StructureGraph.from_dict(record['graph']), args


def energy_reference(graph, args):
    """Dense reference using integrated axial/curvature energy, no FEM helpers.

    Cubic transverse interpolation gives linear curvature; three-point Gauss
    integration is exact for its quadratic energy. Each native edge contributes
    separately, including physical parallel members. This is a linear reference,
    not a buckling or shear-deformation reference.
    """
    ids, eids = sorted(graph.nodes), sorted(graph.edges)
    scale = {'m': 1., 'mm': 1e-3, 'um': 1e-6}[args['length_unit']]
    xy = np.array([graph.nodes[n].position[:2] for n in ids]) * scale
    K = np.zeros((3*len(ids), 3*len(ids))); thermal = np.zeros(3*len(ids))
    data = []
    quadrature = [(0.5*(1.-np.sqrt(3./5.)), 5./18.), (.5, 4./9.),
                  (0.5*(1.+np.sqrt(3./5.)), 5./18.)]
    for eid in eids:
        edge = graph.edges[eid]; a, b = ids.index(edge.node_i), ids.index(edge.node_j)
        delta = xy[b]-xy[a]; length = np.linalg.norm(delta); c, s = delta/length
        radius = edge.radius*scale; area = np.pi*radius**2; inertia = np.pi*radius**4/4
        T = np.zeros((6, 6)); rotation = np.array([[c,s,0],[-s,c,0],[0,0,1]])
        T[:3,:3] = rotation; T[3:,3:] = rotation
        axial = np.array([-1.,0,0,1.,0,0])/length
        def curvature(t):
            return np.array([0,(-6+12*t)/length**2,(-4+6*t)/length,
                             0,(6-12*t)/length**2,(-2+6*t)/length])
        E = args['E_Pa']; local = E*area*length*np.outer(axial,axial)
        for t, weight in quadrature:
            B = curvature(t); local += E*inertia*length*weight*np.outer(B,B)
        eigen = args['expansion_alpha_per_K'][eid]*(
            .5*(args['temperature_K'][edge.node_i]+args['temperature_K'][edge.node_j])
            -args['stress_free_temperature_K'])
        rows = np.array([3*a,3*a+1,3*a+2,3*b,3*b+1,3*b+2])
        K[np.ix_(rows,rows)] += T.T@local@T
        thermal[rows] += T.T@(E*area*length*eigen*axial)
        data.append((rows,T,axial,curvature(0),curvature(1),length,area,inertia,eigen,radius,local))
    u = np.zeros(len(thermal)); mask = np.zeros(len(u),dtype=bool); force = np.zeros(len(u))
    for n, values in args['constraints'].items():
        for component, value in enumerate(values):
            if value is not None: mask[3*ids.index(n)+component] = True; u[3*ids.index(n)+component] = value
    for n, values in args['loads'].items(): force[3*ids.index(n):3*ids.index(n)+3] = values
    free = np.flatnonzero(~mask); fixed = np.flatnonzero(mask)
    u[free] = np.linalg.solve(K[np.ix_(free,free)],
                            (force+thermal)[free]-K[np.ix_(free,fixed)]@u[fixed])
    strain=[]; stress=[]; moments=[]; forces=[]; bending=[]; endpoint_gradient=[]
    for rows,T,axial,b0,b1,length,area,inertia,eigen,radius,local_stiffness in data:
        local_u=T@u[rows]; total=axial@local_u; sigma=args['E_Pa']*(total-eigen)
        m=-args['E_Pa']*inertia*np.array([b0@local_u,b1@local_u])
        strain.append(total); stress.append(sigma); moments.append(m)
        forces.append([sigma*area,(m[1]-m[0])/length,.5*(m[0]+m[1])])
        bending.append(np.max(np.abs(m))*radius/inertia)
        endpoint_gradient.append(local_stiffness@local_u-args['E_Pa']*area*length*eigen*axial)
    return dict(displacement=u.reshape(-1,3), reactions=(K@u-thermal-force).reshape(-1,3),
                total_axial_strain=strain, sigma_axial=stress, moments=moments,
                edge_forces=forces, sigma_bending=bending, endpoint_gradient=endpoint_gradient)


@pytest.mark.parametrize('variant', ['nonuniform','uniform','cooling','zero_alpha',
                                    'reference_temperature','millimetres','reverse_endpoints'])
def test_network_energy_reference(variant):
    graph,args=fixture()
    if variant in ('uniform','cooling','reference_temperature'):
        temperature={'uniform':301.,'cooling':299.,'reference_temperature':300.}[variant]
        args['temperature_K']={n:temperature for n in graph.nodes}
    if variant=='zero_alpha': args['expansion_alpha_per_K']={e:0. for e in graph.edges}
    if variant=='millimetres':
        args['length_unit']='mm'
        for node in graph.nodes.values(): node.position *= 1000.
        for edge in graph.edges.values():
            edge.radius *= 1000.
            if edge.internal_points is not None: edge.internal_points *= 1000.
    if variant=='reverse_endpoints':
        for edge in graph.edges.values(): edge.node_i,edge.node_j=edge.node_j,edge.node_i
    before=copy.deepcopy((graph.to_dict(),args))
    with patch.object(common,'spsolve',wraps=common.spsolve) as sparse_solve, \
         patch.object(np.linalg,'solve',wraps=np.linalg.solve) as dense_solve:
        expected=energy_reference(graph,args)
        record=th.solve_thermoelastic_network_snapshot(graph,**args)
    assert sparse_solve.call_count==1
    assert dense_solve.call_count==1
    result=record.to_dict()
    assert (graph.to_dict(),args)==before
    for field,atol in [('displacement',[1e-12,1e-12,1e-10]),('reactions',[1e-12,1e-12,1e-13]),
                       ('total_axial_strain',1e-10),('sigma_axial',1e-6),
                       ('moments',1e-13),('edge_forces',[1e-12,1e-12,1e-13]),('sigma_bending',1e-6)]:
        assert_components(result[field],expected[field],atol)
    gradient=[]
    for (N,V,_),(Mi,Mj) in zip(result['edge_forces'],result['moments']):
        gradient.append([-N,-V,Mi,N,V,-Mj])
    assert_components(gradient,expected['endpoint_gradient'],
                      [1e-12,1e-12,1e-13,1e-12,1e-12,1e-13])
    assert result['profile']=='thermoelastic_2d_network/1'
    assert result['applicability']['stability_qualified'] is False
    assert result['numerical_regularization'] is False
    detached=record.to_dict(); detached['node_ids'].clear()
    assert len(record.to_dict()['node_ids'])==29
    identity_keys=('profile','node_ids','edge_ids','original_geometry','endpoints_by_id',
                   'positions_m','radii_m','length_unit','E_Pa','stress_free_temperature_K',
                   'temperature_K','expansion_alpha_per_K','constraint_mask',
                   'prescribed_values','loads')
    identity={key:result[key] for key in identity_keys}
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()
    assert result['input_sha256']==digest


@pytest.mark.parametrize('fault',['missing_temperature','missing_alpha','disconnected',
                                 'unsupported','too_hot','invalid_unit'])
def test_network_rejects_out_of_scope(fault):
    graph,args=fixture()
    if fault=='missing_temperature': args['temperature_K'].pop(next(iter(graph.nodes)))
    if fault=='missing_alpha': args['expansion_alpha_per_K'].pop(next(iter(graph.edges)))
    if fault=='disconnected':
        node=graph.add_node([.3,.3],merge=False); args['temperature_K'][node]=300.
    if fault=='unsupported': args['constraints']={}
    if fault=='too_hot': args['temperature_K']={n:400. for n in graph.nodes}
    if fault=='invalid_unit': args['length_unit']='inch'
    before=copy.deepcopy((graph.to_dict(),args))
    with patch.object(BeamFrameFEM,'build_stiffness_2d',side_effect=AssertionError('unexpected assembly')), \
         patch.object(common,'spsolve',side_effect=AssertionError('unexpected sparse solve')), \
         patch.object(np.linalg,'solve',side_effect=AssertionError('unexpected dense solve')):
        with pytest.raises((ValueError,FloatingPointError)):
            th.solve_thermoelastic_network_snapshot(graph,**args)
    assert (graph.to_dict(),args)==before

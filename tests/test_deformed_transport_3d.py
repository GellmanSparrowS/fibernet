"""Check explicit XYZ handoff against tilted analytical and complete 3D cases."""
import json
import math

import numpy as np
import pytest

import fibernet as fn
from fibernet.sim.deformed_transport import (
    prepare_deformed_transport,
    prepare_deformed_transport_3d,
)
from fibernet.sim.linear_transport import solve_linear_transport


def specimen(length_unit):
    scale = {'m': 1., 'mm': 1e-3, 'um': 1e-6}[length_unit]
    return fn.StructureGraph.from_dict({
        'version': '3.1', 'dimension': 3,
        'metadata': {'length_unit': length_unit, 'coordinate_frame': 'lab'},
        'nodes': [
            {'id': nid, 'position': [v * .001 / scale for v in xyz]}
            for nid, xyz in [(19, [0, 0, 0]), (4, [3, 0, 4]), (73, [3, 12, 4])]
        ],
        'edges': [{'id': eid, 'node_i': a, 'node_j': b}
                  for eid, a, b in [(81, 19, 4), (7, 4, 19), (23, 4, 73)]],
    })


@pytest.mark.parametrize('unit', ['m', 'mm', 'um'])
def test_3d_tilt_sparse_parallel_ids_and_chord_readout(unit):
    s = specimen(unit)
    before = s.to_dict()
    u = {19: [0., 0., 0.], 4: [0., 0., .004], 73: np.array([0., 0., .004])}
    g0 = {81: 2., 7: 4., 23: 6.}
    moved, cond, report = prepare_deformed_transport_3d(
        s, translations_m_by_id=u, reference_conductance_S=g0, length_unit=unit,
    )
    equiv_parallel = 6 * 5 / math.sqrt(73)
    expect = equiv_parallel * 6 / (equiv_parallel + 6)
    assert s.to_dict() == before and s is not moved
    assert list(moved.nodes) == list(s.nodes) and list(cond) == list(s.edges)
    assert report['coordinate_frame'] == 'lab'
    assert report['schema'] == 'fibernet.deformed_transport_preparation_3d/1'
    assert report['geometry_dimension'] == 3
    assert report['translation_frame'] == 'source_graph_xyz_axes'
    assert report['endpoints_by_id'] == [[4, 19], [4, 73], [19, 4]]
    np.testing.assert_allclose(report['derived_lengths_m'],
                               [.001 * math.sqrt(73), .012, .001 * math.sqrt(73)], rtol=2e-15)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    result = solve_linear_transport(moved, cond, {19: 1., 73: 0.}, length_unit=unit)
    assert result.dissipated_power_W == pytest.approx(expect, rel=2e-14)
    assert result.current_A[0] > 0 and result.current_A[1] < 0


def test_keep_planar_entry_and_reject_incomplete_xyz():
    s = specimen('mm')
    ref = {81: 2., 7: 4., 23: 6.}
    xy = {i: [0., 0.] for i in s.nodes}
    before = s.to_dict()
    with pytest.raises(ValueError, match='dimension=2'):
        prepare_deformed_transport(s, translations_m_by_id=xy,
                                   reference_conductance_S=ref, length_unit='mm')
    with pytest.raises(ValueError, match='ux_m, uy_m, uz_m'):
        prepare_deformed_transport_3d(s, translations_m_by_id=xy,
                                      reference_conductance_S=ref, length_unit='mm')
    assert s.to_dict() == before


def test_3d_accepts_generator_bool_flags_but_not_numeric_surrogates():
    s = specimen('mm')
    n = s.nodes[19]
    n.boundary = tuple(np.bool_(i % 2) for i in range(6))
    args = dict(translations_m_by_id={i: [0., 0., 0.] for i in s.nodes},
                reference_conductance_S={81: 2., 7: 4., 23: 6.},
                length_unit='mm')
    g, _, _ = prepare_deformed_transport_3d(s, **args)
    assert s.to_dict() == g.to_dict()
    assert all(type(v) is np.bool_ for v in n.boundary)
    assert all(type(v) is np.bool_ for v in g.nodes[19].boundary)
    n.boundary = (1, False, True, False, True, False)
    with pytest.raises(ValueError, match='boundary flags must be six booleans'):
        prepare_deformed_transport_3d(s, **args)


def test_3d_unchanged_chord_keeps_conductance_bits_and_rejects_collapsed_edge():
    g = fn.StructureGraph(dimension=3)
    a = g.add_node([0., 0., 0.])
    b = g.add_node([3., 0., 4.])
    e = g.add_edge(a, b)
    s = {e: math.nextafter(1., math.inf)}
    args = dict(reference_conductance_S=s, length_unit='m')
    v, c, report = prepare_deformed_transport_3d(g, translations_m_by_id={a: [10., 2., -1.], b: [10., 2., -1.]}, **args)
    assert c == s and report['derived_lengths_m'] == report['reference_lengths_m'] == [5.]
    with pytest.raises(ValueError, match='derived chord'):
        prepare_deformed_transport_3d(g, translations_m_by_id={a: [3., 0., 4.], b: [0., 0., 0.]}, **args)
    assert v.nodes[a].position.tolist() == [10., 2., -1.]
    assert g.nodes[a].position.tolist() == [0., 0., 0.]


@pytest.mark.parametrize('bad', [
    [False, 0., 0.], [0., np.inf, 0.], [0., 0., float('nan')],
    [0., 0., 2**53 + 1], np.zeros((1, 3)), [0., 0., 0., 0.],
])
def test_3d_rejects_invalid_xyz_components_without_mutating_graph(bad):
    graph = specimen('mm')
    before = graph.to_dict()
    u = {i: [0., 0., 0.] for i in graph.nodes}
    u[4] = bad
    with pytest.raises(ValueError, match='translation'):
        prepare_deformed_transport_3d(
            graph, translations_m_by_id=u,
            reference_conductance_S={81: 2., 7: 4., 23: 6.}, length_unit='mm',
        )
    assert graph.to_dict() == before


def test_3d_preserves_existing_curve_rejection():
    graph = specimen('mm')
    graph.edges[81].internal_points = np.array([[1., 0., 2.]])
    before = graph.to_dict()
    with pytest.raises(ValueError, match='straight edges only'):
        prepare_deformed_transport_3d(
            graph, translations_m_by_id={i: [0.,0.,0.] for i in graph.nodes},
            reference_conductance_S={81: 2., 7: 4., 23: 6.}, length_unit='mm',
        )
    assert graph.to_dict() == before


def test_complete_3d_bcc_force_to_electrical_dense_oracle():
    from examples.mechanoelectric_network_3d import build_sample

    g,nids,eids,x,ends,grips,sigma,radius=build_sample()
    assert (len(nids),len(eids)) == (91,360)
    s=fn.simulate(g, backend='fem',nonlinear=False,material_mode='graph',
                  length_unit='m',stretch_ratio=1.005,pct=16.5/len(nids))
    deformed=dict(zip(s.metadata['node_ids'],s.deformed_positions))
    u={nid: deformed[nid]-row for nid,row in zip(nids,x)}
    a,b=ends.T; y=np.array([deformed[i] for i in nids])
    assert np.max(np.abs(y[:,2]-x[:,2]))>5e-6
    old=np.linalg.norm(x[a]-x[b],axis=1)
    new=np.linalg.norm(y[a]-y[b],axis=1)
    g0=sigma*np.pi*radius**2/old
    moved, cond, report=prepare_deformed_transport_3d(
        g,translations_m_by_id=u,reference_conductance_S=dict(zip(eids,g0)),length_unit='m',
    )
    observed=np.array([moved.nodes[i].position for i in nids]); np.testing.assert_allclose(observed,y,rtol=2e-15,atol=1e-18)
    independent=sigma*np.pi*radius**2/new
    np.testing.assert_allclose([cond[i] for i in eids],independent,rtol=2e-15)
    assert report['solver_executed'] is False and report['mechanical_result_admitted'] is False
    B=np.zeros((len(eids),len(nids))); ii=np.arange(len(eids)); B[ii,a]=1.; B[ii,b]=-1.
    Y=(B.T*independent)@B
    for pins in grips:
        ids={n:i for i,n in enumerate(nids)}
        f=np.array([ids[n] for n in pins]); q=np.setdiff1d(np.arange(len(nids)),f)
        v=np.zeros(len(nids));v[f]=[pins[n] for n in pins]
        v[q]=np.linalg.solve(Y[np.ix_(q,q)],-Y[np.ix_(q,f)]@v[f])
        independent_current=independent*(B@v)
        result=solve_linear_transport(moved,cond,pins,length_unit='m')
        r={int(ident):i for i,ident in enumerate(result.edge_ids)}
        np.testing.assert_allclose(result.current_A[[r[i] for i in eids]],independent_current,rtol=5e-10,atol=3e-17)
        assert result.dissipated_power_W == pytest.approx(float(np.dot(independent_current,B@v)),rel=5e-13)

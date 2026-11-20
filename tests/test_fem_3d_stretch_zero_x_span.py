"""A nonunit x stretch must not silently become an undeformed 3-D result."""
import numpy as np
import pytest

import fibernet as fn
from fibernet.ml.beam_frame_fem import BeamFrameFEM


def pinned_graph(vertical):
    return fn.StructureGraph.from_dict({'dimension':3,'nodes':[
        {'id':149,'position':[0.,0.,.01] if vertical else [.01,0.,0.]},
        {'id':11,'position':[0.,0.,0.]}],
        'edges':[{'id':81,'node_i':11,'node_j':149,'radius':.0002,
                    'material':fn.Material(youngs_modulus=1e9,poissons_ratio=.3).to_dict()}]})


@pytest.mark.parametrize('ratio',[.99,1.01])
@pytest.mark.parametrize('material_mode',['graph','uniform'])
def test_nonunit_3d_x_stretch_without_x_span_refuses_before_output(tmp_path, ratio, material_mode):
    g=pinned_graph(vertical=True)
    before=g.to_dict()
    kwargs={'material_mode':'graph','length_unit':'m'} if material_mode=='graph' else {}
    with pytest.raises(ValueError,match='nonzero x span'):
        fn.simulate(g,backend='fem',stretch_ratio=ratio,nonlinear=False,
                    save_path=tmp_path/'result.json',**kwargs)
    assert not list(tmp_path.iterdir())
    assert g.to_dict()==before


def test_nonzero_span_matches_series_beam_and_preserves_stable_ids():
    g=pinned_graph(vertical=False)
    r=fn.simulate(g,backend='fem',stretch_ratio=1.01,nonlinear=False,
                  material_mode='graph',length_unit='m')
    assert r.metadata['node_ids']==[149,11] and r.metadata['edge_ids']==[81]
    np.testing.assert_allclose(r.edge_forces,[1e9*np.pi*.0002**2*.01])
    np.testing.assert_allclose(r.edge_stretches,[1.01])


def test_unit_ratio_and_manual_z_load_keep_valid_zero_x_span_uses():
    g=pinned_graph(vertical=True)
    r=fn.simulate(g,backend='fem',stretch_ratio=1.0,nonlinear=False,
                  material_mode='graph',length_unit='m')
    np.testing.assert_array_equal(r.displacements,np.zeros((2,3)))
    # Unit ratio only describes the legacy neutral state. Actually stretching
    # a vertical member requires explicit controls, not the x convenience API.
    p,e,radii=g.to_numpy()
    f=np.zeros_like(p); f[1,2]=1
    raw=BeamFrameFEM(E=1e9).solve_3d(e.T,p,radii,forces=f,fixed_nodes=[0],damping=0.)
    np.testing.assert_allclose(raw['edge_forces'][:,0],[1.])
    np.testing.assert_allclose(raw['u'][1,2],.01/(1e9*np.pi*.0002**2))


def test_tiny_positive_x_span_does_not_trigger_zero_span_guard():
    g=pinned_graph(vertical=True)
    # Distinct spatial positions remain 1 cm apart. Only the projected x
    # span is tiny; no absolute geometric tolerance should be introduced.
    g.set_node_position(149, [1e-15,0.,.01])
    r=fn.simulate(g,backend='fem',stretch_ratio=1.01,nonlinear=False,
                  material_mode='graph',length_unit='m')
    actual=r.displacements[r.metadata['node_ids'].index(149),0]
    np.testing.assert_allclose(actual,1e-15*(1.01-1.),rtol=1e-12,atol=0.)


@pytest.mark.parametrize('invalid',[np.nan,np.inf,-np.inf])
def test_existing_finite_input_boundaries_remain_unchanged(invalid):
    g=pinned_graph(vertical=True)
    with pytest.raises(ValueError,match='finite and positive|finite positive'):
        fn.simulate(g,backend='fem',stretch_ratio=invalid,nonlinear=False)
    with pytest.raises(ValueError,match='finite'):
        g.add_node([invalid,0.,0.])

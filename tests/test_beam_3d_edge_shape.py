"""A 3-D beam solve never treats extra connectivity rows as invisible nodes."""
import numpy as np
import pytest

from fibernet import Material
from fibernet.ml.beam_frame_fem import BeamFrameFEM


@pytest.mark.parametrize('deduplicate', [False, True])
def test_tet4_connectivity_rejected_before_assembly(monkeypatch, deduplicate):
    nodes = np.array([[0., 0., 0.], [.001, 0., 0.], [0., .001, 0.], [0., 0., .001]])
    tetrahedron = np.array([[0, 1, 2, 3]], dtype=np.int64)
    # Positive physical volume; not an intentionally malformed solid cell.
    assert np.linalg.det(nodes[tetrahedron[0, 1:]] - nodes[0]) > 0
    solver = BeamFrameFEM()
    with pytest.raises(ValueError, match=r'two-endpoint beam edges.*tetrahedral connectivity'):
        solver.solve_3d(tetrahedron.T, nodes, np.array([.0001]),
                        fixed_nodes=[0], deduplicate=deduplicate)

    def no_assembly(*args, **kwargs):
        raise AssertionError("invalid connectivity reached material/assembly processing")
    monkeypatch.setattr(solver, '_snapshot_edge_materials', no_assembly)
    with pytest.raises(ValueError, match=r'two-endpoint beam edges.*tetrahedral connectivity'):
        solver.solve_3d(tetrahedron.T, nodes, np.array([.0001]),
                        fixed_nodes=[0], deduplicate=deduplicate)


def test_two_endpoint_3d_series_beams_with_two_materials():
    x = np.array([[0., 0., 0.], [.005, 0., 0.], [.011, 0., 0.]])
    conn = np.array([[0, 1], [1, 2]], dtype=np.int64).T
    r = np.full(2, .00012)
    area = np.pi*r[0]**2
    E = [1e9, 2.4e9]
    F = .001
    force = np.array([[0.,0.,0.],[0.,0.,0.],[F,0.,0.]])
    result = BeamFrameFEM().solve_3d(conn, x, r, forces=force,
        fixed_nodes=[0], damping=0,
        edge_materials=[Material(youngs_modulus=e,poissons_ratio=.25) for e in E])
    du = F*np.diff(x[:,0])/(np.asarray(E)*area)
    np.testing.assert_allclose(result['u'][:,0], np.r_[0.,np.cumsum(du)], atol=1e-16)
    np.testing.assert_allclose(result['edge_forces'][:,0], F, atol=1e-15)
    assert result['reactions'][0,0] == pytest.approx(-F)


def test_empty_two_row_connectivity_keeps_existing_shape_contract():
    result = BeamFrameFEM().solve_3d(np.empty((2,0),dtype=np.int64),
        np.array([[0.,0.,0.],[.005,0.,0.]]),np.empty(0),fixed_nodes=[0])
    assert result['n_edges']==0 and result['edge_forces'].shape==(0,3)
    assert result['K'].shape==(12,12) and result['K'].nnz==0
    np.testing.assert_array_equal(result['u'],np.zeros((2,6)))

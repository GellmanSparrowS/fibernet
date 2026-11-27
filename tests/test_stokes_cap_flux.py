"""Independent exact cap integrals and topology for a quadratic velocity field."""

import numpy as np
import pytest

from fibernet.analysis.stokes_cap_flux import _checked_affine_xyz, _side


def _touching_tets():
    points = [(0,0,0),(0,1,0),(0,0,1),(1,.1,.1),(0,-1,0),(0,0,-1),(1,-.1,-.1)]
    edges = ((0,1),(1,2),(2,0),(0,3),(1,3),(2,3))
    cells = []
    for ids in ((0,1,2,3),(0,4,5,6)):
        row = list(ids)
        for i,j in edges:
            row.append(len(points))
            points.append(tuple((a+b)/2 for a,b in zip(points[ids[i]], points[ids[j]])))
        cells.append(row)
    return np.asarray(points), np.asarray(cells)


@pytest.mark.parametrize("sign", [-1,1])
def test_quadratic_flux_and_full_edge_opening_connectivity(sign):
    p,t = _touching_tets()
    _,y,z = p.T
    u = np.column_stack((sign*(2+y*y+2*y*z+3*z*z), np.zeros_like(y), np.zeros_like(y)))
    report = _side(_checked_affine_xyz(p,t),t,u,False)
    rows = report["openings"]
    assert len(rows)==2 and [x["triangles"] for x in rows]==[1,1]
    np.testing.assert_allclose([x["area_m2"] for x in rows], [.5,.5])
    # The analytic integral of 2+y²+2yz+3z² on a unit right triangle.
    np.testing.assert_allclose([x["outward_flow_m3_s"] for x in rows], [-sign*17/12]*2)
    np.testing.assert_allclose(report["outward_flow_m3_s"], -sign*17/6)


def test_rejects_curved_quadratic_cells():
    p,t = _touching_tets()
    p[t[0,4],1] += .01
    with pytest.raises(ValueError, match="affine"):
        _checked_affine_xyz(p,t)

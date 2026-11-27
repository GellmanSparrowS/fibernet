"""Report signed flux by opening directly from a saved FiberNet P2 VTU."""
import argparse
import json
from pathlib import Path
import numpy as np

from fibernet.analysis._triangle_components import edge_connected_triangles
from fibernet.io._tet4_stokes_velocity_vtu import read_own_quadratic_vtu

_FACE = np.array(((0,1,2),(0,1,3),(0,2,3),(1,2,3)))
_MID = np.array(((4,5,6),(4,7,8),(6,7,9),(5,8,9)))
_PAIRS = np.array(((0,1),(1,2),(2,0),(0,3),(1,3),(2,3)))


def _checked_affine_xyz(points, tet):
    q=points.astype(np.float64,copy=False)
    span=np.ptp(q,axis=0)
    scale=float(span.max())
    if not np.isfinite(span).all() or min(span)<=0:
        raise ValueError("Expected a nonzero 3D extent")
    normalized=(q-q.min(axis=0))/scale
    corners=normalized[tet[:,:4]]
    det=np.linalg.det(corners[:,1:]-corners[:,:1])
    if not np.all(det>0):
        raise ValueError("Nonpositive orientation in saved tetrahedra")
    tol=max(2e-12,8*np.finfo(points.dtype).eps)
    expected=corners[:,_PAIRS].mean(axis=2)
    if not np.allclose(normalized[tet[:,4:]],expected,rtol=0,atol=tol):
        raise ValueError("The tetra10 cells must be affine with edge midpoints")
    return q


def _cap_faces(p, t, high):
    plane = float(p[:, 0].max() if high else p[:, 0].min())
    exterior = (p[:, 0] == plane)[t[:, :4]]
    owning, local = np.where(exterior[:, _FACE].all(axis=2))
    if not len(owning):
        raise ValueError("Expected nonempty planar extreme-x caps")
    faces = t[owning[:, None], _FACE[local]]
    if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
        raise ValueError("Repeated extreme-x cap faces are unsupported")
    return plane, owning, local, faces


def _side(p,t,u,high):
    plane, owning, local, faces = _cap_faces(p, t, high)
    edge_mid=t[owning[:,None],_MID[local]]
    a,b,c=p[faces].transpose(1,0,2)
    d,e=b-a,c-a
    area=.5*np.abs(d[:,1]*e[:,2]-d[:,2]*e[:,1])
    speed=u[edge_mid,0].mean(axis=1,dtype=np.float64)
    # On each affine triangle, degree-2 shape functions integrate to
    # zero at vertices and one-third of the area at edge midpoints.
    flux=(1 if high else -1)*area*speed
    if np.any(area<=0) or not np.isfinite(area).all() or not np.isfinite(flux).all():
        raise ValueError("Invalid cap triangle geometry or flux")
    num,group=edge_connected_triangles(faces)
    weights=np.bincount(group,weights=area,minlength=num)
    centers=a+d/3+e/3
    cy=np.bincount(group,weights=area*centers[:,1],minlength=num)/weights
    cz=np.bincount(group,weights=area*centers[:,2],minlength=num)/weights
    totals=np.bincount(group,weights=flux,minlength=num)
    counts=np.bincount(group,minlength=num)
    total=float(np.sum(flux))
    if not np.isfinite(total) or not all(np.isfinite(q).all() for q in (weights,cy,cz,totals)):
        raise ValueError("Unrepresentable cap integral or centroid")
    return {"plane_x_m":plane,"outward_flow_m3_s":total,
        "openings":[{"centroid_yz_m":[float(cy[i]),float(cz[i])],"area_m2":float(weights[i]),
                     "outward_flow_m3_s":float(totals[i]),"triangles":int(counts[i])}
                    for i in np.lexsort((cz,cy))]}


def audit_saved_stokes(path):
    """Integrate signed SI flux through both extreme-x caps of a native P2 VTU.

    Each opening groups boundary triangles sharing a full corner edge. Flow
    is positive outward, independent of pressure-gradient direction. The
    imbalance compares the two cap integrals; it is not a volumetric
    divergence or wall-leakage check. The bounded FiberNet VTU reader does
    not accept arbitrary mesh exports or curved tetra10 cells.
    """
    points,tet,velocity=read_own_quadratic_vtu(path)
    xyz=_checked_affine_xyz(points,tet)
    low,high=_side(xyz,tet,velocity,False),_side(xyz,tet,velocity,True)
    a=float(low["outward_flow_m3_s"]); b=float(high["outward_flow_m3_s"])
    den=max(abs(a),abs(b))
    if not np.isfinite(a+b):
        raise ValueError("Unrepresentable net outward flux")
    return {"schema":"fibernet-p2-cap-flux-1","cells":int(len(tet)),"points":int(len(points)),
            "x_min":low,"x_max":high,"net_outward_flow_m3_s":a+b,
            "relative_cap_imbalance":abs(a+b)/den if den else 0.,
            "scope":"Signed flux on the two planar extreme-x caps of a saved affine tetra10 field."}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file",type=Path,help="locally exported FiberNet quadratic velocity VTU")
    args=p.parse_args(argv)
    try:
        info=audit_saved_stokes(args.file)
    except (OSError,ValueError) as exc:
        p.error(str(exc))
    print(json.dumps(info,allow_nan=False,indent=2))


if __name__=="__main__":
    main()

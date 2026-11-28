"""Separate throughput and velocity-profile change on identical saved P2 caps."""
import argparse
import json
from math import factorial, fsum, sqrt
from pathlib import Path
import numpy as np

from fibernet.analysis.stokes_cap_flux import _FACE, _PAIRS as _TET_PAIRS, _cap_faces, _checked_affine_xyz
from fibernet.io._tet4_stokes_velocity_vtu import read_own_quadratic_vtu

_PAIRS = ((0, 1), (1, 2), (0, 2))
_VERTEX_POWERS = tuple(tuple(int(i == j) for i in range(3)) for j in range(3))
_POLYNOMIALS = tuple(((2, tuple(2*x for x in exp)), (-1, exp)) for exp in _VERTEX_POWERS) + tuple(
    ((4, tuple(a+b for a,b in zip(_VERTEX_POWERS[i], _VERTEX_POWERS[j]))),) for i,j in _PAIRS)

def _mass_entry(f, g):
    def integral(a, b):
        i,j,k=(x+y for x,y in zip(a,b))
        return 2*factorial(i)*factorial(j)*factorial(k)/factorial(i+j+k+2)
    return fsum(c*d*integral(a,b) for c,a in f for d,b in g)

_MASS = np.array([[_mass_entry(f,g) for g in _POLYNOMIALS] for f in _POLYNOMIALS])
_MASS.setflags(write=False)
_FLUX_EPS = 64*np.finfo(np.float64).eps
# Symmetric tetra10 edge lookup from the same VTK ordering as the flux audit.
_EDGE_COLUMN = np.full((4,4), -1, dtype=int)
for _column, (_i,_j) in enumerate(_TET_PAIRS, start=4):
    _EDGE_COLUMN[_i,_j] = _EDGE_COLUMN[_j,_i] = _column
_EDGE_COLUMN.setflags(write=False)

def _trace(points, tets, velocity, high):
    plane, owner, local, corners = _cap_faces(points, tets, high)
    if np.any(np.diff(np.sort(corners,axis=1),axis=1)==0):
        raise ValueError("Repeated cap corner")
    yz = points[corners,1:]
    order = np.lexsort((yz[:,:,1],yz[:,:,0]),axis=1)
    corner_ids = np.take_along_axis(corners,order,axis=1)
    yz = np.take_along_axis(yz,order[:,:,None],axis=1)
    a,b,c = yz.transpose(1,0,2)
    p,q = b-a,c-a
    area = np.abs(p[:,0]*q[:,1]-q[:,0]*p[:,1])/2
    if np.any(area<=0) or not np.isfinite(area).all():
        raise ValueError("Invalid cap area")
    # Map canonical face edges through each original tet's local numbering.
    ordinal = np.take_along_axis(_FACE[local],order,axis=1)
    ci,cj = np.asarray(_PAIRS).T
    mid = tets[owner[:,None],_EDGE_COLUMN[ordinal[:,ci],ordinal[:,cj]]]
    if not np.all(points[mid,0]==plane):
        raise ValueError("Cap midpoint is off the exact plane")
    mids = points[mid,1:].reshape(-1,6)
    speed = (1 if high else -1)*velocity[np.concatenate((corner_ids,mid),axis=1),0].astype(np.float64,copy=False)
    seen = {tuple(row):(ar,v,tuple(m)) for row,ar,v,m in zip(yz.reshape(-1,6),area,speed,mids)}
    if len(seen)!=len(owner):
        raise ValueError("Repeated cap triangle in physical coordinates")
    return plane,seen


def _compare_trace(before, after):
    x,a=before
    y,b=after
    if x!=y or a.keys()!=b.keys() or any(a[key][2]!=b[key][2] for key in a):
        raise ValueError("The physical cap triangles and midpoint coordinates must match exactly")
    keys=sorted(a)
    areas=np.array([a[k][0] for k in keys])
    u=np.array([a[k][1] for k in keys]); v=np.array([b[k][1] for k in keys])
    area=fsum(areas)
    vel=float(max(np.abs(u).max(),np.abs(v).max()))
    if not np.isfinite(area) or area<=0 or vel<=0 or not np.isfinite(vel):
        raise ValueError("Unrepresentable cap area or velocity")
    weights=areas/area
    u,v=u/vel,v/vel
    def flux(z):
        return fsum(weights*z[:,3:].mean(axis=1))
    q0,q1=flux(u),flux(v)
    def resolved(q, z):
        scale = fsum(weights*np.max(np.abs(z),axis=1))
        return abs(q) > _FLUX_EPS*scale
    if not resolved(q0,u) or not resolved(q1,v) or (q0 < 0) != (q1 < 0):
        raise ValueError("Resolved nonzero reference and candidate flows in the same direction are required")
    gain=q1/q0
    r=v-gain*u
    def norm(z):
        out=fsum(weights*np.einsum('ti,ij,tj->t',z,_MASS,z))
        if not np.isfinite(out) or out<0:
            raise ValueError("Unrepresentable P2 cap squared norm")
        return sqrt(out)
    base=norm(u)
    if base<=0:
        raise ValueError("Unrepresentable reference cap norm")
    f0,f1=area*(vel*q0),area*(vel*q1)
    result=dict(plane_x_m=x,triangles=len(keys),area_m2=area,reference_outward_m3_s=f0,
                candidate_outward_m3_s=f1,throughput_ratio=gain,
                normalized_profile_residual_L2=norm(r)/base,
                residual_signed_flux_over_reference=flux(r)/abs(q0))
    if not all(np.isfinite(k) for k in result.values()) or f0==0 or f1==0:
        raise ValueError("Non-finite or underflowed cap comparison")
    return result

def compare_saved_stokes_caps(reference_path, candidate_path):
    """Compare same-direction saved affine P2 traces on identical physical caps.

    Inputs must have exactly matching cap planes and physical triangles/midpoints;
    interior meshes may differ. Ratios are dimensionless; flows in m³/s are
    outward signed. Difference in throughput need not indicate redistribution.
    This routine does not infer the cause, pressure, viscosity or streamline path.
    """
    def read(path):
        p,t,v=read_own_quadratic_vtu(path)
        return _checked_affine_xyz(p,t),t,v
    before,after=read(reference_path),read(candidate_path)
    return {'schema':'fibernet-p2-cap-comparison-1',
            **{k:_compare_trace(_trace(*before,high),_trace(*after,high))
               for k,high in (('x_min',False),('x_max',True))}}

def main(argv=None):
    """Print the Python API report as JSON for shell workflows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="reference FiberNet quadratic velocity VTU")
    parser.add_argument("candidate", type=Path, help="candidate FiberNet quadratic velocity VTU")
    args = parser.parse_args(argv)
    try:
        report = compare_saved_stokes_caps(args.reference, args.candidate)
        output = json.dumps(report, allow_nan=False, indent=2)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(output)


if __name__ == "__main__":
    main()

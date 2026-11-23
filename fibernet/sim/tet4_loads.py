"""Convert global constant face traction into bounded Tet4 nodal forces."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import math

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle, _binary64_ratio, _float64, _integer, _unmasked_field
from fibernet.sim.tet4 import _LENGTH_TO_M, _check_normal_or_zero, _owned_static_mesh, _validate_topology

__all__ = ['nodal_forces_from_traction']


def _is_triple(value):
    if isinstance(value, np.ndarray):
        return value.shape == (3,)
    return isinstance(value, (list, tuple)) and len(value) == 3


def _reject_mask(value, label):
    # Do not recursively expand an oversized or nested malformed selector.
    if np.ma.isMaskedArray(value):
        _unmasked_field(value, label)


def _prepare_exterior_face_areas(mesh, faces_by_node_id):
    """Validate selectors on an owned Tet4 mesh; prepare exact coordinates.

    One-owner incidence is topological, not a spatial conformity check.
    """
    _reject_mask(faces_by_node_id, 'faces')
    if isinstance(faces_by_node_id, np.ndarray) and (
            faces_by_node_id.ndim != 2 or faces_by_node_id.shape[1] != 3):
        raise ValueError('face array must have shape (M, 3) with exact node IDs')
    if (not isinstance(faces_by_node_id, (tuple, list, np.ndarray))
            or not 1 <= len(faces_by_node_id) <= 4*len(mesh.tetrahedra)):
        raise ValueError('supply a nonempty bounded sequence of selected triangle ID triples')
    node_rows={int(key):i for i,key in enumerate(mesh.node_ids)}
    selected=set()
    for face in faces_by_node_id:
        _reject_mask(face, 'faces')
        if not _is_triple(face):
            raise ValueError('each selected face must contain three exact node IDs')
        face=tuple(sorted(_integer(k,'selected face node IDs') for k in face))
        if len(set(face))!=3 or any(k not in node_rows for k in face):
            raise ValueError('selected face has repeated or foreign node IDs')
        if face in selected:
            raise ValueError('duplicate selected face')
        selected.add(face)
    owners_by_row = _validate_topology(mesh.tetrahedra)
    for face in sorted(selected):
        key = tuple(sorted(node_rows[i] for i in face))
        if owners_by_row.get(key) != 1:
            raise ValueError('selected face is not an exterior Tet4 triangle (interior or absent)')

    # Compute the cross product from represented coordinate operands. This
    # avoids silent false zero for almost parallel edges or DAZ/FTZ
    # erasure of represented subnormal coordinates before taking differences.
    points={}
    for key in set(k for face in selected for k in face):
        raw=mesh.points[node_rows[key]].view(np.uint64).tolist()
        points[key]=tuple(Fraction(*_binary64_ratio(word)) for word in raw)
    return sorted(selected), points, Fraction(str(_LENGTH_TO_M[mesh.length_unit]))**2/2


def _iter_exterior_face_areas(prepared):
    """Yield canonical stable-ID triples and positive normal areas in m².

    Callers choose how to translate an OverflowError from area arithmetic.
    """
    selected, points, half_si = prepared
    for a,b,c in selected:
        start=points[a]
        u=[q-p for q,p in zip(points[b],start)]
        v=[q-p for q,p in zip(points[c],start)]
        cross=[u[j]*v[k]-u[k]*v[j] for j,k in ((1,2),(2,0),(0,1))]
        m=max(map(abs,cross))
        if not m:
            raise ValueError('selected exterior face has zero area')
        area_scale = m * half_si
        # Extract the exponent before converting. The largest cross
        # component alone may be subnormal even if the norm is normal.
        exponent = (area_scale.numerator.bit_length()
                    - area_scale.denominator.bit_length() + 1)
        significand = float(area_scale / (Fraction(2) ** exponent))
        norm = math.hypot(*(float(z/m) for z in cross))
        area = math.ldexp(significand * norm, exponent)
        _check_normal_or_zero(area,'selected triangle area in m²')
        if not (area>0. and math.isfinite(area)):
            raise ValueError('selected triangle area is outside supported normal binary64 range')
        yield (a,b,c), area


def nodal_forces_from_traction(bundle: Tet4Bundle, faces_by_node_id, *,
                               traction_Pa) -> dict[int, tuple[float, float, float]]:
    """Return stable ID -> Fx,Fy,Fz (N) from fixed global traction (Pa).

    The supplied face IDs may be in any order. Each selected face must be
    a distinct three-node face incident to exactly one tetrahedron. Incidence
    is topological; this does not certify a spatial conformity/overlap model.
    No normals, pressure, total-force normalization, supports or solve.
    Mesh counts follow the bounded Tet4 static solver. Intermediate face
    areas and force results must be in finite normal binary64 range.
    """
    mesh = _owned_static_mesh(bundle)
    _reject_mask(traction_Pa, 'traction_Pa')
    if not _is_triple(traction_Pa):
        raise ValueError('traction_Pa must be three finite Cartesian components, not pressure or total force')
    traction = [_float64(x,'traction in Pa') for x in traction_Pa]
    _check_normal_or_zero(traction, 'traction in Pa')
    prepared = _prepare_exterior_face_areas(mesh, faces_by_node_id)
    area_terms=defaultdict(list)
    try:
        for face, area in _iter_exterior_face_areas(prepared):
            for i in face:
                area_terms[i].append(area)
        force={}
        for key,terms in sorted(area_terms.items()):
            # All contributions share one traction sign. Sum dimensionless
            # area ratios so that a representable force is not rejected only
            # because the unweighted area sum is larger than binary64.
            scale=max(terms)
            a,ae=math.frexp(scale)
            ratio_sum=math.fsum(q/scale for q in terms)
            values=[]
            for t in traction:
                if t==0.:
                    values.append(0.)
                    continue
                m,te=math.frexp(t)
                value=math.ldexp((a*m)*(ratio_sum/3),ae+te)
                _check_normal_or_zero(value,'resulting nodal force in N')
                if not value or not math.isfinite(value):
                    raise ValueError('resulting nodal force is outside supported normal binary64 range')
                values.append(value)
            force[key]=tuple(values)
        return force
    except OverflowError as exc:
        raise ValueError('selected area or nodal force is outside supported binary64 range') from exc

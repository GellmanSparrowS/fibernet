"""Bounded small-strain static elasticity on a conforming Tet4Bundle.

Import explicitly from this module. This solver does not accept a
StructureGraph and does not require the optional Meshio file I/O dependency.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import math

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee

from fibernet.io.tet4_meshio import (
    Tet4Bundle, _binary64_ratio, _float64, _integer, _row_count, _unmasked_field, _validate,
)

__all__ = ['Tet4CellFields', 'Tet4StaticResult', 'solve_static_tet4']

_LENGTH_TO_M = {'m': 1., 'cm': 1e-2, 'mm': 1e-3, 'um': 1e-6, 'nm': 1e-9}
_AXIS = {'x': 0, 'y': 1, 'z': 2}
_MAX_POINTS = 1800
_MAX_TETS = 8000
_MAX_FACTOR_ENTRIES = 3_000_000
_MAX_ELEMENT_CONDITION = 1e10
_MIN_NORMAL_BITS = 1 << 52
_FIELD_FAST_PRODUCT_MIN_BITS = (1023 - 900) << 52  # binary64 encoding of 2**-900
_MIN_NORMAL_EXACT = Fraction(1, 1 << 1022)
_RIGID_RANK_TOLERANCE = 128. * np.finfo(np.float64).eps

@dataclass(frozen=True)
class Tet4CellFields:
    """Detached SI tensors, constant in each cell and ordered by cell_ids.

    Strain is dimensionless tensor strain (not engineering shear). Cauchy
    stress is tensile-positive, and both tensors are in global x/y/z axes.
    """

    cell_ids: np.ndarray
    material_ids: np.ndarray
    strain: np.ndarray
    stress_Pa: np.ndarray
    volume_m3: np.ndarray


@dataclass(frozen=True)
class Tet4StaticResult:
    """Nodal arrays follow the returned stable node_ids; all physical values in SI."""

    node_ids: np.ndarray
    displacement_m: np.ndarray
    reaction_N: np.ndarray
    fixed_mask: np.ndarray
    strain_energy_J: float
    half_external_work_J: float
    free_residual_inf_N: float
    normalized_backward_error: float
    relative_energy_work_error: float
    cell_fields: Tet4CellFields | None = None

@dataclass(frozen=True)
class _BandPlan:
    """Pure symbolic plan on the full-to-free DOF map; no stiffness values."""

    free: np.ndarray
    permutation: np.ndarray
    topology_half_bandwidth: int
    L_plus_U_capacity: int
    padded_factor_entries: int


class _BandBudgetExceeded(ValueError):
    """The structural envelope exceeded the frozen budget, before factorization."""

    def __init__(self, plan):
        self.plan = plan
        super().__init__(f'Tet4 topology band exceeds unchanged factor-entry budget '
                         f'({plan.L_plus_U_capacity} > {_MAX_FACTOR_ENTRIES}; '
                         f'free DOFs={len(plan.free)}, h={plan.topology_half_bandwidth})')

def _row_lookup(by_id, id_to_row, label):
    if not isinstance(by_id, Mapping) or len(by_id) > len(id_to_row):
        raise ValueError(f'{label} must be a mapping of mesh node IDs')
    _unmasked_field(by_id, label)
    seen = set()
    for key, value in by_id.items():
        key = _integer(key, label)
        if key not in id_to_row or key in seen:
            raise ValueError(f'{label}: foreign or duplicate stable node ID {key}')
        seen.add(key)
        yield id_to_row[key], value


def _loads_and_constraints(mesh, forces_by_id, fixed_dofs_by_id):
    rows = {int(key): row for row, key in enumerate(mesh.node_ids)}
    force = np.zeros((len(rows), 3), dtype=np.float64)
    fixed = np.zeros(force.shape, dtype=bool)
    for row, value in _row_lookup(forces_by_id, rows, 'forces_by_id'):
        if isinstance(value, np.ndarray):
            if value.shape != (3,) or value.dtype.kind not in 'fiu':
                raise ValueError('nodal force must be a finite real (Fx, Fy, Fz) in N')
        elif not isinstance(value, (tuple, list)) or len(value) != 3:
            raise ValueError('nodal force must be a finite real (Fx, Fy, Fz) in N')
        # Check exactness on the original scalar values, before either array
        # promotion of a mixed sequence or assignment to binary64 can round.
        force[row] = [_float64(component, 'nodal force in N') for component in value]
    for row, axes in _row_lookup(fixed_dofs_by_id, rows, 'fixed_dofs_by_id'):
        if not isinstance(axes, (tuple, list)) or not 1 <= len(axes) <= 3:
            raise ValueError('fixed DOF entries must list x, y, or z')
        for axis in axes:
            if not isinstance(axis, str) or axis not in _AXIS or fixed[row, _AXIS[axis]]:
                raise ValueError('fixed DOFs must be distinct Cartesian axes x, y, or z')
            fixed[row, _AXIS[axis]] = True
    if not fixed.any():
        raise ValueError('underconstrained Tet4 system: no fixed Cartesian DOFs')
    _check_normal_or_zero(force, 'nodal force')
    return force, fixed


def _magnitude_bits(values):
    # Reject represented subnormals even if the current runtime treats
    # subnormal inputs as zero in a floating-point comparison.
    return np.asarray(values, dtype=np.float64).view(np.uint64) & np.uint64((1 << 63) - 1)


def _check_normal_or_zero(values, label):
    bits = _magnitude_bits(values)
    if np.any((bits != 0) & (bits < np.uint64(1 << 52))):
        raise ValueError(f'{label} outside supported normal binary64 range')


def _validate_topology(cells):
    seen, faces = set(), Counter()
    for row, tet in enumerate(cells.tolist()):
        key = tuple(sorted(tet))
        if key in seen:
            raise ValueError(f'duplicate Tet4 connectivity at row {row}')
        seen.add(key)
        for face in combinations(key, 3):
            faces[face] += 1
            if faces[face] > 2:
                raise ValueError(f'non-manifold Tet4 face with more than two cells at row {row}')

def _check_rigid_modes(mesh, fixed):
    cells = mesh.tetrahedra.tolist()
    parent = list(range(len(cells)))

    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    faces = {}
    for i, tet in enumerate(cells):
        for face in combinations(sorted(tet), 3):
            if face in faces:
                parent[find(i)] = find(faces[face])
            else:
                faces[face] = i
    node_components = {}
    rows = defaultdict(list)
    for i, tet in enumerate(cells):
        component = find(i)
        for row in tet:
            if row not in node_components:
                node_components[row] = component
                rows[component].append(row)
            elif node_components[row] != component:
                raise ValueError('Tet4 bodies joined only at a point or edge are outside solver scope')
    if len(node_components) != len(mesh.points):
        raise ValueError('solver refuses unused point rows')
    for component, listing in rows.items():
        coords = mesh.points[listing] - mesh.points[listing[0]]
        scale = np.max(np.abs(coords))
        if not scale > 0. or not np.isfinite(scale):
            raise ValueError('Tet4 component rigid modes could not be evaluated')
        p = coords / scale
        p -= p.mean(axis=0)
        basis = np.zeros((len(p), 3, 6), dtype=np.float64)
        basis[:, :, :3] = np.eye(3)
        x, y, z = p.T
        basis[:, 0, 4], basis[:, 0, 5] = z, -y
        basis[:, 1, 3], basis[:, 1, 5] = -z, x
        basis[:, 2, 3], basis[:, 2, 4] = y, -x
        restraint = basis[fixed[listing]]
        sv = np.linalg.svd(restraint, compute_uv=False)
        if len(sv) < 6 or not sv[-1] > _RIGID_RANK_TOLERANCE * max(restraint.shape) * sv[0]:
            raise ValueError(f'underconstrained Tet4 component {component}: restraints do not suppress six rigid modes')
    return len(rows)

def _free_node_topology(cells, n_points, occupied):
    # Contract tetrahedra to their complete node-pair topology, ignoring
    # held-only nodes. Any physical K free entry must lie in this graph.
    row_group = np.full(n_points, -1, np.int32)
    row_group[occupied] = np.arange(len(occupied), dtype=np.int32)
    t = row_group[cells]
    ii = np.broadcast_to(t[:, :, None], (len(cells), 4, 4)).ravel()
    jj = np.broadcast_to(t[:, None, :], (len(cells), 4, 4)).ravel()
    valid = (ii >= 0) & (jj >= 0)
    graph = sparse.coo_matrix((np.ones(np.count_nonzero(valid), dtype=np.int32),
                              (ii[valid], jj[valid])),
                             shape=(len(occupied), len(occupied))).tocsr()
    graph.data[:] = 1
    return graph


def _canonical_node_order(graph, points, ids):
    # Canonical base-storage order is distinct from canonical hint order.
    # Sorting the hinted storage would alter the RCM tie behavior.
    x, y, z = points.T
    base = np.lexsort((ids, z, y, x))
    hint = np.lexsort((ids, x, y, -z))
    within_base = np.argsort(base)[hint]
    canonical = graph[base, :][:, base]
    canonical.setdiag(0)
    canonical.eliminate_zeros()
    canonical.sort_indices()
    hinted = canonical[within_base, :][:, within_base]
    return hint[reverse_cuthill_mckee(hinted, symmetric_mode=True)]


def _plan_band(mesh, fixed):
    """Plan from a validated Tet4Bundle and its Boolean held-axis mask."""
    if fixed.shape != mesh.points.shape or fixed.dtype != np.bool_:
        raise ValueError('band planning requires a Cartesian Boolean restraint mask')
    free = np.flatnonzero(~fixed.ravel())
    n = len(free)
    if not n:
        order = np.empty(0, dtype=np.int64)
        free.setflags(write=False)
        order.setflags(write=False)
        return _BandPlan(free, order, 0, 0, 0)

    occupied, group = np.unique(free // 3, return_inverse=True)
    graph = _free_node_topology(mesh.tetrahedra, len(mesh.points), occupied)
    node_order = _canonical_node_order(graph, mesh.points[occupied], mesh.node_ids[occupied])
    node_rank = np.empty_like(node_order)
    node_rank[node_order] = np.arange(len(node_order))
    order = np.lexsort((free % 3, node_rank[group]))
    # Each node's retained axes are contiguous. The largest index separation
    # in a complete allowed pair of node blocks occurs at an endpoint.
    first = np.full(len(occupied), n, dtype=np.int64)
    last = np.full(len(occupied), -1, dtype=np.int64)
    np.minimum.at(first, group[order], np.arange(n))
    np.maximum.at(last, group[order], np.arange(n))
    i, j = graph.nonzero()  # including each node's diagonal block
    h = int(np.max(last[i] - first[j], initial=0))
    L_plus_U = (2 * n - h) * (h + 1)
    padded = n * (h + 1)
    free.setflags(write=False)
    order.setflags(write=False)
    plan = _BandPlan(free, order, h, L_plus_U, padded)
    if max(L_plus_U, padded) > _MAX_FACTOR_ENTRIES:
        raise _BandBudgetExceeded(plan)
    return plan


def _pack_lower_band(lower, n, h, shift):
    """Represent the provided triangle exactly in LAPACK lower-band layout."""
    diagonal = lower.row - lower.col
    if int(np.max(diagonal, initial=0)) > h:
        raise ValueError('assembled matrix exceeds its predicted topology band')
    band = np.zeros((h + 1, n), dtype=np.float64, order='F')
    band[diagonal, lower.col] = lower.data
    band[0] -= shift
    return band

def _strain_matrix(grad):
    # Engineering shear, C44=C55=C66=mu. Vector order xx, yy, zz, xy, yz, zx.
    b = np.zeros((6, 12))
    x, y, z = grad.T
    b[0, 0::3], b[1, 1::3], b[2, 2::3] = x, y, z
    b[3, 0::3], b[3, 1::3] = y, x
    b[4, 1::3], b[4, 2::3] = z, y
    b[5, 2::3], b[5, 0::3] = x, z
    return b


def _material_tensor(record):
    e, nu = record['young_modulus_pa'], record['poisson_ratio']
    mu = e / (2. * (1. + nu))
    lam = 2. * mu * nu / (1. - 2. * nu)
    c = np.zeros((6, 6))
    c[:3, :3] = lam
    c[np.arange(3), np.arange(3)] += 2. * mu
    c[np.arange(3, 6), np.arange(3, 6)] = mu
    if not np.isfinite(c).all() or mu <= 0.:
        raise ValueError('material stiffness is outside supported numerical range')
    return c


def _element_kinematics(mesh):
    x = mesh.points * _LENGTH_TO_M[mesh.length_unit]
    if not np.isfinite(x).all():
        raise ValueError('point coordinates in m are outside supported range')
    tensors = {mid: _material_tensor(record) for mid, record in mesh.materials.items()}
    for k, (tet, mid) in enumerate(zip(mesh.tetrahedra, mesh.material_ids)):
        d = (x[tet[1:]] - x[tet[0]]).T
        scale = np.max(np.abs(d))
        if not np.isfinite(scale) or scale <= 0.:
            raise ValueError(f'Tet4 row {k}: invalid geometry after conversion to m')
        d = d / scale
        if np.linalg.cond(d) > _MAX_ELEMENT_CONDITION:
            raise ValueError(f'Tet4 row {k}: element is too ill-conditioned for this solver')
        volume = abs(np.linalg.det(d)) * (scale ** 3) / 6.
        if not np.isfinite(volume) or volume <= 0.:
            raise ValueError(f'Tet4 row {k}: volume is outside supported numerical range')
        corner = np.linalg.inv(d) / scale
        b = _strain_matrix(np.vstack([-corner.sum(axis=0), corner]))
        yield tet, b, tensors[int(mid)], volume


def _element_blocks(mesh):
    for tet, b, tensor, volume in _element_kinematics(mesh):
        dofs = (3 * tet[:, None] + np.arange(3)).ravel()
        kc = volume * (b.T @ tensor @ b)
        # Symmetrize once at the numerical boundary; physical operator is symmetric.
        yield dofs, 0.5 * kc + 0.5 * kc.T

def _exact_field_vector(items):
    return [Fraction(*_binary64_ratio(word)) for word in items.view(np.uint64).tolist()]


def _field_dot(matrix, vector, exact_vector=None):
    """Small dense products, with exact fallback near under/overflow."""
    with np.errstate(under='ignore', over='ignore', invalid='ignore'):
        products = matrix * vector
        result = products.sum(axis=1)
    mag = _magnitude_bits(products)
    nonzero_inputs = (_magnitude_bits(matrix) != 0) & (_magnitude_bits(vector) != 0)
    # Even two exact binary64 significands have at most 106 significant
    # product bits. Cancellation of products >= 2**-900 cannot leave a
    # nonzero subnormal residual, but can produce ordinary rounded zero.
    risky = np.any(nonzero_inputs & (mag < _FIELD_FAST_PRODUCT_MIN_BITS), axis=1) | ~np.isfinite(result)
    if exact_vector is not None:
        risky[:] = True
    if np.any(risky):
        values = _exact_field_vector(vector) if exact_vector is None else exact_vector
        for i in np.flatnonzero(risky):
            exact = sum(a * v for a, v in zip(_exact_field_vector(matrix[i]), values))
            if 0 < abs(exact) < _MIN_NORMAL_EXACT:
                raise ValueError('nonzero Tet4 cell component outside supported normal binary64 range')
            result[i] = float(exact)
    return result


def _tensor_rows(rows):
    return rows[:, [[0, 3, 5], [3, 1, 4], [5, 4, 2]]]


def _recover_cell_fields(mesh, displacement_m):
    n = len(mesh.cell_ids)
    strain, stress = np.empty((n, 6)), np.empty((n, 6))
    volume = np.empty(n)
    for i, (tet, b, tensor, cell_volume) in enumerate(_element_kinematics(mesh)):
        volume[i] = cell_volume
        # Local displacement differences improve translation cancellation.
        u = displacement_m[tet]
        with np.errstate(under='ignore', over='ignore'):
            delta = (u[1:] - u[0]).ravel()
        mag = _magnitude_bits(delta)
        lost = ((mag == 0).reshape(3, 3) & (u[1:] != u[0])).any()
        exact_delta = None
        if lost or np.any((mag != 0) & (mag < _MIN_NORMAL_BITS)) or not np.isfinite(delta).all():
            exact_u = _exact_field_vector(u.ravel())
            exact_delta = [value - exact_u[k % 3]
                           for k, value in enumerate(exact_u[3:])]
        engineering = _field_dot(b[:, 3:], delta, exact_delta)
        stress[i] = _field_dot(tensor, engineering)
        strain[i] = engineering
    _check_normal_or_zero(volume, 'Tet4 cell volume')
    # B stores engineering shear. Halving a nonzero value may itself
    # underflow; check before multiplying, including with FTZ enabled.
    if np.any((_magnitude_bits(strain[:, 3:]) != 0) &
              (_magnitude_bits(strain[:, 3:]) < 2 * _MIN_NORMAL_BITS)):
        raise ValueError('nonzero Tet4 cell strain outside supported normal binary64 range')
    strain[:, 3:] *= 0.5
    arrays = (mesh.cell_ids, mesh.material_ids, _tensor_rows(strain),
              _tensor_rows(stress), volume)
    for a in arrays:
        a.setflags(write=False)
    return Tet4CellFields(*arrays)


def _assemble(mesh):
    n = len(mesh.tetrahedra)
    # The explicit cap keeps assembly bounded; row/column indexes fit int32.
    rr = np.empty((n, 12, 12), dtype=np.int32)
    cc = np.empty_like(rr)
    vv = np.empty((n, 12, 12), dtype=np.float64)
    for i, (dof, block) in enumerate(_element_blocks(mesh)):
        rr[i], cc[i], vv[i] = dof[:, None], dof[None, :], block
    k = sparse.coo_matrix((vv.ravel(), (rr.ravel(), cc.ravel())),
                          shape=(mesh.points.size, mesh.points.size)).tocsr()
    # COO sums duplicated entries independently; their floating reduction
    # order need not agree in transpose pairs. Re-establish bitwise symmetry
    # for the no-pivot symmetric shifted factorization.
    k = 0.5 * k + 0.5 * k.T
    k.eliminate_zeros()
    if not np.isfinite(k.data).all():
        raise ValueError('assembled sparse stiffness is outside supported numerical range')
    return k

def _checked_normal_rescale(value, mantissa, exponent, label):
    physical = np.ldexp(value * mantissa, exponent)
    _check_normal_or_zero(physical, label)
    if (not np.isfinite(physical).all() or
            np.any((_magnitude_bits(value) != 0) & (_magnitude_bits(physical) == 0))):
        raise ValueError(f'{label} outside supported normal binary64 range')
    return physical

def _factor_band(a, floor, plan):
    order = plan.permutation
    n = len(order)
    h = plan.topology_half_bandwidth
    if n == 0 or not 0 <= h < n:
        raise ValueError('invalid factor band dimensions in structural plan')
    capacity, padded = (2 * n - h) * (h + 1), n * (h + 1)
    if capacity != plan.L_plus_U_capacity or padded != plan.padded_factor_entries:
        raise ValueError('inconsistent structural band capacity')
    if max(capacity, padded) > _MAX_FACTOR_ENTRIES:
        raise _BandBudgetExceeded(plan)
    if a.shape != (n, n) or n != len(plan.free):
        raise ValueError('free equations do not match structural band plan')
    ordered = a[order, :][:, order]
    if (ordered - ordered.T).nnz:
        raise ValueError('banded Tet4 requires exact numerical symmetry')
    lower = sparse.tril(ordered, format='coo')

    def run(shift):
        ab = _pack_lower_band(lower, n, h, shift)
        result = linalg.cholesky_banded(ab, lower=True, overwrite_ab=True, check_finite=True)
        diag = result[0]
        if not np.isfinite(diag).all() or np.min(diag) <= 0.:
            raise ValueError('unsafe banded Tet4 Cholesky diagonal')
        return result

    try:
        test = run(floor)
    except np.linalg.LinAlgError as exc:
        raise ValueError('underconstrained or ill-conditioned Tet4: shifted band floor rejected') from exc
    del test
    try:
        actual = run(0.)
    except np.linalg.LinAlgError as exc:
        raise ValueError('unshifted Tet4 band factorization failed') from exc

    def apply(rhs):
        answer = np.empty_like(rhs)
        answer[order] = linalg.cho_solve_banded((actual, True), rhs[order], check_finite=True)
        return answer

    return apply

def _fixed_solve(stiffness, force, plan):
    free = plan.free
    u = np.zeros(force.size)
    if not len(free):
        return u, free, 0., 0., 0., 0.
    rhs = force.ravel()[free]
    reduced = stiffness[free, :][:, free]
    # Scale by a stiffness norm first, then test the spectrum even if all
    # loads are constrained, so a zero RHS cannot conceal a rigid mode.
    kscale = float(np.max(np.abs(reduced.data), initial=0.))
    if not np.isfinite(kscale) or kscale <= 0.:
        raise ValueError('underconstrained Tet4 system: free stiffness has zero scale')
    _check_normal_or_zero(kscale, 'free stiffness scale')
    khat = reduced / kscale
    norm_khat = float(np.asarray(abs(khat).sum(axis=1)).max(initial=0.))
    eps = np.finfo(np.float64).eps
    tolerance = 64. * eps * len(free)
    solve = _factor_band(khat, tolerance, plan)
    fscale = float(np.max(np.abs(rhs)))
    if fscale == 0.:
        return u, free, 0., 0., 0., 0.
    fhat = rhs / fscale
    # A normalized RHS with nonzero components erased by scaling is outside
    # this deliberately bounded solver's supported dynamic range.
    _check_normal_or_zero(fhat, 'normalized free force')
    if np.any((_magnitude_bits(rhs) != 0) & (_magnitude_bits(fhat) == 0)):
        raise ValueError('normalized free force outside supported normal binary64 range')
    uhat = solve(fhat)
    fm, fe = math.frexp(fscale)
    km, ke = math.frexp(kscale)
    u[free] = _checked_normal_rescale(uhat, fm / km, fe - ke, 'displacement')

    # Certify the returned, rescaled displacement, not the ideal internal one.
    actual_uhat = np.ldexp(u[free], ke - fe) * (km / fm)
    internal_hat = khat @ actual_uhat
    residual_hat = internal_hat - fhat
    denominator = norm_khat * np.linalg.norm(actual_uhat, ord=np.inf) + 1.
    backward = float(np.linalg.norm(residual_hat, ord=np.inf) / denominator)
    if not 0. <= backward <= tolerance:
        raise ValueError('Tet4 solve did not meet normalized backward residual criterion')

    # This branch has a represented nonzero free load on an SPD operator,
    # so its work and energy must be strictly positive and agree. A load on
    # supported axes contributes reactions but zero work and is not tested here.
    energy_hat = float(.5 * (actual_uhat @ internal_hat))
    work_hat = float(.5 * (actual_uhat @ fhat))
    if not min(energy_hat, work_hat) > 0.:
        raise ValueError('nonzero free load did not produce positive work and energy')
    relative = abs(energy_hat - work_hat) / max(energy_hat, work_hat)
    if not 0. <= relative <= 4096. * eps * len(free):
        raise ValueError('Tet4 strain energy and half external work do not agree')
    energy, half_work = _checked_normal_rescale(np.array([energy_hat, work_hat]), fm ** 2 / km,
                                          2 * fe - ke, 'positive work/energy')
    return u, free, float(energy), float(half_work), backward, relative

def solve_static_tet4(bundle: Tet4Bundle, *, forces_by_id: Mapping,
                      fixed_dofs_by_id: Mapping,
                      return_cell_fields: bool = False) -> Tet4StaticResult:
    """Solve first-order isotropic static elasticity with zero Cartesian holds.

    Supply nodal forces in N as three-component vectors keyed by stable node
    ID. Holds map stable node IDs to sequences of distinct 'x', 'y' or 'z'
    axes. Unlisted forces are zero and unlisted axes are free. Geometry
    follows the bundle's length unit; its elastic moduli are in Pa.

    Returns SI displacement, reaction, and energy in a detached result, in
    the order of the bundle's node IDs. Unsupported geometry, singular or
    poorly conditioned equations, or an exceeded resource limit raise
    ValueError. Does not solve beam, nonlinear, or time-dependent mechanics.

    With return_cell_fields=True, also recover one constant strain tensor,
    tensile-positive Cauchy stress tensor (Pa) and volume (m³) per validated
    cell. This is an element-wise diagnostic, not a peak stress estimate.
    """
    if not isinstance(return_cell_fields, bool):
        raise TypeError('return_cell_fields must be a bool')
    if not isinstance(bundle, Tet4Bundle):
        raise TypeError('expected a Tet4Bundle')
    if (_row_count(bundle.points, 'points') > _MAX_POINTS or
            _row_count(bundle.tetrahedra, 'tetrahedra') > _MAX_TETS):
        raise ValueError(f'Tet4 solve is limited to {_MAX_POINTS} nodes and {_MAX_TETS} cells')
    mesh = _validate(bundle.points, bundle.tetrahedra, bundle.node_ids, bundle.cell_ids,
                     bundle.material_ids, bundle.materials, bundle.length_unit)
    _validate_topology(mesh.tetrahedra)
    force, fixed = _loads_and_constraints(mesh, forces_by_id, fixed_dofs_by_id)
    try:
        with np.errstate(divide='raise', over='raise', invalid='raise'):
            _check_rigid_modes(mesh, fixed)
            plan = _plan_band(mesh, fixed)
            k = _assemble(mesh)
            u, free, energy, half_work, backward, relative = _fixed_solve(k, force, plan)
            internal = k @ u
            if not np.isfinite(internal).all():
                raise ValueError('internal nodal forces are outside supported numerical range')
            residual = internal.reshape(fixed.shape) - force
            reactions = np.where(fixed, residual, 0.)
            worst_residual = float(np.max(np.abs(residual.ravel()[free]), initial=0.))
            _check_normal_or_zero(reactions, 'support reaction')
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError('Tet4 numerical solve could not be safely evaluated') from exc
    if not np.isfinite([energy, half_work, worst_residual, backward, relative]).all() or energy < 0.:
        raise ValueError('Tet4 energy state is outside supported numerical range')
    u = u.reshape(fixed.shape)
    fields = None
    if return_cell_fields:
        try:
            with np.errstate(divide='raise', over='raise', invalid='raise'):
                fields = _recover_cell_fields(mesh, u)
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            raise ValueError('Tet4 cell fields could not be safely evaluated') from exc
    for array in (mesh.node_ids, u, reactions, fixed):
        array.setflags(write=False)
    return Tet4StaticResult(mesh.node_ids, u, reactions, fixed, energy, half_work,
                            worst_residual, backward, relative, fields)

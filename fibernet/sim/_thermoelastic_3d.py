"""Private bounded 3D frame experiment: mean member thermal eigenstrain only.

No public exports or saved-state dispatch. Welded, straight circular members;
linear static motion, no mesh/continuum, nonlocal constitutive or feedback model.
"""
from collections.abc import Mapping
import math

import numpy as np

from fibernet.core.structure_graph import StructureGraph
from fibernet.ml.beam_frame_fem import BeamFrameFEM

_MAX_NODES, _MAX_EDGES, _MAX_FREE = 96, 192, 384
_SPD_RELATIVE_FLOOR = 1e-12
_EPS = np.finfo(np.float64).eps
_TINY = np.finfo(np.float64).tiny
_UNITS = {"m": 1.0, "mm": 1e-3, "um": 1e-6}
_APPLICABILITY_FIELDS = (
    "endpoint_thermal_strain",
    "nodal_rotation",
    "transverse_slope_bound",
    "total_extreme_fiber_strain",
    "endpoint_elastic_extreme_fiber_strain",
    "twist_shear",
)


def _scalar(x, name):
    if (isinstance(x, (bool, np.bool_, np.timedelta64))
            or not isinstance(x, (int, float, np.integer, np.floating))):
        raise ValueError(f"{name} needs ordinary finite real scalars")
    try:
        y = float(x)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} is not representable as binary64") from exc
    exact = y == int(x) if isinstance(x, (int, np.integer)) else y == x
    if not math.isfinite(y) or not exact or 0.0 < abs(y) < _TINY:
        raise ValueError(f"{name} needs an exact finite normal binary64 scalar or zero")
    return y


def _items(value, wanted, label, *, exact=False):
    if not isinstance(value, Mapping) or len(value) > len(wanted):
        raise ValueError(f"{label} needs a bounded stable-ID mapping")
    out = {}
    for key, x in value.items():
        if (isinstance(key, (bool, np.bool_, np.timedelta64))
                or not isinstance(key, (int, np.integer)) or int(key) not in wanted
                or int(key) in out):
            raise ValueError(f"{label} has an unknown or invalid ID")
        out[int(key)] = x
    if exact and out.keys() != wanted:
        raise ValueError(f"{label} requires exact stable-ID coverage")
    return out


def _normal_geometry(x, scale, label):
    y = _scalar(x, label) * scale
    if not math.isfinite(y) or (x != 0 and abs(y) < _TINY):
        raise ValueError(f"{label} cannot be converted to usable SI geometry")
    return y


def _validated_limits(limits):
    if not isinstance(limits, Mapping) or limits.keys() != set(_APPLICABILITY_FIELDS):
        raise ValueError("applicability_limits requires exactly the six named dimensionless limits")
    result = {name: _scalar(limits[name], name + " limit") for name in _APPLICABILITY_FIELDS}
    if any(limit <= 0 for limit in result.values()):
        raise ValueError("applicability_limits must be strictly positive")
    return result


def _admit(name, value, limits, location):
    if not math.isfinite(value):
        raise ArithmeticError(f"{name} is nonfinite at {location}")
    if value > limits[name]:
        raise ValueError(f"{name} exceeds caller limit at {location}")
    return value


def _norm3(v):
    return math.hypot(*(float(x) for x in v))


def _edge_applicability(eid, a, b, ui, uj, n, L, radius, eps_a, eps_b, limits):
    du = uj[:3] - ui[:3]
    axial = math.fsum(float(x) for x in du * n) / L
    c = (du - (axial * L) * n) / L
    si, sj = np.cross(ui[3:], n), np.cross(uj[3:], n)
    # Derivative of cubic Hermite transverse displacement lies in the
    # convex hull of its three Bernstein controls, so this is a bound.
    slope = max(_norm3(v) for v in (si, 3 * c - si - sj, sj))
    bend_a = 2 * radius / L * _norm3(3 * c - 2 * si - sj)
    bend_b = 2 * radius / L * _norm3(-3 * c + si + 2 * sj)
    total = max(abs(axial) + bend_a, abs(axial) + bend_b)
    elastic = {a: abs(axial - eps_a) + bend_a, b: abs(axial - eps_b) + bend_b}
    twist = radius / L * abs(math.fsum(float(x) for x in (uj[3:] - ui[3:]) * n))
    # Check every raw intermediate for finiteness before any max() can hide
    # a NaN (and before presenting derived values as finite diagnostics).
    if not all(math.isfinite(v) for v in (axial, bend_a, bend_b, *elastic.values(), twist)) or not all(
        np.all(np.isfinite(v)) for v in (c, si, sj)
    ):
        raise ArithmeticError(f"nonfinite applicability diagnostics at edge {eid}")
    location = f"edge {eid}"
    return {
        "endpoint_thermal_strain": max(abs(eps_a), abs(eps_b)),
        "transverse_slope_bound": _admit("transverse_slope_bound", slope, limits, location),
        "total_extreme_fiber_strain": _admit("total_extreme_fiber_strain", total, limits, location),
        "endpoint_elastic_extreme_fiber_strain": _admit(
            "endpoint_elastic_extreme_fiber_strain", max(elastic.values()), limits, location),
        "twist_shear": _admit("twist_shear", twist, limits, location),
        "thermal_strain_by_endpoint_id": {a: eps_a, b: eps_b},
        "elastic_extreme_fiber_strain_by_endpoint_id": elastic,
    }


def _solve_thermoelastic_3d(
    graph, *, temperature_K_by_node_id, stress_free_temperature_K,
    expansion_alpha_per_K_by_edge_id, constraints_by_node_id,
    loads_N_by_node_id=None, length_unit, applicability_limits
):
    """Solve one direct prescribed-temperature snapshot, with fixed geometry.

    Positions/radii are in the declared m/mm/um; Material E is Pa and nu
    dimensionless. Constraint rows are six signed (m,m,m,rad,rad,rad)
    values or None; omitted rows are free. Loads are three signed N values;
    omitted rows are zero. Only geometry is rescaled, not the boundary values.
    Thermal strain is alpha times mean endpoint delta-T [K]. Returned energy
    is the discrete beam energy about this mean strain; it excludes any
    assumed along-rod temperature variance and external-force potential.
    No stiffness shift. A conservative scaled spectral screen is required.
    Explicit dimensionless applicability_limits are per-field kinematic
    screening ceilings, not calibrated error, stability or material limits.
    """
    if (type(graph) is not StructureGraph or graph.dimension != 3
            or not 2 <= len(graph.nodes) <= _MAX_NODES
            or not 1 <= len(graph.edges) <= _MAX_EDGES):
        raise ValueError("requires a bounded ordinary 3-D StructureGraph")
    if type(length_unit) is not str or length_unit not in _UNITS:
        raise ValueError("declare graph geometry length_unit as m, mm or um")
    if "length_unit" in graph.metadata and graph.metadata["length_unit"] != length_unit:
        raise ValueError("declared length_unit conflicts with graph metadata")
    scale = _UNITS[length_unit]
    limits = _validated_limits(applicability_limits)
    node_ids, edge_ids = tuple(graph.nodes), tuple(graph.edges)
    if any(type(k) is not int or not 0 <= k < 2**63 for k in (*node_ids, *edge_ids)):
        raise ValueError("graph IDs must be ordinary nonnegative int64")
    ni, ei = set(node_ids), set(edge_ids)
    rows = dict(zip(node_ids, range(len(node_ids))))
    T = _items(temperature_K_by_node_id, ni, "temperature", exact=True)
    alpha = _items(expansion_alpha_per_K_by_edge_id, ei, "expansion alpha", exact=True)
    constraints = _items(constraints_by_node_id, ni, "constraints")
    loads = _items({} if loads_N_by_node_id is None else loads_N_by_node_id, ni, "forces")
    ref = _scalar(stress_free_temperature_K, "stress-free temperature")
    T = {nid: _scalar(t, "temperature") for nid, t in T.items()}
    if ref <= 0 or any(t <= 0 for t in T.values()):
        raise ValueError("temperatures must be strictly positive Kelvin")

    fixed = {}
    for nid, v in constraints.items():
        if not isinstance(v, (tuple, list)) or len(v) != 6:
            raise ValueError("constraint rows must have six components or None by component")
        for j, value in enumerate(v):
            if value is not None:
                fixed[6 * rows[nid] + j] = _scalar(value, "constraint")
    ndof = 6 * len(node_ids)
    if ndof - len(fixed) > _MAX_FREE:
        raise ValueError("reduced mechanical DOF budget exceeded")
    external = np.zeros((len(node_ids), 6))
    for nid, v in loads.items():
        if not isinstance(v, (tuple, list)) or len(v) != 3:
            raise ValueError("force rows must have exactly three translational components")
        external[rows[nid], :3] = [_scalar(x, "force") for x in v]
    pos = np.array([[_normal_geometry(x, scale, "position") for x in graph.nodes[n].position]
                    for n in node_ids])
    if pos.shape != (len(node_ids), 3):
        raise ValueError("requires XYZ node positions")

    fem = BeamFrameFEM()
    edges = [graph.edges[e] for e in edge_ids]
    materials = fem._snapshot_edge_materials([e.material for e in edges], len(edges))
    radii = np.array([_normal_geometry(e.radius, scale, "radius") for e in edges])
    rhs_th = np.zeros_like(external)
    topology, parameters, used, endpoint_thermal = [], [], set(), []
    for idx, (eid, edge, radius, material) in enumerate(zip(edge_ids, edges, radii, materials)):
        a, b = edge.node_i, edge.node_j
        if a == b or a not in ni or b not in ni or (edge.internal_points is not None and len(edge.internal_points)):
            raise ValueError(f"edge {eid} must have distinct existing endpoints and no internal points")
        i, j = rows[a], rows[b]
        delta = pos[j] - pos[i]
        length = np.linalg.norm(delta)
        if not np.isfinite(length) or length < 1e-12:
            raise ValueError(f"edge {eid} length is below the 3-D linear assembly floor or nonfinite")
        area = fem.section_properties(radius)["A"]
        unit = delta / length
        # Average differences first: no addition of two huge Kelvin values.
        dT = 0.5 * (T[a] - ref) + 0.5 * (T[b] - ref)
        aT = _scalar(alpha[eid], "expansion alpha")
        # Never let opposite-sign endpoint thermal strains cancel the
        # applicability check. This occurs before stiffness or solve.
        ea_T, eb_T = aT * (T[a] - ref), aT * (T[b] - ref)
        _admit("endpoint_thermal_strain", abs(ea_T), limits, f"edge {eid}, node {a}")
        _admit("endpoint_thermal_strain", abs(eb_T), limits, f"edge {eid}, node {b}")
        endpoint_thermal.append((ea_T, eb_T))
        eps = aT * dT
        ea = material.youngs_modulus * area
        equivalent = ea * eps * unit
        if (not math.isfinite(eps) or not math.isfinite(ea) or ea < _TINY
                or not np.all(np.isfinite(equivalent))):
            raise ArithmeticError(f"edge {eid} thermal assembly exceeds the numerical range")
        rhs_th[i, :3] -= equivalent
        rhs_th[j, :3] += equivalent
        topology.append((i, j))
        used.update((i, j))
        parameters.append((length, unit, area, ea, eps))
    if len(used) != len(node_ids):
        raise ValueError("isolated node outside the private owner domain")
    conn = np.array(topology, dtype=np.int64).T
    order = np.arange(len(edges))
    K = fem._assemble_3d_physical_stiffness(conn, pos, radii, materials, order, ndof)

    rhs = (external + rhs_th).ravel()
    if not np.all(np.isfinite(rhs)):
        raise ArithmeticError("combined nodal forces are nonfinite")
    u = np.zeros(ndof)
    known = np.array(sorted(fixed), dtype=np.int64)
    u[known] = [fixed[i] for i in known]
    free = np.setdiff1d(np.arange(ndof), known)
    spectral_ratio = None
    if len(free):
        # Scale generalized rotations to displacements by a characteristic
        # length. Work-conjugate duals use the matching transpose scale.
        char = max(v[0] for v in parameters)
        S = np.where(free % 6 < 3, 1., 1. / char)
        A = (K[free][:, free].toarray() * S[None, :]) * S[:, None]
        if not np.all(np.isfinite(A)):
            raise ArithmeticError("scaled free stiffness is nonfinite")
        magnitude = float(np.max(np.abs(A)))
        if magnitude < _TINY:
            raise ValueError("free stiffness is numerically unresolved")
        A /= magnitude
        if np.max(np.abs(A - A.T)) > 32 * _EPS:
            raise ValueError("free stiffness is not symmetric within roundoff")
        A = (A + A.T) * 0.5
        spectrum = np.linalg.eigvalsh(A)
        if (not np.all(np.isfinite(spectrum)) or spectrum[-1] <= 0
                or spectrum[0] <= _SPD_RELATIVE_FLOOR * spectrum[-1]):
            raise ValueError("free stiffness is underconstrained or near singular")
        spectral_ratio = float(spectrum[0] / spectrum[-1])
        physical_rhs = (rhs - K @ u)[free]
        normalized_rhs = (physical_rhs / magnitude) * S
        if not np.all(np.isfinite(normalized_rhs)):
            raise ArithmeticError("scaled right hand side is nonfinite")
        if np.any((physical_rhs != 0) & (np.abs(normalized_rhs) < _TINY)):
            raise ArithmeticError("scaled nonzero right hand side below the normal range")
        factor = np.linalg.cholesky(A)
        w = np.linalg.solve(factor.T, np.linalg.solve(factor, normalized_rhs))
        backward_scale = np.linalg.norm(A, np.inf) * np.linalg.norm(w, np.inf)
        backward_scale += np.linalg.norm(normalized_rhs, np.inf)
        deviation = np.linalg.norm(A @ w - normalized_rhs, np.inf)
        if (not math.isfinite(backward_scale) or not math.isfinite(deviation)
                or deviation > 1e-11 * backward_scale):
            raise ArithmeticError("free solve failed the scaled backward-residual check")
        u[free] = S * w
        if (np.any((w != 0) & (np.abs(u[free]) < _TINY))
                or (np.any(normalized_rhs) and not np.any(u[free]))):
            raise ArithmeticError("nonzero free solution below the normal range")
    reactions = (K @ u - rhs).reshape(-1, 6)
    if not all(np.all(np.isfinite(x)) for x in (u, rhs, reactions)):
        raise ArithmeticError("nonfinite displacement/load/reaction")
    u = u.reshape(-1, 6)

    applicability = {
        "scope": "linear_kinematic_diagnostic_bounds_not_stability_or_material_validation",
        "limits": limits,
        "nodal_rotation_rad_by_id": {
            nid: _admit("nodal_rotation", _norm3(u[rows[nid], 3:]), limits, f"node {nid}")
            for nid in node_ids
        },
        "edge_diagnostics_by_id": {
            eid: _edge_applicability(
                eid, node_ids[i], node_ids[j], u[i], u[j], n, L, radii[idx],
                *endpoint_thermal[idx], limits,
            )
            for idx, (eid, (i, j), (L, n, *unused)) in enumerate(zip(edge_ids, topology, parameters))
        },
    }

    # Compute stable energy about the thermal reference on each element by
    # removing one rigid motion and one thermal axial translation. Use the
    # SAME assembly kernel, at most one temporary 12x12 per edge, no second
    # stiffness formula and no retained local-matrix table.
    axial_N, stress_Pa, energies, bounds = [], [], [], []
    two = np.array([[0], [1]])
    for idx, ((i, j), (L, n, area, ea, eps)) in enumerate(zip(topology, parameters)):
        du = u[j, :3] - u[i, :3]
        elong = math.fsum(float(x) for x in du * n)
        relative = elong - L * eps
        force = ea * (relative / L)
        remaining = du - elong * n - np.cross(u[i, 3:], L * n)
        w = np.r_[np.zeros(6), remaining + relative * n, u[j, 3:] - u[i, 3:]]
        local = fem._assemble_3d_physical_stiffness(two, pos[[i, j]], radii[idx:idx+1],
                                                   materials[idx:idx+1], (0,), 12)
        value = 0.5 * float(w @ (local @ w))
        bound = 32 * _EPS * float(abs(w) @ (abs(local) @ abs(w)))
        stress = force / area
        if not all(math.isfinite(x) for x in (value, bound, force, stress)) or value < -bound:
            raise ArithmeticError("element storage cannot be certified")
        if not value and np.any(w) and not bound:
            raise ArithmeticError("nonzero element deformation has no resolved storage")
        axial_N.append(force)
        stress_Pa.append(stress)
        energies.append(max(0., value))
        bounds.append(bound)
    return {
        "node_ids": node_ids, "edge_ids": edge_ids,
        "displacement_m_and_rad": u, "reactions_N_and_Nm": reactions,
        "axial_force_N": np.array(axial_N), "axial_stress_Pa": np.array(stress_Pa),
        "mean_thermal_strain": np.array([v[-1] for v in parameters]),
        "elastic_energy_J": math.fsum(energies), "edge_elastic_energy_J": np.array(energies),
        "element_energy_roundoff_bound_J": math.fsum(bounds),
        "scaled_stiffness_min_over_max": spectral_ratio,
        "applicability": applicability,
    }

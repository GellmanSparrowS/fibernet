"""Bounded sparse operator; extracted functions retain original equations."""
from __future__ import annotations
import math
import json
JSON_CAP = 1024**2
RESULT_CAP = 4*1024**2

RESOURCES = {"cpus": 1, "timeout_seconds": 180, "memory_budget_mb": 2048,
             "reserve_memory_mb": 4096}
SMALL_PROFILE = "small_700_refuse_unanchored"
LARGE_PROFILE = "large_sparse_loaded_components_8g8g"
PROFILES = {
    SMALL_PROFILE: {"nodes": 700, "DOFs": 4200, "segments": 4200, "config_json_bytes": JSON_CAP,
        "result_bytes": RESULT_CAP, "resources": RESOURCES, "unanchored_policy": "refuse",
        "allocation_model": "dense_fill_allowance"},
    LARGE_PROFILE: {"nodes": 6000, "DOFs": 36000, "segments": 7000, "config_json_bytes": 8*1024**2,
        "result_bytes": 64*1024**2, "resources": {"cpus": 1, "timeout_seconds": 180,
            "memory_budget_mb": 8192, "reserve_memory_mb": 8192},
        "unanchored_policy": "retain_undefined_exclude_dofs", "allocation_model": "sparse_fill_allowance",
        "declared_fill_multiplier": 256, "bytes_per_fill_entry": 24}}

def numeric_modules():
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import splu
    from . import _beam
    return np, sparse, splu, _beam

def number(value, label, *, positive=False):
    if type(value) not in (int, float):
        raise ValueError("Actual finite numeric " + label + " required")
    try:
        answer = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("Unrepresentable " + label) from error
    if not math.isfinite(answer) or (positive and answer <= 0):
        raise ValueError("Finite " + label + " required")
    if type(value) is int and int(answer) != value:
        raise ValueError("Inexact physical integer conversion: " + label)
    return answer

def identifier(value):
    if type(value) is not int or value < 0:
        raise ValueError("Nonnegative literal integer ID required; original zero-based IDs are retained")
    return value

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

def study_profile(name):
    if type(name) is not str or name not in PROFILES:
        raise ValueError("Explicit known study profile required")
    return json.loads(canonical(PROFILES[name]))

def checked_network(network, *, profile_name=SMALL_PROFILE):
    profile = study_profile(profile_name)
    nodes, segments = network.get("nodes"), network.get("segments")
    if type(nodes) is not list or type(segments) is not list:
        raise ValueError("Ordered node and segment lists required")
    if (not 2 <= len(nodes) <= profile["nodes"] or 6*len(nodes) > profile["DOFs"] or
            not 1 <= len(segments) <= profile["segments"]):
        raise MemoryError("Separate study node/DOF/segment cap exceeded")
    order, positions, edges, seen = [], {}, [], set()
    for node in nodes:
        nid = identifier(node["id"])
        if nid in positions or node.get("junction_type", "welded") != "welded":
            raise ValueError("Unique shared welded node required")
        p = node["position"]
        if type(p) is not list or len(p) != 3:
            raise ValueError("Three-dimensional position required")
        positions[nid] = [number(x, "position") for x in p]
        order.append(nid)
    for segment in segments:
        sid, a, b = (identifier(segment[k]) for k in ("id", "node_i", "node_j"))
        if sid in seen or a == b or a not in positions or b not in positions:
            raise ValueError("Unique valid original beam endpoints required")
        if math.dist(positions[a], positions[b]) <= 0:
            raise ValueError("Positive finite chord required")
        seen.add(sid)
        edges.append((sid, a, b))
    for link in network.get("crosslinks", []):
        if link.get("type") != "welded" or len(link.get("node_ids", [])) != 1 or link["node_ids"][0] not in positions:
            raise ValueError("Separate-node/contact/sliding/hinged coupling unsupported")
    return order, positions, edges

def components(order, edges):
    graph = {nid: set() for nid in order}
    for _, a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    unseen, finite, isolates = set(order), [], []
    for start in order:
        if start not in unseen:
            continue
        pending, group = [start], []
        unseen.remove(start)
        while pending:
            nid = pending.pop()
            group.append(nid)
            for neighbor in graph[nid]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    pending.append(neighbor)
        (finite if graph[start] else isolates).append(sorted(group) if graph[start] else start)
    return finite, isolates

def allocation_forecast(node_count, segment_count, profile_name):
    profile = study_profile(profile_name)
    if type(node_count) is not int or type(segment_count) is not int or not 2 <= node_count <= profile["nodes"] or not 1 <= segment_count <= profile["segments"]:
        raise MemoryError("Allocation dimensions outside study profile")
    n = node_count*6
    if profile["allocation_model"] == "dense_fill_allowance":
        forecast = 12*n*n*8 + segment_count*32768 + n*8192
    else:
        structural_nnz_bound = 144*segment_count+6*node_count
        forecast = (144*segment_count*32 + segment_count*32768 + n*8192 +
                    structural_nnz_bound*profile["declared_fill_multiplier"]*profile["bytes_per_fill_entry"])
    if forecast > profile["resources"]["memory_budget_mb"]*1024**2:
        raise MemoryError("Sparse study factorization forecast exceeds payload budget")
    return forecast

def assemble_sparse(network, *, diameter_m, young_modulus_Pa, poisson_ratio, model, kappa=1., guard=None, profile_name=SMALL_PROFILE):
    """New sparse assembly; exact unchanged circular-section element operator."""
    order, positions, edges = checked_network(network, profile_name=profile_name)
    n, m = len(order)*6, len(edges)
    # Dense-fill LU workspace allowance; forecast, not an OS hard memory cap.
    forecast = allocation_forecast(len(order), m, profile_name)
    if guard:
        guard.check(forecast)
    np, sparse, _, reference = numeric_modules()
    section = reference.circular_properties(number(diameter_m, "diameter", positive=True)/2,
                                            young_modulus_Pa, poisson_ratio)
    index = {nid: i for i, nid in enumerate(order)}
    rows = np.empty(m*144, dtype=np.int32)
    cols = np.empty_like(rows)
    data = np.empty(m*144)
    elements = []
    for part, (sid, a, b) in enumerate(edges):
        if guard:
            guard.check(32768)
        stiffness = reference.element_stiffness(positions[a], positions[b], section, model=model, kappa=kappa)
        dofs = np.array([6*index[nid]+j for nid in (a, b) for j in range(6)])
        start = part*144
        rows[start:start+144] = np.repeat(dofs, 12)
        cols[start:start+144] = np.tile(dofs, 12)
        data[start:start+144] = stiffness.ravel()
        elements.append({"id": sid, "node_ids": [a, b], "dofs": dofs, "stiffness": stiffness,
                         "length_m": math.dist(positions[a], positions[b])})
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    matrix.eliminate_zeros()
    return matrix, order, elements, forecast

def solve_sparse(matrix, prescribed, *, omitted_dofs=(), forces=None, guard=None, characteristic_length_m=.1, profile_name=SMALL_PROFILE, factorization_forecast=None):
    """Symmetric diagonally scaled SuperLU; no pseudoinverse or weak springs."""
    np, sparse, splu, _ = numeric_modules()
    profile = study_profile(profile_name)
    n = matrix.shape[0]
    if matrix.shape != (n, n) or n % 6 or n > profile["DOFs"] or not np.isfinite(matrix.data).all():
        raise ValueError("Bounded finite square CSR matrix required")
    length_scale = number(characteristic_length_m, "residual moment length scale", positive=True)
    asymmetry = matrix-matrix.T
    if asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1e-12*max(float(np.max(np.abs(matrix.data))), 1e-30):
        raise ValueError("Conservative symmetric operator required")
    if not prescribed:
        raise ValueError("Prescribed DOFs required")
    for dof, value in prescribed.items():
        if type(dof) is not int or not 0 <= dof < n:
            raise ValueError("Literal bounded prescribed DOF required")
        number(value, "prescribed value")
    omitted = set(omitted_dofs)
    if any(type(i) is not int or not 0 <= i < n for i in omitted) or omitted & set(prescribed):
        raise ValueError("Explicit disjoint omitted isolate DOFs required")
    if omitted and (matrix[list(omitted), :].nnz or matrix[:, list(omitted)].nnz):
        raise ValueError("Only exact zero operator rows in the declared response slice can be omitted")
    f = np.zeros(n) if forces is None else np.asarray(forces, dtype=float)
    if f.shape != (n,) or not np.isfinite(f).all() or (omitted and np.any(f[list(omitted)] != 0)):
        raise ValueError("Finite loads with no isolate forcing required")
    constrained = np.array(sorted(prescribed), dtype=int)
    free = np.array([i for i in range(n) if i not in prescribed and i not in omitted], dtype=int)
    u = np.zeros(n)
    u[constrained] = [prescribed[int(i)] for i in constrained]
    if len(free):
        if factorization_forecast is not None and (type(factorization_forecast) is not int or factorization_forecast <= 0 or
                factorization_forecast > profile["resources"]["memory_budget_mb"]*1024**2):
            raise ValueError("Typed bounded factorization byte forecast required")
        if guard:
            if factorization_forecast is None:
                if profile_name != SMALL_PROFILE:
                    raise ValueError("Explicit large sparse factorization forecast required")
                factorization_forecast = 12*n*n*8
            guard.check(factorization_forecast)
        reduced = matrix[free, :][:, free].tocsc()
        diagonal = reduced.diagonal()
        if np.any(diagonal <= 0):
            raise ValueError("Unrestrained zero-stiffness DOF")
        scale = 1/np.sqrt(diagonal)
        normalized = sparse.diags(scale) @ reduced @ sparse.diags(scale)
        rhs = (f[free]-matrix[free, :][:, constrained]@u[constrained])*scale
        try:
            factor = splu(normalized.tocsc())
            u[free] = scale*factor.solve(rhs)
        except RuntimeError as error:
            raise ValueError("Singular finite frame; no pseudoinverse allowed") from error
    internal = matrix@u
    residual = internal-f
    translation = np.array([i for i in range(n) if i % 6 < 3], dtype=int)
    rotation = np.array([i for i in range(n) if i % 6 >= 3], dtype=int)
    force_scale = max(float(np.max(np.abs(internal[translation]))), float(np.max(np.abs(f[translation]))), 1e-30)
    moment_scale = max(float(np.max(np.abs(internal[rotation]))), float(np.max(np.abs(f[rotation]))), force_scale*length_scale, 1e-30)
    free_translation, free_rotation = free[free % 6 < 3], free[free % 6 >= 3]
    force_error = float(np.max(np.abs(residual[free_translation])))/force_scale if len(free_translation) else 0.
    moment_error = float(np.max(np.abs(residual[free_rotation])))/moment_scale if len(free_rotation) else 0.
    free_error = max(force_error, moment_error)
    if not np.isfinite(u).all() or not np.isfinite(residual).all() or free_error > 1e-8:
        raise ValueError("Finite free-equilibrium residual gate failed")
    if guard:
        guard.check()
    return u, residual, free, {"relative_free_residual": free_error, "relative_free_force_residual": force_error,
                              "relative_free_moment_residual": moment_error, "moment_residual_length_scale_m": length_scale,
                              "factorization": "scaled_sparse_SuperLU",
                              "positive_definiteness_numerically_certified": False}

"""Preparation-only boundary for an external sealed planar L3 provider.

No solver, launcher, provider import, remeshing or result parser is provided.
StructureGraph remains authoritative. See L3_OPTIONAL_BACKEND.md.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

__all__ = ["L3InputError", "L3ProviderIdentityError", "L3Binding", "L3Plan",
           "L3Resources", "L3ProviderSpec", "bind_l3", "prepare_l3"]


class L3InputError(ValueError):
    """An explicit input cannot be represented without changing semantics."""


class L3ProviderIdentityError(ValueError):
    """The external provider does not match the required immutable source seal."""


@dataclass(frozen=True)
class L3Binding:
    """Detached immutable input bytes, not another graph or a mechanics receipt."""
    canonical: bytes
    sha256: str


@dataclass(frozen=True)
class L3Plan:
    """Immutable preparation record; does not certify runtime or execution."""
    canonical: bytes
    sha256: str
    native_request_sha256: str
    status: str = "prepared_not_executable"


@dataclass(frozen=True)
class L3Resources:
    """Explicit future parent ceilings; only input size is enforced here."""
    max_nodes: int
    max_edges: int
    timeout_s: float
    total_rss_bytes: int
    output_bytes: int
    cpu_count: int


@dataclass(frozen=True)
class L3ProviderSpec:
    """Explicit external source and interpreter; no discovery or provisioning."""
    directory: str
    interpreter: str


_UNITS = {"m": 1.0, "mm": 1e-3, "um": 1e-6}
_MODEL = {"joint": "welded", "section": "circular",
          "constitutive": "isotropic_elastic_brittle",
          "loading": "clamped_positive_y", "contact": False}
_SEAL = {
    "acceptance_version": "L3_strengthCV05_portable_1.0.0",
    "python": "3.14.3", "numpy": "2.4.6", "scipy": "1.18.0", "psutil": "7.2.2",
    "files_sha256": {
        "run_case.py": "169eb510f4de1bd89ca8be65d70a7ec3a6551b1443692af8a41599ed646b018e",
        "lightweight_cv05/__init__.py": "c97a6366b27d95045623221fb4bdb6653bdfcfcf489a43fe151ca6905535397c",
        "lightweight_cv05/controller.py": "ab5354ceeff03e8f4fd8053ca73aa9829823fe6b751eb546fed2f66a3b91b381",
        "lightweight_cv05/provider.py": "e3a7df9f3b81b282141a7082283856b92af23a91b0b6edfeb7f2ac940dbd47b1",
        "lightweight_cv05/stability_gate.py": "406f21fafb5e67ccfe0509f2d434938214980dca4289519c9707847abff001d2",
        "lightweight_cv05/runtime.py": "c78d1c1414fa01885d2bdfe362a6a0f9997de24bdd69941d6a09cec8084e2da9",
    },
}
_POSITIVE_SETTINGS = {
    "young_Pa", "radius_m", "shear_correction", "load_increment_m",
    "minimum_increment_m", "event_displacement_tolerance_m",
    "max_displacement_m", "residual_tolerance", "timeout_s",
}
_REQUIRED_SETTINGS = _POSITIVE_SETTINGS | {
    "poisson", "post_rupture_opening_m", "event_bisection_cap", "frame_cap",
    "notched", "material_source", "failure_parameter_status",
    "memory_budget_mb", "reserve_memory_mb",
}
_OPTIONAL_SETTINGS = {"strength_field_mean", "strength_field_cv"}
_ALIASES = {"memory_budget_MiB": "memory_budget_mb", "reserve_MiB": "reserve_memory_mb"}
_EDGE_REQUIRED = {"failure_strain", "physical_span_id", "source_edge_id",
                  "initially_active", "reference_length_m"}
_EDGE_OPTIONAL = {"native_lineage", "parent_fiber_id", "source_named_path_id", "provenance"}


def _error(message):
    raise L3InputError(message)


def _number(value, label, *, positive=False, nonnegative=False):
    if type(value) not in (int, float):
        _error(label + " must be a literal finite number, not bool or a scalar wrapper")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (positive and value <= 0) or (nonnegative and value < 0):
        _error(label + " has an invalid finite range")
    return value


def _graph_number(value, label, *, positive=False):
    """Normalize graph-native real scalars without lossy longdouble coercion."""
    import numpy as np

    if isinstance(value, np.generic) and value.dtype.kind not in "iuf":
        _error(label + " must have a real numeric NumPy scalar dtype")
    if isinstance(value, np.integer) and not isinstance(value, np.bool_):
        value = int(value)
    elif isinstance(value, np.floating):
        native = float(value)
        _number(native, label, positive=positive)
        if type(value)(native) != value:
            _error(label + " cannot be represented losslessly as a Python float")
        value = native
    return _number(value, label, positive=positive)


def _integer(value, label, *, positive=False):
    if type(value) is not int or not (1 if positive else 0) <= value < 2**63:
        _error(label + " must be a literal nonnegative signed-int64 integer")
    return value


def _text(value, label):
    if type(value) is not str or not value.strip():
        _error(label + " must be nonempty text")
    return value


def _json_value(value, depth=0):
    if depth > 80:
        _error("JSON input is cyclic or exceeds 80 nesting levels")
    if value is None or type(value) in (str, bool):
        return
    if type(value) in (int, float):
        _number(value, "JSON number")
        return
    if type(value) is list:
        for item in value:
            _json_value(item, depth + 1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_value(item, depth + 1)
        return
    _error("Only finite literal JSON values with string object keys are accepted")


def _canonical(value):
    _json_value(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _copy(value):
    return json.loads(_canonical(value))


def _ids(values, label):
    if type(values) not in (list, tuple) or not values:
        _error(label + " requires explicit nonempty IDs")
    for value in values:
        _integer(value, label)
    if len(set(values)) != len(values):
        _error(label + " contains duplicate IDs")
    return sorted(values)


def _scaled(value, scale, label, *, positive=False):
    _number(value, label, positive=positive)
    result = value * scale
    _number(result, label + " in SI", positive=positive)
    if value != 0 and result == 0:
        _error(label + " underflows during SI conversion")
    return result


def _flags(metadata):
    if type(metadata) is not dict:
        _error("metadata must be a JSON object")
    _json_value(metadata)
    hints = {"joint": "welded", "joint_type": "welded", "section": "circular",
             "section_type": "circular", "contact": False,
             "loading": "clamped_positive_y", "load_axis": "Y",
             "constitutive": "isotropic_elastic_brittle"}
    for key, expected in hints.items():
        if key in metadata and (type(metadata[key]) is not type(expected) or metadata[key] != expected):
            _error("Unsupported mechanical metadata: " + key)


def _graph_record(graph, coordinate_unit, radius_unit, model):
    # No optional engine/provider import; the existing graph is the only model.
    from fibernet.core.structure_graph import StructureGraph, SNode, SEdge
    from fibernet.core.material import Material
    if not isinstance(graph, StructureGraph):
        raise TypeError("graph must be the existing StructureGraph")
    if type(coordinate_unit) is not str or coordinate_unit not in _UNITS:
        _error("coordinate_unit must explicitly be m, mm or um")
    if type(radius_unit) is not str or radius_unit not in _UNITS:
        _error("radius_unit must explicitly be m, mm or um")
    if _canonical(model) != _canonical(_MODEL):
        _error("Explicit restricted L3 model declaration required")
    if type(graph.dimension) is not int or graph.dimension != 2:
        _error("L3 requires dimension=2, XY planar geometry")
    _flags(graph.metadata)
    for key, value in (("coordinate_units", coordinate_unit), ("coordinate_unit", coordinate_unit),
                       ("radius_units", radius_unit), ("radius_unit", radius_unit)):
        if key in graph.metadata and graph.metadata[key] != value:
            _error("Graph metadata unit conflict: " + key)
    if type(graph.nodes) is not dict or type(graph.edges) is not dict:
        _error("Graph node and edge stores must be dictionaries")
    if not graph.nodes or not graph.edges:
        _error("Nonempty graph required")
    nodes, edges, xy = [], [], {}
    for key, node in graph.nodes.items():
        _integer(key, "node dictionary ID")
        if type(node) is not SNode:
            _error("Graph node store must contain SNode objects")
        if type(node.node_id) is not int or node.node_id != key:
            _error("Node object/dictionary ID mismatch")
        p = node.position
        if (type(p).__module__ != "numpy" or type(p).__name__ != "ndarray"
                or p.shape != (3,) or p.dtype.kind not in "iuf"):
            _error("Node position must be an unmasked real numeric ndarray of shape (3,)")
        values = p.tolist()
        for v in values:
            _number(v, "coordinate")
        if values[2] != 0:
            _error("Nonzero Z cannot be flattened")
        _flags(node.metadata)
        if (type(node.boundary) not in (tuple, list) or len(node.boundary) != 6
                or any(type(v) is not bool for v in node.boundary)):
            _error("Node boundary flags must be six literal booleans")
        point = [_scaled(v, _UNITS[coordinate_unit], "coordinate") for v in values[:2]]
        xy[key] = point
        nodes.append({"id": key, "position": point})
    common = None
    for key, edge in graph.edges.items():
        _integer(key, "edge dictionary ID")
        if type(edge) is not SEdge:
            _error("Graph edge store must contain SEdge objects")
        if type(edge.edge_id) is not int or edge.edge_id != key:
            _error("Edge object/dictionary ID mismatch")
        for endpoint in (edge.node_i, edge.node_j):
            _integer(endpoint, "endpoint ID")
            if endpoint not in xy:
                _error("Unknown endpoint")
        if edge.node_i == edge.node_j:
            _error("Self edges unsupported")
        if type(edge.segments) is not int or edge.segments != 1:
            _error("Each graph edge must already be one physical segment (segments=1)")
        if edge.internal_points is not None:
            p = edge.internal_points
            if (type(p).__module__ != "numpy" or type(p).__name__ != "ndarray"
                    or p.shape != (0, 3) or p.dtype.kind not in "iuf"):
                _error("Nonempty or malformed internal_points unsupported; no chord reduction")
        _flags(edge.metadata)
        mat = edge.material
        if type(mat) is not Material or mat.stiffness_tensor is not None:
            _error("An explicit isotropic Material is required")
        young = _graph_number(mat.youngs_modulus, "youngs_modulus", positive=True)
        nu = _graph_number(mat.poissons_ratio, "poissons_ratio")
        if not -1 < nu < 0.5:
            _error("Poisson ratio outside (-1, .5)")
        shear = _graph_number(mat.shear_modulus, "shear_modulus", positive=True)
        expected_shear = young / (2 * (1 + nu))
        if not math.isfinite(expected_shear) or not math.isclose(shear, expected_shear, rel_tol=1e-12, abs_tol=0):
            _error("Shear modulus differs from isotropic E/[2(1+nu)]")
        source_radius = _graph_number(edge.radius, "radius", positive=True)
        radius = _scaled(source_radius, _UNITS[radius_unit], "radius", positive=True)
        profile = (young, nu, radius)
        if common is not None and profile != common:
            _error("Nonuniform E/nu/radius unsupported")
        common = profile
        length = math.dist(xy[edge.node_i], xy[edge.node_j])
        _number(length, "SI reference centerline length", positive=True)
        edges.append({"id": key, "node_i": edge.node_i, "node_j": edge.node_j,
                      "radius": radius, "rest_length": length})
    # Raw object validation precedes serialization (which can coerce flags/IDs).
    serialized = graph.to_dict()
    # Serialization keeps ndarray-native radius values unchanged. Convert only
    # these already validated scalars in the detached snapshot, never the graph.
    for edge in serialized["edges"]:
        edge["radius"] = _graph_number(edge["radius"], "serialized radius", positive=True)
    serialized = _copy(serialized)
    return {"graph": serialized, "graph_sha256": _digest(_canonical(serialized)),
            "nodes": sorted(nodes, key=lambda n: n["id"]),
            "segments": sorted(edges, key=lambda e: e["id"]),
            "young_Pa": common[0], "poisson": common[1], "radius_m": common[2]}


def bind_l3(graph, *, coordinate_unit, radius_unit, edge_data,
            fixed_node_ids, loaded_node_ids, model):
    """Bind explicit strain, span, grips and units to an unchanged StructureGraph.

    No thresholds, topology, materials or grips are inferred. Native lineage is
    optional for synthetic graphs; if supplied it is retained byte-semantically
    as canonical JSON and checked against corresponding explicit fields.
    """
    record = _graph_record(graph, coordinate_unit, radius_unit, model)
    node_ids = {n["id"] for n in record["nodes"]}
    edge_ids = {e["id"] for e in record["segments"]}
    fixed, loaded = _ids(fixed_node_ids, "fixed grip"), _ids(loaded_node_ids, "loaded grip")
    if set(fixed) & set(loaded) or not (set(fixed) | set(loaded)) <= node_ids:
        _error("Grips must be disjoint known nodes")
    if (type(edge_data) is not dict or any(type(k) is not int for k in edge_data)
            or set(edge_data) != edge_ids):
        _error("edge_data must exactly cover stable edge IDs")
    associated, spans = [], {}
    for edge in record["segments"]:
        data = _copy(edge_data[edge["id"]])
        if type(data) is not dict or not _EDGE_REQUIRED <= set(data) or set(data) - _EDGE_REQUIRED - _EDGE_OPTIONAL:
            _error("Each edge needs explicit strain/span/source/active/reference length")
        threshold = _number(data["failure_strain"], "failure_strain", positive=True)
        span = data["physical_span_id"]
        if type(span) is int:
            _integer(span, "physical_span_id")
        else:
            _text(span, "physical_span_id")
        typed_span = (type(span), span)
        if typed_span in spans and spans[typed_span] != threshold:
            _error("Conflicting thresholds within one physical span")
        spans[typed_span] = threshold
        _integer(data["source_edge_id"], "source_edge_id")
        if type(data["initially_active"]) is not bool:
            _error("initially_active must be a literal bool")
        rest = _number(data["reference_length_m"], "reference_length_m", positive=True)
        if not math.isclose(rest, edge["rest_length"], rel_tol=1e-8, abs_tol=1e-13):
            _error("Associated reference length differs from graph")
        lineage = data.get("native_lineage")
        if lineage is not None:
            if type(lineage) is not dict:
                _error("native_lineage must be a JSON object")
            if "strength_span_id" in lineage and (type(lineage["strength_span_id"]) is not type(span) or lineage["strength_span_id"] != span):
                _error("Native span identity mismatch")
            if "initial_kerf_inactive" in lineage and (type(lineage["initial_kerf_inactive"]) is not bool or lineage["initial_kerf_inactive"] == data["initially_active"]):
                _error("Native inactive mapping mismatch")
        associated.append({"edge_id": edge["id"], "data": data})
    if not any(item["data"]["initially_active"] for item in associated):
        _error("Initially active material required")
    payload = {"schema": "fibernet.l3.binding/1", "record": record,
               "coordinate_unit": coordinate_unit, "radius_unit": radius_unit,
               "coordinate_scale_to_m": _UNITS[coordinate_unit],
               "radius_scale_to_m": _UNITS[radius_unit], "model": _copy(model),
               "edge_data": associated, "fixed_node_ids": fixed, "loaded_node_ids": loaded}
    raw = _canonical(payload)
    return L3Binding(raw, _digest(raw))


def _regular_path(value, label, *, directory=False):
    try:
        if type(value) is not str or not value.strip() or "\x00" in value:
            raise L3ProviderIdentityError(label + " must be nonempty path text")
        p = Path(value)
        if not p.is_absolute() or any(part == ".." for part in p.parts):
            raise L3ProviderIdentityError(label + " must be an explicit absolute path")
        if any(item.is_symlink() for item in (p, *p.parents)):
            raise L3ProviderIdentityError(label + " may not traverse symlinks")
        if not (p.is_dir() if directory else p.is_file()):
            raise L3ProviderIdentityError(label + " is absent or not regular")
        return p
    except (OSError, RuntimeError) as exc:
        raise L3ProviderIdentityError(label + " cannot be inspected") from exc


def _provider_record(spec):
    if type(spec) is not L3ProviderSpec:
        raise TypeError("provider must be L3ProviderSpec")
    root = _regular_path(spec.directory, "provider directory", directory=True)
    interpreter = _regular_path(spec.interpreter, "interpreter")
    seal_path = _regular_path(str(root / "SOURCE_IDENTITY.json"), "provider seal")
    if seal_path.stat().st_size > 16384:
        raise L3ProviderIdentityError("Oversized provider seal")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate provider seal key")
            result[key] = value
        return result

    try:
        seal = json.loads(seal_path.read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object)
        seal_raw = _canonical(seal)
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        raise L3ProviderIdentityError("Unreadable or invalid provider seal") from exc
    if seal_raw != _canonical(_SEAL):
        raise L3ProviderIdentityError("Provider seal differs from required release")
    for name, expected in _SEAL["files_sha256"].items():
        path = _regular_path(str(root / name), "sealed source")
        if path.stat().st_size > 4 * 1024**2:
            raise L3ProviderIdentityError("Oversized sealed source")
        if _digest(path.read_bytes()) != expected:
            raise L3ProviderIdentityError("Changed sealed source: " + name)
    return {"directory": str(root), "interpreter": str(interpreter),
            "seal": seal, "source_verified": True, "runtime_verified": False,
            "execution_qualified": False}


def _provider(spec):
    try:
        return _provider_record(spec)
    except OSError as exc:
        raise L3ProviderIdentityError("Provider identity files became unreadable") from exc


def _settings(settings, record, resources):
    original = _copy(settings)
    if type(original) is not dict:
        _error("settings must be an object")
    normalized = _copy(original)
    for alias, canonical in _ALIASES.items():
        if alias in normalized:
            _integer(normalized[alias], alias, positive=True)
            if canonical in normalized:
                _integer(normalized[canonical], canonical, positive=True)
                if normalized[canonical] != normalized[alias]:
                    _error("Conflicting memory alias: " + alias)
            normalized[canonical] = normalized.pop(alias)
    if not _REQUIRED_SETTINGS <= set(normalized) or set(normalized) - _REQUIRED_SETTINGS - _OPTIONAL_SETTINGS:
        _error("Missing or unknown L3 settings; provider defaults are not used")
    for key in _POSITIVE_SETTINGS:
        _number(normalized[key], key, positive=True)
    for key in ("memory_budget_mb", "reserve_memory_mb", "frame_cap", "event_bisection_cap"):
        _integer(normalized[key], key, positive=True)
    _number(normalized["post_rupture_opening_m"], "post_rupture_opening_m", nonnegative=True)
    _number(normalized["poisson"], "poisson")
    for key in ("young_Pa", "poisson", "radius_m"):
        if normalized[key] != record[key]:
            _error("Settings cannot override graph material/section: " + key)
    if type(normalized["notched"]) is not bool:
        _error("notched must be literal bool")
    if type(normalized["material_source"]) is not dict:
        _error("material_source must be an explicit JSON object")
    _text(normalized["failure_parameter_status"], "failure_parameter_status")
    if normalized["minimum_increment_m"] > normalized["load_increment_m"]:
        _error("minimum increment exceeds load increment")
    if normalized["load_increment_m"] > normalized["max_displacement_m"]:
        _error("load increment exceeds maximum displacement")
    if normalized["timeout_s"] > resources.timeout_s:
        _error("Native timeout exceeds explicit parent ceiling")
    for key in _OPTIONAL_SETTINGS & set(normalized):
        _number(normalized[key], key, positive=(key == "strength_field_mean"), nonnegative=True)
    return original, normalized


def prepare_l3(graph, binding, *, case, settings, acceptance, resources, provider):
    """Validate and prepare immutable data; never execute or import the provider.

    The source hash check does not verify the supplied interpreter runtime.
    Canonical bytes include private paths/provenance: review before sharing.
    """
    _text(case, "case")
    if type(binding) is not L3Binding or type(binding.canonical) is not bytes:
        raise TypeError("binding must be L3Binding")
    if _digest(binding.canonical) != binding.sha256:
        _error("Binding integrity mismatch")
    try:
        b = json.loads(binding.canonical)
        rebuilt = bind_l3(graph, coordinate_unit=b["coordinate_unit"], radius_unit=b["radius_unit"],
                          edge_data={item["edge_id"]: item["data"] for item in b["edge_data"]},
                          fixed_node_ids=b["fixed_node_ids"], loaded_node_ids=b["loaded_node_ids"], model=b["model"])
    except (KeyError, TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise L3InputError("Invalid or stale L3 binding") from exc
    if rebuilt.canonical != binding.canonical:
        _error("Graph or bound data changed; explicit rebinding required")
    if type(resources) is not L3Resources:
        raise TypeError("resources must be L3Resources")
    limits = dict(vars(resources))
    for key in ("max_nodes", "max_edges", "total_rss_bytes", "output_bytes", "cpu_count"):
        _integer(limits[key], key, positive=True)
    _number(limits["timeout_s"], "parent timeout_s", positive=True)
    record = b["record"]
    if len(record["nodes"]) > resources.max_nodes or len(record["segments"]) > resources.max_edges:
        _error("Input exceeds explicit node/edge ceiling")
    original, normalized = _settings(settings, record, resources)
    a = _copy(acceptance)
    inactive = [item["edge_id"] for item in b["edge_data"] if not item["data"]["initially_active"]]
    if type(a) is not dict or "initial_inactive_segment_ids" not in a:
        _error("acceptance requires explicit initial_inactive_segment_ids")
    got = a["initial_inactive_segment_ids"]
    if type(got) is not list or any(type(i) is not int for i in got) or sorted(got) != inactive:
        _error("Acceptance inactive IDs differ from bound physical segments")
    if inactive and not normalized["notched"]:
        _error("notched=False would reactivate bound inactive segments")
    segments = []
    for edge, item in zip(record["segments"], b["edge_data"]):
        segment = dict(edge, failure_strain=item["data"]["failure_strain"])
        # Lineage remains authoritative in the binding, not silently interpreted.
        segments.append(segment)
    inactive_set = set(inactive)
    active = [e for e in segments if e["id"] not in inactive_set]
    try:
        total = math.fsum(e["rest_length"] for e in active)
        _number(total, "active reference length", positive=True)
        weights = [e["rest_length"] / total for e in active]
        if any(w == 0 for w in weights):
            _error("Active reference-length weights underflow")
        mean = math.fsum(w * e["failure_strain"] for w, e in zip(weights, active))
        _number(mean, "active-length mean threshold", positive=True)
        # Normalize before squaring; avoid squaring a large dimensional threshold.
        variance_ratio = math.fsum(w * (e["failure_strain"] / mean - 1)**2
                                   for w, e in zip(weights, active))
        cv = math.sqrt(variance_ratio)
    except (OverflowError, ZeroDivisionError) as exc:
        raise L3InputError("Strength statistics exceed finite numeric range") from exc
    for value in (mean, cv):
        _number(value, "active-length strength statistic", nonnegative=True)
    for key, actual, relative, absolute in (("strength_field_mean", mean, 1e-9, 1e-12),
                                            ("strength_field_cv", cv, 1e-7, 1e-10)):
        if key in normalized and not math.isclose(normalized[key], actual, rel_tol=relative, abs_tol=absolute):
            _error("Declared " + key + " differs from actual post-notch length-weighted field")
    request = {"case": case, "network": {"nodes": record["nodes"], "segments": segments},
               "boundary": {"fixed_nodes": b["fixed_node_ids"], "loaded_nodes": b["loaded_node_ids"]},
               "settings": normalized, "acceptance": a}
    native_digest = _digest(_canonical({key: request[key] for key in ("network", "boundary", "settings", "acceptance")}))
    payload = {"schema": "fibernet.l3.plan/1", "status": "prepared_not_executable",
               "binding": b, "binding_sha256": binding.sha256, "case": case,
               "original_settings": original, "normalized_settings": normalized,
               "resources": limits, "resource_enforcement": "input_size_only_no_worker",
               "provider": _provider(provider), "request": request,
               "native_request_sha256": native_digest,
               "node_ids": [n["id"] for n in record["nodes"]],
               "edge_ids": [e["id"] for e in record["segments"]],
               "strength_active_length_weighted_mean": mean,
               "strength_active_length_weighted_cv": cv}
    raw = _canonical(payload)
    return L3Plan(raw, _digest(raw), native_digest)

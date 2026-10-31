"""Geometry-only interchange with the R10 source-fibre release.

The StructureGraph is authoritative. A bound sidecar retains the original
fused-domain data; it is not a second mutable graph or a solver-ready mesh.
"""
from __future__ import annotations

import hashlib
import json
import math

from fibernet.core.material import Material
from fibernet.core.structure_graph import StructureGraph, _material_record

SCHEMA = "FiberNet_original100_source_graph_material_v1"
_BINDING = "fibernet.r10.geometry_binding.v1"
_PROFILE = {"C10_MPa": 6.0, "C01_MPa": 3.0, "D1_per_MPa": 0.495}


def _error(message):
    raise ValueError("R10 geometry: " + message)


def _json_copy(value):
    # Reject implicit tuple/NumPy conversion, non-string keys and nonfinite
    # extension fields. Unknown JSON fields are retained rather than discarded.
    def check(item, depth=0):
        if depth > 80:
            _error("JSON nesting exceeds 80 levels")
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for v in item:
                check(v, depth + 1)
            return
        if type(item) is dict and all(type(k) is str for k in item):
            for v in item.values():
                check(v, depth + 1)
            return
        _error("expected finite JSON data with string keys")
    check(value)
    return json.loads(json.dumps(value, allow_nan=False))


def _number(value, label, positive=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(float(value))
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        _error(label + " must be finite numeric data (not boolean)")
    if positive and value <= 0:
        _error(label + " must be positive")
    return value


def _ids(values, count, label):
    if (type(values) is not list or len(values) != count or
            any(type(v) is not int or not 0 <= v < 2 ** 63 - 1 for v in values) or
            len(set(values)) != count):
        _error(label + " must be unique nonnegative int64 identities with room for next-ID")


def _point(value, label):
    if type(value) is not list or len(value) != 2:
        _error(label + " must be an XY pair")
    for v in value:
        _number(v, label)


def _ring(value, label):
    if type(value) is not list or len(value) < 4:
        _error(label + " requires a closed ring with at least three vertices")
    for p in value:
        _point(p, label)
    if value[0] != value[-1] or len({tuple(p) for p in value[:-1]}) < 3:
        _error(label + " must be closed with three distinct vertices")


def validate_r10_geometry(payload, *, length_unit):
    """Return a detached JSON record after structural, not geometric, checks.

    Only mm source coordinates and the original Y/1-mm/no-contact profile are
    supported. This does NOT establish polygon validity, material connectivity,
    source-to-outline membership, meshing, or mechanical validity.
    """
    if length_unit != "mm":
        _error("explicit length_unit='mm' is required; no implicit conversion")
    g = _json_copy(payload)
    if type(g) is not dict or g.get("schema") != SCHEMA:
        _error("unsupported source schema")
    required = ("case_id", "nodes_xy_mm", "node_ids", "edges", "edge_ids",
                "parent_fiber_ids", "source_edge_ids", "source_named_path_ids",
                "outline", "grips", "material", "load_axis",
                "reference_thickness_mm", "nominal_gauge_mm", "self_contact")
    if any(k not in g for k in required):
        _error("missing required source/profile/identity field")
    if not isinstance(g["case_id"], str) or not g["case_id"]:
        _error("case_id must be nonempty text")
    xy, edges = g["nodes_xy_mm"], g["edges"]
    if type(xy) is not list or not xy or type(edges) is not list or not edges:
        _error("nonempty node and edge lists required")
    for p in xy:
        _point(p, "nodes_xy_mm")
    _ids(g["node_ids"], len(xy), "node_ids")
    _ids(g["edge_ids"], len(edges), "edge_ids")
    for e in edges:
        if (type(e) is not list or len(e) != 2 or
                any(type(i) is not int or not 0 <= i < len(xy) for i in e)):
            _error("edges must contain two in-range row indices, not source IDs")
        if e[0] == e[1] or xy[e[0]] == xy[e[1]]:
            _error("self-loop or zero-length source edge")
    for key in ("parent_fiber_ids", "source_edge_ids", "source_named_path_ids"):
        if type(g[key]) is not list or len(g[key]) != len(edges):
            _error(key + " must have one identity record per source edge")
    if type(g["material"]) is not dict or set(g["material"]) != set(_PROFILE):
        _error("unsupported material profile fields")
    for key, expected in _PROFILE.items():
        if _number(g["material"][key], key) != expected:
            _error("unsupported Mooney-Rivlin profile")
    if (g["load_axis"] != "Y" or
            _number(g["reference_thickness_mm"], "reference_thickness_mm") != 1 or
            g["self_contact"] is not False):
        _error("only Y loading, 1 mm reference thickness and no contact supported")
    _number(g["nominal_gauge_mm"], "nominal_gauge_mm", positive=True)
    outline = g["outline"]
    if type(outline) is not dict or "exterior" not in outline or type(outline.get("holes")) is not list:
        _error("explicit outline exterior and holes required; no inferred outline")
    _ring(outline["exterior"], "outline.exterior")
    for hole in outline["holes"]:
        _ring(hole, "outline.holes")
    grips = g["grips"]
    if type(grips) is not dict:
        _error("explicit grips required")
    for side in ("bottom", "top"):
        grip = grips.get(side)
        segments = grip.get("segments_xy_mm") if type(grip) is dict else None
        if type(segments) is not list or not segments:
            _error(side + " grip requires physical segments_xy_mm")
        for segment in segments:
            if type(segment) is not list or len(segment) != 2:
                _error("grip segment must have two XY endpoints")
            for p in segment:
                _point(p, "grip segment")
            if segment[0] == segment[1]:
                _error("zero-length grip segment")
    return g


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def import_r10_geometry(payload, *, edge_properties, length_unit):
    """Return ``(StructureGraph, binding)`` with explicit beam annotations.

    edge_properties maps every stable edge ID to radius (mm), Material (SI
    documented units), and positive integer segments. These annotations are NOT
    inferred from the fixed continuum profile and do not configure its solver.
    """
    source = validate_r10_geometry(payload, length_unit=length_unit)
    if (type(edge_properties) is not dict or
            any(type(k) is not int for k in edge_properties) or
            set(edge_properties) != set(source["edge_ids"])):
        _error("edge_properties must exactly cover stable edge IDs")
    nodes = [{"id": nid, "position": [*p, 0.0]} for nid, p in
             zip(source["node_ids"], source["nodes_xy_mm"])]
    edges, retained = [], {}
    for eid, endpoints in zip(source["edge_ids"], source["edges"]):
        props = edge_properties[eid]
        if type(props) is not dict or set(props) != {"radius", "material", "segments"}:
            _error("each edge requires explicit radius, material and segments")
        radius = _number(props["radius"], "radius", positive=True)
        segments = props["segments"]
        if type(segments) is not int or segments < 1:
            _error("segments must be a positive integer")
        if not isinstance(props["material"], Material):
            _error("material must be an explicit Material instance")
        material = _material_record(props["material"].to_dict())
        retained[str(eid)] = dict(radius=radius, material=material, segments=segments)
        edges.append(dict(id=eid, node_i=source["node_ids"][endpoints[0]],
                          node_j=source["node_ids"][endpoints[1]], **retained[str(eid)]))
    graph = StructureGraph.from_dict(dict(version="3.1", dimension=2, nodes=nodes, edges=edges,
                                         metadata={"coordinate_units": "mm",
                                                   "r10_geometry_only": True}))
    binding = dict(schema=_BINDING, length_unit="mm", source=source,
                   source_sha256=_digest(source), graph_sha256=_digest(graph.to_dict()),
                   edge_properties=retained, status="geometry_only_not_solver_ready")
    return graph, _json_copy(binding)


def export_r10_geometry(graph, binding, *, length_unit):
    """Invert an unchanged import; NOT an arbitrary StructureGraph exporter.

    Returns the original source record. Keep the binding's edge_properties
    separately: bare R10 has no beam Material/radius/segments fields. Edits need
    explicit fused-domain rebuilding and reimport, never a stale sidecar.
    """
    if not isinstance(graph, StructureGraph):
        raise TypeError("graph must be the existing StructureGraph")
    b = _json_copy(binding)
    if (type(b) is not dict or b.get("schema") != _BINDING or
            b.get("length_unit") != length_unit or length_unit != "mm"):
        _error("invalid geometry binding or unit mismatch")
    source = validate_r10_geometry(b.get("source"), length_unit=length_unit)
    if b.get("source_sha256") != _digest(source):
        _error("associated source changed; explicit revalidation/reimport required")
    if graph.dimension != 2 or any(n.position[2] != 0 for n in graph.nodes.values()):
        _error("3D/nonplanar graphs are unsupported")
    if any(e.internal_points is not None and len(e.internal_points) for e in graph.edges.values()):
        _error("curved/discretized edges unsupported; no silent flattening")
    if b.get("graph_sha256") != _digest(graph.to_dict()):
        _error("graph changed; bound domain/profile/beam annotations cannot be silently reused")
    props = {}
    records = b.get("edge_properties")
    if type(records) is not dict:
        _error("retained beam annotations must be an object")
    for key, value in records.items():
        if (not key.isdecimal() or not key.isascii() or str(int(key)) != key or
                type(value) is not dict or set(value) != {"radius", "material", "segments"}):
            _error("invalid retained beam annotations")
        try:
            material = Material.from_dict(_material_record(value["material"]))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError("R10 geometry: invalid retained Material record") from exc
        props[int(key)] = dict(value, material=material)
    rebuilt, _ = import_r10_geometry(source, edge_properties=props, length_unit=length_unit)
    if _digest(rebuilt.to_dict()) != b["graph_sha256"]:
        _error("retained beam annotations do not match graph binding")
    return source

"""Strict native NetworkX interchange for the existing StructureGraph.

This module does not change the legacy analysis projection. Native v1 uses
one namespaced record per entity, never a hidden whole-graph snapshot.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from fibernet.core.material import Material
from fibernet.core.structure_graph import _material_record


_NS = "_fibernet"
_FORMAT = "fibernet.structure_graph.networkx"
_HEADER = {"format", "version", "structure_graph_version", "dimension",
           "tolerance", "box_size", "next_node_id", "next_edge_id",
           "node_order", "edge_order", "metadata"}
_NODE = {"id", "position", "boundary", "metadata"}
_EDGE = {"id", "node_i", "node_j", "radius", "segments", "material",
         "internal_points", "metadata"}
_MATERIAL_REQUIRED = {"name", "density", "youngs_modulus", "poissons_ratio",
                      "shear_modulus"}


def _error(label, detail):
    raise ValueError(f"native NetworkX {label}: {detail}")


def _unmasked(value, label):
    if np.ma.isMaskedArray(value):
        _error(label, "masked arrays/scalars are unsupported, including all-false masks")


def _fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != expected:
        _error(label, "missing or unknown fields")


def _record(attrs, fields, label):
    _fields(attrs, {_NS}, label + " attributes")
    record = attrs[_NS]
    _fields(record, fields, label + " record")
    return record


def _integer(value, label):
    _unmasked(value, label)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        _error(label, "must be a nonnegative integer, excluding bool")
    return int(value)


def _number(value, label):
    _unmasked(value, label)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        _error(label, "must be a finite real number, excluding bool")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        _error(label, "must be a finite real number")
    if not math.isfinite(result):
        _error(label, "must be a finite real number")
    # Python-int comparison avoids NumPy's mixed uint64/float comparison
    # rounding. Wider floating values must already be exactly float64-valued.
    original = int(value) if isinstance(value, Integral) else value
    if original != result:
        _error(label, "cannot be represented exactly as float64")
    return result


def _json(value, label, depth=0, ancestors=None):
    """Copy JSON data without arbitrary object conversion or shared containers."""
    _unmasked(value, label)
    if depth > 100:
        _error(label, "JSON nesting exceeds 100 levels")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _error(label, "JSON numbers must be finite")
        return value
    if not isinstance(value, (dict, list)):
        _error(label, "requires JSON values; unsupported object or container")
    ancestors = set() if ancestors is None else ancestors
    if id(value) in ancestors:
        _error(label, "cyclic JSON data")
    ancestors.add(id(value))
    try:
        if isinstance(value, list):
            return [_json(v, label, depth + 1, ancestors) for v in value]
        if any(type(k) is not str for k in value):
            _error(label, "JSON dictionary keys must be strings")
        return {k: _json(v, label, depth + 1, ancestors) for k, v in value.items()}
    finally:
        ancestors.remove(id(value))


def _metadata(value, label):
    if not isinstance(value, dict):
        _error(label, "metadata must be a JSON object")
    return _json(value, label)


def _vector(value, label):
    if not isinstance(value, list) or len(value) != 3:
        _error(label, "must be a three-coordinate list")
    return [_number(v, label) for v in value]


def _order(value, label):
    if not isinstance(value, list):
        _error(label, "must be an ID list")
    order = [_integer(v, label) for v in value]
    if len(set(order)) != len(order):
        _error(label, "duplicate IDs")
    return order


def _header(record):
    _fields(record, _HEADER, "graph record")
    if (record["format"] != _FORMAT or type(record["version"]) is not int
            or record["version"] != 1 or record["structure_graph_version"] != "3.1"):
        _error("schema", "unsupported format or version")
    dimension = _integer(record["dimension"], "dimension")
    if dimension not in (2, 3):
        _error("dimension", "must be 2 or 3")
    tolerance = _number(record["tolerance"], "tolerance")
    if tolerance <= 0:
        _error("tolerance", "must be positive")
    return {"version": "3.1", "dimension": dimension,
            "tolerance": tolerance,
            "box_size": None if record["box_size"] is None else _vector(record["box_size"], "box_size"),
            "next_node_id": _integer(record["next_node_id"], "next_node_id"),
            "next_edge_id": _integer(record["next_edge_id"], "next_edge_id"),
            "metadata": _metadata(record["metadata"], "graph metadata")}


def _node(record, key):
    _fields(record, _NODE, "node record")
    nid = _integer(record["id"], "node record ID")
    if nid != _integer(key, "node key"):
        _error("node identity", "key differs from record ID; relabeling is unsupported")
    boundary = record["boundary"]
    if (not isinstance(boundary, list) or len(boundary) != 6
            or any(type(v) is not bool for v in boundary)):
        _error("boundary", "must be a list of six bools")
    return {"id": nid, "position": _vector(record["position"], "position"),
            "boundary": boundary.copy(), "metadata": _metadata(record["metadata"], "node metadata")}


def _edge(record, key, u, v):
    _fields(record, _EDGE, "edge record")
    eid = _integer(record["id"], "edge record ID")
    if eid != _integer(key, "edge key"):
        _error("edge identity", "key differs from record ID")
    i, j = (_integer(record[k], k) for k in ("node_i", "node_j"))
    if i == j:
        _error("topology", "self-loops are unsupported")
    if {i, j} != {_integer(u, "edge endpoint"), _integer(v, "edge endpoint")}:
        _error("topology", "live incidence differs from stored ordered endpoints")
    points = record["internal_points"]
    if points is not None:
        if not isinstance(points, list):
            _error("internal_points", "must be a list of three-coordinate lists or null")
        points = [_vector(p, "internal_points") for p in points]
    material = _json(record["material"], "material")
    if not isinstance(material, dict) or not _MATERIAL_REQUIRED.issubset(material):
        _error("material", "must contain the complete schema-3.1 core fields")
    # The shared material validator normalizes numbers to float64. First
    # reject values for which that normalization would change numeric value.
    for name, value in material.items():
        if name not in ("name", "extra", "stiffness_tensor") and value is not None:
            _number(value, f"material {name}")
    tensor = material.get("stiffness_tensor")
    if isinstance(tensor, list):
        for row in tensor:
            if isinstance(row, list):
                for value in row:
                    _number(value, "material stiffness_tensor")
    try:
        material = _material_record(material)
    except (ValueError, TypeError, OverflowError) as exc:
        _error("material", str(exc))
    # Material(None shear) computes a new value on construction. Native v1 must
    # reject that unrepresentable state rather than silently change it.
    if material["shear_modulus"] is None:
        _error("material", "null shear_modulus cannot roundtrip without recomputation")
    return {"id": eid, "node_i": i, "node_j": j,
            "radius": _number(record["radius"], "radius"),
            "segments": _integer(record["segments"], "segments"),
            "material": material, "internal_points": points,
            "metadata": _metadata(record["metadata"], "edge metadata")}


def _networkx():
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("networkx required: pip install networkx") from exc
    return nx


def _payload(G):
    nx = _networkx()
    if not isinstance(G, nx.MultiGraph) or G.is_directed() or not G.is_multigraph():
        _error("graph type", "requires an undirected NetworkX MultiGraph")
    header = _record(G.graph, _HEADER, "graph")
    payload = _header(header)
    node_order = _order(header["node_order"], "node_order")
    edge_order = _order(header["edge_order"], "edge_order")
    nodes, edges = {}, {}
    for key, attrs in G.nodes(data=True):
        record = _node(_record(attrs, _NODE, "node"), key)
        nodes[record["id"]] = record
    for u, v, key, attrs in G.edges(keys=True, data=True):
        record = _edge(_record(attrs, _EDGE, "edge"), key, u, v)
        eid = record["id"]
        if eid in edges:
            _error("edge identity", "edge IDs must be globally unique across endpoint pairs")
        edges[eid] = record
    if set(node_order) != set(nodes) or set(edge_order) != set(edges):
        _error("coverage", "orders must exactly cover current entities; topology edits/subgraphs are unsupported")
    for kind, records in (("node", nodes), ("edge", edges)):
        if payload[f"next_{kind}_id"] <= max(records, default=-1):
            _error(f"next_{kind}_id", "must exceed all retained IDs")
    payload["nodes"] = [nodes[nid] for nid in node_order]
    payload["edges"] = [edges[eid] for eid in edge_order]
    return payload


def _from_networkx_lossless(G, graph_cls):
    payload = _payload(G)
    graph = graph_cls.from_dict(payload)
    # JSON [] cannot encode the second dimension of an empty point array.
    # The v1 curve contract fixes it at (0, 3), independently of JSON reader.
    for record in payload["edges"]:
        if record["internal_points"] == []:
            graph.edges[record["id"]].internal_points = np.empty((0, 3))
    return graph


def _to_networkx_lossless(graph):
    nx = _networkx()
    # to_dict deliberately canonicalizes record order. Preserve actual source
    # orders separately and reconstruct from those orders on import.
    _unmasked(graph._box_size, "box_size")
    for nid, node in graph.nodes.items():
        _unmasked(node.position, "position")
        _unmasked(node.boundary, "boundary")
        if _integer(nid, "node key") != _integer(node.node_id, "node record ID"):
            _error("node identity", "source key differs from node.node_id")
        if (not isinstance(node.boundary, (tuple, list, np.ndarray))
                or (isinstance(node.boundary, np.ndarray) and node.boundary.ndim != 1)
                or len(node.boundary) != 6
                or any(not isinstance(v, (bool, np.bool_)) for v in node.boundary)):
            _error("boundary", "source boundary must contain six bools")
    for eid, edge in graph.edges.items():
        if _integer(eid, "edge key") != _integer(edge.edge_id, "edge record ID"):
            _error("edge identity", "source key differs from edge.edge_id")
        if type(edge.material) is not Material:
            _error("material", "native v1 supports the schema-3.1 Material class only")
        for name in Material.__dataclass_fields__:
            value = getattr(edge.material, name)
            _unmasked(value, f"material {name}")
            if name == "extra":
                _metadata(value, "material extra")
            elif name == "stiffness_tensor" and value is not None:
                if (not isinstance(value, np.ndarray) or value.shape != (6, 6)
                        or value.dtype.kind not in 'iuf'):
                    _error("material stiffness_tensor", "requires a numeric (6, 6) array")
                for number in value.flat:
                    _number(number, "material stiffness_tensor")
            elif name != "name" and value is not None:
                _number(value, f"material {name}")
        _unmasked(edge.radius, "radius")
        _unmasked(edge.segments, "segments")
        _unmasked(edge.internal_points, "internal_points")
        if edge.internal_points is not None:
            points = edge.internal_points
            if (not isinstance(points, np.ndarray) or points.ndim != 2
                    or points.shape[1] != 3 or points.dtype.kind not in 'iuf'
                    or not np.isfinite(points).all()):
                _error("internal_points", "source curves must be finite numeric (M, 3) arrays")
    try:
        data = graph.to_dict()
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        _error("source", f"cannot produce native schema-3.1 records: {exc}")
    header = {"format": _FORMAT, "version": 1, "structure_graph_version": "3.1",
              **{k: data[k] for k in ("dimension", "tolerance", "box_size", "next_node_id", "next_edge_id", "metadata")},
              "node_order": list(graph.nodes), "edge_order": list(graph.edges)}
    G = nx.MultiGraph()
    G.graph[_NS] = header
    for node in data["nodes"]:
        G.add_node(node["id"], **{_NS: node})
    for edge in data["edges"]:
        G.add_edge(edge["node_i"], edge["node_j"], key=edge["id"], **{_NS: edge})
    # Validation also detaches all mutable source values. Rebuild only the
    # outgoing native records; never retain a snapshot beside current attrs.
    payload = _payload(G)
    G = nx.MultiGraph()
    G.graph[_NS] = {"format": _FORMAT, "version": 1, "structure_graph_version": "3.1",
                   **{k: payload[k] for k in ("dimension", "tolerance", "box_size", "next_node_id", "next_edge_id", "metadata")},
                   "node_order": [_integer(v, "node_order") for v in header["node_order"]],
                   "edge_order": [_integer(v, "edge_order") for v in header["edge_order"]]}
    for node in payload["nodes"]:
        G.add_node(node["id"], **{_NS: node})
    for edge in payload["edges"]:
        G.add_edge(edge["node_i"], edge["node_j"], key=edge["id"], **{_NS: edge})
    return G

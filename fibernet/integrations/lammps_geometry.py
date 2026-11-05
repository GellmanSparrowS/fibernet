"""Explicit geometry-only preparation for the bounded LAMMPS adapter.

No native runtime, forces, masses or material-to-bond conversion are inferred.
"""
import copy
import json
import math
from numbers import Integral

import numpy as np

from fibernet.core.structure_graph import StructureGraph
from fibernet.core.material import Material
from fibernet.core.single_junction_import import _real_coordinates as _read_coordinates
from fibernet.integrations import _harmonic_profile as _bounds
from fibernet.integrations.lammps_backend import LAMMPSContractError


def _real_coordinates(value, label):
    try:
        return _read_coordinates(value, label)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LAMMPSContractError(str(exc)) from exc


def expand_edge_samples(graph, *, edge_ids):
    """Promote explicitly selected edge samples to independent graph nodes.

    Return ``(new_graph, report)`` without modifying ``graph``. Every selected
    edge must contain ordered internal points. Each is replaced by a chain of
    straight edges through those positions. Only stored vertices are retained;
    any continuous curve between them is approximated, not preserved. Old
    endpoints and all untouched
    stable IDs are retained. New nodes use ``merge=False``: intersections and
    coincident points on different edges are not inferred as junctions. Exact
    sample collisions with any other node or promoted sample are rejected so
    the caller must resolve ambiguous topology. No coordinates are snapped.

    This accepts only a finite JSON-roundtrippable 3D StructureGraph with at
    most 256 output nodes, 1024 output edges and 256 input samples. Selected
    IDs are explicit, unique and nonnegative. Source coordinates must be exactly
    representable in float64. The new graph is reconstructed through the native
    schema and its insertion order may differ. Original selected edge IDs are
    retired; the report maps each to new node and edge IDs in path order.
    Radius, a detached material and edge metadata are copied to each new edge;
    its ``segments`` is set to 1 and added nodes have no boundary flags or
    metadata. Metadata is copied verbatim: embedded old IDs, cross-section
    orientations and local frames are not remapped or transported. Units and
    the graph box are not changed.

    This is an explicit topology/discretization change, not mechanical
    equivalence or LAMMPS execution admission. New nodal masses, per-edge bond
    coefficients, restraints and other controls must be supplied by the user.
    Any unselected internal points remain and the backend may still reject them.
    Count/JSON limits do not certify total peak memory or downstream runtime.
    """
    if (type(graph) is not StructureGraph or isinstance(graph.dimension, (bool, np.bool_))
            or not isinstance(graph.dimension, Integral) or graph.dimension != 3):
        raise LAMMPSContractError('requires an ordinary 3D StructureGraph')
    if type(edge_ids) not in (tuple, list) or not edge_ids:
        raise LAMMPSContractError('edge_ids must be an explicit nonempty list or tuple')
    if len(edge_ids) > 1024:
        raise LAMMPSContractError('edge selection exceeds bound')
    def stable_id(x):
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) or not 0 <= x < 2**31:
            raise LAMMPSContractError('IDs must be nonnegative signed-32-bit integers')
        return int(x)
    chosen = [stable_id(x) for x in edge_ids]
    if len(set(chosen)) != len(chosen) or not set(chosen) <= set(graph.edges):
        raise LAMMPSContractError('selected edge IDs must be unique and present')
    if len(graph.nodes) > 256 or len(graph.edges) > 1024:
        raise LAMMPSContractError('input graph exceeds geometry preparation bounds')
    if graph.box_size is not None and _real_coordinates(graph.box_size, 'box size').shape != (3,):
        raise LAMMPSContractError('box size must have three coordinates')
    for nid, node in graph.nodes.items():
        if stable_id(nid) != stable_id(node.node_id):
            raise LAMMPSContractError('node record ID mismatch')
        if _real_coordinates(node.position, 'node position').shape != (3,):
            raise LAMMPSContractError('node positions must have three coordinates')
    samples, total = {}, 0
    for eid, edge in graph.edges.items():
        if stable_id(eid) != stable_id(edge.edge_id):
            raise LAMMPSContractError('edge record ID mismatch')
        stable_id(edge.node_i); stable_id(edge.node_j)
        if edge.internal_points is None:
            continue
        arr = _real_coordinates(edge.internal_points, 'internal points')
        if arr.shape == (0,):
            arr = np.empty((0, 3))
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise LAMMPSContractError('internal points must have shape (N, 3)')
        total += len(arr)
        if total > 256:
            raise LAMMPSContractError('input sample count exceeds bound')
        samples[eid] = arr
    if any(eid not in samples or len(samples[eid]) == 0 for eid in chosen):
        raise LAMMPSContractError('selected edges require internal points')
    added = sum(len(samples[eid]) for eid in chosen)
    if len(graph.nodes) + added > 256 or len(graph.edges) + added > 1024:
        raise LAMMPSContractError('expanded graph exceeds bounded LAMMPS pilot counts')
    if (stable_id(graph._next_node_id) + added > 2**31 or
            stable_id(graph._next_edge_id) + added + len(chosen) > 2**31):
        raise LAMMPSContractError('new IDs would exceed the LAMMPS stable ID range')
    # A geometry point never becomes a junction merely because positions match.
    existing = [tuple(_real_coordinates(n.position, 'node position')) for n in graph.nodes.values()]
    promoted = [tuple(p) for eid in chosen for p in samples[eid]]
    existing_positions = set(existing)
    if len(set(promoted)) != len(promoted) or any(p in existing_positions for p in promoted):
        raise LAMMPSContractError('promoted sample coincides with another node or sample; resolve topology explicitly')
    for eid in chosen:
        edge = graph.edges[eid]
        if edge.node_i not in graph.nodes or edge.node_j not in graph.nodes:
            raise LAMMPSContractError('edge endpoint missing')
        points = [graph.nodes[edge.node_i].position, *samples[eid], graph.nodes[edge.node_j].position]
        if any(math.dist(a, b) == 0 for a, b in zip(points, points[1:])):
            raise LAMMPSContractError('zero-length segment is unsupported')
    try:
        # Charge metadata before copying; this is a bounded JSON-data contract,
        # not a claim about total process memory or arbitrary object support.
        _bounds.json_bytes([graph.metadata] + [n.metadata for n in graph.nodes.values()] +
                           [e.metadata for e in graph.edges.values()], 1024 * 1024)
        for edge in graph.edges.values():
            if type(edge.material) is not Material:
                raise LAMMPSContractError('only ordinary Material records are supported')
            tensor = edge.material.stiffness_tensor
            if tensor is not None and getattr(tensor, 'size', 37) > 36:
                raise LAMMPSContractError('material tensor exceeds bound')
        _bounds.json_bytes([e.material.to_dict() for e in graph.edges.values()], 1024 * 1024)
        record = graph.to_dict()
        encoded = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if len(encoded.encode('utf-8')) > 1024 * 1024:
            raise LAMMPSContractError('source graph JSON exceeds one MiB')
        result = StructureGraph.from_dict(json.loads(encoded))
        if json.dumps(result.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False) != encoded:
            raise LAMMPSContractError('graph cannot be reconstructed without record changes')
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:
        if isinstance(exc, LAMMPSContractError):
            raise
        raise LAMMPSContractError('graph must be finite native JSON-roundtrippable') from exc
    mapping = {}
    for eid in chosen:
        edge = result.edges[eid]
        start, end = edge.node_i, edge.node_j
        new_nodes = [result.add_node(point, merge=False) for point in samples[eid]]
        result.remove_edge(eid)
        chain = [start, *new_nodes, end]
        new_edges = []
        for a, b in zip(chain, chain[1:]):
            new_id = result.add_edge(a, b, radius=edge.radius, material=copy.deepcopy(edge.material),
                                     segments=1, allow_parallel=True)
            result.edges[new_id].metadata = copy.deepcopy(edge.metadata)
            new_edges.append(new_id)
        mapping[eid] = {'new_node_ids': new_nodes, 'new_edge_ids': new_edges}
    return result, {'schema': 'fibernet.lammps.sample-expansion/1',
                    'source_edge_to_new_ids': mapping,
                    'scope': 'geometry and explicit topology only; no masses, coefficients or mechanics inferred'}

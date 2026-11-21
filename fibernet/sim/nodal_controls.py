"""Explicit stable-ID control preparation for existing row-based solvers.

No solver is constructed. Force components retain caller units; this is not a
universal boundary-condition, material, time-history, or unit-conversion API.
"""
from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping

import numpy as np


def _signature(graph):
    # Unlike StructureGraph.fingerprint(), retain materials, internal geometry,
    # full numeric precision AND insertion order. No parallel graph authority.
    payload = [list(graph.nodes), list(graph.edges), graph.to_dict()]
    try:
        raw = json.dumps(payload, sort_keys=True, allow_nan=False,
                         separators=(',', ':')).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise ValueError('preparation requires a finite JSON-serializable StructureGraph') from exc
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class PreparedNodalControls:
    """Detached controls; call a kwargs method immediately before a solve.

    Any graph change requires fresh preparation, including insertion reordering.
    Returned kwargs are independent copies. Mutating them is the caller's explicit
    responsibility. This object is not a solver plan or a restart checkpoint.
    """
    node_ids: tuple
    edge_ids: tuple
    fixed_rows: tuple
    forces: tuple
    dimension: int
    graph_signature: str

    def _check(self, graph):
        if _signature(graph) != self.graph_signature:
            raise ValueError('graph changed since preparation; prepare_nodal_controls again')

    def taichi_kwargs(self, graph):
        """Ordinary TaichiEngine.dynamics row controls; requires 3 components."""
        self._check(graph)
        if self.dimension != 3:
            raise ValueError('ordinary Taichi dynamics requires dimension=3 controls')
        return {'fixed_nodes': list(self.fixed_rows),
                'external_force': np.array(self.forces, dtype=float).reshape(-1, 3)}

    def fem_kwargs(self, graph):
        """BeamFrameFEM.solve_2d/solve_3d translational force row controls.

        FEM fixed_nodes clamps rotations as well as translations. Taichi has no
        corresponding rotational DOFs. This helper does not equate those models.
        """
        self._check(graph)
        return {'fixed_nodes': list(self.fixed_rows),
                'forces': np.array(self.forces, dtype=float).reshape(-1, self.dimension)}

    def to_dict(self):
        """JSON-ready provenance sidecar, not an executable deserialization API."""
        return {'schema': 'fibernet.nodal_controls/1',
                'node_ids': list(self.node_ids), 'edge_ids': list(self.edge_ids),
                'fixed_rows': list(self.fixed_rows),
                'fixed_node_ids': [self.node_ids[i] for i in self.fixed_rows],
                'forces': [list(row) for row in self.forces],
                'dimension': self.dimension, 'graph_signature': self.graph_signature,
                'units': 'caller-supplied; no conversion'}


def prepare_nodal_controls(graph, *, fixed_node_ids=None, forces_by_id=None,
                           dimension=3, fixed_nodes=None, external_force=None):
    """Prepare stable-ID supports and sparse forces in existing adapter row order.

    IDs must be nonnegative integers (Python and NumPy integers may be mixed),
    never strings, floats or bools. Missing force IDs mean zero force. Unknown IDs,
    repeated support IDs, invalid vectors, and dual row/ID spellings fail early.
    Optional fixed_nodes/external_force retain row meanings within this helper;
    existing solver methods and their accepted legacy calls are unchanged.

    dimension is the number of translational force components, explicitly 2 or 3.
    Use 3 for ordinary Taichi and 2/3 for the corresponding raw FEM solve.
    The graph must be finite and JSON-serializable for exact mutation detection.
    """
    from fibernet.core.structure_graph import StructureGraph
    from .accelerated import (_graph_to_arrays, _strict_state_ids,
                              _validated_fixed_rows, _finite_real_array)
    if not isinstance(graph, StructureGraph):
        raise TypeError('graph must be a StructureGraph')
    if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, (int, np.integer)) or dimension not in (2, 3):
        raise ValueError('dimension must be 2 or 3')
    if fixed_node_ids is not None and fixed_nodes is not None:
        raise ValueError('choose fixed_node_ids or fixed_nodes, not both')
    if forces_by_id is not None and external_force is not None:
        raise ValueError('choose forces_by_id or external_force, not both')
    def force_array(value, name, shape):
        out = _finite_real_array(value, name, shape)
        if any(isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat):
            raise ValueError(f'{name} must not contain boolean components')
        return out

    node_ids = _strict_state_ids(list(graph.nodes), nonnegative=True)
    edge_ids = _strict_state_ids(list(graph.edges), 'edge_ids', nonnegative=True)
    if not node_ids:
        raise ValueError('graph must have at least one node')
    _, _, adapter_ids, rows = _graph_to_arrays(graph)
    assert adapter_ids == node_ids
    if fixed_node_ids is not None:
        ids = _strict_state_ids(list(fixed_node_ids), 'fixed_node_ids', nonnegative=True)
        unknown = [nid for nid in ids if nid not in rows]
        if unknown:
            raise ValueError(f'unknown fixed_node_ids: {unknown}')
        fixed = [rows[nid] for nid in ids]
    else:
        fixed = _validated_fixed_rows(fixed_nodes, len(node_ids)) if fixed_nodes is not None else []
    forces = np.zeros((len(node_ids), dimension))
    if forces_by_id is not None:
        if not isinstance(forces_by_id, Mapping):
            raise ValueError('forces_by_id must be an ID-to-vector mapping')
        ids = _strict_state_ids(list(forces_by_id), 'forces_by_id keys', nonnegative=True)
        unknown = [nid for nid in ids if nid not in rows]
        if unknown:
            raise ValueError(f'unknown forces_by_id keys: {unknown}')
        for nid in ids:
            forces[rows[nid]] = force_array(forces_by_id[nid], f'force for node {nid}', (dimension,))
    elif external_force is not None:
        forces = force_array(external_force, 'external_force', (len(node_ids), dimension))
    return PreparedNodalControls(tuple(node_ids), tuple(edge_ids), tuple(fixed),
                                 tuple(tuple(row) for row in forces.tolist()),
                                 int(dimension), _signature(graph))


@dataclass(frozen=True)
class PreparedTaichiSchedule:
    """Detached displacement schedule, bound to the graph at preparation."""
    _graph_signature: str
    _schedule: tuple

    def taichi_kwargs(self, graph):
        """Return fresh native dynamics kwargs after checking the graph."""
        if _signature(graph) != self._graph_signature:
            raise ValueError('graph changed since schedule preparation; prepare again')
        return {'displacement_schedule': {
            row: [(step, np.array(vector)) for step, vector in points]
            for row, points in self._schedule}}


def _schedule_scalar(value):
    if type(value) not in (int, float) and not (
            isinstance(value, np.generic) and value.dtype.kind in 'iuf'):
        raise ValueError('schedule values must be real integer or floating scalars, not time or boolean')
    try:
        out = float(value)
    except OverflowError as exc:
        raise ValueError('schedule values must be finite binary64 numbers') from exc
    # Check before changing waypoint positions or turning a small move into zero.
    if not np.isfinite(out) or out != (value.item() if isinstance(value, np.generic) else value):
        raise ValueError('schedule values must be exactly representable in binary64')
    return out


def _schedule_sequence(value, label, size=None):
    # Do not infer a component order from a set, mapping, iterator, or
    # ndarray subclass (for example, an implicitly lossy masked array).
    if type(value) is np.ndarray:
        if size is None:
            valid = value.ndim == 1 or (value.ndim == 2 and value.shape[1] == 2)
        else:
            valid = value.shape == (size,)
    else:
        valid = type(value) in (list, tuple) and (size is None or len(value) == size)
    if not valid:
        raise ValueError(f'{label} must be an ordered list, tuple or ndarray of the required shape')
    return value

def prepare_taichi_schedule_by_id(graph, schedule_by_id):
    """Prepare absolute XYZ displacement waypoints for ordinary Taichi dynamics.

    Keys are stable nonnegative integer node IDs. Steps use the engine's
    integration-step coordinates, including fractions, not seconds. There
    is no unit conversion. Each schedule, waypoint and XYZ vector must be
    an ordered list, tuple or ndarray. The engine still validates and sorts
    the timeline.
    Call the returned object's `taichi_kwargs(graph)`
    immediately before solving; graph edits require fresh preparation.
    """
    from fibernet.core.structure_graph import StructureGraph

    if not isinstance(graph, StructureGraph):
        raise TypeError('graph must be a StructureGraph')
    if not isinstance(schedule_by_id, Mapping):
        raise ValueError('schedule_by_id must be an ID-to-waypoints mapping')
    signature = _signature(graph)
    # Same authoritative insertion order as accelerated._graph_to_arrays;
    # StructureGraph.to_numpy() is sorted by ID and cannot supply these rows.
    rows = {node_id: row for row, node_id in enumerate(graph.nodes)}
    if not rows:
        raise ValueError('graph must have at least one node')
    row_schedule, seen = {}, set()
    for node_id, waypoints in schedule_by_id.items():
        if type(node_id) is not int and not (
                isinstance(node_id, np.generic) and node_id.dtype.kind in 'iu'):
            raise ValueError('schedule_by_id keys must be nonnegative integer node IDs')
        node_id = int(node_id)
        if node_id < 0:
            raise ValueError('schedule_by_id keys must be nonnegative integer node IDs')
        if node_id not in rows:
            raise ValueError(f'unknown schedule_by_id key: {node_id}')
        row = rows[node_id]
        if row in seen:
            raise ValueError('schedule_by_id keys must be unique')
        seen.add(row)
        points = []
        for waypoint in _schedule_sequence(waypoints, 'schedule'):
            step, displacement = _schedule_sequence(waypoint, '(step, displacement) waypoint', 2)
            vector = tuple(_schedule_scalar(v) for v in
                           _schedule_sequence(displacement, 'XYZ displacement', 3))
            points.append((_schedule_scalar(step), vector))
        if points:
            row_schedule[row] = tuple(points)
    return PreparedTaichiSchedule(signature, tuple(row_schedule.items()))

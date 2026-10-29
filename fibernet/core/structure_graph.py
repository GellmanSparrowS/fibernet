"""
StructureGraph — The unified graph representation for FiberNet.

Design Principles
-----------------
1. **NumPy-native**: Node positions stored as float64 arrays for fast computation.
2. **Functional API**: Transformations return new graphs (immutable by default).
3. **Dimension-agnostic**: Same API for 2D (z=0) and 3D structures.
4. **Edge discretization**: Each edge carries N internal points for deformation.
5. **Boundary-aware**: Nodes carry boundary flags for periodic tiling/welding.
6. **Metadata-rich**: Every node and edge carries extensible metadata dicts.
7. **Conversion-ready**: Seamless interop with networkx, numpy, taichi, FiberNetwork.

The StructureGraph is the *authoritative* topological and geometric model.
All generators produce StructureGraph; all simulators consume StructureGraph;
all visualizers render StructureGraph.

Topology Model
--------------
- Nodes = junction points in space (with position, boundary flags, metadata)
- Edges = beam/strut segments between nodes (with radius, material, internal points)
- A node shared by ≥2 edges becomes a crosslink/junction automatically

Edge Discretization
-------------------
Each edge stores its endpoints (node_i, node_j positions) plus optional
internal_points — an (M, 3) array of intermediate positions along the edge.
This enables:
  - Curved beam elements in FEM
  - Deformation visualization (displaced internal points)
  - Accurate arc-length computation
  - High-fidelity rendering of complex structures

Examples
--------
>>> from fibernet.core.structure_graph import StructureGraph
>>> g = StructureGraph(dimension=2)
>>> n0 = g.add_node([0, 0])
>>> n1 = g.add_node([10, 0])
>>> n2 = g.add_node([5, 8.66])
>>> g.add_edge(n0, n1, radius=0.5, n_internal=4)
>>> g.add_edge(n1, n2, radius=0.5, n_internal=4)
>>> g.add_edge(n2, n0, radius=0.5, n_internal=4)
>>> print(g)
StructureGraph(dim=2, nodes=3, edges=3)
"""

from __future__ import annotations

import copy
import hashlib
import json
from numbers import Real
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from fibernet.core.material import Material


def _material_record(data: Any) -> dict:
    """Validate and detach the JSON-only material record in graph schema 3.1.

    This is serialization validation, not a constitutive-model validity check.
    Legacy graphs without a material record are handled by the graph reader.
    """
    if not isinstance(data, dict):
        raise ValueError("edge material must be an object")
    allowed = set(Material.__dataclass_fields__)
    if any(not isinstance(key, str) or key not in allowed for key in data):
        raise ValueError("edge material contains unknown fields")

    def number(value, label):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{label} must be a finite real number")
        try:
            value = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{label} must be a finite real number") from exc
        if not np.isfinite(value):
            raise ValueError(f"{label} must be a finite real number")
        return value

    def json_value(value, depth=0):
        if depth > 100:
            raise ValueError('material extra is cyclic or exceeds 100 nesting levels')
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, Real):
            return number(value, 'material extra value')
        if isinstance(value, list):
            return [json_value(item, depth + 1) for item in value]
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            return {key: json_value(item, depth + 1) for key, item in value.items()}
        raise ValueError("material extra must contain only JSON data with string keys")

    result = {}
    required_numbers = {'density', 'youngs_modulus', 'poissons_ratio'}
    for key, value in data.items():
        if key == 'name':
            if not isinstance(value, str):
                raise ValueError('material name must be a string')
            result[key] = value
        elif key == 'extra':
            if not isinstance(value, dict):
                raise ValueError('material extra must be an object')
            result[key] = json_value(value)
        elif key == 'stiffness_tensor':
            if value is None:
                result[key] = None
            else:
                try:
                    tensor = np.asarray(value)
                    valid = (tensor.shape == (6, 6) and tensor.dtype.kind in 'iuf'
                             and np.isfinite(tensor).all())
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    raise ValueError('material stiffness_tensor must be a finite 6x6 array')
                result[key] = tensor.astype(float).tolist()
        elif value is None and key not in required_numbers:
            result[key] = None
        else:
            result[key] = number(value, f'material {key}')
    return result


# ---------------------------------------------------------------------------
# Node / Edge data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=False)
class SNode:
    """A node (junction point) in the structure graph.

    Parameters
    ----------
    position : np.ndarray
        3D position (for 2D structures, z=0).
    node_id : int
        Unique integer identifier within the graph.
    boundary : tuple of bool
        (on_x_min, on_x_max, on_y_min, on_y_max, on_z_min, on_z_max).
        True if node lies on that cell boundary face. Used for periodic welding.
    metadata : dict
        Extensible metadata (e.g., generation source, symmetry group).
    """

    position: np.ndarray
    node_id: int = -1
    boundary: Tuple[bool, ...] = (False,) * 6
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return 2 if abs(self.position[2]) < 1e-12 else 3


@dataclass(frozen=False)
class SEdge:
    """An edge (beam/strut segment) in the structure graph.

    Parameters
    ----------
    node_i, node_j : int
        Endpoint node IDs.
    edge_id : int
        Unique integer identifier within the graph.
    radius : float
        Cross-section radius (or characteristic dimension).
    material : Material
        Beam material properties.
    internal_points : np.ndarray or None
        (M, 3) array of intermediate positions along the edge.
        Used for curved beams, deformation visualization, and high-fidelity rendering.
    segments : int
        Number of FEM discretization segments (for simulation meshing).
    metadata : dict
        Extensible metadata.
    """

    node_i: int
    node_j: int
    edge_id: int = -1
    radius: float = 0.1
    material: Material = field(default_factory=Material)
    internal_points: Optional[np.ndarray] = None
    segments: int = 4
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main StructureGraph class
# ---------------------------------------------------------------------------

class StructureGraph:
    """The unified graph for fiber networks, lattices, and metamaterials.

    Parameters
    ----------
    dimension : int
        Spatial dimension (2 or 3).
    tolerance : float
        Node-merging tolerance. Nodes closer than this are merged.
    box_size : array-like or None
        Unit cell dimensions [Lx, Ly, Lz] for periodic structures.
    """

    def __init__(
        self,
        dimension: int = 2,
        tolerance: float = 1e-6,
        box_size: Optional[Sequence[float]] = None,
    ):
        if dimension not in (2, 3):
            raise ValueError("dimension must be 2 or 3")
        self._dimension = dimension
        self._tolerance = tolerance

        if box_size is not None:
            bs = np.asarray(box_size, dtype=np.float64)
            if len(bs) == 2:
                bs = np.append(bs, 0.0)
            self._box_size = bs
        else:
            self._box_size = None

        self._nodes: Dict[int, SNode] = {}
        self._edges: Dict[int, SEdge] = {}
        self._next_node_id = 0
        self._next_edge_id = 0
        self._metadata: Dict[str, Any] = {}

        # Spatial hash for fast node merging
        self._spatial_hash: Dict[Tuple[int, int, int], List[int]] = {}
        # Edge set for deduplication: frozenset({node_i, node_j}) → edge_id
        self._edge_set: Dict[frozenset, int] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @property
    def box_size(self) -> Optional[np.ndarray]:
        return self._box_size.copy() if self._box_size is not None else None

    @box_size.setter
    def box_size(self, value):
        if value is not None:
            bs = np.asarray(value, dtype=np.float64)
            if len(bs) == 2:
                bs = np.append(bs, 0.0)
            self._box_size = bs
        else:
            self._box_size = None

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    @property
    def nodes(self) -> Dict[int, SNode]:
        return self._nodes

    @property
    def edges(self) -> Dict[int, SEdge]:
        return self._edges

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def _pos_key(self, pos: np.ndarray) -> Tuple[int, int, int]:
        """Hash a position to a spatial grid cell for fast lookup."""
        scale = 1.0 / max(self._tolerance, 1e-12)
        return (
            int(round(pos[0] * scale)),
            int(round(pos[1] * scale)),
            int(round(pos[2] * scale)),
        )

    def _nearby_keys(self, key: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Return neighboring grid cell keys."""
        kx, ky, kz = key
        offsets = [
            (0, 0, 0), (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        ]
        # Nearby points can lie across an edge or corner of a hash cell.
        # Keep the existing center/axis search order when several nodes match.
        offsets.extend(
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if (dx != 0) + (dy != 0) + (dz != 0) >= 2
        )
        return [(kx + dx, ky + dy, kz + dz) for dx, dy, dz in offsets]

    def _checked_position_key(self, position):
        """Check the destination lookup key before changing graph state."""
        if not np.isfinite(position).all():
            raise ValueError("position must be finite")
        try:
            with np.errstate(over='raise', invalid='raise'):
                return self._pos_key(position)
        except (ValueError, OverflowError, FloatingPointError, IndexError) as exc:
            raise ValueError("position exceeds the spatial hash coordinate range or has missing coordinates") from exc

    def add_node(
        self,
        position: Sequence[float],
        *,
        boundary: Optional[Tuple[bool, ...]] = None,
        merge: bool = True,
        **metadata: Any,
    ) -> int:
        """Add a node at the given position.

        If ``merge=True`` (default), returns the ID of an existing node
        within tolerance instead of creating a duplicate.

        Returns
        -------
        int
            Node ID.
        """
        pos = np.asarray(position, dtype=np.float64).ravel()
        if len(pos) == 2:
            pos = np.append(pos, 0.0)
        if len(pos) != 3:
            raise ValueError(f"position must be 2D or 3D, got shape {pos.shape}")

        key = self._checked_position_key(pos)
        if merge:
            for nk in self._nearby_keys(key):
                for nid in self._spatial_hash.get(nk, []):
                    if np.linalg.norm(self._nodes[nid].position - pos) < self._tolerance:
                        return nid

        nid = self._next_node_id
        bnd = boundary if boundary is not None else (False,) * 6
        node = SNode(position=pos.copy(), node_id=nid, boundary=bnd, metadata=dict(metadata))
        self._nodes[nid] = node
        self._next_node_id += 1

        self._spatial_hash.setdefault(key, []).append(nid)
        return nid

    def remove_node(self, node_id: int):
        """Remove a node and all its incident edges."""
        if node_id not in self._nodes:
            return
        # Remove incident edges
        edges_to_remove = [
            eid for eid, e in self._edges.items()
            if e.node_i == node_id or e.node_j == node_id
        ]
        for eid in edges_to_remove:
            self.remove_edge(eid)
        # Remove from spatial hash
        pos = self._nodes[node_id].position
        key = self._pos_key(pos)
        if key in self._spatial_hash and node_id in self._spatial_hash[key]:
            self._spatial_hash[key].remove(node_id)
        del self._nodes[node_id]

    # ------------------------------------------------------------------
    # Node position manipulation (for RL / parametric design)
    # ------------------------------------------------------------------

    def displace_node(self, node_id: int, displacement: Sequence[float]):
        """Move a node by a displacement vector (dx, dy[, dz]).

        Parameters
        ----------
        node_id : int
            Node to move.
        displacement : array-like
            Displacement vector (dx, dy) or (dx, dy, dz).

        Raises
        ------
        KeyError
            If node_id not found.

        Examples
        --------
        >>> g.displace_node(5, [0.1, 0.2])  # move node 5 by (0.1, 0.2)
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id} not found")
        disp = np.asarray(displacement, dtype=float)
        if len(disp) == 2:
            disp = np.append(disp, 0.0)
        try:
            with np.errstate(over='raise', invalid='raise'):
                position = self._nodes[node_id].position + disp
        except FloatingPointError as exc:
            raise ValueError("displacement must produce a finite position") from exc
        self.set_node_position(node_id, position)

    def set_node_position(self, node_id: int, position: Sequence[float]):
        """Set the absolute position of a node.

        Parameters
        ----------
        node_id : int
            Node to reposition.
        position : array-like
            New position (x, y) or (x, y, z).
        """
        self.set_node_positions({node_id: position})

    def set_node_positions(self, positions: Dict[int, Sequence[float]]):
        """Batch-set positions, validating every update before changing any.

        Rejected edits leave positions, merge lookup and stable IDs unchanged.
        Coordinates are flattened and normalized as in ``add_node``.

        Parameters
        ----------
        positions : dict
            Mapping {node_id: (x, y[, z])}.

        Examples
        --------
        >>> g.set_node_positions({1: [0.5, 0.0], 3: [1.0, 2.0]})
        """
        updates = []
        for nid, value in positions.items():
            if nid not in self._nodes:
                raise KeyError(f"Node {nid} not found")
            pos = np.asarray(value, dtype=np.float64).ravel()
            if len(pos) == 2:
                pos = np.append(pos, 0.0)
            if len(pos) != 3:
                raise ValueError(f"position must be 2D or 3D, got shape {pos.shape}")
            pos = pos.copy()
            new_key = self._checked_position_key(pos)
            old_key = self._pos_key(self._nodes[nid].position)
            updates.append((nid, pos, old_key, new_key))
        for nid, pos, old_key, new_key in updates:
            if old_key in self._spatial_hash and nid in self._spatial_hash[old_key]:
                self._spatial_hash[old_key].remove(nid)
            self._nodes[nid].position = pos
            self._spatial_hash.setdefault(new_key, []).append(nid)

    def get_internal_nodes(self) -> List[int]:
        """Return node IDs of non-boundary (internal) nodes.

        Useful for RL: these are the nodes whose positions can be
        parameterized as continuous actions.
        """
        result = []
        for nid, node in self._nodes.items():
            if not any(node.boundary):
                result.append(nid)
        return sorted(result)

    def get_boundary_nodes(self) -> List[int]:
        """Return node IDs of boundary nodes."""
        result = []
        for nid, node in self._nodes.items():
            if any(node.boundary):
                result.append(nid)
        return sorted(result)

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(
        self,
        node_i: int,
        node_j: int,
        *,
        radius: float = 0.1,
        material: Optional[Material] = None,
        n_internal: int = 0,
        internal_points: Optional[np.ndarray] = None,
        segments: int = 4,
        allow_parallel: bool = False,
        **metadata: Any,
    ) -> int:
        """Add an edge between two existing nodes.

        Parameters
        ----------
        node_i, node_j : int
            Endpoint node IDs.
        radius : float
            Cross-section radius.
        material : Material, optional
            Beam material. Defaults to generic Material().
        n_internal : int
            Number of equally-spaced internal points to generate along the edge.
            These are stored in ``internal_points``.
        internal_points : np.ndarray, optional
            Explicit (M, 3) internal point array. Overrides ``n_internal``.
        segments : int
            Number of FEM beam segments.
        allow_parallel : bool
            Keep a distinct fiber when another edge has the same endpoints.
            The default preserves the existing simple-graph behavior.

        Returns
        -------
        int
            Edge ID, or -1 if node_i == node_j.
        """
        if node_i == node_j:
            return -1
        if node_i not in self._nodes or node_j not in self._nodes:
            raise ValueError(f"Node {node_i} or {node_j} not in graph")

        # Deduplicate: if edge between these nodes already exists, return existing
        edge_key = frozenset({node_i, node_j})
        if edge_key in self._edge_set and not allow_parallel:
            return self._edge_set[edge_key]

        if internal_points is None and n_internal > 0:
            pi = self._nodes[node_i].position
            pj = self._nodes[node_j].position
            t = np.linspace(0, 1, n_internal + 2)[1:-1]
            internal_points = pi[None, :] * (1 - t[:, None]) + pj[None, :] * t[:, None]

        eid = self._next_edge_id
        edge = SEdge(
            node_i=node_i,
            node_j=node_j,
            edge_id=eid,
            radius=radius,
            material=material or Material(),
            internal_points=internal_points,
            segments=segments,
            metadata=dict(metadata),
        )
        self._edges[eid] = edge
        self._edge_set.setdefault(edge_key, eid)
        self._next_edge_id += 1
        return eid

    def remove_edge(self, edge_id: int):
        """Remove an edge by ID."""
        edge = self._edges.pop(edge_id, None)
        if edge is not None:
            key = frozenset({edge.node_i, edge.node_j})
            if self._edge_set.get(key) == edge_id:
                replacement = next((eid for eid, other in self._edges.items()
                                    if frozenset({other.node_i, other.node_j}) == key), None)
                if replacement is None:
                    self._edge_set.pop(key, None)
                else:
                    self._edge_set[key] = replacement

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def add_polyline(
        self,
        points: Sequence[Sequence[float]],
        *,
        closed: bool = False,
        radius: float = 0.1,
        material: Optional[Material] = None,
        n_internal: int = 0,
        segments: int = 4,
        **metadata: Any,
    ) -> List[int]:
        """Add a chain of edges through a sequence of points.

        Parameters
        ----------
        points : sequence of (x, y[, z])
            Polyline vertices.
        closed : bool
            If True, close the polyline by connecting last to first.
        radius, material, n_internal, segments
            Passed to ``add_edge``.

        Returns
        -------
        list of int
            Edge IDs created.
        """
        pts = [np.asarray(p, dtype=np.float64) for p in points]
        nids = [self.add_node(p) for p in pts]
        edge_ids = []
        pairs = list(zip(nids[:-1], nids[1:]))
        if closed and len(nids) > 2:
            pairs.append((nids[-1], nids[0]))
        for ni, nj in pairs:
            eid = self.add_edge(
                ni, nj, radius=radius, material=material,
                n_internal=n_internal, segments=segments, **metadata,
            )
            if eid >= 0:
                edge_ids.append(eid)
        return edge_ids

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def neighbors(self, node_id: int) -> List[int]:
        """Return list of neighbor node IDs."""
        nbrs = []
        for e in self._edges.values():
            if e.node_i == node_id:
                nbrs.append(e.node_j)
            elif e.node_j == node_id:
                nbrs.append(e.node_i)
        return nbrs

    def degree(self, node_id: int) -> int:
        """Return the degree (number of incident edges) of a node."""
        return sum(
            1 for e in self._edges.values()
            if e.node_i == node_id or e.node_j == node_id
        )

    def node_positions(self) -> np.ndarray:
        """Return (N, 3) array of all node positions, ordered by node_id."""
        if not self._nodes:
            return np.empty((0, 3))
        ids = sorted(self._nodes.keys())
        return np.array([self._nodes[nid].position for nid in ids])

    def edge_array(self) -> np.ndarray:
        """Return (M, 2) array of edge endpoint node IDs."""
        if not self._edges:
            return np.empty((0, 2), dtype=int)
        return np.array([[e.node_i, e.node_j] for e in self._edges.values()], dtype=int)

    def edge_lengths(self) -> np.ndarray:
        """Return array of edge lengths."""
        lengths = []
        for e in self._edges.values():
            pi = self._nodes[e.node_i].position
            pj = self._nodes[e.node_j].position
            lengths.append(np.linalg.norm(pj - pi))
        return np.array(lengths)

    def edge_midpoints(self) -> np.ndarray:
        """Return (M, 3) array of edge midpoints."""
        mids = []
        for e in self._edges.values():
            pi = self._nodes[e.node_i].position
            pj = self._nodes[e.node_j].position
            mids.append((pi + pj) / 2.0)
        return np.array(mids) if mids else np.empty((0, 3))

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min_corner, max_corner) of the bounding box."""
        if not self._nodes:
            return np.zeros(3), np.zeros(3)
        pos = self.node_positions()
        return pos.min(axis=0), pos.max(axis=0)

    def total_edge_length(self) -> float:
        """Sum of all edge lengths."""
        return float(self.edge_lengths().sum()) if self._edges else 0.0

    def _adjacency_lists(self) -> Dict[int, List[int]]:
        """Build operation-local neighbor lists in physical-edge order."""
        adjacency = {nid: [] for nid in self._nodes}
        for edge in self._edges.values():
            adjacency.setdefault(edge.node_i, []).append(edge.node_j)
            if edge.node_j != edge.node_i:
                adjacency.setdefault(edge.node_j, []).append(edge.node_i)
        return adjacency

    def is_connected(self) -> bool:
        """Check connectivity in O(nodes + physical edges) time."""
        if not self._nodes:
            return True
        adjacency = self._adjacency_lists()
        visited = set()
        start = next(iter(self._nodes))
        stack = [start]
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            stack.extend(adjacency[nid])
        return len(visited) == len(self._nodes)

    def connected_components(self) -> List[set]:
        """Return component node sets in node insertion order, in O(N + E)."""
        adjacency = self._adjacency_lists()
        visited = set()
        components = []
        for start in self._nodes:
            if start in visited:
                continue
            comp = set()
            stack = [start]
            while stack:
                nid = stack.pop()
                if nid in visited:
                    continue
                visited.add(nid)
                comp.add(nid)
                stack.extend(n for n in adjacency[nid] if n not in visited)
            components.append(comp)
        return components

    # ------------------------------------------------------------------
    # Discretize edges (add internal points for deformation support)
    # ------------------------------------------------------------------

    def discretize_edges(self, n_points_per_edge: int = 8) -> "StructureGraph":
        """Return a copy, filling edges that have no stored internal samples.

        Edges with ``internal_points=None`` or an empty array receive
        ``n_points_per_edge`` equally spaced endpoint-linear samples. Existing
        nonempty samples are preserved exactly, including collinear points;
        their count need not equal ``n_points_per_edge``. In particular, zero
        requests no new samples and does not clear stored geometry.

        Physical nodes, edges, IDs, materials and ``segments`` are unchanged.
        These geometry samples are not additional mechanical nodes and do not
        supply an interior displacement field.
        """
        g = self.copy()
        for eid, edge in g._edges.items():
            pi = g._nodes[edge.node_i].position
            pj = g._nodes[edge.node_j].position
            t = np.linspace(0, 1, n_points_per_edge + 2)[1:-1]
            if edge.internal_points is not None and len(edge.internal_points) > 0:
                continue
            edge.internal_points = (
                pi[None, :] * (1 - t[:, None]) + pj[None, :] * t[:, None]
            )
        return g

    # ------------------------------------------------------------------
    # Copy / merge
    # ------------------------------------------------------------------

    def copy(self) -> "StructureGraph":
        """Deep copy of this graph."""
        g = StructureGraph(
            dimension=self._dimension,
            tolerance=self._tolerance,
            box_size=self._box_size.copy() if self._box_size is not None else None,
        )
        g._metadata = copy.deepcopy(self._metadata)
        for nid, node in self._nodes.items():
            new_node = SNode(
                position=node.position.copy(),
                node_id=node.node_id,
                boundary=tuple(node.boundary),
                metadata=copy.deepcopy(node.metadata),
            )
            g._nodes[nid] = new_node
            key = g._pos_key(node.position)
            g._spatial_hash.setdefault(key, []).append(nid)
        g._next_node_id = self._next_node_id
        g._edge_set = dict(self._edge_set)
        for eid, edge in self._edges.items():
            new_edge = SEdge(
                node_i=edge.node_i,
                node_j=edge.node_j,
                edge_id=edge.edge_id,
                radius=edge.radius,
                material=copy.deepcopy(edge.material),
                internal_points=edge.internal_points.copy() if edge.internal_points is not None else None,
                segments=edge.segments,
                metadata=copy.deepcopy(edge.metadata),
            )
            g._edges[eid] = new_edge
        g._next_edge_id = self._next_edge_id
        return g

    def merge(self, other: "StructureGraph") -> "StructureGraph":
        """Merge another graph into this one (union). Returns new graph."""
        g = self.copy()
        id_map = {}
        for nid, node in other._nodes.items():
            new_nid = g.add_node(
                node.position, boundary=tuple(node.boundary),
                merge=False,
            )
            # Metadata is data, including keys that name constructor arguments.
            g._nodes[new_nid].metadata = copy.deepcopy(node.metadata)
            id_map[nid] = new_nid
        for eid, edge in other._edges.items():
            new_eid = g.add_edge(
                id_map[edge.node_i], id_map[edge.node_j],
                radius=edge.radius, material=copy.deepcopy(edge.material),
                internal_points=edge.internal_points.copy() if edge.internal_points is not None else None,
                segments=edge.segments, allow_parallel=True,
            )
            g._edges[new_eid].metadata = copy.deepcopy(edge.metadata)
        return g

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_networkx(self):
        """Convert to a NetworkX graph, retaining parallel fibers when present."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx required: pip install networkx")
        pairs = [frozenset((e.node_i, e.node_j)) for e in self._edges.values()]
        G = nx.MultiGraph() if len(set(pairs)) < len(pairs) else nx.Graph()
        for nid, node in self._nodes.items():
            G.add_node(
                nid,
                pos=tuple(node.position[:2]) if self._dimension == 2 else tuple(node.position),
                boundary=node.boundary,
                **node.metadata,
            )
        for eid, edge in self._edges.items():
            G.add_edge(
                edge.node_i, edge.node_j,
                radius=edge.radius,
                edge_id=eid,
                segments=edge.segments,
                **edge.metadata,
            )
        return G

    @classmethod
    def from_networkx(cls, G, dimension: int = 2, tolerance: float = 1e-6) -> "StructureGraph":
        """Create a StructureGraph from a NetworkX Graph or MultiGraph.

        Expects nodes to have a 'pos' attribute (2-tuple or 3-tuple).
        """
        g = cls(dimension=dimension, tolerance=tolerance)
        id_map = {}
        for n, data in G.nodes(data=True):
            pos = data.get("pos", (0, 0, 0))
            bnd = data.get("boundary", (False,) * 6)
            nid = g.add_node(pos, boundary=bnd, merge=False)
            id_map[n] = nid
        for u, v, data in G.edges(data=True):
            g.add_edge(
                id_map[u], id_map[v],
                radius=data.get("radius", 0.1),
                segments=data.get("segments", 4),
                allow_parallel=G.is_multigraph(),
            )
        return g

    def to_networkx_lossless(self):
        """Export a detached native-v1 MultiGraph, preserving IDs and ordering.

        Native records live under ``_fibernet``; arbitrary NetworkX topology
        edits are not supported by this complete-record interchange protocol.
        """
        from fibernet.io.networkx_adapter import _to_networkx_lossless
        return _to_networkx_lossless(self)

    @classmethod
    def from_networkx_lossless(cls, G) -> "StructureGraph":
        """Import current native-v1 records with strict identity validation.

        The stored dimension/tolerance are authoritative. Use ``from_networkx``
        for the existing projection importer and its configuration overrides.
        """
        from fibernet.io.networkx_adapter import _from_networkx_lossless
        return _from_networkx_lossless(G, cls)

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Export to numpy arrays for simulation.

        Returns
        -------
        positions : (N, 3) float64
        edges : (M, 2) int32
        radii : (M,) float64
        """
        if not self._nodes:
            return np.empty((0, 3)), np.empty((0, 2), dtype=np.int32), np.empty((0,))
        id_list = sorted(self._nodes.keys())
        id_to_idx = {nid: idx for idx, nid in enumerate(id_list)}
        positions = np.array([self._nodes[nid].position for nid in id_list])
        edges = np.array(
            [[id_to_idx[e.node_i], id_to_idx[e.node_j]] for e in self._edges.values()],
            dtype=np.int32,
        ).reshape(-1, 2)
        radii = np.array([e.radius for e in self._edges.values()])
        return positions, edges, radii

    def to_fiber_network(self, *, profile=None):
        """Convert through the existing endpoint-chord compatibility bridge.

        With no profile, return the unchanged legacy ``FiberNetwork``.
        ``FiberNetworkProfile`` opts into bounded, explicit loss acceptance and
        returns ``(FiberNetwork, detached_report)``. The report describes this
        conversion only, not downstream mesh export or a lossless roundtrip.
        """
        if profile is not None:
            from fibernet.core.fiber_network_profile import _checked_to_fiber_network
            return _checked_to_fiber_network(self, profile)
        from fibernet.core.fiber import Fiber
        from fibernet.core.network import FiberNetwork, Crosslink

        net = FiberNetwork(dimension=self._dimension, box_size=self.box_size)
        net.metadata = copy.deepcopy(self._metadata)

        node_to_fibers: Dict[int, List[Tuple[int, float]]] = {}
        for nid in self._nodes:
            node_to_fibers[nid] = []

        for eid, edge in self._edges.items():
            pi = self._nodes[edge.node_i].position
            pj = self._nodes[edge.node_j].position
            dist = np.linalg.norm(pj - pi)
            if dist < 1e-12:
                continue
            fid = net.num_fibers
            n_seg = max(edge.segments, max(4, int(dist / max(edge.radius * 2, 0.05))))
            fiber = Fiber.straight(
                pi, pj, radius=edge.radius, material=copy.deepcopy(edge.material),
                fiber_id=fid, segments=n_seg,
            )
            net.add_fiber(fiber)
            node_to_fibers[edge.node_i].append((fid, 0.0))
            node_to_fibers[edge.node_j].append((fid, 1.0))

        for nid, fiber_list in node_to_fibers.items():
            if len(fiber_list) < 2:
                continue
            pos = self._nodes[nid].position
            for i in range(len(fiber_list)):
                for j in range(i + 1, len(fiber_list)):
                    fi, pi_val = fiber_list[i]
                    fj, pj_val = fiber_list[j]
                    net.add_crosslink(Crosslink(
                        fiber_i=fi, fiber_j=fj,
                        param_i=pi_val, param_j=pj_val,
                        position=pos.copy(),
                        crosslink_type="welded",
                    ))
        return net

    @classmethod
    def from_fiber_network(cls, net, tolerance: float = 1e-6) -> "StructureGraph":
        """Create a StructureGraph from a FiberNetwork (approximate)."""
        g = cls(dimension=net.dimension, tolerance=tolerance, box_size=net.box_size)
        for fiber in net.fibers:
            n_start = g.add_node(fiber.start_point, merge=True)
            n_end = g.add_node(fiber.end_point, merge=True)
            g.add_edge(n_start, n_end, radius=fiber.radius, material=fiber.material,
                       allow_parallel=True)
        return g

    # ------------------------------------------------------------------
    # Hash / fingerprint for caching
    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """Compute a deterministic hash of the graph topology and geometry.

        Useful for caching simulation results and detecting changes.
        """
        h = hashlib.sha256()
        h.update(f"dim={self._dimension}".encode())
        for nid in sorted(self._nodes.keys()):
            pos = self._nodes[nid].position
            h.update(f"n{nid}:{pos[0]:.10f},{pos[1]:.10f},{pos[2]:.10f}".encode())
        for eid in sorted(self._edges.keys()):
            e = self._edges[eid]
            h.update(f"e{eid}:{e.node_i}-{e.node_j}:r{e.radius:.6f}".encode())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize graph schema 3.1, including each edge's material.

        Material records contain only JSON data, never serialized Python objects.
        """
        def material_record(material):
            if not isinstance(material, Material):
                raise ValueError('edge material must be a Material instance')
            try:
                return _material_record(material.to_dict())
            except (AttributeError, TypeError) as exc:
                raise ValueError('edge material cannot be serialized as JSON data') from exc

        return {
            "version": "3.1",
            "dimension": self._dimension,
            "tolerance": self._tolerance,
            "next_node_id": self._next_node_id,
            "next_edge_id": self._next_edge_id,
            "box_size": self._box_size.tolist() if self._box_size is not None else None,
            "metadata": self._metadata,
            "nodes": [
                {
                    "id": nid,
                    "position": node.position.tolist(),
                    "boundary": [bool(b) for b in node.boundary],
                    "metadata": node.metadata,
                }
                for nid, node in sorted(self._nodes.items())
            ],
            "edges": [
                {
                    "id": eid,
                    "node_i": edge.node_i,
                    "node_j": edge.node_j,
                    "radius": edge.radius,
                    "material": material_record(edge.material),
                    "segments": edge.segments,
                    "internal_points": edge.internal_points.tolist() if edge.internal_points is not None else None,
                    "metadata": edge.metadata,
                }
                for eid, edge in sorted(self._edges.items())
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructureGraph":
        """Deserialize without renumbering nodes or parallel fibers.

        Legacy records without IDs use their zero-based list positions.
        Missing next-ID counters are inferred from the largest stored ID.
        Legacy missing materials use a fresh generic Material. Versions 3.0,
        3.1 and version-less records are accepted; unknown versions are rejected.
        """
        if data.get('version', '3.0') not in ('3.0', '3.1'):
            raise ValueError('unsupported StructureGraph schema version')
        g = cls(
            dimension=data.get("dimension", 2),
            tolerance=data.get("tolerance", 1e-6),
            box_size=data.get("box_size"),
        )
        def identifier(value, name):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
            return int(value)

        g._metadata = copy.deepcopy(data.get("metadata", {}))
        for index, nd in enumerate(data.get("nodes", [])):
            nid = identifier(nd.get("id", index), 'node ID')
            if nid in g._nodes:
                raise ValueError(f'duplicate node ID: {nid}')
            g._next_node_id = nid
            g.add_node(
                nd["position"],
                boundary=tuple(nd.get("boundary", (False,) * 6)),
                merge=False,
            )
            g._nodes[nid].metadata = copy.deepcopy(nd.get("metadata", {}))
        for index, ed in enumerate(data.get("edges", [])):
            eid = identifier(ed.get("id", index), 'edge ID')
            if eid in g._edges:
                raise ValueError(f'duplicate edge ID: {eid}')
            node_i = identifier(ed['node_i'], 'edge endpoint')
            node_j = identifier(ed['node_j'], 'edge endpoint')
            if node_i == node_j:
                raise ValueError('serialized edges must have distinct endpoints')
            g._next_edge_id = eid
            ip = np.array(ed["internal_points"]) if ed.get("internal_points") is not None else None
            if ip is not None:
                # JSON [] loses the coordinate axis of a valid empty curve.
                if ip.shape == (0,):
                    ip = np.empty((0, 3))
                elif ip.ndim != 2 or ip.shape[1] != 3:
                    raise ValueError('serialized internal_points must have shape (M, 3)')
            try:
                material = (Material.from_dict(_material_record(ed['material']))
                            if 'material' in ed else Material())
            except (ValueError, TypeError, ZeroDivisionError) as exc:
                raise ValueError(f'invalid material for edge {eid}: {exc}') from exc
            g.add_edge(
                node_i, node_j,
                radius=ed.get("radius", 0.1),
                material=material,
                internal_points=ip,
                segments=ed.get("segments", 4),
                allow_parallel=True,
            )
            g._edges[eid].metadata = copy.deepcopy(ed.get("metadata", {}))
        for name, entries in (('node', g._nodes), ('edge', g._edges)):
            minimum = max(entries, default=-1) + 1
            counter = identifier(data.get(f'next_{name}_id', minimum), f'next {name} ID')
            if counter < minimum:
                raise ValueError(f'next {name} ID must exceed all stored IDs')
            setattr(g, f'_next_{name}_id', counter)
        return g

    def save_json(self, path: str):
        """Save graph JSON, preserving an existing file if serialization fails.

        Serialization completes before opening the destination. Parent directories
        are not created. This protects against serialization errors, not partial
        filesystem writes or interruption after the destination has been opened.
        """
        payload = json.dumps(self.to_dict(), indent=2)
        with open(path, "w") as f:
            f.write(payload)

    @classmethod
    def load_json(cls, path: str) -> "StructureGraph":
        """Load graph from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        box_str = ""
        if self._box_size is not None:
            box_str = f", box={self._box_size[:2].tolist()}"
        return (
            f"StructureGraph(dim={self._dimension}, "
            f"nodes={self.num_nodes}, edges={self.num_edges}{box_str})"
        )

    def summary(self) -> str:
        """Return a multi-line summary string."""
        bb_min, bb_max = self.bounding_box()
        n_comp = len(self.connected_components())
        lines = [
            f"StructureGraph (dim={self._dimension})",
            f"  Nodes: {self.num_nodes}",
            f"  Edges: {self.num_edges}",
            f"  Components: {n_comp}",
            f"  Bounding box: {bb_min.tolist()} → {bb_max.tolist()}",
            f"  Total edge length: {self.total_edge_length():.4f}",
        ]
        if self._box_size is not None:
            lines.append(f"  Box size: {self._box_size.tolist()}")
        if self._metadata:
            lines.append(f"  Metadata: {self._metadata}")
        return "\n".join(lines)

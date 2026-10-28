"""Provider-free geometry-policy preflight for an existing StructureGraph.

This module reads the existing ``graph.edges.items()`` duck interface and
``edge.internal_points``. It neither creates a graph nor evaluates mechanics.
``n_pts_per_side``-style actual nodes are not edge-owned internal points.

A returned record describes the selected endpoint-chord interpretation. It is
not evidence that a solver ran or that curve mechanics were preserved. Only a
successful solver caller may attach it to a result.
"""


class GeometryPolicyError(ValueError):
    """Base for geometry policy selection or acceptance failures."""


class InvalidGeometryPolicyError(GeometryPolicyError):
    """The requested geometry policy is outside the closed supported set."""


class UnsupportedInternalPointsError(GeometryPolicyError):
    """Strict endpoint-only acceptance rejected edge-owned internal points."""

    def __init__(self, edge_ids):
        # StructureGraph owns integer stable edge IDs. Do not sort, renumber,
        # collapse parallel edges, or retain its mutable edge mapping/list.
        self.rejected_edge_ids = tuple(edge_ids)
        super().__init__(
            "geometry_policy='reject_internal_points' rejects nonempty "
            "edge-owned internal_points on stable edge IDs "
            + repr(self.rejected_edge_ids)
            + ". Choose geometry_policy='endpoint_chord' explicitly to ignore "
            "those points mechanically, or author an actual-node straight-edge "
            "StructureGraph with separately specified mechanics. Neither choice "
            "establishes curved-beam mechanical equivalence."
        )


def validate_geometry_policy(geometry_policy):
    """Return one exact built-in string policy, without reading any graph.

    Closed values are ``reject_internal_points`` and ``endpoint_chord``.
    No coercion, case folding, alias, boolean, or str subclass is accepted.
    The separate validator permits streamed callers to reject invalid
    options before provider construction, without preloading their graphs.
    """
    if type(geometry_policy) is not str or geometry_policy not in (
        "reject_internal_points", "endpoint_chord"
    ):
        raise InvalidGeometryPolicyError(
            "geometry_policy must be exactly 'reject_internal_points' or "
            "'endpoint_chord' (a built-in str)"
        )
    return geometry_policy


def preflight_taichi_geometry(graph, *, geometry_policy="reject_internal_points"):
    """Validate geometry acceptance and return a fresh JSON-native record.

    The authoritative graph contract supplies insertion-ordered integer edge
    IDs and internal_points that are None or a sized point array. Only None or
    length zero is accepted under the strict policy. All nonempty arrays are
    rejected, including collinear, duplicate, off-chord, or backtracking points;
    no coordinates, collinearity tolerance, radius, material, or units are read.

    Actual intermediate graph nodes connected by independent straight edges
    remain eligible. This says nothing about their mechanical equivalence to a
    curve. Parallel physical edges remain distinct because the edge mapping is
    scanned as-is, never grouped by endpoints or reconstructed.

    This is not general graph/schema validation: malformed duck objects keep
    their ordinary attribute/length errors. The record owns its dict/list, and
    uses existing immutable integer IDs without conversion. It must not be
    retroactively added to legacy results as proof of their interpretation.
    """
    policy = validate_geometry_policy(geometry_policy)
    internal_point_edge_ids = []
    for edge_id, edge in graph.edges.items():
        points = edge.internal_points
        if points is not None and len(points) > 0:
            internal_point_edge_ids.append(edge_id)

    if policy == "reject_internal_points" and internal_point_edge_ids:
        raise UnsupportedInternalPointsError(internal_point_edge_ids)

    return {
        "schema": "taichi_endpoint_chord/1",
        "policy": policy,
        "representation": "endpoint_chord",
        "ignored_internal_point_edge_ids": internal_point_edge_ids,
    }

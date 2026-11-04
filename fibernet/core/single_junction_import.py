"""Opt-in conversion of exact legacy welded crossings.

The single-junction entry point retains its original narrow contract. The
separate multi-junction entry point accepts bounded explicit welded networks.
Both map topology and geometry without claiming mechanical equivalence; neither
snaps nearby points. The legacy endpoint-only converter remains unchanged.
"""
import copy
import math
from numbers import Integral, Real

import numpy as np

from fibernet.core.fiber import Fiber, CrossSection
from fibernet.core.material import Material
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.core.structure_graph import StructureGraph


def _real_coordinates(value, label):
    """Read plain real coordinates without discarding masks or imaginary parts."""
    def contains_mask(item):
        if np.ma.isMaskedArray(item) or np.ma.is_masked(item):
            return True
        if isinstance(item, (list, tuple)):
            return any(contains_mask(child) for child in item)
        return False
    if contains_mask(value):
        raise ValueError(label + ' must not contain masked values')
    def check_integers(item):
        if isinstance(item, (bool, np.bool_)) or (isinstance(item, np.ndarray) and item.dtype.kind not in 'iuf'):
            raise ValueError(label + ' must contain real non-boolean coordinates')
        if isinstance(item, (list, tuple)):
            for child in item:
                check_integers(child)
        elif isinstance(item, np.ndarray) and item.dtype.kind in 'iu':
            if item.size > 30000:
                raise ValueError(label + ' sample limit exceeded')
            for child in item.flat:
                check_integers(child)
        elif isinstance(item, Integral) and not isinstance(item, (bool, np.bool_)):
            try:
                converted = float(item)
                exact = math.isfinite(converted) and int(converted) == int(item)
            except (OverflowError, ValueError):
                exact = False
            if not exact:
                raise ValueError(label + ' cannot be represented exactly as float64')
    check_integers(value)
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(label + ' must contain real coordinates') from exc
    if raw.dtype.kind not in 'iuf':
        raise ValueError(label + ' must contain real coordinates')
    if raw.size > 30000:
        raise ValueError(label + ' sample limit exceeded')
    with np.errstate(over='ignore', invalid='ignore'):
        result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(label + ' must contain finite float64 coordinates')
    if raw.dtype.kind in 'iu':
        exact = all(int(source) == int(converted)
                    for source, converted in zip(raw.flat, result.flat))
    else:
        exact = bool(np.array_equal(raw, result.astype(raw.dtype)))
    if not exact:
        raise ValueError(label + ' cannot be represented exactly as float64')
    return result


def convert_single_welded_crosslink(network):
    """Return ``(graph, mapping)`` for two straight fibers and one exact weld.

    Each input centerline must be an index-uniform straight line and the weld
    must be strictly interior on both fibers. Parameters are interpreted by
    linear interpolation between successive centerline samples, consistent with
    the legacy generators. Both interpolated positions and the stored weld
    position must be exactly equal in float64; coordinates must be exactly representable
    as float64 and no geometric snapping is done.
    Nonperiodic circular fibers with positive radii, ordinary materials and
    infinite weld strength/stiffness are required. Each centerline is limited
    to 10,000 samples; 2D coordinates must have zero z. Fiber-specific section
    parameters are unsupported. Node IDs and edge IDs are newly allocated;
    mapping keys refer to fiber *list indices*, not ``fiber_id`` values.

    The output contains four straight edges with four segments each. Source
    discretization, crosslink strength/stiffness, and cross-section metadata are
    not represented. Radius and detached Material copies are retained. Network
    metadata and box are copied. No other coincident endpoints are merged, and
    this function does not infer mechanics, constraints or units.
    """
    if not isinstance(network, FiberNetwork):
        raise TypeError('network must be a FiberNetwork')
    if (isinstance(network.dimension, (bool, np.bool_)) or
            not isinstance(network.dimension, Integral) or network.dimension not in (2, 3)):
        raise ValueError('network dimension must be 2 or 3')
    if network.periodic is not False:
        raise ValueError('periodic networks are unsupported')
    if len(network.fibers) != 2 or len(network.crosslinks) != 1:
        raise ValueError('requires exactly two fibers and one crosslink')
    link = network.crosslinks[0]
    if not isinstance(link, Crosslink) or link.crosslink_type != 'welded':
        raise ValueError('requires one welded Crosslink')
    if (type(link.fiber_i) is not int or type(link.fiber_j) is not int or
            {link.fiber_i, link.fiber_j} != {0, 1}):
        raise ValueError('crosslink must reference the two fiber list indices')
    for name in ('strength', 'stiffness'):
        v = getattr(link, name)
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or v != float('inf'):
            raise ValueError('finite or unspecified weld ' + name + ' is unsupported')
    params = {link.fiber_i: link.param_i, link.fiber_j: link.param_j}
    points = []
    for index, fiber in enumerate(network.fibers):
        if not isinstance(fiber, Fiber) or type(fiber.material) is not Material:
            raise TypeError('each fiber must have an ordinary Material')
        if isinstance(fiber.fiber_id, (bool, np.bool_)) or not isinstance(fiber.fiber_id, Integral):
            raise ValueError('source fiber IDs must be integers')
        if fiber.cross_section != CrossSection.CIRCULAR or fiber.cross_section_params:
            raise ValueError('only circular fibers without section parameters are supported')
        if (isinstance(fiber.radius, (bool, np.bool_)) or not isinstance(fiber.radius, Real)
                or not math.isfinite(fiber.radius) or fiber.radius <= 0):
            raise ValueError('fiber radius must be finite and positive')
        xyz = _real_coordinates(fiber.centerline, 'centerline')
        if xyz.ndim != 2 or xyz.shape[1:] != (3,) or len(xyz) < 2 or not np.isfinite(xyz).all():
            raise ValueError('each centerline must contain finite 3D points')
        if len(xyz) > 10000:
            raise ValueError('centerline sample limit exceeded')
        if network.dimension == 2 and np.any(xyz[:, 2] != 0):
            raise ValueError('2D centerlines must have zero z')
        if not np.array_equal(xyz, np.linspace(xyz[0], xyz[-1], len(xyz))):
            raise ValueError('only index-uniform straight centerlines are supported')
        p = params[index]
        if isinstance(p, (bool, np.bool_)) or not isinstance(p, Real) or not math.isfinite(p) or not 0 < p < 1:
            raise ValueError('crosslink parameters must be strictly interior')
        t = float(p) * (len(xyz) - 1)
        i = min(int(t), len(xyz) - 2)
        f = t - i
        point = xyz[i] * (1.0 - f) + xyz[i + 1] * f
        points.append(point)
    pos = _real_coordinates(link.position, 'crosslink position')
    if pos.shape != (3,) or not np.isfinite(pos).all() or not np.array_equal(pos, points[0]) or not np.array_equal(pos, points[1]):
        raise ValueError('interpolated fiber points and weld position must exactly coincide; no snapping is performed')
    endpoints = [_real_coordinates(f.centerline[k], 'endpoint') for f in network.fibers for k in (0, -1)]
    if (any(np.array_equal(a, b) for i, a in enumerate(endpoints) for b in endpoints[i+1:])
            or any(np.array_equal(a, pos) for a in endpoints)):
        raise ValueError('coincident fiber endpoints or a zero-length split are unsupported')
    box = None if network.box_size is None else _real_coordinates(network.box_size, 'box size')
    if box is not None and box.shape not in ((2,), (3,)):
        raise ValueError('box size must have two or three coordinates')
    graph = StructureGraph(dimension=int(network.dimension), box_size=box)
    graph.metadata.update(copy.deepcopy(network.metadata))
    junction = graph.add_node(pos, merge=False)
    mapping = {}
    for index, fiber in enumerate(network.fibers):
        start = graph.add_node(fiber.start_point, merge=False)
        end = graph.add_node(fiber.end_point, merge=False)
        edges = [graph.add_edge(start, junction, radius=fiber.radius, material=copy.deepcopy(fiber.material), segments=4, allow_parallel=True),
                 graph.add_edge(junction, end, radius=fiber.radius, material=copy.deepcopy(fiber.material), segments=4, allow_parallel=True)]
        mapping[index] = {'source_fiber_id': int(fiber.fiber_id), 'edge_ids': edges}
    return graph, {'crosslink_index': 0, 'junction_node_id': junction, 'fibers': mapping}


_MAX_FIBERS = 1024
_MAX_LINKS = 4096
_MAX_POINTS_PER_FIBER = 10000
_MAX_TOTAL_POINTS = 100000


def _bounded_vector(value, label, lengths):
    """Reject oversized or nested vectors before recursively reading values."""
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or len(value) not in lengths:
            raise ValueError(label + ' has unsupported shape')
    elif isinstance(value, (list, tuple)):
        if len(value) not in lengths or any(isinstance(x, (list, tuple, np.ndarray)) for x in value):
            raise ValueError(label + ' has unsupported shape')
    else:
        raise ValueError(label + ' has unsupported shape')
    return _real_coordinates(value, label)


def convert_exact_welded_crosslinks(network):
    """Return ``(graph, report)`` for bounded, explicitly welded straight fibers.

    Crosslinks must reference distinct fiber list indices, use strictly interior
    index-interpolated parameters, and coincide exactly in float64 with both
    centerlines and their stored position. One record per unordered fiber pair
    is permitted; duplicate, reversed duplicate and multiple contacts on the
    same pair are rejected. A fiber may have multiple welds with different
    fibers. At a coordinate, links connected through a common fiber share one
    graph node; disjoint link groups at the same coordinate remain separate.
    Equal coordinates on one fiber must have the same parameter. No unrecorded
    crossings or coincident endpoints are welded.

    Only nonperiodic, index-uniform straight, circular fibers without section
    parameters, exact ordinary Materials, positive finite radii, and welded
    links with infinite strength and stiffness are accepted. Limits are 1,024
    fibers, 4,096 links, 10,000 samples per fiber and 100,000 samples total.
    Copied metadata and Material contents are not covered by those limits.
    Coordinates must be finite, exactly representable float64 values; in 2D,
    centerline z coordinates are zero. Coincident endpoints or an endpoint at
    any weld are rejected. Parameters must be exactly representable float64.

    The report maps source fiber list indices to original fiber IDs, ordered
    split parameters, junction node IDs and ordered edge IDs. It also maps
    each original crosslink index to its new junction node ID. IDs are newly
    allocated; source discretization and weld strength/stiffness are not
    represented. Each resulting edge has four segments and a detached Material
    copy; radius, network metadata and box are copied. A two-coordinate box is
    normalized by StructureGraph to three coordinates with zero z. No units, boundary
    conditions, physical contact, persistence compatibility or mechanical
    equivalence are inferred.
    """
    if not isinstance(network, FiberNetwork):
        raise TypeError('network must be a FiberNetwork')
    if (isinstance(network.dimension, (bool, np.bool_)) or
            not isinstance(network.dimension, Integral) or network.dimension not in (2, 3)):
        raise ValueError('network dimension must be 2 or 3')
    if network.periodic is not False:
        raise ValueError('periodic networks are unsupported')
    if not 2 <= len(network.fibers) <= _MAX_FIBERS:
        raise ValueError('fiber count must be between 2 and 1024')
    if not 1 <= len(network.crosslinks) <= _MAX_LINKS:
        raise ValueError('crosslink count must be between 1 and 4096')

    coordinates = []
    endpoint_keys = set()
    total_points = 0
    for fiber in network.fibers:
        if not isinstance(fiber, Fiber) or type(fiber.material) is not Material:
            raise TypeError('each fiber must have an ordinary Material')
        if isinstance(fiber.fiber_id, (bool, np.bool_)) or not isinstance(fiber.fiber_id, Integral):
            raise ValueError('source fiber IDs must be integers')
        if fiber.cross_section != CrossSection.CIRCULAR or fiber.cross_section_params:
            raise ValueError('only circular fibers without section parameters are supported')
        if (isinstance(fiber.radius, (bool, np.bool_)) or not isinstance(fiber.radius, Real)
                or not math.isfinite(fiber.radius) or fiber.radius <= 0):
            raise ValueError('fiber radius must be finite and positive')
        try:
            count = len(fiber.centerline)
        except TypeError as exc:
            raise ValueError('centerline must contain samples') from exc
        total_points += count
        if count > _MAX_POINTS_PER_FIBER or total_points > _MAX_TOTAL_POINTS:
            raise ValueError('centerline sample limit exceeded')
        source = fiber.centerline
        if isinstance(source, np.ndarray):
            if source.ndim != 2 or source.shape[1:] != (3,):
                raise ValueError('each centerline must contain 3D points')
        elif isinstance(source, (list, tuple)):
            for row in source:
                if (not isinstance(row, (list, tuple, np.ndarray)) or len(row) != 3 or
                        any(isinstance(item, (list, tuple, np.ndarray)) for item in row)):
                    raise ValueError('each centerline must contain 3D points')
        else:
            raise ValueError('each centerline must contain 3D points')
        xyz = _real_coordinates(fiber.centerline, 'centerline')
        if xyz.ndim != 2 or xyz.shape[1:] != (3,) or len(xyz) < 2:
            raise ValueError('each centerline must contain finite 3D points')
        if network.dimension == 2 and np.any(xyz[:, 2] != 0):
            raise ValueError('2D centerlines must have zero z')
        if not np.array_equal(xyz, np.linspace(xyz[0], xyz[-1], len(xyz))):
            raise ValueError('only index-uniform straight centerlines are supported')
        for endpoint in (xyz[0], xyz[-1]):
            key = tuple(endpoint)
            if key in endpoint_keys:
                raise ValueError('coincident fiber endpoints are unsupported')
            endpoint_keys.add(key)
        coordinates.append(xyz)

    seen_pairs = set()
    # Each fiber's positions map to exactly one index parameter.
    per_fiber = [dict() for _ in coordinates]
    by_position = {}
    for link_index, link in enumerate(network.crosslinks):
        if not isinstance(link, Crosslink) or link.crosslink_type != 'welded':
            raise ValueError('every crosslink must be welded')
        i, j = link.fiber_i, link.fiber_j
        if (type(i) is not int or type(j) is not int or i == j or
                not 0 <= i < len(coordinates) or not 0 <= j < len(coordinates)):
            raise ValueError('crosslinks must reference distinct fiber list indices')
        pair = (min(i, j), max(i, j))
        if pair in seen_pairs:
            raise ValueError('only one crosslink per unordered fiber pair is supported')
        seen_pairs.add(pair)
        for name in ('strength', 'stiffness'):
            value = getattr(link, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or value != float('inf'):
                raise ValueError('finite or unspecified weld ' + name + ' is unsupported')
        position = _bounded_vector(link.position, 'crosslink position', (3,))
        key = tuple(position)
        if key in endpoint_keys:
            raise ValueError('a weld coinciding with any fiber endpoint is unsupported')
        for index, param in ((i, link.param_i), (j, link.param_j)):
            if (isinstance(param, (bool, np.bool_)) or not isinstance(param, Real)
                    or not math.isfinite(param) or not 0 < param < 1):
                raise ValueError('crosslink parameters must be strictly interior')
            p = float(param)
            if p != param:
                raise ValueError('crosslink parameters must be exactly representable as float64')
            xyz = coordinates[index]
            t = p * (len(xyz) - 1)
            k = min(int(t), len(xyz) - 2)
            fraction = t - k
            point = xyz[k] * (1.0 - fraction) + xyz[k + 1] * fraction
            if not np.array_equal(point, position):
                raise ValueError('interpolated points and weld position must exactly coincide; no snapping is performed')
            previous = per_fiber[index].get(key)
            if previous is not None and previous != p:
                raise ValueError('one position on a fiber cannot have multiple parameters')
            per_fiber[index][key] = p
        by_position.setdefault(key, []).append((link_index, i, j))

    # Build connected components of explicit links at each exact coordinate.
    # Merely coincident groups without a link path remain separate nodes.
    groups = []
    for position, links in by_position.items():
        parent = {}
        def find(item):
            parent.setdefault(item, item)
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item
        for _, i, j in links:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
        components = {}
        for link_index, i, j in links:
            root = find(i)
            component = components.setdefault(root, {'fibers': set(), 'links': []})
            component['fibers'].update((i, j))
            component['links'].append(link_index)
        for component in components.values():
            groups.append((min(component['links']), position, component))
    groups.sort(key=lambda group: group[0])

    box = None if network.box_size is None else _bounded_vector(network.box_size, 'box size', (2, 3))
    graph = StructureGraph(dimension=int(network.dimension), box_size=box)
    graph.metadata.update(copy.deepcopy(network.metadata))
    node_for = {}
    link_nodes = [None] * len(network.crosslinks)
    for _, position, component in groups:
        node = graph.add_node(position, merge=False)
        for index in component['fibers']:
            node_for[(index, position)] = node
        for link_index in component['links']:
            link_nodes[link_index] = node
    mapping = {}
    for index, fiber in enumerate(network.fibers):
        start = graph.add_node(coordinates[index][0], merge=False)
        end = graph.add_node(coordinates[index][-1], merge=False)
        splits = sorted((param, node_for[(index, pos)]) for pos, param in per_fiber[index].items())
        nodes = [start] + [node for _, node in splits] + [end]
        edges = [graph.add_edge(a, b, radius=fiber.radius, material=copy.deepcopy(fiber.material),
                                segments=4, allow_parallel=True)
                 for a, b in zip(nodes, nodes[1:])]
        mapping[index] = {'source_fiber_id': int(fiber.fiber_id),
                          'parameters': [param for param, _ in splits],
                          'junction_node_ids': [node for _, node in splits],
                          'edge_ids': edges}
    return graph, {'fibers': mapping, 'crosslink_junction_node_ids': link_nodes}

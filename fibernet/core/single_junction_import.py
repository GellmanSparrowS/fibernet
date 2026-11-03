"""Opt-in narrow conversion of one exact legacy welded crossing.

This is a topology/geometry conversion, not a mechanical-equivalence claim.
The legacy endpoint-only converter is unchanged. Nearby noncoincident points,
curves, multiple crosslinks, periodic networks and other connection types are
outside this initial capability and are rejected rather than snapped.
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

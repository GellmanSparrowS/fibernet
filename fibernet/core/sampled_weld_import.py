"""Convert sampled legacy fiber geometry with explicitly recorded welds."""

from copy import deepcopy
import math
from numbers import Integral, Real

from networkx.utils import UnionFind
import numpy as np

from fibernet.core.fiber import CrossSection, Fiber
from fibernet.core.material import Material
from fibernet.core.network import Crosslink, FiberNetwork
from fibernet.core.single_junction_import import (
    _MAX_FIBERS,
    _MAX_LINKS,
    _MAX_POINTS_PER_FIBER,
    _MAX_TOTAL_POINTS,
    _bounded_vector,
    _real_coordinates,
)
from fibernet.core.structure_graph import StructureGraph


def _sample_index(parameter, count):
    """Resolve a canonical normalized index, without rounding to a nearby knot."""
    if (isinstance(parameter, (bool, np.bool_)) or not isinstance(parameter, Real)
            or not 0 <= parameter <= 1):
        raise ValueError('weld parameter must be a real number in [0, 1]')
    value = float(parameter)
    sample = round(value * (count - 1))
    # A float64 quotient k/(n-1) need not multiply back to integer k.
    if value != parameter or value != sample / (count - 1):
        raise ValueError('weld parameter must be a canonical sample-knot parameter')
    return sample


def convert_sampled_welded_fibers(network):
    """Return a piecewise-straight graph and detached source-to-graph report.

    Keep every sampled segment of bounded, nonperiodic 2D or 3D circular
    fibers. Distinct adjacent samples and finite exact float64 coordinates
    are required. Radii must be positive and finite and materials ordinary.
    Welds must carry infinite strength and stiffness and identify two
    distinct fiber list indices, each with a canonical float64 sample index
    parameter k/(number of samples - 1). Endpoints are valid samples.
    Both identified samples must equal the weld position exactly.
    Welds inside a sample segment are unsupported.

    Welds merge only their stated sample occurrences. Coincidence and
    crossings without a record do not create a junction. Parallel segments
    remain distinct. The report lists ordered sample nodes and segment
    edges per source fiber list index, and a node per source crosslink.
    Graph identities are new. Radius, box and independent material copies
    are retained. The report's ``source_metadata`` holds a detached copy
    of the original metadata. The graph records the import format. Source
    metadata is not interpreted as controls for the new IDs. No units or
    mechanical equivalence are inferred.
    """
    if not isinstance(network, FiberNetwork):
        raise TypeError('network must be a FiberNetwork')
    if (isinstance(network.dimension, (bool, np.bool_))
            or not isinstance(network.dimension, Integral) or network.dimension not in (2, 3)):
        raise ValueError('network dimension must be 2 or 3')
    if network.periodic is not False:
        raise ValueError('periodic networks are unsupported')
    if not 1 <= len(network.fibers) <= _MAX_FIBERS:
        raise ValueError('fiber count must be between 1 and 1024')
    if len(network.crosslinks) > _MAX_LINKS:
        raise ValueError('crosslink count exceeds 4096')

    centerlines = []
    total_samples = 0
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
        source = fiber.centerline
        if not isinstance(source, np.ndarray) or source.ndim != 2 or source.shape[1] != 3:
            raise ValueError('each centerline must be an array of 3D samples')
        count = len(source)
        total_samples += count
        if count > _MAX_POINTS_PER_FIBER or total_samples > _MAX_TOTAL_POINTS:
            raise ValueError('centerline sample limit exceeded')
        if count < 2:
            raise ValueError('each centerline must contain at least two samples')
        samples = _real_coordinates(source, 'centerline')
        if network.dimension == 2 and np.any(samples[:, 2] != 0):
            raise ValueError('2D centerlines must have zero z')
        if np.any(np.all(samples[:-1] == samples[1:], axis=1)):
            raise ValueError('consecutive centerline samples must be distinct')
        centerlines.append(samples)

    connected_samples = UnionFind()
    link_samples = []
    for link in network.crosslinks:
        if not isinstance(link, Crosslink) or link.crosslink_type != 'welded':
            raise ValueError('every crosslink must be welded')
        first, second = link.fiber_i, link.fiber_j
        if (type(first) is not int or type(second) is not int or first == second
                or not 0 <= first < len(centerlines) or not 0 <= second < len(centerlines)):
            raise ValueError('crosslinks must reference distinct fiber list indices')
        for name in ('strength', 'stiffness'):
            value = getattr(link, name)
            if not isinstance(value, Real) or value != math.inf:
                raise ValueError('weld ' + name + ' must be positive infinity')
        position = _bounded_vector(link.position, 'crosslink position', (3,))
        first_sample = _sample_index(link.param_i, len(centerlines[first]))
        second_sample = _sample_index(link.param_j, len(centerlines[second]))
        if (not np.array_equal(centerlines[first][first_sample], position)
                or not np.array_equal(centerlines[second][second_sample], position)):
            raise ValueError('both weld samples must equal its stored position exactly')
        first_key = (first, first_sample)
        connected_samples.union(first_key, (second, second_sample))
        link_samples.append(first_key)

    box = None if network.box_size is None else _bounded_vector(network.box_size, 'box size', (2, 3))
    graph = StructureGraph(dimension=int(network.dimension), box_size=box)
    graph.metadata['sampled_weld_import'] = 'fibernet.sampled_weld_import/1'
    component_nodes = {}
    mapping = {}
    for fiber_index, (fiber, samples) in enumerate(zip(network.fibers, centerlines)):
        nodes = []
        for sample_index, position in enumerate(samples):
            component = connected_samples[fiber_index, sample_index]
            if component not in component_nodes:
                component_nodes[component] = graph.add_node(position, merge=False)
            nodes.append(component_nodes[component])
        edges = [
            graph.add_edge(start, end, radius=fiber.radius, material=deepcopy(fiber.material),
                           segments=4, allow_parallel=True)
            for start, end in zip(nodes, nodes[1:])
        ]
        mapping[fiber_index] = {'source_fiber_id': int(fiber.fiber_id),
                                'sample_node_ids': nodes, 'segment_edge_ids': edges}
    return graph, {
        'fibers': mapping,
        'crosslink_junction_node_ids': [component_nodes[connected_samples[sample]]
                                       for sample in link_samples],
        'source_metadata': deepcopy(network.metadata),
    }

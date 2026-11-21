"""Summarize the face connectivity of a segmented 3D Boolean array."""
from collections import Counter, defaultdict
from itertools import product
from numbers import Real
import json

import numpy as np
from scipy import ndimage

from fibernet import StructureGraph
from fibernet.analysis import analyze_topology

BLOCK = 10
MAX_AXIS = 100
MAX_NODES = 10_000
MAX_EDGES = 50_000
AXES = ('x', 'y', 'z')


def mask_component_graph(mask, *, voxel_size_um=None):
    """Use mask[z, y, x], with a positive isotropic calibration if supplied."""
    if type(mask) is not np.ndarray or mask.ndim != 3 or mask.dtype != np.bool_:
        raise ValueError('mask must be a 3D Boolean numpy.ndarray in [z, y, x] order')
    if any(not 1 <= n <= MAX_AXIS for n in mask.shape):
        raise ValueError('each mask axis must have between 1 and 100 voxels')
    if not np.any(mask):
        raise ValueError('the mask must contain foreground voxels')
    if voxel_size_um is None:
        step, coordinate_unit = 1., 'voxel'
    else:
        if isinstance(voxel_size_um, (bool, np.bool_)) or not isinstance(voxel_size_um, Real):
            raise ValueError('voxel_size_um must be a real isotropic spacing')
        if not 1e-6 <= voxel_size_um <= 1e6:
            raise ValueError('voxel_size_um must be between 1e-6 and 1e6')
        step, coordinate_unit = float(voxel_size_um), 'um'

    graph = StructureGraph(dimension=3, tolerance=1e-9 * step,
                           box_size=[n * step for n in reversed(mask.shape)])
    graph.metadata.update(representation='voxel block-component topology; not a fiber model',
                          shape_zyx=list(mask.shape), array_axis_order='zyx',
                          block_shape_zyx=[BLOCK] * 3, coordinate_unit=coordinate_unit,
                          voxel_size_um=None if voxel_size_um is None else step)
    six = ndimage.generate_binary_structure(3, 1)
    lookup = np.full(mask.shape, -1, dtype=np.int32)
    starts = (range(0, n, BLOCK) for n in mask.shape)
    for start in product(*starts):
        cube = tuple(slice(i, i + BLOCK) for i in start)
        labels, count = ndimage.label(mask[cube], structure=six)
        if len(graph.nodes) + count > MAX_NODES:
            raise ValueError('mask exceeds the example component-node budget')
        ids = np.full(count + 1, -1, dtype=np.int32)
        for local_id, sl in enumerate(ndimage.find_objects(labels), 1):
            members = np.argwhere(labels[sl] == local_id) + [a.start + b for a, b in zip(sl, start)]
            flags = tuple(bool(np.any(members[:, axis] == end))
                          for axis in (2, 1, 0) for end in (0, mask.shape[axis] - 1))
            ids[local_id] = graph.add_node(
                ((members[0, ::-1] + .5) * step).tolist(), boundary=flags, merge=False,
                block_zyx=[i // BLOCK for i in start], first_member_zyx=members[0].tolist(),
                voxel_count=len(members),
                voxel_centroid_xyz=((members[:, ::-1].mean(axis=0) + .5) * step).tolist())
        lookup[cube] = ids[labels]

    contacts = defaultdict(lambda: [0, 0, 0])
    for axis in range(3):
        view = np.moveaxis(lookup, axis, 0)
        for i in range(BLOCK, len(view), BLOCK):
            lower, upper = view[i - 1], view[i]
            touching = (lower >= 0) & (upper >= 0)
            for a, b in zip(lower[touching], upper[touching]):
                contacts[tuple(sorted((int(a), int(b))))][axis] += 1
                if len(contacts) > MAX_EDGES:
                    raise ValueError('mask exceeds the example contact-edge budget')
    for (a, b), witnesses in sorted(contacts.items()):
        graph.add_edge(a, b, segments=1, relation='cross-block occupied voxel faces',
                       face_witnesses_zyx=witnesses)
    return graph


def component_summary(graph):
    """Report weighted component sizes and occupied opposite face contacts."""
    topology = analyze_topology(graph)
    mass = Counter()
    flags = defaultdict(lambda: np.zeros(6, dtype=bool))
    for node_id, component in zip(topology.node_ids, topology.component_by_node):
        node = graph.nodes[node_id]
        mass[component] += node.metadata['voxel_count']
        flags[component] |= node.boundary
    return {'nodes': topology.n_nodes, 'edges': topology.n_physical_edges,
            'foreground_voxels': sum(mass.values()), 'component_sizes_voxels': sorted(mass.values()),
            'spanning_sizes_voxels': {
                axis: sorted(mass[c] for c, touch in flags.items() if touch[2*i] and touch[2*i+1])
                for i, axis in enumerate(AXES)}}


def main():
    # The diagonal pair straddles a block boundary but touches no voxel face.
    mask = np.zeros((20, 10, 10), dtype=bool)
    mask[:, 0, :2] = True
    mask[9, 3, 3] = mask[10, 4, 3] = mask[0, 9, 9] = True
    print(json.dumps(component_summary(mask_component_graph(mask)), indent=2))


if __name__ == '__main__':
    main()

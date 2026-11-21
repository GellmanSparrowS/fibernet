"""Small independent face-connectivity oracle for the topology-only example."""
from itertools import product

import numpy as np
import pytest

from examples import segmented_mask_topology as example
from fibernet import StructureGraph


def voxel_oracle(mask):
    """Traverse raw occupied voxels; know nothing about graph blocks or IDs."""
    remaining = set(map(tuple, np.argwhere(mask)))
    sizes, spans = [], {a: [] for a in 'xyz'}
    while remaining:
        stack = [remaining.pop()]
        members = []
        while stack:
            p = stack.pop()
            members.append(p)
            for axis, delta in product(range(3), (-1, 1)):
                neighbor = tuple(x + (delta if j == axis else 0) for j, x in enumerate(p))
                if neighbor in remaining:
                    stack.append(neighbor)
                    remaining.remove(neighbor)
        sizes.append(len(members))
        for col, axis in enumerate('zyx'):
            values = {p[col] for p in members}
            if 0 in values and mask.shape[col] - 1 in values:
                spans[axis].append(len(members))
    return sorted(sizes), {a: sorted(s) for a, s in spans.items()}


def test_face_contacts_weights_axis_and_roundtrip(tmp_path):
    mask = np.zeros((20, 10, 30), dtype=bool)
    mask[:, 0, :2] = True  # Two distinct witnesses for one collapsed z contact.
    mask[9, 3, 3] = mask[10, 4, 3] = True  # Diagonal across the z cut is not contact.
    mask[2, 8, :] = True  # X spans two internal cuts without spanning z.
    graph = example.mask_component_graph(mask, voxel_size_um=1.3)
    path = tmp_path / 'topology.json'
    graph.save_json(path)
    saved = StructureGraph.load_json(path)
    assert saved.to_dict() == graph.to_dict()
    report = example.component_summary(saved)
    masses, spans = voxel_oracle(mask)
    assert report['foreground_voxels'] == int(mask.sum()) == 72
    assert report['component_sizes_voxels'] == masses == [1, 1, 30, 40]
    assert report['spanning_sizes_voxels'] == spans == {'x': [30], 'y': [], 'z': [40]}
    assert (report['nodes'], report['edges']) == (7, 3)
    assert sorted(tuple(e.metadata['face_witnesses_zyx']) for e in saved.edges.values()) == [
        (0, 0, 1), (0, 0, 1), (2, 0, 0)]
    for node in saved.nodes.values():
        index = node.metadata['first_member_zyx']
        assert mask[tuple(index)]
        np.testing.assert_array_equal(node.position, (np.array(index[::-1]) + .5) * 1.3)


def test_partial_blocks_reach_actual_array_faces(tmp_path):
    mask = np.zeros((13, 11, 25), dtype=bool)
    mask[12, 4, :] = True  # x crosses the 10 and 20 cuts, then ends at x = 24.
    mask[0, :, 1] = True  # y crosses into a one-voxel trailing block.
    mask[3, 3, 24] = True  # One outer-face voxel cannot span x.
    graph = example.mask_component_graph(mask)
    path = tmp_path / 'partial.json'
    graph.save_json(path)
    saved = StructureGraph.load_json(path)
    masses, spans = voxel_oracle(mask)
    summary = example.component_summary(saved)
    assert summary['component_sizes_voxels'] == masses == [1, 11, 25]
    assert summary['spanning_sizes_voxels'] == spans == {'x': [25], 'y': [11], 'z': []}
    assert (summary['nodes'], summary['edges']) == (6, 3)
    assert any(n.metadata['first_member_zyx'] == [3, 3, 24] and n.boundary[1]
               for n in saved.nodes.values())


def test_singleton_axis_foreground_touches_both_index_faces():
    mask = np.zeros((1, 3, 11), dtype=bool)
    mask[0, 1, 4] = True
    graph = example.mask_component_graph(mask)
    (node,) = graph.nodes.values()
    assert node.boundary == (False, False, False, False, True, True)
    masses, spans = voxel_oracle(mask)
    report = example.component_summary(graph)
    assert report['component_sizes_voxels'] == masses == [1]
    assert report['spanning_sizes_voxels'] == spans == {'x': [], 'y': [], 'z': [1]}


@pytest.mark.parametrize('mask', [np.zeros((10, 10, 10), dtype=np.uint8),
                                  np.zeros((0, 10, 10), dtype=bool),
                                  np.zeros((10, 10), dtype=bool),
                                  np.zeros((10, 10, 10), dtype=bool),
                                  np.zeros((110, 10, 10), dtype=bool)])
def test_rejects_nonboolean_empty_or_outside_example_domain(mask):
    with pytest.raises(ValueError, match='mask'):
        example.mask_component_graph(mask)


def test_rejects_invalid_calibration_and_guards_graph_growth(monkeypatch):
    mask = np.zeros((20, 10, 10), dtype=bool)
    mask[:, :1, :1] = True
    for spacing in (True, np.nan, 0, np.inf):
        with pytest.raises(ValueError, match='voxel_size_um'):
            example.mask_component_graph(mask, voxel_size_um=spacing)
    monkeypatch.setattr(example, 'MAX_NODES', 1)
    with pytest.raises(ValueError, match='component-node budget'):
        example.mask_component_graph(mask)
    monkeypatch.setattr(example, 'MAX_NODES', 2)
    monkeypatch.setattr(example, 'MAX_EDGES', 0)
    with pytest.raises(ValueError, match='contact-edge budget'):
        example.mask_component_graph(mask)

"""Verify that the APP graph retains every compiled manufacturing fiber."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def main():
    from fslab.manufacturing import compile_planar
    from fslab.structure import StructureFactory

    for unit in ('square', 'hexagon', 'ring', 'kagome'):
        for points in (0, 1, 2):
            factory = StructureFactory(unit=unit, grid_x=2, grid_y=2,
                                       n_pts_per_side=points)
            network = compile_planar(factory)
            graph = factory.build()
            assert graph.num_nodes == len(network.positions), (unit, points, 'nodes')
            assert graph.num_edges == len(network.edges), (unit, points, 'edges')
            assert np.array_equal(graph.node_positions(), network.positions), (unit, points, 'positions')
            assert np.array_equal(graph.edge_array(), network.edges), (unit, points, 'connectivity')
            assert len(graph.metadata['route_edges']) == graph.num_edges
            assert max(graph.metadata['route_edges']) < graph.num_edges
            assert graph.copy().num_edges == graph.num_edges
            assert type(graph).from_dict(graph.to_dict()).num_edges == graph.num_edges
    print('[selftest_graph_contract] PASS')


if __name__ == '__main__':
    main()

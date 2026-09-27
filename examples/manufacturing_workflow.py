"""Independent planar and curved-route example for FiberNet.

Run: python -m examples.manufacturing_workflow --output-dir demo_routes
Saved NPZ files contain graph geometry and edge-ordered Euler routes.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from fibernet.gen.manufacturing import (PlanarManufacturingConfig,
                                       compile_planar, compile_surface)


class ManufacturingWorkflow:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def _save(self, name, network):
        destination = self.output_dir / (name + '.npz')
        temporary = destination.with_suffix('.tmp.npz')
        try:
            np.savez_compressed(temporary, reference=network.reference,
                                positions=network.positions, edges=network.edges,
                                route_nodes=network.route_nodes,
                                route_edges=network.route_edges)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        with np.load(destination, allow_pickle=False) as saved:
            if not np.array_equal(saved['edges'], network.edges):
                raise AssertionError('saved route edges changed')
        return destination

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        config = PlanarManufacturingConfig(
            unit='kagome', grid_x=2, grid_y=2, n_pts_per_side=2,
            line_displacements=[[0.02, -0.01], [0.01, 0.03]], seed=23)
        planar = compile_planar(config)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.2],
                             [0, 1, 0.1], [1, 1, 0.2], [2, 1, 0.4]])
        faces = [[0, 1, 4, 3], [1, 2, 5, 4]]
        curved = compile_surface(vertices, faces, config)
        summary = {}
        for name, network in (('planar', planar), ('surface', curved)):
            if (network.health['components'] != 1 or network.health['odd'] != 0
                    or len(network.route_edges) != len(network.edges)):
                raise AssertionError(name + ' has no continuous closed route')
            self._save(name, network)
            summary[name] = {
                'nodes': len(network.positions),
                'edges': len(network.edges),
                'topology_id': network.topology_id,
                'route_closed': bool(network.route_nodes[0] == network.route_nodes[-1]),
            }
        summary['scope'] = ('Euler route and geometry only; printer, material, '
                            'collision and bond constraints are not certified')
        destination = self.output_dir / 'summary.json'
        temporary = destination.with_suffix('.tmp.json')
        try:
            temporary.write_text(json.dumps(summary, indent=2) + '\n',
                                 encoding='utf-8')
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        print('[manufacturing_workflow] planar and surface routes saved')
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='demo_routes')
    args = parser.parse_args()
    ManufacturingWorkflow(args.output_dir).run()

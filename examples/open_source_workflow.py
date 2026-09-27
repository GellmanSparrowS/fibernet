"""Runnable FiberNet workflow from graph generation to tensile recruitment.

Run: python -m examples.open_source_workflow --output-dir demo_output
The FEM example uses a small strain and does not calibrate experimental forces.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import fibernet as fn
from fibernet.analysis import analyze_tensile_recruitment


class OpenSourceWorkflow:
    def __init__(self, output_dir, target_stretch=1.01, alpha=0.05):
        self.output_dir = Path(output_dir)
        self.target_stretch = float(target_stretch)
        self.alpha = float(alpha)

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        graph = fn.pattern_2d(unit='honeycomb', box=(10, 10), grid=(2, 2),
                              radius=0.05, seed=23)
        figure = fn.show(graph, theme='publication',
                         save_path=str(self.output_dir / 'network.png'))
        import matplotlib.pyplot as plt
        plt.close(figure)

        result_path = self.output_dir / 'fem_result.json'
        simulation = fn.simulate(graph, backend='fem', mode='stretch',
                                 strain=self.target_stretch,
                                 save_path=str(result_path))
        restored = fn.SimResult.load(str(result_path))
        if not np.isclose(simulation.max_displacement,
                          restored.max_displacement):
            raise AssertionError('FEM result did not round-trip')

        node_ids = list(graph.nodes)
        node_index = {node_id: i for i, node_id in enumerate(node_ids)}
        positions = np.asarray([graph.nodes[i].position for i in node_ids],
                               dtype=float)[:, :2]
        edges = np.asarray([[node_index[e.node_i], node_index[e.node_j]]
                            for e in graph.edges.values()], dtype=int)
        final_xy = np.asarray(simulation.deformed_positions)[:, :2]
        start_length = np.linalg.norm(
            positions[edges[:, 1]] - positions[edges[:, 0]], axis=1)
        final_length = np.linalg.norm(
            final_xy[edges[:, 1]] - final_xy[edges[:, 0]], axis=1)
        edge_strain = np.stack([np.zeros(len(edges)),
                                final_length / start_length - 1.0])
        xmin, xmax = positions[:, 0].min(), positions[:, 0].max()
        grip_width = 0.1 * (xmax - xmin)
        left = np.flatnonzero(positions[:, 0] <= xmin + grip_width)
        right = np.flatnonzero(positions[:, 0] >= xmax - grip_width)
        recruitment = analyze_tensile_recruitment(
            edge_strain, edges, left, right, n_nodes=len(node_ids),
            alpha=self.alpha)

        summary = {
            'nodes': graph.num_nodes,
            'edges': graph.num_edges,
            'target_stretch': self.target_stretch,
            'max_displacement': simulation.max_displacement,
            'max_force': simulation.max_force,
            'recruited_edge_fraction': float(recruitment.active_edges[-1].mean()),
            'spanning_edge_fraction': float(recruitment.backbone_frac[-1]),
            'first_spanning_frame': int(recruitment.perc_frame),
            'analysis': 'positive axial edge strain threshold, not total force flow',
        }
        (self.output_dir / 'summary.json').write_text(
            json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        print('[open_source_workflow] complete: %d nodes, %d edges' %
              (graph.num_nodes, graph.num_edges))
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='demo_output')
    parser.add_argument('--target-stretch', type=float, default=1.01)
    parser.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args()
    OpenSourceWorkflow(args.output_dir, args.target_stretch, args.alpha).run()

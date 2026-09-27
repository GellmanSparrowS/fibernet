"""Small, resumable intervention audit for thresholded tensile recruitment.

Run: python benchmarks/percolation_pruning_pilot.py
The output is exploratory numerical evidence, not a manufacturing claim.
"""
import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'FiberScope'))

from fslab.cell_rules import graph_health
from fslab.engine2 import Engine2, Engine2Config
from fslab.percolation import compute_percolation
from fslab.simcache import ENGINE_SRC_HASH, ENGINE_VERSION
from fslab.structure import StructureFactory


@dataclass
class PilotConfig:
    units: tuple = ('square', 'hexagon', 'ring', 'kagome')
    grid: int = 2
    points_per_edge: int = 1
    target_stretch: float = 1.4
    increments: int = 20
    steps: int = 2000
    remove_fraction: float = 0.10
    alpha: float = 0.05
    seed: int = 23
    max_nodes: int = 3000
    max_edges: int = 5000


class PruningPilot:
    def __init__(self, config=None):
        self.config = config or PilotConfig()

    @staticmethod
    def connected(n_nodes, edges, removed):
        adjacency = [[] for _ in range(n_nodes)]
        for eid, (a, b) in enumerate(edges):
            if eid not in removed:
                adjacency[int(a)].append(int(b))
                adjacency[int(b)].append(int(a))
        seen, stack = {0}, [0]
        while stack:
            for node in adjacency[stack.pop()]:
                if node not in seen:
                    seen.add(node)
                    stack.append(node)
        return len(seen) == n_nodes

    def choose(self, graph, ordered, count):
        edges = np.asarray(graph.edge_array(), dtype=int)
        selected = set()
        for eid in ordered:
            candidate = selected | {int(eid)}
            if self.connected(graph.num_nodes, edges, candidate):
                selected = candidate
                if len(selected) == count:
                    break
        if len(selected) != count:
            raise ValueError('cannot match removal count while keeping graph connected')
        return sorted(selected)

    def simulate(self, graph):
        cfg = self.config
        engine_cfg = Engine2Config(
            target_stretch=cfg.target_stretch, num_steps=cfg.steps,
            n_increments=cfg.increments, use_contact=False,
            use_bending=True)
        return Engine2(graph, engine_cfg).run()

    def summary(self, graph, run, removed_length, original_length):
        percolation = compute_percolation(run, alpha=self.config.alpha)
        health = graph_health(graph.node_positions(), graph.edge_array())
        return {
            'nodes': int(graph.num_nodes),
            'edges': int(graph.num_edges),
            'removed_length_fraction': float(removed_length / original_length),
            'final_raw_reaction': float(run.force_curve[-1]),
            'work_proxy': float(trapezoid(run.force_curve, run.strain_levels)),
            'spanning_frame': int(percolation.perc_frame),
            'final_backbone_fraction': float(percolation.backbone_frac[-1]),
            'odd_nodes': int(health['odd']),
            'connected_components': int(health['components']),
        }

    def one(self, unit):
        cfg = self.config
        factory = StructureFactory(unit=unit, grid_x=cfg.grid, grid_y=cfg.grid,
                                   n_pts_per_side=cfg.points_per_edge, seed=cfg.seed)
        graph = factory.build()
        if graph.num_nodes > cfg.max_nodes or graph.num_edges > cfg.max_edges:
            raise MemoryError('pilot graph exceeds configured node/edge budget')
        baseline = self.simulate(graph)
        recruitment = compute_percolation(baseline, alpha=cfg.alpha)
        never = ~recruitment.active_edges.any(axis=0)
        strain_score = np.maximum(baseline.edge_strain, 0).max(axis=0)
        rest = np.asarray(baseline.edge_rest, dtype=float)
        total_length = float(rest.sum())
        count = max(1, int(round(graph.num_edges * cfg.remove_fraction)))
        if count >= graph.num_edges:
            raise ValueError('removal budget must be below edge count')
        random_order = np.random.default_rng(
            cfg.seed + sum(ord(letter) for letter in unit)).permutation(graph.num_edges)
        rankings = {
            'low_recruitment': np.lexsort((np.arange(graph.num_edges), strain_score, ~never)),
            'random': random_order,
            'high_recruitment': np.lexsort((np.arange(graph.num_edges), -strain_score)),
        }
        result = {
            'baseline': self.summary(graph, baseline, 0.0, total_length),
            'never_recruited_count': int(never.sum()),
            'interventions': {},
        }
        for name, order in rankings.items():
            removed = self.choose(graph, order, count)
            candidate = graph.copy()
            for eid in removed:
                candidate.remove_edge(eid)
            for key in ('route_nodes', 'route_edges', 'topology_id'):
                candidate.metadata.pop(key, None)
            run = self.simulate(candidate)
            outcome = self.summary(candidate, run, float(rest[removed].sum()),
                                   total_length)
            outcome['removed_ids'] = removed
            outcome['never_recruited_removed'] = int(never[removed].sum())
            result['interventions'][name] = outcome
        return result

    def run(self, output):
        output = Path(output)
        signature = asdict(self.config)
        signature['units'] = list(signature['units'])
        if output.exists():
            data = json.loads(output.read_text(encoding='utf-8'))
            old_config = dict(data['config'])
            old_config.pop('units', None)
            new_config = dict(signature)
            new_config.pop('units', None)
            if (old_config != new_config or data['engine_hash'] != ENGINE_SRC_HASH
                    or not set(data['cases']).issubset(signature['units'])):
                raise ValueError('existing checkpoint uses different configuration or engine')
            data['config']['units'] = signature['units']
        else:
            data = {'config': signature, 'engine_version': ENGINE_VERSION,
                    'engine_hash': ENGINE_SRC_HASH, 'cases': {}}
        for unit in self.config.units:
            if unit in data['cases']:
                continue
            data['cases'][unit] = self.one(unit)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(output.suffix + '.tmp')
            temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
            os.replace(temporary, output)
            print('[pruning_pilot] %s complete' % unit)
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT / 'benchmarks' / 'results' /
                                                'percolation_pruning_pilot.json'))
    parser.add_argument('--units', nargs='+', default=list(PilotConfig.units))
    args = parser.parse_args()
    PruningPilot(PilotConfig(units=tuple(args.units))).run(args.output)

"""Length-stratified, resumable pruning screen with independent reruns.

Run: python benchmarks/percolation_pruning_matched.py
This is a hindsight simulation study, not an early predictor or print claim.
"""
import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.percolation_pruning_pilot import (
    PilotConfig, PruningPilot, ROOT, StructureFactory, compute_percolation,
    ENGINE_SRC_HASH, ENGINE_VERSION,
)


@dataclass
class MatchedConfig(PilotConfig):
    random_repeats: int = 5
    length_round_decimals: int = 5


class MatchedPruningPilot(PruningPilot):
    def __init__(self, config=None):
        super().__init__(config or MatchedConfig())

    @staticmethod
    def length_classes(lengths, decimals):
        rounded = np.round(np.asarray(lengths, dtype=float), decimals)
        _, classes = np.unique(rounded, return_inverse=True)
        return classes

    def choose_matched(self, graph, order, classes, quotas):
        """Select one edge at a time while matching exact length-class counts."""
        edges = np.asarray(graph.edge_array(), dtype=int)
        selected = set()
        left = dict(quotas)
        for eid in order:
            eid = int(eid)
            group = int(classes[eid])
            if left.get(group, 0) == 0:
                continue
            candidate = selected | {eid}
            if self.connected(graph.num_nodes, edges, candidate):
                selected = candidate
                left[group] -= 1
                if all(value == 0 for value in left.values()):
                    return sorted(selected)
        raise ValueError('length-stratified connected match unavailable: %s' % left)

    @staticmethod
    def cycle_screen(graph, grip_nodes, lengths, target_length):
        """Screen a cycle basis; geometry and route validity are not inferred."""
        edges = np.asarray(graph.edge_array(), dtype=int)
        simple = nx.Graph()
        simple.add_nodes_from(range(graph.num_nodes))
        simple.add_edges_from((int(a), int(b)) for a, b in edges)
        edge_lookup = {}
        for eid, (a, b) in enumerate(edges):
            edge_lookup.setdefault(frozenset((int(a), int(b))), eid)
        candidates = []
        for cycle in nx.cycle_basis(simple):
            ids = sorted({edge_lookup[frozenset((int(cycle[i]),
                                               int(cycle[(i + 1) % len(cycle)])))]
                          for i in range(len(cycle))})
            keep = np.ones(len(edges), dtype=bool)
            keep[ids] = False
            degree = np.bincount(edges[keep].ravel(), minlength=graph.num_nodes)
            active = np.flatnonzero(degree > 0)
            if active.size == 0 or np.any(degree[np.asarray(grip_nodes, dtype=int)] == 0):
                continue
            subgraph = nx.Graph()
            subgraph.add_nodes_from(int(n) for n in active)
            subgraph.add_edges_from((int(a), int(b)) for a, b in edges[keep])
            if nx.is_connected(subgraph) and not np.any(degree % 2):
                candidates.append({
                    'removed_ids': ids,
                    'removed_length_fraction': float(lengths[ids].sum() / lengths.sum()),
                    'isolated_nodes': int(graph.num_nodes - active.size),
                    'budget_error_fraction': float(abs(lengths[ids].sum() - target_length)
                                                   / lengths.sum()),
                })
        candidates.sort(key=lambda item: item['budget_error_fraction'])
        return {'basis_cycle_count': len(nx.cycle_basis(simple)),
                'topologically_feasible_count': len(candidates),
                'closest_candidates': candidates[:3]}

    def one(self, unit):
        cfg = self.config
        graph = StructureFactory(unit=unit, grid_x=cfg.grid, grid_y=cfg.grid,
                                 n_pts_per_side=cfg.points_per_edge,
                                 seed=cfg.seed).build()
        if graph.num_nodes > cfg.max_nodes or graph.num_edges > cfg.max_edges:
            raise MemoryError('matched pilot graph exceeds node/edge budget')
        baseline = self.simulate(graph)
        recruitment = compute_percolation(baseline, alpha=cfg.alpha)
        never = ~recruitment.active_edges.any(axis=0)
        score = np.maximum(baseline.edge_strain, 0).max(axis=0)
        lengths = np.asarray(baseline.edge_rest, dtype=float)
        classes = self.length_classes(lengths, cfg.length_round_decimals)
        count = max(1, int(round(graph.num_edges * cfg.remove_fraction)))
        if count >= graph.num_edges:
            raise ValueError('removal budget must be below edge count')
        low_order = np.lexsort((np.arange(graph.num_edges), score, ~never))
        low_ids = self.choose(graph, low_order, count)
        quotas = {int(group): int(np.count_nonzero(classes[low_ids] == group))
                  for group in np.unique(classes[low_ids])}
        target_length = float(lengths[low_ids].sum())
        orders = {'low_recruitment': low_ids,
                  'high_recruitment_matched': np.lexsort((np.arange(graph.num_edges),
                                                        -score))}
        for repeat in range(cfg.random_repeats):
            rng = np.random.default_rng(cfg.seed + 1009 * repeat
                                        + sum(ord(ch) for ch in unit))
            orders['random_%02d' % repeat] = rng.permutation(graph.num_edges)
        result = {
            'baseline': self.summary(graph, baseline, 0.0, float(lengths.sum())),
            'target_edge_count': len(low_ids),
            'target_removed_length_fraction': target_length / float(lengths.sum()),
            'length_class_quotas': {str(k): v for k, v in quotas.items()},
            'never_recruited_count': int(never.sum()),
            'interventions': {},
            'cycle_screen': self.cycle_screen(
                graph, np.union1d(baseline.left_nodes, baseline.right_nodes),
                lengths, target_length),
        }
        for name, order in orders.items():
            removed = self.choose_matched(graph, order, classes, quotas)
            candidate = graph.copy()
            for eid in removed:
                candidate.remove_edge(eid)
            for key in ('route_nodes', 'route_edges', 'topology_id'):
                candidate.metadata.pop(key, None)
            rerun = self.simulate(candidate)
            outcome = self.summary(candidate, rerun, float(lengths[removed].sum()),
                                   float(lengths.sum()))
            outcome['removed_ids'] = removed
            outcome['never_recruited_removed'] = int(never[removed].sum())
            outcome['budget_error_fraction'] = float(
                abs(lengths[removed].sum() - target_length) / lengths.sum())
            result['interventions'][name] = outcome
        return result

    def run(self, output):
        path = Path(output)
        signature = asdict(self.config)
        signature['units'] = list(signature['units'])
        source_hash = hashlib.sha256(b''.join(path.read_bytes() for path in (
            Path(__file__), ROOT / 'benchmarks' / 'percolation_pruning_pilot.py',
            ROOT / 'fibernet' / 'analysis' / 'tensile_recruitment.py',
        ))).hexdigest()
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
            old_config = dict(data['config'])
            old_config.pop('units', None)
            new_config = dict(signature)
            new_config.pop('units', None)
            if (old_config != new_config or data['engine_hash'] != ENGINE_SRC_HASH
                    or data['study_source_sha256'] != source_hash
                    or not set(data['cases']).issubset(signature['units'])):
                raise ValueError('checkpoint has incompatible configuration or source')
            data['config']['units'] = signature['units']
        else:
            data = {'config': signature, 'engine_version': ENGINE_VERSION,
                    'engine_hash': ENGINE_SRC_HASH,
                    'study_source_sha256': source_hash, 'cases': {}}
        for unit in signature['units']:
            if unit in data['cases']:
                continue
            case = self.one(unit)
            data['cases'][unit] = case
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
            os.replace(temporary, path)
            print('[matched_pilot] %s complete' % unit)
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT / 'benchmarks' / 'results' /
                                                'percolation_pruning_matched.json'))
    parser.add_argument('--units', nargs='+', default=list(MatchedConfig.units))
    args = parser.parse_args()
    MatchedPruningPilot(MatchedConfig(units=tuple(args.units))).run(args.output)

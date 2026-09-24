"""Screen and rerun even-degree cycle removals with compacted node IDs.

Run: python benchmarks/percolation_cycle_intervention.py
Even degree and Euler routing are necessary screens, not print validation.
"""
import argparse
import copy
import hashlib
import itertools
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'FiberScope'))

from benchmarks.percolation_pruning_matched import MatchedPruningPilot
from benchmarks.percolation_pruning_pilot import PilotConfig
from fibernet.core.structure_graph import StructureGraph
from fslab.cell_rules import graph_health
from fslab.percolation import compute_percolation
from fslab.simcache import ENGINE_SRC_HASH, ENGINE_VERSION
from fslab.structure import StructureFactory


@dataclass
class CycleConfig(PilotConfig):
    budget_tolerance_fraction: float = 0.01
    max_cycle_candidates: int = 200


class CycleInterventionStudy(MatchedPruningPilot):
    def __init__(self, config=None):
        super().__init__(config or CycleConfig())

    @staticmethod
    def compact_graph(graph, removed_indices):
        """Retain fiber attributes while dropping only newly isolated nodes."""
        removed = set(removed_indices)
        original_edges = list(graph.edges.values())
        kept = [edge for i, edge in enumerate(original_edges) if i not in removed]
        retained_nodes = sorted({int(edge.node_i) for edge in kept} |
                                {int(edge.node_j) for edge in kept})
        if not retained_nodes:
            raise ValueError('removal left no edges')
        new_graph = StructureGraph(dimension=graph.dimension,
                                   tolerance=graph.tolerance,
                                   box_size=graph.box_size)
        old_to_new = {}
        for old_id in retained_nodes:
            node = graph.nodes[old_id]
            old_to_new[old_id] = new_graph.add_node(
                node.position, boundary=tuple(node.boundary), merge=False,
                **copy.deepcopy(node.metadata))
        for edge in kept:
            new_graph.add_edge(
                old_to_new[int(edge.node_i)], old_to_new[int(edge.node_j)],
                radius=edge.radius, material=copy.deepcopy(edge.material),
                internal_points=(edge.internal_points.copy()
                                 if edge.internal_points is not None else None),
                segments=edge.segments, allow_parallel=True,
                **copy.deepcopy(edge.metadata))
        new_graph.metadata.update(copy.deepcopy(graph.metadata))
        for key in ('route_nodes', 'route_edges', 'topology_id', 'weld_triplets'):
            new_graph.metadata.pop(key, None)
        return new_graph, old_to_new

    @staticmethod
    def cycle_basis_edge_ids(edges, n_nodes):
        graph = nx.Graph()
        graph.add_nodes_from(range(n_nodes))
        graph.add_edges_from((int(a), int(b)) for a, b in edges)
        lookup = {}
        for eid, (a, b) in enumerate(edges):
            lookup.setdefault(frozenset((int(a), int(b))), eid)
        cycles = []
        for nodes in nx.cycle_basis(graph):
            ids = tuple(sorted({lookup[frozenset((int(nodes[i]),
                                                 int(nodes[(i + 1) % len(nodes)])))]
                                for i in range(len(nodes))}))
            cycles.append(ids)
        return sorted(set(cycles))

    @staticmethod
    def topologically_valid(edges, n_nodes, grip_nodes, removed):
        keep = np.ones(len(edges), dtype=bool)
        keep[list(removed)] = False
        remaining = edges[keep]
        degree = np.bincount(remaining.ravel(), minlength=n_nodes)
        if (remaining.size == 0 or np.any(degree % 2)
                or np.any(degree[np.asarray(grip_nodes, dtype=int)] == 0)):
            return False, 0
        active = np.flatnonzero(degree > 0)
        graph = nx.Graph()
        graph.add_nodes_from(int(n) for n in active)
        graph.add_edges_from((int(a), int(b)) for a, b in remaining)
        return nx.is_connected(graph), int(n_nodes - active.size)

    def candidates(self, graph, grips, lengths, edge_activity, target_length):
        edges = np.asarray(graph.edge_array(), dtype=int)
        basis = self.cycle_basis_edge_ids(edges, graph.num_nodes)
        collected = {}
        for size in (1, 2):
            for subset in itertools.combinations(basis, size):
                removed = tuple(sorted(set().union(*[set(ids) for ids in subset])))
                if removed in collected:
                    continue
                valid, isolated = self.topologically_valid(
                    edges, graph.num_nodes, grips, removed)
                if not valid:
                    continue
                mass = float(lengths[list(removed)].sum())
                score = float(np.average(edge_activity[list(removed)],
                                         weights=lengths[list(removed)]))
                collected[removed] = {
                    'removed_ids': list(removed),
                    'basis_cycles_used': size,
                    'removed_length_fraction': mass / float(lengths.sum()),
                    'budget_error_fraction': abs(mass - target_length)
                                             / float(lengths.sum()),
                    'isolated_nodes': isolated,
                    'removed_edge_recruitment': score,
                }
        all_candidates = sorted(collected.values(),
                                key=lambda x: (x['budget_error_fraction'],
                                               x['removed_ids']))
        eligible = [item for item in all_candidates
                    if item['budget_error_fraction']
                    <= self.config.budget_tolerance_fraction]
        if len(eligible) > self.config.max_cycle_candidates:
            eligible = eligible[:self.config.max_cycle_candidates]
        return len(basis), len(all_candidates), len([item for item in all_candidates
                if item['budget_error_fraction']
                <= self.config.budget_tolerance_fraction]), eligible

    def one(self, unit):
        cfg = self.config
        graph = StructureFactory(unit=unit, grid_x=cfg.grid, grid_y=cfg.grid,
                                 n_pts_per_side=cfg.points_per_edge,
                                 seed=cfg.seed).build()
        if graph.num_nodes > cfg.max_nodes or graph.num_edges > cfg.max_edges:
            raise MemoryError('cycle study graph exceeds node/edge budget')
        baseline = self.simulate(graph)
        recruitment = compute_percolation(baseline, alpha=cfg.alpha)
        never = ~recruitment.active_edges.any(axis=0)
        strain_score = np.maximum(baseline.edge_strain, 0).max(axis=0)
        count = max(1, int(round(graph.num_edges * cfg.remove_fraction)))
        low_order = np.lexsort((np.arange(graph.num_edges), strain_score, ~never))
        reference = self.choose(graph, low_order, count)
        lengths = np.asarray(baseline.edge_rest, dtype=float)
        target = float(lengths[reference].sum())
        grips = np.union1d(baseline.left_nodes, baseline.right_nodes)
        basis_count, feasible_count, in_budget_count, options = self.candidates(
            graph, grips, lengths, recruitment.active_edges.mean(axis=0), target)
        case = {
            'baseline': self.summary(graph, baseline, 0.0, float(lengths.sum())),
            'target_removed_length_fraction': target / float(lengths.sum()),
            'basis_cycle_count': basis_count,
            'topology_candidate_count': feasible_count,
            'within_budget_count': in_budget_count,
            'simulated_candidate_count': len(options),
            'selection_truncated': in_budget_count > len(options),
            'candidates': [],
        }
        for option in options:
            candidate_graph, mapping = self.compact_graph(graph, option['removed_ids'])
            health = graph_health(candidate_graph.node_positions(),
                                  candidate_graph.edge_array())
            if health['components'] != 1 or health['odd'] != 0:
                raise AssertionError('candidate failed compacted Euler screen')
            multigraph = nx.MultiGraph()
            multigraph.add_nodes_from(range(candidate_graph.num_nodes))
            multigraph.add_edges_from((int(a), int(b)) for a, b in
                                      candidate_graph.edge_array())
            if not nx.is_eulerian(multigraph):
                raise AssertionError('candidate has no Euler circuit')
            new_grips = np.array(sorted(mapping[int(old)] for old in grips),
                                 dtype=int)
            rerun = self.simulate(candidate_graph)
            if not np.array_equal(new_grips, np.union1d(
                    rerun.left_nodes, rerun.right_nodes)):
                raise AssertionError('cycle removal changed grip identity')
            summary = self.summary(candidate_graph, rerun,
                                   float(lengths[option['removed_ids']].sum()),
                                   float(lengths.sum()))
            case['candidates'].append(dict(option, **summary,
                                           euler_circuit_exists=True))
        return case

    def run(self, output):
        path = Path(output)
        config = asdict(self.config)
        config['units'] = list(config['units'])
        source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        matched_hash = hashlib.sha256((ROOT / 'benchmarks' /
                                       'percolation_pruning_matched.py').read_bytes()).hexdigest()
        pilot_hash = hashlib.sha256((ROOT / 'benchmarks' /
                                     'percolation_pruning_pilot.py').read_bytes()).hexdigest()
        analysis_hash = hashlib.sha256((ROOT / 'fibernet' / 'analysis' /
                                        'tensile_recruitment.py').read_bytes()).hexdigest()
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
            old = dict(data['config'])
            old.pop('units', None)
            new = dict(config)
            new.pop('units', None)
            if (old != new or data['engine_hash'] != ENGINE_SRC_HASH
                    or data['source_sha256'] != source_hash
                    or data['matched_source_sha256'] != matched_hash
                    or data['pilot_source_sha256'] != pilot_hash
                    or data['analysis_source_sha256'] != analysis_hash
                    or not set(data['cases']).issubset(config['units'])):
                raise ValueError('cycle checkpoint incompatible with current study')
            data['config']['units'] = config['units']
        else:
            data = {'config': config, 'engine_version': ENGINE_VERSION,
                    'engine_hash': ENGINE_SRC_HASH, 'source_sha256': source_hash,
                    'matched_source_sha256': matched_hash,
                    'pilot_source_sha256': pilot_hash,
                    'analysis_source_sha256': analysis_hash, 'cases': {}}
        for unit in config['units']:
            if unit in data['cases']:
                continue
            data['cases'][unit] = self.one(unit)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
            os.replace(temporary, path)
            print('[cycle_study] %s complete' % unit)
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT / 'benchmarks' / 'results' /
                                                'percolation_cycle_intervention.json'))
    parser.add_argument('--units', nargs='+', default=list(CycleConfig.units))
    args = parser.parse_args()
    CycleInterventionStudy(CycleConfig(units=tuple(args.units))).run(args.output)

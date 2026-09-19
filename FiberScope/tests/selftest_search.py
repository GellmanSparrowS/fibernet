"""Actual RL adapter training, checkpoint restart and per-topology exploration."""
import sys
import tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fslab.search_algorithms import run_rl, SEARCH_SPECS
from fslab.structure import StructureFactory, all_unit_keys
from fslab.inverse import run_inverse


def main():
    for algorithm in SEARCH_SPECS:
        if algorithm == 'cem':
            continue
        calls = []
        def objective(x):
            calls.append(x.copy())
            return float(np.sum((x - .2) ** 2))
        with tempfile.TemporaryDirectory() as folder:
            first = run_rl(algorithm, objective, np.zeros(3), budget=36, seed=2, checkpoint=folder)
            assert first['evaluations'] == 36 and len(calls) == 36
            second = run_rl(algorithm, objective, np.zeros(3), budget=40, seed=2, checkpoint=folder)
            assert second['evaluations'] == 40 and len(calls) == 40
            assert second['best_value'] <= first['best_value']
            assert (Path(folder) / 'policy.zip').exists()
        print('[search] %s actual policy training and resume PASS' % algorithm)
    for unit in all_unit_keys():
        factory = StructureFactory(unit=unit, grid_x=2, grid_y=1, n_pts_per_side=2,
                                   line_displacements=[[.1, .05], [-.1, .04]], expansion_rule='mirror')
        inputs = []
        def builder(key, pert, ld):
            inputs.append(ld)
            return replace(factory, unit=key, perturbation=pert, line_displacements=ld).build()
        def evaluate(graph, target):
            return float(np.var(graph.node_positions())), None
        result = run_inverse(builder, 'J', budget=4, fixed_unit=unit, pts=2,
                             initial_spec=factory, evaluator=evaluate)
        assert len(result['records']) == 4
        assert inputs[0] == factory.spectrum() and inputs[1] != inputs[0]
        assert result['best_dist'] <= result['records'][0].dist
    print('[search] all topologies deform from current spectrum within budget PASS')


if __name__ == '__main__':
    main()

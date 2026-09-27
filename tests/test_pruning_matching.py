"""Check that the intervention comparison matches physical edge length."""
import numpy as np

from benchmarks.percolation_pruning_matched import (
    MatchedConfig, MatchedPruningPilot,
)


def test_ring_interventions_match_length_and_edge_count():
    study = MatchedPruningPilot(MatchedConfig(units=('ring',), random_repeats=3))
    case = study.one('ring')
    target = case['target_removed_length_fraction']
    count = case['target_edge_count']
    for result in case['interventions'].values():
        assert len(result['removed_ids']) == count
        assert abs(result['removed_length_fraction'] - target) < 1e-7
        assert result['budget_error_fraction'] < 1e-7
        assert result['connected_components'] == 1
    assert case['cycle_screen']['topologically_feasible_count'] > 0
    assert np.median([result['odd_nodes'] for result in
                      case['interventions'].values()]) > 2

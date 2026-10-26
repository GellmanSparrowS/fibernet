"""Adversarial bin-boundary axis test: mocked figure, no fit or physics."""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.viz import plot2d
from _legacy_flow_worker import no_computation

class AxisReversal(unittest.TestCase):
    def test_all_edges_and_nextafter_neighbors(self):
        mismatches = []
        edges = np.linspace(0, np.pi, 37)
        with no_computation():
            for edge_index, edge in enumerate(edges):
                for direction in (-np.inf, None, np.inf):
                    theta = edge if direction is None else np.nextafter(edge, direction)
                    d = np.array([np.cos(theta), np.sin(theta), 0.])
                    histograms = []
                    for endpoints in ([np.zeros(3), d], [d, np.zeros(3)]):
                        network = FiberNetwork([Fiber(endpoints)], dimension=2)
                        ax = MagicMock()
                        with patch.object(plot2d.plt, 'subplots', return_value=(MagicMock(), ax)):
                            plot2d.plot_orientation_distribution(network)
                        histograms.append(ax.bar.call_args.args[1])
                    if not np.array_equal(*histograms):
                        mismatches.append((edge_index, str(direction), float(theta),
                            np.flatnonzero(histograms[0]).tolist(), np.flatnonzero(histograms[1]).tolist()))
        self.assertEqual(mismatches, [], f'{len(mismatches)} reversal mismatches: {mismatches}')

if __name__ == '__main__':
    unittest.main()

"""Fresh-process boundary for the 111-case undirected-axis regression."""
import unittest
from _legacy_flow_subprocess import run_worker


class TestAxisReversalIsolated(unittest.TestCase):
    def test_bin_edges_and_nextafter_neighbors(self):
        print(run_worker('_orientation_axis_worker.py'))

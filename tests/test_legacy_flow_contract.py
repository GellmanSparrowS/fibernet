"""Collection is stdlib-only; plotting and config cases run in a fresh process."""
import unittest
from _legacy_flow_subprocess import run_worker


class TestLegacyFlowIsolated(unittest.TestCase):
    def test_thirteen_synthetic_plot_and_preflight_cases(self):
        print(run_worker('_legacy_flow_worker.py'))

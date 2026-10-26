"""Public feature imports preserve the caller's numerical diagnostics."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module", [
    "fibernet",
    "fibernet.analysis.graph_features",
    "fibernet.analysis.graph_features_3d",
])
def test_import_preserves_warning_as_error_policy(module):
    code = f"""
import importlib
import warnings
warnings.simplefilter('error', RuntimeWarning)
warnings.simplefilter('error', FutureWarning)
importlib.import_module({module!r})
for category in (RuntimeWarning, FutureWarning):
    try:
        warnings.warn('caller diagnostic', category)
    except category:
        pass
    else:
        raise AssertionError(category.__name__ + ' was globally suppressed')
"""
    process = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True,
        text=True, timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_ml_graph_feature_export_runs_supported_extractor():
    import fibernet as fn
    from fibernet.analysis import GraphFeatureExtractor as AnalysisExtractor
    from fibernet.ml import GraphFeatureExtractor

    assert GraphFeatureExtractor is AnalysisExtractor is fn.GraphFeatureExtractor
    graph = fn.pattern_2d(unit="square", box=(2, 2), grid=(1, 1))
    values = GraphFeatureExtractor(canvas_size=32).extract(graph, allow_unavailable=True)
    assert values["n_node"] == graph.num_nodes
    assert values["n_edge"] == graph.num_edges

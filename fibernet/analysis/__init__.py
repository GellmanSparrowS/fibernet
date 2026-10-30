"""
Analysis tools for FiberNet.

- graph_features: Structural and topological feature extraction
"""

# Reload clears the facade cache without reloading the feature implementation.
globals().pop("GraphFeatureExtractor", None)
from fibernet.analysis.tensile_recruitment import (
    PercolationResult, analyze_tensile_recruitment, compute_percolation,
)
from fibernet.analysis.snapshot_features import SnapshotFeatureExtractor
from fibernet.analysis.width_contact import ContactConfig

__all__ = [
    "GraphFeatureExtractor",
    "PercolationResult",
    "analyze_tensile_recruitment",
    "compute_percolation",
    "SnapshotFeatureExtractor",
    "ContactConfig",
]

from fibernet.analysis.topology_metrics import TopologyMetricsV1, analyze_topology

__all__ += ["TopologyMetricsV1", "analyze_topology"]

from fibernet.analysis.geometry_paths import GeometryPathMetricsV1, analyze_geometry_paths

__all__ += ["GeometryPathMetricsV1", "analyze_geometry_paths"]

from fibernet.analysis.spectrum_metrics import LaplacianSpectrumV1, analyze_spectrum

__all__ += ["LaplacianSpectrumV1", "analyze_spectrum"]


def __getattr__(name):
    if name == "GraphFeatureExtractor":
        from fibernet.analysis.graph_features import GraphFeatureExtractor
        globals()[name] = GraphFeatureExtractor
        return GraphFeatureExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

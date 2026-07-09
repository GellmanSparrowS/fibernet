"""
Analysis tools for fiber networks.
分析工具模块，用于纤维网络结构分析。

Submodules / 子模块:
- topology: Graph-theoretic analysis / 图论分析
- morphology: Geometric characterization / 几何表征
- properties: Effective property estimation / 有效性能估算
- advanced: Spectral analysis, pore distribution, anisotropy, fingerprinting
- stress_strain: Stress-strain curve extraction and analysis
"""

from fibernet.analysis.morphology import MorphologyAnalyzer
from fibernet.analysis.properties import PropertyEstimator

# Topology requires networkx; guard import for CI minimal installs
try:
    from fibernet.analysis.topology import TopologyAnalyzer
except (ImportError, NameError):
    TopologyAnalyzer = None

try:
    from fibernet.analysis.advanced import (
        SpectralAnalyzer, PoreAnalyzer, AnisotropyAnalyzer, StructuralFingerprint,
    )
except ImportError:
    pass

try:
    from fibernet.analysis.stress_strain import (
        StressStrainCurve, extract_stress_strain, compare_curves,
    )
except ImportError:
    pass

__all__ = [
    "MorphologyAnalyzer", "PropertyEstimator",
]

if TopologyAnalyzer is not None:
    __all__.append("TopologyAnalyzer")

# Statistical analysis
from fibernet.analysis.statistics import StatisticalAnalyzer

# NetworkX integration
try:
    from fibernet.analysis.networkx_integration import (
        to_networkx, compute_centrality, detect_communities,
        compute_graph_metrics, find_shortest_path, compute_small_world_metrics,
    )
except ImportError:
    pass

__all__ += [
    "StatisticalAnalyzer",
]

# Percolation analysis
from fibernet.analysis.percolation import (
    PercolationAnalyzer, PercolationResult, estimate_percolation_threshold,
)
__all__ += [
    "PercolationAnalyzer", "PercolationResult", "estimate_percolation_threshold",
]

# Graph feature extraction (94-dim)
from fibernet.analysis.graph_features import GraphFeatureExtractor
__all__ += [
    "GraphFeatureExtractor",
]

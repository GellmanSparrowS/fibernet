"""
Analysis tools for fiber networks.

Submodules:
- topology: Graph-theoretic analysis
- morphology: Geometric characterization
- properties: Effective property estimation
- advanced: Spectral analysis, pore distribution, anisotropy, fingerprinting
- stress_strain: Stress-strain curve extraction and analysis
"""

from fibernet.analysis.topology import TopologyAnalyzer
from fibernet.analysis.morphology import MorphologyAnalyzer
from fibernet.analysis.properties import PropertyEstimator
from fibernet.analysis.advanced import (
    SpectralAnalyzer, PoreAnalyzer, AnisotropyAnalyzer, StructuralFingerprint,
)
from fibernet.analysis.stress_strain import (
    StressStrainCurve, extract_stress_strain, compare_curves,
)

__all__ = [
    "TopologyAnalyzer", "MorphologyAnalyzer", "PropertyEstimator",
    "SpectralAnalyzer", "PoreAnalyzer", "AnisotropyAnalyzer", "StructuralFingerprint",
    "StressStrainCurve", "extract_stress_strain", "compare_curves",
]

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

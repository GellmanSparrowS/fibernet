"""
FiberNet Integrations Module

Provides integration with popular open-source scientific libraries:
- NetworkX: Advanced graph analysis and community detection
- MDAnalysis: Molecular dynamics trajectory analysis
- LAMMPS: Molecular dynamics simulations
- OVITO: Advanced visualization and analysis

All integrated libraries are optional dependencies.
"""

# NetworkX integration (BSD license)
try:
    from .networkx_integration import (
        NetworkXBridge, GraphAnalysisResult,
        analyze_network_topology
    )
    _NETWORKX_AVAILABLE = True
except ImportError:
    _NETWORKX_AVAILABLE = False

# MDAnalysis integration (GPL v2 license)
try:
    from .mdanalysis_integration import (
        MDAnalysisBridge, MDAnalysisResult,
        analyze_fiber_dynamics
    )
    _MDANALYSIS_AVAILABLE = True
except ImportError:
    _MDANALYSIS_AVAILABLE = False

# LAMMPS integration (GPL v2 license)
try:
    from .lammps_integration import (
        LAMMPSBridge, LAMMPSResult,
        run_lammps_md
    )
    _LAMMPS_AVAILABLE = True
except ImportError:
    _LAMMPS_AVAILABLE = False

# OVITO integration (GPL v3 license)
try:
    from .ovito_integration import (
        OVITOBridge, OVITOAnalysisResult,
        render_network_ovito
    )
    _OVITO_AVAILABLE = True
except ImportError:
    _OVITO_AVAILABLE = False

__all__ = []

if _NETWORKX_AVAILABLE:
    __all__.extend([
        "NetworkXBridge", "GraphAnalysisResult",
        "analyze_network_topology"
    ])

if _MDANALYSIS_AVAILABLE:
    __all__.extend([
        "MDAnalysisBridge", "MDAnalysisResult",
        "analyze_fiber_dynamics"
    ])

if _LAMMPS_AVAILABLE:
    __all__.extend([
        "LAMMPSBridge", "LAMMPSResult",
        "run_lammps_md"
    ])

if _OVITO_AVAILABLE:
    __all__.extend([
        "OVITOBridge", "OVITOAnalysisResult",
        "render_network_ovito"
    ])

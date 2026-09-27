"""APP compatibility layer for FiberNet tensile recruitment analysis."""

from .fibernet_bridge import ensure_fibernet

ensure_fibernet()

from fibernet.analysis.tensile_recruitment import (
    PercolationResult,
    analyze_tensile_recruitment,
    compute_percolation,
)

__all__ = ['PercolationResult', 'analyze_tensile_recruitment',
           'compute_percolation']

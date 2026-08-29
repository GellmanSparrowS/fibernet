"""fslab: pure-numpy science core for FiberScope (no Qt imports here)."""
from .structure import StructureFactory, UNIT_PRESETS
from .simcache import RunConfig, StretchRun, run_stretch
from .percolation import PercolationResult, compute_percolation

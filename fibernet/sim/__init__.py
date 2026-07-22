"""
Simulation engines for FiberNet.

Core backend:
- accelerated: TaichiEngine (mass-spring dynamics, GPU-accelerated)
- rl_env: FiberNetworkEnv (RL environment)
"""

try:
    from fibernet.sim.accelerated import TaichiEngine, SimResult
except ImportError:
    pass

__all__ = [
    "TaichiEngine", "SimResult",
]

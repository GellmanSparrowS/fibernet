"""
Simulation engines for FiberNet.

Core backend:
- accelerated: TaichiEngine (mass-spring dynamics, GPU-accelerated)
- rl_env: FiberNetworkEnv (RL environment)
"""

# Keep importing a simulation contract independent of unrelated native engines.
_LAZY_EXPORTS = {
    "prepare_nodal_controls": "fibernet.sim.nodal_controls",
    "PreparedNodalControls": "fibernet.sim.nodal_controls",
    "TaichiEngine": "fibernet.sim.accelerated",
    "SimResult": "fibernet.sim.accelerated",
    "ReducedBeamConfig": "fibernet.sim.reduced_beam",
    "ReducedBeamResult": "fibernet.sim.reduced_beam",
    "ReducedBeamSolver": "fibernet.sim.reduced_beam",
}

__all__ = list(_LAZY_EXPORTS)

# A package reload must not retain classes from a reloaded owner module.
for _name in _LAZY_EXPORTS:
    globals().pop(_name, None)
del _name


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        from importlib import import_module
        value = getattr(import_module(_LAZY_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

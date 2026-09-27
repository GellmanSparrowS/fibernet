"""FiberScope compatibility entry points for the shared curve inverse search."""
from fibernet.ml.curve_inverse import (
    LD_AMP, PTS, TARGETS, SCALARS, InverseRecord,
    target_curve, loading_phase, curve_of, metrics_of, distance,
    objective_of, decode_line_params, save_log,
    run_inverse as _shared_run_inverse,
)
from .engine2 import Engine2  # Legacy patch point used by APP integrations.
from .simcache import RunConfig
from .structure import all_unit_keys


def fast_cfg(stretch: float) -> RunConfig:
    """Keep the APP worker's RunConfig interface for older callers."""
    return RunConfig(target_stretch=stretch, num_steps=6000,
                     n_increments=60, save_interval=250, ramp_fraction=0.7)


def run_inverse(*args, **kwargs):
    """Run the shared search with the APP's registered unit keys."""
    kwargs.setdefault("unit_keys", all_unit_keys())
    return _shared_run_inverse(*args, **kwargs)

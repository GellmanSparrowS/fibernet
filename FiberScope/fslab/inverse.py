"""Inverse design (two-stage, callback-driven for live GUI):

  stage 1  topology screen: every unit type with a pristine lattice
  stage 2  CEM refinement on the winning unit over the intermediate-point
           values of ONE reference fiber line (2*pts dims) + perturbation,
           i.e. the optimizer tunes the same point values the user edits.

Targets: normalized curve shapes (J/C/linear/multi) OR scalar objectives
(peak force / initial stiffness / toughness, max or min). The force curve
is measured on the LOADING phase only (monotonic strain), so damped
hold-phase relaxation does not pollute the shape metric.
"""
from dataclasses import dataclass
import json

import numpy as np

from .engine2 import Engine2
from .simcache import RunConfig
from .structure import UNIT_PRESETS

LD_AMP = 0.25          # max line-point displacement fraction in CEM
PTS = 2                # intermediate points used by the optimizer

TARGETS = {
    "J": lambda t: t ** 2.2,                    # delayed stiffening
    "C": lambda t: 1 - (1 - t) ** 2.2,          # early stiffening
    "linear": lambda t: t,
    "multi": lambda t: np.clip(0.5 * t + 0.5 * np.maximum(t - 0.55, 0) / 0.45,
                               0, 1),
}

# name -> (metric key, sign): sign=-1 means "maximize"
SCALARS = {
    "max_peak": ("peak", -1.0),
    "min_peak": ("peak", 1.0),
    "max_stiffness": ("stiff", -1.0),
    "min_stiffness": ("stiff", 1.0),
    "max_toughness": ("tough", -1.0),
}


def target_curve(name: str, n: int = 24) -> np.ndarray:
    t = np.linspace(0, 1, n)
    y = TARGETS[name](t)
    return (y / max(y.max(), 1e-12)).astype(np.float32)


def loading_phase(run):
    s = run.strain_levels
    F = run.force_curve
    keep = [0]
    for i in range(1, len(s)):
        if s[i] > s[keep[-1]] + 1e-6:
            keep.append(i)
    return s[keep], F[keep]


def curve_of(run, n: int = 24) -> np.ndarray:
    """Normalized force vs strain on the loading (monotonic) phase only."""
    s, F = loading_phase(run)
    s0, s1 = s[0], max(s[-1], s[0] + 1e-9)
    t = np.clip((s - s0) / (s1 - s0), 0, 1)
    y = np.interp(np.linspace(0, 1, n), t, F)
    return (y / max(y.max(), 1e-12)).astype(np.float32)


def metrics_of(run) -> dict:
    s, F = loading_phase(run)
    if len(s) < 2:
        return dict(peak=0.0, stiff=0.0, tough=0.0)
    s0, s1 = s[0], max(s[-1], s[0] + 1e-9)
    t = (s - s0) / (s1 - s0)
    return dict(peak=float(F.max()),
                stiff=float(np.interp(0.2, t, F) - F[0]),
                tough=float(np.trapz(F, s)))


def distance(run, target: np.ndarray) -> float:
    return float(np.mean((curve_of(run, len(target)) - target) ** 2) ** 0.5)


def objective_of(run, target_name: str, target: np.ndarray):
    """(objective_to_minimize, metrics_dict)"""
    m = metrics_of(run)
    if target_name in SCALARS:
        key, sign = SCALARS[target_name]
        return sign * m[key], m
    return distance(run, target), m


def fast_cfg(stretch: float) -> RunConfig:
    return RunConfig(target_stretch=stretch, num_steps=6000,
                     n_increments=60, save_interval=250, ramp_fraction=0.7)


@dataclass
class InverseRecord:
    eval_id: int
    stage: str
    label: str
    params: list
    dist: float
    best_dist: float


def decode_line_params(x, pts: int = PTS):
    """CEM vector -> (line_displacements, perturbation)."""
    ld = [[float(x[2 * k]) * LD_AMP, float(x[2 * k + 1]) * LD_AMP]
          for k in range(pts)]
    pert = float(np.clip(x[2 * pts], 0, 1) * 0.5)
    return ld, pert


def run_inverse(factory_builder, target_name: str, budget: int = 60,
                seed: int = 0, stretch: float = 2.0, callback=None,
                fixed_unit: str = None):
    """factory_builder(unit, pert, line_displacements) -> graph.
    callback(rec, run_if_best)."""
    rng = np.random.default_rng(seed)
    target = target_curve(target_name) if target_name in TARGETS else None
    best_dist = np.inf
    best_run = None
    best_label = ""
    best_spec = None
    records = []
    ev = 0

    def eval_one(stage, label, params, unit, pert, ld):
        nonlocal ev, best_dist, best_run, best_label, best_spec
        g = factory_builder(unit, pert, ld)
        run = Engine2(g, fast_cfg(stretch).engine_cfg()).run()
        d, _ = objective_of(run, target_name, target)
        ev += 1
        is_best = d < best_dist
        if is_best:
            best_dist, best_run, best_label = d, run, label
            best_spec = dict(unit=unit, pert=pert, line_displacements=ld)
        rec = InverseRecord(ev, stage, label, [float(x) for x in params],
                            d, best_dist)
        records.append(rec)
        if callback is not None:
            callback(rec, run if is_best else None)
        return d

    # stage 1: topology screen (locked to the caller's current unit:
    # inverse design reshapes the same primitive, never swaps topology)
    screen_units = [fixed_unit] if fixed_unit else list(UNIT_PRESETS)
    dists = []
    for unit in screen_units:
        d = eval_one("screen", unit, [], unit, 0.0, None)
        dists.append((d, unit))
    dists.sort()
    win_unit = dists[0][1]

    # stage 2: CEM over the reference-line point values (+ perturbation)
    dim = 2 * PTS + 1
    mean = np.zeros(dim)
    std = np.array([0.5] * (2 * PTS) + [0.3])
    pop, elite = 10, 3
    X = None
    while ev < budget:
        if X is None:
            X = rng.uniform(-1, 1, (pop, dim))
        else:
            X = np.clip(mean + std[None, :] * rng.standard_normal((pop, dim)),
                        -1, 1)
        X = X[: budget - ev]
        ds = []
        for x in X:
            ld, pert = decode_line_params(x)
            d = eval_one("refine", win_unit, x, win_unit, pert, ld)
            ds.append(d)
        order = np.argsort(ds)
        elites = X[order[:elite]]
        mean = elites.mean(0)
        std = np.clip(elites.std(0) + 1e-3, 0.05, 1.0)

    return {"best_dist": best_dist, "best_label": best_label,
            "best_spec": best_spec, "records": records,
            "best_run": best_run}


def save_log(path, result):
    with open(path, "w", encoding="utf-8") as fh:
        for r in result["records"]:
            fh.write(json.dumps({"eval_id": r.eval_id, "stage": r.stage,
                                 "label": r.label, "params": r.params,
                                 "dist": round(r.dist, 5),
                                 "best_dist": round(r.best_dist, 5)}) + "\n")

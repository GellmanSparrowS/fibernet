"""Target-curve and scalar inverse search on the shared reduced solver.

The caller supplies a graph builder and, for a topology screen, unit keys.
Results are numerical model objectives and require independent validation.

Inverse design (two-stage, callback-driven for live GUI):

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
from scipy.integrate import trapezoid

from fibernet.sim.reduced_beam import ReducedBeamConfig, ReducedBeamSolver

LD_AMP = 0.45          # max line-point displacement fraction in CEM
PTS = 2                # intermediate points used by the optimizer

TARGETS = {
    "J": lambda t: t ** 2.2,                    # delayed stiffening
    "C": lambda t: 1 - (1 - t) ** 2.2,          # early stiffening
    "linear": lambda t: t,
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
                tough=float(trapezoid(F, s)))


def distance(run, target: np.ndarray) -> float:
    return float(np.mean((curve_of(run, len(target)) - target) ** 2) ** 0.5)


def objective_of(run, target_name: str, target: np.ndarray):
    """(objective_to_minimize, metrics_dict)"""
    m = metrics_of(run)
    if target_name in SCALARS:
        key, sign = SCALARS[target_name]
        return sign * m[key], m
    return distance(run, target), m


def fast_cfg(stretch: float) -> ReducedBeamConfig:
    return ReducedBeamConfig(target_stretch=stretch, num_steps=6000,
                             n_increments=60, save_interval=250,
                             ramp_fraction=0.7, drag=200.0)


@dataclass
class InverseRecord:
    eval_id: int
    stage: str
    label: str
    params: list
    dist: float
    best_dist: float


def decode_line_params(x, pts: int = PTS, amplitude=LD_AMP):
    """CEM vector -> (line_displacements, perturbation)."""
    ld = [[float(x[2 * k]) * amplitude, float(x[2 * k + 1]) * amplitude]
          for k in range(pts)]
    pert = float(np.clip(x[2 * pts], 0, 1) * 0.5)
    return ld, pert


def run_inverse(factory_builder, target_name: str, budget: int = 60,
                seed: int = 0, stretch: float = 2.0, callback=None,
                fixed_unit: str = None, pts: int = None, stop_cb=None,
                initial_spec=None, evaluator=None, amplitude=.6,
                unit_keys=None):
    """factory_builder(unit, pert, line_displacements) -> graph.
    callback(rec, current_run); rec.dist == rec.best_dist identifies a best candidate.
    stop_cb() -> True aborts between evaluations (same contract as
    mlmodel.gen_dataset); the best result found so far is still returned
    with result["stopped"] = True."""
    budget = max(1, int(budget))
    if not .05 <= amplitude <= 1.:
        raise ValueError('deformation amplitude must be 0.05..1.0')
    if target_name not in TARGETS and target_name not in SCALARS:
        raise ValueError('unknown inverse objective')
    pts = int(pts) if pts else PTS
    rng = np.random.default_rng(seed)
    target = target_curve(target_name) if target_name in TARGETS else None
    best_dist = np.inf
    best_run = None
    best_label = ""
    best_spec = None
    records = []
    ev = 0
    stopped = False

    def _stop():
        return bool(stop_cb()) if stop_cb is not None else False

    def eval_one(stage, label, params, unit, pert, ld):
        nonlocal ev, best_dist, best_run, best_label, best_spec
        g = factory_builder(unit, pert, ld)
        if evaluator is None:
            run = ReducedBeamSolver(g, fast_cfg(stretch)).run()
            d, _ = objective_of(run, target_name, target)
        else:
            d, run = evaluator(g, target_name)
        if not np.isfinite(d):
            raise ValueError('non-finite inverse objective')
        ev += 1
        is_best = d < best_dist
        if is_best:
            best_dist, best_run, best_label = d, run, label
            best_spec = dict(unit=unit, pert=pert, line_displacements=ld)
        rec = InverseRecord(ev, stage, label, [float(x) for x in params],
                            d, best_dist)
        records.append(rec)
        del records[:-2000]
        if callback is not None:
            callback(rec, run)
        return d

    # stage 1: topology screen (locked to the caller's current unit:
    # inverse design reshapes the same primitive, never swaps topology)
    if fixed_unit is None and not unit_keys:
        raise ValueError('unit_keys are required when fixed_unit is omitted')
    screen_units = [fixed_unit] if fixed_unit else list(unit_keys)
    dists = []
    for unit in screen_units:
        if ev >= budget or _stop():
            stopped = _stop()
            break
        ld0 = initial_spec.spectrum() if initial_spec is not None else None
        pert0 = initial_spec.perturbation if initial_spec is not None else 0.
        d = eval_one('screen', unit, [], unit, pert0, ld0)
        dists.append((d, unit))
        if _stop():
            stopped = True
            break
    dists.sort()
    if not dists:
        return dict(best_dist=None, best_label='', best_spec=None, records=[], best_run=None, stopped=True)
    win_unit = dists[0][1]

    # stage 2: CEM over the reference-line point values (+ perturbation).
    # The exploration radius starts wide (explore broadly) and shrinks as
    # the budget is spent, so later evaluations polish rather than wander.
    dim = 2 * pts + 1
    mean = np.zeros(dim)
    if initial_spec is not None:
        mean[:-1] = np.asarray(initial_spec.spectrum() or [[0., 0.]] * pts).ravel() / amplitude
        mean[-1] = initial_spec.perturbation / .5
        mean = np.clip(mean, -1, 1)
    std = np.array([0.85] * (2 * pts) + [0.5])
    pop = max(3, min(10, int(budget - ev) if budget - ev > 0 else 3))
    elite = max(2, min(3, pop // 3))
    X = None
    while ev < budget and not stopped:
        remaining = max(budget - ev, 1)
        pop_n = min(pop, remaining)
        if X is None:
            X = rng.uniform(-1, 1, (pop_n, dim))
            local = max(1, pop_n // 5)
            X[:local] = np.clip(mean + .85 * rng.standard_normal((local, dim)), -1, 1)
        else:
            X = np.clip(mean + std[None, :] * rng.standard_normal((pop_n, dim)),
                        -1, 1)
            global_count = max(1, pop_n // 4)
            X[-global_count:] = rng.uniform(-1, 1, (global_count, dim))
        X = X[:remaining]
        ds = []
        for x in X:
            if _stop():
                stopped = True
                break
            ld, pert = decode_line_params(x, pts, amplitude)
            d = eval_one("refine", win_unit, x, win_unit, pert, ld)
            ds.append(d)
        if not ds:
            break
        order = np.argsort(ds)
        elites = X[order[:max(1, min(elite, len(order)))]]
        mean = elites.mean(0)
        shrink = max(0.35, 1.0 - (ev / max(budget, 1)))
        std = np.clip((elites.std(0) + 1e-3) * shrink, 0.06, 0.9)

    return {"best_dist": best_dist, "best_label": best_label,
            "best_spec": best_spec, "records": records,
            "best_run": best_run, "stopped": stopped, "evaluations": ev}


def save_log(path, result):
    with open(path, "w", encoding="utf-8") as fh:
        for r in result["records"]:
            fh.write(json.dumps({"eval_id": r.eval_id, "stage": r.stage,
                                 "label": r.label, "params": r.params,
                                 "dist": round(r.dist, 5),
                                 "best_dist": round(r.best_dist, 5)}) + "\n")


class CurveInverseDesigner:
    """Configure a callback-driven target-curve search over a graph builder."""

    def __init__(self, factory_builder, unit_keys=None, budget=60, seed=0,
                 stretch=2.0, amplitude=0.6):
        if not callable(factory_builder):
            raise TypeError("factory_builder must be callable")
        self.factory_builder = factory_builder
        self.unit_keys = tuple(unit_keys) if unit_keys is not None else None
        self.budget = int(budget)
        self.seed = int(seed)
        self.stretch = float(stretch)
        self.amplitude = float(amplitude)

    def run(self, target_name, fixed_unit=None, initial_spec=None,
            callback=None, stop_cb=None, evaluator=None, pts=None):
        return run_inverse(
            self.factory_builder, target_name, budget=self.budget,
            seed=self.seed, stretch=self.stretch, callback=callback,
            fixed_unit=fixed_unit, pts=pts, stop_cb=stop_cb,
            initial_spec=initial_spec, evaluator=evaluator,
            amplitude=self.amplitude, unit_keys=self.unit_keys)


run_curve_inverse = run_inverse

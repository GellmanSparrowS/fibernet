"""Offline exploration run: novelty-search agent vs R0 uniform random,
same eval budget, checkpointed JSONL log (resumable).

Usage: python scripts/run_exploration.py [--budget 120] [--seed 0]
Output: data/exploration_log.jsonl  (one record per eval, append-only)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fslab import StructureFactory
from fslab.engine2 import Engine2
from fslab.inverse import curve_of
from fslab.simcache import RunConfig
from fslab.structure import UNIT_PRESETS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "data", "exploration_log.jsonl")

UNITS = list(UNIT_PRESETS)


def behavior_of(run):
    c = curve_of(run, 12)
    e = run.energies
    tot = e["axial"][-1] + e["bend"][-1] + e["contact"][-1] + 1e-12
    return np.concatenate([c, [e["bend"][-1] / tot, e["contact"][-1] / tot,
                               run.force_curve[-1] / 1e6]])


def evaluate(unit, pert, seed):
    g = StructureFactory(unit=unit, grid_x=3, grid_y=3, n_pts_per_side=2,
                         seed=seed, perturbation=pert).build()
    run = Engine2(g, RunConfig(target_stretch=2.0, num_steps=8000,
                               save_interval=1000).engine_cfg()).run()
    return run, behavior_of(run)


def load_done():
    done = []
    if os.path.exists(LOG):
        with open(LOG, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.append(json.loads(line))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    done = load_done()
    n_agent = sum(1 for r in done if r["method"] == "agent")
    n_r0 = sum(1 for r in done if r["method"] == "r0")
    rng = np.random.default_rng(args.seed + len(done))

    archive = [np.array(r["behavior"]) for r in done if r["method"] == "agent"
               and r["accepted"]]

    def novelty(b):
        if not archive:
            return 10.0
        return float(min(np.linalg.norm(b - a) for a in archive))

    with open(LOG, "a", encoding="utf-8") as fh:
        while n_agent < args.budget or n_r0 < args.budget:
            if n_agent < args.budget:
                # agent: mutate a random archived member, else random
                if archive and rng.random() < 0.8:
                    base = None
                    for r in done:
                        if r["method"] == "agent" and r["accepted"] and \
                                np.allclose(r["behavior"], archive[int(
                                    rng.integers(len(archive)))]):
                            base = r
                            break
                    unit = base["unit"] if base else str(rng.choice(UNITS))
                    pert = float(np.clip((base["pert"] if base else 0.2) +
                                         rng.normal(0, 0.15), 0, 0.6))
                    seed = int(rng.integers(1000))
                else:
                    unit = str(rng.choice(UNITS))
                    pert = float(rng.uniform(0, 0.6))
                    seed = int(rng.integers(1000))
                method = "agent"
            else:
                method = "r0"
                unit = str(rng.choice(UNITS))
                pert = float(rng.uniform(0, 0.6))
                seed = int(rng.integers(1000))

            run, b = evaluate(unit, pert, seed)
            nov = novelty(b)
            accepted = (method == "r0") or (nov > 0.35)
            if method == "agent" and accepted:
                archive.append(b)
            rec = {"id": len(done), "method": method, "unit": unit,
                   "pert": round(pert, 3), "seed": seed,
                   "novelty": round(nov, 4), "accepted": bool(accepted),
                   "behavior": [round(float(x), 4) for x in b],
                   "bend_frac": round(float(run.energies["bend"][-1] /
                                            max(run.energies["axial"][-1] +
                                                run.energies["bend"][-1] +
                                                run.energies["contact"][-1],
                                                1e-9)), 3),
                   "contact": int(run.contact_counts.max()),
                   "force": round(float(run.force_curve[-1]), 1)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            done.append(rec)
            if method == "agent":
                n_agent += 1
            else:
                n_r0 += 1
            print(f"[explore] {rec['id']:3d} {method:5s} {unit:11s} "
                  f"nov={nov:.3f} acc={int(accepted)}", flush=True)
    print(f"[explore] done: agent={n_agent} r0={n_r0}")


if __name__ == "__main__":
    main()

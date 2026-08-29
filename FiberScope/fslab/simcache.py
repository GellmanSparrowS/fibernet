"""Stretch simulation runner on engine2 with per-frame extraction + disk cache.

One simulation produces every frame the GUI needs; the slider never re-runs
the solver. Cache files are written atomically (tmp + os.replace).
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import os

import numpy as np

from .engine2 import Engine2, Engine2Config
from .fibernet_bridge import ensure_fibernet

MAX_FRAMES = 240     # downsample cap for trajectory frames


@dataclass
class RunConfig:
    target_stretch: float = 2.0
    stiffness: float = 1.0e5
    k_bend_frac: float = 0.25
    k_contact: float = 2.0e5
    r_contact: float = 0.8
    use_bending: bool = True
    use_contact: bool = True
    drag: float = 200.0
    # explicit steps: fibernet auto_steps was non-deterministic; engine2 is
    # deterministic, 16000 steps x 1e-5 s settles these samples (~1s wall)
    num_steps: int = 16000
    save_interval: int = 1000
    ramp_fraction: float = 0.2
    n_increments: int = 120  # quasi-static load increments (frames-1)
    auto_scale: bool = True  # scale step budget with network size

    def engine_cfg(self, n_nodes=None) -> Engine2Config:
        steps = self.num_steps
        if self.auto_scale and n_nodes:
            fac = min(max((n_nodes / 600.0) ** 0.5, 1.0), 2.0)
            steps = int(steps * fac)
        return Engine2Config(
            target_stretch=self.target_stretch, stiffness=self.stiffness,
            k_bend_frac=self.k_bend_frac, k_contact=self.k_contact,
            r_contact=self.r_contact, use_bending=self.use_bending,
            use_contact=self.use_contact, drag=self.drag,
            num_steps=steps, save_interval=self.save_interval,
            ramp_fraction=self.ramp_fraction,
            n_increments=self.n_increments)


@dataclass
class StretchRun:
    frames_xy: np.ndarray       # (F, N, 2) float32
    edges: np.ndarray           # (E, 2) int32
    edge_rest: np.ndarray       # (E,) float32
    edge_strain: np.ndarray     # (F, E) float32
    strain_levels: np.ndarray   # (F,) macro stretch ratio
    force_curve: np.ndarray     # (F,) grip reaction force
    left_nodes: np.ndarray
    right_nodes: np.ndarray
    energies: dict              # axial/bend/contact/kinetic/work (F,)
    contact_pairs_last: np.ndarray
    contact_frames: list
    contact_counts: np.ndarray
    metadata: dict

    @property
    def n_frames(self):
        return int(self.frames_xy.shape[0])

    @property
    def n_edges(self):
        return int(self.edges.shape[0])


def _downsample(res):
    F = res.frames_xy.shape[0]
    if F <= MAX_FRAMES:
        return res
    idx = np.unique(np.linspace(0, F - 1, MAX_FRAMES).round().astype(int))
    res.frames_xy = res.frames_xy[idx]
    res.edge_strain = res.edge_strain[idx]
    res.strain_levels = res.strain_levels[idx]
    res.force_curve = res.force_curve[idx]
    for k in list(res.energies):
        res.energies[k] = res.energies[k][idx]
    res.contact_counts = res.contact_counts[idx]
    return res


def run_stretch(factory, cfg: RunConfig = None, cache_dir: str = None,
                use_cache: bool = True, progress_cb=None) -> StretchRun:
    ensure_fibernet()
    cfg = cfg or RunConfig()
    graph = factory.build()
    n_nodes = len(graph.node_positions())

    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        raw = json.dumps({"s": factory.key(), "c": asdict(cfg)}, sort_keys=True)
        tag = hashlib.sha1(raw.encode()).hexdigest()[:16]
        path = os.path.join(cache_dir, f"run_{tag}.npz")
        if use_cache and os.path.exists(path):
            with np.load(path, allow_pickle=True) as z:
                energies = {str(k): z["energies_" + k] for k in
                            ("axial", "bend", "contact", "kinetic", "work")}
                cf = list(z["contact_frames"]) if "contact_frames" in z \
                    else [np.zeros((0, 2), dtype=np.int64)] * len(z["frames_xy"])
                return StretchRun(
                    z["frames_xy"], z["edges"], z["edge_rest"], z["edge_strain"],
                    z["strain_levels"], z["force_curve"], z["left_nodes"],
                    z["right_nodes"], energies, z["contact_pairs_last"], cf,
                    z["contact_counts"], json.loads(str(z["meta"])))

    res = Engine2(graph, cfg.engine_cfg(n_nodes)).run(progress_cb=progress_cb)
    # no fracture in this model -> grip force must not soften; clip the
    # tiny numerical transients so the curve is monotone non-decreasing
    res.force_curve = np.maximum.accumulate(res.force_curve)
    res = _downsample(res)
    run = StretchRun(res.frames_xy, res.edges, res.edge_rest, res.edge_strain,
                     res.strain_levels, res.force_curve, res.left_nodes,
                     res.right_nodes, res.energies, res.contact_pairs_last,
                     res.contact_frames, res.contact_counts, res.metadata)

    if path:
        tmp = path[:-4] + ".tmp.npz"
        kw = dict(frames_xy=run.frames_xy, edges=run.edges,
                  edge_rest=run.edge_rest, edge_strain=run.edge_strain,
                  strain_levels=run.strain_levels, force_curve=run.force_curve,
                  left_nodes=run.left_nodes, right_nodes=run.right_nodes,
                  contact_pairs_last=run.contact_pairs_last,
                  contact_frames=np.array(run.contact_frames, dtype=object),
                  contact_counts=run.contact_counts,
                  meta=json.dumps(run.metadata))
        for k, v in run.energies.items():
            kw["energies_" + k] = v
        np.savez_compressed(tmp, **kw)
        os.replace(tmp, path)
    return run

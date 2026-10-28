"""Stretch simulation runner on engine2 with per-frame extraction + disk cache.

One simulation produces every frame the GUI needs; the slider never re-runs
the solver. Cache files are written atomically (tmp + os.replace).
"""
from dataclasses import dataclass, asdict
import hashlib
import json
from numbers import Integral
import os
import shutil
import tempfile

import numpy as np

from .engine2 import Engine2, Engine2Config
from .fibernet_bridge import FIBERNET_ROOT, ensure_fibernet
from fibernet.sim.reduced_beam import _validate_planar_geometry

MAX_FRAMES = 240     # downsample cap for trajectory frames

# --- cache identity ------------------------------------------------------
# A cached npz is only valid for the numerics that produced it, so the key
# carries an explicit engine version.  After changing the solver, the
# downsampling, structure builder or shared graph run scripts/bump_engine_version.py;
# tests/selftest.py fails loudly when the recorded source hash drifts.
ENGINE_VERSION = "e2v14"
ENGINE_SRC_HASH = "9e41b1fb4d5d"
ENGINE_SRC_FILES = ("engine2.py", "simcache.py", "structure.py", "cell_rules.py", "manufacturing.py", "cell_cycles.py", "welding.py")


def engine_src_hash():
    """sha1 of the numeric core sources; None when they are not readable.

    The two identity constants below are masked out, otherwise writing a
    new hash would immediately invalidate itself.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha1()
    try:
        for name in ENGINE_SRC_FILES:
            with open(os.path.join(base, name), "rb") as fh:
                data = fh.read().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            if name == "simcache.py":
                data = b"\n".join(
                    ln for ln in data.split(b"\n")
                    if not ln.startswith((b"ENGINE_VERSION =",
                                          b"ENGINE_SRC_HASH =")))
            h.update(data)
        for shared in (("core", "structure_graph.py"),
                       ("gen", "spectrum.py"),
                       ("gen", "cell_rules.py"),
                       ("gen", "cell_cycles.py"),
                       ("gen", "manufacturing.py"),
                       ("gen", "surface_geometry.py"),
                       ("sim", "reduced_beam.py")):
            with open(os.path.join(FIBERNET_ROOT, "fibernet", *shared), "rb") as fh:
                h.update(fh.read().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    except OSError:
        return None
    return h.hexdigest()[:12]


_CACHE_DIR = None


def atomic_replace(tmp, dst):
    """os.replace with a shutil.move fallback.

    Some Windows filter drivers reject MoveFileEx(REPLACE_EXISTING) with
    WinError 17 even inside one directory; shutil.move copes.
    """
    try:
        os.replace(tmp, dst)
    except OSError:
        shutil.move(tmp, dst)


def _probe_cache_dir(cand):
    """True when cand supports the real write+atomic-rename cycle."""
    try:
        os.makedirs(cand, exist_ok=True)
        tmp = os.path.join(cand, ".probe.tmp.npz")
        dst = os.path.join(cand, ".probe.npz")
        np.savez_compressed(tmp, ok=np.zeros(1))
        atomic_replace(tmp, dst)
        os.remove(dst)
        return True
    except OSError:
        for f in (os.path.join(cand, ".probe.tmp.npz"),
                  os.path.join(cand, ".probe.npz")):
            try:
                os.remove(f)
            except OSError:
                pass
        return False


def default_cache_dir():
    """First user cache dir that really works; None when nothing does.

    Deliberately NOT next to the executable: judges may run the frozen app
    from a read-only location (Program Files, synced or network folders),
    where a failing write used to abort the whole simulation.
    """
    global _CACHE_DIR
    if _CACHE_DIR is not None:
        return _CACHE_DIR or None
    roots = [os.environ.get("LOCALAPPDATA"),
             os.path.join(os.path.expanduser("~"), ".fiberscope"),
             tempfile.gettempdir()]
    for root in roots:
        if not root:
            continue
        cand = os.path.join(root, "FiberScope", "cache")
        if _probe_cache_dir(cand):
            _CACHE_DIR = cand
            return cand
    _CACHE_DIR = ""
    return None


def _engine_input_digest(graph) -> str:
    """Versioned, exact Engine2 input identity (not a full graph fingerprint).

    Node rows are sorted by stable ID; physical edges retain insertion order
    and direction. A constant Z offset, radius and material are not included.
    Nonplanar geometry and edge-owned curve samples reject before cache lookup.
    Validate before coercion so masks, fractional IDs and overflow cannot alias.
    """
    def identity(value):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, Integral) or
                not -(2**63) <= int(value) < 2**63):
            raise ValueError("Engine2 cache IDs must be signed int64 integers")
        return int(value)

    def reject_masks(value):
        if np.ma.isMaskedArray(value) or value is np.ma.masked:
            raise ValueError("Engine2 cache inputs must not be masked")
        if isinstance(value, (list, tuple)):
            for item in value:
                reject_masks(item)
        elif isinstance(value, np.ndarray) and value.dtype.kind == "O":
            for item in value.flat:
                reject_masks(item)

    node_ids = sorted(identity(node) for node in graph.nodes)
    node_index = {node: row for row, node in enumerate(node_ids)}
    edge_ids = [identity(edge) for edge in graph.edges]
    # node_positions() and edge_array() can otherwise discard masks/coerce IDs.
    for node in graph.nodes.values():
        reject_masks(node.position)
    positions = graph.node_positions()
    if np.ma.isMaskedArray(positions):
        raise ValueError("Engine2 cache positions must not be masked")
    positions = np.asarray(positions)
    if (positions.ndim != 2 or positions.shape[0] != len(node_ids) or
            positions.shape[1] < 2 or positions.dtype.kind not in "iuf"):
        raise ValueError("Engine2 cache positions must be numeric (N, >=2)")
    _validate_planar_geometry(graph, positions)
    xy = np.asarray(positions[:, :2], dtype="<f8")
    if not np.isfinite(xy).all():
        raise ValueError("Engine2 cache XY must be finite")
    endpoints = []
    for edge in graph.edges.values():
        pair = [identity(edge.node_i), identity(edge.node_j)]
        if any(node not in node_index for node in pair):
            raise ValueError("Engine2 cache edges must reference existing nodes")
        endpoints.append([node_index[node] for node in pair])
    extra = graph.metadata.get("weld_triplets", [])
    reject_masks(extra)
    # Object conversion preserves individual integer types (including bool).
    trips = np.asarray(extra, dtype=object)
    if trips.shape == (0,):
        trips = trips.reshape(0, 3)
    if trips.ndim != 2 or trips.shape[1] != 3:
        raise ValueError("Engine2 cache weld_triplets must have shape (T, 3)")
    mapped = []
    for trip in trips:
        ids = [identity(node) for node in trip]
        if any(node not in node_index for node in ids):
            raise ValueError("Engine2 cache weld_triplets must reference existing nodes")
        mapped.append([node_index[node] for node in ids])
    h = hashlib.sha256(b"fiberscope-engine2-input-v1\0")
    for name, array in (
            ("node_ids", np.asarray(node_ids, dtype="<i8")),
            ("xy", xy),
            ("edge_ids", np.asarray(edge_ids, dtype="<i8")),
            ("endpoint_rows", np.asarray(endpoints, dtype="<i8").reshape(-1, 2)),
            ("weld_triplet_rows", np.asarray(mapped, dtype="<i8").reshape(-1, 3))):
        header = json.dumps([name, array.dtype.str, array.shape],
                            separators=(",", ":")).encode("ascii")
        h.update(len(header).to_bytes(8, "little"))
        h.update(header)
        h.update(array.tobytes(order="C"))
    return h.hexdigest()


def cache_tag(factory, cfg, *, graph) -> str:
    """Key for an already built, post-weld graph; never invokes the builder."""
    raw = json.dumps({"v": ENGINE_VERSION, "h": ENGINE_SRC_HASH,
                      "s": factory.key(), "c": asdict(cfg),
                      "mf": MAX_FRAMES, "input_v1": _engine_input_digest(graph)},
                     sort_keys=True, default=str)
    # Explicit namespace prevents an old configuration-only filename hit.
    return "g1_" + hashlib.sha256(raw.encode()).hexdigest()


def _load_cache(path):
    """Read a cached run; None when unreadable so the caller recomputes."""
    try:
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
    except Exception as e:
        print("[simcache] drop unreadable cache %s: %s"
              % (os.path.basename(path), e))
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def _save_cache(path, run):
    """Atomic write; a failure must never break a simulation."""
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
    try:
        np.savez_compressed(tmp, **kw)
        atomic_replace(tmp, path)
    except OSError as e:
        print("[simcache] cache write skipped (%s)" % e)
        try:
            os.remove(tmp)
        except OSError:
            pass


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
    grip_pct: float = 0.05   # width fraction clamped at each end
    weld_intersections: bool = False

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
            n_increments=self.n_increments, grip_pct=self.grip_pct)


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
    cache_hit: bool = False     # runtime only, never persisted in NPZ
    cache_key: str = ""         # empty when no cache path is selected

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
    _validate_planar_geometry(graph, graph.node_positions())
    if cfg.weld_intersections:
        from .welding import IntersectionWelder
        graph = IntersectionWelder().apply(graph)
    n_nodes = len(graph.node_positions())

    path = None
    key = ""
    if cache_dir:
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError:
            cache_dir = None      # read-only location -> run without cache
    if cache_dir:
        key = cache_tag(factory, cfg, graph=graph)
        path = os.path.join(cache_dir, f"run_{key}.npz")
        if use_cache and os.path.exists(path):
            cached = _load_cache(path)
            if cached is not None:
                cached.cache_hit = True
                cached.cache_key = key
                return cached

    res = Engine2(graph, cfg.engine_cfg(n_nodes)).run(progress_cb=progress_cb)
    res.metadata['weld_intersections'] = cfg.weld_intersections
    # no fracture in this model -> grip force must not soften; clip the
    # tiny numerical transients so the curve is monotone non-decreasing
    res.force_curve = np.maximum.accumulate(res.force_curve)
    res = _downsample(res)
    run = StretchRun(res.frames_xy, res.edges, res.edge_rest, res.edge_strain,
                     res.strain_levels, res.force_curve, res.left_nodes,
                     res.right_nodes, res.energies, res.contact_pairs_last,
                     res.contact_frames, res.contact_counts, res.metadata)

    run.cache_key = key
    if path:
        _save_cache(path, run)
    return run

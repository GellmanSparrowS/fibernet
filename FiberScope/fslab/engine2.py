"""engine2: numpy mass-spring core with bending + contact + energy bookkeeping.

Physics model (all tunables in Engine2Config):
  axial   : springs on graph edges, F = k * (L-L0)/L0          (fibernet-consistent)
  bending : next-nearest springs on degree-2 nodes (discretized beam bending;
            degree-2 nodes are points where a single fiber passes through)
  contact : pairwise repulsion for non-neighbour nodes closer than r_contact
Boundary: left 10% percentile nodes fixed, right 10% displaced (fibernet grip).
"""
from dataclasses import dataclass, field
import time

import numpy as np


@dataclass
class Engine2Config:
    target_stretch: float = 2.0
    stiffness: float = 1.0e5
    k_bend_frac: float = 0.25      # k_bend = k_bend_frac * stiffness
    k_contact: float = 2.0e5
    r_contact: float = 0.8
    use_bending: bool = True
    use_contact: bool = True
    dt: float = 1.0e-5
    drag: float = 1500.0           # v *= exp(-drag*dt) each step
    num_steps: int = 16000
    save_interval: int = 1000
    ramp_fraction: float = 0.2
    n_increments: int = 0      # >0: quasi-static increments (affine predictor + relaxation); 0 = legacy ramp dynamics
    contact_rebuild: int = 100     # rebuild contact candidates every N steps
    grip_pct: float = 0.10


@dataclass
class Engine2Result:
    frames_xy: np.ndarray
    edges: np.ndarray
    edge_rest: np.ndarray
    edge_strain: np.ndarray
    strain_levels: np.ndarray
    force_curve: np.ndarray
    left_nodes: np.ndarray
    right_nodes: np.ndarray
    energies: dict                 # axial/bend/contact/kinetic/work (F,)
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


def _grips(x: np.ndarray, pct: float):
    n = max(int(len(x) * pct), 1)
    order = np.argsort(x)
    return (np.sort(order[:n]).astype(np.int32),
            np.sort(order[-n:]).astype(np.int32))


def _degree2_triplets(edges: np.ndarray, n: int):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    trips = []
    for v in range(n):
        if len(adj[v]) == 2:
            a, c = sorted(adj[v])
            trips.append((a, v, c))
    return np.array(trips, dtype=np.int64).reshape(-1, 3) if trips else \
        np.zeros((0, 3), dtype=np.int64)


def _excluded_pairs(edges, trips, n):
    ex = set()
    for a, b in edges:
        ex.add((min(int(a), int(b)), max(int(a), int(b))))
    for a, b, c in trips:
        ex.add((min(int(a), int(c)), max(int(a), int(c))))
    return ex


class Engine2:
    def __init__(self, graph, cfg: Engine2Config = None):
        self.cfg = cfg or Engine2Config()
        pos = np.asarray(graph.node_positions(), dtype=np.float64)[:, :2]
        edges = np.asarray(graph.edge_array(), dtype=np.int64)[:, :2]
        self.n = pos.shape[0]
        self.pos0 = pos
        self.edges = edges
        self.ea, self.eb = edges[:, 0], edges[:, 1]
        self.rest = np.linalg.norm(pos[self.eb] - pos[self.ea], axis=1)
        self.trips = _degree2_triplets(edges, self.n)
        if len(self.trips):
            self.na, self.nb, self.nc = (self.trips[:, 0], self.trips[:, 1],
                                         self.trips[:, 2])
            self.nnn_rest = np.linalg.norm(pos[self.nc] - pos[self.na], axis=1)
        else:
            self.na = self.nb = self.nc = np.zeros(0, dtype=np.int64)
            self.nnn_rest = np.zeros(0)
        self.excluded = _excluded_pairs(edges, self.trips, self.n)
        self.left, self.right = _grips(pos[:, 0], self.cfg.grip_pct)
        self._right_disp0 = pos[self.right].copy()
        # pairs already in contact in the as-printed state never count as
        # NEW contact events (they are part of the manufactured structure)
        all_cand = self._contact_candidates(pos)
        if len(all_cand):
            d0 = np.linalg.norm(pos[all_cand[:, 1]] - pos[all_cand[:, 0]],
                                axis=1)
            self._rest_close = set(
                tuple(pair) for pair, dd in zip(all_cand, d0)
                if dd < 1.2 * self.cfg.r_contact)
        else:
            self._rest_close = set()

    # ---------------- contact candidates ----------------
    def _contact_candidates(self, pos, fresh=False):
        rc = self.cfg.r_contact
        cell = np.floor(pos / rc).astype(np.int64)
        buckets = {}
        for i in range(self.n):
            buckets.setdefault((cell[i, 0], cell[i, 1]), []).append(i)
        pairs = []
        seen = set()
        for (cx, cy), ids in buckets.items():
            neigh = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neigh.extend(buckets.get((cx + dx, cy + dy), ()))
            for i in ids:
                for j in neigh:
                    if j <= i:
                        continue
                    key = (i, j)
                    if key in seen or key in self.excluded:
                        continue
                    if fresh and key in self._rest_close:
                        continue
                    seen.add(key)
                    pairs.append(key)
        return np.array(pairs, dtype=np.int64).reshape(-1, 2)

    # ---------------- force / record helpers ----------------
    def _forces(self, pos, cand, step):
        cfg = self.cfg
        k = cfg.stiffness
        kb = cfg.k_bend_frac * cfg.stiffness
        kc = cfg.k_contact
        rc = cfg.r_contact
        f = np.zeros_like(pos)

        # axial springs
        d = pos[self.eb] - pos[self.ea]
        L = np.linalg.norm(d, axis=1)
        Ls = np.maximum(L, 1e-12)
        dirv = d / Ls[:, None]
        T = k * (L - self.rest) / self.rest
        Fv = T[:, None] * dirv
        np.add.at(f, self.ea, Fv)
        np.add.at(f, self.eb, -Fv)

        # bending (next-nearest springs along fibers)
        if cfg.use_bending and len(self.na):
            db = pos[self.nc] - pos[self.na]
            Lb = np.linalg.norm(db, axis=1)
            Lbs = np.maximum(Lb, 1e-12)
            dirb = db / Lbs[:, None]
            Tb = kb * (Lb - self.nnn_rest) / np.maximum(self.nnn_rest, 1e-9)
            Fb = Tb[:, None] * dirb
            np.add.at(f, self.na, Fb)
            np.add.at(f, self.nc, -Fb)

        # contact repulsion
        if cfg.use_contact:
            if step % cfg.contact_rebuild == 1:
                cand = self._contact_candidates(pos, fresh=True)
            if len(cand):
                dc = pos[cand[:, 1]] - pos[cand[:, 0]]
                Lc = np.linalg.norm(dc, axis=1)
                mask = (Lc < rc) & (Lc > 1e-9)
                if mask.any():
                    idx = cand[mask]
                    dcm, Lcm = dc[mask], Lc[mask]
                    dirc = dcm / Lcm[:, None]
                    Fcv = (kc * (rc - Lcm))[:, None] * dirc
                    np.add.at(f, idx[:, 0], -Fcv)
                    np.add.at(f, idx[:, 1], Fcv)
        return f, cand

    def _record(self, pos, cand, Fg, vel, W, rec):
        cfg = self.cfg
        k = cfg.stiffness
        kb = cfg.k_bend_frac * cfg.stiffness
        kc = cfg.k_contact
        rc = cfg.r_contact
        rec["frames"].append(pos.copy())
        dS = pos[self.eb] - pos[self.ea]
        LS = np.linalg.norm(dS, axis=1)
        rec["strains"].append((LS - self.rest) / self.rest)
        rec["spans"].append(pos[self.right, 0].max() - pos[self.left, 0].min())
        rec["e_ax"].append(0.5 * k * (LS - self.rest) ** 2 / self.rest)
        if cfg.use_bending and len(self.na):
            dB = pos[self.nc] - pos[self.na]
            LB = np.linalg.norm(dB, axis=1)
            rec["e_bn"].append(0.5 * kb * (LB - self.nnn_rest) ** 2 /
                               np.maximum(self.nnn_rest, 1e-9))
        else:
            rec["e_bn"].append(np.zeros(0))
        if cfg.use_contact and len(cand):
            dC = pos[cand[:, 1]] - pos[cand[:, 0]]
            LC = np.linalg.norm(dC, axis=1)
            m = LC < rc
            rec["c_pairs"].append(cand[m] if m.any() else
                                  np.zeros((0, 2), dtype=np.int64))
            rec["c_counts"].append(int(m.sum()))
            rec["e_ct"].append(0.5 * kc * (rc - LC[m]) ** 2 if m.any()
                               else np.zeros(0))
        else:
            rec["c_pairs"].append(np.zeros((0, 2), dtype=np.int64))
            rec["c_counts"].append(0)
            rec["e_ct"].append(np.zeros(0))
        rec["forces"].append(Fg)
        rec["e_kin"].append(0.5 * (vel ** 2).sum())
        rec["work"].append(W)

    # ---------------- main loop ----------------
    def run(self, progress_cb=None) -> Engine2Result:
        cfg = self.cfg
        n = self.n
        pos = self.pos0.copy()
        vel = np.zeros_like(pos)
        Lx = pos[:, 0].max() - pos[:, 0].min()
        target_disp = Lx * (cfg.target_stretch - 1.0)
        decay = np.exp(-cfg.drag * cfg.dt)

        fixed = set(self.left.tolist())
        right_set = set(self.right.tolist())
        free = np.array([i not in fixed and i not in right_set
                         for i in range(n)])

        rec = {kk: [] for kk in ("frames", "strains", "spans", "forces",
                                 "c_pairs", "c_counts", "e_ax", "e_bn",
                                 "e_ct", "e_kin", "work")}
        W = 0.0
        prev_disp = 0.0
        cand = self._contact_candidates(pos, fresh=True) if cfg.use_contact \
            else np.zeros((0, 2), dtype=np.int64)

        # frame 0 (undeformed) so every saved array matches frames length
        self._record(pos, cand, 0.0, vel, 0.0, rec)

        t0 = time.time()
        total = cfg.num_steps
        if cfg.n_increments > 0:
            # quasi-static continuation: affine predictor per load increment
            # plus short dynamic relaxation. Grip-only dynamics diffuses too
            # slowly across big samples (left end would lag); the predictor
            # keeps the strain field uniform while forces add the non-affine
            # physics (buckling / contact / bending).
            M = int(cfg.n_increments)
            R = max(cfg.num_steps // M, 1)
            total = R * M
            x0min = float(self.pos0[:, 0].min())
            done = 0
            for inc in range(1, M + 1):
                disp = target_disp * (inc / M)
                cur = float(pos[:, 0].max() - x0min)
                if cur > 1e-9:
                    pos[:, 0] = x0min + (pos[:, 0] - x0min) \
                        * ((Lx + disp) / cur)
                    pos[self.left] = self.pos0[self.left]
                for _ in range(R):
                    f, cand = self._forces(pos, cand, done + 1)
                    vel[free] = (vel[free] + f[free] * cfg.dt) * decay
                    pos[free] += vel[free] * cfg.dt
                    vel[self.left] = 0.0
                    pos[self.right] = self._right_disp0
                    pos[self.right, 0] += disp
                    vel[self.right] = 0.0
                    done += 1
                Fg = -f[self.right, 0].sum()
                W += Fg * (disp - prev_disp)
                prev_disp = disp
                self._record(pos, cand, Fg, vel, W, rec)
                if progress_cb is not None:
                    progress_cb(done, total)
        else:
            ramp = int(cfg.num_steps * cfg.ramp_fraction)
            for step in range(1, cfg.num_steps + 1):
                f, cand = self._forces(pos, cand, step)
                vel[free] = (vel[free] + f[free] * cfg.dt) * decay
                pos[free] += vel[free] * cfg.dt
                vel[self.left] = 0.0
                s = min(step / ramp, 1.0) if ramp > 0 else 1.0
                disp = target_disp * s
                pos[self.right] = self._right_disp0
                pos[self.right, 0] += disp
                vel[self.right] = 0.0
                Fg = -f[self.right, 0].sum()
                W += Fg * (disp - prev_disp)
                prev_disp = disp
                if step % cfg.save_interval == 0 or step == cfg.num_steps:
                    self._record(pos, cand, Fg, vel, W, rec)
                    if progress_cb is not None:
                        progress_cb(step, cfg.num_steps)
        if progress_cb is not None:
            progress_cb(total, total)

        frames = np.array(rec["frames"], dtype=np.float32)
        span0 = self.pos0[self.right, 0].max() - self.pos0[self.left, 0].min()
        meta = {"engine": "engine2", "num_steps": cfg.num_steps,
                "dt": cfg.dt, "target_stretch": cfg.target_stretch,
                "use_bending": cfg.use_bending, "use_contact": cfg.use_contact,
                "n_increments": cfg.n_increments,
                "wall_seconds": round(time.time() - t0, 2)}
        return Engine2Result(
            frames_xy=frames, edges=self.edges.astype(np.int32),
            edge_rest=self.rest.astype(np.float32),
            edge_strain=np.array(rec["strains"], dtype=np.float32),
            strain_levels=(np.array(rec["spans"]) / span0).astype(np.float32),
            force_curve=np.array(rec["forces"], dtype=np.float32),
            left_nodes=self.left, right_nodes=self.right,
            energies={"axial": np.array([float(x.sum()) for x in rec["e_ax"]]),
                      "bend": np.array([float(x.sum()) for x in rec["e_bn"]]),
                      "contact": np.array([float(x.sum()) for x in rec["e_ct"]]),
                      "kinetic": np.array(rec["e_kin"]),
                      "work": np.array(rec["work"])},
            contact_pairs_last=(rec["c_pairs"][-1] if rec["c_pairs"] else
                                np.zeros((0, 2), dtype=np.int64)),
            contact_frames=rec["c_pairs"],
            contact_counts=np.array(rec["c_counts"], dtype=np.int32),
            metadata=meta)

"""NumPy-only structural descriptors and three-target surrogate MLP."""
import numpy as np

def light_features(pos, edges):
    """Pure-numpy structural descriptors (~14-dim, millisecond cost)."""
    pos = np.asarray(pos, float)
    raw_edges = np.asarray(edges)
    if (pos.ndim != 2 or pos.shape[1] < 2 or len(pos) < 2 or
            not np.isfinite(pos).all() or
            raw_edges.ndim != 2 or raw_edges.shape[1] < 2 or
            not len(raw_edges) or
            not np.issubdtype(raw_edges.dtype, np.integer)):
        raise ValueError('features require finite nodes and integer edges')
    pos = pos[:, :2]
    edges = np.asarray(raw_edges[:, :2], int)
    if (edges.min() < 0 or edges.max() >= len(pos) or
            np.any(edges[:, 0] == edges[:, 1])):
        raise ValueError('feature edges contain invalid node indices')
    n_nodes = pos.shape[0]
    n_edges = edges.shape[0]
    d = pos[edges[:, 1]] - pos[edges[:, 0]]
    lens = np.linalg.norm(d, axis=1)
    lm = float(lens.mean())
    ls = float(lens.std())
    cv = ls / max(lm, 1e-9)
    # orientation entropy: edge angles (axial, mod pi) in 12 bins
    ang = np.arctan2(d[:, 1], d[:, 0]) % np.pi
    hist, _ = np.histogram(ang, bins=12, range=(0.0, np.pi))
    p = hist / max(n_edges, 1)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(12.0))
    # degree distribution
    deg = np.bincount(edges.ravel(), minlength=n_nodes)
    d2 = float((deg == 2).mean())
    d3 = float((deg == 3).mean())
    d4 = float((deg == 4).mean())
    do = float(max(0.0, 1.0 - d2 - d3 - d4))
    c = pos - pos.mean(axis=0)
    rg = float(np.sqrt((c ** 2).sum(axis=1).mean()))
    span = np.ptp(pos, axis=0)
    diag = max(float(np.hypot(span[0], span[1])), 1e-9)
    aspect = span[0] / max(span[1], 1e-9)
    return np.array([
        lm, ls, cv, ent,            # edge-length stats + orientation entropy
        d2, d3, d4, do,             # degree distribution
        rg / diag,                  # radius of gyration (normalized)
        n_nodes / 1000.0, n_edges / 1000.0, n_edges / max(n_nodes, 1),
        aspect, lm / diag,
    ], dtype=np.float64)


class MLP:
    """Numpy MLP regressor: (d, hidden, 3), tanh hidden, linear output."""

    def __init__(self, hidden=32, seed=0):
        self.hidden = int(hidden)
        self.rng = np.random.default_rng(seed)
        self.W1 = self.b1 = self.W2 = self.b2 = None
        self.x_mean = self.x_std = self.y_mean = self.y_std = None

    def _init_weights(self, d):
        h = self.hidden
        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / (d + h)), (d, h))
        self.b1 = np.zeros(h)
        self.W2 = self.rng.normal(0.0, np.sqrt(2.0 / (h + 3)), (h, 3))
        self.b2 = np.zeros(3)

    def _forward(self, Xn):
        H = np.tanh(Xn @ self.W1 + self.b1)
        return H, H @ self.W2 + self.b2

    def train(self, X, Y, lr=1e-3, epochs=200, batch=16, val_frac=0.2,
              patience=15, min_delta=1e-4, epoch_cb=None, stop_cb=None):
        """Adam training with standardized I/O and early stopping on val MSE.

        Returns a history dict (train/val loss lists, epochs_run,
        best_epoch, early_stopped); best weights are restored at the end.
        """
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        n, d = X.shape
        if self.W1 is None or self.W1.shape[0] != d:
            self._init_weights(d)
        idx = self.rng.permutation(n)
        nv = int(n * val_frac) if n >= 5 else 0
        vi, ti = idx[:nv], idx[nv:]
        self.val_indices, self.train_indices = vi, ti
        self.x_mean = X[ti].mean(axis=0)
        self.x_std = X[ti].std(axis=0)
        self.x_std[self.x_std < 1e-12] = 1.0
        self.y_mean = Y[ti].mean(axis=0)
        self.y_std = Y[ti].std(axis=0)
        self.y_std[self.y_std < 1e-12] = 1.0
        Xn = (X - self.x_mean) / self.x_std
        Yn = (Y - self.y_mean) / self.y_std
        if len(ti) == 0:
            ti = idx

        def mse(sel):
            _, out = self._forward(Xn[sel])
            return float(((out - Yn[sel]) ** 2).mean())

        state = {k: (np.zeros_like(p), np.zeros_like(p)) for k, p in
                 (('W1', self.W1), ('b1', self.b1),
                  ('W2', self.W2), ('b2', self.b2))}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        def step(name, grad):
            m, v = state[name]
            m[:] = beta1 * m + (1.0 - beta1) * grad
            v[:] = beta2 * v + (1.0 - beta2) * grad * grad
            return lr * (m / (1.0 - beta1 ** t)) / \
                (np.sqrt(v / (1.0 - beta2 ** t)) + eps)

        hist_tr, hist_va = [], []
        best, best_epoch, bad, best_w = np.inf, 0, 0, None
        for ep in range(1, int(epochs) + 1):
            if stop_cb is not None and stop_cb():
                break
            order = self.rng.permutation(len(ti))
            for s0 in range(0, len(order), int(batch)):
                bidx = ti[order[s0:s0 + int(batch)]]
                t += 1
                H, out = self._forward(Xn[bidx])
                err = 2.0 * (out - Yn[bidx]) / out.size
                gW2 = H.T @ err
                gb2 = err.sum(axis=0)
                dH = (err @ self.W2.T) * (1.0 - H ** 2)
                gW1 = Xn[bidx].T @ dH
                gb1 = dH.sum(axis=0)
                self.W2 -= step('W2', gW2)
                self.b2 -= step('b2', gb2)
                self.W1 -= step('W1', gW1)
                self.b1 -= step('b1', gb1)
            tr = mse(ti)
            va = mse(vi) if nv else tr
            hist_tr.append(tr)
            hist_va.append(va)
            if epoch_cb is not None:
                epoch_cb(ep, tr, va)
            if va < best - min_delta:
                best, best_epoch, bad = va, ep, 0
                best_w = (self.W1.copy(), self.b1.copy(),
                          self.W2.copy(), self.b2.copy())
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_w is not None:
            self.W1, self.b1, self.W2, self.b2 = best_w
        return {'train': hist_tr, 'val': hist_va,
                'epochs_run': len(hist_tr), 'best_epoch': best_epoch,
                'early_stopped': bad >= patience}

    def predict(self, x):
        x = np.asarray(x, float)
        single = (x.ndim == 1)
        Xn = (x - self.x_mean) / self.x_std
        if single:
            Xn = Xn[None, :]
        _, out = self._forward(Xn)
        out = out * self.y_std + self.y_mean
        return out[0] if single else out


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')

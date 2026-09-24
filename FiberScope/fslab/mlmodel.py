"""Surrogate modeling: structure features -> mechanical properties.

light_features : fast pure-numpy graph descriptors (~14-dim)
make_sample    : build + features + quick quasi-static engine2 run -> labels
gen_dataset    : resumable incremental dataset generation (npz append)
MLP            : numpy MLP regressor (tanh hidden, linear out, Adam,
                 standardization, early stopping)
"""
import gc
import json
import os

import numpy as np

from .engine2 import Engine2, Engine2Config
from .structure import StructureFactory

N_PTS = 5            # spectrum points sampled per deformation spectrum
TARGET_NAMES = ('peak', 'stiffness', 'toughness')

# fast sim config for label extraction (quasi-static, small step budget)
SIM_KW = dict(target_stretch=1.8, num_steps=1500, save_interval=500,
              n_increments=24)

from fibernet.ml.physical_surrogate import MLP, light_features, r2_score

def make_sample(factory):
    """One sample: x = light features, y = [peak, stiffness, toughness]."""
    g = factory.build()
    pos = np.asarray(g.node_positions(), float)[:, :2]
    edges = np.asarray(g.edge_array(), int)[:, :2]
    x = light_features(pos, edges)
    res = Engine2(g, Engine2Config(**SIM_KW)).run()
    del g
    f = np.asarray(res.force_curve, float)
    s = np.asarray(res.strain_levels, float)
    peak = float(f.max())
    # stiffness: linear fit on the early curve (5%..40% of the strain span)
    s0, s1 = float(s.min()), float(s.max())
    m = (s >= s0 + 0.05 * (s1 - s0)) & (s <= s0 + 0.40 * (s1 - s0))
    if int(m.sum()) < 3:
        m = np.ones(len(s), bool)
        m[0] = False
    k = float(np.polyfit(s[m], f[m], 1)[0])
    toughness = float((0.5 * (f[1:] + f[:-1]) * np.diff(s)).sum())
    return x, np.array([peak, k, toughness], dtype=np.float64)


def gen_dataset(unit, n=60, amp=.2, pert=.1, seed0=0, path=None,
                progress_cb=None, stop_cb=None, mode='physics', acquisition='random', model=None):
    from .dataset_stream import generate_dataset
    return generate_dataset(unit, n, amp, pert, seed0, path, progress_cb, stop_cb,
                            mode, acquisition, model)


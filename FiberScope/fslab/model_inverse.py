"""Learned scalar evaluation; never fabricate a physical force curve."""
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
from .inverse import run_inverse, SCALARS
from .mlmodel import light_features
from .storage import atomic_replace


@dataclass
class PredictedStructure:
    positions: np.ndarray
    edges: np.ndarray
    prediction: np.ndarray


def predict_structure(graph, model, target):
    if target not in SCALARS:
        raise ValueError('the learned model supports scalar targets, not force-curve shapes')
    positions = np.asarray(graph.node_positions(), float)[:, :2]
    edges = np.asarray(graph.edge_array(), int)[:, :2]
    prediction = np.asarray(model.predict(light_features(positions, edges)), float).reshape(3)
    if not np.isfinite(prediction).all():
        raise ValueError('non-finite model prediction')
    metric, sign = SCALARS[target]
    column = {'peak': 0, 'stiff': 1, 'tough': 2}[metric]
    return float(sign * prediction[column]), PredictedStructure(positions, edges, prediction)


def save_prediction(path, preview):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, positions=preview.positions, edges=preview.edges, prediction=preview.prediction)
    atomic_replace(temporary, path)


def load_prediction(path):
    with np.load(path, allow_pickle=False) as data:
        return PredictedStructure(data['positions'], data['edges'], data['prediction'])


def run_model_inverse(factory, model, target, budget=40, seed=7, amplitude=.6, callback=None, stop_cb=None):
    def build(unit, perturbation, spectrum):
        return replace(factory, unit=unit, seed=seed, perturbation=perturbation,
                       line_displacements=spectrum).build()
    result = run_inverse(build, target, budget, seed=seed, fixed_unit=factory.unit,
                         pts=max(1, factory.n_pts_per_side), initial_spec=factory,
                         amplitude=amplitude, callback=callback, stop_cb=stop_cb,
                         evaluator=lambda graph, goal: predict_structure(graph, model, goal))
    preview = result.pop('best_run')
    result.update(best_run=None, best_preview=preview, evaluation_mode='surrogate',
                  best_prediction=preview.prediction.tolist() if preview is not None else None)
    return result

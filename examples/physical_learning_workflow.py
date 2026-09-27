"""Train the shared three-target physical surrogate from an NPZ dataset.

Run: python -m examples.physical_learning_workflow --output-dir demo_learning
The built-in fallback is synthetic data for an API check, not physical labels.
Provide --data data.npz with X (N, 14) and Y (N, 3) for real training.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from fibernet.ml import PhysicalRegressor
from fibernet.ml.physical_learning import FEATURE_SCHEMA


class PhysicalLearningWorkflow:
    def __init__(self, output_dir, data_path=None, model="ridge", seed=23):
        self.output_dir = Path(output_dir)
        self.data_path = Path(data_path) if data_path else None
        self.model = model
        self.seed = int(seed)

    def _data(self):
        if self.data_path:
            with np.load(self.data_path, allow_pickle=False) as saved:
                X = np.asarray(saved["X"], float)
                Y = np.asarray(saved["Y"], float)
            source = str(self.data_path)
        else:
            rng = np.random.default_rng(self.seed)
            X = rng.normal(size=(80, 14))
            Y = np.column_stack((2 * X[:, 0] + X[:, 1],
                                 X[:, 2] - X[:, 3],
                                 X[:, 4] + 0.5 * X[:, 5]))
            source = "synthetic API demonstration; not physical labels"
        if (X.ndim != 2 or X.shape[1] != 14 or
                Y.shape != (len(X), 3) or len(X) < 10 or
                not np.isfinite(X).all() or not np.isfinite(Y).all()):
            raise ValueError("expected finite X(N,14), Y(N,3), N>=10")
        return X, Y, source

    def run(self):
        X, Y, source = self._data()
        regressor = PhysicalRegressor(key=self.model, seed=self.seed)
        history = regressor.train(X, Y)
        ids = regressor.val_indices
        prediction = np.asarray(regressor.predict(X[ids]), float)
        rmse = np.sqrt(np.mean((prediction - Y[ids]) ** 2, axis=0))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / "held_out_predictions.npz"
        temporary = target.with_suffix(".tmp.npz")
        try:
            np.savez_compressed(temporary, row_indices=ids,
                                actual=Y[ids], predicted=prediction)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        summary = {"source": source, "schema": FEATURE_SCHEMA,
                   "model": self.model, "seed": self.seed,
                   "samples": len(X), "validation_rows": len(ids),
                   "rmse_by_target": rmse.tolist(),
                   "targets": ["peak", "stiffness", "toughness"],
                   "best_step": history["best_epoch"]}
        target = self.output_dir / "summary.json"
        temporary = target.with_suffix(".tmp.json")
        try:
            temporary.write_text(json.dumps(summary, indent=2) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        print("[physical_learning_workflow] held-out predictions saved")
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--output-dir", default="demo_learning")
    parser.add_argument("--model", default="ridge")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    PhysicalLearningWorkflow(args.output_dir, args.data,
                             args.model, args.seed).run()

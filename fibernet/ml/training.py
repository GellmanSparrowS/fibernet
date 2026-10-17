"""
Advanced Training Utilities for FiberNet ML Pipeline.

Features
--------
- Training loop with checkpointing, early stopping, LR scheduling
- Memory-safe DataLoader with batch size control
- Learning rate warmup + cosine annealing
- Gradient clipping for stability
- Training history tracking and export
- Model comparison and selection
- Cross-validation with PyTorch models

Examples
--------
>>> from fibernet.ml.training import train_model, cross_validate_torch
>>> history = train_model(
...     model, X_train, y_train,
...     X_val=X_val, y_val=y_val,
...     epochs=100, batch_size=64,
...     early_stopping_patience=10,
...     checkpoint_path="checkpoints/best.pt",
... )
>>> print(f"Best val loss: {history['best_val_loss']:.4f}")
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _regression_metrics(y_true, y_pred):
    """Score tabular point predictions without broadcasting output axes.

    R² is the uniform average of per-output scores. Constant outputs score
    one for exact predictions and zero otherwise; fewer than two rows yield
    NaN R². Only scalar (N,) and column (N, 1) shapes are interchangeable.
    """
    targets = np.asarray(y_true)
    preds = np.asarray(y_pred)
    if targets.ndim not in (1, 2) or preds.ndim not in (1, 2):
        raise ValueError("regression metrics require (N,) or (N, K) arrays")
    if targets.ndim == 1:
        targets = targets[:, None]
    if preds.ndim == 1:
        preds = preds[:, None]
    if targets.shape != preds.shape:
        raise ValueError(
            f"prediction shape {preds.shape} does not match target shape {targets.shape}"
        )
    if not targets.shape[0] or not targets.shape[1]:
        raise ValueError("regression metrics require nonempty arrays")
    residual = targets - preds
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    if len(targets) < 2:
        r2 = float("nan")
    else:
        ss_res = np.sum(residual ** 2, axis=0)
        ss_tot = np.sum((targets - targets.mean(axis=0)) ** 2, axis=0)
        scores = np.ones_like(ss_res, dtype=float)
        varying = ss_tot != 0
        scores[varying] = 1.0 - ss_res[varying] / ss_tot[varying]
        scores[~varying & (ss_res != 0)] = 0.0
        r2 = float(np.mean(scores))
    return {"r2": r2, "rmse": rmse}


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch required: pip install torch")


if HAS_TORCH:

    def _point_prediction(raw, target, adapter):
        """Validate declared point predictions before any array broadcasting."""
        if adapter is not None:
            # Clone also isolates loss inputs from an adapter's in-place edits.
            with torch.no_grad():
                try:
                    point = adapter(raw.detach().clone())
                except Exception as exc:
                    raise ValueError("metric_prediction adapter failed") from exc
            if not isinstance(point, torch.Tensor):
                raise TypeError("metric_prediction must return a torch.Tensor")
            if point.device != raw.device:
                raise ValueError("metric_prediction must preserve the input device")
        else:
            point = raw
        if point.is_complex() or point.dtype == torch.bool:
            if adapter is not None:
                raise ValueError("metric_prediction must return real numeric predictions")
            reason = "unsupported_prediction_dtype"
            return None, {"status": "unavailable", "reason": reason,
                          "r2": reason, "rmse": reason}
        shape = tuple(point.shape)
        target_shape = tuple(target.shape)
        normalized = shape if len(shape) == 2 else shape + (1,)
        normalized_target = target_shape if len(target_shape) == 2 else target_shape + (1,)
        valid = (point.ndim in (1, 2) and target.ndim in (1, 2)
                 and normalized == normalized_target and all(normalized))
        if not valid:
            if adapter is not None:
                raise ValueError(
                    f"metric_prediction shape {shape} does not match tabular "
                    f"target shape {target_shape}; expected (N,) or (N, K)"
                )
            return None, {
                "status": "unavailable", "reason": "prediction_shape_mismatch",
                "prediction_shape": list(shape), "target_shape": list(target_shape),
                "r2": "prediction_shape_mismatch", "rmse": "prediction_shape_mismatch",
            }
        if adapter is not None and not torch.isfinite(point).all():
            raise ValueError("metric_prediction must return finite point predictions")
        if not torch.isfinite(point).all() or not torch.isfinite(target).all():
            reason = "nonfinite_prediction_or_target"
            return None, {"status": "unavailable", "reason": reason,
                          "r2": reason, "rmse": reason}
        return point.detach(), None


    class TrainingHistory:
        """Training history tracker.

        Stores per-epoch metrics and supports export to JSON/CSV.
        """

        def __init__(self):
            self.train_losses: List[float] = []
            self.val_losses: List[float] = []
            self.train_metrics: Dict[str, List[float]] = {}
            self.val_metrics: Dict[str, List[Optional[float]]] = {}
            self.history_schema_version: Optional[int] = None
            self.metric_definition: Optional[str] = None
            self.metric_prediction: Optional[str] = None
            self.val_metric_status: List[Dict[str, Any]] = []
            self.lr_history: List[float] = []
            self.epoch_times: List[float] = []
            self.best_val_loss: float = float("inf")
            self.best_epoch: int = 0

        def update(
            self,
            epoch: int,
            train_loss: float,
            val_loss: Optional[float] = None,
            train_metrics: Optional[Dict[str, float]] = None,
            val_metrics: Optional[Dict[str, Optional[float]]] = None,
            lr: float = 0.0,
            epoch_time: float = 0.0,
        ):
            self.train_losses.append(train_loss)
            if val_loss is not None:
                self.val_losses.append(val_loss)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
            self.lr_history.append(lr)
            self.epoch_times.append(epoch_time)

            if train_metrics:
                for k, v in train_metrics.items():
                    self.train_metrics.setdefault(k, []).append(v)
            if val_metrics:
                for k, v in val_metrics.items():
                    self.val_metrics.setdefault(k, []).append(v)

        def to_dict(self) -> Dict[str, Any]:
            data = {
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "train_metrics": self.train_metrics,
                "val_metrics": self.val_metrics,
                "lr_history": self.lr_history,
                "epoch_times": self.epoch_times,
                "best_val_loss": self.best_val_loss,
                "best_epoch": self.best_epoch,
            }

            # Generic and legacy histories must not acquire point semantics.
            if self.metric_definition is not None:
                data.update(
                    history_schema_version=self.history_schema_version,
                    metric_definition=self.metric_definition,
                    metric_prediction=self.metric_prediction,
                    val_metric_status=self.val_metric_status,
                )
            return data

        def save(self, path: str):
            with open(path, "w") as fh:
                json.dump(self.to_dict(), fh, indent=2)

        @classmethod
        def load(cls, path: str) -> "TrainingHistory":
            with open(path) as fh:
                data = json.load(fh)
            hist = cls()
            hist.train_losses = data.get("train_losses", [])
            hist.val_losses = data.get("val_losses", [])
            hist.train_metrics = data.get("train_metrics", {})
            hist.val_metrics = data.get("val_metrics", {})
            hist.lr_history = data.get("lr_history", [])
            hist.epoch_times = data.get("epoch_times", [])
            hist.best_val_loss = data.get("best_val_loss", float("inf"))
            hist.best_epoch = data.get("best_epoch", 0)
            hist.history_schema_version = data.get("history_schema_version")
            hist.metric_definition = data.get("metric_definition")
            hist.metric_prediction = data.get("metric_prediction")
            hist.val_metric_status = data.get("val_metric_status", [])
            return hist

        def plot(self, save_path: Optional[str] = None, show: bool = False):
            """Plot training curves."""
            import matplotlib
            if not show:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_facecolor("#0a0a0f")

            # Loss curve
            ax = axes[0]
            ax.set_facecolor("#0a0a0f")
            epochs = range(1, len(self.train_losses) + 1)
            ax.plot(epochs, self.train_losses, color="#b388ff", linewidth=1.5, label="Train")
            if self.val_losses:
                ax.plot(epochs, self.val_losses, color="#82b1ff", linewidth=1.5, label="Val")
                ax.axhline(self.best_val_loss, color="#ff8a80", linestyle="--",
                          alpha=0.6, label=f"Best: {self.best_val_loss:.4f}")
            ax.set_xlabel("Epoch", color="#aaa")
            ax.set_ylabel("Loss", color="#aaa")
            ax.set_title("Loss Curve", color="#ddd")
            ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#aaa")
            ax.tick_params(colors="#888")
            for sp in ax.spines.values():
                sp.set_color("#333")

            # LR curve
            ax = axes[1]
            ax.set_facecolor("#0a0a0f")
            if self.lr_history:
                ax.plot(epochs, self.lr_history, color="#69f0ae", linewidth=1.5)
            ax.set_xlabel("Epoch", color="#aaa")
            ax.set_ylabel("Learning Rate", color="#aaa")
            ax.set_title("LR Schedule", color="#ddd")
            ax.tick_params(colors="#888")
            for sp in ax.spines.values():
                sp.set_color("#333")

            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight",
                           facecolor=fig.get_facecolor())
            if show:
                plt.show()
            return fig


    def get_lr_scheduler(
        optimizer: optim.Optimizer,
        schedule_type: str = "cosine",
        epochs: int = 100,
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        **kwargs,
    ):
        """Create learning rate scheduler.

        Parameters
        ----------
        schedule_type : str
            "cosine", "step", "exponential", "plateau", "warmup_cosine"
        epochs : int
            Total training epochs.
        warmup_epochs : int
            Linear warmup epochs (for warmup_cosine).
        min_lr : float
            Minimum learning rate.
        """
        if schedule_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=min_lr,
            )
        elif schedule_type == "step":
            step_size = kwargs.get("step_size", max(epochs // 3, 1))
            gamma = kwargs.get("gamma", 0.5)
            return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif schedule_type == "exponential":
            gamma = kwargs.get("gamma", 0.98)
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif schedule_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=kwargs.get("factor", 0.5),
                patience=kwargs.get("patience", 10), min_lr=min_lr,
            )
        elif schedule_type == "warmup_cosine":
            warmup = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_epochs,
            )
            cosine = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=min_lr,
            )
            return optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
            )
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")


    def train_model(
        model: nn.Module,
        X_train: Union[np.ndarray, "torch.Tensor"],
        y_train: Union[np.ndarray, "torch.Tensor"],
        *,
        X_val: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
        y_val: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_fn: Optional[Callable] = None,
        metric_prediction: Optional[Callable] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler_type: str = "warmup_cosine",
        warmup_epochs: int = 5,
        early_stopping_patience: int = 15,
        early_stopping_min_delta: float = 1e-5,
        gradient_clip: float = 1.0,
        checkpoint_path: Optional[str] = None,
        verbose: bool = True,
        device: Optional[str] = None,
        num_workers: int = 0,
    ) -> TrainingHistory:
        """Train a PyTorch model with full feature support.

        Parameters
        ----------
        model : nn.Module
            Model to train.
        X_train, y_train : array-like
            Training data.
        X_val, y_val : array-like, optional
            Validation data; supply both arrays with at least one row.
        epochs : int
            Maximum training epochs.
        batch_size : int
            Mini-batch size (controls memory usage).
        lr : float
            Initial learning rate.
        weight_decay : float
            L2 regularization.
        loss_fn : callable, optional
            Loss function. Default: MSELoss.
        metric_prediction : callable, optional
            Detached raw output Tensor -> tabular point Tensor on the same
            device. Used only for metrics, never loss. See ml-metric-usage.md.
            Without an adapter incompatible outputs have unavailable metrics.
        optimizer : Optimizer, optional
            Custom optimizer. Default: AdamW.
        scheduler_type : str
            LR schedule type.
        warmup_epochs : int
            Warmup epochs for warmup_cosine schedule.
        early_stopping_patience : int
            Stop after N validation epochs without improvement. 0 = disabled.
            Without validation, all requested epochs run.
        early_stopping_min_delta : float
            Minimum improvement to count as progress.
        gradient_clip : float
            Max gradient norm. 0 = disabled.
        checkpoint_path : str, optional
            Save this run's best validation model here. Best weights are also
            restored in memory without a path; existing files are never read.
        verbose : bool
            Print epoch progress.
        device : str, optional
            "cuda" or "cpu". Auto-detected if None.
        num_workers : int
            DataLoader workers.

        Returns
        -------
        TrainingHistory
        """
        if metric_prediction is not None and not callable(metric_prediction):
            raise TypeError("metric_prediction must be callable or None")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        model = model.to(device)

        if loss_fn is None:
            loss_fn = nn.MSELoss()
        if optimizer is None:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = get_lr_scheduler(
            optimizer, scheduler_type, epochs, warmup_epochs,
        )

        # Build DataLoaders
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be supplied together")
        X_t = torch.as_tensor(X_train, dtype=torch.float32)
        y_t = torch.as_tensor(y_train, dtype=torch.float32)
        train_loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            X_v = torch.as_tensor(X_val, dtype=torch.float32)
            y_v = torch.as_tensor(y_val, dtype=torch.float32)
            if len(X_v) == 0:
                raise ValueError("Validation data must contain at least one row")
            val_loader = DataLoader(
                TensorDataset(X_v, y_v),
                batch_size=batch_size, shuffle=False, num_workers=num_workers,
            )

        history = TrainingHistory()
        history.history_schema_version = 2
        history.metric_definition = "point_regression_uniform_output_v1"
        history.metric_prediction = "explicit_callable" if metric_prediction is not None else "identity"
        patience_counter = 0
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            t0 = time.time()

            # Train
            model.train()
            train_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                # Handle shape mismatch (e.g., pred=[B,1] vs yb=[B])
                if pred.dim() > yb.dim() and pred.shape[-1] == 1:
                    pred = pred.squeeze(-1)
                elif yb.dim() > pred.dim() and yb.shape[-1] == 1:
                    yb = yb.squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(n_batches, 1)

            # Validate
            val_loss = None
            val_metrics = {"r2": None, "rmse": None}
            metric_status = {
                "status": "unavailable", "reason": "validation_unavailable",
                "r2": "validation_unavailable", "rmse": "validation_unavailable",
            }
            if val_loader is not None:
                model.eval()
                v_loss = 0.0
                v_n = 0
                all_preds = []
                all_targets = []
                unavailable = None
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        pred = model(xb)
                        raw_pred, raw_target = pred, yb
                        if pred.dim() > yb.dim() and pred.shape[-1] == 1:
                            pred = pred.squeeze(-1)
                        elif yb.dim() > pred.dim() and yb.shape[-1] == 1:
                            yb = yb.squeeze(-1)
                        loss = loss_fn(pred, yb)
                        v_loss += loss.item()
                        v_n += 1
                        point, reason = _point_prediction(raw_pred, raw_target, metric_prediction)
                        if reason is not None:
                            unavailable = reason
                        else:
                            all_preds.append(point.cpu().reshape(len(yb), -1))
                            all_targets.append(raw_target.cpu().reshape(len(yb), -1))

                val_loss = v_loss / max(v_n, 1)

                if unavailable is not None:
                    metric_status = unavailable
                else:
                    preds = torch.cat(all_preds).numpy()
                    targets = torch.cat(all_targets).numpy()
                    val_metrics = _regression_metrics(targets, preds)
                    metric_status = {"status": "available", "r2": "available", "rmse": "available"}
                    if len(targets) < 2:
                        val_metrics["r2"] = None
                        metric_status.update(status="partial", r2="insufficient_samples")
                    for name, value in val_metrics.items():
                        if value is not None and not np.isfinite(value):
                            val_metrics[name] = None
                            metric_status[name] = "nonfinite_metric"
                    if any(value is None for value in val_metrics.values()):
                        metric_status["status"] = (
                            "unavailable" if all(value is None for value in val_metrics.values())
                            else "partial"
                        )
            history.val_metric_status.append(metric_status)

            # LR schedule
            current_lr = optimizer.param_groups[0]["lr"]
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if val_loss is not None else train_loss)
            else:
                scheduler.step()

            epoch_time = time.time() - t0
            history.update(epoch, train_loss, val_loss, val_metrics=val_metrics,
                          lr=current_lr, epoch_time=epoch_time)

            # Select only a validation state produced during this invocation.
            if val_loss is not None:
                if val_loss < best_val_loss - early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = copy.deepcopy(model.state_dict())
                    if checkpoint_path:
                        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                        torch.save({
                            "model_state_dict": best_state,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "val_loss": val_loss,
                        }, checkpoint_path)
                else:
                    patience_counter += 1

                if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (patience={early_stopping_patience})")
                    break

            if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
                val_str = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
                if val_loss is not None:
                    r2 = val_metrics["r2"]
                    val_str += (f", R²={r2:.3f}" if r2 is not None
                                else f", R²=unavailable ({metric_status['r2']})")
                print(f"Epoch {epoch:3d} | loss={train_loss:.4f}{val_str} "
                      f"| lr={current_lr:.2e} | {epoch_time:.1f}s")

        # Deepcopy preserves tensor values, non-tensor extra state, and metadata.
        if best_state is not None:
            model.load_state_dict(best_state)

        return history


    def cross_validate_torch(
        model_factory: Callable[[], nn.Module],
        X: Union[np.ndarray, "torch.Tensor"],
        y: Union[np.ndarray, "torch.Tensor"],
        *,
        n_folds: int = 5,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        random_state: int = 42,
        metric_prediction: Optional[Callable] = None,
        verbose: bool = False,
        **train_kwargs,
    ) -> Dict[str, Any]:
        """K-fold cross-validation for tabular point-regression models.

        Scores use uniform-average per-output R², with exact constant outputs
        scoring one and other constant outputs zero. R² is NaN for validation
        folds with fewer than two rows. Scalar (N,) and (N, 1) shapes are
        interchangeable. Distribution Tensors require an explicit
        metric_prediction adapter; incompatible identity predictions are rejected.
        The adapter runs both during training and on restored-model scoring.

        Parameters
        ----------
        model_factory : callable
            Function that returns a fresh model instance.
        X, y : array-like
            Data.
        n_folds : int
            Number of folds.
        epochs : int
            Training epochs per fold.

        Returns
        -------
        dict
            Keys: fold_r2s, fold_rmses, mean_r2, std_r2, mean_rmse, std_rmse
        """
        from sklearn.model_selection import KFold

        X = np.asarray(X) if not isinstance(X, np.ndarray) else X
        y = np.asarray(y) if not isinstance(y, np.ndarray) else y

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_r2s = []
        fold_rmses = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            model = model_factory()
            history = train_model(
                model, X[train_idx], y[train_idx],
                X_val=X[val_idx], y_val=y[val_idx],
                epochs=epochs, batch_size=batch_size, lr=lr,
                verbose=verbose, metric_prediction=metric_prediction, **train_kwargs,
            )

            # Evaluate on val fold
            model.eval()
            device = next(model.parameters()).device
            with torch.no_grad():
                X_v = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
                y_v = y[val_idx]
                raw = model(X_v)
                if metric_prediction is not None:
                    point, reason = _point_prediction(
                        raw, torch.as_tensor(y_v, device=device), metric_prediction
                    )
                    if reason is not None:
                        raise ValueError(f"CV point metrics unavailable: {reason['reason']}")
                else:
                    point = raw
                preds = point.detach().cpu().numpy()

            metrics = _regression_metrics(y_v, preds)
            r2, rmse = metrics["r2"], metrics["rmse"]

            fold_r2s.append(float(r2))
            fold_rmses.append(rmse)

            if verbose:
                print(f"Fold {fold}: R²={r2:.3f}, RMSE={rmse:.4f}")

        return {
            "fold_r2s": fold_r2s,
            "fold_rmses": fold_rmses,
            "mean_r2": float(np.mean(fold_r2s)),
            "std_r2": float(np.std(fold_r2s)),
            "mean_rmse": float(np.mean(fold_rmses)),
            "std_rmse": float(np.std(fold_rmses)),
        }


    def save_model_bundle(
        model: nn.Module,
        path: str,
        *,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
        history: Optional[TrainingHistory] = None,
        metadata: Optional[Dict] = None,
    ):
        """Save model + metadata as a single .pt bundle.

        Parameters
        ----------
        model : nn.Module
            Trained model.
        path : str
            Output path (.pt).
        feature_names, target_names : list of str
            Feature/target names for inference.
        history : TrainingHistory, optional
            Training history.
        metadata : dict, optional
            Extra metadata.
        """
        bundle = {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
            "feature_names": feature_names or [],
            "target_names": target_names or [],
            "metadata": metadata or {},
        }
        if history:
            bundle["history"] = history.to_dict()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, path)


    def load_model_bundle(
        path: str,
        model: Optional[nn.Module] = None,
        model_class: Optional[type] = None,
        **model_kwargs,
    ) -> Tuple[nn.Module, Dict]:
        """Load model from bundle.

        Parameters
        ----------
        path : str
            Path to .pt file.
        model : nn.Module, optional
            Pre-created model to load weights into.
        model_class : type, optional
            Model class to instantiate (if model not given).
        **model_kwargs
            Arguments for model_class constructor.

        Returns
        -------
        model : nn.Module
        bundle_info : dict
            feature_names, target_names, metadata, history.
        """
        bundle = torch.load(path, weights_only=False)

        if model is None:
            if model_class is None:
                raise ValueError("Either model or model_class must be provided")
            model = model_class(**model_kwargs)

        model.load_state_dict(bundle["model_state_dict"])
        model.eval()

        info = {
            "feature_names": bundle.get("feature_names", []),
            "target_names": bundle.get("target_names", []),
            "metadata": bundle.get("metadata", {}),
            "history": bundle.get("history"),
        }

        return model, info

else:
    # Stubs
    class TrainingHistory:
        def __init__(self, *a, **kw):
            _require_torch()

    def train_model(*a, **kw):
        _require_torch()

    def cross_validate_torch(*a, **kw):
        _require_torch()

    def save_model_bundle(*a, **kw):
        _require_torch()

    def load_model_bundle(*a, **kw):
        _require_torch()

    def get_lr_scheduler(*a, **kw):
        _require_torch()

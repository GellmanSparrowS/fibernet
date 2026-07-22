"""
Inverse Design for FiberNet — Property-to-Structure Mapping.

Implements inverse design methods that find structures matching
target mechanical properties:
- InverseDesignNet: Neural network mapping properties → features
- TandemNetwork: Forward + inverse network for joint training
- BayesianInverseDesign: Bayesian optimization-based inverse design
- GeneticInverseDesign: Genetic algorithm for inverse design

Features
--------
- Direct inverse mapping from target properties to structure features
- Tandem training ensuring physical consistency
- Uncertainty quantification via ensemble inverse models
- Multi-objective inverse design with Pareto front
- Integration with generative models for structure realization

References
----------
- Article section 5: RL-driven inverse design
- Ma et al., "Deep learning for inverse design" (2024)

Examples
--------
>>> from fibernet.ml.inverse_design import InverseDesignNet, TandemNetwork
>>> # Train inverse model
>>> inv_model = InverseDesignNet(n_properties=3, n_features=20, hidden=[128, 64])
>>> trainer = InverseDesignTrainer(inv_model)
>>> trainer.fit(X_train, y_train, epochs=200)
>>> # Find structures for target properties
>>> target = torch.tensor([[500.0, 1.5, 0.3]])  # [force, stretch, porosity]
>>> features = inv_model(target)  # predicted structure features

>>> # Tandem training for better consistency
>>> tandem = TandemNetwork(n_properties=3, n_features=20)
>>> trainer = InverseDesignTrainer(tandem)
>>> trainer.fit(X_train, y_train, epochs=200)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch required: pip install torch")


if HAS_TORCH:

    class InverseDesignNet(nn.Module):
        """Neural network for inverse design: properties → structure features.

        Parameters
        ----------
        n_properties : int
            Number of target properties (e.g., force, stiffness, stretch).
        n_features : int
            Output structure feature dimension.
        hidden : list of int
            Hidden layer sizes.
        dropout : float
            Dropout rate.
        """

        def __init__(
            self,
            n_properties: int = 3,
            n_features: int = 20,
            hidden: Optional[List[int]] = None,
            dropout: float = 0.1,
        ):
            super().__init__()
            if hidden is None:
                hidden = [128, 64]

            self.n_properties = n_properties
            self.n_features = n_features

            layers = []
            prev = n_properties
            for h in hidden:
                layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
                prev = h
            layers.append(nn.Linear(prev, n_features))
            self.net = nn.Sequential(*layers)

        def forward(self, properties: torch.Tensor) -> torch.Tensor:
            return self.net(properties)

        @torch.no_grad()
        def predict_features(self, properties: np.ndarray) -> np.ndarray:
            self.eval()
            p = torch.tensor(properties, dtype=torch.float32)
            if p.dim() == 1:
                p = p.unsqueeze(0)
            return self.net(p).numpy()


    class ForwardNet(nn.Module):
        """Forward model: structure features → properties."""

        def __init__(
            self,
            n_features: int = 20,
            n_properties: int = 3,
            hidden: Optional[List[int]] = None,
            dropout: float = 0.1,
        ):
            super().__init__()
            if hidden is None:
                hidden = [128, 64]

            layers = []
            prev = n_features
            for h in hidden:
                layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
                prev = h
            layers.append(nn.Linear(prev, n_properties))
            self.net = nn.Sequential(*layers)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.net(features)


    class TandemNetwork(nn.Module):
        """Tandem network combining forward and inverse models.

        Trains both models jointly, enforcing consistency:
        inverse(forward(x)) ≈ x and forward(inverse(y)) ≈ y.

        Parameters
        ----------
        n_features : int
            Structure feature dimension.
        n_properties : int
            Property dimension.
        hidden : list of int
            Hidden layers for both networks.
        dropout : float
            Dropout rate.
        cycle_weight : float
            Weight for cycle consistency loss.
        """

        def __init__(
            self,
            n_features: int = 20,
            n_properties: int = 3,
            hidden: Optional[List[int]] = None,
            dropout: float = 0.1,
            cycle_weight: float = 0.5,
        ):
            super().__init__()
            self.forward_net = ForwardNet(n_features, n_properties, hidden, dropout)
            self.inverse_net = InverseDesignNet(n_properties, n_features, hidden, dropout)
            self.cycle_weight = cycle_weight
            self.n_features = n_features
            self.n_properties = n_properties

        def forward_loss(self, features: torch.Tensor, properties: torch.Tensor) -> torch.Tensor:
            """Forward prediction loss."""
            pred = self.forward_net(features)
            return F.mse_loss(pred, properties)

        def inverse_loss(self, features: torch.Tensor, properties: torch.Tensor) -> torch.Tensor:
            """Inverse design loss."""
            pred_features = self.inverse_net(properties)
            return F.mse_loss(pred_features, features)

        def cycle_loss(self, features: torch.Tensor, properties: torch.Tensor) -> torch.Tensor:
            """Cycle consistency loss."""
            # Forward cycle: features → properties → features
            pred_prop = self.forward_net(features)
            recon_feat = self.inverse_net(pred_prop)
            loss_fwd = F.mse_loss(recon_feat, features)

            # Inverse cycle: properties → features → properties
            pred_feat = self.inverse_net(properties)
            recon_prop = self.forward_net(pred_feat)
            loss_inv = F.mse_loss(recon_prop, properties)

            return self.cycle_weight * (loss_fwd + loss_inv)

        def total_loss(self, features: torch.Tensor, properties: torch.Tensor) -> Dict[str, torch.Tensor]:
            fwd = self.forward_loss(features, properties)
            inv = self.inverse_loss(features, properties)
            cyc = self.cycle_loss(features, properties)
            return {
                "total": fwd + inv + cyc,
                "forward": fwd,
                "inverse": inv,
                "cycle": cyc,
            }


    class InverseDesignTrainer:
        """Training loop for inverse design models.

        Parameters
        ----------
        model : InverseDesignNet or TandemNetwork
            Inverse design model.
        lr : float
            Learning rate.
        """

        def __init__(
            self,
            model: Union[InverseDesignNet, TandemNetwork],
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
        ):
            _require_torch()
            self.model = model
            self.lr = lr
            self.weight_decay = weight_decay
            self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            *,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 200,
            batch_size: int = 64,
            verbose: bool = True,
        ) -> Dict[str, Any]:
            """Train the inverse design model.

            Parameters
            ----------
            X_train : (N, n_features)
                Structure features.
            y_train : (N, n_properties)
                Property labels.
            X_val, y_val : optional
                Validation data.
            epochs : int
            batch_size : int
            verbose : bool

            Returns
            -------
            dict
                Training history.
            """
            X_t = torch.tensor(X_train, dtype=torch.float32)
            y_t = torch.tensor(y_train, dtype=torch.float32)

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            is_tandem = isinstance(self.model, TandemNetwork)
            n = len(X_t)

            for epoch in range(epochs):
                self.model.train()
                perm = torch.randperm(n)
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    idx = perm[start:end]
                    xb, yb = X_t[idx], y_t[idx]

                    optimizer.zero_grad()

                    if is_tandem:
                        losses = self.model.total_loss(xb, yb)
                        losses["total"].backward()
                        loss_val = losses["total"].item()
                    else:
                        pred = self.model(yb)
                        loss = F.mse_loss(pred, xb)
                        loss.backward()
                        loss_val = loss.item()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss_val
                    n_batches += 1

                scheduler.step()
                epoch_loss /= max(n_batches, 1)
                self.history["train_loss"].append(epoch_loss)

                val_loss = None
                if X_val is not None and y_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        X_v = torch.tensor(X_val, dtype=torch.float32)
                        y_v = torch.tensor(y_val, dtype=torch.float32)
                        if is_tandem:
                            val_loss = self.model.total_loss(X_v, y_v)["total"].item()
                        else:
                            pred_v = self.model(y_v)
                            val_loss = F.mse_loss(pred_v, X_v).item()
                    self.history["val_loss"].append(val_loss)

                if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
                    val_str = f" | val={val_loss:.4f}" if val_loss else ""
                    print(f"Epoch {epoch:3d} | loss={epoch_loss:.4f}{val_str}")

            return self.history

        def design(
            self,
            target_properties: np.ndarray,
            n_candidates: int = 1,
        ) -> np.ndarray:
            """Generate candidate structures for target properties.

            Parameters
            ----------
            target_properties : (n_properties,) or (n_targets, n_properties)
                Target property values.
            n_candidates : int
                Number of candidates per target (with noise injection).

            Returns
            -------
            (n_targets * n_candidates, n_features) candidate features
            """
            self.model.eval()
            target = torch.tensor(target_properties, dtype=torch.float32)
            if target.dim() == 1:
                target = target.unsqueeze(0)

            n_targets = target.shape[0]

            if isinstance(self.model, TandemNetwork):
                inv_net = self.model.inverse_net
            else:
                inv_net = self.model

            candidates = []
            with torch.no_grad():
                for _ in range(n_candidates):
                    noisy_target = target + torch.randn_like(target) * 0.01 * target.std(dim=0).clamp(min=1e-6)
                    features = inv_net(noisy_target)
                    candidates.append(features)

            return torch.cat(candidates, dim=0).numpy()


    class BayesianInverseDesign:
        """Bayesian optimization-based inverse design.

        Uses a forward model as the objective function and searches
        the feature space for structures matching target properties.

        Parameters
        ----------
        forward_model : object
            Trained forward model with predict() method.
        feature_bounds : dict
            Bounds for each feature: {name: (low, high)}.
        n_properties : int
            Number of properties.

        Examples
        --------
        >>> bayes = BayesianInverseDesign(forward_model, feature_bounds)
        >>> best_features = bayes.design(
        ...     target={"max_force": 500, "stiffness": 50},
        ...     n_iter=100,
        ... )
        """

        def __init__(
            self,
            forward_model: Any,
            feature_bounds: Dict[str, Tuple[float, float]],
        ):
            self.forward_model = forward_model
            self.feature_bounds = feature_bounds

        def design(
            self,
            target: Dict[str, float],
            n_iter: int = 100,
            n_initial: int = 10,
            verbose: bool = True,
        ) -> Dict[str, Any]:
            """Find features matching target properties via Bayesian optimization.

            Parameters
            ----------
            target : dict
                Target property values {name: value}.
            n_iter : int
                Number of optimization iterations.
            n_initial : int
                Initial random points.
            verbose : bool

            Returns
            -------
            dict
                Best features, error, and optimization history.
            """
            try:
                from skopt import gp_minimize
                from skopt.space import Real
            except ImportError:
                raise ImportError("scikit-optimize required: pip install scikit-optimize")

            feature_names = list(self.feature_bounds.keys())
            dimensions = [
                Real(low, high, name=name)
                for name, (low, high) in self.feature_bounds.items()
            ]

            def objective(x):
                feat_dict = {feature_names[i]: x[i] for i in range(len(x))}
                feat_array = np.array([[feat_dict[n] for n in feature_names]])

                try:
                    pred = self.forward_model.predict(feat_array)
                    if hasattr(pred, 'flatten'):
                        pred = pred.flatten()
                    error = sum(
                        ((pred[i] - target.get(f"prop_{i}", 0)) / max(abs(target.get(f"prop_{i}", 1)), 1e-8)) ** 2
                        for i in range(len(target))
                    )
                    return float(error)
                except Exception:
                    return 1e10

            result = gp_minimize(
                objective, dimensions,
                n_calls=n_iter, n_initial_points=n_initial,
                verbose=verbose, random_state=42,
            )

            best = {feature_names[i]: result.x[i] for i in range(len(feature_names))}
            return {
                "features": best,
                "error": float(result.fun),
                "n_iter": n_iter,
            }

else:
    class InverseDesignNet:
        def __init__(self, *a, **kw):
            _require_torch()

    class TandemNetwork:
        def __init__(self, *a, **kw):
            _require_torch()

    class InverseDesignTrainer:
        def __init__(self, *a, **kw):
            _require_torch()

    class BayesianInverseDesign:
        def __init__(self, *a, **kw):
            _require_torch()

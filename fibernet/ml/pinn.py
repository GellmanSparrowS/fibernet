"""
Physics-Informed Neural Networks (PINN) for FiberNet.

Embeds physical laws into neural network training:
- Equilibrium conditions (force balance)
- Constitutive laws (stress-strain relations)
- Boundary conditions
- Energy conservation

Components
----------
- PhysicsLoss: Computes physics-based loss terms
- PINNModel: Wraps any FiberNet model with physics constraints
- train_pinn: Training loop with physics + data loss

Examples
--------
>>> from fibernet.ml.pinn import PhysicsLoss, PINNModel, train_pinn
>>> from fibernet.ml.models import FiberMLP
>>> base_model = FiberMLP(n_features=20, n_outputs=3)  # [ux, uy, sigma]
>>> pinn = PINNModel(base_model, physics_loss=PhysicsLoss(lambda_eq=1.0))
>>> history = train_pinn(pinn, X_train, y_train, collocation_points=X_coll)
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

    class PhysicsLoss(nn.Module):
        """Physics-based loss terms for fiber network mechanics.

        Implements:
        - Equilibrium: ∇·σ + f = 0 (force balance)
        - Constitutive: σ = E·ε (Hooke's law)
        - Boundary conditions (Dirichlet/Neumann)
        - Energy minimization (strain energy)

        Parameters
        ----------
        lambda_eq : float
            Weight for equilibrium loss.
        lambda_const : float
            Weight for constitutive law loss.
        lambda_bc : float
            Weight for boundary condition loss.
        lambda_energy : float
            Weight for energy minimization.
        youngs_modulus : float
            Material Young's modulus (E).
        """

        def __init__(
            self,
            lambda_eq: float = 1.0,
            lambda_const: float = 0.5,
            lambda_bc: float = 1.0,
            lambda_energy: float = 0.1,
            youngs_modulus: float = 1e9,
        ):
            super().__init__()
            self.lambda_eq = lambda_eq
            self.lambda_const = lambda_const
            self.lambda_bc = lambda_bc
            self.lambda_energy = lambda_energy
            self.E = youngs_modulus

        def equilibrium_loss(
            self,
            pred_stress: torch.Tensor,
            coords: torch.Tensor,
        ) -> torch.Tensor:
            """Force balance: ∂σ_xx/∂x + ∂σ_xy/∂y = 0, etc.

            Parameters
            ----------
            pred_stress : (N, 3) stress components [σ_xx, σ_yy, σ_xy]
            coords : (N, 2) spatial coordinates [x, y], requires_grad=True
            """
            if not coords.requires_grad:
                coords = coords.requires_grad_(True)

            # Compute gradients via autograd
            sigma_xx = pred_stress[:, 0:1]
            sigma_yy = pred_stress[:, 1:2]
            sigma_xy = pred_stress[:, 2:3]

            grad_sigma_xx = torch.autograd.grad(
                sigma_xx, coords, grad_outputs=torch.ones_like(sigma_xx),
                create_graph=True, retain_graph=True,
            )[0]
            grad_sigma_xy = torch.autograd.grad(
                sigma_xy, coords, grad_outputs=torch.ones_like(sigma_xy),
                create_graph=True, retain_graph=True,
            )[0]
            grad_sigma_yy = torch.autograd.grad(
                sigma_yy, coords, grad_outputs=torch.ones_like(sigma_yy),
                create_graph=True, retain_graph=True,
            )[0]

            # ∂σ_xx/∂x + ∂σ_xy/∂y = 0
            eq_x = grad_sigma_xx[:, 0] + grad_sigma_xy[:, 1]
            # ∂σ_xy/∂x + ∂σ_yy/∂y = 0
            eq_y = grad_sigma_xy[:, 0] + grad_sigma_yy[:, 1]

            return (eq_x ** 2).mean() + (eq_y ** 2).mean()

        def constitutive_loss(
            self,
            pred_stress: torch.Tensor,
            pred_strain: torch.Tensor,
        ) -> torch.Tensor:
            """Hooke's law: σ = E·ε (1D simplification).

            Parameters
            ----------
            pred_stress : (N, ...) predicted stress
            pred_strain : (N, ...) predicted strain
            """
            return F.mse_loss(pred_stress, self.E * pred_strain)

        def boundary_loss(
            self,
            pred_displacement: torch.Tensor,
            target_displacement: torch.Tensor,
            boundary_mask: torch.Tensor,
        ) -> torch.Tensor:
            """Dirichlet BC: u = u_target on boundary.

            Parameters
            ----------
            pred_displacement : (N, d) predicted displacement
            target_displacement : (N, d) target displacement
            boundary_mask : (N,) boolean mask for boundary nodes
            """
            if boundary_mask.sum() == 0:
                return torch.tensor(0.0)
            pred_bc = pred_displacement[boundary_mask]
            target_bc = target_displacement[boundary_mask]
            return F.mse_loss(pred_bc, target_bc)

        def energy_loss(
            self,
            pred_stress: torch.Tensor,
            pred_strain: torch.Tensor,
        ) -> torch.Tensor:
            """Strain energy: U = 0.5 * σ:ε should be minimized."""
            energy = 0.5 * (pred_stress * pred_strain).sum(dim=-1)
            return energy.mean() ** 2

        def forward(
            self,
            predictions: Dict[str, torch.Tensor],
            coords: Optional[torch.Tensor] = None,
            boundary_mask: Optional[torch.Tensor] = None,
            target_bc: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Compute total physics loss.

            Parameters
            ----------
            predictions : dict
                Model outputs. Expected keys depend on model configuration:
                - "displacement": (N, 2) displacement field
                - "stress": (N, 3) stress field [σ_xx, σ_yy, σ_xy]
                - "strain": (N, 3) strain field [ε_xx, ε_yy, ε_xy]
            coords : (N, 2), optional
                Spatial coordinates for equilibrium loss.
            boundary_mask : (N,), optional
                Boolean mask for boundary nodes.
            target_bc : (N, 2), optional
                Target boundary displacements.

            Returns
            -------
            dict
                Loss components and total.
            """
            losses = {}
            total = torch.tensor(0.0)

            if "stress" in predictions and "strain" in predictions:
                # Constitutive
                losses["constitutive"] = self.constitutive_loss(
                    predictions["stress"], predictions["strain"],
                )
                total = total + self.lambda_const * losses["constitutive"]

                # Energy
                losses["energy"] = self.energy_loss(
                    predictions["stress"], predictions["strain"],
                )
                total = total + self.lambda_energy * losses["energy"]

            if "stress" in predictions and coords is not None:
                losses["equilibrium"] = self.equilibrium_loss(
                    predictions["stress"], coords,
                )
                total = total + self.lambda_eq * losses["equilibrium"]

            if "displacement" in predictions and boundary_mask is not None and target_bc is not None:
                losses["boundary"] = self.boundary_loss(
                    predictions["displacement"], target_bc, boundary_mask,
                )
                total = total + self.lambda_bc * losses["boundary"]

            losses["total_physics"] = total
            return losses


    class PINNModel(nn.Module):
        """Physics-Informed Neural Network wrapper.

        Wraps a base model and adds physics-constrained output heads.

        Parameters
        ----------
        base_model : nn.Module
            Base neural network (e.g., FiberMLP, FiberResNet).
        physics_loss : PhysicsLoss
            Physics loss configuration.
        output_mode : str
            "field": Predict displacement + stress + strain fields
            "scalar": Predict scalar properties with physics regularization
        n_spatial_dims : int
            Number of spatial dimensions (2 or 3).

        Examples
        --------
        >>> base = FiberMLP(n_features=22, n_outputs=64)  # 20 features + 2 coords → 64 hidden
        >>> pinn = PINNModel(base, PhysicsLoss(lambda_eq=1.0))
        >>> # Input: [features, x, y]
        >>> out = pinn(torch.randn(32, 22))
        >>> # out = {"displacement": (32, 2), "stress": (32, 3), "strain": (32, 3)}
        """

        def __init__(
            self,
            base_model: nn.Module,
            physics_loss: Optional[PhysicsLoss] = None,
            output_mode: str = "field",
            n_spatial_dims: int = 2,
        ):
            _require_torch()
            super().__init__()

            self.base_model = base_model
            self.physics_loss = physics_loss or PhysicsLoss()
            self.output_mode = output_mode
            self.n_spatial_dims = n_spatial_dims

            # Determine base model output dimension
            base_out = self._infer_output_dim(base_model)

            if output_mode == "field":
                # Displacement (2 or 3) + Stress (3 or 6) + Strain (3 or 6)
                n_disp = n_spatial_dims
                n_stress = 3 if n_spatial_dims == 2 else 6
                n_strain = 3 if n_spatial_dims == 2 else 6

                self.disp_head = nn.Sequential(
                    nn.Linear(base_out, 32), nn.ReLU(),
                    nn.Linear(32, n_disp),
                )
                self.stress_head = nn.Sequential(
                    nn.Linear(base_out, 32), nn.ReLU(),
                    nn.Linear(32, n_stress),
                )
                self.strain_head = nn.Sequential(
                    nn.Linear(base_out, 32), nn.ReLU(),
                    nn.Linear(32, n_strain),
                )
            else:
                self.scalar_head = nn.Linear(base_out, 1)

        def _infer_output_dim(self, model):
            """Infer output dimension of base model."""
            if hasattr(model, "n_outputs"):
                # Check if it's actually the last hidden dim
                # For FiberMLP, n_outputs is the final layer output
                return model.n_outputs
            # Try a forward pass
            try:
                dummy = torch.randn(1, model.n_features if hasattr(model, "n_features") else 10)
                out = model(dummy)
                return out.shape[-1]
            except Exception:
                return 64

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """Forward pass.

            Parameters
            ----------
            x : (batch, n_features)
                Input features (may include spatial coords as last columns).

            Returns
            -------
            dict with keys based on output_mode.
            """
            h = self.base_model(x)

            if self.output_mode == "field":
                return {
                    "displacement": self.disp_head(h),
                    "stress": self.stress_head(h),
                    "strain": self.strain_head(h),
                }
            else:
                return {"prediction": self.scalar_head(h)}

        def compute_loss(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            coords: Optional[torch.Tensor] = None,
            boundary_mask: Optional[torch.Tensor] = None,
            target_bc: Optional[torch.Tensor] = None,
            data_weight: float = 1.0,
        ) -> Dict[str, torch.Tensor]:
            """Compute combined data + physics loss.

            Parameters
            ----------
            x : input features
            y : target values
            coords : spatial coordinates (for equilibrium)
            boundary_mask : boundary node mask
            target_bc : target boundary values
            data_weight : weight for data loss vs physics loss

            Returns
            -------
            dict of loss components
            """
            predictions = self.forward(x)
            losses = {}

            # Data loss
            if self.output_mode == "field":
                data_loss = F.mse_loss(predictions["displacement"], y)
            else:
                data_loss = F.mse_loss(predictions["prediction"], y)
            losses["data"] = data_loss

            # Physics loss
            if coords is not None or boundary_mask is not None:
                physics_losses = self.physics_loss(
                    predictions, coords, boundary_mask, target_bc,
                )
                losses.update(physics_losses)
                total = data_weight * data_loss + physics_losses["total_physics"]
            else:
                total = data_weight * data_loss

            losses["total"] = total
            return losses


    def train_pinn(
        model: PINNModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        coords: Optional[np.ndarray] = None,
        boundary_mask: Optional[np.ndarray] = None,
        target_bc: Optional[np.ndarray] = None,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        data_weight: float = 1.0,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train PINN with physics-informed loss.

        Parameters
        ----------
        model : PINNModel
            Physics-informed model.
        X_train, y_train : array-like
            Training data.
        X_val, y_val : array-like, optional
            Validation data.
        coords : array-like, optional
            Spatial coordinates for physics loss.
        boundary_mask : array-like, optional
            Boolean boundary mask.
        target_bc : array-like, optional
            Boundary condition values.
        epochs : int
            Training epochs.
        lr : float
            Learning rate.
        batch_size : int
            Batch size.
        data_weight : float
            Weight for data loss.

        Returns
        -------
        dict
            Training history.
        """
        from fibernet.ml.training import TrainingHistory

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        coords_t = torch.tensor(coords, dtype=torch.float32, requires_grad=True) if coords is not None else None
        bc_mask_t = torch.tensor(boundary_mask, dtype=torch.bool) if boundary_mask is not None else None
        target_bc_t = torch.tensor(target_bc, dtype=torch.float32) if target_bc is not None else None

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        history = TrainingHistory()

        n = len(X_t)
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]

                xb = X_t[idx]
                yb = y_t[idx]
                cb = coords_t[idx] if coords_t is not None else None
                bm = bc_mask_t[idx] if bc_mask_t is not None else None
                tb = target_bc_t[idx] if target_bc_t is not None else None

                optimizer.zero_grad()
                losses = model.compute_loss(xb, yb, cb, bm, tb, data_weight)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += losses["total"].item()
                n_batches += 1

            scheduler.step()
            epoch_loss /= max(n_batches, 1)

            # Validation
            val_loss = None
            if X_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    X_v = torch.tensor(X_val, dtype=torch.float32)
                    y_v = torch.tensor(y_val, dtype=torch.float32)
                    pred = model(X_v)
                    if model.output_mode == "field":
                        val_loss = F.mse_loss(pred["displacement"], y_v).item()
                    else:
                        val_loss = F.mse_loss(pred["prediction"], y_v).item()

            history.update(epoch, epoch_loss, val_loss,
                          lr=optimizer.param_groups[0]["lr"])

            if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
                val_str = f" | val={val_loss:.4f}" if val_loss else ""
                print(f"Epoch {epoch:3d} | loss={epoch_loss:.4f}{val_str}")

        return {"history": history, "final_loss": epoch_loss}


    def generate_collocation_points(
        n_points: int,
        bounds: Tuple[Tuple[float, float], ...],
        seed: int = 42,
    ) -> np.ndarray:
        """Generate collocation points for PINN physics loss.

        Parameters
        ----------
        n_points : int
            Number of points.
        bounds : tuple of (min, max) per dimension
            Domain bounds, e.g., ((0, 10), (0, 10)).
        seed : int
            Random seed.

        Returns
        -------
        (n_points, n_dims) array
        """
        rng = np.random.RandomState(seed)
        n_dims = len(bounds)
        points = np.zeros((n_points, n_dims))
        for d in range(n_dims):
            lo, hi = bounds[d]
            points[:, d] = rng.uniform(lo, hi, n_points)
        return points

else:
    class PhysicsLoss:
        def __init__(self, *a, **kw):
            _require_torch()

    class PINNModel:
        def __init__(self, *a, **kw):
            _require_torch()

    def train_pinn(*a, **kw):
        _require_torch()

    def generate_collocation_points(*a, **kw):
        _require_torch()

"""
Generative Models for FiberNet Structure Design.

Implements:
- FiberVAE: Variational Autoencoder for structure latent space
- FiberCVAE: Conditional VAE for property-guided generation
- LatentSpaceExplorer: Interpolation and sampling in latent space

The VAE learns a compressed representation of fiber network features,
enabling:
- Structure generation from latent vectors
- Property-guided sampling (CVAE)
- Latent space interpolation between structures
- Anomaly detection (reconstruction error)

Examples
--------
>>> from fibernet.ml.generative import FiberVAE, FiberCVAE
>>> vae = FiberVAE(n_features=20, latent_dim=8, hidden=[128, 64])
>>> # Train
>>> from fibernet.ml.generative import train_vae
>>> history = train_vae(vae, X_train, epochs=50)
>>> # Generate new structures
>>> z = vae.sample(n=10)
>>> X_gen = vae.decode(z)

>>> # Conditional generation
>>> cvae = FiberCVAE(n_features=20, n_conditions=3, latent_dim=8)
>>> # Generate with target properties [max_force=100, stiffness=50, energy=200]
>>> X_new = cvae.generate(conditions=torch.tensor([[100, 50, 200]]), n=5)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

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

    class FiberVAE(nn.Module):
        """Variational Autoencoder for fiber network structures.

        Parameters
        ----------
        n_features : int
            Input feature dimension.
        latent_dim : int
            Latent space dimension.
        hidden : list of int
            Encoder hidden layers. Decoder mirrors the architecture.
        beta : float
            KL divergence weight (beta-VAE). Default: 1.0.

        Examples
        --------
        >>> vae = FiberVAE(n_features=20, latent_dim=8, hidden=[128, 64])
        >>> x_recon, mu, log_var = vae(torch.randn(32, 20))
        >>> z = vae.encode(torch.randn(32, 20))
        >>> x_new = vae.decode(z)
        """

        def __init__(
            self,
            n_features: int,
            latent_dim: int = 8,
            hidden: Optional[List[int]] = None,
            beta: float = 1.0,
        ):
            _require_torch()
            super().__init__()

            if hidden is None:
                hidden = [128, 64]

            self.n_features = n_features
            self.latent_dim = latent_dim
            self.beta = beta

            # Encoder
            enc_layers = []
            prev = n_features
            for h in hidden:
                enc_layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)])
                prev = h
            self.encoder = nn.Sequential(*enc_layers)

            self.fc_mu = nn.Linear(hidden[-1], latent_dim)
            self.fc_logvar = nn.Linear(hidden[-1], latent_dim)

            # Decoder (mirror)
            dec_layers = []
            prev = latent_dim
            for h in reversed(hidden):
                dec_layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)])
                prev = h
            dec_layers.append(nn.Linear(prev, n_features))
            self.decoder = nn.Sequential(*dec_layers)

        def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Encode to latent distribution parameters."""
            h = self.encoder(x)
            mu = self.fc_mu(h)
            log_var = self.fc_logvar(h)
            return mu, log_var

        def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
            """Sample z from N(mu, sigma^2) using reparameterization trick."""
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            """Decode from latent space to feature space."""
            return self.decoder(z)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Full forward pass: encode → sample → decode.

            Returns
            -------
            x_recon : (batch, n_features) reconstructed features
            mu : (batch, latent_dim) mean
            log_var : (batch, latent_dim) log variance
            """
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            x_recon = self.decode(z)
            return x_recon, mu, log_var

        def sample(self, n: int = 10, device: str = "cpu") -> torch.Tensor:
            """Sample from prior N(0, I) and decode.

            Returns
            -------
            (n, n_features) generated feature vectors
            """
            z = torch.randn(n, self.latent_dim, device=device)
            return self.decode(z)

        def loss_function(
            self, x: torch.Tensor, x_recon: torch.Tensor,
            mu: torch.Tensor, log_var: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            """VAE loss = reconstruction + beta * KL divergence."""
            recon_loss = F.mse_loss(x_recon, x, reduction="sum") / x.shape[0]
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.shape[0]
            total = recon_loss + self.beta * kl_loss
            return {
                "total": total,
                "reconstruction": recon_loss,
                "kl_divergence": kl_loss,
            }

        def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
            """Per-sample reconstruction error (for anomaly detection)."""
            self.eval()
            with torch.no_grad():
                x_recon, _, _ = self.forward(x)
                return ((x - x_recon) ** 2).sum(dim=-1)


    class FiberCVAE(nn.Module):
        """Conditional VAE for property-guided structure generation.

        Conditions on target properties (e.g., desired max_force, stiffness)
        to generate structures that meet specific criteria.

        Parameters
        ----------
        n_features : int
            Input feature dimension.
        n_conditions : int
            Number of condition variables.
        latent_dim : int
            Latent space dimension.
        hidden : list of int
            Hidden layer sizes.
        beta : float
            KL weight.

        Examples
        --------
        >>> cvae = FiberCVAE(n_features=20, n_conditions=3, latent_dim=8)
        >>> # Train with conditions = target properties
        >>> x_recon, mu, log_var = cvae(X, conditions)
        >>> # Generate with target properties
        >>> X_new = cvae.generate(conditions=torch.tensor([[100, 50, 200]]), n=5)
        """

        def __init__(
            self,
            n_features: int,
            n_conditions: int = 1,
            latent_dim: int = 8,
            hidden: Optional[List[int]] = None,
            beta: float = 1.0,
        ):
            _require_torch()
            super().__init__()

            if hidden is None:
                hidden = [128, 64]

            self.n_features = n_features
            self.n_conditions = n_conditions
            self.latent_dim = latent_dim
            self.beta = beta

            # Encoder: input + conditions
            enc_layers = []
            prev = n_features + n_conditions
            for h in hidden:
                enc_layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)])
                prev = h
            self.encoder = nn.Sequential(*enc_layers)
            self.fc_mu = nn.Linear(hidden[-1], latent_dim)
            self.fc_logvar = nn.Linear(hidden[-1], latent_dim)

            # Decoder: latent + conditions
            dec_layers = []
            prev = latent_dim + n_conditions
            for h in reversed(hidden):
                dec_layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)])
                prev = h
            dec_layers.append(nn.Linear(prev, n_features))
            self.decoder = nn.Sequential(*dec_layers)

        def encode(self, x, c):
            h = self.encoder(torch.cat([x, c], dim=-1))
            return self.fc_mu(h), self.fc_logvar(h)

        def decode(self, z, c):
            return self.decoder(torch.cat([z, c], dim=-1))

        def forward(self, x, c):
            mu, log_var = self.encode(x, c)
            z = FiberVAE.reparameterize(self, mu, log_var)
            x_recon = self.decode(z, c)
            return x_recon, mu, log_var

        def generate(
            self,
            conditions: torch.Tensor,
            n: int = 10,
            temperature: float = 1.0,
        ) -> torch.Tensor:
            """Generate structures conditioned on target properties.

            Parameters
            ----------
            conditions : (1, n_conditions) or (n, n_conditions)
                Target properties.
            n : int
                Number of samples to generate.
            temperature : float
                Sampling temperature (lower = closer to mean).

            Returns
            -------
            (n, n_features) generated features
            """
            if conditions.shape[0] == 1:
                conditions = conditions.expand(n, -1)

            z = torch.randn(n, self.latent_dim) * temperature
            return self.decode(z, conditions)

        def loss_function(self, x, x_recon, mu, log_var):
            recon_loss = F.mse_loss(x_recon, x, reduction="sum") / x.shape[0]
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.shape[0]
            total = recon_loss + self.beta * kl_loss
            return {"total": total, "reconstruction": recon_loss, "kl_divergence": kl_loss}


    class LatentSpaceExplorer:
        """Explore and manipulate VAE/CVAE latent spaces.

        Parameters
        ----------
        model : FiberVAE or FiberCVAE
            Trained generative model.
        """

        def __init__(self, model: Union[FiberVAE, FiberCVAE]):
            self.model = model
            self.model.eval()

        def interpolate(
            self,
            z_start: torch.Tensor,
            z_end: torch.Tensor,
            n_steps: int = 10,
        ) -> torch.Tensor:
            """Linear interpolation in latent space.

            Parameters
            ----------
            z_start, z_end : (1, latent_dim) or (latent_dim,)
            n_steps : int

            Returns
            -------
            (n_steps, n_features) interpolated features
            """
            z_start = z_start.reshape(1, -1)
            z_end = z_end.reshape(1, -1)

            alphas = torch.linspace(0, 1, n_steps).unsqueeze(-1)
            z_interp = (1 - alphas) * z_start + alphas * z_end

            with torch.no_grad():
                return self.model.decode(z_interp)

        def find_nearest(
            self,
            target_features: torch.Tensor,
            dataset_features: np.ndarray,
            top_k: int = 5,
        ) -> List[int]:
            """Find nearest structures in latent space.

            Parameters
            ----------
            target_features : (1, n_features)
            dataset_features : (N, n_features)
            top_k : int

            Returns
            -------
            list of indices
            """
            with torch.no_grad():
                if isinstance(self.model, FiberCVAE):
                    raise ValueError("Use latent_from_cvae for CVAE models")
                z_target, _ = self.model.encode(target_features)
                z_all, _ = self.model.encode(torch.tensor(dataset_features, dtype=torch.float32))

                dists = ((z_all - z_target) ** 2).sum(dim=-1)
                return dists.topk(top_k, largest=False).indices.tolist()

        def latent_traversal(
            self,
            z_base: torch.Tensor,
            dim: int = 0,
            n_steps: int = 10,
            range_val: float = 3.0,
        ) -> torch.Tensor:
            """Traverse one latent dimension while keeping others fixed.

            Parameters
            ----------
            z_base : (1, latent_dim) base point
            dim : int which dimension to traverse
            n_steps : int
            range_val : float traversal range [-range_val, +range_val]

            Returns
            -------
            (n_steps, n_features)
            """
            z_base = z_base.reshape(1, -1).clone()
            z_traversal = z_base.repeat(n_steps, 1)
            z_traversal[:, dim] = torch.linspace(-range_val, range_val, n_steps)

            with torch.no_grad():
                return self.model.decode(z_traversal)


    def train_vae(
        model: Union[FiberVAE, FiberCVAE],
        X_train: np.ndarray,
        *,
        conditions: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        conditions_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train VAE or CVAE.

        Parameters
        ----------
        model : FiberVAE or FiberCVAE
        X_train : (N, n_features) training data
        conditions : (N, n_conditions) condition variables (for CVAE)
        X_val : validation data
        conditions_val : validation conditions
        epochs, lr, batch_size : training params

        Returns
        -------
        dict with history
        """
        from fibernet.ml.training import TrainingHistory

        X_t = torch.tensor(X_train, dtype=torch.float32)
        C_t = torch.tensor(conditions, dtype=torch.float32) if conditions is not None else None

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        history = TrainingHistory()

        n = len(X_t)
        is_cvae = isinstance(model, FiberCVAE)

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]
                xb = X_t[idx]

                optimizer.zero_grad()

                if is_cvae:
                    cb = C_t[idx]
                    x_recon, mu, log_var = model(xb, cb)
                else:
                    x_recon, mu, log_var = model(xb)

                losses = model.loss_function(xb, x_recon, mu, log_var)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += losses["total"].item()
                epoch_recon += losses["reconstruction"].item()
                epoch_kl += losses["kl_divergence"].item()
                n_batches += 1

            scheduler.step()
            epoch_loss /= max(n_batches, 1)

            val_loss = None
            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    X_v = torch.tensor(X_val, dtype=torch.float32)
                    if is_cvae:
                        C_v = torch.tensor(conditions_val, dtype=torch.float32)
                        x_r, mu, lv = model(X_v, C_v)
                    else:
                        x_r, mu, lv = model(X_v)
                    losses = model.loss_function(X_v, x_r, mu, lv)
                    val_loss = losses["total"].item()

            history.update(epoch, epoch_loss, val_loss,
                          train_metrics={"recon": epoch_recon / max(n_batches, 1),
                                        "kl": epoch_kl / max(n_batches, 1)},
                          lr=optimizer.param_groups[0]["lr"])

            if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
                val_str = f" | val={val_loss:.4f}" if val_loss else ""
                print(f"Epoch {epoch:3d} | loss={epoch_loss:.4f} "
                      f"recon={epoch_recon/max(n_batches,1):.4f} "
                      f"kl={epoch_kl/max(n_batches,1):.4f}{val_str}")

        return {"history": history, "final_loss": epoch_loss}

else:
    class FiberVAE:
        def __init__(self, *a, **kw):
            _require_torch()

    class FiberCVAE:
        def __init__(self, *a, **kw):
            _require_torch()

    class LatentSpaceExplorer:
        def __init__(self, *a, **kw):
            _require_torch()

    def train_vae(*a, **kw):
        _require_torch()

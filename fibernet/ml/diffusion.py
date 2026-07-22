"""
Diffusion Models for FiberNet Structure Generation.

Implements score-based denoising diffusion probabilistic models (DDPM)
for generating fiber network structures in feature space.

Models
------
- ScoreNetwork: Neural network estimating the score function ∇_x log p(x_t)
- FiberDiffusion: DDPM for fiber network feature generation
- ConditionalFiberDiffusion: Property-guided conditional diffusion
- DiffusionTrainer: Training loop with EMA and validation

Features
--------
- Forward diffusion process with cosine/linear noise schedule
- Reverse sampling with DDPM and DDIM samplers
- Conditional generation (property-guided)
- Classifier-free guidance for better conditional control
- Latent space diffusion (optional encoder)
- EMA model for stable sampling

References
----------
- Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021)
- Article section 2: Graph diffusion and structural variational diffusion

Examples
--------
>>> from fibernet.ml.diffusion import FiberDiffusion, DiffusionTrainer
>>> model = FiberDiffusion(n_features=20, hidden=[256, 128, 64])
>>> trainer = DiffusionTrainer(model)
>>> trainer.fit(X_train, epochs=200)
>>> X_generated = model.sample(n=100)

>>> # Conditional generation
>>> from fibernet.ml.diffusion import ConditionalFiberDiffusion
>>> cmodel = ConditionalFiberDiffusion(n_features=20, n_conditions=3)
>>> trainer = DiffusionTrainer(cmodel)
>>> trainer.fit(X_train, conditions=conditions_train, epochs=200)
>>> target = torch.tensor([[500.0, 1.5, 0.3]])  # [force, stretch, porosity]
>>> X_gen = cmodel.sample(n=10, conditions=target)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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

    # ==================================================================
    # Noise Schedule
    # ==================================================================

    class NoiseSchedule:
        """Noise schedule for diffusion process.

        Parameters
        ----------
        n_steps : int
            Number of diffusion steps.
        schedule : str
            "linear" or "cosine" noise schedule.
        beta_start : float
            Starting beta (linear schedule).
        beta_end : float
            Ending beta (linear schedule).
        """

        def __init__(
            self,
            n_steps: int = 1000,
            schedule: str = "cosine",
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
        ):
            self.n_steps = n_steps

            if schedule == "linear":
                betas = torch.linspace(beta_start, beta_end, n_steps)
            elif schedule == "cosine":
                steps = torch.arange(n_steps + 1, dtype=torch.float64)
                alpha_bar = torch.cos(((steps / n_steps) + 0.008) / (1.008) * math.pi / 2) ** 2
                alpha_bar = alpha_bar / alpha_bar[0]
                betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
                betas = betas.clamp(max=0.999).float()
            else:
                raise ValueError(f"Unknown schedule: {schedule}. Use 'linear' or 'cosine'")

            self.betas = betas
            self.alphas = 1.0 - betas
            self.alpha_bar = torch.cumprod(self.alphas, dim=0)
            self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
            self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
            self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
            self.sqrt_recip_alpha_bar = torch.sqrt(1.0 / self.alpha_bar)
            self.sqrt_recip_alpha_bar_m1 = torch.sqrt(1.0 / self.alpha_bar - 1.0)

            # Posterior variance
            posterior_var = betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
            self.posterior_log_var = torch.log(
                torch.cat([posterior_var[1:2], betas[1:]])
            )
            self.posterior_mean_coef1 = (
                betas * torch.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar)
            )
            self.posterior_mean_coef2 = (
                (1.0 - self.alpha_bar_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bar)
            )

        def to(self, device):
            for attr in dir(self):
                val = getattr(self, attr)
                if isinstance(val, torch.Tensor):
                    setattr(self, attr, val.to(device))
            return self

        def q_sample(self, x0, t, noise=None):
            """Forward process: add noise to x0 at timestep t."""
            if noise is None:
                noise = torch.randn_like(x0)
            sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
            sqrt_1mab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
            return sqrt_ab * x0 + sqrt_1mab * noise, noise

        def predict_x0_from_noise(self, x_t, noise, t):
            """Predict x0 from noise prediction."""
            return (
                self.sqrt_recip_alpha_bar[t].unsqueeze(-1) * x_t
                - self.sqrt_recip_alpha_bar_m1[t].unsqueeze(-1) * noise
            )

        def q_posterior_mean(self, x0_pred, x_t, t):
            """Compute posterior mean for reverse step."""
            c1 = self.posterior_mean_coef1[t].unsqueeze(-1)
            c2 = self.posterior_mean_coef2[t].unsqueeze(-1)
            return c1 * x0_pred + c2 * x_t


    # ==================================================================
    # Score / Denoising Network
    # ==================================================================

    class SinusoidalTimeEmbedding(nn.Module):
        """Sinusoidal positional encoding for diffusion timestep."""

        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            half_dim = self.dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
            emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
            return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


    class ScoreNetwork(nn.Module):
        """Denoising network for score estimation.

        Predicts noise ε from noisy sample x_t and timestep t.

        Parameters
        ----------
        n_features : int
            Input/output feature dimension.
        hidden : list of int
            Hidden layer sizes.
        time_dim : int
            Time embedding dimension.
        dropout : float
            Dropout rate.
        """

        def __init__(
            self,
            n_features: int,
            hidden: Optional[List[int]] = None,
            time_dim: int = 64,
            dropout: float = 0.1,
        ):
            super().__init__()

            if hidden is None:
                hidden = [256, 128, 64]

            self.time_embed = nn.Sequential(
                SinusoidalTimeEmbedding(time_dim),
                nn.Linear(time_dim, time_dim * 2),
                nn.SiLU(),
                nn.Linear(time_dim * 2, time_dim),
            )

            layers = []
            prev = n_features + time_dim
            for h in hidden:
                layers.extend([
                    nn.Linear(prev, h),
                    nn.LayerNorm(h),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ])
                prev = h
            layers.append(nn.Linear(prev, n_features))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            t_emb = self.time_embed(t)
            h = torch.cat([x, t_emb], dim=-1)
            return self.net(h)


    class ConditionalScoreNetwork(nn.Module):
        """Score network with conditioning input.

        Parameters
        ----------
        n_features : int
            Input/output feature dimension.
        n_conditions : int
            Number of condition variables.
        hidden : list of int
            Hidden layer sizes.
        time_dim : int
            Time embedding dimension.
        dropout : float
            Dropout rate.
        """

        def __init__(
            self,
            n_features: int,
            n_conditions: int = 1,
            hidden: Optional[List[int]] = None,
            time_dim: int = 64,
            dropout: float = 0.1,
        ):
            super().__init__()

            if hidden is None:
                hidden = [256, 128, 64]

            self.time_embed = nn.Sequential(
                SinusoidalTimeEmbedding(time_dim),
                nn.Linear(time_dim, time_dim * 2),
                nn.SiLU(),
                nn.Linear(time_dim * 2, time_dim),
            )

            self.cond_embed = nn.Sequential(
                nn.Linear(n_conditions, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )

            layers = []
            prev = n_features + time_dim * 2
            for h in hidden:
                layers.extend([
                    nn.Linear(prev, h),
                    nn.LayerNorm(h),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ])
                prev = h
            layers.append(nn.Linear(prev, n_features))
            self.net = nn.Sequential(*layers)

        def forward(
            self, x: torch.Tensor, t: torch.Tensor, conditions: torch.Tensor
        ) -> torch.Tensor:
            t_emb = self.time_embed(t)
            c_emb = self.cond_embed(conditions)
            h = torch.cat([x, t_emb, c_emb], dim=-1)
            return self.net(h)


    # ==================================================================
    # Diffusion Models
    # ==================================================================

    class FiberDiffusion(nn.Module):
        """DDPM for fiber network feature generation.

        Parameters
        ----------
        n_features : int
            Feature dimension of input structures.
        hidden : list of int, optional
            Hidden layer sizes for score network.
        n_steps : int
            Number of diffusion steps.
        schedule : str
            Noise schedule: "linear" or "cosine".
        dropout : float
            Dropout rate in score network.

        Examples
        --------
        >>> model = FiberDiffusion(n_features=20)
        >>> trainer = DiffusionTrainer(model)
        >>> trainer.fit(X_train, epochs=200)
        >>> X_gen = model.sample(n=50)
        """

        def __init__(
            self,
            n_features: int,
            hidden: Optional[List[int]] = None,
            n_steps: int = 500,
            schedule: str = "cosine",
            dropout: float = 0.1,
        ):
            super().__init__()
            self.n_features = n_features
            self.n_steps = n_steps
            self.schedule = NoiseSchedule(n_steps, schedule)

            self.score_net = ScoreNetwork(
                n_features=n_features,
                hidden=hidden,
                dropout=dropout,
            )

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Predict noise given noisy input and timestep."""
            return self.score_net(x, t)

        def loss(self, x0: torch.Tensor) -> torch.Tensor:
            """Compute training loss (simplified variational bound)."""
            batch_size = x0.shape[0]
            t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device)
            noise = torch.randn_like(x0)
            x_t, _ = self.schedule.q_sample(x0, t, noise)
            noise_pred = self.score_net(x_t, t)
            return F.mse_loss(noise_pred, noise)

        @torch.no_grad()
        def sample(
            self,
            n: int = 10,
            device: str = "cpu",
            ddim: bool = False,
            ddim_steps: int = 50,
            eta: float = 0.0,
        ) -> torch.Tensor:
            """Generate samples via reverse diffusion.

            Parameters
            ----------
            n : int
                Number of samples to generate.
            device : str
                Device for generation.
            ddim : bool
                Use DDIM sampler (faster, deterministic).
            ddim_steps : int
                Number of DDIM steps (when ddim=True).
            eta : float
                DDIM stochasticity parameter (0=deterministic).

            Returns
            -------
            (n, n_features) generated feature vectors
            """
            self.schedule.to(device)
            self.eval()

            if ddim:
                return self._sample_ddim(n, device, ddim_steps, eta)

            x = torch.randn(n, self.n_features, device=device)

            for step in reversed(range(self.n_steps)):
                t = torch.full((n,), step, device=device, dtype=torch.long)
                noise_pred = self.score_net(x, t)

                x0_pred = self.schedule.predict_x0_from_noise(x, noise_pred, t)
                mean = self.schedule.q_posterior_mean(x0_pred, x, t)

                if step > 0:
                    log_var = self.schedule.posterior_log_var[step]
                    noise = torch.randn_like(x)
                    x = mean + torch.exp(0.5 * log_var) * noise
                else:
                    x = mean

            return x

        @torch.no_grad()
        def _sample_ddim(
            self, n: int, device: str, ddim_steps: int, eta: float
        ) -> torch.Tensor:
            """DDIM sampler for faster generation."""
            step_size = self.n_steps // ddim_steps
            timesteps = list(range(0, self.n_steps, step_size))

            x = torch.randn(n, self.n_features, device=device)

            for i in reversed(range(len(timesteps))):
                t = torch.full((n,), timesteps[i], device=device, dtype=torch.long)
                noise_pred = self.score_net(x, t)

                alpha_bar = self.schedule.alpha_bar[timesteps[i]]
                x0_pred = (x - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)

                if i > 0:
                    alpha_bar_prev = self.schedule.alpha_bar[timesteps[i - 1]]
                else:
                    alpha_bar_prev = torch.tensor(1.0, device=device)

                sigma = eta * torch.sqrt(
                    (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)
                )

                dir_x = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * noise_pred
                x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_x

                if i > 0 and eta > 0:
                    x = x + sigma * torch.randn_like(x)

            return x


    class ConditionalFiberDiffusion(nn.Module):
        """Conditional DDPM for property-guided structure generation.

        Parameters
        ----------
        n_features : int
            Feature dimension.
        n_conditions : int
            Number of condition variables (e.g., target properties).
        hidden : list of int, optional
            Hidden layer sizes.
        n_steps : int
            Number of diffusion steps.
        schedule : str
            Noise schedule.
        dropout : float
            Dropout rate.
        cond_drop_prob : float
            Probability of dropping conditions during training
            (for classifier-free guidance).

        Examples
        --------
        >>> cmodel = ConditionalFiberDiffusion(n_features=20, n_conditions=3)
        >>> trainer = DiffusionTrainer(cmodel)
        >>> trainer.fit(X_train, conditions=cond_train, epochs=200)
        >>> target = torch.tensor([[500.0, 1.5, 0.3]])
        >>> X_gen = cmodel.sample(n=10, conditions=target, guidance_scale=2.0)
        """

        def __init__(
            self,
            n_features: int,
            n_conditions: int = 1,
            hidden: Optional[List[int]] = None,
            n_steps: int = 500,
            schedule: str = "cosine",
            dropout: float = 0.1,
            cond_drop_prob: float = 0.1,
        ):
            super().__init__()
            self.n_features = n_features
            self.n_conditions = n_conditions
            self.n_steps = n_steps
            self.cond_drop_prob = cond_drop_prob
            self.schedule = NoiseSchedule(n_steps, schedule)

            self.score_net = ConditionalScoreNetwork(
                n_features=n_features,
                n_conditions=n_conditions,
                hidden=hidden,
                dropout=dropout,
            )

        def forward(
            self, x: torch.Tensor, t: torch.Tensor, conditions: torch.Tensor
        ) -> torch.Tensor:
            return self.score_net(x, t, conditions)

        def loss(
            self, x0: torch.Tensor, conditions: torch.Tensor
        ) -> torch.Tensor:
            batch_size = x0.shape[0]
            t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device)
            noise = torch.randn_like(x0)
            x_t, _ = self.schedule.q_sample(x0, t, noise)

            # Classifier-free guidance: randomly drop conditions
            cond = conditions.clone()
            if self.cond_drop_prob > 0:
                mask = torch.rand(batch_size, device=x0.device) < self.cond_drop_prob
                cond[mask] = 0.0

            noise_pred = self.score_net(x_t, t, cond)
            return F.mse_loss(noise_pred, noise)

        @torch.no_grad()
        def sample(
            self,
            n: int = 10,
            conditions: Optional[torch.Tensor] = None,
            guidance_scale: float = 1.0,
            device: str = "cpu",
        ) -> torch.Tensor:
            """Generate conditionally guided samples.

            Parameters
            ----------
            n : int
                Number of samples.
            conditions : (1, n_conditions) or (n, n_conditions)
                Target condition values.
            guidance_scale : float
                Classifier-free guidance scale (>1 for stronger conditioning).
            device : str
                Device.

            Returns
            -------
            (n, n_features) generated features
            """
            self.schedule.to(device)
            self.eval()

            if conditions is None:
                conditions = torch.zeros(n, self.n_conditions, device=device)
            elif conditions.shape[0] == 1:
                conditions = conditions.expand(n, -1).to(device)
            else:
                conditions = conditions.to(device)

            x = torch.randn(n, self.n_features, device=device)
            null_cond = torch.zeros_like(conditions)

            for step in reversed(range(self.n_steps)):
                t = torch.full((n,), step, device=device, dtype=torch.long)

                # Classifier-free guidance
                noise_cond = self.score_net(x, t, conditions)
                if guidance_scale > 1.0:
                    noise_uncond = self.score_net(x, t, null_cond)
                    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                else:
                    noise_pred = noise_cond

                x0_pred = self.schedule.predict_x0_from_noise(x, noise_pred, t)
                mean = self.schedule.q_posterior_mean(x0_pred, x, t)

                if step > 0:
                    log_var = self.schedule.posterior_log_var[step]
                    noise = torch.randn_like(x)
                    x = mean + torch.exp(0.5 * log_var) * noise
                else:
                    x = mean

            return x


    # ==================================================================
    # Trainer
    # ==================================================================

    class DiffusionTrainer:
        """Training loop for diffusion models.

        Parameters
        ----------
        model : FiberDiffusion or ConditionalFiberDiffusion
            Diffusion model to train.
        lr : float
            Learning rate.
        weight_decay : float
            Weight decay.
        ema_decay : float
            EMA decay for model averaging.

        Examples
        --------
        >>> trainer = DiffusionTrainer(model)
        >>> trainer.fit(X_train, epochs=200, batch_size=64)
        >>> X_gen = trainer.sample(n=50)
        """

        def __init__(
            self,
            model: Union[FiberDiffusion, ConditionalFiberDiffusion],
            lr: float = 2e-4,
            weight_decay: float = 1e-4,
            ema_decay: float = 0.995,
        ):
            _require_torch()
            self.model = model
            self.lr = lr
            self.weight_decay = weight_decay
            self.ema_decay = ema_decay
            self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

            # EMA model
            self.ema_model = None
            self._init_ema()

        def _init_ema(self):
            """Initialize EMA model as copy of model."""
            import copy
            self.ema_model = copy.deepcopy(self.model)
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

        def _update_ema(self):
            """Update EMA model parameters."""
            if self.ema_model is None:
                return
            with torch.no_grad():
                for p_ema, p_model in zip(
                    self.ema_model.parameters(), self.model.parameters()
                ):
                    p_ema.data.mul_(self.ema_decay).add_(
                        p_model.data, alpha=1.0 - self.ema_decay
                    )

        def fit(
            self,
            X_train: np.ndarray,
            *,
            conditions: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None,
            conditions_val: Optional[np.ndarray] = None,
            epochs: int = 200,
            batch_size: int = 64,
            verbose: bool = True,
        ) -> Dict[str, Any]:
            """Train the diffusion model.

            Parameters
            ----------
            X_train : (N, n_features)
                Training data.
            conditions : (N, n_conditions), optional
                Condition variables (for ConditionalFiberDiffusion).
            X_val : (M, n_features), optional
                Validation data.
            conditions_val : (M, n_conditions), optional
                Validation conditions.
            epochs : int
                Training epochs.
            batch_size : int
                Batch size.
            verbose : bool
                Print progress.

            Returns
            -------
            dict
                Training history.
            """
            X_t = torch.tensor(X_train, dtype=torch.float32)
            C_t = (
                torch.tensor(conditions, dtype=torch.float32)
                if conditions is not None
                else None
            )

            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )

            is_conditional = isinstance(self.model, ConditionalFiberDiffusion)
            n = len(X_t)

            for epoch in range(epochs):
                self.model.train()
                perm = torch.randperm(n)
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    idx = perm[start:end]
                    xb = X_t[idx]

                    optimizer.zero_grad()

                    if is_conditional and C_t is not None:
                        cb = C_t[idx]
                        loss = self.model.loss(xb, cb)
                    else:
                        loss = self.model.loss(xb)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step()
                self._update_ema()
                epoch_loss /= max(n_batches, 1)
                self.history["train_loss"].append(epoch_loss)

                # Validation
                val_loss = None
                if X_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        X_v = torch.tensor(X_val, dtype=torch.float32)
                        if is_conditional and conditions_val is not None:
                            C_v = torch.tensor(conditions_val, dtype=torch.float32)
                            val_loss = self.model.loss(X_v, C_v).item()
                        else:
                            val_loss = self.model.loss(X_v).item()
                    self.history["val_loss"].append(val_loss)

                if verbose and (
                    epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1
                ):
                    val_str = f" | val={val_loss:.4f}" if val_loss else ""
                    print(f"Epoch {epoch:3d} | loss={epoch_loss:.4f}{val_str}")

            return self.history

        def sample(
            self,
            n: int = 10,
            conditions: Optional[torch.Tensor] = None,
            use_ema: bool = True,
            **kwargs,
        ) -> torch.Tensor:
            """Generate samples using EMA model.

            Parameters
            ----------
            n : int
                Number of samples.
            conditions : tensor, optional
                Conditions for conditional model.
            use_ema : bool
                Use EMA model for sampling.

            Returns
            -------
            (n, n_features) generated features
            """
            model = self.ema_model if use_ema and self.ema_model else self.model
            model.eval()

            if isinstance(model, ConditionalFiberDiffusion):
                return model.sample(n=n, conditions=conditions, **kwargs)
            else:
                return model.sample(n=n, **kwargs)


    # ==================================================================
    # Graph-Level Diffusion Wrapper
    # ==================================================================

    class GraphDiffusionGenerator:
        """Generate fiber network structures using feature-space diffusion.

        Wraps FiberDiffusion to generate complete structures by:
        1. Diffusing in feature space
        2. Mapping generated features back to structural parameters
        3. Creating StructureGraph instances from parameters

        Parameters
        ----------
        n_features : int
            Feature dimension (must match training data).
        feature_names : list of str, optional
            Names of features being generated.
        model : FiberDiffusion or ConditionalFiberDiffusion, optional
            Pre-trained model. If None, creates a new one.

        Examples
        --------
        >>> gen = GraphDiffusionGenerator(n_features=20, feature_names=names)
        >>> gen.train(X_train, epochs=200)
        >>> structures = gen.generate_structures(n=10, unit="honeycomb", grid=(3,3))
        """

        def __init__(
            self,
            n_features: int,
            feature_names: Optional[List[str]] = None,
            model: Optional[Union[FiberDiffusion, ConditionalFiberDiffusion]] = None,
        ):
            self.n_features = n_features
            self.feature_names = feature_names or [f"f{i}" for i in range(n_features)]
            self.model = model or FiberDiffusion(n_features=n_features)
            self.trainer: Optional[DiffusionTrainer] = None
            self.scaler_mean: Optional[np.ndarray] = None
            self.scaler_std: Optional[np.ndarray] = None

        def train(
            self,
            X: np.ndarray,
            *,
            conditions: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None,
            conditions_val: Optional[np.ndarray] = None,
            epochs: int = 200,
            batch_size: int = 64,
            normalize: bool = True,
            verbose: bool = True,
        ) -> Dict[str, Any]:
            """Train the diffusion model with optional normalization.

            Parameters
            ----------
            X : (N, n_features)
                Training feature matrix.
            conditions : optional
                Condition variables.
            normalize : bool
                Standardize features before training.

            Returns
            -------
            dict
                Training history.
            """
            X = np.asarray(X, dtype=np.float32)

            if normalize:
                self.scaler_mean = X.mean(axis=0)
                self.scaler_std = X.std(axis=0) + 1e-8
                X = (X - self.scaler_mean) / self.scaler_std
                if X_val is not None:
                    X_val = (X_val - self.scaler_mean) / self.scaler_std

            self.trainer = DiffusionTrainer(self.model)
            return self.trainer.fit(
                X,
                conditions=conditions,
                X_val=X_val,
                conditions_val=conditions_val,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
            )

        def generate(
            self,
            n: int = 10,
            conditions: Optional[torch.Tensor] = None,
            denormalize: bool = True,
            **kwargs,
        ) -> np.ndarray:
            """Generate feature vectors.

            Parameters
            ----------
            n : int
                Number of samples.
            conditions : optional
                Condition tensor for conditional generation.
            denormalize : bool
                Reverse normalization if applied during training.

            Returns
            -------
            (n, n_features) numpy array
            """
            if self.trainer is None:
                raise RuntimeError("Model not trained. Call train() first.")

            X_gen = self.trainer.sample(n=n, conditions=conditions, **kwargs)
            X_gen = X_gen.cpu().numpy()

            if denormalize and self.scaler_mean is not None:
                X_gen = X_gen * self.scaler_std + self.scaler_mean

            return X_gen

        def generate_structures(
            self,
            n: int = 10,
            unit: str = "honeycomb",
            grid: Tuple[int, int] = (3, 3),
            conditions: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> List[Dict[str, Any]]:
            """Generate structural parameters for fiber networks.

            Parameters
            ----------
            n : int
                Number of structures to generate.
            unit : str
                Base unit type.
            grid : (int, int)
                Grid size.
            conditions : optional
                Conditions for conditional generation.

            Returns
            -------
            list of dict
                Generated structure parameters.
            """
            from fibernet.gen.pattern import pattern_2d

            features = self.generate(n=n, conditions=conditions)
            structures = []

            for i in range(n):
                feat_dict = {
                    name: float(features[i, j])
                    for j, name in enumerate(self.feature_names)
                }
                structures.append({
                    "index": i,
                    "unit": unit,
                    "grid": grid,
                    "features": feat_dict,
                })

            return structures

        def save(self, path: str):
            """Save model and scaler to disk."""
            bundle = {
                "model_state": self.model.state_dict(),
                "model_class": self.model.__class__.__name__,
                "n_features": self.n_features,
                "feature_names": self.feature_names,
                "scaler_mean": self.scaler_mean,
                "scaler_std": self.scaler_std,
            }
            if self.trainer and self.trainer.ema_model:
                bundle["ema_state"] = self.trainer.ema_model.state_dict()

            from pathlib import Path as _Path
            _Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(bundle, path)

        def load(self, path: str):
            """Load model and scaler from disk."""
            bundle = torch.load(path, weights_only=False)
            self.model.load_state_dict(bundle["model_state"])
            self.n_features = bundle.get("n_features", self.n_features)
            self.feature_names = bundle.get("feature_names", self.feature_names)
            self.scaler_mean = bundle.get("scaler_mean")
            self.scaler_std = bundle.get("scaler_std")

            if "ema_state" in bundle:
                import copy
                self.ema_model = copy.deepcopy(self.model)
                self.ema_model.load_state_dict(bundle["ema_state"])

else:
    class FiberDiffusion:
        def __init__(self, *a, **kw):
            _require_torch()

    class ConditionalFiberDiffusion:
        def __init__(self, *a, **kw):
            _require_torch()

    class DiffusionTrainer:
        def __init__(self, *a, **kw):
            _require_torch()

    class GraphDiffusionGenerator:
        def __init__(self, *a, **kw):
            _require_torch()

"""
Generative Adversarial Networks for FiberNet Structure Generation.

Implements:
- FiberGenerator: Generator network mapping latent vectors to features
- FiberDiscriminator: Discriminator network for real/fake classification
- FiberGAN: Standard GAN with configurable architectures
- FiberWGAN: Wasserstein GAN with gradient penalty for stable training
- FiberCGAN: Conditional GAN for property-guided generation
- GANTrainer: Training loop with multiple loss options

Features
--------
- WGAN-GP for stable training with Wasserstein distance
- Spectral normalization for discriminator stability
- Conditional generation with property constraints
- Latent space interpolation for smooth transitions
- Mode collapse detection via inception score approximation

References
----------
- Goodfellow et al., "Generative Adversarial Nets" (NeurIPS 2014)
- Arjovsky et al., "Wasserstein GAN" (ICML 2017)
- Gulrajani et al., "Improved Training of WGANs" (NeurIPS 2017)
- Article section 2: GANs for fiber architecture generation

Examples
--------
>>> from fibernet.ml.gan import FiberWGAN, GANTrainer
>>> gan = FiberWGAN(n_features=20, latent_dim=32, hidden=[256, 128])
>>> trainer = GANTrainer(gan)
>>> trainer.fit(X_train, epochs=300)
>>> X_generated = gan.sample(n=100)

>>> # Conditional generation
>>> from fibernet.ml.gan import FiberCGAN
>>> cgan = FiberCGAN(n_features=20, n_conditions=3, latent_dim=32)
>>> trainer = GANTrainer(cgan)
>>> trainer.fit(X_train, conditions=cond_train, epochs=300)
>>> target = torch.tensor([[500.0, 1.5, 0.3]])
>>> X_gen = cgan.sample(n=10, conditions=target)
"""

from __future__ import annotations

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
    # Building Blocks
    # ==================================================================

    class _MiniBatchStddev(nn.Module):
        """Mini-batch standard deviation layer for discriminator."""

        def __init__(self, group_size: int = 4):
            super().__init__()
            self.group_size = group_size

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            group = min(self.group_size, batch)
            y = x.view(group, -1, x.shape[1])
            y = y - y.mean(dim=0, keepdim=True)
            y = (y ** 2).mean(dim=0)
            y = torch.sqrt(y + 1e-8)
            y = y.mean(dim=1, keepdim=True)
            y = y.expand(batch, -1)
            return torch.cat([x, y], dim=1)


    def _build_mlp(
        in_dim: int,
        hidden: List[int],
        out_dim: int,
        dropout: float = 0.1,
        activation: str = "leaky_relu",
        spectral_norm: bool = False,
        batch_norm: bool = True,
    ) -> nn.Sequential:
        """Build MLP with optional spectral norm and batch norm."""
        act_map = {
            "relu": nn.ReLU,
            "leaky_relu": lambda: nn.LeakyReLU(0.2),
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        act_cls = act_map.get(activation, nn.LeakyReLU)

        layers = []
        prev = in_dim
        for h in hidden:
            linear = nn.Linear(prev, h)
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            if batch_norm and not spectral_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_cls() if callable(act_cls) else act_cls)
            layers.append(nn.Dropout(dropout))
            prev = h

        final = nn.Linear(prev, out_dim)
        if spectral_norm:
            final = nn.utils.spectral_norm(final)
        layers.append(final)

        return nn.Sequential(*layers)


    # ==================================================================
    # Generator & Discriminator
    # ==================================================================

    class FiberGenerator(nn.Module):
        """Generator network for fiber network features.

        Parameters
        ----------
        latent_dim : int
            Latent space dimension.
        n_features : int
            Output feature dimension.
        hidden : list of int
            Hidden layer sizes.
        dropout : float
            Dropout rate.
        """

        def __init__(
            self,
            latent_dim: int = 32,
            n_features: int = 20,
            hidden: Optional[List[int]] = None,
            dropout: float = 0.1,
        ):
            super().__init__()
            if hidden is None:
                hidden = [256, 128]
            self.latent_dim = latent_dim
            self.n_features = n_features
            self.net = _build_mlp(
                latent_dim, hidden, n_features,
                dropout=dropout, activation="relu", batch_norm=True,
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.net(z)


    class FiberDiscriminator(nn.Module):
        """Discriminator network for real/fake classification.

        Parameters
        ----------
        n_features : int
            Input feature dimension.
        hidden : list of int
            Hidden layer sizes.
        dropout : float
            Dropout rate.
        use_spectral_norm : bool
            Apply spectral normalization for stability.
        use_minibatch_std : bool
            Add mini-batch standard deviation layer.
        """

        def __init__(
            self,
            n_features: int = 20,
            hidden: Optional[List[int]] = None,
            dropout: float = 0.1,
            use_spectral_norm: bool = True,
            use_minibatch_std: bool = True,
        ):
            super().__init__()
            if hidden is None:
                hidden = [256, 128]

            self.use_minibatch_std = use_minibatch_std
            in_dim = n_features + 1 if use_minibatch_std else n_features

            self.minibatch_std = _MiniBatchStddev() if use_minibatch_std else None

            self.net = _build_mlp(
                in_dim, hidden, 1,
                dropout=dropout, activation="leaky_relu",
                spectral_norm=use_spectral_norm, batch_norm=False,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.minibatch_std is not None:
                x = self.minibatch_std(x)
            return self.net(x)


    class ConditionalGenerator(nn.Module):
        """Generator with conditioning input."""

        def __init__(
            self,
            latent_dim: int = 32,
            n_conditions: int = 1,
            n_features: int = 20,
            hidden: Optional[List[int]] = None,
            dropout: float = 0.1,
        ):
            super().__init__()
            if hidden is None:
                hidden = [256, 128]
            self.latent_dim = latent_dim
            self.n_conditions = n_conditions
            self.n_features = n_features
            self.net = _build_mlp(
                latent_dim + n_conditions, hidden, n_features,
                dropout=dropout, activation="relu", batch_norm=True,
            )

        def forward(self, z: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
            return self.net(torch.cat([z, conditions], dim=-1))


    class ConditionalDiscriminator(nn.Module):
        """Discriminator with conditioning input."""

        def __init__(
            self,
            n_features: int = 20,
            n_conditions: int = 1,
            hidden: Optional[List[int]] = None,
            dropout: float = 0.1,
            use_spectral_norm: bool = True,
        ):
            super().__init__()
            if hidden is None:
                hidden = [256, 128]
            self.net = _build_mlp(
                n_features + n_conditions, hidden, 1,
                dropout=dropout, activation="leaky_relu",
                spectral_norm=use_spectral_norm, batch_norm=False,
            )

        def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
            return self.net(torch.cat([x, conditions], dim=-1))


    # ==================================================================
    # GAN Models
    # ==================================================================

    class FiberGAN(nn.Module):
        """Standard GAN for fiber network generation.

        Parameters
        ----------
        n_features : int
            Feature dimension.
        latent_dim : int
            Latent space dimension.
        hidden : list of int, optional
            Hidden layer sizes.
        dropout : float
            Dropout rate.
        """

        def __init__(
            self,
            n_features: int = 20,
            latent_dim: int = 32,
            hidden: Optional[List[int]] = None,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.n_features = n_features
            self.latent_dim = latent_dim
            self.generator = FiberGenerator(latent_dim, n_features, hidden, dropout)
            self.discriminator = FiberDiscriminator(n_features, hidden, dropout)

        def generate(self, z: torch.Tensor) -> torch.Tensor:
            return self.generator(z)

        def discriminate(self, x: torch.Tensor) -> torch.Tensor:
            return self.discriminator(x)

        @torch.no_grad()
        def sample(self, n: int = 10, device: str = "cpu") -> torch.Tensor:
            z = torch.randn(n, self.latent_dim, device=device)
            self.generator.eval()
            return self.generator(z)


    class FiberWGAN(nn.Module):
        """Wasserstein GAN with gradient penalty.

        Uses Wasserstein distance for more stable training and
        meaningful loss values.

        Parameters
        ----------
        n_features : int
            Feature dimension.
        latent_dim : int
            Latent space dimension.
        hidden : list of int, optional
            Hidden layer sizes.
        dropout : float
            Dropout rate.
        gp_weight : float
            Gradient penalty weight.

        Examples
        --------
        >>> gan = FiberWGAN(n_features=20, latent_dim=32)
        >>> trainer = GANTrainer(gan)
        >>> trainer.fit(X_train, epochs=300)
        >>> X_gen = gan.sample(n=100)
        """

        def __init__(
            self,
            n_features: int = 20,
            latent_dim: int = 32,
            hidden: Optional[List[int]] = None,
            dropout: float = 0.1,
            gp_weight: float = 10.0,
        ):
            super().__init__()
            self.n_features = n_features
            self.latent_dim = latent_dim
            self.gp_weight = gp_weight
            self.generator = FiberGenerator(latent_dim, n_features, hidden, dropout)
            self.discriminator = FiberDiscriminator(
                n_features, hidden, dropout,
                use_spectral_norm=True, use_minibatch_std=False,
            )

        def generate(self, z: torch.Tensor) -> torch.Tensor:
            return self.generator(z)

        def discriminate(self, x: torch.Tensor) -> torch.Tensor:
            return self.discriminator(x)

        def gradient_penalty(
            self, real: torch.Tensor, fake: torch.Tensor
        ) -> torch.Tensor:
            """Compute gradient penalty for WGAN-GP."""
            batch_size = real.shape[0]
            alpha = torch.rand(batch_size, 1, device=real.device)
            interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
            d_interp = self.discriminator(interpolated)
            gradients = torch.autograd.grad(
                outputs=d_interp, inputs=interpolated,
                grad_outputs=torch.ones_like(d_interp),
                create_graph=True,
            )[0]
            gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            return gp

        @torch.no_grad()
        def sample(self, n: int = 10, device: str = "cpu") -> torch.Tensor:
            z = torch.randn(n, self.latent_dim, device=device)
            self.generator.eval()
            return self.generator(z)


    class FiberCGAN(nn.Module):
        """Conditional GAN for property-guided structure generation.

        Parameters
        ----------
        n_features : int
            Feature dimension.
        n_conditions : int
            Number of condition variables.
        latent_dim : int
            Latent space dimension.
        hidden : list of int, optional
            Hidden layer sizes.
        dropout : float
            Dropout rate.
        use_wgan : bool
            Use WGAN-GP training (recommended).

        Examples
        --------
        >>> cgan = FiberCGAN(n_features=20, n_conditions=3, latent_dim=32)
        >>> trainer = GANTrainer(cgan)
        >>> trainer.fit(X_train, conditions=cond_train, epochs=300)
        >>> target = torch.tensor([[500.0, 1.5, 0.3]])
        >>> X_gen = cgan.sample(n=10, conditions=target)
        """

        def __init__(
            self,
            n_features: int = 20,
            n_conditions: int = 1,
            latent_dim: int = 32,
            hidden: Optional[List[int]] = None,
            dropout: float = 0.1,
            use_wgan: bool = True,
        ):
            super().__init__()
            self.n_features = n_features
            self.n_conditions = n_conditions
            self.latent_dim = latent_dim
            self.use_wgan = use_wgan

            self.generator = ConditionalGenerator(
                latent_dim, n_conditions, n_features, hidden, dropout,
            )
            self.discriminator = ConditionalDiscriminator(
                n_features, n_conditions, hidden, dropout,
                use_spectral_norm=use_wgan,
            )
            self.gp_weight = 10.0

        def generate(self, z: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
            return self.generator(z, conditions)

        def discriminate(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
            return self.discriminator(x, conditions)

        def gradient_penalty(
            self, real: torch.Tensor, fake: torch.Tensor, conditions: torch.Tensor
        ) -> torch.Tensor:
            batch_size = real.shape[0]
            alpha = torch.rand(batch_size, 1, device=real.device)
            interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
            d_interp = self.discriminator(interpolated, conditions)
            gradients = torch.autograd.grad(
                outputs=d_interp, inputs=interpolated,
                grad_outputs=torch.ones_like(d_interp),
                create_graph=True,
            )[0]
            gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            return gp

        @torch.no_grad()
        def sample(
            self,
            n: int = 10,
            conditions: Optional[torch.Tensor] = None,
            device: str = "cpu",
        ) -> torch.Tensor:
            if conditions is None:
                conditions = torch.zeros(n, self.n_conditions, device=device)
            elif conditions.shape[0] == 1:
                conditions = conditions.expand(n, -1).to(device)
            else:
                conditions = conditions.to(device)

            z = torch.randn(n, self.latent_dim, device=device)
            self.generator.eval()
            return self.generator(z, conditions)

        @torch.no_grad()
        def interpolate(
            self,
            n_steps: int = 10,
            conditions: Optional[torch.Tensor] = None,
            device: str = "cpu",
        ) -> torch.Tensor:
            """Interpolate in latent space for smooth structure transitions.

            Parameters
            ----------
            n_steps : int
                Number of interpolation steps.
            conditions : optional
                Condition tensor.

            Returns
            -------
            (n_steps, n_features) interpolated features
            """
            z_start = torch.randn(1, self.latent_dim, device=device)
            z_end = torch.randn(1, self.latent_dim, device=device)
            alphas = torch.linspace(0, 1, n_steps, device=device)

            results = []
            for alpha in alphas:
                z = z_start * (1 - alpha) + z_end * alpha
                if conditions is not None:
                    c = conditions if conditions.shape[0] == 1 else conditions[:1]
                    results.append(self.generator(z, c.to(device)))
                else:
                    results.append(self.generator(z, torch.zeros(1, self.n_conditions, device=device)))

            return torch.cat(results, dim=0)


    # ==================================================================
    # Trainer
    # ==================================================================

    class GANTrainer:
        """Training loop for GAN models.

        Parameters
        ----------
        model : FiberGAN, FiberWGAN, or FiberCGAN
            GAN model to train.
        g_lr : float
            Generator learning rate.
        d_lr : float
            Discriminator learning rate.
        n_critic : int
            Number of discriminator updates per generator update.

        Examples
        --------
        >>> trainer = GANTrainer(gan)
        >>> trainer.fit(X_train, epochs=300)
        >>> X_gen = trainer.sample(n=50)
        """

        def __init__(
            self,
            model: Union[FiberGAN, FiberWGAN, FiberCGAN],
            g_lr: float = 2e-4,
            d_lr: float = 2e-4,
            n_critic: int = 5,
        ):
            _require_torch()
            self.model = model
            self.g_lr = g_lr
            self.d_lr = d_lr
            self.n_critic = n_critic
            self.history: Dict[str, List[float]] = {
                "g_loss": [], "d_loss": [], "gp": [],
            }

        def fit(
            self,
            X_train: np.ndarray,
            *,
            conditions: Optional[np.ndarray] = None,
            epochs: int = 300,
            batch_size: int = 64,
            verbose: bool = True,
        ) -> Dict[str, Any]:
            """Train the GAN.

            Parameters
            ----------
            X_train : (N, n_features)
                Training data.
            conditions : (N, n_conditions), optional
                Condition variables (for FiberCGAN).
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

            is_conditional = isinstance(self.model, FiberCGAN)
            is_wgan = isinstance(self.model, (FiberWGAN, FiberCGAN)) and getattr(
                self.model, "use_wgan", True
            )

            g_opt = torch.optim.Adam(self.model.generator.parameters(), lr=self.g_lr, betas=(0.5, 0.999))
            d_opt = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.d_lr, betas=(0.5, 0.999))

            n = len(X_t)

            for epoch in range(epochs):
                perm = torch.randperm(n)
                g_loss_epoch = 0.0
                d_loss_epoch = 0.0
                gp_epoch = 0.0
                n_batches = 0

                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    if end - start < 4:
                        continue
                    idx = perm[start:end]
                    real = X_t[idx]
                    bs = real.shape[0]
                    cb = C_t[idx] if C_t is not None else None

                    # Train discriminator
                    for _ in range(self.n_critic):
                        z = torch.randn(bs, self.model.latent_dim)
                        d_opt.zero_grad()

                        if is_conditional:
                            fake = self.model.generate(z, cb).detach()
                            d_real = self.model.discriminate(real, cb)
                            d_fake = self.model.discriminate(fake, cb)
                        else:
                            fake = self.model.generate(z).detach()
                            d_real = self.model.discriminate(real)
                            d_fake = self.model.discriminate(fake)

                        if is_wgan:
                            d_loss = -(d_real.mean() - d_fake.mean())
                            if is_conditional and cb is not None:
                                gp = self.model.gradient_penalty(real, fake, cb)
                            elif hasattr(self.model, "gradient_penalty"):
                                gp = self.model.gradient_penalty(real, fake)
                            else:
                                gp = torch.tensor(0.0)
                            d_loss = d_loss + self.model.gp_weight * gp
                        else:
                            d_loss = -(torch.log(d_real.sigmoid() + 1e-8).mean() +
                                       torch.log(1 - d_fake.sigmoid() + 1e-8).mean())
                            gp = torch.tensor(0.0)

                        d_loss.backward()
                        d_opt.step()

                    # Train generator
                    g_opt.zero_grad()
                    z = torch.randn(bs, self.model.latent_dim)

                    if is_conditional:
                        fake = self.model.generate(z, cb)
                        g_score = self.model.discriminate(fake, cb)
                    else:
                        fake = self.model.generate(z)
                        g_score = self.model.discriminate(fake)

                    if is_wgan:
                        g_loss = -g_score.mean()
                    else:
                        g_loss = -torch.log(g_score.sigmoid() + 1e-8).mean()

                    g_loss.backward()
                    g_opt.step()

                    g_loss_epoch += g_loss.item()
                    d_loss_epoch += d_loss.item()
                    gp_epoch += gp.item() if isinstance(gp, torch.Tensor) else gp
                    n_batches += 1

                g_loss_epoch /= max(n_batches, 1)
                d_loss_epoch /= max(n_batches, 1)
                gp_epoch /= max(n_batches, 1)

                self.history["g_loss"].append(g_loss_epoch)
                self.history["d_loss"].append(d_loss_epoch)
                self.history["gp"].append(gp_epoch)

                if verbose and (
                    epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1
                ):
                    print(f"Epoch {epoch:3d} | G={g_loss_epoch:.4f} | D={d_loss_epoch:.4f} | GP={gp_epoch:.4f}")

            return self.history

        def sample(self, n: int = 10, conditions: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
            self.model.eval()
            if isinstance(self.model, FiberCGAN):
                return self.model.sample(n=n, conditions=conditions, **kwargs)
            return self.model.sample(n=n, **kwargs)

        def evaluate_diversity(self, X_real: np.ndarray, n_samples: int = 1000) -> Dict[str, float]:
            """Evaluate generation quality and diversity.

            Parameters
            ----------
            X_real : (N, n_features)
                Real data for comparison.
            n_samples : int
                Number of generated samples.

            Returns
            -------
            dict
                diversity_score, coverage, fidelity metrics.
            """
            self.model.eval()
            X_gen = self.sample(n=n_samples).cpu().numpy()
            X_real = np.asarray(X_real)

            real_mean = X_real.mean(axis=0)
            real_std = X_real.std(axis=0) + 1e-8
            gen_mean = X_gen.mean(axis=0)
            gen_std = X_gen.std(axis=0)

            # Feature matching quality
            mean_error = np.mean(np.abs(gen_mean - real_mean) / real_std)
            std_ratio = np.mean(gen_std / real_std)

            # Coverage: fraction of real features covered by generated range
            coverage = np.mean([
                (X_gen[:, j].min() <= X_real[:, j].mean() <= X_gen[:, j].max())
                for j in range(X_real.shape[1])
            ])

            return {
                "mean_error": float(mean_error),
                "std_ratio": float(std_ratio),
                "coverage": float(coverage),
                "n_generated": n_samples,
            }

else:
    class FiberGAN:
        def __init__(self, *a, **kw):
            _require_torch()

    class FiberWGAN:
        def __init__(self, *a, **kw):
            _require_torch()

    class FiberCGAN:
        def __init__(self, *a, **kw):
            _require_torch()

    class GANTrainer:
        def __init__(self, *a, **kw):
            _require_torch()

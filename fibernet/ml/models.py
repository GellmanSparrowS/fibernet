"""
PyTorch Model Zoo for FiberNet ML Pipeline.

Provides ready-to-use neural networks for structure-property prediction:
- FiberMLP: Standard MLP with dropout and batch norm
- FiberResNet: Residual network for tabular data
- FiberAttentionNet: Multi-head attention for feature importance
- FiberUncertaintyNet: Aleatoric + epistemic uncertainty estimation
- FiberMultiTaskNet: Shared backbone with multiple property heads
- FiberEnsemble: Model ensemble for robust predictions

All models accept (batch, n_features) input and produce (batch, n_outputs).

Examples
--------
>>> from fibernet.ml.models import FiberMLP, FiberResNet, FiberEnsemble
>>> model = FiberMLP(n_features=20, n_outputs=1, hidden=[128, 64])
>>> y_pred = model(torch.randn(32, 20))

>>> ensemble = FiberEnsemble(
...     models=[FiberMLP(20, 1, [128,64]), FiberResNet(20, 1, [64,64])],
... )
>>> y_pred, y_std = ensemble.predict_with_uncertainty(torch.randn(32, 20))
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


# ======================================================================
# Building Blocks
# ======================================================================

if HAS_TORCH:

    class _ResidualBlock(nn.Module):
        """Residual block with batch norm."""

        def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, out_dim)
            self.bn1 = nn.BatchNorm1d(out_dim)
            self.fc2 = nn.Linear(out_dim, out_dim)
            self.bn2 = nn.BatchNorm1d(out_dim)
            self.dropout = nn.Dropout(dropout)

            if in_dim != out_dim:
                self.shortcut = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                )
            else:
                self.shortcut = nn.Identity()

        def forward(self, x):
            residual = self.shortcut(x)
            out = F.relu(self.bn1(self.fc1(x)))
            out = self.dropout(out)
            out = self.bn2(self.fc2(out))
            return F.relu(out + residual)


    class _AttentionLayer(nn.Module):
        """Multi-head self-attention for tabular data."""

        def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            assert dim % n_heads == 0, "dim must be divisible by n_heads"

            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # x: (batch, seq_len, dim) — treat features as "tokens"
            B, N, D = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            out = (attn @ v).transpose(1, 2).reshape(B, N, D)
            out = self.proj(out)
            return self.norm(x + out)


    # ======================================================================
    # Models
    # ======================================================================

    class FiberMLP(nn.Module):
        """Multi-layer perceptron for tabular structure-property prediction.

        Parameters
        ----------
        n_features : int
            Number of input features.
        n_outputs : int
            Number of output targets.
        hidden : list of int
            Hidden layer sizes. Default: [128, 64, 32].
        dropout : float
            Dropout rate. Default: 0.1.
        activation : str
            Activation function: "relu", "gelu", "silu". Default: "relu".

        Examples
        --------
        >>> model = FiberMLP(n_features=20, n_outputs=3, hidden=[256, 128, 64])
        >>> out = model(torch.randn(16, 20))
        >>> print(out.shape)  # (16, 3)
        """

        def __init__(
            self,
            n_features: int,
            n_outputs: int = 1,
            hidden: Optional[List[int]] = None,
            dropout: float = 0.1,
            activation: str = "relu",
        ):
            _require_torch()
            super().__init__()

            if hidden is None:
                hidden = [128, 64, 32]

            act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
            act_cls = act_map.get(activation, nn.ReLU)

            layers = []
            prev = n_features
            for h in hidden:
                layers.extend([
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    act_cls(),
                    nn.Dropout(dropout),
                ])
                prev = h

            layers.append(nn.Linear(prev, n_outputs))
            self.net = nn.Sequential(*layers)
            self.n_features = n_features
            self.n_outputs = n_outputs

        def forward(self, x):
            return self.net(x)


    class FiberResNet(nn.Module):
        """Residual network for tabular data.

        Better than plain MLP for deeper architectures and noisy data.

        Parameters
        ----------
        n_features : int
            Number of input features.
        n_outputs : int
            Number of output targets.
        hidden : list of int
            Hidden dimensions for each residual block. Default: [128, 128, 128].
        dropout : float
            Dropout rate.

        Examples
        --------
        >>> model = FiberResNet(n_features=20, n_outputs=1, hidden=[256, 256])
        >>> out = model(torch.randn(16, 20))
        """

        def __init__(
            self,
            n_features: int,
            n_outputs: int = 1,
            hidden: Optional[List[int]] = None,
            dropout: float = 0.1,
        ):
            _require_torch()
            super().__init__()

            if hidden is None:
                hidden = [128, 128, 128]

            self.input_proj = nn.Linear(n_features, hidden[0])
            self.input_bn = nn.BatchNorm1d(hidden[0])

            blocks = []
            for i in range(len(hidden)):
                in_dim = hidden[i]
                out_dim = hidden[min(i + 1, len(hidden) - 1)] if i < len(hidden) - 1 else hidden[-1]
                blocks.append(_ResidualBlock(in_dim, out_dim, dropout))
            self.blocks = nn.ModuleList(blocks)

            self.head = nn.Linear(hidden[-1], n_outputs)
            self.n_features = n_features
            self.n_outputs = n_outputs

        def forward(self, x):
            x = F.relu(self.input_bn(self.input_proj(x)))
            for block in self.blocks:
                x = block(x)
            return self.head(x)


    class FiberAttentionNet(nn.Module):
        """Attention-based model that learns feature interactions.

        Treats each feature as a token and applies multi-head self-attention,
        enabling the model to discover non-linear feature combinations.

        Parameters
        ----------
        n_features : int
            Number of input features.
        n_outputs : int
            Number of output targets.
        embed_dim : int
            Embedding dimension per feature token. Default: 32.
        n_heads : int
            Number of attention heads. Default: 4.
        n_layers : int
            Number of attention layers. Default: 2.
        dropout : float
            Dropout rate.

        Examples
        --------
        >>> model = FiberAttentionNet(n_features=20, n_outputs=1, embed_dim=32)
        >>> out = model(torch.randn(16, 20))
        """

        def __init__(
            self,
            n_features: int,
            n_outputs: int = 1,
            embed_dim: int = 32,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.1,
        ):
            _require_torch()
            super().__init__()

            self.feature_embed = nn.Linear(1, embed_dim)
            self.pos_embed = nn.Parameter(torch.randn(1, n_features, embed_dim) * 0.02)

            attn_layers = []
            for _ in range(n_layers):
                attn_layers.append(_AttentionLayer(embed_dim, n_heads, dropout))
            self.attn_layers = nn.ModuleList(attn_layers)

            self.pool = nn.Linear(embed_dim, 1)
            self.head = nn.Sequential(
                nn.LayerNorm(n_features),
                nn.Linear(n_features, n_features // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(n_features // 2, n_outputs),
            )
            self.n_features = n_features
            self.n_outputs = n_outputs

        def forward(self, x):
            # x: (batch, n_features) → (batch, n_features, 1) → embed
            tokens = self.feature_embed(x.unsqueeze(-1)) + self.pos_embed

            for layer in self.attn_layers:
                tokens = layer(tokens)

            # Pool: (batch, n_features, embed) → (batch, n_features)
            pooled = self.pool(tokens).squeeze(-1)
            return self.head(pooled)


    class FiberUncertaintyNet(nn.Module):
        """Uncertainty-aware model: predicts mean + variance (aleatoric).

        Outputs 2 values per target: [mu, log_sigma2].
        Use with Gaussian NLL loss for proper uncertainty calibration.

        Parameters
        ----------
        n_features : int
            Number of input features.
        n_outputs : int
            Number of output targets.
        hidden : list of int
            Hidden layer sizes.
        dropout : float
            Dropout rate (also used for MC-Dropout epistemic uncertainty).

        Examples
        --------
        >>> model = FiberUncertaintyNet(n_features=20, n_outputs=1)
        >>> out = model(torch.randn(16, 20))
        >>> mu = out[:, 0:1]       # predicted mean
        >>> log_var = out[:, 1:2]  # log variance
        >>> # Gaussian NLL: loss = 0.5 * (log_var + (y - mu)**2 / exp(log_var))
        """

        def __init__(
            self,
            n_features: int,
            n_outputs: int = 1,
            hidden: Optional[List[int]] = None,
            dropout: float = 0.15,
        ):
            _require_torch()
            super().__init__()

            if hidden is None:
                hidden = [128, 64]

            layers = []
            prev = n_features
            for h in hidden:
                layers.extend([
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev = h
            self.backbone = nn.Sequential(*layers)
            self.mu_head = nn.Linear(hidden[-1], n_outputs)
            self.logvar_head = nn.Linear(hidden[-1], n_outputs)
            self.n_features = n_features
            self.n_outputs = n_outputs
            self._dropout = dropout

        def forward(self, x):
            h = self.backbone(x)
            mu = self.mu_head(h)
            log_var = self.logvar_head(h)
            log_var = torch.clamp(log_var, min=-6, max=6)
            return torch.cat([mu, log_var], dim=-1)

        def predict_with_uncertainty(self, x, n_mc: int = 20):
            """MC-Dropout for epistemic + aleatoric uncertainty.

            Parameters
            ----------
            x : torch.Tensor
                Input features.
            n_mc : int
                Number of MC forward passes.

            Returns
            -------
            mu : (batch, n_outputs) mean prediction
            sigma : (batch, n_outputs) total uncertainty (std)
            """
            self.train()  # keep dropout active
            mus = []
            vars_ = []
            for _ in range(n_mc):
                out = self.forward(x)
                mu = out[:, :self.n_outputs]
                log_var = out[:, self.n_outputs:]
                mus.append(mu)
                vars_.append(torch.exp(log_var))

            mus = torch.stack(mus)      # (n_mc, batch, n_outputs)
            vars_ = torch.stack(vars_)

            # Epistemic: variance of means
            epistemic = mus.var(dim=0)
            # Aleatoric: mean of variances
            aleatoric = vars_.mean(dim=0)
            # Total uncertainty
            sigma = torch.sqrt(epistemic + aleatoric)
            mu = mus.mean(dim=0)

            self.eval()
            return mu, sigma


    class FiberMultiTaskNet(nn.Module):
        """Multi-task model with shared backbone and per-task heads.

        Parameters
        ----------
        n_features : int
            Number of input features.
        task_names : list of str
            Names of prediction tasks.
        hidden : list of int
            Shared backbone hidden layers.
        head_hidden : list of int
            Per-task head hidden layers.

        Examples
        --------
        >>> model = FiberMultiTaskNet(
        ...     n_features=20,
        ...     task_names=["max_force", "E_star", "poisson_ratio"],
        ... )
        >>> outputs = model(torch.randn(16, 20))
        >>> # outputs = {"max_force": ..., "E_star": ..., "poisson_ratio": ...}
        """

        def __init__(
            self,
            n_features: int,
            task_names: List[str],
            hidden: Optional[List[int]] = None,
            head_hidden: Optional[List[int]] = None,
            dropout: float = 0.1,
        ):
            _require_torch()
            super().__init__()

            if hidden is None:
                hidden = [128, 64]
            if head_hidden is None:
                head_hidden = [32]

            self.task_names = task_names
            self.n_features = n_features

            # Build shared backbone directly
            layers = []
            prev = n_features
            for h in hidden:
                layers.extend([
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev = h
            self.shared_backbone = nn.Sequential(*layers)

            self.heads = nn.ModuleDict()
            for name in task_names:
                head_layers = []
                prev = hidden[-1]
                for h in head_hidden:
                    head_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
                    prev = h
                head_layers.append(nn.Linear(prev, 1))
                self.heads[name] = nn.Sequential(*head_layers)

        def forward(self, x):
            shared = self.shared_backbone(x)
            return {name: head(shared).squeeze(-1) for name, head in self.heads.items()}

        def predict_tensor(self, x) -> torch.Tensor:
            """Return concatenated predictions as a single tensor."""
            outputs = self.forward(x)
            return torch.stack([outputs[n] for n in self.task_names], dim=-1)


    class FiberEnsemble:
        """Ensemble of models for robust predictions with uncertainty.

        Parameters
        ----------
        models : list of nn.Module
            Trained models.

        Examples
        --------
        >>> ensemble = FiberEnsemble([model1, model2, model3])
        >>> mu, sigma = ensemble.predict_with_uncertainty(X)
        """

        def __init__(self, models: List[nn.Module]):
            _require_torch()
            self.models = models

        def predict(self, x) -> torch.Tensor:
            """Mean prediction across ensemble."""
            preds = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    preds.append(model(x))
            return torch.stack(preds).mean(dim=0)

        def predict_with_uncertainty(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
            """Predict with uncertainty (std across ensemble).

            Returns
            -------
            mu : (batch, n_outputs)
            sigma : (batch, n_outputs)
            """
            preds = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    preds.append(model(x))
            preds = torch.stack(preds)
            return preds.mean(dim=0), preds.std(dim=0)

        def to(self, device):
            for m in self.models:
                m.to(device)
            return self

        def eval(self):
            for m in self.models:
                m.eval()
            return self


    # ======================================================================
    # Loss Functions
    # ======================================================================

    def gaussian_nll_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Gaussian NLL loss for uncertainty models.

        Parameters
        ----------
        pred : (batch, 2*n_outputs)
            Concatenated [mu, log_var] predictions.
        target : (batch, n_outputs)
            Ground truth.

        Returns
        -------
        loss : scalar
        """
        n_out = target.shape[-1]
        mu = pred[:, :n_out]
        log_var = pred[:, n_out:]
        return 0.5 * (log_var + (target - mu).pow(2) / (log_var.exp() + 1e-8)).mean()


    def multi_task_loss(
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Weighted MSE loss for multi-task models.

        Parameters
        ----------
        outputs : dict
            Model outputs {task_name: predictions}.
        targets : dict
            Ground truth {task_name: targets}.
        weights : dict, optional
            Per-task loss weights. Default: uniform.

        Returns
        -------
        loss : scalar
        """
        if weights is None:
            weights = {name: 1.0 for name in outputs}

        total_loss = torch.tensor(0.0)
        for name in outputs:
            if name in targets:
                total_loss = total_loss + weights[name] * F.mse_loss(outputs[name], targets[name])
        return total_loss / max(len(targets), 1)

else:
    # Stub classes when torch is not available
    class FiberMLP:
        def __init__(self, *a, **kw):
            _require_torch()

    class FiberResNet:
        def __init__(self, *a, **kw):
            _require_torch()

    class FiberAttentionNet:
        def __init__(self, *a, **kw):
            _require_torch()

    class FiberUncertaintyNet:
        def __init__(self, *a, **kw):
            _require_torch()

    class FiberMultiTaskNet:
        def __init__(self, *a, **kw):
            _require_torch()

    class FiberEnsemble:
        def __init__(self, *a, **kw):
            _require_torch()

    def gaussian_nll_loss(*a, **kw):
        _require_torch()

    def multi_task_loss(*a, **kw):
        _require_torch()

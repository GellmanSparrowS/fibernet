"""
Neural Operators for FiberNet — Learning Mappings Between Function Spaces.

Implements operator learning architectures:
- FiberFNO: Fourier Neural Operator for field-to-field mapping
- FiberDeepONet: Deep Operator Network with branch-trunk decomposition
- SpectralConv: Spectral convolution layer using FFT
- NeuralOperatorTrainer: Training loop for operator learning

Features
--------
- Resolution-invariant learning (train at low res, evaluate at high res)
- Fourier spectral convolutions for global pattern capture
- Deep operator networks for parametric PDE solutions
- Multi-channel input/output support
- Compatible with fiber network structure → field mapping

References
----------
- Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations" (ICLR 2021)
- Lu et al., "DeepONet: Learning nonlinear operators" (Nat. Mach. Intell. 2021)
- Article section 4.3: Neural operators for field prediction

Examples
--------
>>> from fibernet.ml.neural_operator import FiberFNO, NeuralOperatorTrainer
>>> fno = FiberFNO(in_channels=3, out_channels=2, modes=16, width=64)
>>> trainer = NeuralOperatorTrainer(fno)
>>> trainer.fit(X_train, y_train, epochs=100)
>>> pred = fno(X_test)  # field prediction

>>> # DeepONet
>>> from fibernet.ml.neural_operator import FiberDeepONet
>>> don = FiberDeepONet(branch_dim=20, trunk_dim=2, hidden=[128, 64])
>>> trainer = NeuralOperatorTrainer(don)
>>> trainer.fit(X_train, y_train, coords_train, epochs=100)
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
    # Spectral Convolution
    # ==================================================================

    class SpectralConv2d(nn.Module):
        """2D Spectral convolution layer using FFT.

        Performs multiplication in Fourier space with learnable weights
        for low-frequency modes, enabling global pattern capture.

        Parameters
        ----------
        in_channels : int
            Input channels.
        out_channels : int
            Output channels.
        modes1 : int
            Number of Fourier modes in x direction.
        modes2 : int
            Number of Fourier modes in y direction.
        """

        def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.modes1 = modes1
            self.modes2 = modes2

            scale = 1.0 / (in_channels * out_channels)
            self.weights1 = nn.Parameter(
                scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
            )
            self.weights2 = nn.Parameter(
                scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
            )

        def _complex_mul(self, x, weights):
            """Complex multiplication via einsum."""
            return torch.einsum("bixy,ioxy->boxy", x, weights)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            x_ft = torch.fft.rfft2(x)

            out_ft = torch.zeros(
                batch, self.out_channels, x.shape[-2], x.shape[-1] // 2 + 1,
                device=x.device, dtype=torch.cfloat,
            )

            # Convert stored real/imag weights to complex
            w1 = torch.complex(self.weights1[..., 0], self.weights1[..., 1])
            w2 = torch.complex(self.weights2[..., 0], self.weights2[..., 1])

            # Upper frequency modes
            x_upper = x_ft[:, :, :self.modes1, :self.modes2]
            out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
                "bixy,ioxy->boxy", x_upper, w1
            )

            # Lower frequency modes (negative)
            x_lower = x_ft[:, :, -self.modes1:, :self.modes2]
            out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
                "bixy,ioxy->boxy", x_lower, w2
            )

            return torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))


    class SpectralConv1d(nn.Module):
        """1D Spectral convolution layer."""

        def __init__(self, in_channels: int, out_channels: int, modes: int):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.modes = modes

            scale = 1.0 / (in_channels * out_channels)
            self.weights = nn.Parameter(
                scale * torch.rand(in_channels, out_channels, modes, 2)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            x_ft = torch.fft.rfft(x)

            out_ft = torch.zeros(
                batch, self.out_channels, x.shape[-1] // 2 + 1,
                device=x.device, dtype=torch.cfloat,
            )

            w = torch.complex(self.weights[..., 0], self.weights[..., 1])
            x_modes = x_ft[:, :, :self.modes]
            out_ft[:, :, :self.modes] = torch.einsum(
                "bim,iom->bom", x_modes, w
            )

            return torch.fft.irfft(out_ft, n=x.shape[-1])


    # ==================================================================
    # Fourier Neural Operator
    # ==================================================================

    class FNOBlock(nn.Module):
        """Single FNO block: spectral conv + pointwise conv + activation."""

        def __init__(self, channels: int, modes1: int, modes2: int):
            super().__init__()
            self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
            self.pointwise = nn.Conv2d(channels, channels, 1)
            self.norm = nn.InstanceNorm2d(channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.spectral(x) + self.pointwise(x)
            return F.gelu(self.norm(h))


    class FiberFNO(nn.Module):
        """Fourier Neural Operator for field-to-field mapping.

        Maps input fields (e.g., structure encoding) to output fields
        (e.g., stress, displacement) using spectral convolutions.

        Parameters
        ----------
        in_channels : int
            Input field channels.
        out_channels : int
            Output field channels.
        modes : int
            Number of Fourier modes.
        width : int
            Hidden channel width.
        n_layers : int
            Number of FNO blocks.

        Examples
        --------
        >>> fno = FiberFNO(in_channels=3, out_channels=2, modes=16, width=64)
        >>> X = torch.randn(8, 3, 64, 64)  # 8 structures as 3-channel images
        >>> Y = fno(X)  # (8, 2, 64, 64) stress field prediction
        """

        def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 2,
            modes: int = 16,
            width: int = 64,
            n_layers: int = 4,
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.modes = modes
            self.width = width
            self.n_layers = n_layers

            # Lifting layer
            self.lift = nn.Conv2d(in_channels, width, 1)

            # FNO blocks
            self.blocks = nn.ModuleList([
                FNOBlock(width, modes, modes) for _ in range(n_layers)
            ])

            # Projection
            self.proj = nn.Sequential(
                nn.Conv2d(width, width * 2, 1),
                nn.GELU(),
                nn.Conv2d(width * 2, out_channels, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.lift(x)
            for block in self.blocks:
                x = x + block(x)  # Residual connections
            return self.proj(x)


    # ==================================================================
    # Deep Operator Network
    # ==================================================================

    class FiberDeepONet(nn.Module):
        """Deep Operator Network with branch-trunk architecture.

        Learns operator G: u → G(u)(y) where:
        - Branch network encodes input function u (structure features)
        - Trunk network encodes evaluation points y (spatial coordinates)
        - Output = inner_product(branch(u), trunk(y))

        Parameters
        ----------
        branch_dim : int
            Branch input dimension (structure feature vector).
        trunk_dim : int
            Trunk input dimension (spatial coordinates, e.g., 2 for x,y).
        hidden : list of int
            Hidden layer sizes for both networks.
        n_outputs : int
            Number of output functions (e.g., 2 for [stress_x, stress_y]).
        activation : str
            Activation function.

        Examples
        --------
        >>> don = FiberDeepONet(branch_dim=20, trunk_dim=2, hidden=[128, 64])
        >>> features = torch.randn(32, 20)  # structure features
        >>> coords = torch.randn(32, 100, 2)  # 100 evaluation points per sample
        >>> output = don(features, coords)  # (32, 100, 2) field at each point
        """

        def __init__(
            self,
            branch_dim: int,
            trunk_dim: int = 2,
            hidden: Optional[List[int]] = None,
            n_outputs: int = 1,
            activation: str = "relu",
        ):
            super().__init__()
            if hidden is None:
                hidden = [128, 64]

            self.branch_dim = branch_dim
            self.trunk_dim = trunk_dim
            self.n_outputs = n_outputs
            self.p = hidden[-1]  # inner product dimension

            act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
            act_cls = act_map.get(activation, nn.ReLU)

            # Branch network
            branch_layers = []
            prev = branch_dim
            for h in hidden:
                branch_layers.extend([nn.Linear(prev, h), act_cls(), nn.Dropout(0.1)])
                prev = h
            branch_layers.append(nn.Linear(prev, self.p * n_outputs))
            self.branch = nn.Sequential(*branch_layers)

            # Trunk network
            trunk_layers = []
            prev = trunk_dim
            for h in hidden:
                trunk_layers.extend([nn.Linear(prev, h), act_cls()])
                prev = h
            trunk_layers.append(nn.Linear(prev, self.p))
            self.trunk = nn.Sequential(*trunk_layers)

            # Bias
            self.bias = nn.Parameter(torch.zeros(n_outputs))

        def forward(
            self, features: torch.Tensor, coords: torch.Tensor
        ) -> torch.Tensor:
            """Forward pass.

            Parameters
            ----------
            features : (batch, branch_dim)
                Structure feature vectors.
            coords : (batch, n_points, trunk_dim)
                Spatial evaluation points.

            Returns
            -------
            (batch, n_points, n_outputs) field values at evaluation points.
            """
            batch = features.shape[0]
            n_points = coords.shape[1]

            # Branch: (batch, p * n_outputs)
            b = self.branch(features)
            b = b.view(batch, self.n_outputs, self.p)

            # Trunk: (batch * n_points, p)
            coords_flat = coords.reshape(-1, self.trunk_dim)
            t = self.trunk(coords_flat)
            t = t.view(batch, n_points, self.p)

            # Inner product
            out = torch.einsum("bop,bnp->bno", b, t)
            out = out + self.bias

            return out

        def predict_field(
            self,
            features: torch.Tensor,
            grid_size: Tuple[int, int] = (64, 64),
            bounds: Tuple[Tuple[float, float], ...] = ((0, 1), (0, 1)),
        ) -> torch.Tensor:
            """Predict field on a regular grid.

            Parameters
            ----------
            features : (batch, branch_dim)
            grid_size : (H, W)
            bounds : domain bounds per dimension

            Returns
            -------
            (batch, n_outputs, H, W) field prediction
            """
            H, W = grid_size
            device = features.device

            # Create grid coordinates
            xs = torch.linspace(bounds[0][0], bounds[0][1], W, device=device)
            ys = torch.linspace(bounds[1][0], bounds[1][1], H, device=device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            coords = coords.unsqueeze(0).expand(features.shape[0], -1, -1)

            # Predict
            out = self.forward(features, coords)  # (batch, n_points, n_outputs)

            # Reshape to grid
            batch = features.shape[0]
            return out.view(batch, H, W, self.n_outputs).permute(0, 3, 1, 2)


    # ==================================================================
    # Trainer
    # ==================================================================

    class NeuralOperatorTrainer:
        """Training loop for neural operator models.

        Parameters
        ----------
        model : FiberFNO or FiberDeepONet
            Neural operator model.
        lr : float
            Learning rate.
        weight_decay : float
            Weight decay.

        Examples
        --------
        >>> trainer = NeuralOperatorTrainer(fno)
        >>> trainer.fit(X_train, y_train, epochs=100)
        >>> pred = trainer.predict(X_test)
        """

        def __init__(
            self,
            model: Union[FiberFNO, FiberDeepONet],
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
            coords: Optional[np.ndarray] = None,
            *,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            coords_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 16,
            verbose: bool = True,
        ) -> Dict[str, Any]:
            """Train the neural operator.

            Parameters
            ----------
            X_train : array-like
                For FNO: (N, C_in, H, W) input fields.
                For DeepONet: (N, branch_dim) feature vectors.
            y_train : array-like
                For FNO: (N, C_out, H, W) output fields.
                For DeepONet: (N, n_points) or (N, n_points, n_outputs).
            coords : array-like, optional
                For DeepONet: (N, n_points, trunk_dim) evaluation points.
            X_val, y_val, coords_val : optional
                Validation data.
            epochs, batch_size : training params.
            verbose : bool

            Returns
            -------
            dict
                Training history.
            """
            from fibernet.ml.training import TrainingHistory

            X_t = torch.tensor(X_train, dtype=torch.float32)
            y_t = torch.tensor(y_train, dtype=torch.float32)
            C_t = torch.tensor(coords, dtype=torch.float32) if coords is not None else None

            is_deeponet = isinstance(self.model, FiberDeepONet)

            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )
            history = TrainingHistory()

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

                    if is_deeponet:
                        cb = C_t[idx] if C_t is not None else None
                        if cb is None:
                            # Generate default coordinates
                            bs = end - start
                            n_pts = y_t.shape[1] if y_t.dim() > 1 else 1
                            cb = torch.rand(bs, n_pts, self.model.trunk_dim)
                        pred = self.model(xb, cb)
                        yb = y_t[idx]
                        if yb.dim() == 2:
                            yb = yb.unsqueeze(-1)
                    else:
                        pred = self.model(xb)
                        yb = y_t[idx]

                    loss = F.mse_loss(pred, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step()
                epoch_loss /= max(n_batches, 1)
                history.update(epoch, epoch_loss, lr=optimizer.param_groups[0]["lr"])

                # Validation
                val_loss = None
                if X_val is not None and y_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        X_v = torch.tensor(X_val, dtype=torch.float32)
                        y_v = torch.tensor(y_val, dtype=torch.float32)
                        if is_deeponet:
                            C_v = torch.tensor(coords_val, dtype=torch.float32) if coords_val is not None else None
                            if C_v is None:
                                n_pts = y_v.shape[1] if y_v.dim() > 1 else 1
                                C_v = torch.rand(len(X_v), n_pts, self.model.trunk_dim)
                            pred_v = self.model(X_v, C_v)
                            if y_v.dim() == 2:
                                y_v = y_v.unsqueeze(-1)
                        else:
                            pred_v = self.model(X_v)
                        val_loss = F.mse_loss(pred_v, y_v).item()
                    history.update(epoch, epoch_loss, val_loss, lr=optimizer.param_groups[0]["lr"])

                if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
                    val_str = f" | val={val_loss:.4f}" if val_loss else ""
                    print(f"Epoch {epoch:3d} | loss={epoch_loss:.4f}{val_str}")

            self.history["train_loss"] = history.train_losses
            self.history["val_loss"] = history.val_losses
            return {"history": history, "final_loss": epoch_loss}

        def predict(self, X: np.ndarray, coords: Optional[np.ndarray] = None) -> np.ndarray:
            """Run inference."""
            self.model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X, dtype=torch.float32)
                if isinstance(self.model, FiberDeepONet):
                    if coords is None:
                        n_pts = 100
                        C_t = torch.rand(len(X_t), n_pts, self.model.trunk_dim)
                    else:
                        C_t = torch.tensor(coords, dtype=torch.float32)
                    return self.model(X_t, C_t).cpu().numpy()
                return self.model(X_t).cpu().numpy()

else:
    class FiberFNO:
        def __init__(self, *a, **kw):
            _require_torch()

    class FiberDeepONet:
        def __init__(self, *a, **kw):
            _require_torch()

    class NeuralOperatorTrainer:
        def __init__(self, *a, **kw):
            _require_torch()

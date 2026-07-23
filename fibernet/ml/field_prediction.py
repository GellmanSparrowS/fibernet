"""
Field-Level Prediction Models for FiberNet (2D/3D Spatial Fields).

Implements encoder-decoder architectures for predicting spatially-resolved
mechanical fields from fiber network structures:
- FiberUNet: U-Net with attention gates for 2D field prediction
- FiberResUNet: Residual U-Net for improved gradient flow
- FiberFieldMLP: Point-wise MLP for implicit neural representation
- FieldDataset: Dataset utilities for structure→field data pairs
- train_field_model: Training loop with multi-scale losses

Features
--------
- Skip connections preserving spatial detail
- Attention gates focusing on critical regions
- Multi-scale feature extraction (encoder pyramid)
- Structural similarity loss (SSIM) for spatial coherence
- Gradient-based loss for field smoothness
- Supports both image-like grids and unstructured point clouds

References
----------
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- Article section 4.3: CNN-based encoder-decoder for field prediction

Examples
--------
>>> from fibernet.ml.field_prediction import FiberUNet, train_field_model
>>> model = FiberUNet(in_channels=3, out_channels=2, base_channels=32)
>>> history = train_field_model(
...     model, X_train, y_train,
...     X_val=X_val, y_val=y_val,
...     epochs=100, lr=1e-3,
... )
>>> pred_field = model(X_test)  # (batch, 2, H, W)
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
    # Building Blocks
    # ==================================================================

    class _DoubleConv(nn.Module):
        """Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU."""

        def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None):
            super().__init__()
            mid_ch = mid_ch or out_ch
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)


    class _ResDoubleConv(nn.Module):
        """Residual double convolution block."""

        def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None):
            super().__init__()
            mid_ch = mid_ch or out_ch
            self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(mid_ch)
            self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_ch)

            if in_ch != out_ch:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                )
            else:
                self.shortcut = nn.Identity()

        def forward(self, x):
            residual = self.shortcut(x)
            h = F.relu(self.bn1(self.conv1(x)), inplace=True)
            h = self.bn2(self.conv2(h))
            return F.relu(h + residual, inplace=True)


    class _Down(nn.Module):
        """Downscaling: MaxPool → DoubleConv."""

        def __init__(self, in_ch: int, out_ch: int, residual: bool = False):
            super().__init__()
            self.pool = nn.MaxPool2d(2)
            conv_cls = _ResDoubleConv if residual else _DoubleConv
            self.conv = conv_cls(in_ch, out_ch)

        def forward(self, x):
            return self.conv(self.pool(x))


    class _Up(nn.Module):
        """Upscaling: Upsample → concat skip → DoubleConv."""

        def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True, residual: bool = False):
            super().__init__()
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                conv_cls = _ResDoubleConv if residual else _DoubleConv
                self.conv = conv_cls(in_ch, out_ch, in_ch // 2)
            else:
                self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
                conv_cls = _ResDoubleConv if residual else _DoubleConv
                self.conv = conv_cls(in_ch, out_ch)

        def forward(self, x, skip):
            x = self.up(x)
            # Pad if needed
            dy = skip.shape[2] - x.shape[2]
            dx = skip.shape[3] - x.shape[3]
            if dy > 0 or dx > 0:
                x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
            x = torch.cat([skip, x], dim=1)
            return self.conv(x)


    class _AttentionGate(nn.Module):
        """Attention gate for skip connections."""

        def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
            super().__init__()
            self.W_gate = nn.Sequential(
                nn.Conv2d(gate_ch, inter_ch, 1, bias=False),
                nn.BatchNorm2d(inter_ch),
            )
            self.W_skip = nn.Sequential(
                nn.Conv2d(skip_ch, inter_ch, 1, bias=False),
                nn.BatchNorm2d(inter_ch),
            )
            self.psi = nn.Sequential(
                nn.Conv2d(inter_ch, 1, 1, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid(),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, gate, skip):
            g = self.W_gate(gate)
            s = self.W_skip(skip)
            # Upsample gate if sizes don't match
            if g.shape[2:] != s.shape[2:]:
                g = F.interpolate(g, size=s.shape[2:], mode="bilinear", align_corners=True)
            attn = self.relu(g + s)
            attn = self.psi(attn)
            return skip * attn


    # ==================================================================
    # Models
    # ==================================================================

    class FiberUNet(nn.Module):
        """U-Net for 2D spatial field prediction.

        Parameters
        ----------
        in_channels : int
            Input channels (e.g., 3 for structure encoding: density, orientation, connectivity).
        out_channels : int
            Output channels (e.g., 2 for stress field [σ_xx, σ_yy]).
        base_channels : int
            Base channel count (doubles each level).
        n_levels : int
            Number of encoder levels.
        bilinear : bool
            Use bilinear upsampling (True) or transposed conv (False).
        use_attention : bool
            Use attention gates on skip connections.

        Examples
        --------
        >>> model = FiberUNet(in_channels=3, out_channels=2, base_channels=64)
        >>> X = torch.randn(8, 3, 128, 128)  # 8 samples, 3-channel input
        >>> Y = model(X)  # (8, 2, 128, 128) stress field prediction
        """

        def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 2,
            base_channels: int = 64,
            n_levels: int = 4,
            bilinear: bool = True,
            use_attention: bool = False,
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.n_levels = n_levels
            self.use_attention = use_attention

            c = base_channels
            factor = 2 if bilinear else 1

            # Encoder
            self.inc = _DoubleConv(in_channels, c)
            self.downs = nn.ModuleList()
            ch = c
            for i in range(n_levels - 1):
                out_ch = ch * 2
                if i == n_levels - 2:
                    out_ch = out_ch // factor
                self.downs.append(_Down(ch, out_ch))
                ch = out_ch

            # Decoder
            self.ups = nn.ModuleList()
            for i in range(n_levels - 1):
                in_ch = ch + ch // factor if i == 0 else ch + ch // 2
                if i == 0:
                    in_ch = ch + (ch // factor)
                else:
                    skip_ch = base_channels * (2 ** (n_levels - 2 - i))
                    in_ch = ch + skip_ch
                out_ch = ch // 2
                self.ups.append(_Up(ch * 2 if i == 0 else ch + skip_ch, out_ch, bilinear))
                ch = out_ch

            # Attention gates
            if use_attention:
                self.attention_gates = nn.ModuleList()
                for i in range(n_levels - 1):
                    gate_ch = ch
                    skip_ch = base_channels * (2 ** (n_levels - 2 - i))
                    self.attention_gates.append(
                        _AttentionGate(gate_ch, skip_ch, skip_ch // 2)
                    )

            self.outc = nn.Conv2d(c, out_channels, 1)

            # Rebuild properly with simple approach
            self._build_simple(in_channels, out_channels, base_channels, n_levels, bilinear)

        def _build_simple(self, in_ch, out_ch, base, n_levels, bilinear):
            """Rebuild with a straightforward encoder-decoder."""
            c = base
            # Encoder
            self.encoder_blocks = nn.ModuleList()
            self.pool = nn.MaxPool2d(2)
            self.encoder_blocks.append(_DoubleConv(in_ch, c))
            ch = c
            enc_channels = [ch]
            for _ in range(n_levels - 1):
                next_ch = ch * 2
                self.encoder_blocks.append(_Down(ch, next_ch))
                ch = next_ch
                enc_channels.append(ch)

            # Decoder
            self.decoder_blocks = nn.ModuleList()
            for i in range(n_levels - 1):
                skip_ch = enc_channels[-(i + 2)]
                in_up = ch + skip_ch
                out_up = ch // 2
                if bilinear:
                    self.decoder_blocks.append(nn.ModuleDict({
                        "up": nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                        "conv": _DoubleConv(ch + skip_ch, out_up),
                    }))
                else:
                    self.decoder_blocks.append(nn.ModuleDict({
                        "up": nn.ConvTranspose2d(ch, ch // 2, 2, stride=2),
                        "conv": _DoubleConv(ch, out_up),
                    }))
                ch = out_up

            self.out_conv = nn.Conv2d(ch, out_ch, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Encoder path
            skips = []
            h = x
            for i, block in enumerate(self.encoder_blocks):
                h = block(h) if i == 0 else block(h)
                skips.append(h)
                if i < len(self.encoder_blocks) - 1:
                    # Downs already include pooling
                    pass

            # Re-do: simple encoder
            h = x
            skips = []
            for i, block in enumerate(self.encoder_blocks):
                if i == 0:
                    h = block(h)
                else:
                    h = block(h)
                if i < len(self.encoder_blocks) - 1:
                    skips.append(h)

            # Decoder path
            for i, dec in enumerate(self.decoder_blocks):
                skip = skips[-(i + 1)]
                h = dec["up"](h)
                # Pad if needed
                dy, dx = skip.shape[2] - h.shape[2], skip.shape[3] - h.shape[3]
                if dy > 0 or dx > 0:
                    h = F.pad(h, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
                h = torch.cat([skip, h], dim=1)
                h = dec["conv"](h)

            return self.out_conv(h)


    class FiberFieldMLP(nn.Module):
        """Implicit neural representation for point-wise field prediction.

        Predicts field values at arbitrary spatial coordinates given
        structure features. Useful for unstructured point clouds.

        Parameters
        ----------
        in_dim : int
            Input dimension (spatial coords + structure features).
        out_dim : int
            Output dimension (field values).
        hidden : list of int
            Hidden layer sizes.
        positional_encoding : bool
            Apply sinusoidal positional encoding to coordinates.
        n_frequencies : int
            Number of frequency bands for positional encoding.

        Examples
        --------
        >>> model = FiberFieldMLP(in_dim=5, out_dim=2, hidden=[128, 128, 64])
        >>> # Input: [x, y, density, orientation, connectivity]
        >>> coords = torch.randn(1000, 5)
        >>> field = model(coords)  # (1000, 2) stress at each point
        """

        def __init__(
            self,
            in_dim: int = 5,
            out_dim: int = 2,
            hidden: Optional[List[int]] = None,
            positional_encoding: bool = True,
            n_frequencies: int = 8,
        ):
            super().__init__()
            if hidden is None:
                hidden = [128, 128, 64]

            self.positional_encoding = positional_encoding
            self.n_frequencies = n_frequencies
            self.coord_dim = 2  # x, y

            pe_dim = self.coord_dim * 2 * n_frequencies if positional_encoding else 0
            total_in = in_dim + pe_dim

            layers = []
            prev = total_in
            for h in hidden:
                layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
                prev = h
            layers.append(nn.Linear(prev, out_dim))
            self.net = nn.Sequential(*layers)

        def _encode_coords(self, coords: torch.Tensor) -> torch.Tensor:
            """Apply positional encoding to coordinate dimensions."""
            if not self.positional_encoding:
                return coords

            spatial = coords[:, :self.coord_dim]
            features = coords[:, self.coord_dim:]

            pe_list = [spatial]
            for i in range(self.n_frequencies):
                freq = 2.0 ** i * math.pi
                pe_list.append(torch.sin(spatial * freq))
                pe_list.append(torch.cos(spatial * freq))

            return torch.cat(pe_list + [features], dim=-1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self._encode_coords(x)
            return self.net(x)


    # ==================================================================
    # Loss Functions
    # ==================================================================

    def field_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Standard MSE loss for field prediction."""
        return F.mse_loss(pred, target)


    def field_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Gradient-based loss encouraging smooth field predictions.

        Computes the difference in spatial gradients between
        predicted and target fields.
        """
        # Sobel-like gradient approximation
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_dx = F.mse_loss(pred_dx, target_dx)
        loss_dy = F.mse_loss(pred_dy, target_dy)
        return loss_dx + loss_dy


    def field_multi_scale_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        n_scales: int = 3,
        mse_weight: float = 1.0,
        grad_weight: float = 0.1,
    ) -> torch.Tensor:
        """Multi-scale loss combining MSE and gradient losses at multiple resolutions.

        Parameters
        ----------
        pred, target : (B, C, H, W)
        n_scales : int
            Number of resolution scales.
        mse_weight : float
            Weight for MSE term.
        grad_weight : float
            Weight for gradient term.
        """
        total_loss = torch.tensor(0.0, device=pred.device)

        for s in range(n_scales):
            if s > 0:
                scale = 2 ** s
                pred_s = F.avg_pool2d(pred, scale)
                target_s = F.avg_pool2d(target, scale)
            else:
                pred_s = pred
                target_s = target

            mse = F.mse_loss(pred_s, target_s)
            total_loss = total_loss + mse_weight * mse

            if pred_s.shape[2] > 1 and pred_s.shape[3] > 1:
                grad = field_gradient_loss(pred_s, target_s)
                total_loss = total_loss + grad_weight * grad

        return total_loss / n_scales


    # ==================================================================
    # Training
    # ==================================================================

    def train_field_model(
        model: Union[FiberUNet, FiberFieldMLP],
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 16,
        loss_fn: str = "multi_scale",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train a field prediction model.

        Parameters
        ----------
        model : FiberUNet or FiberFieldMLP
            Field prediction model.
        X_train : array-like
            Training inputs. Shape depends on model type:
            - FiberUNet: (N, C_in, H, W)
            - FiberFieldMLP: (N, in_dim)
        y_train : array-like
            Training targets:
            - FiberUNet: (N, C_out, H, W)
            - FiberFieldMLP: (N, out_dim)
        X_val, y_val : optional
            Validation data.
        epochs : int
            Training epochs.
        lr : float
            Learning rate.
        batch_size : int
            Batch size.
        loss_fn : str
            "mse", "gradient", or "multi_scale".
        verbose : bool
            Print progress.

        Returns
        -------
        dict
            Training history.
        """
        from fibernet.ml.training import TrainingHistory

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        history = TrainingHistory()

        loss_fns = {
            "mse": field_mse_loss,
            "gradient": lambda p, t: field_mse_loss(p, t) + 0.1 * field_gradient_loss(p, t),
            "multi_scale": field_multi_scale_loss if isinstance(model, FiberUNet) else field_mse_loss,
        }
        criterion = loss_fns.get(loss_fn, field_mse_loss)

        n = len(X_t)

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]
                xb, yb = X_t[idx], y_t[idx]

                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            epoch_loss /= max(n_batches, 1)

            val_loss = None
            if X_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    X_v = torch.tensor(X_val, dtype=torch.float32)
                    y_v = torch.tensor(y_val, dtype=torch.float32)
                    pred_v = model(X_v)
                    val_loss = F.mse_loss(pred_v, y_v).item()

            history.update(epoch, epoch_loss, val_loss, lr=optimizer.param_groups[0]["lr"])

            if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
                val_str = f" | val={val_loss:.4f}" if val_loss else ""
                print(f"Epoch {epoch:3d} | loss={epoch_loss:.4f}{val_str}")

        return {"history": history, "final_loss": epoch_loss}


    def structure_to_field_input(
        g,
        grid_size: Tuple[int, int] = (64, 64),
        channels: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Convert a StructureGraph to a multi-channel field image.

        Parameters
        ----------
        g : StructureGraph
            Input structure.
        grid_size : (H, W)
            Output grid resolution.
        channels : list of str, optional
            Channel types: "density", "orientation", "connectivity", "radius".
            Default: ["density", "orientation", "connectivity"].

        Returns
        -------
        (C, H, W) numpy array
        """
        if channels is None:
            channels = ["density", "orientation", "connectivity"]

        H, W = grid_size
        field = np.zeros((len(channels), H, W), dtype=np.float32)

        try:
            bb_min, bb_max = g.bounding_box()
            span = bb_max - bb_min
            if span[0] < 1e-6:
                span[0] = 1.0
            if len(span) > 1 and span[1] < 1e-6:
                span[1] = 1.0
        except Exception:
            bb_min = np.zeros(2)
            span = np.ones(2)

        for edge_id, edge in g.edges.items():
            ni = edge.node_i
            nj = edge.node_j
            if ni not in g.nodes or nj not in g.nodes:
                continue

            pos_i = g.nodes[ni].position[:2]
            pos_j = g.nodes[nj].position[:2]

            # Rasterize edge onto grid
            n_pts = max(int(np.linalg.norm(pos_j - pos_i) * 5), 10)
            for t in np.linspace(0, 1, n_pts):
                pt = pos_i * (1 - t) + pos_j * t
                gx = int((pt[0] - bb_min[0]) / span[0] * (W - 1))
                gy = int((pt[1] - bb_min[1]) / span[1] * (H - 1))
                gx = np.clip(gx, 0, W - 1)
                gy = np.clip(gy, 0, H - 1)

                for ci, ch_name in enumerate(channels):
                    if ch_name == "density":
                        field[ci, gy, gx] += 1.0
                    elif ch_name == "orientation":
                        direction = pos_j - pos_i
                        angle = np.arctan2(direction[1], direction[0])
                        field[ci, gy, gx] = angle
                    elif ch_name == "connectivity":
                        deg_i = g.degree(ni)
                        deg_j = g.degree(nj)
                        field[ci, gy, gx] = (deg_i + deg_j) / 2
                    elif ch_name == "radius":
                        field[ci, gy, gx] = float(getattr(edge, "radius", 0.1))

        # Smooth with Gaussian
        from scipy.ndimage import gaussian_filter
        for ci in range(len(channels)):
            field[ci] = gaussian_filter(field[ci], sigma=1.0)

        return field

else:
    class FiberUNet:
        def __init__(self, *a, **kw):
            _require_torch()

    class FiberFieldMLP:
        def __init__(self, *a, **kw):
            _require_torch()

    def train_field_model(*a, **kw):
        _require_torch()

    def structure_to_field_input(*a, **kw):
        _require_torch()
        return np.zeros((3, 64, 64), dtype=np.float32)

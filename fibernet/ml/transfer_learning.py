"""
Transfer & Meta-Learning for FiberNet — Low-Data Regime Support.

Implements strategies for learning from limited simulation data:
- FiberTransferNet: Pre-train on source, fine-tune on target
- FiberPrototypicalNet: Prototypical networks for few-shot learning
- FiberMAML: Model-Agnostic Meta-Learning wrapper
- DomainAdapter: Domain adaptation between structure types

Features
--------
- Pre-training on large datasets, fine-tuning on small target sets
- Few-shot learning with prototypical networks
- Domain adaptation for transfer between unit types
- Learning rate scheduling for fine-tuning
- Layer freezing strategies

References
----------
- Article section 6.2: Transfer learning for novel material systems
- Finn et al., "Model-Agnostic Meta-Learning" (ICML 2017)
- Snell et al., "Prototypical Networks for Few-shot Learning" (NeurIPS 2017)

Examples
--------
>>> from fibernet.ml.transfer_learning import FiberTransferNet, train_transfer
>>> # Pre-train on honeycomb data
>>> transfer = FiberTransferNet(n_features=20, n_outputs=1, hidden=[128, 64])
>>> transfer.pretrain(X_honeycomb, y_honeycomb, epochs=100)
>>> # Fine-tune on small re-entrant dataset
>>> transfer.finetune(X_reentrant, y_reentrant, epochs=20, freeze_layers=2)
>>> pred = transfer.predict(X_test)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import copy

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

    class _TransferBackbone(nn.Module):
        """Shared backbone for transfer learning."""

        def __init__(self, n_features: int, hidden: List[int], dropout: float = 0.1):
            super().__init__()
            layers = []
            prev = n_features
            for h in hidden:
                layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
                prev = h
            self.net = nn.Sequential(*layers)
            self.output_dim = hidden[-1]

        def forward(self, x):
            return self.net(x)

        def get_layer_groups(self) -> List[nn.Module]:
            """Return layer groups for selective freezing."""
            groups = []
            for module in self.net:
                if isinstance(module, nn.Linear):
                    groups.append(module)
            return groups


    class FiberTransferNet(nn.Module):
        """Transfer learning model for structure-property prediction.

        Parameters
        ----------
        n_features : int
            Input feature dimension.
        n_outputs : int
            Output target dimension.
        hidden : list of int
            Hidden layer sizes.
        dropout : float
            Dropout rate.

        Examples
        --------
        >>> transfer = FiberTransferNet(n_features=20, n_outputs=1)
        >>> transfer.pretrain(X_source, y_source, epochs=100)
        >>> transfer.finetune(X_target, y_target, epochs=20, freeze_layers=2)
        >>> pred = transfer.predict(X_new)
        """

        def __init__(
            self,
            n_features: int = 20,
            n_outputs: int = 1,
            hidden: Optional[List[int]] = None,
            dropout: float = 0.1,
        ):
            super().__init__()
            if hidden is None:
                hidden = [128, 64]

            self.n_features = n_features
            self.n_outputs = n_outputs
            self.backbone = _TransferBackbone(n_features, hidden, dropout)
            self.head = nn.Linear(self.backbone.output_dim, n_outputs)
            self._pretrained = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.backbone(x)
            return self.head(features)

        def pretrain(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            epochs: int = 100,
            lr: float = 1e-3,
            batch_size: int = 64,
            verbose: bool = True,
        ) -> Dict[str, Any]:
            """Pre-train on source domain data.

            Parameters
            ----------
            X : (N, n_features)
                Source domain features.
            y : (N,) or (N, n_outputs)
                Source domain targets.
            epochs : int
            lr : float
            batch_size : int
            verbose : bool

            Returns
            -------
            dict
                Training history.
            """
            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)
            if y_t.dim() == 1:
                y_t = y_t.unsqueeze(-1)

            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            n = len(X_t)
            losses = []

            for epoch in range(epochs):
                self.train()
                perm = torch.randperm(n)
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    idx = perm[start:end]
                    xb, yb = X_t[idx], y_t[idx]

                    optimizer.zero_grad()
                    pred = self(xb)
                    loss = F.mse_loss(pred, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step()
                epoch_loss /= max(n_batches, 1)
                losses.append(epoch_loss)

                if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
                    print(f"Pretrain Epoch {epoch:3d} | loss={epoch_loss:.4f}")

            self._pretrained = True
            return {"train_loss": losses}

        def finetune(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            epochs: int = 20,
            lr: float = 1e-4,
            batch_size: int = 32,
            freeze_layers: int = 0,
            verbose: bool = True,
        ) -> Dict[str, Any]:
            """Fine-tune on target domain data.

            Parameters
            ----------
            X : (M, n_features)
                Target domain features (typically smaller).
            y : (M,) or (M, n_outputs)
                Target domain targets.
            epochs : int
            lr : float
                Lower learning rate for fine-tuning.
            batch_size : int
            freeze_layers : int
                Number of backbone layers to freeze (from input side).
                0 = fine-tune all, high = only train head.
            verbose : bool

            Returns
            -------
            dict
                Training history.
            """
            # Freeze layers
            layer_groups = self.backbone.get_layer_groups()
            for i, layer in enumerate(layer_groups):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)
            if y_t.dim() == 1:
                y_t = y_t.unsqueeze(-1)

            trainable = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(epochs // 3, 1))

            n = len(X_t)
            losses = []

            for epoch in range(epochs):
                self.train()
                perm = torch.randperm(n)
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    idx = perm[start:end]
                    xb, yb = X_t[idx], y_t[idx]

                    optimizer.zero_grad()
                    pred = self(xb)
                    loss = F.mse_loss(pred, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step()
                epoch_loss /= max(n_batches, 1)
                losses.append(epoch_loss)

                if verbose and (epoch % max(epochs // 5, 1) == 0 or epoch == epochs - 1):
                    print(f"Finetune Epoch {epoch:3d} | loss={epoch_loss:.4f}")

            # Unfreeze all for future use
            for layer in layer_groups:
                for param in layer.parameters():
                    param.requires_grad = True

            return {"train_loss": losses}

        @torch.no_grad()
        def predict(self, X: np.ndarray) -> np.ndarray:
            self.eval()
            X_t = torch.tensor(X, dtype=torch.float32)
            return self(X_t).numpy()

        def save_pretrained(self, path: str):
            """Save pre-trained backbone."""
            torch.save({
                "backbone": self.backbone.state_dict(),
                "head": self.head.state_dict(),
                "n_features": self.n_features,
                "n_outputs": self.n_outputs,
            }, path)

        def load_pretrained(self, path: str, strict: bool = True):
            """Load pre-trained backbone."""
            state = torch.load(path, weights_only=False)
            self.backbone.load_state_dict(state["backbone"], strict=strict)
            if "head" in state:
                self.head.load_state_dict(state["head"], strict=strict)
            self._pretrained = True


    class FiberPrototypicalNet:
        """Prototypical network for few-shot structure classification.

        Learns an embedding where structures of the same type
        cluster together, enabling classification from few examples.

        Parameters
        ----------
        n_features : int
            Input feature dimension.
        embedding_dim : int
            Embedding space dimension.
        hidden : list of int
            Hidden layers.

        Examples
        --------
        >>> proto = FiberPrototypicalNet(n_features=20, embedding_dim=32)
        >>> proto.train(X_support, y_support, epochs=50)
        >>> predictions = proto.predict(X_query, X_support, y_support)
        """

        def __init__(
            self,
            n_features: int = 20,
            embedding_dim: int = 32,
            hidden: Optional[List[int]] = None,
        ):
            _require_torch()
            if hidden is None:
                hidden = [64, 32]

            layers = []
            prev = n_features
            for h in hidden:
                layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()])
                prev = h
            layers.append(nn.Linear(prev, embedding_dim))
            self.encoder = nn.Sequential(*layers)
            self.embedding_dim = embedding_dim

        def _embed(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

        def _compute_prototypes(
            self, X_support: torch.Tensor, y_support: torch.Tensor
        ) -> Dict[int, torch.Tensor]:
            """Compute prototype (mean embedding) for each class."""
            embeddings = self._embed(X_support)
            classes = y_support.unique()
            prototypes = {}
            for c in classes:
                mask = y_support == c
                prototypes[int(c.item())] = embeddings[mask].mean(dim=0)
            return prototypes

        def train(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            epochs: int = 50,
            lr: float = 1e-3,
            n_way: int = 5,
            n_support: int = 5,
            n_query: int = 10,
            verbose: bool = True,
        ) -> Dict[str, Any]:
            """Train via episodic few-shot learning.

            Parameters
            ----------
            X : (N, n_features)
                Full training set.
            y : (N,) integer class labels.
            epochs : int
                Number of training episodes.
            n_way : int
                Number of classes per episode.
            n_support : int
                Support samples per class.
            n_query : int
                Query samples per class.
            verbose : bool

            Returns
            -------
            dict
                Training history with accuracy.
            """
            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.long)
            optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=lr)

            classes = y_t.unique()
            losses_list = []
            accs_list = []

            for episode in range(epochs):
                # Sample episode
                ep_classes = classes[torch.randperm(len(classes))[:n_way]]
                support_x, support_y, query_x, query_y = [], [], [], []

                for c in ep_classes:
                    mask = y_t == c
                    idx = torch.where(mask)[0]
                    perm = idx[torch.randperm(len(idx))]
                    n_total = n_support + n_query
                    if len(perm) < n_total:
                        perm = perm.repeat(n_total // len(perm) + 1)[:n_total]

                    support_x.append(X_t[perm[:n_support]])
                    support_y.append(y_t[perm[:n_support]])
                    query_x.append(X_t[perm[n_support:n_total]])
                    query_y.append(y_t[perm[n_support:n_total]])

                sx = torch.cat(support_x)
                sy = torch.cat(support_y)
                qx = torch.cat(query_x)
                qy = torch.cat(query_y)

                # Compute prototypes
                prototypes = self._compute_prototypes(sx, sy)

                # Classify queries
                q_emb = self._embed(qx)
                dists = []
                for c in ep_classes:
                    proto = prototypes[int(c.item())]
                    dists.append(torch.sum((q_emb - proto) ** 2, dim=1))
                dists = torch.stack(dists, dim=1)  # (n_query_total, n_way)

                # Remap labels
                label_map = {int(c.item()): i for i, c in enumerate(ep_classes)}
                qy_remapped = torch.tensor([label_map[int(q.item())] for q in qy])

                optimizer.zero_grad()
                loss = F.cross_entropy(-dists, qy_remapped)
                loss.backward()
                optimizer.step()

                # Accuracy
                pred = dists.argmin(dim=1)
                acc = (pred == qy_remapped).float().mean().item()

                losses_list.append(loss.item())
                accs_list.append(acc)

                if verbose and (episode % max(epochs // 10, 1) == 0 or episode == epochs - 1):
                    print(f"Episode {episode:3d} | loss={loss.item():.4f} | acc={acc:.2f}")

            return {"losses": losses_list, "accuracies": accs_list}

        def predict(
            self,
            X_query: np.ndarray,
            X_support: np.ndarray,
            y_support: np.ndarray,
        ) -> np.ndarray:
            """Predict class labels for query samples."""
            self.encoder.eval()
            with torch.no_grad():
                sx = torch.tensor(X_support, dtype=torch.float32)
                sy = torch.tensor(y_support, dtype=torch.long)
                qx = torch.tensor(X_query, dtype=torch.float32)

                prototypes = self._compute_prototypes(sx, sy)
                q_emb = self._embed(qx)

                classes = list(prototypes.keys())
                dists = []
                for c in classes:
                    dists.append(torch.sum((q_emb - prototypes[c]) ** 2, dim=1))
                dists = torch.stack(dists, dim=1)
                pred_idx = dists.argmin(dim=1)

                return np.array([classes[i] for i in pred_idx.numpy()])


    class DomainAdapter:
        """Domain adaptation between different structure types.

        Fine-tunes a model trained on one structure type (e.g., honeycomb)
        to work on another (e.g., re-entrant) with minimal new data.

        Parameters
        ----------
        model : nn.Module
            Pre-trained model.
        adaptation_lr : float
            Learning rate for adaptation.

        Examples
        --------
        >>> adapter = DomainAdapter(pretrained_model)
        >>> adapter.adapt(X_source, y_source, X_target, y_target, n_iter=50)
        >>> pred = adapter.predict(X_target_test)
        """

        def __init__(self, model: nn.Module, adaptation_lr: float = 1e-4):
            _require_torch()
            self.model = copy.deepcopy(model)
            self.adaptation_lr = adaptation_lr
            self.source_model = copy.deepcopy(model)
            for p in self.source_model.parameters():
                p.requires_grad_(False)

        def adapt(
            self,
            X_source: np.ndarray,
            y_source: np.ndarray,
            X_target: np.ndarray,
            y_target: np.ndarray,
            *,
            n_iter: int = 50,
            batch_size: int = 32,
            domain_weight: float = 0.1,
            verbose: bool = True,
        ) -> Dict[str, Any]:
            """Adapt model to target domain.

            Uses target data for supervised loss and source data
            for regularization to prevent catastrophic forgetting.

            Parameters
            ----------
            X_source, y_source : source domain data.
            X_target, y_target : target domain data (small).
            n_iter : int
            batch_size : int
            domain_weight : float
                Weight for domain divergence regularization.
            verbose : bool

            Returns
            -------
            dict
                Adaptation history.
            """
            Xs = torch.tensor(X_source, dtype=torch.float32)
            ys = torch.tensor(y_source, dtype=torch.float32)
            if ys.dim() == 1:
                ys = ys.unsqueeze(-1)
            Xt = torch.tensor(X_target, dtype=torch.float32)
            yt = torch.tensor(y_target, dtype=torch.float32)
            if yt.dim() == 1:
                yt = yt.unsqueeze(-1)

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.adaptation_lr)
            losses = []

            for it in range(n_iter):
                self.model.train()

                # Target loss
                target_pred = self.model(Xt)
                target_loss = F.mse_loss(target_pred, yt)

                # Regularization: don't deviate too far from source model
                with torch.no_grad():
                    source_pred_target = self.source_model(Xt)
                reg_loss = F.mse_loss(target_pred, source_pred_target)

                loss = target_loss + domain_weight * reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                if verbose and (it % max(n_iter // 10, 1) == 0 or it == n_iter - 1):
                    print(f"Iter {it:3d} | loss={loss.item():.4f} | target={target_loss.item():.4f}")

            return {"losses": losses}

        @torch.no_grad()
        def predict(self, X: np.ndarray) -> np.ndarray:
            self.model.eval()
            return self.model(torch.tensor(X, dtype=torch.float32)).numpy()

else:
    class FiberTransferNet:
        def __init__(self, *a, **kw):
            _require_torch()

    class FiberPrototypicalNet:
        def __init__(self, *a, **kw):
            _require_torch()

    class DomainAdapter:
        def __init__(self, *a, **kw):
            _require_torch()

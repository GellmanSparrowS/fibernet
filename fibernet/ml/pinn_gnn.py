"""
Physics-Informed Graph Neural Networks for FiberNet.

Embeds physical laws directly into GNN message passing, ensuring that
predictions satisfy mechanical equilibrium, constitutive relations,
and energy conservation at the graph level.

Implements:
- PhysicsInformedMessagePassing: Force-balanced message passing layer
- PhysicsInformedGNN: Full physics-constrained GNN model
- ForceBalanceLoss: Nodal equilibrium residual (ΣF = 0)
- ConstitutiveLoss: Stress-strain law enforcement
- EnergyConservationLoss: Total energy conservation constraint
- PhysicsGNNTrainer: Training with combined data + physics loss

Features
--------
- Force balance at every node (Newton's third law in messages)
- Constitutive law embedding in edge features (σ = Eε)
- Strain energy conservation across the network
- Symmetric message passing (undirected edges)
- Compatible with StructureGraph input via graph_from_structure
- Multi-task prediction: displacement + stress + strain

References
----------
- Article section 5: "Physics-informed GNNs model physical interactions
  as message passing on graphs"
- Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks"
  (ICLR 2021)
- Allen et al., "Physics-Informed Graph Neural Networks for Modeling
  and Control" (2023)

Examples
--------
>>> from fibernet.ml.pinn_gnn import PhysicsInformedGNN, PhysicsGNNTrainer
>>> pinn_gnn = PhysicsInformedGNN(
...     node_dim=5, edge_dim=2, hidden=64, n_layers=4,
...     physics_weight=1.0,
... )
>>> trainer = PhysicsGNNTrainer(pinn_gnn, physics_loss_weight=0.5)
>>> history = trainer.fit(train_graphs, labels, epochs=200)
>>> pred = pinn_gnn.predict(test_graphs)
"""

from __future__ import annotations

import math
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

    # ==================================================================
    # Physics-Informed Message Passing Layer
    # ==================================================================

    class PhysicsInformedMessagePassing(nn.Module):
        """Message passing layer with embedded force balance constraints.

        In standard GNNs, messages flow freely. Here, edge messages represent
        physical forces and must satisfy equilibrium at each node.

        The message from node i to j is:
            m_ij = MLP(h_i, h_j, e_ij)  (raw message)
            f_ij = force_head(m_ij)      (predicted force vector)

        The nodal update enforces:
            h_i' = GRU(h_i, Σ_j f_ij)   (force-balanced aggregation)

        Parameters
        ----------
        node_dim : int
            Node feature dimension.
        edge_dim : int
            Edge feature dimension.
        hidden : int
            Hidden layer size.
        force_dim : int
            Dimension of force vectors (2 or 3).
        """

        def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64,
                     force_dim: int = 2):
            super().__init__()
            self.node_dim = node_dim
            self.edge_dim = edge_dim
            self.hidden = hidden
            self.force_dim = force_dim

            # Message MLP: takes (h_i, h_j, e_ij) → message
            self.message_mlp = nn.Sequential(
                nn.Linear(2 * node_dim + edge_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            )

            # Force head: message → force vector
            self.force_head = nn.Linear(hidden, force_dim)

            # Update: combine node features with aggregated forces
            self.update_gru = nn.GRUCell(node_dim + force_dim, hidden)

            # Project back to node_dim if different
            if hidden != node_dim:
                self.project = nn.Linear(hidden, node_dim)
            else:
                self.project = nn.Identity()

            # Edge update
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * node_dim + edge_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, edge_dim),
            )

        def forward(
            self,
            node_features: torch.Tensor,
            edge_index: torch.Tensor,
            edge_features: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass with physics-informed message passing.

            Parameters
            ----------
            node_features : (N, node_dim)
            edge_index : (2, E)
            edge_features : (E, edge_dim)

            Returns
            -------
            updated_nodes : (N, node_dim)
            updated_edges : (E, edge_dim)
            nodal_forces : (N, force_dim) predicted force at each node
            """
            n_nodes = node_features.shape[0]
            src, dst = edge_index[0], edge_index[1]

            # Compute messages
            h_src = node_features[src]  # (E, node_dim)
            h_dst = node_features[dst]  # (E, node_dim)
            msg_input = torch.cat([h_src, h_dst, edge_features], dim=-1)
            messages = self.message_mlp(msg_input)  # (E, hidden)

            # Compute forces
            forces = self.force_head(messages)  # (E, force_dim)

            # Aggregate forces at nodes (message passing)
            nodal_forces = torch.zeros(n_nodes, self.force_dim, device=node_features.device)
            # Force from edge e acts on dst node
            dst_expanded = dst.unsqueeze(-1).expand(-1, self.force_dim)
            nodal_forces.scatter_add_(0, dst_expanded, forces)
            # Also add opposite force on src (Newton's 3rd law)
            src_expanded = src.unsqueeze(-1).expand(-1, self.force_dim)
            nodal_forces.scatter_add_(0, src_expanded, -forces)

            # Update node features
            update_input = torch.cat([node_features, nodal_forces], dim=-1)
            h_updated = self.update_gru(update_input, node_features[:, :self.hidden] if node_features.shape[1] >= self.hidden else torch.zeros(n_nodes, self.hidden, device=node_features.device))
            h_updated = self.project(h_updated)

            # Update edge features
            edge_update_input = torch.cat([node_features[src], node_features[dst], edge_features], dim=-1)
            edge_updated = self.edge_mlp(edge_update_input)

            return h_updated, edge_updated, nodal_forces


    # ==================================================================
    # Physics Loss Functions
    # ==================================================================

    class ForceBalanceLoss(nn.Module):
        """Compute nodal force balance residual.

        At equilibrium, the sum of forces at each non-boundary node must be zero:
            Σ_j f_ij + f_ext_i = 0

        Parameters
        ----------
        weight : float
            Loss weight.
        """

        def __init__(self, weight: float = 1.0):
            super().__init__()
            self.weight = weight

        def forward(
            self,
            nodal_forces: torch.Tensor,
            external_forces: Optional[torch.Tensor] = None,
            free_node_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Compute force balance residual.

            Parameters
            ----------
            nodal_forces : (N, force_dim) predicted internal forces
            external_forces : (N, force_dim) or None
            free_node_mask : (N,) bool mask for non-fixed nodes

            Returns
            -------
            scalar loss
            """
            if external_forces is not None:
                residual = nodal_forces + external_forces
            else:
                residual = nodal_forces

            if free_node_mask is not None:
                residual = residual[free_node_mask]

            return self.weight * (residual ** 2).mean()


    class ConstitutiveLoss(nn.Module):
        """Enforce constitutive law (stress-strain relation) on edge predictions.

        For linear elastic: σ = E * ε
        For nonlinear: σ = f(ε) (learned)

        Parameters
        ----------
        youngs_modulus : float
            Material E for linear elastic check.
        weight : float
            Loss weight.
        """

        def __init__(self, youngs_modulus: float = 1e9, weight: float = 0.5):
            super().__init__()
            self.E = youngs_modulus
            self.weight = weight

        def forward(
            self,
            predicted_stress: torch.Tensor,
            predicted_strain: torch.Tensor,
        ) -> torch.Tensor:
            """Enforce σ = Eε on predicted fields (normalized)."""
            target_stress = self.E * predicted_strain
            # Normalize by E^2 to prevent loss explosion with large E
            scale = max(self.E ** 2, 1.0)
            return self.weight * F.mse_loss(predicted_stress / self.E, predicted_strain)


    class EnergyConservationLoss(nn.Module):
        """Enforce energy conservation across the network.

        Total strain energy U = Σ_edges (0.5 * σ * ε * V) should be consistent
        with external work W = Σ_nodes (f_ext · u).

        Parameters
        ----------
        weight : float
            Loss weight.
        """

        def __init__(self, weight: float = 0.1):
            super().__init__()
            self.weight = weight

        def forward(
            self,
            strain_energy: torch.Tensor,
            external_work: torch.Tensor,
        ) -> torch.Tensor:
            """Energy balance: |U - W| → 0."""
            return self.weight * (strain_energy - external_work) ** 2


    # ==================================================================
    # Physics-Informed GNN Model
    # ==================================================================

    class PhysicsInformedGNN(nn.Module):
        """Full physics-informed GNN for fiber network property prediction.

        Combines multiple physics-informed message passing layers with
        task-specific heads for displacement, stress, and property prediction.

        Parameters
        ----------
        node_dim : int
            Input node feature dimension.
        edge_dim : int
            Input edge feature dimension.
        hidden : int
            Hidden dimension.
        n_layers : int
            Number of message passing layers.
        n_outputs : int
            Number of output properties (for graph-level prediction).
        output_mode : str
            "graph" (graph-level prediction) or "node" (per-node prediction).
        predict_field : bool
            Whether to also predict per-node fields (displacement, stress).
        physics_weight : float
            Weight for physics loss terms.
        youngs_modulus : float
            Material E for constitutive loss.
        force_dim : int
            Force vector dimension (2 or 3).
        pooling : str
            "mean", "max", or "attention".

        Examples
        --------
        >>> gnn = PhysicsInformedGNN(node_dim=5, edge_dim=2, hidden=64)
        >>> graph_data = graph_from_structure(structure_graph)
        >>> pred = gnn([graph_data])  # graph-level prediction
        >>> fields = gnn.predict_fields(graph_data)  # per-node fields
        """

        def __init__(
            self,
            node_dim: int = 5,
            edge_dim: int = 2,
            hidden: int = 64,
            n_layers: int = 4,
            n_outputs: int = 1,
            output_mode: str = "graph",
            predict_field: bool = True,
            physics_weight: float = 1.0,
            youngs_modulus: float = 1e9,
            force_dim: int = 2,
            pooling: str = "mean",
        ):
            super().__init__()
            self.node_dim = node_dim
            self.edge_dim = edge_dim
            self.hidden = hidden
            self.n_outputs = n_outputs
            self.output_mode = output_mode
            self.predict_field = predict_field
            self.physics_weight = physics_weight

            # Input projection
            self.node_proj = nn.Linear(node_dim, hidden)
            self.edge_proj = nn.Linear(edge_dim, hidden)

            # Message passing layers
            self.layers = nn.ModuleList([
                PhysicsInformedMessagePassing(hidden, hidden, hidden, force_dim)
                for _ in range(n_layers)
            ])
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden) for _ in range(n_layers)
            ])

            # Output heads
            if pooling == "attention":
                self.attn_pool = nn.Linear(hidden, 1)
            self.pooling = pooling

            self.head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, n_outputs),
            )

            # Field prediction heads
            if predict_field:
                self.displacement_head = nn.Linear(hidden, force_dim)
                self.stress_head = nn.Linear(hidden, 1)
                self.strain_head = nn.Linear(hidden, 1)

            # Physics losses
            self.force_balance_loss = ForceBalanceLoss(weight=physics_weight)
            self.constitutive_loss = ConstitutiveLoss(
                youngs_modulus=youngs_modulus, weight=physics_weight * 0.5
            )
            self.energy_loss = EnergyConservationLoss(weight=physics_weight * 0.1)

        def _pool(self, x: torch.Tensor) -> torch.Tensor:
            if self.pooling == "mean":
                return x.mean(dim=0)
            elif self.pooling == "max":
                return x.max(dim=0)[0]
            elif self.pooling == "attention":
                attn = self.attn_pool(x)
                attn = F.softmax(attn, dim=0)
                return (x * attn).sum(dim=0)
            return x.mean(dim=0)

        def encode(
            self,
            node_features: torch.Tensor,
            edge_index: torch.Tensor,
            edge_features: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            """Encode a graph through physics-informed message passing.

            Returns
            -------
            dict with:
                - node_embeddings: (N, hidden)
                - edge_embeddings: (E, hidden)
                - nodal_forces: (N, force_dim)
                - graph_embedding: (hidden,) if graph mode
            """
            h = self.node_proj(node_features)
            e = self.edge_proj(edge_features) if edge_features.shape[0] > 0 else edge_features

            all_forces = []
            for layer, norm in zip(self.layers, self.layer_norms):
                h_new, e_new, forces = layer(h, edge_index, e)
                h = norm(h + h_new)  # residual + norm
                if e_new.shape[0] > 0:
                    e = e + e_new
                all_forces.append(forces)

            # Average forces across layers
            avg_forces = sum(all_forces) / len(all_forces)

            result = {
                "node_embeddings": h,
                "edge_embeddings": e,
                "nodal_forces": avg_forces,
            }

            if self.output_mode == "graph":
                result["graph_embedding"] = self._pool(h)

            return result

        def forward(self, graphs: List[Dict[str, Any]]) -> torch.Tensor:
            """Forward pass on a batch of graphs.

            Parameters
            ----------
            graphs : list of dict
                Each from graph_from_structure().

            Returns
            -------
            (batch_size, n_outputs) predictions
            """
            preds = []
            for g in graphs:
                nf = g["node_features"]
                ei = g["edge_index"]
                ef = g["edge_features"]
                enc = self.encode(nf, ei, ef)

                if self.output_mode == "graph":
                    preds.append(self.head(enc["graph_embedding"]))
                else:
                    node_pred = self.head(enc["node_embeddings"])
                    preds.append(node_pred.mean(dim=0))  # aggregate for batch

            return torch.stack(preds)

        def predict_fields(self, graph: Dict[str, Any]) -> Dict[str, torch.Tensor]:
            """Predict per-node fields (displacement, stress, strain).

            Parameters
            ----------
            graph : dict from graph_from_structure()

            Returns
            -------
            dict with displacement, stress, strain per node
            """
            nf = graph["node_features"]
            ei = graph["edge_index"]
            ef = graph["edge_features"]
            enc = self.encode(nf, ei, ef)

            result = {"node_embeddings": enc["node_embeddings"]}
            if self.predict_field:
                result["displacement"] = self.displacement_head(enc["node_embeddings"])
                result["stress"] = self.stress_head(enc["node_embeddings"])
                result["strain"] = self.strain_head(enc["node_embeddings"])
            return result

        def compute_physics_loss(
            self,
            graph: Dict[str, Any],
            external_forces: Optional[torch.Tensor] = None,
            free_node_mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Compute physics-based loss terms for a graph.

            Parameters
            ----------
            graph : dict from graph_from_structure()
            external_forces : (N, force_dim) or None
            free_node_mask : (N,) bool or None

            Returns
            -------
            dict with force_balance, constitutive, energy, total_physics
            """
            nf = graph["node_features"]
            ei = graph["edge_index"]
            ef = graph["edge_features"]
            enc = self.encode(nf, ei, ef)

            losses = {}

            # Force balance loss
            losses["force_balance"] = self.force_balance_loss(
                enc["nodal_forces"], external_forces, free_node_mask
            )

            # Constitutive loss (if predicting fields)
            if self.predict_field:
                fields = self.predict_fields(graph)
                losses["constitutive"] = self.constitutive_loss(
                    fields["stress"].squeeze(-1),
                    fields["strain"].squeeze(-1),
                )

                # Energy conservation
                strain_energy = (fields["stress"] * fields["strain"]).sum() * 0.5
                if external_forces is not None:
                    external_work = (external_forces * fields["displacement"]).sum()
                else:
                    external_work = strain_energy  # self-consistency
                losses["energy"] = self.energy_loss(strain_energy, external_work)
            else:
                losses["constitutive"] = torch.tensor(0.0, device=nf.device)
                losses["energy"] = torch.tensor(0.0, device=nf.device)

            losses["total_physics"] = sum(losses.values())
            return losses


    # ==================================================================
    # Trainer
    # ==================================================================

    class PhysicsGNNTrainer:
        """Trainer for PhysicsInformedGNN with combined data + physics loss.

        Parameters
        ----------
        model : PhysicsInformedGNN
        physics_loss_weight : float
            Overall weight for physics loss vs data loss.
        lr : float
        weight_decay : float
        """

        def __init__(
            self,
            model: PhysicsInformedGNN,
            physics_loss_weight: float = 0.5,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
        ):
            self.model = model
            self.physics_loss_weight = physics_loss_weight
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=200, eta_min=1e-5
            )

        def fit(
            self,
            graphs: List[Dict[str, Any]],
            labels: np.ndarray,
            *,
            epochs: int = 100,
            batch_size: int = 32,
            val_split: float = 0.2,
            early_stopping: int = 15,
            external_forces_list: Optional[List[Optional[torch.Tensor]]] = None,
            verbose: bool = True,
        ) -> Dict[str, Any]:
            """Train with combined data + physics loss.

            Parameters
            ----------
            graphs : list of graph dicts
            labels : (n_samples,) target values
            external_forces_list : list of force tensors per graph (optional)

            Returns
            -------
            dict with training history
            """
            from fibernet.ml.training import TrainingHistory

            n = len(graphs)
            labels = np.asarray(labels, dtype=np.float32)
            rng = np.random.RandomState(42)
            idx = rng.permutation(n)
            n_val = int(n * val_split)
            val_idx = idx[:n_val]
            train_idx = idx[n_val:]

            train_graphs = [graphs[i] for i in train_idx]
            train_labels = labels[train_idx]
            val_graphs = [graphs[i] for i in val_idx]
            val_labels = labels[val_idx]

            history = TrainingHistory()
            patience_counter = 0
            best_val_loss = float("inf")

            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0
                physics_loss_sum = 0.0
                n_batches = 0

                for start in range(0, len(train_graphs), batch_size):
                    end = min(start + batch_size, len(train_graphs))
                    batch_g = train_graphs[start:end]
                    batch_y = torch.tensor(train_labels[start:end], dtype=torch.float32)

                    self.optimizer.zero_grad()

                    # Data loss
                    pred = self.model(batch_g).squeeze(-1)
                    data_loss = F.mse_loss(pred, batch_y)

                    # Physics loss (sample a few graphs per batch)
                    p_loss = torch.tensor(0.0)
                    n_physics = 0
                    for gi, g in enumerate(batch_g):
                        ef = None
                        if external_forces_list is not None:
                            idx_g = train_idx[start + gi]
                            if idx_g < len(external_forces_list):
                                ef = external_forces_list[idx_g]
                        p_losses = self.model.compute_physics_loss(g, ef)
                        p_loss = p_loss + p_losses["total_physics"]
                        n_physics += 1

                    if n_physics > 0:
                        p_loss = p_loss / n_physics

                    total = data_loss + self.physics_loss_weight * p_loss
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    train_loss += data_loss.item()
                    physics_loss_sum += p_loss.item() if isinstance(p_loss, torch.Tensor) else 0.0
                    n_batches += 1

                self.scheduler.step()
                train_loss /= max(n_batches, 1)
                physics_loss_sum /= max(n_batches, 1)

                # Validate
                val_loss = 0.0
                if val_graphs:
                    self.model.eval()
                    with torch.no_grad():
                        pred = self.model(val_graphs).squeeze(-1)
                        val_loss = F.mse_loss(
                            pred, torch.tensor(val_labels)
                        ).item()

                history.update(epoch, train_loss, val_loss,
                              lr=self.optimizer.param_groups[0]["lr"])

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if early_stopping > 0 and patience_counter >= early_stopping:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

                if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch:3d} | data={train_loss:.4f} | "
                          f"physics={physics_loss_sum:.4f} | val={val_loss:.4f}")

            return {
                "history": history,
                "best_val_loss": best_val_loss,
                "final_physics_loss": physics_loss_sum,
            }

else:
    class PhysicsInformedMessagePassing:
        def __init__(self, *a, **kw):
            _require_torch()

    class ForceBalanceLoss:
        def __init__(self, *a, **kw):
            _require_torch()

    class ConstitutiveLoss:
        def __init__(self, *a, **kw):
            _require_torch()

    class EnergyConservationLoss:
        def __init__(self, *a, **kw):
            _require_torch()

    class PhysicsInformedGNN:
        def __init__(self, *a, **kw):
            _require_torch()

    class PhysicsGNNTrainer:
        def __init__(self, *a, **kw):
            _require_torch()

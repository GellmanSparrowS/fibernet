"""
Graph Neural Networks for FiberNet Structure-Property Prediction.

Implements GCN, GAT, GraphSAGE in pure PyTorch with optional PyG support.
Operates directly on StructureGraph objects.

Models
------
- GraphConvLayer: Basic graph convolution (GCN-style)
- GraphAttentionLayer: Graph attention (GAT-style)
- GraphSAGELayer: GraphSAGE aggregation
- FiberGNN: Configurable GNN for graph-level property prediction
- FiberGAT: Multi-head attention GNN

Features
--------
- Direct StructureGraph → GNN input (no manual conversion)
- Global pooling (mean, max, attention)
- Optional edge features
- Batch processing with sparse adjacency

Examples
--------
>>> from fibernet.ml.gnn import FiberGNN, graph_from_structure
>>> from fibernet import pattern_2d
>>> g = pattern_2d("honeycomb", box=(10,10), grid=(4,4))
>>> graph_data = graph_from_structure(g)
>>> model = FiberGNN(
...     node_dim=graph_data["node_features"].shape[1],
...     hidden=64, n_outputs=1, n_layers=3,
... )
>>> pred = model([graph_data])  # list for batch
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
# Graph Conversion
# ======================================================================

def graph_from_structure(
    g,
    node_features: Optional[List[str]] = None,
    edge_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convert StructureGraph to GNN-ready dictionary.

    Parameters
    ----------
    g : StructureGraph
        Input structure graph.
    node_features : list of str, optional
        Node feature types: "position", "degree", "boundary", "coord_count".
        Default: all available.
    edge_features : list of str, optional
        Edge feature types: "length", "angle", "radius".
        Default: all available.

    Returns
    -------
    dict
        Keys: node_features (N, F_n), edge_index (2, E), edge_features (E, F_e),
        adjacency (sparse N×N), n_nodes, n_edges.
    """
    _require_torch()

    node_ids = sorted(g.nodes.keys())
    node_map = {nid: i for i, nid in enumerate(node_ids)}
    n_nodes = len(node_ids)

    # Node features
    nf = []
    for nid in node_ids:
        node = g.nodes[nid]
        feats = []
        feats.extend(node.position[:2].tolist())  # x, y
        feats.append(float(g.degree(nid)))

        # Boundary flag
        try:
            boundary_nodes = set(g.get_boundary_nodes())
            feats.append(1.0 if nid in boundary_nodes else 0.0)
        except Exception:
            feats.append(0.0)

        # Coordination number (connected edges)
        feats.append(float(g.degree(nid)))
        nf.append(feats)

    node_features_tensor = torch.tensor(nf, dtype=torch.float32)

    # Edge index and features
    src_list, dst_list = [], []
    ef_list = []
    for edge_id, edge in g.edges.items():
        src = node_map.get(edge.node_i)
        dst = node_map.get(edge.node_j)
        if src is not None and dst is not None:
            src_list.extend([src, dst])
            dst_list.extend([dst, src])

            # Edge features
            ef = []
            pos_a = g.nodes[edge.node_i].position[:2]
            pos_b = g.nodes[edge.node_j].position[:2]
            length = float(np.linalg.norm(pos_b - pos_a))
            ef.append(length)
            ef.append(float(getattr(edge, "radius", 0.1)))
            ef_list.append(ef)
            ef_list.append(ef)  # symmetric

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_features_tensor = torch.tensor(ef_list, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_features_tensor = torch.zeros((0, 2), dtype=torch.float32)

    # Normalized adjacency (for GCN)
    adjacency = _build_normalized_adjacency(n_nodes, edge_index)

    return {
        "node_features": node_features_tensor,
        "edge_index": edge_index,
        "edge_features": edge_features_tensor,
        "adjacency": adjacency,
        "n_nodes": n_nodes,
        "n_edges": len(g.edges),
    }


def _build_normalized_adjacency(n_nodes: int, edge_index: torch.Tensor):
    """Build symmetric normalized adjacency D^{-1/2} A D^{-1/2} as sparse tensor."""
    if edge_index.shape[1] == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros(0),
            (n_nodes, n_nodes),
        )

    row, col = edge_index[0], edge_index[1]

    # Add self-loops
    all_row = torch.cat([row, torch.arange(n_nodes)])
    all_col = torch.cat([col, torch.arange(n_nodes)])

    # Degree
    deg = torch.zeros(n_nodes)
    deg.scatter_add_(0, all_row, torch.ones(all_row.shape[0], dtype=torch.float32))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

    # Normalized weights
    values = deg_inv_sqrt[all_row] * deg_inv_sqrt[all_col]

    indices = torch.stack([all_row, all_col])
    return torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes))


# ======================================================================
# GNN Layers (pure PyTorch)
# ======================================================================

if HAS_TORCH:

    class GraphConvLayer(nn.Module):
        """Graph convolution layer (GCN-style).

        h' = σ(D^{-1/2} A D^{-1/2} h W)

        Parameters
        ----------
        in_dim : int
            Input feature dimension.
        out_dim : int
            Output feature dimension.
        bias : bool
            Whether to add bias.
        """

        def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
            self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
            nn.init.xavier_uniform_(self.weight)

        def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x : (N, in_dim) node features
            adj : (N, N) sparse normalized adjacency

            Returns
            -------
            (N, out_dim) updated features
            """
            support = x @ self.weight
            output = torch.sparse.mm(adj, support)
            if self.bias is not None:
                output = output + self.bias
            return output


    class GraphAttentionLayer(nn.Module):
        """Graph attention layer (GAT-style).

        Computes attention-weighted message passing.

        Parameters
        ----------
        in_dim : int
            Input feature dimension.
        out_dim : int
            Output feature dimension.
        n_heads : int
            Number of attention heads.
        dropout : float
            Dropout on attention weights.
        """

        def __init__(
            self,
            in_dim: int,
            out_dim: int,
            n_heads: int = 4,
            dropout: float = 0.1,
            concat: bool = True,
        ):
            super().__init__()
            self.n_heads = n_heads
            self.out_dim = out_dim
            self.concat = concat

            self.W = nn.Parameter(torch.empty(n_heads, in_dim, out_dim))
            self.a_src = nn.Parameter(torch.empty(n_heads, out_dim, 1))
            self.a_dst = nn.Parameter(torch.empty(n_heads, out_dim, 1))

            nn.init.xavier_uniform_(self.W)
            nn.init.xavier_uniform_(self.a_src)
            nn.init.xavier_uniform_(self.a_dst)

            self.leaky_relu = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x : (N, in_dim)
            edge_index : (2, E)

            Returns
            -------
            (N, out_dim * n_heads) if concat, else (N, out_dim)
            """
            N = x.shape[0]
            src, dst = edge_index[0], edge_index[1]

            # Project: (N, in_dim) → (n_heads, N, out_dim)
            h = torch.einsum("ni,hio->hno", x, self.W)

            # Attention scores
            attn_src = (h * self.a_src.squeeze(-1).unsqueeze(1)).sum(dim=-1)  # (H, N)
            attn_dst = (h * self.a_dst.squeeze(-1).unsqueeze(1)).sum(dim=-1)  # (H, N)

            # Edge attention: a_ij = leaky_relu(a_src_i + a_dst_j)
            edge_attn = self.leaky_relu(attn_src[:, src] + attn_dst[:, dst])  # (H, E)

            # Softmax per destination node
            attn_weights = torch.full((self.n_heads, N), float("-inf"))
            # Scatter-based softmax
            for h_idx in range(self.n_heads):
                max_vals = torch.full((N,), float("-inf"))
                max_vals.scatter_reduce_(0, dst, edge_attn[h_idx], reduce="amax", include_self=False)
                exp_attn = torch.exp(edge_attn[h_idx] - max_vals[dst])
                sum_exp = torch.zeros(N).scatter_add_(0, dst, exp_attn)
                attn_weights[h_idx] = float("-inf")  # will be filled per node

            # Simpler approach: softmax per node via scatter
            attn_weights = self._sparse_softmax(edge_attn, dst, N)
            attn_weights = self.dropout(attn_weights)

            # Message passing
            msg = h[:, src, :]  # (H, E, out_dim)
            weighted_msg = msg * attn_weights.unsqueeze(-1)  # (H, E, out_dim)

            # Aggregate to destination
            out = torch.zeros(self.n_heads, N, self.out_dim)
            out.scatter_add_(1, dst.unsqueeze(0).unsqueeze(-1).expand(self.n_heads, -1, self.out_dim),
                            weighted_msg)

            if self.concat:
                return out.permute(1, 0, 2).reshape(N, -1)
            else:
                return out.mean(dim=0)

        def _sparse_softmax(self, scores, index, n_nodes):
            """Sparse softmax: softmax grouped by destination node."""
            H, E = scores.shape

            # Max per node
            max_vals = torch.full((H, n_nodes), float("-inf"))
            max_vals.scatter_reduce_(1, index.unsqueeze(0).expand(H, -1), scores,
                                     reduce="amax", include_self=False)
            max_per_edge = max_vals.gather(1, index.unsqueeze(0).expand(H, -1))

            exp_scores = torch.exp(scores - max_per_edge)

            # Sum per node
            sum_exp = torch.zeros(H, n_nodes).scatter_add_(
                1, index.unsqueeze(0).expand(H, -1), exp_scores,
            )
            sum_per_edge = sum_exp.gather(1, index.unsqueeze(0).expand(H, -1))

            return exp_scores / (sum_per_edge + 1e-12)


    class GraphSAGELayer(nn.Module):
        """GraphSAGE layer with mean aggregation.

        Parameters
        ----------
        in_dim : int
            Input dimension.
        out_dim : int
            Output dimension.
        aggregator : str
            "mean", "max", "sum"
        """

        def __init__(self, in_dim: int, out_dim: int, aggregator: str = "mean"):
            super().__init__()
            self.aggregator = aggregator
            self.self_weight = nn.Linear(in_dim, out_dim, bias=False)
            self.neigh_weight = nn.Linear(in_dim, out_dim, bias=False)
            self.bias = nn.Parameter(torch.zeros(out_dim))

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            N = x.shape[0]
            src, dst = edge_index[0], edge_index[1]

            # Aggregate neighbor features
            neigh_feat = torch.zeros(N, x.shape[1])
            if self.aggregator == "mean":
                neigh_feat.scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.shape[1]), x[src])
                # Count neighbors
                count = torch.zeros(N).scatter_add_(0, dst, torch.ones(src.shape[0]))
                count = count.clamp(min=1).unsqueeze(-1)
                neigh_feat = neigh_feat / count
            elif self.aggregator == "sum":
                neigh_feat.scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.shape[1]), x[src])
            elif self.aggregator == "max":
                neigh_feat.fill_(float("-inf"))
                neigh_feat.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, x.shape[1]),
                                          x[src], reduce="amax", include_self=False)
                neigh_feat[neigh_feat == float("-inf")] = 0.0

            out = self.self_weight(x) + self.neigh_weight(neigh_feat) + self.bias
            # L2 normalize
            out = F.normalize(out, p=2, dim=-1)
            return out


    # ======================================================================
    # GNN Models
    # ======================================================================

    class FiberGNN(nn.Module):
        """Configurable GNN for graph-level property prediction.

        Parameters
        ----------
        node_dim : int
            Input node feature dimension.
        hidden : int
            Hidden dimension.
        n_outputs : int
            Number of output predictions.
        n_layers : int
            Number of GNN layers.
        layer_type : str
            "gcn", "gat", "sage"
        pooling : str
            "mean", "max", "attention"
        dropout : float
            Dropout rate.

        Examples
        --------
        >>> model = FiberGNN(node_dim=5, hidden=64, n_outputs=3, n_layers=3)
        >>> graph_data = graph_from_structure(structure_graph)
        >>> pred = model([graph_data])
        """

        def __init__(
            self,
            node_dim: int,
            hidden: int = 64,
            n_outputs: int = 1,
            n_layers: int = 3,
            layer_type: str = "gcn",
            pooling: str = "mean",
            dropout: float = 0.1,
            n_heads: int = 4,
        ):
            _require_torch()
            super().__init__()

            self.layer_type = layer_type
            self.pooling = pooling
            self.n_layers = n_layers

            # Input projection
            self.input_proj = nn.Linear(node_dim, hidden)

            # GNN layers
            self.layers = nn.ModuleList()
            self.norms = nn.ModuleList()

            for i in range(n_layers):
                in_d = hidden
                if layer_type == "gcn":
                    self.layers.append(GraphConvLayer(in_d, hidden))
                elif layer_type == "gat":
                    out_d = hidden // n_heads if i < n_layers - 1 else hidden
                    self.layers.append(GraphAttentionLayer(
                        in_d, out_d, n_heads=n_heads,
                        concat=(i < n_layers - 1),
                    ))
                    hidden_actual = out_d * n_heads if i < n_layers - 1 else hidden
                elif layer_type == "sage":
                    self.layers.append(GraphSAGELayer(in_d, hidden))
                else:
                    raise ValueError(f"Unknown layer_type: {layer_type}")

                self.norms.append(nn.LayerNorm(hidden))

            # Attention pooling
            if pooling == "attention":
                self.attn_pool = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, 1),
                )

            # Output head
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, n_outputs),
            )

            self.dropout = nn.Dropout(dropout)

        def forward(self, graphs: List[Dict[str, Any]]) -> torch.Tensor:
            """Forward pass over a batch of graph data dicts.

            Parameters
            ----------
            graphs : list of dict
                Each from graph_from_structure().

            Returns
            -------
            (batch_size, n_outputs) predictions
            """
            graph_embeds = []
            for gd in graphs:
                x = gd["node_features"]
                adj = gd["adjacency"]
                edge_index = gd["edge_index"]

                x = F.relu(self.input_proj(x))
                x = self.dropout(x)

                for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
                    if self.layer_type == "gcn":
                        h = F.relu(layer(x, adj))
                    elif self.layer_type == "gat":
                        h = F.elu(layer(x, edge_index))
                    elif self.layer_type == "sage":
                        h = F.relu(layer(x, edge_index))
                    else:
                        h = layer(x, adj)

                    h = norm(h)
                    h = self.dropout(h)

                    # Residual
                    if h.shape == x.shape:
                        x = x + h
                    else:
                        x = h

                # Pool
                embed = self._pool(x)
                graph_embeds.append(embed)

            batch = torch.stack(graph_embeds)
            return self.head(batch)

        def _pool(self, x: torch.Tensor) -> torch.Tensor:
            """Global pooling over nodes."""
            if self.pooling == "mean":
                return x.mean(dim=0)
            elif self.pooling == "max":
                return x.max(dim=0)[0]
            elif self.pooling == "attention":
                attn = self.attn_pool(x)
                attn = F.softmax(attn, dim=0)
                return (x * attn).sum(dim=0)
            else:
                return x.mean(dim=0)


    def train_gnn(
        model: FiberGNN,
        graphs: List[Dict[str, Any]],
        labels: np.ndarray,
        *,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        val_split: float = 0.2,
        early_stopping: int = 15,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train a GNN model on graph data.

        Parameters
        ----------
        model : FiberGNN
            GNN model.
        graphs : list of dict
            Graph data from graph_from_structure().
        labels : array-like
            Target values (n_samples,).
        epochs : int
            Max training epochs.
        lr : float
            Learning rate.
        batch_size : int
            Batch size (number of graphs per batch).
        val_split : float
            Validation fraction.
        early_stopping : int
            Patience for early stopping.
        verbose : bool
            Print progress.

        Returns
        -------
        dict
            Training history.
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

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        history = TrainingHistory()
        patience_counter = 0
        best_val_loss = float("inf")

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            n_batches = 0
            for start in range(0, len(train_graphs), batch_size):
                end = min(start + batch_size, len(train_graphs))
                batch_graphs = train_graphs[start:end]
                batch_labels = torch.tensor(train_labels[start:end], dtype=torch.float32)

                optimizer.zero_grad()
                pred = model(batch_graphs).squeeze(-1)
                loss = F.mse_loss(pred, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(n_batches, 1)

            # Validate
            val_loss = 0.0
            if val_graphs:
                model.eval()
                with torch.no_grad():
                    pred = model(val_graphs).squeeze(-1)
                    val_loss = F.mse_loss(pred, torch.tensor(val_labels)).item()

            history.update(epoch, train_loss, val_loss,
                          lr=optimizer.param_groups[0]["lr"])

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
                print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        return {"history": history, "best_val_loss": best_val_loss}


else:
    class FiberGNN:
        def __init__(self, *a, **kw):
            _require_torch()

    class GraphConvLayer:
        def __init__(self, *a, **kw):
            _require_torch()

    class GraphAttentionLayer:
        def __init__(self, *a, **kw):
            _require_torch()

    class GraphSAGELayer:
        def __init__(self, *a, **kw):
            _require_torch()

    def graph_from_structure(*a, **kw):
        _require_torch()
        return {}

    def train_gnn(*a, **kw):
        _require_torch()

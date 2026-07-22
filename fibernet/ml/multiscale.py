"""
Multi-Scale Learning for FiberNet — Hierarchical Feature Extraction.

Implements multi-scale feature extraction and learning:
- MultiScaleFeatureExtractor: Extract features at multiple spatial scales
- HierarchicalEncoder: Hierarchical neural network for multi-scale encoding
- ScaleBridgeModel: Bridge micro→macro predictions via homogenization
- MultiScaleTrainer: Training loop for multi-scale models

Features
--------
- Coarse-graining at multiple resolution levels
- Homogenization theory integration (RVE-based)
- Cross-scale information transfer
- Compatible with StructureGraph inputs

References
----------
- Article section 5.3: Multi-scale physics-informed learning
- Storm et al., "Microstructure-based GNN for multiscale simulations" (2024)

Examples
--------
>>> from fibernet.ml.multiscale import MultiScaleFeatureExtractor, HierarchicalEncoder
>>> extractor = MultiScaleFeatureExtractor(scales=[1.0, 2.0, 4.0])
>>> features = extractor.extract(graph)  # multi-scale feature dict
>>> encoder = HierarchicalEncoder(n_features=60, n_outputs=1)
>>> pred = encoder(torch.tensor(features))
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import math

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


class MultiScaleFeatureExtractor:
    """Extract features at multiple spatial scales from fiber networks.

    Parameters
    ----------
    scales : list of float
        Spatial scale factors (1.0 = original, 2.0 = 2x coarser).
    rve_method : str
        "uniform" (uniform grid), "random" (random RVE positions).
    n_rve_per_scale : int
        Number of Representative Volume Elements per scale.

    Examples
    --------
    >>> extractor = MultiScaleFeatureExtractor(scales=[1.0, 2.0, 4.0])
    >>> features = extractor.extract(graph)
    >>> # Returns {"scale_1.0": {...}, "scale_2.0": {...}, ...}
    """

    def __init__(
        self,
        scales: Optional[List[float]] = None,
        rve_method: str = "uniform",
        n_rve_per_scale: int = 4,
    ):
        if scales is None:
            scales = [1.0, 2.0, 4.0]
        self.scales = scales
        self.rve_method = rve_method
        self.n_rve_per_scale = n_rve_per_scale

    def extract(self, g) -> Dict[str, Any]:
        """Extract multi-scale features from a StructureGraph.

        Parameters
        ----------
        g : StructureGraph

        Returns
        -------
        dict
            Multi-scale features flattened into a single vector, plus per-scale breakdowns.
        """
        try:
            bb_min, bb_max = g.bounding_box()
            span = bb_max - bb_min
        except Exception:
            bb_min = np.zeros(2)
            span = np.ones(2) * 10.0

        all_features = {}

        for scale in self.scales:
            rve_size = span / scale
            scale_features = self._extract_at_scale(g, bb_min, rve_size)
            scale_key = f"scale_{scale:.1f}"
            all_features[scale_key] = scale_features

        # Flatten into single feature vector
        flat = {}
        for scale_key, feats in all_features.items():
            for feat_name, feat_val in feats.items():
                flat[f"{scale_key}__{feat_name}"] = feat_val

        # Cross-scale features
        flat["scale_ratio_max_min"] = max(self.scales) / min(self.scales)
        flat["n_scales"] = len(self.scales)

        # Add global features
        flat["global__n_nodes"] = g.num_nodes
        flat["global__n_edges"] = g.num_edges
        flat["global__density"] = g.num_edges / max(g.num_nodes, 1)

        try:
            degrees = [g.degree(nid) for nid in g.nodes]
            flat["global__mean_degree"] = float(np.mean(degrees)) if degrees else 0.0
            flat["global__std_degree"] = float(np.std(degrees)) if degrees else 0.0
        except Exception:
            flat["global__mean_degree"] = 0.0
            flat["global__std_degree"] = 0.0

        return flat

    def _extract_at_scale(self, g, bb_min, rve_size) -> Dict[str, float]:
        """Extract features within a single RVE at given scale."""
        features = {}
        rve_center = bb_min[:2] + rve_size[:2] / 2
        half_size = rve_size[:2] / 2

        # Count nodes/edges in RVE
        nodes_in_rve = []
        for nid, node in g.nodes.items():
            pos = node.position[:2]
            if (np.abs(pos - rve_center) <= half_size).all():
                nodes_in_rve.append(nid)

        features["n_nodes"] = len(nodes_in_rve)

        edges_in_rve = 0
        edge_lengths = []
        for eid, edge in g.edges.items():
            if edge.node_i in nodes_in_rve or edge.node_j in nodes_in_rve:
                edges_in_rve += 1
                try:
                    pi = g.nodes[edge.node_i].position[:2]
                    pj = g.nodes[edge.node_j].position[:2]
                    edge_lengths.append(float(np.linalg.norm(pj - pi)))
                except Exception:
                    pass

        features["n_edges"] = edges_in_rve
        features["density"] = edges_in_rve / max(len(nodes_in_rve), 1)

        if edge_lengths:
            features["mean_edge_length"] = float(np.mean(edge_lengths))
            features["std_edge_length"] = float(np.std(edge_lengths))
        else:
            features["mean_edge_length"] = 0.0
            features["std_edge_length"] = 0.0

        # Degree statistics
        if nodes_in_rve:
            degrees = []
            for nid in nodes_in_rve:
                try:
                    degrees.append(g.degree(nid))
                except Exception:
                    degrees.append(0)
            features["mean_degree"] = float(np.mean(degrees))
            features["max_degree"] = float(np.max(degrees)) if degrees else 0.0
        else:
            features["mean_degree"] = 0.0
            features["max_degree"] = 0.0

        # Area / volume
        rve_area = float(np.prod(rve_size[:2]))
        features["rve_area"] = rve_area
        total_length = sum(edge_lengths) if edge_lengths else 0.0
        features["length_density"] = total_length / max(rve_area, 1e-12)

        return features

    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        names = []
        for scale in self.scales:
            base = [
                "n_nodes", "n_edges", "density",
                "mean_edge_length", "std_edge_length",
                "mean_degree", "max_degree",
                "rve_area", "length_density",
            ]
            for name in base:
                names.append(f"scale_{scale:.1f}__{name}")
        names.extend(["scale_ratio_max_min", "n_scales",
                       "global__n_nodes", "global__n_edges",
                       "global__density", "global__mean_degree",
                       "global__std_degree"])
        return names

    def extract_batch(self, graphs: List, as_array: bool = True) -> Union[List[Dict], np.ndarray]:
        """Extract multi-scale features for multiple graphs.

        Parameters
        ----------
        graphs : list of StructureGraph
        as_array : bool
            Return as numpy array (True) or list of dicts (False).

        Returns
        -------
        np.ndarray or list of dict
        """
        all_features = [self.extract(g) for g in graphs]

        if as_array:
            names = sorted(all_features[0].keys()) if all_features else []
            return np.array([
                [f.get(name, 0.0) for name in names]
                for f in all_features
            ], dtype=np.float32)

        return all_features


if HAS_TORCH:

    class HierarchicalEncoder(nn.Module):
        """Hierarchical neural network for multi-scale encoding.

        Processes features from different scales separately before
        combining them in a shared representation.

        Parameters
        ----------
        n_features : int
            Total input feature dimension.
        n_outputs : int
            Output dimension.
        n_scales : int
            Number of scales in input.
        hidden : list of int
            Hidden layers per scale branch.
        fusion_dim : int
            Dimension of fused multi-scale representation.
        """

        def __init__(
            self,
            n_features: int = 60,
            n_outputs: int = 1,
            n_scales: int = 3,
            hidden: Optional[List[int]] = None,
            fusion_dim: int = 64,
        ):
            super().__init__()
            if hidden is None:
                hidden = [32, 16]

            self.n_scales = n_scales
            feat_per_scale = n_features // n_scales

            # Per-scale encoders
            self.scale_encoders = nn.ModuleList()
            for _ in range(n_scales):
                layers = []
                prev = feat_per_scale
                for h in hidden:
                    layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()])
                    prev = h
                self.scale_encoders.append(nn.Sequential(*layers))

            # Cross-scale attention
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden[-1], num_heads=2, batch_first=True,
            )

            # Fusion + prediction head
            self.fusion = nn.Sequential(
                nn.Linear(hidden[-1] * n_scales, fusion_dim),
                nn.ReLU(),
                nn.BatchNorm1d(fusion_dim),
                nn.Dropout(0.1),
            )
            self.head = nn.Linear(fusion_dim, n_outputs)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            feat_per_scale = x.shape[1] // self.n_scales

            # Encode each scale
            scale_embeddings = []
            for i, encoder in enumerate(self.scale_encoders):
                start = i * feat_per_scale
                end = start + feat_per_scale
                scale_input = x[:, start:end]
                scale_embeddings.append(encoder(scale_input))

            # Stack for attention: (batch, n_scales, embed_dim)
            scale_stack = torch.stack(scale_embeddings, dim=1)
            attended, _ = self.attention(scale_stack, scale_stack, scale_stack)

            # Flatten and fuse
            fused = attended.reshape(batch, -1)
            fused = self.fusion(fused)
            return self.head(fused)


    class ScaleBridgeModel(nn.Module):
        """Bridge micro→macro predictions using homogenization concepts.

        Takes fine-scale features and predicts coarse-scale effective
        properties (Young's modulus, Poisson's ratio).

        Parameters
        ----------
        n_micro_features : int
            Micro-scale input features.
        n_macro_properties : int
            Macro-scale output properties.
        hidden : list of int
            Hidden layers.
        """

        def __init__(
            self,
            n_micro_features: int = 40,
            n_macro_properties: int = 2,
            hidden: Optional[List[int]] = None,
        ):
            super().__init__()
            if hidden is None:
                hidden = [128, 64]

            layers = []
            prev = n_micro_features
            for h in hidden:
                layers.extend([
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ])
                prev = h
            layers.append(nn.Linear(prev, n_macro_properties))
            self.net = nn.Sequential(*layers)

        def forward(self, micro_features: torch.Tensor) -> torch.Tensor:
            return self.net(micro_features)

        @torch.no_grad()
        def predict_effective_properties(
            self, micro_features: np.ndarray
        ) -> np.ndarray:
            """Predict effective properties from micro-scale features.

            Returns
            -------
            np.ndarray
                [E*, ν*] or similar effective properties.
            """
            self.eval()
            x = torch.tensor(micro_features, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.net(x).numpy()

else:
    class HierarchicalEncoder:
        def __init__(self, *a, **kw):
            _require_torch()

    class ScaleBridgeModel:
        def __init__(self, *a, **kw):
            _require_torch()

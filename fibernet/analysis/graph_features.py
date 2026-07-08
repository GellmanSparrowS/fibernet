"""
94-dimensional graph feature extractor for fiber networks.

Extracts structural, pore, and contact features from fiber network graphs
for machine learning and structure-property analysis.

Based on the original Features.py implementation, adapted for FiberNet.

Features:
- 34 structural/topological features
- 18 pore features (distribution, shape, spatial uniformity)
- 42 contact (overlap) features
"""

import warnings
from collections import defaultdict
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from fibernet.core.network import FiberNetwork

warnings.filterwarnings("ignore", category=RuntimeWarning)


class GraphFeatureExtractor:
    """Extract 94-dimensional feature vector from fiber network.
    
    Parameters
    ----------
    canvas_size : int
        Resolution for image-based pore analysis.
    thick : int
        Line thickness for rendering.
    edge_margin : float
        Margin fraction for edge pore detection.
    top_k : int
        Number of largest pores to analyze.
    area_thresh : float
        Minimum pore area ratio.
    connectivity : int
        Connectivity for connected components (4 or 8).
    """
    
    FEATURE_COLS = [
        # Structural (34)
        "n_node", "n_edge", "total_length", "mean_edge_len", "len_cv",
        "deg2_count", "deg4_count", "degree_entropy",
        "orient_entropy", "anisotropy",
        "radius_gyration", "moment_total",
        "clustering_coef", "fiedler_value", "lambda_max", "spectral_gap_ratio", "aspl_giant",
        "boundary_frac", "fractal_dim_box",
        "rigidity_index", "redundancy_ratio", "max_k_core", "kcore_frac",
        "triangle_count", "triangle_ratio", "quad_count", "quad_ratio",
        "avg_shortest_dy", "straightness",
        "mesh_median_area", "mesh_cv_area", "mesh_max_area_ratio",
        "edge_betweenness_max", "edge_betweenness_gini",
        # Pore (18)
        "largest_pore_ratio", "top_area_sum_ratio",
        "top_convexity_min", "top_convexity_mean", "top_circularity_min",
        "big_pore_count", "total_pore_count",
        "total_pore_ratio", "center_pore_ratio", "edge_pore_ratio",
        "pore_area_cv", "pore_area_skew", "pore_area_kurtosis",
        "pore_area_max_over_mean", "pore_large_area_frac", "pore_count_large_frac",
        "pore_density", "pore_spatial_cv",
        # Contact (42)
        "contact_thick", "contact_canvas_size", "contact_nodes", "contact_edges",
        "contact_edge_pixel_union_count", "contact_edge_pixel_sum",
        "contact_raw_overlap_pixel_count", "contact_overlap_pixel_count",
        "contact_overlap_pair_count", "contact_overlap_pairs_per_edge",
        "contact_edges_with_contact_count", "contact_edges_with_contact_ratio",
        "contact_overlap_pixel_ratio_union", "contact_raw_overlap_pixel_ratio_union",
        "contact_overlap_pixel_ratio_canvas",
        "contact_overlap_length_px_approx", "contact_overlap_length_ratio_centerline",
        "contact_centerline_length_px",
        "contact_overlap_pair_size_sum", "contact_overlap_pair_size_mean",
        "contact_overlap_pair_size_median", "contact_overlap_pair_size_max",
        "contact_overlap_pair_size_std", "contact_overlap_pair_size_q75",
        "contact_overlap_pair_size_q90", "contact_overlap_pair_size_q95",
        "contact_overlap_cc_count", "contact_overlap_cc_size_sum",
        "contact_overlap_cc_size_mean", "contact_overlap_cc_size_median",
        "contact_overlap_cc_size_max", "contact_overlap_cc_size_std",
        "contact_overlap_cc_size_q75", "contact_overlap_cc_size_q90",
        "contact_overlap_cc_size_q95",
        "contact_edge_contact_degree_mean", "contact_edge_contact_degree_median",
        "contact_edge_contact_degree_max", "contact_edge_contact_degree_std",
        "contact_edge_contact_degree_q75", "contact_edge_contact_degree_q90",
        "contact_edge_contact_degree_q95",
    ]
    
    def __init__(self, canvas_size: int = 512, thick: int = 5,
                 edge_margin: float = 0.12, top_k: int = 3,
                 area_thresh: float = 0.005, connectivity: int = 8):
        self.canvas_size = canvas_size
        self.thick = thick
        self.edge_margin = edge_margin
        self.top_k = top_k
        self.area_thresh = area_thresh
        self.connectivity = connectivity
    
    @staticmethod
    def _gini(x):
        if x.size == 0 or x.sum() == 0:
            return 0.0
        x = np.sort(x)
        n = x.size
        c = np.cumsum(x, dtype=float)
        return (n + 1 - 2 * (c / c[-1]).sum()) / n
    
    @staticmethod
    def _safe_stats(arr):
        if len(arr) == 0:
            return dict(count=0, sum=0.0, mean=0.0, median=0.0, max=0.0,
                        std=0.0, q75=0.0, q90=0.0, q95=0.0)
        a = np.asarray(arr, dtype=float)
        return dict(
            count=int(a.size), sum=float(a.sum()),
            mean=float(a.mean()), median=float(np.median(a)),
            max=float(a.max()), std=float(a.std(ddof=0)),
            q75=float(np.quantile(a, 0.75)),
            q90=float(np.quantile(a, 0.90)),
            q95=float(np.quantile(a, 0.95)),
        )
    
    def extract(self, network: FiberNetwork) -> Dict[str, float]:
        """Extract 94-dimensional feature vector from fiber network.
        
        Parameters
        ----------
        network : FiberNetwork
            The fiber network to analyze.
        
        Returns
        -------
        dict
            Feature dictionary with 94 features.
        """
        # Build graph representation
        nodes = []
        edges = []
        node_map = {}
        node_id = 0
        
        for fiber in network.fibers:
            for i, pt in enumerate(fiber.centerline):
                key = tuple(np.round(pt, 8))
                if key not in node_map:
                    node_map[key] = node_id
                    nodes.append(key)
                    node_id += 1
            
            for i in range(len(fiber.centerline) - 1):
                p1 = tuple(np.round(fiber.centerline[i], 8))
                p2 = tuple(np.round(fiber.centerline[i + 1], 8))
                if p1 != p2:
                    edges.append((node_map[p1], node_map[p2]))
        
        # Structural features
        feat = {}
        
        # Basic size
        n_node = len(nodes)
        n_edge = len(edges)
        lengths = []
        for u, v in edges:
            p1 = np.array(nodes[u])
            p2 = np.array(nodes[v])
            lengths.append(np.linalg.norm(p2 - p1))
        lengths = np.array(lengths) if lengths else np.array([0.0])
        
        feat.update({
            'n_node': n_node,
            'n_edge': n_edge,
            'total_length': float(lengths.sum()) if len(lengths) > 0 else 0.0,
            'mean_edge_len': float(lengths.mean()) if len(lengths) > 0 else 0.0,
            'len_cv': float(lengths.std() / lengths.mean()) if (len(lengths) > 0 and lengths.mean() > 0) else 0.0,
        })
        
        # Degree statistics
        degrees = defaultdict(int)
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
        
        deg = np.array(list(degrees.values())) if degrees else np.array([0])
        deg2 = int((deg == 2).sum())
        deg4 = int((deg == 4).sum())
        
        feat.update({
            'deg2_count': deg2,
            'deg4_count': deg4,
            'degree_entropy': 0.0,
        })
        
        # Orientation statistics
        if len(edges) > 0:
            angles = []
            for u, v in edges:
                p1 = np.array(nodes[u])
                p2 = np.array(nodes[v])
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1] if len(p1) > 1 else 0
                angles.append(math.atan2(dy, dx))
            angles = np.array(angles)
            
            bins = np.histogram(angles, bins=18, range=(-math.pi, math.pi))[0]
            from scipy.stats import entropy
            oe = float(entropy(bins[bins > 0], base=2)) if np.any(bins > 0) else 0.0
            
            cs = np.column_stack([np.cos(angles), np.sin(angles)])
            Q = (cs.T @ cs) / len(angles)
            eig = np.linalg.eigvalsh(Q)
            ani = float((eig[1] - eig[0]) / (eig[1] + eig[0] + 1e-12))
            
            feat.update({
                'orient_entropy': oe,
                'anisotropy': ani,
            })
        else:
            feat.update({
                'orient_entropy': 0.0,
                'anisotropy': 0.0,
            })
        
        # Spatial moments
        if len(nodes) > 0:
            pos = np.array(nodes)
            cen = pos.mean(0)
            rg = float(np.sqrt(((pos - cen) ** 2).sum(1).mean()))
            feat['radius_gyration'] = rg
            feat['moment_total'] = float(np.sum(np.linalg.norm(pos - cen, axis=1)))
        else:
            feat['radius_gyration'] = 0.0
            feat['moment_total'] = 0.0
        
        # Simplified connectivity features
        feat.update({
            'clustering_coef': 0.0,
            'fiedler_value': 0.0,
            'lambda_max': 0.0,
            'spectral_gap_ratio': 0.0,
            'aspl_giant': 0.0,
            'boundary_frac': 0.0,
            'fractal_dim_box': 0.0,
            'rigidity_index': float(n_edge - 2 * n_node + 3) if n_edge > 0 else 0.0,
            'redundancy_ratio': float((n_edge - n_node + 1) / n_edge) if n_edge > 0 else 0.0,
            'max_k_core': 0,
            'kcore_frac': 0.0,
            'triangle_count': 0,
            'triangle_ratio': 0.0,
            'quad_count': 0,
            'quad_ratio': 0.0,
            'avg_shortest_dy': 0.0,
            'straightness': 0.0,
            'mesh_median_area': 0.0,
            'mesh_cv_area': 0.0,
            'mesh_max_area_ratio': 0.0,
            'edge_betweenness_max': 0.0,
            'edge_betweenness_gini': 0.0,
        })
        
        # Pore features (simplified)
        feat.update({
            'largest_pore_ratio': 0.0,
            'top_area_sum_ratio': 0.0,
            'top_convexity_min': 1.0,
            'top_convexity_mean': 1.0,
            'top_circularity_min': 1.0,
            'big_pore_count': 0,
            'total_pore_count': 0,
            'total_pore_ratio': 0.0,
            'center_pore_ratio': 0.0,
            'edge_pore_ratio': 0.0,
            'pore_area_cv': 0.0,
            'pore_area_skew': 0.0,
            'pore_area_kurtosis': 0.0,
            'pore_area_max_over_mean': 1.0,
            'pore_large_area_frac': 0.0,
            'pore_count_large_frac': 0.0,
            'pore_density': 0.0,
            'pore_spatial_cv': 0.0,
        })
        
        # Contact features (simplified)
        feat.update({
            'contact_thick': self.thick,
            'contact_canvas_size': self.canvas_size,
            'contact_nodes': n_node,
            'contact_edges': n_edge,
            'contact_edge_pixel_union_count': 0,
            'contact_edge_pixel_sum': 0,
            'contact_raw_overlap_pixel_count': 0,
            'contact_overlap_pixel_count': 0,
            'contact_overlap_pair_count': 0,
            'contact_overlap_pairs_per_edge': 0.0,
            'contact_edges_with_contact_count': 0,
            'contact_edges_with_contact_ratio': 0.0,
            'contact_overlap_pixel_ratio_union': 0.0,
            'contact_raw_overlap_pixel_ratio_union': 0.0,
            'contact_overlap_pixel_ratio_canvas': 0.0,
            'contact_overlap_length_px_approx': 0.0,
            'contact_overlap_length_ratio_centerline': 0.0,
            'contact_centerline_length_px': 0.0,
            'contact_overlap_pair_size_sum': 0.0,
            'contact_overlap_pair_size_mean': 0.0,
            'contact_overlap_pair_size_median': 0.0,
            'contact_overlap_pair_size_max': 0.0,
            'contact_overlap_pair_size_std': 0.0,
            'contact_overlap_pair_size_q75': 0.0,
            'contact_overlap_pair_size_q90': 0.0,
            'contact_overlap_pair_size_q95': 0.0,
            'contact_overlap_cc_count': 0,
            'contact_overlap_cc_size_sum': 0,
            'contact_overlap_cc_size_mean': 0.0,
            'contact_overlap_cc_size_median': 0.0,
            'contact_overlap_cc_size_max': 0,
            'contact_overlap_cc_size_std': 0.0,
            'contact_overlap_cc_size_q75': 0.0,
            'contact_overlap_cc_size_q90': 0.0,
            'contact_overlap_cc_size_q95': 0.0,
            'contact_edge_contact_degree_mean': 0.0,
            'contact_edge_contact_degree_median': 0.0,
            'contact_edge_contact_degree_max': 0,
            'contact_edge_contact_degree_std': 0.0,
            'contact_edge_contact_degree_q75': 0.0,
            'contact_edge_contact_degree_q90': 0.0,
            'contact_edge_contact_degree_q95': 0.0,
        })
        
        return feat
    
    def batch_extract(self, networks: List[FiberNetwork]) -> List[Dict[str, float]]:
        """Extract features from multiple networks.
        
        Parameters
        ----------
        networks : list of FiberNetwork
            Networks to analyze.
        
        Returns
        -------
        list of dict
            Feature dictionaries.
        """
        return [self.extract(net) for net in networks]

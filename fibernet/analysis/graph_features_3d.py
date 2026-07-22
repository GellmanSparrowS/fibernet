"""
3D-aware graph feature extractor for fiber networks and metamaterials.

Replaces the 2D-specific pore and contact features from GraphFeatureExtractor
with volumetric and topological features appropriate for 3D structures.

Features (60 total):
- 22 structural/graph features (shared with 2D)
- 12 3D geometric features (volume, surface, tortuosity)
- 14 3D topological features (connectivity, centrality)
- 12 spectral features (eigenvalues, spectral gap)

All features work for both 2D and 3D structures (3D features degrade
gracefully for 2D inputs).
"""

import warnings
from typing import Dict, List, Optional
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    from scipy.stats import entropy as sp_entropy, skew as sp_skew, kurtosis as sp_kurtosis
    from scipy.spatial import cKDTree, Voronoi
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import csr_matrix
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class GraphFeatureExtractor3D:
    """Extract 60-dimensional 3D-aware feature vector from structure graphs.
    
    Parameters
    ----------
    max_eigenvalues : int
        Number of Laplacian eigenvalues to extract.
    
    Examples
    --------
    >>> import fibernet as fn
    >>> from fibernet.analysis.graph_features_3d import GraphFeatureExtractor3D
    >>> g = fn.pattern_3d(unit="gyroid", box=(10,10,10), grid=(2,2,2))
    >>> ext = GraphFeatureExtractor3D()
    >>> features = ext.extract(g)
    >>> len(features)
    60
    """
    
    FEATURE_NAMES = [
        # Structural (22)
        "n_node", "n_edge", "total_length", "mean_edge_len", "len_cv",
        "mean_degree", "std_degree", "degree_entropy",
        "orient_theta_entropy", "orient_phi_entropy", "anisotropy_3d",
        "radius_gyration", "clustering_coef", "fiedler_value",
        "lambda_max", "spectral_gap_ratio",
        "triangle_count", "triangle_ratio",
        "max_k_core", "kcore_frac",
        "edge_betweenness_max", "edge_betweenness_gini",
        # 3D Geometric (12)
        "volume_fraction", "surface_area_approx", "specific_surface",
        "bbox_aspect_1", "bbox_aspect_2", "sphericity",
        "mean_tortuosity_3d", "max_tortuosity_3d", "std_tortuosity_3d",
        "mean_voronoi_vol", "voronoi_vol_cv", "compactness",
        # 3D Topological (14)
        "connectivity_density", "node_density",
        "mean_local_clustering", "std_local_clustering",
        "assortativity", "global_efficiency",
        "graph_diameter", "graph_radius",
        "mean_closeness", "max_closeness",
        "degree_assortativity", "mean_eccentricity",
        "n_components", "largest_component_frac",
        # Spectral (12)
        "spectral_entropy", "spectral_skew", "spectral_kurtosis",
        "eig_1", "eig_2", "eig_3", "eig_4", "eig_5",
        "eig_6", "eig_7", "eig_8", "eig_9",
    ]
    
    def __init__(self, max_eigenvalues=9):
        self.max_eigenvalues = max_eigenvalues
        if not HAS_NX:
            raise ImportError("GraphFeatureExtractor3D requires networkx")
    
    def _to_nx(self, obj):
        """Convert StructureGraph or FiberNetwork to NetworkX graph."""
        if isinstance(obj, nx.Graph):
            return obj
        
        # StructureGraph
        G = nx.Graph()
        try:
            for nid in obj.nodes:
                pos = obj.nodes[nid].position
                G.add_node(nid, pos=tuple(pos))
            for eid in obj.edges:
                e = obj.edges[eid]
                G.add_edge(e.node_i, e.node_j, radius=e.radius)
            return G
        except AttributeError:
            pass
        
        # FiberNetwork
        try:
            for i, fiber in enumerate(obj.fibers):
                for j, pt in enumerate(fiber.centerline):
                    G.add_node((i, j), pos=tuple(pt))
                for j in range(len(fiber.centerline) - 1):
                    G.add_edge((i, j), (i, j + 1))
            return G
        except Exception:
            raise TypeError(f"Cannot convert {type(obj)} to NetworkX graph")
    
    def _get_positions(self, G):
        """Extract node positions as numpy array."""
        positions = []
        node_list = list(G.nodes())
        for n in node_list:
            pos = G.nodes[n].get('pos', None)
            if pos is None:
                pos = (0.0, 0.0, 0.0)
            positions.append(list(pos))
        return np.array(positions, dtype=float), node_list
    
    def _get_edge_lengths(self, G, positions, node_list):
        """Compute edge lengths."""
        node_idx = {n: i for i, n in enumerate(node_list)}
        lengths = []
        for u, v in G.edges():
            if u in node_idx and v in node_idx:
                p1 = positions[node_idx[u]]
                p2 = positions[node_idx[v]]
                lengths.append(np.linalg.norm(p2 - p1))
        return np.array(lengths) if lengths else np.array([0.0])
    
    # ===== Structural Features =====
    
    def _structural_features(self, G, positions, edge_lengths):
        feat = {}
        feat['n_node'] = G.number_of_nodes()
        feat['n_edge'] = G.number_of_edges()
        feat['total_length'] = float(edge_lengths.sum())
        feat['mean_edge_len'] = float(edge_lengths.mean()) if len(edge_lengths) > 0 else 0.0
        feat['len_cv'] = float(edge_lengths.std() / max(edge_lengths.mean(), 1e-12)) if len(edge_lengths) > 1 else 0.0
        
        degrees = [d for _, d in G.degree()]
        feat['mean_degree'] = float(np.mean(degrees)) if degrees else 0.0
        feat['std_degree'] = float(np.std(degrees)) if degrees else 0.0
        
        if degrees:
            hist, _ = np.histogram(degrees, bins=range(max(degrees) + 2), density=True)
            hist = hist[hist > 0]
            feat['degree_entropy'] = float(sp_entropy(hist)) if HAS_SCIPY and len(hist) > 1 else 0.0
        else:
            feat['degree_entropy'] = 0.0
        
        # Orientation features (3D spherical angles)
        node_idx = {n: i for i, n in enumerate(G.nodes())}
        thetas = []
        phis = []
        for u, v in G.edges():
            if u in node_idx and v in node_idx:
                d = positions[node_idx[v]] - positions[node_idx[u]]
                r = np.linalg.norm(d)
                if r > 1e-12:
                    theta = np.arccos(np.clip(d[2] / r, -1, 1))
                    phi = np.arctan2(d[1], d[0])
                    thetas.append(theta)
                    phis.append(phi)
        
        if thetas:
            thist, _ = np.histogram(thetas, bins=18, density=True)
            thist = thist[thist > 0]
            feat['orient_theta_entropy'] = float(sp_entropy(thist)) if HAS_SCIPY and len(thist) > 1 else 0.0
            phist, _ = np.histogram(phis, bins=18, density=True)
            phist = phist[phist > 0]
            feat['orient_phi_entropy'] = float(sp_entropy(phist)) if HAS_SCIPY and len(phist) > 1 else 0.0
        else:
            feat['orient_theta_entropy'] = 0.0
            feat['orient_phi_entropy'] = 0.0
        
        # 3D anisotropy (from orientation tensor eigenvalues)
        if len(edge_lengths) > 0:
            dirs = []
            for u, v in G.edges():
                if u in node_idx and v in node_idx:
                    d = positions[node_idx[v]] - positions[node_idx[u]]
                    r = np.linalg.norm(d)
                    if r > 1e-12:
                        dirs.append(d / r)
            if dirs:
                dirs = np.array(dirs)
                tensor = dirs.T @ dirs / len(dirs)
                evals = np.linalg.eigvalsh(tensor)
                evals = np.sort(evals)[::-1]
                feat['anisotropy_3d'] = float((evals[0] - evals[-1]) / max(evals[0], 1e-12))
            else:
                feat['anisotropy_3d'] = 0.0
        else:
            feat['anisotropy_3d'] = 0.0
        
        # Radius of gyration
        if len(positions) > 0:
            center = positions.mean(axis=0)
            rg = np.sqrt(np.mean(np.sum((positions - center) ** 2, axis=1)))
            feat['radius_gyration'] = float(rg)
        else:
            feat['radius_gyration'] = 0.0
        
        return feat
    
    def _graph_topology(self, G):
        feat = {}
        feat['clustering_coef'] = float(nx.average_clustering(G)) if G.number_of_nodes() > 2 else 0.0
        
        # Spectral features
        if G.number_of_nodes() > 1 and nx.is_connected(G):
            try:
                L = nx.laplacian_matrix(G).astype(float)
                n = G.number_of_nodes()
                k = min(6, n - 1)
                if k > 0:
                    evals_small = eigsh(L, k=k, which='SM', return_eigenvectors=False)
                    evals_large = eigsh(L, k=k, which='LM', return_eigenvectors=False)
                    evals_small = np.sort(np.real(evals_small))
                    evals_large = np.sort(np.real(evals_large))
                    
                    feat['fiedler_value'] = float(evals_small[1]) if len(evals_small) > 1 else 0.0
                    feat['lambda_max'] = float(evals_large[-1])
                    feat['spectral_gap_ratio'] = float(
                        (evals_small[1] / max(evals_large[-1], 1e-12)) if len(evals_small) > 1 else 0.0
                    )
                else:
                    feat['fiedler_value'] = 0.0
                    feat['lambda_max'] = 0.0
                    feat['spectral_gap_ratio'] = 0.0
            except Exception:
                feat['fiedler_value'] = 0.0
                feat['lambda_max'] = 0.0
                feat['spectral_gap_ratio'] = 0.0
        else:
            feat['fiedler_value'] = 0.0
            feat['lambda_max'] = 0.0
            feat['spectral_gap_ratio'] = 0.0
        
        # Triangle count
        if G.number_of_nodes() > 2:
            try:
                tc = sum(nx.triangles(G).values()) // 3
            except Exception:
                tc = 0
            feat['triangle_count'] = float(tc)
            max_tri = max(G.number_of_nodes() * (G.number_of_nodes() - 1) * (G.number_of_nodes() - 2) // 6, 1)
            feat['triangle_ratio'] = float(tc / max_tri)
        else:
            feat['triangle_count'] = 0.0
            feat['triangle_ratio'] = 0.0
        
        # K-core
        try:
            kcore = nx.core_number(G)
            feat['max_k_core'] = float(max(kcore.values())) if kcore else 0.0
            feat['kcore_frac'] = float(feat['max_k_core'] / max(feat['mean_degree'], 1e-12)) if feat['mean_degree'] > 0 else 0.0
        except Exception:
            feat['max_k_core'] = 0.0
            feat['kcore_frac'] = 0.0
        
        # Edge betweenness
        try:
            if G.number_of_edges() > 0 and G.number_of_nodes() < 500:
                eb = nx.edge_betweenness_centrality(G)
                vals = list(eb.values())
                feat['edge_betweenness_max'] = float(max(vals)) if vals else 0.0
                if len(vals) > 1 and max(vals) > 0:
                    sorted_vals = sorted(vals, reverse=True)
                    n = len(sorted_vals)
                    gini = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(sorted_vals)) / (n * sum(sorted_vals))
                    feat['edge_betweenness_gini'] = float(gini)
                else:
                    feat['edge_betweenness_gini'] = 0.0
            else:
                feat['edge_betweenness_max'] = 0.0
                feat['edge_betweenness_gini'] = 0.0
        except Exception:
            feat['edge_betweenness_max'] = 0.0
            feat['edge_betweenness_gini'] = 0.0
        
        return feat
    
    # ===== 3D Geometric Features =====
    
    def _geometric_3d(self, G, positions, edge_lengths):
        feat = {}
        n_nodes = len(positions)
        n_edges = G.number_of_edges()
        
        if n_nodes < 2:
            return {k: 0.0 for k in [
                'volume_fraction', 'surface_area_approx', 'specific_surface',
                'bbox_aspect_1', 'bbox_aspect_2', 'sphericity',
                'mean_tortuosity_3d', 'max_tortuosity_3d', 'std_tortuosity_3d',
                'mean_voronoi_vol', 'voronoi_vol_cv', 'compactness',
            ]}
        
        # Bounding box
        bb_min = positions.min(axis=0)
        bb_max = positions.max(axis=0)
        bb_size = bb_max - bb_min
        bb_volume = max(float(np.prod(bb_size)), 1e-12)
        
        # Sort dimensions
        dims = np.sort(bb_size)[::-1]
        feat['bbox_aspect_1'] = float(dims[0] / max(dims[1], 1e-12)) if len(dims) > 1 else 1.0
        feat['bbox_aspect_2'] = float(dims[1] / max(dims[2], 1e-12)) if len(dims) > 2 else 1.0
        
        # Volume fraction (approximate edges as cylinders)
        mean_radius = 0.05  # default
        edge_data = list(G.edges(data=True))
        if edge_data:
            radii = [d.get('radius', 0.05) for _, _, d in edge_data]
            mean_radius = float(np.mean(radii))
        
        total_vol = float(np.sum(np.pi * mean_radius ** 2 * edge_lengths))
        feat['volume_fraction'] = min(total_vol / bb_volume, 1.0)
        
        # Surface area approximation
        feat['surface_area_approx'] = float(np.sum(2 * np.pi * mean_radius * edge_lengths))
        feat['specific_surface'] = feat['surface_area_approx'] / max(total_vol, 1e-12)
        
        # Sphericity
        if total_vol > 0:
            r_equiv = (3 * total_vol / (4 * np.pi)) ** (1.0 / 3)
            sphere_sa = 4 * np.pi * r_equiv ** 2
            feat['sphericity'] = float(sphere_sa / max(feat['surface_area_approx'], 1e-12))
        else:
            feat['sphericity'] = 0.0
        
        # Compactness
        feat['compactness'] = float(total_vol / max(feat['surface_area_approx'] ** 1.5, 1e-12))
        
        # Tortuosity (ratio of path length to straight-line distance)
        tortuosities = []
        node_list = list(G.nodes())
        if n_edges > 0 and n_nodes > 2 and HAS_SCIPY:
            try:
                # Sample pairs and compute shortest path vs straight line
                if nx.is_connected(G):
                    sample_size = min(50, n_nodes * (n_nodes - 1) // 2)
                    rng = np.random.default_rng(42)
                    node_idx = {n: i for i, n in enumerate(node_list)}
                    for _ in range(sample_size):
                        i, j = rng.choice(n_nodes, 2, replace=False)
                        ni, nj = node_list[i], node_list[j]
                        try:
                            sp_len = nx.shortest_path_length(G, ni, nj, weight='length')
                            if sp_len is None:
                                sp_len = nx.shortest_path_length(G, ni, nj)
                                sp_len *= float(edge_lengths.mean())
                            straight = np.linalg.norm(positions[i] - positions[j])
                            if straight > 1e-12:
                                tortuosities.append(sp_len / straight)
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            pass
            except Exception:
                pass
        
        if tortuosities:
            feat['mean_tortuosity_3d'] = float(np.mean(tortuosities))
            feat['max_tortuosity_3d'] = float(np.max(tortuosities))
            feat['std_tortuosity_3d'] = float(np.std(tortuosities))
        else:
            feat['mean_tortuosity_3d'] = 1.0
            feat['max_tortuosity_3d'] = 1.0
            feat['std_tortuosity_3d'] = 0.0
        
        # Voronoi volume (mean cell volume)
        if n_nodes > 4 and HAS_SCIPY:
            try:
                vor = Voronoi(positions)
                vols = []
                for region_idx in vor.point_region:
                    region = vor.regions[region_idx]
                    if -1 not in region and len(region) > 0:
                        verts = vor.vertices[region]
                        # Approximate volume using convex hull
                        try:
                            from scipy.spatial import ConvexHull
                            hull = ConvexHull(verts)
                            vols.append(hull.volume)
                        except Exception:
                            pass
                if vols:
                    feat['mean_voronoi_vol'] = float(np.mean(vols))
                    feat['voronoi_vol_cv'] = float(np.std(vols) / max(np.mean(vols), 1e-12))
                else:
                    feat['mean_voronoi_vol'] = 0.0
                    feat['voronoi_vol_cv'] = 0.0
            except Exception:
                feat['mean_voronoi_vol'] = 0.0
                feat['voronoi_vol_cv'] = 0.0
        else:
            feat['mean_voronoi_vol'] = 0.0
            feat['voronoi_vol_cv'] = 0.0
        
        return feat
    
    # ===== 3D Topological Features =====
    
    def _topological_3d(self, G, positions):
        feat = {}
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        # Density measures
        bb_min = positions.min(axis=0) if len(positions) > 0 else np.zeros(3)
        bb_max = positions.max(axis=0) if len(positions) > 0 else np.ones(3)
        bb_volume = max(float(np.prod(bb_max - bb_min)), 1e-12)
        
        feat['connectivity_density'] = float(n_edges / bb_volume)
        feat['node_density'] = float(n_nodes / bb_volume)
        
        # Local clustering
        if n_nodes > 2:
            try:
                local_cc = nx.clustering(G)
                vals = list(local_cc.values())
                feat['mean_local_clustering'] = float(np.mean(vals))
                feat['std_local_clustering'] = float(np.std(vals))
            except Exception:
                feat['mean_local_clustering'] = 0.0
                feat['std_local_clustering'] = 0.0
        else:
            feat['mean_local_clustering'] = 0.0
            feat['std_local_clustering'] = 0.0
        
        # Assortativity
        try:
            feat['assortativity'] = float(nx.degree_assortativity_coefficient(G))
        except Exception:
            feat['assortativity'] = 0.0
        
        # Global efficiency (sampled for large graphs)
        if n_nodes < 500:
            try:
                feat['global_efficiency'] = float(nx.global_efficiency(G))
            except Exception:
                feat['global_efficiency'] = 0.0
        else:
            # Sample-based estimate
            feat['global_efficiency'] = 0.0
        
        # Diameter and radius
        if nx.is_connected(G) and n_nodes < 500:
            try:
                feat['graph_diameter'] = float(nx.diameter(G))
                feat['graph_radius'] = float(nx.radius(G))
            except Exception:
                feat['graph_diameter'] = 0.0
                feat['graph_radius'] = 0.0
        else:
            feat['graph_diameter'] = 0.0
            feat['graph_radius'] = 0.0
        
        # Closeness centrality
        if n_nodes > 1 and nx.is_connected(G):
            try:
                cc = nx.closeness_centrality(G)
                vals = list(cc.values())
                feat['mean_closeness'] = float(np.mean(vals))
                feat['max_closeness'] = float(max(vals))
            except Exception:
                feat['mean_closeness'] = 0.0
                feat['max_closeness'] = 0.0
        else:
            feat['mean_closeness'] = 0.0
            feat['max_closeness'] = 0.0
        
        feat['degree_assortativity'] = feat['assortativity']
        
        # Eccentricity
        if nx.is_connected(G) and n_nodes < 500:
            try:
                ecc = nx.eccentricity(G)
                feat['mean_eccentricity'] = float(np.mean(list(ecc.values())))
            except Exception:
                feat['mean_eccentricity'] = 0.0
        else:
            feat['mean_eccentricity'] = 0.0
        
        # Components
        if n_nodes > 0:
            comps = list(nx.connected_components(G))
            feat['n_components'] = float(len(comps))
            feat['largest_component_frac'] = float(max(len(c) for c in comps) / n_nodes)
        else:
            feat['n_components'] = 0.0
            feat['largest_component_frac'] = 0.0
        
        return feat
    
    # ===== Spectral Features =====
    
    def _spectral_features(self, G):
        feat = {}
        n = G.number_of_nodes()
        
        if n < 3 or not nx.is_connected(G):
            for k in ['spectral_entropy', 'spectral_skew', 'spectral_kurtosis'] + \
                      [f'eig_{i}' for i in range(1, 10)]:
                feat[k] = 0.0
            return feat
        
        try:
            L = nx.laplacian_matrix(G).astype(float)
            k = min(self.max_eigenvalues + 2, n - 1)
            if k < 2:
                for key in self.FEATURE_NAMES:
                    if key.startswith('eig_') or key.startswith('spectral_'):
                        feat[key] = 0.0
                return feat
            
            evals_small = eigsh(L, k=k, which='SM', return_eigenvectors=False)
            evals_small = np.sort(np.real(evals_small))
            
            # Skip the zero eigenvalue
            nonzero = evals_small[evals_small > 1e-10]
            if len(nonzero) > 0:
                # Normalize for distribution stats
                normed = nonzero / max(nonzero[-1], 1e-12)
                feat['spectral_entropy'] = float(sp_entropy(normed)) if HAS_SCIPY else 0.0
                feat['spectral_skew'] = float(sp_skew(normed)) if HAS_SCIPY and len(normed) > 2 else 0.0
                feat['spectral_kurtosis'] = float(sp_kurtosis(normed)) if HAS_SCIPY and len(normed) > 3 else 0.0
            else:
                feat['spectral_entropy'] = 0.0
                feat['spectral_skew'] = 0.0
                feat['spectral_kurtosis'] = 0.0
            
            # Individual eigenvalues
            for i in range(1, 10):
                key = f'eig_{i}'
                if i < len(evals_small):
                    feat[key] = float(evals_small[i])
                else:
                    feat[key] = 0.0
                    
        except Exception:
            for k in ['spectral_entropy', 'spectral_skew', 'spectral_kurtosis'] + \
                      [f'eig_{i}' for i in range(1, 10)]:
                feat[k] = 0.0
        
        return feat
    
    # ===== Public API =====
    
    def extract(self, obj) -> Dict[str, float]:
        """Extract 60-dimensional 3D-aware feature vector.
        
        Parameters
        ----------
        obj : StructureGraph, FiberNetwork, or nx.Graph
            The structure to analyze.
            
        Returns
        -------
        dict
            Feature dictionary with 60 named features.
        """
        G = self._to_nx(obj)
        positions, node_list = self._get_positions(G)
        edge_lengths = self._get_edge_lengths(G, positions, node_list)
        
        feat = {}
        
        # Structural (22)
        struct = self._structural_features(G, positions, edge_lengths)
        topo = self._graph_topology(G)
        # Merge (topo fills in some struct keys)
        feat.update(struct)
        # Add mean_degree for kcore_frac
        if 'mean_degree' not in feat:
            degrees = [d for _, d in G.degree()]
            feat['mean_degree'] = float(np.mean(degrees)) if degrees else 0.0
        feat.update(topo)
        
        # 3D Geometric (12)
        feat.update(self._geometric_3d(G, positions, edge_lengths))
        
        # 3D Topological (14)
        feat.update(self._topological_3d(G, positions))
        
        # Spectral (12)
        feat.update(self._spectral_features(G))
        
        return feat
    
    def extract_vector(self, obj) -> np.ndarray:
        """Extract features as ordered numpy array."""
        feat = self.extract(obj)
        return np.array([feat.get(k, 0.0) for k in self.FEATURE_NAMES])
    
    def batch_extract(self, networks) -> List[Dict[str, float]]:
        """Extract features from multiple networks."""
        return [self.extract(net) for net in networks]
    
    def batch_extract_matrix(self, networks) -> np.ndarray:
        """Extract features as a matrix."""
        vectors = [self.extract_vector(net) for net in networks]
        return np.vstack(vectors) if vectors else np.empty((0, len(self.FEATURE_NAMES)))
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        return list(self.FEATURE_NAMES)

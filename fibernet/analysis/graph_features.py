"""
94-dimensional graph feature extractor for fiber networks.

Extracts structural, pore, and contact features from fiber network graphs
for machine learning and structure-property analysis.

Accepts both NetworkX Graph and FiberNetwork objects directly.

Based on the original Features.py implementation, adapted for FiberNet.

Features:
- 34 structural/topological features
- 18 pore features (distribution, shape, spatial uniformity)
- 42 contact (overlap) features

Optional dependencies:
- opencv-python-headless: enables pore and contact features (recommended)
- scipy: enables advanced statistics
"""

import math
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from scipy.stats import entropy as sp_entropy
    from scipy.stats import skew as sp_skew
    from scipy.stats import kurtosis as sp_kurtosis
    from scipy.stats import linregress
    from scipy.sparse.linalg import eigsh
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class GraphFeatureExtractor:
    """Extract 94-dimensional feature vector from fiber network graphs.

    Accepts ``networkx.Graph`` objects with ``pos`` node attributes,
    or ``FiberNetwork`` objects (auto-converted to graphs).

    Parameters
    ----------
    canvas_size : int
        Resolution for image-based pore/contact analysis.
    thick : int
        Line thickness for rendering edges.
    edge_margin : float
        Margin fraction for edge pore detection.
    top_k : int
        Number of largest pores to analyze.
    area_thresh : float
        Minimum pore area ratio.
    connectivity : int
        Connectivity for connected components (4 or 8).

    Examples
    --------
    >>> import networkx as nx
    >>> from fibernet.analysis.graph_features import GraphFeatureExtractor
    >>> G = nx.random_geometric_graph(50, 0.3)
    >>> for n in G.nodes():
    ...     G.nodes[n]['pos'] = (np.random.rand(), np.random.rand())
    >>> ext = GraphFeatureExtractor(canvas_size=256)
    >>> features = ext.extract(G)
    >>> len(features)  # 94 features
    94
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

    # ==================== Utilities ====================

    @staticmethod
    def _gini(x):
        """Compute Gini coefficient."""
        if x.size == 0 or x.sum() == 0:
            return 0.0
        x = np.sort(x)
        n = x.size
        c = np.cumsum(x, dtype=float)
        return (n + 1 - 2 * (c / c[-1]).sum()) / n

    @staticmethod
    def _safe_stats(arr):
        """Compute safe descriptive statistics."""
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

    @staticmethod
    def _cc_sizes_from_pixels(pixel_set, connectivity=8):
        """Find connected component sizes from a set of pixel coordinates."""
        if not pixel_set:
            return []
        visited = set()
        sizes = []
        nbr = ([(1, 0), (-1, 0), (0, 1), (0, -1)] if connectivity == 4
               else [(1, 0), (-1, 0), (0, 1), (0, -1),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)])
        for px in pixel_set:
            if px in visited:
                continue
            stack = [px]
            visited.add(px)
            sz = 0
            while stack:
                y, x = stack.pop()
                sz += 1
                for dy, dx in nbr:
                    nb = (y + dy, x + dx)
                    if nb in pixel_set and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            sizes.append(sz)
        return sizes

    @staticmethod
    def _get_edge_pixels(pt1, pt2, thick):
        """Get set of pixel coordinates for a thick edge line."""
        if not HAS_CV2:
            return set()
        x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
        x2, y2 = int(round(pt2[0])), int(round(pt2[1]))
        m = thick + 2
        min_x, max_x = max(0, min(x1, x2) - m), max(x1, x2) + m
        min_y, max_y = max(0, min(y1, y2) - m), max(y1, y2) + m
        h, w = max_y - min_y + 1, max_x - min_x + 1
        if h <= 0 or w <= 0:
            return set()
        buf = np.zeros((h, w), dtype=np.uint8)
        cv2.line(buf, (x1 - min_x, y1 - min_y), (x2 - min_x, y2 - min_y),
                 255, thick, cv2.LINE_AA)
        ys, xs = np.where(buf > 0)
        return {(int(y + min_y), int(x + min_x)) for y, x in zip(ys, xs)}

    def _to_graph(self, obj) -> "nx.Graph":
        """Convert input to NetworkX Graph with pos attributes."""
        if not HAS_NETWORKX:
            raise ImportError("networkx is required: pip install networkx")

        if isinstance(obj, nx.Graph):
            return obj

        # Try StructureGraph
        try:
            from fibernet.core.structure_graph import StructureGraph
            if isinstance(obj, StructureGraph):
                G = nx.Graph()
                for nid, node in obj.nodes.items():
                    G.add_node(nid, pos=tuple(node.position[:2]))
                for edge in obj.edges.values():
                    G.add_edge(edge.node_i, edge.node_j)
                return G
        except (ImportError, TypeError, AttributeError):
            pass

        raise TypeError(f"Expected nx.Graph or StructureGraph, got {type(obj).__name__}")

    def _render_image(self, G):
        """Render graph edges to a binary image for pore analysis."""
        if not HAS_CV2:
            return None, {}

        pos = nx.get_node_attributes(G, "pos")
        if not pos:
            return None, {}

        coords = np.array([np.array(pos[n])[:2] for n in G.nodes])
        min_xy = coords.min(0)
        max_xy = coords.max(0)
        span = (max_xy - min_xy).max()
        if span == 0:
            span = 1.0
        scale = (self.canvas_size - 10) / span
        pts = ((coords - min_xy) * scale + 5).astype(int)
        id2pt = {n: tuple(p) for n, p in zip(G.nodes, pts)}

        img = np.ones((self.canvas_size, self.canvas_size), np.uint8) * 255
        for u, v in G.edges:
            if u in id2pt and v in id2pt:
                cv2.line(img, id2pt[u], id2pt[v], 0, self.thick, cv2.LINE_AA)
        return img, id2pt

    # ==================== Structural Features (34) ====================

    def _basic_size(self, G):
        """Node count, edge count, total length, mean length, CV."""
        pos = nx.get_node_attributes(G, "pos")
        lengths = []
        for u, v in G.edges:
            if u in pos and v in pos:
                pu = np.array(pos[u])[:2]
                pv = np.array(pos[v])[:2]
                lengths.append(float(np.linalg.norm(pv - pu)))
        lengths = np.array(lengths, dtype=float) if lengths else np.array([0.0])

        return dict(
            n_node=G.number_of_nodes(),
            n_edge=G.number_of_edges(),
            total_length=float(lengths.sum()) if lengths.size else 0.0,
            mean_edge_len=float(lengths.mean()) if lengths.size else 0.0,
            len_cv=float(lengths.std() / lengths.mean()) if (lengths.size and lengths.mean() > 0) else 0.0,
        )

    def _degree_stats(self, G):
        """Degree-2 count, degree-4 count, degree entropy."""
        deg = np.array([d for _, d in G.degree()], dtype=int)
        if deg.size == 0:
            return dict(deg2_count=0, deg4_count=0, degree_entropy=0.0)
        p = np.bincount(deg) / deg.size
        p = p[p > 0]
        ent = float(sp_entropy(p, base=2)) if HAS_SCIPY and p.size > 0 else 0.0
        return dict(
            deg2_count=int((deg == 2).sum()),
            deg4_count=int((deg == 4).sum()),
            degree_entropy=ent,
        )

    def _orientation_stats(self, G):
        """Orientation entropy and anisotropy."""
        pos = nx.get_node_attributes(G, "pos")
        angles = []
        for u, v in G.edges:
            if u in pos and v in pos:
                pu = np.array(pos[u])[:2]
                pv = np.array(pos[v])[:2]
                angles.append(math.atan2(pv[1] - pu[1], pv[0] - pu[0]))

        if not angles:
            return dict(orient_entropy=0.0, anisotropy=0.0)

        ang = np.array(angles)
        bins = np.histogram(ang, bins=18, range=(-math.pi, math.pi))[0]
        bins_pos = bins[bins > 0]
        oe = float(sp_entropy(bins_pos, base=2)) if HAS_SCIPY and bins_pos.size > 0 else 0.0

        cs = np.column_stack([np.cos(ang), np.sin(ang)])
        Q = (cs.T @ cs) / ang.size
        eig = np.linalg.eigvalsh(Q)
        ani = float((eig[1] - eig[0]) / (eig[1] + eig[0] + 1e-12)) if eig.size >= 2 else 0.0

        return dict(orient_entropy=oe, anisotropy=ani)

    def _spatial_moments(self, G):
        """Radius of gyration and weighted moment."""
        pos = nx.get_node_attributes(G, "pos")
        if not pos:
            return dict(radius_gyration=0.0, moment_total=0.0)
        coords = np.array([np.array(pos[n])[:2] for n in G.nodes])
        deg = np.array([d for _, d in G.degree()])
        cen = coords.mean(0)
        rg = float(np.sqrt(((coords - cen) ** 2).sum(1).mean()))
        mt = float(np.sum(np.linalg.norm(coords - cen, axis=1) * deg))
        return dict(radius_gyration=rg, moment_total=mt)

    def _path_connectivity(self, G):
        """Clustering, Fiedler value, spectral gap, ASPL."""
        if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
            return dict(clustering_coef=0.0, fiedler_value=0.0,
                       lambda_max=0.0, spectral_gap_ratio=0.0, aspl_giant=0.0)

        cc_coef = float(nx.average_clustering(G))

        try:
            fiedler = float(nx.algebraic_connectivity(G))
        except (nx.NetworkXError, Exception):
            fiedler = 0.0

        try:
            components = list(nx.connected_components(G))
            if components:
                gc = G.subgraph(max(components, key=len))
                aspl = float(nx.average_shortest_path_length(gc)) if gc.number_of_nodes() > 1 else 0.0
            else:
                aspl = 0.0
        except (nx.NetworkXError, Exception):
            aspl = 0.0

        lmax = 0.0
        if HAS_SCIPY and G.number_of_nodes() > 2:
            try:
                L = nx.laplacian_matrix(G).astype(float)
                lmax = float(eigsh(L, k=1, which="LA", return_eigenvectors=False)[0])
            except Exception:
                lmax = 0.0

        sgr = float(fiedler / (lmax + 1e-12)) if lmax > 0 else 0.0

        return dict(clustering_coef=cc_coef, fiedler_value=fiedler,
                    lambda_max=lmax, spectral_gap_ratio=sgr, aspl_giant=aspl)

    def _boundary_fractal(self, G):
        """Boundary fraction and box-counting fractal dimension."""
        pos = nx.get_node_attributes(G, "pos")
        ne = G.number_of_edges()
        if not pos or ne == 0:
            return dict(boundary_frac=0.0, fractal_dim_box=0.0)

        coords = np.array([np.array(pos[n])[:2] for n in G.nodes])
        eps = 1e-6
        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()

        bn = set()
        for n, p in pos.items():
            x, y = p[0], p[1]
            if abs(x - xmin) < eps or abs(x - xmax) < eps or abs(y - ymin) < eps or abs(y - ymax) < eps:
                bn.add(n)

        be = [(u, v) for u, v in G.edges if u in bn or v in bn]
        bf = len(be) / ne if ne else 0.0

        # Box-counting fractal dimension
        sizes, counts = [], []
        for k in range(1, 7):
            s = 1.0 / 2 ** k
            idx = np.floor(coords / max(s, 1e-10)).astype(int)
            counts.append(len({tuple(i) for i in idx}))
            sizes.append(1.0 / s)

        if HAS_SCIPY and len(sizes) >= 2 and all(c > 0 for c in counts):
            fd = float(linregress(np.log(sizes), np.log(counts)).slope)
        else:
            fd = 0.0

        return dict(boundary_frac=bf, fractal_dim_box=fd)

    def _redundancy_kcore(self, G):
        """Rigidity index, redundancy ratio, k-core stats."""
        nn = G.number_of_nodes()
        ne = G.number_of_edges()
        ri = float(ne - 2 * nn + 3) if ne > 0 else 0.0
        rr = float((ne - nn + 1) / ne) if ne > 0 else 0.0

        if nn > 0 and ne > 0:
            try:
                kc = max(nx.core_number(G).values())
                kcf = float(kc / nn)
            except Exception:
                kc = 0
                kcf = 0.0
        else:
            kc = 0
            kcf = 0.0

        return dict(rigidity_index=ri, redundancy_ratio=rr,
                    max_k_core=kc, kcore_frac=kcf)

    def _cycle_features(self, G):
        """Triangle and quad counts."""
        ne = G.number_of_edges()
        if ne == 0:
            return dict(triangle_count=0, triangle_ratio=0.0,
                       quad_count=0, quad_ratio=0.0)
        try:
            tri = sum(nx.triangles(G).values()) // 3
        except Exception:
            tri = 0

        # Approximate quad count from 4-cycles
        quad = 0
        if G.number_of_nodes() <= 500:
            try:
                for cycle in nx.cycle_basis(G):
                    if len(cycle) == 4:
                        quad += 1
            except Exception:
                pass

        return dict(
            triangle_count=tri,
            triangle_ratio=float(tri / ne) if ne else 0.0,
            quad_count=quad,
            quad_ratio=float(quad / ne) if ne else 0.0,
        )

    def _vertical_shortestness(self, G):
        """Average vertical shortest path and straightness."""
        pos = nx.get_node_attributes(G, "pos")
        if not pos or G.number_of_edges() == 0:
            return dict(avg_shortest_dy=0.0, straightness=0.0)

        coords = {n: np.array(pos[n])[:2] for n in G.nodes}
        ymin = min(p[1] for p in coords.values())
        ymax = max(p[1] for p in coords.values())
        if ymax - ymin < 1e-10:
            return dict(avg_shortest_dy=0.0, straightness=0.0)

        top = {n for n, p in coords.items() if abs(p[1] - ymax) < 1e-6}
        bot = {n for n, p in coords.items() if abs(p[1] - ymin) < 1e-6}

        if not top or not bot:
            return dict(avg_shortest_dy=0.0, straightness=0.0)

        # Add weight = vertical distance
        Gw = G.copy()
        for u, v in Gw.edges:
            dy = abs(coords[u][1] - coords[v][1])
            Gw.edges[u, v]["w"] = dy if dy > 0 else 1e-6

        dists = []
        for s in list(top)[:5]:  # Limit for performance
            try:
                d = nx.single_source_dijkstra_path_length(Gw, s, weight="w")
                dists.extend(d[t] for t in bot if t in d)
            except Exception:
                pass

        avg = float(np.mean(dists)) if dists else 0.0
        return dict(avg_shortest_dy=avg,
                    straightness=float(avg / (ymax - ymin + 1e-12)))

    def _mesh_holes(self, img):
        """Mesh hole area statistics from rendered image."""
        if img is None or not HAS_CV2:
            return dict(mesh_median_area=0.0, mesh_cv_area=0.0, mesh_max_area_ratio=0.0)

        res = self.canvas_size
        work = img.copy()
        cv2.floodFill(work, None, (0, 0), 128)
        mask = work == 255
        if mask.sum() == 0:
            return dict(mesh_median_area=0.0, mesh_cv_area=0.0, mesh_max_area_ratio=0.0)

        _, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        areas = st[1:, cv2.CC_STAT_AREA].astype(float)
        if areas.size == 0:
            return dict(mesh_median_area=0.0, mesh_cv_area=0.0, mesh_max_area_ratio=0.0)

        return dict(
            mesh_median_area=float(np.median(areas)),
            mesh_cv_area=float(np.std(areas) / areas.mean()) if areas.mean() > 0 else 0.0,
            mesh_max_area_ratio=float(areas.max() / (res * res)),
        )

    def _betweenness_edges(self, G):
        """Edge betweenness centrality statistics."""
        if G.number_of_edges() == 0:
            return dict(edge_betweenness_max=0.0, edge_betweenness_gini=0.0)

        try:
            if G.number_of_nodes() <= 2500:
                bc = nx.edge_betweenness_centrality(G, normalized=True)
            else:
                bc = nx.edge_betweenness_centrality(G, k=200, normalized=True, seed=0)
            vals = np.array(list(bc.values()))
            return dict(
                edge_betweenness_max=float(vals.max()) if vals.size else 0.0,
                edge_betweenness_gini=float(self._gini(vals)),
            )
        except Exception:
            return dict(edge_betweenness_max=0.0, edge_betweenness_gini=0.0)

    # ==================== Pore Features (18) ====================

    def _pore_features(self, img):
        """Pore size distribution, shape, and spatial uniformity."""
        zero = dict(
            largest_pore_ratio=0.0, top_area_sum_ratio=0.0,
            top_convexity_min=1.0, top_convexity_mean=1.0, top_circularity_min=1.0,
            big_pore_count=0, total_pore_count=0,
            total_pore_ratio=0.0, center_pore_ratio=0.0, edge_pore_ratio=0.0,
            pore_area_cv=0.0, pore_area_skew=0.0, pore_area_kurtosis=0.0,
            pore_area_max_over_mean=1.0, pore_large_area_frac=0.0,
            pore_count_large_frac=0.0, pore_density=0.0, pore_spatial_cv=0.0,
        )

        if img is None or not HAS_CV2:
            return zero

        res = self.canvas_size
        total_px = res * res
        work = img.copy()
        cv2.floodFill(work, None, (0, 0), 128)
        mask = (work == 255).astype(np.uint8)
        cc_n, labels, st, cen = cv2.connectedComponentsWithStats(mask, 8)

        if cc_n <= 1:
            return zero

        all_areas = st[1:, cv2.CC_STAT_AREA].astype(float)
        all_cxy = cen[1:]
        n_pores = len(all_areas)

        # Distribution statistics
        if n_pores >= 2 and HAS_SCIPY:
            mean_a = float(all_areas.mean())
            std_a = float(all_areas.std(ddof=0))
            pore_area_cv = std_a / mean_a if mean_a > 0 else 0.0
            pore_area_skew = float(sp_skew(all_areas))
            pore_area_kurtosis = float(sp_kurtosis(all_areas))
            pore_area_max_over_mean = float(all_areas.max() / mean_a) if mean_a > 0 else 1.0
            large_mask = all_areas > 2.0 * mean_a
            total_area_sum = float(all_areas.sum())
            pore_large_area_frac = float(all_areas[large_mask].sum() / total_area_sum) if total_area_sum > 0 else 0.0
            pore_count_large_frac = float(large_mask.sum() / n_pores)
        else:
            mean_a = float(all_areas.mean()) if n_pores > 0 else 0.0
            pore_area_cv = 0.0
            pore_area_skew = 0.0
            pore_area_kurtosis = 0.0
            pore_area_max_over_mean = float(all_areas.max() / mean_a) if (n_pores > 0 and mean_a > 0) else 1.0
            pore_large_area_frac = 0.0
            pore_count_large_frac = 0.0

        pore_density = float(n_pores / total_px)

        # Spatial uniformity (3x3 grid CV)
        grid_n = 3
        grid_area = np.zeros((grid_n, grid_n))
        for k in range(n_pores):
            cx, cy = all_cxy[k]
            xi = min(int(cx / res * grid_n), grid_n - 1)
            yi = min(int(cy / res * grid_n), grid_n - 1)
            grid_area[yi, xi] += all_areas[k]
        nz = grid_area[grid_area > 0]
        pore_spatial_cv = float(nz.std() / nz.mean()) if len(nz) > 1 else 0.0

        new_feats = dict(
            pore_area_cv=pore_area_cv,
            pore_area_skew=pore_area_skew,
            pore_area_kurtosis=pore_area_kurtosis,
            pore_area_max_over_mean=pore_area_max_over_mean,
            pore_large_area_frac=pore_large_area_frac,
            pore_count_large_frac=pore_count_large_frac,
            pore_density=pore_density,
            pore_spatial_cv=pore_spatial_cv,
        )

        # Center / edge split
        margin = res * self.edge_margin
        cp, ep = [], []
        for i in range(1, cc_n):
            a = st[i, cv2.CC_STAT_AREA]
            r = a / total_px
            cx, cy = cen[i]
            if margin < cx < res - margin and margin < cy < res - margin:
                cp.append((i, a, r))
            else:
                ep.append((i, a, r))

        tpc = len(cp) + len(ep)
        tpr = (sum(a for _, a, _ in cp) + sum(a for _, a, _ in ep)) / total_px
        cpr = sum(a for _, a, _ in cp) / total_px
        epr = sum(a for _, a, _ in ep) / total_px

        big = sorted([(i, a, r) for i, a, r in cp if r >= self.area_thresh],
                     key=lambda x: x[1], reverse=True)
        bpc = len(big)
        top = big[:self.top_k]

        base = dict(big_pore_count=bpc, total_pore_count=tpc,
                    total_pore_ratio=tpr, center_pore_ratio=cpr, edge_pore_ratio=epr)

        if not top:
            return {**zero, **base, **new_feats}

        def _shape(lid):
            m = (labels == lid).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return 1.0, 1.0
            c = cnts[0]
            ar = cv2.contourArea(c)
            ha = cv2.contourArea(cv2.convexHull(c))
            pe = cv2.arcLength(c, True)
            return (ar / ha if ha else 1.0), (4 * np.pi * ar / pe ** 2 if pe else 1.0)

        shapes = [_shape(i) for i, _, _ in top]
        ars = [r for _, _, r in top]
        return dict(
            largest_pore_ratio=ars[0], top_area_sum_ratio=sum(ars),
            top_convexity_min=min(s[0] for s in shapes),
            top_convexity_mean=float(np.mean([s[0] for s in shapes])),
            top_circularity_min=min(s[1] for s in shapes),
            **base, **new_feats)

    # ==================== Contact Features (42) ====================

    def _contact_features(self, G, id2pt):
        """Contact/overlap analysis from pixel-level edge rendering."""
        edges = list(G.edges())
        E = len(edges)
        N = G.number_of_nodes()

        if not HAS_CV2 or not id2pt or E == 0:
            return self._contact_zeros(N, E)

        node_to_edges = defaultdict(set)
        for i, (u, v) in enumerate(edges):
            node_to_edges[u].add(i)
            node_to_edges[v].add(i)
        adj_pairs = set()
        for eset in node_to_edges.values():
            lst = list(eset)
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    adj_pairs.add(tuple(sorted((lst[i], lst[j]))))

        edge_pixels = [self._get_edge_pixels(id2pt[u], id2pt[v], self.thick)
                       for u, v in edges]
        ep_counts = [len(s) for s in edge_pixels]
        ep_sum = int(np.sum(ep_counts))
        ep_union = set().union(*edge_pixels) if edge_pixels else set()
        ep_union_n = len(ep_union)

        px2e = defaultdict(list)
        for i, pxs in enumerate(edge_pixels):
            for px in pxs:
                px2e[px].append(i)

        raw_olap = set()
        olap = defaultdict(set)
        for px, el in px2e.items():
            if len(el) >= 2:
                raw_olap.add(px)
                for i in range(len(el)):
                    for j in range(i + 1, len(el)):
                        pair = tuple(sorted((el[i], el[j])))
                        if pair not in adj_pairs:
                            olap[pair].add(px)

        olap_all = set()
        ecd = np.zeros(E, dtype=int)
        for (a, b), pxs in olap.items():
            if pxs:
                olap_all.update(pxs)
                ecd[a] += 1
                ecd[b] += 1

        olap_n = len(olap_all)
        raw_n = len(raw_olap)
        pair_n = len(olap)

        cc_st = self._safe_stats(self._cc_sizes_from_pixels(olap_all, self.connectivity))
        pr_st = self._safe_stats([len(p) for p in olap.values()])
        ec_st = self._safe_stats(ecd.tolist())

        cl_len = float(sum(
            math.dist(id2pt.get(u, (0, 0)), id2pt.get(v, (0, 0)))
            for u, v in edges
        ))
        ca = self.canvas_size ** 2
        ewc = int((ecd > 0).sum())
        ol_approx = olap_n / max(self.thick, 1)

        return {
            "contact_thick": int(self.thick),
            "contact_canvas_size": int(self.canvas_size),
            "contact_nodes": N,
            "contact_edges": E,
            "contact_edge_pixel_union_count": ep_union_n,
            "contact_edge_pixel_sum": ep_sum,
            "contact_raw_overlap_pixel_count": raw_n,
            "contact_overlap_pixel_count": olap_n,
            "contact_overlap_pair_count": pair_n,
            "contact_overlap_pairs_per_edge": pair_n / E if E else 0.0,
            "contact_edges_with_contact_count": ewc,
            "contact_edges_with_contact_ratio": ewc / E if E else 0.0,
            "contact_overlap_pixel_ratio_union": olap_n / ep_union_n if ep_union_n else 0.0,
            "contact_raw_overlap_pixel_ratio_union": raw_n / ep_union_n if ep_union_n else 0.0,
            "contact_overlap_pixel_ratio_canvas": olap_n / ca if ca else 0.0,
            "contact_overlap_length_px_approx": ol_approx,
            "contact_overlap_length_ratio_centerline": ol_approx / cl_len if cl_len else 0.0,
            "contact_centerline_length_px": cl_len,
            "contact_overlap_pair_size_sum": pr_st["sum"],
            "contact_overlap_pair_size_mean": pr_st["mean"],
            "contact_overlap_pair_size_median": pr_st["median"],
            "contact_overlap_pair_size_max": pr_st["max"],
            "contact_overlap_pair_size_std": pr_st["std"],
            "contact_overlap_pair_size_q75": pr_st["q75"],
            "contact_overlap_pair_size_q90": pr_st["q90"],
            "contact_overlap_pair_size_q95": pr_st["q95"],
            "contact_overlap_cc_count": cc_st["count"],
            "contact_overlap_cc_size_sum": cc_st["sum"],
            "contact_overlap_cc_size_mean": cc_st["mean"],
            "contact_overlap_cc_size_median": cc_st["median"],
            "contact_overlap_cc_size_max": cc_st["max"],
            "contact_overlap_cc_size_std": cc_st["std"],
            "contact_overlap_cc_size_q75": cc_st["q75"],
            "contact_overlap_cc_size_q90": cc_st["q90"],
            "contact_overlap_cc_size_q95": cc_st["q95"],
            "contact_edge_contact_degree_mean": ec_st["mean"],
            "contact_edge_contact_degree_median": ec_st["median"],
            "contact_edge_contact_degree_max": ec_st["max"],
            "contact_edge_contact_degree_std": ec_st["std"],
            "contact_edge_contact_degree_q75": ec_st["q75"],
            "contact_edge_contact_degree_q90": ec_st["q90"],
            "contact_edge_contact_degree_q95": ec_st["q95"],
        }

    def _contact_zeros(self, N=0, E=0):
        """Return zero-valued contact features."""
        return {
            "contact_thick": int(self.thick),
            "contact_canvas_size": int(self.canvas_size),
            "contact_nodes": N,
            "contact_edges": E,
            "contact_edge_pixel_union_count": 0,
            "contact_edge_pixel_sum": 0,
            "contact_raw_overlap_pixel_count": 0,
            "contact_overlap_pixel_count": 0,
            "contact_overlap_pair_count": 0,
            "contact_overlap_pairs_per_edge": 0.0,
            "contact_edges_with_contact_count": 0,
            "contact_edges_with_contact_ratio": 0.0,
            "contact_overlap_pixel_ratio_union": 0.0,
            "contact_raw_overlap_pixel_ratio_union": 0.0,
            "contact_overlap_pixel_ratio_canvas": 0.0,
            "contact_overlap_length_px_approx": 0.0,
            "contact_overlap_length_ratio_centerline": 0.0,
            "contact_centerline_length_px": 0.0,
            "contact_overlap_pair_size_sum": 0.0,
            "contact_overlap_pair_size_mean": 0.0,
            "contact_overlap_pair_size_median": 0.0,
            "contact_overlap_pair_size_max": 0.0,
            "contact_overlap_pair_size_std": 0.0,
            "contact_overlap_pair_size_q75": 0.0,
            "contact_overlap_pair_size_q90": 0.0,
            "contact_overlap_pair_size_q95": 0.0,
            "contact_overlap_cc_count": 0,
            "contact_overlap_cc_size_sum": 0,
            "contact_overlap_cc_size_mean": 0.0,
            "contact_overlap_cc_size_median": 0.0,
            "contact_overlap_cc_size_max": 0,
            "contact_overlap_cc_size_std": 0.0,
            "contact_overlap_cc_size_q75": 0.0,
            "contact_overlap_cc_size_q90": 0.0,
            "contact_overlap_cc_size_q95": 0.0,
            "contact_edge_contact_degree_mean": 0.0,
            "contact_edge_contact_degree_median": 0.0,
            "contact_edge_contact_degree_max": 0,
            "contact_edge_contact_degree_std": 0.0,
            "contact_edge_contact_degree_q75": 0.0,
            "contact_edge_contact_degree_q90": 0.0,
            "contact_edge_contact_degree_q95": 0.0,
        }

    # ==================== Public API ====================

    def extract(self, obj) -> Dict[str, float]:
        """Extract 94-dimensional feature vector.

        Parameters
        ----------
        obj : nx.Graph or FiberNetwork
            The network to analyze. For nx.Graph, nodes must have
            a ``pos`` attribute (2D or 3D tuple).

        Returns
        -------
        dict
            Feature dictionary with 94 named features.

        Examples
        --------
        >>> ext = GraphFeatureExtractor(canvas_size=256)
        >>> G = my_generator.generate()  # nx.Graph with pos
        >>> features = ext.extract(G)
        >>> assert len(features) == 94
        """
        G = self._to_graph(obj)

        img, id2pt = self._render_image(G)

        feat = {}
        feat.update(self._basic_size(G))
        feat.update(self._degree_stats(G))
        feat.update(self._orientation_stats(G))
        feat.update(self._spatial_moments(G))
        feat.update(self._path_connectivity(G))
        feat.update(self._boundary_fractal(G))
        feat.update(self._redundancy_kcore(G))
        feat.update(self._cycle_features(G))
        feat.update(self._vertical_shortestness(G))
        feat.update(self._mesh_holes(img))
        feat.update(self._betweenness_edges(G))
        feat.update(self._pore_features(img))
        feat.update(self._contact_features(G, id2pt))

        return feat

    def extract_vector(self, obj) -> np.ndarray:
        """Extract features as ordered numpy array.

        Parameters
        ----------
        obj : nx.Graph or FiberNetwork
            The network to analyze.

        Returns
        -------
        np.ndarray
            94-element feature vector ordered by FEATURE_COLS.
        """
        feat = self.extract(obj)
        return np.array([feat.get(k, 0.0) for k in self.FEATURE_COLS])

    def batch_extract(self, networks) -> List[Dict[str, float]]:
        """Extract features from multiple networks.

        Parameters
        ----------
        networks : list
            List of nx.Graph or FiberNetwork objects.

        Returns
        -------
        list of dict
            Feature dictionaries.
        """
        return [self.extract(net) for net in networks]

    def batch_extract_matrix(self, networks) -> np.ndarray:
        """Extract features as a matrix.

        Parameters
        ----------
        networks : list
            List of nx.Graph or FiberNetwork objects.

        Returns
        -------
        np.ndarray
            Matrix of shape (n_networks, 94).
        """
        vectors = [self.extract_vector(net) for net in networks]
        return np.vstack(vectors) if vectors else np.empty((0, len(self.FEATURE_COLS)))

"""
94-dimensional graph feature extractor for fiber networks.

Extracts structural, pore, and contact features from fiber network graphs
for machine learning and structure-property analysis.

Accepts simple undirected NetworkX Graph and StructureGraph objects directly.
This historical feature set uses the XY projection and straight endpoint chords;
it does not describe three-dimensional or curved-fiber geometry. Parallel
physical edges are rejected because collapsing them would change fiber counts
and total length without changing the feature schema.

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
import json as _json
import sys as _sys
from numbers import Integral as _Integral, Real as _Real
from collections.abc import Mapping as _Mapping
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

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



def _mark_status(status, names, state, reason):
    if status is not None:
        for name in names:
            status[name] = {"state": state, "reason": reason}


def _image_unavailable(status, names, missing_reason):
    _mark_status(status, names, "unavailable",
                 "dependency_missing.cv2" if not HAS_CV2 else missing_reason)


class FeatureUnavailableError(RuntimeError):
    """Ordinary extraction encountered unavailable audited features.

    ``unavailable_features`` is a tuple in FEATURE_COLS order; ``reasons`` maps
    every unavailable name to its status reason in the same stable order.
    """

    def __init__(self, reasons):
        self.reasons = dict(reasons)
        self.unavailable_features = tuple(self.reasons)
        details = "; ".join(f"{name}: {reason}" for name, reason in self.reasons.items())
        super().__init__(
            "Graph features are unavailable: " + details + ". "
            "For dependency_missing.cv2, install opencv-python-headless "
            "(pip install opencv-python-headless). For input.no_positions or "
            "render_missing, provide positioned nodes and check rendering. "
            "Use extract_with_status() to inspect values and reasons, or pass "
            "allow_unavailable=True explicitly to retain legacy placeholders."
        )


def _validate_allow_unavailable(value):
    if type(value) is not bool:
        raise TypeError("allow_unavailable must be a bool")


class GraphFeatureExtractor:
    """Extract 94-dimensional feature vector from fiber network graphs.

    Accepts simple undirected ``networkx.Graph`` objects with ``pos`` node
    attributes, or ``StructureGraph`` objects without parallel edges.
    Coordinates are projected onto XY and edges are represented by endpoint
    chords. Use a geometry-aware feature set when curve shape or 3D geometry
    matters; this class preserves the historical 94-feature definitions.

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

    STATUS_SCHEMA = "fibernet.graph_features94_result/1"
    VALUE_SCHEMA = "fibernet.graph_features94_legacy/1"
    GEOMETRY_SCHEMA = "simple_undirected_xy_endpoint_chords_legacy_v1"
    DEFINITION_ID = "graph94-image-availability-v1"
    MESH_FEATURES = ("mesh_median_area", "mesh_cv_area", "mesh_max_area_ratio")
    PORE_FEATURES = tuple(FEATURE_COLS[34:52])
    CONTACT_CONFIGURATION = ("contact_thick", "contact_canvas_size",
                             "contact_nodes", "contact_edges")
    CONTACT_MEASUREMENTS = tuple(FEATURE_COLS[56:])
    AUDITED_NAMES = MESH_FEATURES + PORE_FEATURES + CONTACT_CONFIGURATION + CONTACT_MEASUREMENTS
    PARAMETER_NAMES = ("canvas_size", "thick", "edge_margin", "top_k",
                       "area_thresh", "connectivity")

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
        """Convert supported inputs without silently collapsing physical edges."""
        if not HAS_NETWORKX:
            raise ImportError("networkx is required: pip install networkx")

        if isinstance(obj, nx.Graph):
            if obj.is_multigraph() or obj.is_directed():
                raise ValueError(
                    "The 94-feature GraphFeatureExtractor requires a simple "
                    "undirected graph. MultiGraph and directed inputs are not "
                    "supported; make any topology projection explicit before "
                    "extracting this historical feature schema."
                )
            return obj

        from fibernet.core.structure_graph import StructureGraph
        if isinstance(obj, StructureGraph):
            G = nx.Graph()
            for nid, node in obj.nodes.items():
                G.add_node(nid, pos=tuple(node.position[:2]))
            for edge in obj.edges.values():
                if G.has_edge(edge.node_i, edge.node_j):
                    raise ValueError(
                        "The 94-feature GraphFeatureExtractor cannot preserve "
                        f"parallel physical edges between nodes {edge.node_i} "
                        f"and {edge.node_j}. Use a feature set that preserves "
                        "physical edge identity, or explicitly construct a "
                        "simple NetworkX graph if a topology projection is "
                        "intended."
                    )
                G.add_edge(edge.node_i, edge.node_j)
            return G

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

    def _mesh_holes(self, img, *, _status=None, _missing_reason="render_missing"):
        """Mesh hole area statistics from rendered image."""
        if img is None or not HAS_CV2:
            _image_unavailable(_status, self.MESH_FEATURES, _missing_reason)
            return dict(mesh_median_area=0.0, mesh_cv_area=0.0, mesh_max_area_ratio=0.0)

        res = self.canvas_size
        work = img.copy()
        cv2.floodFill(work, None, (0, 0), 128)
        mask = work == 255
        if mask.sum() == 0:
            _mark_status(_status, self.MESH_FEATURES, "not_applicable", "domain.no_pores")
            return dict(mesh_median_area=0.0, mesh_cv_area=0.0, mesh_max_area_ratio=0.0)

        _, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        areas = st[1:, cv2.CC_STAT_AREA].astype(float)
        if areas.size == 0:
            _mark_status(_status, self.MESH_FEATURES, "not_applicable", "domain.no_pores")
            return dict(mesh_median_area=0.0, mesh_cv_area=0.0, mesh_max_area_ratio=0.0)

        _mark_status(_status, self.MESH_FEATURES, "computed", "evaluated")
        if _status is not None and areas.mean() <= 0:
            _mark_status(_status, ("mesh_cv_area",), "not_applicable", "domain.zero_mean_area")
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

    def _pore_features(self, img, *, _status=None, _missing_reason="render_missing"):
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
            _image_unavailable(_status, self.PORE_FEATURES, _missing_reason)
            return zero

        res = self.canvas_size
        total_px = res * res
        work = img.copy()
        cv2.floodFill(work, None, (0, 0), 128)
        mask = (work == 255).astype(np.uint8)
        cc_n, labels, st, cen = cv2.connectedComponentsWithStats(mask, 8)

        if cc_n <= 1:
            _mark_status(_status, self.PORE_FEATURES, "not_applicable", "domain.no_pores")
            _mark_status(_status, ("big_pore_count", "total_pore_count", "top_area_sum_ratio"),
                         "defined_empty", "domain.no_pores")
            _mark_status(_status, ("total_pore_ratio", "center_pore_ratio", "edge_pore_ratio",
                                  "pore_density"), "computed", "domain.no_pores")
            return zero

        all_areas = st[1:, cv2.CC_STAT_AREA].astype(float)
        all_cxy = cen[1:]
        n_pores = len(all_areas)
        _mark_status(_status, self.PORE_FEATURES, "computed", "evaluated")

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

        if _status is not None:
            if n_pores >= 2 and not HAS_SCIPY:
                _mark_status(_status, ("pore_area_cv", "pore_area_skew", "pore_area_kurtosis",
                                      "pore_large_area_frac", "pore_count_large_frac"),
                             "unavailable", "dependency_missing.scipy")
            elif n_pores < 2:
                _mark_status(_status, ("pore_area_skew", "pore_area_kurtosis"),
                             "not_applicable", "domain.insufficient_pores")
                _mark_status(_status, ("pore_area_cv", "pore_large_area_frac",
                                      "pore_count_large_frac"), "fallback", "stat.single_pore_default")
            if mean_a <= 0:
                _mark_status(_status, ("pore_area_max_over_mean",),
                             "not_applicable", "domain.zero_mean_area")
                if HAS_SCIPY and n_pores >= 2:
                    _mark_status(_status, ("pore_area_cv",),
                                 "not_applicable", "domain.zero_mean_area")

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
        if len(nz) <= 1:
            _mark_status(_status, ("pore_spatial_cv",), "fallback", "stat.single_occupied_bin_default")

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
            reason = "selection.disabled_top_k" if self.top_k == 0 else "domain.no_selected_pores"
            _mark_status(_status, ("largest_pore_ratio", "top_convexity_min",
                                  "top_convexity_mean", "top_circularity_min"), "not_applicable", reason)
            _mark_status(_status, ("top_area_sum_ratio",), "defined_empty", reason)
            return {**zero, **base, **new_feats}

        def _shape(lid):
            m = (labels == lid).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                _mark_status(_status, ("top_convexity_min", "top_convexity_mean",
                                      "top_circularity_min"), "fallback", "shape.no_contour")
                return 1.0, 1.0
            c = cnts[0]
            ar = cv2.contourArea(c)
            ha = cv2.contourArea(cv2.convexHull(c))
            pe = cv2.arcLength(c, True)
            if not ha:
                _mark_status(_status, ("top_convexity_min", "top_convexity_mean"),
                             "fallback", "shape.zero_hull_area")
            if not pe:
                _mark_status(_status, ("top_circularity_min",), "fallback", "shape.zero_perimeter")
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

    def _contact_features(self, G, id2pt, *, _status=None, _missing_reason="render_missing"):
        """Contact/overlap analysis from pixel-level edge rendering."""
        edges = list(G.edges())
        E = len(edges)
        N = G.number_of_nodes()

        _mark_status(_status, self.CONTACT_CONFIGURATION, "computed", "input_recorded")
        if not HAS_CV2 or not id2pt or E == 0:
            if not HAS_CV2 or not id2pt:
                _image_unavailable(_status, self.CONTACT_MEASUREMENTS, _missing_reason)
            else:
                _mark_status(_status, self.CONTACT_MEASUREMENTS, "not_applicable", "domain.no_edges")
            return self._contact_zeros(N, E)
        _mark_status(_status, self.CONTACT_MEASUREMENTS, "computed", "evaluated")

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

        if _status is not None:
            if not pair_n:
                _mark_status(_status, (name for name in self.CONTACT_MEASUREMENTS
                                      if name.startswith("contact_overlap_pair_size_")),
                             "not_applicable", "domain.no_overlap_pairs")
                _mark_status(_status, ("contact_overlap_pair_size_sum",),
                             "defined_empty", "domain.no_overlap_pairs")
            if not cc_st["count"]:
                _mark_status(_status, (name for name in self.CONTACT_MEASUREMENTS
                                      if name.startswith("contact_overlap_cc_")),
                             "not_applicable", "domain.no_overlap_components")
                _mark_status(_status, ("contact_overlap_cc_count", "contact_overlap_cc_size_sum"),
                             "defined_empty", "domain.no_overlap_components")
            if not ep_union_n:
                _mark_status(_status, ("contact_overlap_pixel_ratio_union",
                                      "contact_raw_overlap_pixel_ratio_union"),
                             "not_applicable", "domain.zero_pixel_union")
            if not cl_len:
                _mark_status(_status, ("contact_overlap_length_ratio_centerline",),
                             "not_applicable", "domain.zero_centerline_length")
            if not ca:
                _mark_status(_status, ("contact_overlap_pixel_ratio_canvas",),
                             "not_applicable", "domain.zero_canvas_area")
            if any(not pixels for pixels in edge_pixels):
                _mark_status(_status, self.CONTACT_MEASUREMENTS, "fallback", "render_empty_edge")

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

    def extract(self, obj, *, allow_unavailable: bool = False) -> Dict[str, float]:
        """Extract 94-dimensional feature vector.

        Parameters
        ----------
        obj : nx.Graph or StructureGraph
            Simple undirected network without parallel physical edges.
            NetworkX nodes must have a ``pos`` attribute. Only XY coordinates
            and straight endpoint chords enter this historical feature set.

        allow_unavailable : bool, keyword-only
            False (default) raises FeatureUnavailableError for any audited
            unavailable feature. True preserves historical numeric placeholders.
            Use extract_with_status() to inspect availability without rejection.
            Unknown, fallback, defined_empty and not_applicable are not rejected.

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
        _validate_allow_unavailable(allow_unavailable)
        if allow_unavailable:
            return self._extract(obj)
        result = self.extract_with_status(obj)
        reasons = {name: result["status"][name]["reason"]
                   for name in self.FEATURE_COLS
                   if result["status"][name]["state"] == "unavailable"}
        if reasons:
            raise FeatureUnavailableError(reasons)
        return result["values"]

    def _extract(self, obj, *, _status=None):
        G = self._to_graph(obj)
        img, id2pt = self._render_image(G)
        status_kw = {}
        if _status is not None:
            missing_reason = "input.no_positions" if not nx.get_node_attributes(G, "pos") else "render_missing"
            status_kw = {"_status": _status, "_missing_reason": missing_reason}

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
        feat.update(self._mesh_holes(img, **status_kw))
        feat.update(self._betweenness_edges(G))
        feat.update(self._pore_features(img, **status_kw))
        feat.update(self._contact_features(G, id2pt, **status_kw))

        return feat

    def extract_with_status(self, obj) -> dict:
        """Return unchanged legacy values and explicitly scoped availability.

        All 94 status names are present; 31 unaudited structural features stay
        unknown. This Python result can contain legacy NaN/Inf and is NOT an
        unconditional strict-JSON payload. No existing training policy changes.
        """
        status = {name: {"state": "unknown", "reason": "outside_audited_scope"}
                  for name in self.FEATURE_COLS}
        values = self._extract(obj, _status=status)
        if set(values) != set(self.FEATURE_COLS):
            raise ValueError("94-feature implementation returned an unexpected name set")
        for name in self.AUDITED_NAMES:
            if not np.isfinite(values[name]):
                _mark_status(status, (name,), "fallback", "stat.nonfinite")
        parameters = {name: getattr(self, name) for name in self.PARAMETER_NAMES}
        parameters = {key: val.item() if isinstance(val, np.generic) else val
                      for key, val in parameters.items()}
        dependencies = {}
        for name, available in (("cv2", HAS_CV2), ("scipy", HAS_SCIPY),
                                ("numpy", True), ("networkx", HAS_NETWORKX)):
            module = _sys.modules.get(name)
            version = getattr(module, "__version__", None) if available else None
            dependencies[name] = {"available": bool(available),
                                  "version": str(version) if version is not None else None}
        return {"schema": self.STATUS_SCHEMA, "feature_schema": self.VALUE_SCHEMA,
                "definition_id": self.DEFINITION_ID, "geometry": self.GEOMETRY_SCHEMA,
                "feature_names": list(self.FEATURE_COLS), "audited_names": list(self.AUDITED_NAMES),
                "parameters": parameters, "dependencies": dependencies,
                "status": status, "values": values}

    def extract_vector(self, obj, *, allow_unavailable: bool = False) -> np.ndarray:
        """Extract features as ordered numpy array.

        Parameters
        ----------
        obj : nx.Graph or StructureGraph
            The network to analyze.

        allow_unavailable : bool, keyword-only
            False (default) raises FeatureUnavailableError for any audited
            unavailable feature. True preserves historical numeric placeholders.
            Use extract_with_status() to inspect availability without rejection.
            Unknown, fallback, defined_empty and not_applicable are not rejected.

        Returns
        -------
        np.ndarray
            94-element feature vector ordered by FEATURE_COLS.
        """
        _validate_allow_unavailable(allow_unavailable)
        feat = (self.extract(obj, allow_unavailable=True) if allow_unavailable
                else self.extract(obj))
        return np.array([feat.get(k, 0.0) for k in self.FEATURE_COLS])

    def batch_extract(self, networks, *, allow_unavailable: bool = False) -> List[Dict[str, float]]:
        """Extract features from multiple networks.

        Parameters
        ----------
        networks : list
            List of simple nx.Graph or StructureGraph objects.

        allow_unavailable : bool, keyword-only
            False (default) raises FeatureUnavailableError for any audited
            unavailable feature. True preserves historical numeric placeholders.
            Use extract_with_status() to inspect availability without rejection.
            Unknown, fallback, defined_empty and not_applicable are not rejected.

        Returns
        -------
        list of dict
            Feature dictionaries.
        """
        _validate_allow_unavailable(allow_unavailable)
        return [(self.extract(net, allow_unavailable=True) if allow_unavailable
                 else self.extract(net)) for net in networks]

    def batch_extract_matrix(self, networks, *, allow_unavailable: bool = False) -> np.ndarray:
        """Extract features as a matrix.

        Parameters
        ----------
        networks : list
            List of simple nx.Graph or StructureGraph objects.

        allow_unavailable : bool, keyword-only
            False (default) raises FeatureUnavailableError for any audited
            unavailable feature. True preserves historical numeric placeholders.
            Use extract_with_status() to inspect availability without rejection.
            Unknown, fallback, defined_empty and not_applicable are not rejected.

        Returns
        -------
        np.ndarray
            Matrix of shape (n_networks, 94).
        """
        _validate_allow_unavailable(allow_unavailable)
        vectors = [(self.extract_vector(net, allow_unavailable=True) if allow_unavailable
                    else self.extract_vector(net)) for net in networks]
        return np.vstack(vectors) if vectors else np.empty((0, len(self.FEATURE_COLS)))


def _feature_status_rules():
    """Frozen per-name state/reason pairs, not a client-supplied policy."""
    cls = GraphFeatureExtractor
    audited = set(cls.AUDITED_NAMES)
    rules = {name: ({("computed", "evaluated"), ("fallback", "stat.nonfinite")}
                    if name in audited else {("unknown", "outside_audited_scope")})
             for name in cls.FEATURE_COLS}

    def add(names, state, *reasons):
        for name in names:
            rules[name].update((state, reason) for reason in reasons)

    for name in cls.CONTACT_CONFIGURATION:
        rules[name] = {("computed", "input_recorded"), ("fallback", "stat.nonfinite")}
    images = cls.MESH_FEATURES + cls.PORE_FEATURES + cls.CONTACT_MEASUREMENTS
    add(images, "unavailable", "dependency_missing.cv2", "input.no_positions", "render_missing")
    add(cls.MESH_FEATURES, "not_applicable", "domain.no_pores")
    add(("mesh_cv_area", "pore_area_cv", "pore_area_max_over_mean"),
        "not_applicable", "domain.zero_mean_area")
    no_pore_counts = ("big_pore_count", "total_pore_count", "top_area_sum_ratio")
    no_pore_ratios = ("total_pore_ratio", "center_pore_ratio", "edge_pore_ratio", "pore_density")
    add(no_pore_counts, "defined_empty", "domain.no_pores")
    add(no_pore_ratios, "computed", "domain.no_pores")
    add(set(cls.PORE_FEATURES) - set(no_pore_counts + no_pore_ratios),
        "not_applicable", "domain.no_pores")
    add(("pore_area_cv", "pore_area_skew", "pore_area_kurtosis",
         "pore_large_area_frac", "pore_count_large_frac"),
        "unavailable", "dependency_missing.scipy")
    add(("pore_area_skew", "pore_area_kurtosis"), "not_applicable", "domain.insufficient_pores")
    add(("pore_area_cv", "pore_large_area_frac", "pore_count_large_frac"),
        "fallback", "stat.single_pore_default")
    add(("pore_spatial_cv",), "fallback", "stat.single_occupied_bin_default")
    add(("largest_pore_ratio", "top_convexity_min", "top_convexity_mean", "top_circularity_min"),
        "not_applicable", "domain.no_selected_pores", "selection.disabled_top_k")
    add(("top_area_sum_ratio",), "defined_empty", "domain.no_selected_pores", "selection.disabled_top_k")
    add(("top_convexity_min", "top_convexity_mean", "top_circularity_min"),
        "fallback", "shape.no_contour")
    add(("top_convexity_min", "top_convexity_mean"), "fallback", "shape.zero_hull_area")
    add(("top_circularity_min",), "fallback", "shape.zero_perimeter")
    add(cls.CONTACT_MEASUREMENTS, "not_applicable", "domain.no_edges")
    add(cls.CONTACT_MEASUREMENTS, "fallback", "render_empty_edge")
    pairs = [name for name in cls.CONTACT_MEASUREMENTS if name.startswith("contact_overlap_pair_size_")]
    comps = [name for name in cls.CONTACT_MEASUREMENTS if name.startswith("contact_overlap_cc_")]
    add(set(pairs) - {"contact_overlap_pair_size_sum"}, "not_applicable", "domain.no_overlap_pairs")
    add(("contact_overlap_pair_size_sum",), "defined_empty", "domain.no_overlap_pairs")
    add(set(comps) - {"contact_overlap_cc_count", "contact_overlap_cc_size_sum"},
        "not_applicable", "domain.no_overlap_components")
    add(("contact_overlap_cc_count", "contact_overlap_cc_size_sum"),
        "defined_empty", "domain.no_overlap_components")
    add(("contact_overlap_pixel_ratio_union", "contact_raw_overlap_pixel_ratio_union"),
        "not_applicable", "domain.zero_pixel_union")
    add(("contact_overlap_length_ratio_centerline",), "not_applicable", "domain.zero_centerline_length")
    add(("contact_overlap_pixel_ratio_canvas",), "not_applicable", "domain.zero_canvas_area")
    return rules


_FEATURE_STATUS_RULES = _feature_status_rules()
# Only documented finite default conventions can be requested. Raster failures,
# dependencies, unknowns and nonfinite results are NEVER policy exceptions.
_FEATURE_ALLOWED_CONVENTIONS = frozenset(
    (state, reason) for pairs in _FEATURE_STATUS_RULES.values() for state, reason in pairs
    if state == "not_applicable" or
    (state == "fallback" and reason.startswith(("stat.single_", "shape."))))


def _checked_feature_policy(allow_reasons):
    if isinstance(allow_reasons, (str, bytes)):
        raise ValueError("allow_reasons must be a sequence of registered convention reasons")
    try:
        requested = tuple(allow_reasons)
    except TypeError as exc:
        raise ValueError("allow_reasons must be a sequence") from exc
    known = {reason for _, reason in _FEATURE_ALLOWED_CONVENTIONS}
    if any(not isinstance(reason, str) or reason not in known for reason in requested):
        raise ValueError("allow_reasons contains a forbidden or unknown convention reason")
    return frozenset(pair for pair in _FEATURE_ALLOWED_CONVENTIONS if pair[1] in requested)


def _checked_feature_contract(contract):
    """Validate metadata only; return a conservative comparable-space identity."""
    cls = GraphFeatureExtractor
    if not isinstance(contract, _Mapping):
        raise ValueError("contract.malformed: feature_contract must be a mapping")
    required = {"schema", "feature_schema", "definition_id", "geometry", "feature_names",
                "audited_names", "parameters", "dependencies", "status"}
    if set(contract) != required:
        raise ValueError("contract.malformed: missing/extra feature_contract fields")
    expected = {"schema": cls.STATUS_SCHEMA, "feature_schema": cls.VALUE_SCHEMA,
                "definition_id": cls.DEFINITION_ID, "geometry": cls.GEOMETRY_SCHEMA}
    for field, value in expected.items():
        if contract[field] != value:
            raise ValueError(f"schema.unsupported: {field}")
    if contract["feature_names"] != list(cls.FEATURE_COLS):
        raise ValueError("contract.malformed: exact legacy94 feature_names/order required")
    if contract["audited_names"] != list(cls.AUDITED_NAMES):
        raise ValueError("contract.malformed: exact audited_names required")
    statuses = contract["status"]
    if not isinstance(statuses, _Mapping) or set(statuses) != set(cls.FEATURE_COLS):
        raise ValueError("contract.malformed: exactly94 named statuses required")
    for name in cls.FEATURE_COLS:
        record = statuses[name]
        if (not isinstance(record, _Mapping) or set(record) != {"state", "reason"}
                or not isinstance(record["state"], str) or not isinstance(record["reason"], str)
                or (record["state"], record["reason"]) not in _FEATURE_STATUS_RULES[name]):
            raise ValueError(f"contract.malformed: feature {name!r} state/reason pair")
    parameters = contract["parameters"]
    if not isinstance(parameters, _Mapping) or set(parameters) != set(cls.PARAMETER_NAMES):
        raise ValueError("contract.malformed: all six actual parameters required")
    for name in ("canvas_size", "thick", "top_k", "connectivity"):
        value = parameters[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, _Integral):
            raise ValueError(f"contract.malformed: parameter {name} must be integer, not bool")
    for name in ("edge_margin", "area_thresh"):
        value = parameters[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, _Real):
            raise ValueError(f"contract.malformed: parameter {name} must be finite real")
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"contract.malformed: parameter {name} must be finite real") from exc
        if not finite:
            raise ValueError(f"contract.malformed: parameter {name} must be finite real")
    if (parameters["canvas_size"] <= 10 or parameters["thick"] <= 0
            or parameters["top_k"] < 0 or parameters["connectivity"] not in (4, 8)
            or not 0 <= parameters["edge_margin"] < .5 or not 0 <= parameters["area_thresh"] <= 1):
        raise ValueError("contract.malformed: parameter outside checked image regime")
    deps = contract["dependencies"]
    if not isinstance(deps, _Mapping) or set(deps) != {"cv2", "scipy", "numpy", "networkx"}:
        raise ValueError("contract.malformed: dependency provenance required")
    for name, record in deps.items():
        if (not isinstance(record, _Mapping) or set(record) != {"available", "version"}
                or type(record["available"]) is not bool
                or not (record["version"] is None or isinstance(record["version"], str))
                or (not record["available"] and record["version"] is not None)):
            raise ValueError(f"contract.malformed: dependency {name}")
    try:
        _json.dumps(contract, allow_nan=False)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("contract.malformed: metadata must be strict JSON") from exc
    return tuple(contract[field] for field in ("feature_schema", "definition_id", "geometry")) + tuple(
        parameters[name] for name in cls.PARAMETER_NAMES)

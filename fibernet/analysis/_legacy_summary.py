"""Strict endpoint-only legacy summary; not a physical graph conversion.

MIT License, Copyright (c) 2026 FiberNet Contributors.
"""
import math
import numpy as np
from fibernet.core.network import FiberNetwork
from fibernet.core.fiber import Fiber


def analyze_legacy_endpoints(network):
    """Return the historical twelve fields only for unambiguous endpoint topology."""
    if not isinstance(network, FiberNetwork):
        raise TypeError('fibernet.api.analyze requires FiberNetwork; for StructureGraph use '
                        'fibernet.analysis.analyze_topology and analyze_geometry_paths '
                        '(declare length_unit and coordinate_frame); no conversion is performed')
    if isinstance(network.dimension, bool) or network.dimension not in (2, 3):
        raise ValueError('legacy summary requires dimension 2 or 3')
    if network.periodic:
        raise ValueError('legacy endpoint summary does not interpret periodic adjacency')
    if network.crosslinks:
        raise ValueError('legacy endpoint summary does not infer crosslink incidence; '
                         "prepare explicit StructureGraph connectivity and call analyze(graph, kind='topology'); "
                         'from_fiber_network is approximate, not lossless')
    endpoints, pairs, lengths, tortuosities = {}, set(), [], []
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        for fiber in network.fibers:
            if not isinstance(fiber, Fiber):
                raise TypeError('fibers must contain Fiber instances')
            points = fiber.centerline
            if (not isinstance(points, np.ndarray) or np.ma.isMaskedArray(points)
                    or points.dtype != np.dtype('float64') or points.ndim != 2
                    or points.shape[1] != 3 or len(points) < 2
                    or not np.isfinite(points).all()):
                raise ValueError('centerline must be a finite float64 (N, 3) array, N >= 2')
            if network.dimension == 2 and np.any(points[:, 2] != points[0, 2]):
                raise ValueError('2D fiber centerlines must lie in an XY-parallel plane')
            differences = np.diff(points, axis=0)
            segments = np.linalg.norm(differences, axis=1)
            if np.any((segments == 0) & np.any(differences != 0, axis=1)):
                raise ValueError('segment norm underflow in legacy morphology')
            chord = float(np.linalg.norm(points[-1] - points[0]))
            if not math.isfinite(chord) or chord <= 1e-12:
                raise ValueError('endpoint chord must exceed legacy orientation threshold 1e-12')
            length = float(fiber.length)
            tortuosity = float(fiber.tortuosity())
            if not math.isfinite(length) or not math.isfinite(tortuosity):
                raise ValueError('nonfinite legacy morphology')
            keys = []
            for point in (points[0], points[-1]):
                exact, rounded = tuple(point), tuple(np.round(point, decimals=6))
                if rounded in endpoints and endpoints[rounded] != exact:
                    raise ValueError('distinct endpoints collide under legacy six-decimal rounding')
                endpoints[rounded] = exact
                keys.append(rounded)
            pair = frozenset(keys)
            if len(pair) != 2 or pair in pairs:
                raise ValueError('legacy endpoint summary cannot collapse loops or parallel fibers')
            pairs.add(pair)
            lengths.append(length)
            tortuosities.append(tortuosity)
        # Direct reuse: no density, pore, all-pairs path or centrality report.
        from fibernet.analysis._archived.morphology import MorphologyAnalyzer
        from fibernet.analysis._archived.topology import TopologyAnalyzer
        import networkx as nx
        graph = TopologyAnalyzer(network).graph
        n, e = graph.number_of_nodes(), graph.number_of_edges()
        components = nx.number_connected_components(graph)
        total = math.fsum(lengths)
        mean_tortuosity = math.fsum(tortuosities) / len(tortuosities) if tortuosities else 0.0
        nematic = MorphologyAnalyzer(network).nematic_order_parameter()
        if not all(math.isfinite(value) for value in (total, mean_tortuosity, nematic)):
            raise ValueError('nonfinite legacy summary')
    return {
        'num_fibers': network.num_fibers,
        'num_crosslinks': network.num_crosslinks,
        'dimension': int(network.dimension),
        'nematic_order': float(nematic),
        'mean_length': total / len(lengths) if lengths else 0.0,
        'total_length': total,
        'mean_tortuosity': mean_tortuosity,
        'num_nodes': n,
        'num_edges': e,
        'mean_degree': 2.0 * e / n if n else 0.0,
        'is_connected': bool(n and components == 1),
        'num_components': components,
    }

"""
Feature Extraction for Fiber Networks

Converts fiber network structures into feature vectors suitable for ML models.
"""

import numpy as np
import warnings
from numbers import Integral
from typing import Dict, List, Optional, Any
from fibernet.core.network import FiberNetwork


def _require_finite(value, quantity):
    """Reject invalid derived quantities before normalization or clipping."""
    if not np.isfinite(value).all():
        raise ValueError(f'Legacy GNN {quantity} is not finite; rescale the input explicitly')
    return value


def _checked_norm(vector, quantity):
    """Reject a zero norm of a nonzero vector, including silent BLAS underflow."""
    value = _require_finite(np.linalg.norm(vector), quantity)
    if value == 0 and np.any(vector != 0):
        raise ValueError(f'Legacy GNN {quantity} underflowed; rescale the input explicitly')
    return value


def _float32_features(values, shape, quantity):
    """Cast only when finite values stay finite and nonzero values stay nonzero."""
    values = np.asarray(values, dtype=np.float64).reshape(shape)
    _require_finite(values, quantity)
    if np.any(np.abs(values) > np.finfo(np.float32).max):
        raise ValueError(f'Legacy GNN {quantity} exceeds float32 range; rescale the input explicitly')
    # Subnormal rounding is checked below rather than returned as a false zero.
    with np.errstate(under='ignore'):
        result = values.astype(np.float32)
    if np.any((values != 0) & (result == 0)):
        raise ValueError(f'Legacy GNN {quantity} underflows float32; rescale the input explicitly')
    return result


class FeatureExtractor:
    """Extract basic counts without substituting a different feature schema.

    Only the all-groups-disabled configuration is currently supported. Enabled
    legacy groups raise NotImplementedError with migration guidance; they never
    return placeholder zeros. This class is not the 94-feature
    GraphFeatureExtractor, and existing trained models must not interchange them.
    
    Parameters
    ----------
    include_morphology : bool
        Include morphological features (length, orientation, tortuosity)
    include_topology : bool
        Include topological features (degree, connectivity, components)
    include_spectral : bool
        Include spectral features (eigenvalues, spectral gap)
    include_pore : bool
        Include pore structure features (pore size distribution)
    include_anisotropy : bool
        Include anisotropy features (orientation tensors)
    """
    
    def __init__(
        self,
        include_morphology: bool = True,
        include_topology: bool = True,
        include_spectral: bool = True,
        include_pore: bool = True,
        include_anisotropy: bool = True,
    ):
        self.include_morphology = include_morphology
        self.include_topology = include_topology
        self.include_spectral = include_spectral
        self.include_pore = include_pore
        self.include_anisotropy = include_anisotropy

    def _require_supported_groups(self) -> None:
        reasons = {
            'morphology': 'std_radius has no historical report definition',
            'topology': 'legacy graph projection and report field mapping are unresolved',
            'spectral': 'eigenvalue method and missing-mode schema require migration',
            'pore': 'sampled-vertex pore approximation requires an explicit contract',
            'anisotropy': 'legacy 3D tensor and empty-input conventions require an explicit contract',
        }
        enabled = [name for name in reasons if getattr(self, 'include_' + name)]
        if enabled:
            details = '; '.join(name + ': ' + reasons[name] for name in enabled)
            raise NotImplementedError(
                'Legacy FeatureExtractor groups for FiberNetwork are unavailable (' + details + '). '
                'For counts only, construct FeatureExtractor(include_morphology=False, '
                'include_topology=False, include_spectral=False, include_pore=False, '
                'include_anisotropy=False). No complete legacy vector is returned. '
                'GraphFeatureExtractor is a different 94-feature schema, not a drop-in replacement.'
            )
    
    def extract(self, network: FiberNetwork) -> Dict[str, float]:
        """Return basic counts, or fail explicitly for enabled legacy groups.
        
        Parameters
        ----------
        network : FiberNetwork
            Input fiber network
            
        Returns
        -------
        Dict[str, float]
            Feature dictionary with string keys and numeric values
        """
        if not isinstance(network, FiberNetwork):
            raise TypeError('FeatureExtractor requires FiberNetwork; no implicit StructureGraph conversion')
        self._require_supported_groups()
        return {
            'num_fibers': network.num_fibers,
            'num_crosslinks': network.num_crosslinks,
            'dimension': network.dimension,
        }
    
    def extract_to_array(self, network: FiberNetwork) -> np.ndarray:
        """Extract features as a numpy array.
        
        Parameters
        ----------
        network : FiberNetwork
            Input fiber network
            
        Returns
        -------
        np.ndarray
            1D array of feature values
        """
        features = self.extract(network)
        return np.array(list(features.values()))
    
    def get_feature_names(self) -> List[str]:
        """Return supported names without generating or analyzing a network.

        Enabled legacy groups raise the same diagnostic as extract. This is not
        a declaration that a complete 38-value legacy vector is available.
        
        Returns
        -------
        List[str]
            Feature names in consistent order
        """
        self._require_supported_groups()
        return ['num_fibers', 'num_crosslinks', 'dimension']


def extract_features(
    network: FiberNetwork,
    as_array: bool = False,
) -> Any:
    """Legacy convenience entry; raises until enabled groups are migrated.

    To request basic counts only, use FeatureExtractor with all five include
    flags False. This function never silently substitutes a different schema.
    
    Parameters
    ----------
    network : FiberNetwork
        Input fiber network
    as_array : bool
        If True, return numpy array; otherwise return dict
        
    Returns
    -------
    Dict[str, float] or np.ndarray
        Extracted features
    """
    extractor = FeatureExtractor()
    if as_array:
        return extractor.extract_to_array(network)
    else:
        return extractor.extract(network)


class GNNFeatureExtractor:
    """
    Extract graph neural network features from fiber networks.
    
    Converts fiber networks into graph representations suitable for GNN models.
    Supports PyTorch Geometric and DGL formats.

    This is a simple fiber-contact graph: fibers are nodes indexed by list
    position, and crosslinks are edges. It is not a geometric junction graph or
    a lossless physical serialization. Repeated contacts and self-contacts are
    rejected. Materials, crosslink constitutive data and full centerlines are
    not encoded. Position is the endpoint midpoint (not the arc centroid),
    angle uses endpoint chords, and length is arc length. The optional weight
    column remains the historical constant 1, not stiffness or strength.
    Angle normalization retains ``norm + 1e-10``: at small chord lengths this
    is a regularized feature, not the exact geometric angle. Selected angle
    features reject chord norms <=1e-12, including for isolated fibers.

    Feature lists retain their historical membership semantics and fixed output
    order: position, degree, centrality, then length/radius for nodes; length,
    angle, weight for edges. Empty lists retain the historical default selection.
    Each undirected edge appears once; no reverse edges are added for ML tools.
    Nonfinite derived geometry and float32 overflow/round-to-zero underflow are
    rejected rather than clipped or replaced. Caller-directed rescaling must
    include its own unit/schema policy; this class never rescales automatically.
    
    Parameters
    ----------
    node_features : list of str
        Node feature names to extract. Options: 'position', 'degree', 'centrality'
    edge_features : list of str
        Edge feature names to extract. Options: 'length', 'angle', 'weight'
    """
    
    def __init__(
        self,
        node_features: Optional[List[str]] = None,
        edge_features: Optional[List[str]] = None,
    ):
        self.node_features = node_features or ['position', 'degree']
        self.edge_features = edge_features or ['length', 'angle']
    
    def extract_graph(self, network: FiberNetwork) -> Dict[str, Any]:
        """
        Extract graph representation from fiber network.
        
        Parameters
        ----------
        network : FiberNetwork
            Input fiber network
            
        Returns
        -------
        Dict[str, Any]
            Graph data with node_features, edge_index, edge_features
        """
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise', under='raise'):
                # Keep the historical formulas, but reject unsafe intermediates.
                G = self._to_networkx(network)
                node_feat_matrix = self._extract_node_features(G, network)
                edge_index, edge_feat_matrix = self._extract_edge_features(G, network)
        except FloatingPointError as exc:
            raise ValueError(
                'Legacy GNN geometry or features exceeded the numerical range; '
                'rescale the input explicitly rather than using placeholder values'
            ) from exc
        
        return {
            'node_features': node_feat_matrix,
            'edge_index': edge_index,
            'edge_features': edge_feat_matrix,
            'num_nodes': node_feat_matrix.shape[0],
            'num_edges': edge_index.shape[1],
            'graph': G,
        }
    
    def to_pytorch_geometric(self, network: FiberNetwork, label: Optional[float] = None):
        """
        Convert to PyTorch Geometric Data object.
        
        Parameters
        ----------
        network : FiberNetwork
            Input fiber network
        label : float, optional
            Graph-level label for supervised learning
            
        Returns
        -------
        torch_geometric.data.Data
            PyTorch Geometric data object
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError("PyTorch Geometric required. Install with: pip install torch torch-geometric")
        
        graph_data = self.extract_graph(network)
        
        # Convert to tensors
        x = torch.tensor(graph_data['node_features'], dtype=torch.float32)
        edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(graph_data['edge_features'], dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        if label is not None:
            data.y = torch.tensor([label], dtype=torch.float32)
        
        return data
    
    def to_dgl(self, network: FiberNetwork, label: Optional[float] = None):
        """
        Convert to DGL graph object.
        
        Parameters
        ----------
        network : FiberNetwork
            Input fiber network
        label : float, optional
            Graph-level label
            
        Returns
        -------
        dgl.DGLGraph
            DGL graph object
        """
        try:
            import torch
            import dgl
        except ImportError:
            raise ImportError("DGL required. Install with: pip install dgl")
        
        graph_data = self.extract_graph(network)
        
        # Create DGL graph
        edge_index = graph_data['edge_index']
        src = edge_index[0]
        dst = edge_index[1]
        
        g = dgl.graph((src, dst), num_nodes=graph_data['num_nodes'])
        
        # Add node features
        g.ndata['feat'] = torch.tensor(graph_data['node_features'], dtype=torch.float32)
        
        # Add edge features
        g.edata['feat'] = torch.tensor(graph_data['edge_features'], dtype=torch.float32)
        
        # Add label
        if label is not None:
            g.label = torch.tensor([label], dtype=torch.float32)
        
        return g
    
    def _to_networkx(self, network: FiberNetwork):
        """Convert fiber network to NetworkX graph."""
        self._validate_network(network)
        import networkx as nx
        
        G = nx.Graph()
        
        # Add nodes (fibers)
        for i, fiber in enumerate(network.fibers):
            pos = (fiber.start_point + fiber.end_point) / 2  # Center position
            _require_finite(pos, f'fiber {i} endpoint midpoint')
            length = _require_finite(fiber.length, f'fiber {i} arc length')
            if length == 0 and np.any(fiber.centerline[1:] != fiber.centerline[:-1]):
                raise ValueError(f'Legacy GNN fiber {i} arc length underflowed; rescale the input explicitly')
            G.add_node(i, pos=pos, length=length, radius=fiber.radius)
        
        # Add edges (crosslinks)
        for crosslink in network.crosslinks:
            i = crosslink.fiber_i
            j = crosslink.fiber_j
            G.add_edge(i, j, position=crosslink.position)
        
        return G

    def _validate_network(self, network: FiberNetwork) -> None:
        """Reject silent contact collapse; disclose the historical projection."""
        if not isinstance(network, FiberNetwork):
            raise TypeError('GNNFeatureExtractor requires FiberNetwork; no implicit StructureGraph conversion')
        has_internal_points = False
        for index, fiber in enumerate(network.fibers):
            points = np.asarray(fiber.centerline)
            if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                    or not np.isfinite(points).all()):
                raise ValueError(f'fiber {index} requires a finite centerline with shape (N, 3), N >= 2')
            if points.dtype != np.dtype(np.float64):
                raise ValueError(
                    f'fiber {index} centerline must retain the Fiber constructor float64 dtype; '
                    'reconstruct the Fiber explicitly instead of mutating its coordinate dtype'
                )
            if not np.isfinite(fiber.radius) or fiber.radius < 0:
                raise ValueError(f'fiber {index} requires a finite nonnegative radius')
            if 'angle' in self.edge_features:
                chord = _require_finite(points[-1] - points[0], f'fiber {index} endpoint chord')
                norm = _checked_norm(chord, f'fiber {index} endpoint-chord norm')
                if norm <= 1e-12:
                    raise ValueError(f'fiber {index} has no defined endpoint-chord angle')
            has_internal_points = has_internal_points or len(points) > 2

        pairs = set()
        for index, crosslink in enumerate(network.crosslinks):
            i, j = crosslink.fiber_i, crosslink.fiber_j
            if any(not isinstance(k, Integral) or isinstance(k, (bool, np.bool_))
                   or k < 0 or k >= network.num_fibers for k in (i, j)):
                raise ValueError(f'crosslink {index} must reference existing integer fiber list indices')
            if i == j:
                raise ValueError(f'crosslink {index}: self-contacts are not supported by this legacy GNN projection')
            pair = (min(i, j), max(i, j))
            if pair in pairs:
                raise ValueError(
                    f'crosslink {index}: repeated fiber-pair contacts would be collapsed by the '
                    'legacy simple GNN graph; choose an explicit contact representation instead'
                )
            pairs.add(pair)
            position = np.asarray(crosslink.position)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError(f'crosslink {index} requires a finite 3D position')

        if has_internal_points and ('position' in self.node_features
                                    or 'length' in self.edge_features
                                    or 'angle' in self.edge_features):
            warnings.warn(
                'Legacy GNN geometry uses endpoint midpoints and chord angles; internal '
                'centerline points are not encoded except through fiber arc length. '
                'This is not a lossless curved-fiber representation.',
                UserWarning, stacklevel=3,
            )
        if 'weight' in self.edge_features:
            warnings.warn(
                'Legacy GNN weight is the constant 1.0, not crosslink stiffness or strength.',
                UserWarning, stacklevel=3,
            )
    
    def _extract_node_features(self, G, network: FiberNetwork) -> np.ndarray:
        """Extract node feature matrix."""
        import networkx as nx
        
        features = []
        centrality = nx.betweenness_centrality(G) if 'centrality' in self.node_features else None
        
        for node in G.nodes():
            node_feat = []
            
            if 'position' in self.node_features:
                pos = G.nodes[node]['pos']
                node_feat.extend(pos)
            
            if 'degree' in self.node_features:
                degree = G.degree(node)
                node_feat.append(degree)
            
            if 'centrality' in self.node_features:
                node_feat.append(centrality[node])
            
            # Add fiber properties
            fiber = network.fibers[node]
            node_feat.extend([fiber.length, fiber.radius])
            
            features.append(node_feat)
        
        width = (2 + 3 * ('position' in self.node_features)
                 + ('degree' in self.node_features) + ('centrality' in self.node_features))
        return _float32_features(features, (G.number_of_nodes(), width), 'node features')
    
    def _extract_edge_features(self, G, network: FiberNetwork) -> tuple:
        """Extract edge index and edge feature matrix."""
        edges = list(G.edges())
        
        if len(edges) == 0:
            # No edges
            width = sum(name in self.edge_features for name in ('length', 'angle', 'weight'))
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, width), dtype=np.float32)
        
        # Edge index (source, target)
        edge_index = np.array(edges, dtype=np.int64).T
        
        # Edge features
        edge_features = []
        for u, v in edges:
            edge_feat = []
            
            if 'length' in self.edge_features:
                # Distance between fiber centers
                pos_u = G.nodes[u]['pos']
                pos_v = G.nodes[v]['pos']
                delta = _require_finite(np.array(pos_v) - np.array(pos_u), 'fiber midpoint difference')
                dist = _checked_norm(delta, 'fiber midpoint distance')
                edge_feat.append(dist)
            
            if 'angle' in self.edge_features:
                # Angle between fibers
                fiber_u = network.fibers[u]
                fiber_v = network.fibers[v]
                
                dir_u = _require_finite(fiber_u.end_point - fiber_u.start_point, 'source endpoint chord')
                dir_v = _require_finite(fiber_v.end_point - fiber_v.start_point, 'target endpoint chord')
                
                norm_u = _checked_norm(dir_u, 'source endpoint-chord norm')
                norm_v = _checked_norm(dir_v, 'target endpoint-chord norm')
                dir_u = dir_u / (norm_u + 1e-10)
                dir_v = dir_v / (norm_v + 1e-10)
                
                raw_cosine = _require_finite(np.dot(dir_u, dir_v), 'chord cosine')
                cos_angle = np.clip(raw_cosine, -1, 1)
                angle = np.arccos(cos_angle)
                edge_feat.append(angle)
            
            if 'weight' in self.edge_features:
                # Crosslink stiffness or strength
                edge_feat.append(1.0)  # Default weight
            
            edge_features.append(edge_feat)
        
        width = sum(name in self.edge_features for name in ('length', 'angle', 'weight'))
        edge_feat_matrix = _float32_features(edge_features, (len(edges), width), 'edge features')
        
        return edge_index, edge_feat_matrix
    
    def create_dataset(
        self,
        networks: List[FiberNetwork],
        labels: Optional[List[float]] = None,
        format: str = 'pyg',
    ) -> Any:
        """
        Create a dataset from multiple networks.
        
        Parameters
        ----------
        networks : list of FiberNetwork
            List of fiber networks
        labels : list of float, optional
            Labels for each network
        format : str
            Output format: 'pyg' (PyTorch Geometric) or 'dgl'
            
        Returns
        -------
        Dataset object
        """
        if format == 'pyg':
            try:
                from torch_geometric.data import InMemoryDataset
            except ImportError:
                raise ImportError("PyTorch Geometric required")
            
            data_list = []
            for i, network in enumerate(networks):
                label = labels[i] if labels is not None else None
                data = self.to_pytorch_geometric(network, label)
                data_list.append(data)
            
            return data_list
        
        elif format == 'dgl':
            try:
                import dgl
            except ImportError:
                raise ImportError("DGL required")
            
            graph_list = []
            for i, network in enumerate(networks):
                label = labels[i] if labels is not None else None
                g = self.to_dgl(network, label)
                graph_list.append(g)
            
            return graph_list
        
        else:
            raise ValueError(f"Unknown format: {format}. Use 'pyg' or 'dgl'")

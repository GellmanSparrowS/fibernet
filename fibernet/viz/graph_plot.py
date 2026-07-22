"""
Professional graph visualization for fiber networks.

Follows the principle from the research workflow: nodes are hidden,
only edges (fibers) are drawn. This focuses attention on the fiber
structure rather than individual vertices.
"""

import numpy as np
from typing import Optional, Tuple
from pathlib import Path

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_graph(G: "nx.Graph",
               pos_attr: str = "pos",
               show_nodes: bool = False,
               show_edges: bool = True,
               edge_color: str = "black",
               edge_width: float = 2.0,
               node_color: str = "red",
               node_size: int = 10,
               figsize: Tuple[float, float] = (10, 10),
               dpi: int = 150,
               title: Optional[str] = None,
               save_path: Optional[str] = None,
               ax=None,
               **kwargs):
    """Plot a fiber network graph.
    
    By default, only edges are shown (nodes hidden) to focus on
    the fiber structure.
    
    Parameters
    ----------
    G : nx.Graph
        Graph with position attributes.
    pos_attr : str
        Name of position attribute.
    show_nodes : bool
        If True, draw nodes. Default False (hide nodes).
    show_edges : bool
        If True, draw edges. Default True.
    edge_color : str
        Edge color.
    edge_width : float
        Edge line width.
    node_color : str
        Node color (if show_nodes=True).
    node_size : int
        Node size (if show_nodes=True).
    figsize : tuple
        Figure size.
    dpi : int
        Resolution.
    title : str, optional
        Figure title.
    save_path : str, optional
        Path to save figure.
    ax : matplotlib axes, optional
        Existing axes to draw on.
    
    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is required: pip install networkx")
    if not HAS_MPL:
        raise ImportError("matplotlib is required: pip install matplotlib")
    
    pos = nx.get_node_attributes(G, pos_attr)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    
    # Draw edges using LineCollection for performance
    if show_edges and pos:
        edge_coords = []
        for u, v in G.edges():
            if u in pos and v in pos:
                pu = np.array(pos[u])[:2]
                pv = np.array(pos[v])[:2]
                edge_coords.append([pu, pv])
        
        if edge_coords:
            lc = LineCollection(edge_coords, colors=edge_color,
                               linewidths=edge_width, zorder=1)
            ax.add_collection(lc)
    
    # Draw nodes (optional)
    if show_nodes:
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color,
                              node_size=node_size)
    
    # Formatting
    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=14, pad=10)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    return fig, ax


def plot_graph_comparison(G_list, labels=None, 
                          pos_attr: str = "pos",
                          figsize: Tuple[float, float] = (15, 5),
                          save_path: Optional[str] = None,
                          **kwargs):
    """Plot multiple graphs side by side for comparison.
    
    Parameters
    ----------
    G_list : list of nx.Graph
        Graphs to compare.
    labels : list of str, optional
        Labels for each graph.
    pos_attr : str
        Position attribute name.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    fig, axes
        Matplotlib figure and axes array.
    """
    if not HAS_MPL:
        raise ImportError("matplotlib is required: pip install matplotlib")
    
    n = len(G_list)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    if labels is None:
        labels = [f"Graph {i+1}" for i in range(n)]
    
    for G, label, ax in zip(G_list, labels, axes):
        plot_graph(G, pos_attr=pos_attr, ax=ax, **kwargs)
        ax.set_title(label, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    
    return fig, axes


def plot_structure_stats(G, pos_attr: str = "pos",
                         figsize: Tuple[float, float] = (12, 4),
                         save_path: Optional[str] = None):
    """Plot structure with basic statistics.
    
    Creates a 3-panel figure:
    1. Structure visualization
    2. Degree distribution
    3. Edge length distribution
    """
    if not HAS_MPL:
        raise ImportError("matplotlib is required: pip install matplotlib")
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Panel 1: Structure
    plot_graph(G, pos_attr=pos_attr, ax=axes[0])
    axes[0].set_title(f"Structure\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Panel 2: Degree distribution
    degrees = [d for _, d in G.degree()]
    axes[1].hist(degrees, bins=range(min(degrees), max(degrees)+2), 
                edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Degree')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Degree Distribution')
    
    # Panel 3: Edge length distribution
    pos = nx.get_node_attributes(G, pos_attr)
    lengths = []
    for u, v in G.edges():
        pu = np.array(pos[u])[:2]
        pv = np.array(pos[v])[:2]
        lengths.append(np.linalg.norm(pv - pu))
    
    axes[2].hist(lengths, bins=30, edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Edge Length')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'Edge Length Distribution\n(mean={np.mean(lengths):.4f})')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    
    return fig, axes

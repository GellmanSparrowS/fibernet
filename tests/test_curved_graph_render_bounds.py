"""The complete rendered fiber stays inside the view in 2D and 3D."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fibernet.core.structure_graph import StructureGraph


@pytest.mark.parametrize("dimensions", [2, 3])
def test_curved_fiber_is_inside_plot_bounds(dimensions):
    from fibernet.viz.render import render_graph, render_graph_3d
    import matplotlib.pyplot as plt

    graph = StructureGraph(dimension=dimensions)
    left = graph.add_node([0., 0., 0.])
    right = graph.add_node([1., 0., 0.])
    points = np.array([[0.25, -2., 0. if dimensions == 2 else -3.],
                       [0.75, 3., 0. if dimensions == 2 else 4.]])
    graph.add_edge(left, right, internal_points=points)
    figure = (render_graph(graph) if dimensions == 2 else render_graph_3d(graph))
    try:
        axis = figure.axes[0]
        limits = [axis.get_xlim(), axis.get_ylim()]
        if dimensions == 3:
            limits.append(axis.get_zlim())
        for dim, (lower, upper) in enumerate(limits):
            assert lower < points[:, dim].min()
            assert upper > points[:, dim].max()
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_length_coloring_accepts_edgeless_graph():
    from fibernet.viz.render import render_graph
    import matplotlib.pyplot as plt

    graph = StructureGraph(dimension=2)
    graph.add_node([0., 0.])
    figure = render_graph(graph, color_by="length", show_nodes=True)
    figure.canvas.draw()
    plt.close(figure)


def test_render_import_preserves_explicit_matplotlib_backend():
    root = Path(__file__).resolve().parents[1]
    code = """
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
existing = plt.figure()
from fibernet.viz.render import render_graph
assert matplotlib.get_backend().lower() == 'svg'
assert plt.fignum_exists(existing.number)
plt.close(existing)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root)), capture_output=True,
        text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

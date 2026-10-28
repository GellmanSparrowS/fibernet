"""Colormap contracts using installed Matplotlib and a 3.5 API-surface fixture.

The fixture is NOT an actual Matplotlib 3.5 runtime. PyVista is a call recorder,
not a renderer: these tests do not require PyVista, VTK, a display, or a GPU.
"""
from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.core.structure_graph import StructureGraph
from fibernet.viz import plot2d, render3d


class RecordingPlotter:
    def __init__(self, **kwargs):
        self.meshes = []

    def add_mesh(self, mesh, **kwargs):
        self.meshes.append((mesh, kwargs))

    def add_axes(self):
        pass


def spline(points, **kwargs):
    return SimpleNamespace(tube=lambda **kwargs: np.asarray(points).copy())


class TestColormapCompatibility(unittest.TestCase):
    registry_surface_fixture = False

    def setUp(self):
        self.network = FiberNetwork(fibers=[
            Fiber([[0, 0, 0], [1, 0, 0]]),
            Fiber([[1, 0, 0], [1, 2, 0]]),
        ], dimension=2)
        self.result = {
            'trajectory': [np.array([[0., 0., 0.], [1.1, 0., 0.], [1.1, 2., 0.]])],
            'edges': [(0, 1), (1, 2)], 'rest_lengths': [1., 2.],
        }
        self.custom = ListedColormap(['red', 'blue'], name='custom-unregistered')

    def tearDown(self):
        plt.close('all')

    def assert_palette(self, figure, expected):
        actual = figure.axes[0].collections[0].cmap
        if isinstance(expected, str):
            self.assertEqual(actual.name, expected)
        else:
            self.assertIs(actual, expected)

    def test_dynamics_named_palette(self):
        self.assert_palette(plot2d.plot_dynamics_result(self.result, colormap='plasma'), 'plasma')

    def test_metamaterial_named_palette(self):
        self.assert_palette(plot2d.plot_metamaterial(self.network, colormap='plasma'), 'plasma')

    def test_dynamics_custom_and_default_palette(self):
        self.assert_palette(plot2d.plot_dynamics_result(self.result, colormap=self.custom), self.custom)
        with matplotlib.rc_context({'image.cmap': 'magma'}):
            self.assert_palette(plot2d.plot_dynamics_result(self.result, colormap=None), 'magma')

    def test_metamaterial_custom_and_default_palette(self):
        self.assert_palette(plot2d.plot_metamaterial(self.network, colormap=self.custom), self.custom)
        with matplotlib.rc_context({'image.cmap': 'magma'}):
            self.assert_palette(plot2d.plot_metamaterial(self.network, colormap=None), 'magma')

    def check_network_palette(self, color_by):
        adapter = SimpleNamespace(Plotter=RecordingPlotter, Spline=spline)
        with patch.object(render3d, '_require_pyvista', return_value=adapter):
            for value in ('plasma', None, self.custom):
                with self.subTest(value=str(value)):
                    with matplotlib.rc_context({'image.cmap': 'magma'}):
                        expected = plt.get_cmap(value)
                        result = render3d.render_network_3d(
                            self.network, color_by=color_by, colormap=value,
                            show_crosslinks=False, show_axes=False,
                        )
                    self.assertEqual(len(result.meshes), 2)
                    for index, (_, kwargs) in enumerate(result.meshes):
                        if color_by == 'orientation':
                            direction = self.network.fibers[index].direction
                            normalized = (np.arctan2(direction[1], direction[0]) + np.pi) / (2 * np.pi)
                        else:
                            normalized = float(index)
                        self.assertEqual(kwargs['color'], [int(c * 255) for c in expected(normalized)[:3]])

    def test_network_orientation_palette(self):
        self.check_network_palette('orientation')

    def test_network_length_palette(self):
        self.check_network_palette('length')

    def test_graph_fixed_palettes(self):
        graph = StructureGraph(dimension=3)
        nodes = [graph.add_node(p) for p in ([0, 0, 0], [1, 0, 0], [1, 2, 0])]
        graph.add_edge(nodes[0], nodes[1])
        graph.add_edge(nodes[1], nodes[2])
        result = SimpleNamespace(
            metadata={}, deformed_positions=None, positions_trajectory=[],
            displacements=np.array([[0., 0., 0.], [.1, 0., 0.], [.2, 0., 0.]]),
            edge_forces=np.array([1., 3.]), edge_stretches=np.array([1., 1.1]),
        )
        adapter = SimpleNamespace(Plotter=RecordingPlotter, Spline=spline)
        with patch.object(render3d, '_require_pyvista', return_value=adapter):
            for mode, palette in [('stress', 'RdYlBu_r'), ('stretch', 'hot'),
                                  ('displacement', 'coolwarm'), ('orientation', 'hsv')]:
                with self.subTest(mode=mode):
                    output = render3d.render_structure_graph_3d(graph, result, color_by=mode, show_nodes=False)
                    self.assertEqual(len(output.meshes), 2)
                    normalized = {'stress': [0., 1.], 'stretch': [0., 1.],
                                  'displacement': [.25, .75], 'orientation': [.5, .75]}[mode]
                    for index, (_, kwargs) in enumerate(output.meshes):
                        self.assertEqual(kwargs['color'], [int(c * 255) for c in plt.get_cmap(palette)(normalized[index])[:3]])

    def test_network_invalid_palette_errors(self):
        adapter = SimpleNamespace(Plotter=RecordingPlotter, Spline=spline)
        with patch.object(render3d, '_require_pyvista', return_value=adapter):
            for value, error in [('definitely-not-a-colormap', ValueError), (42, TypeError)]:
                with self.subTest(value=value), self.assertRaises(error):
                    render3d.render_network_3d(self.network, color_by='length', colormap=value)

    def test_resolver_contract(self):
        resolve = render3d._resolve_colormap
        self.assertEqual(resolve('plasma').name, 'plasma')
        self.assertIs(resolve(self.custom), self.custom)
        with matplotlib.rc_context({'image.cmap': 'magma'}):
            self.assertEqual(resolve(None).name, 'magma')
        with self.assertRaises(ValueError):
            resolve('definitely-not-a-colormap')
        with self.assertRaises(TypeError):
            resolve(42)

    def test_actual_matplotlib_agg_draw(self):
        if self.registry_surface_fixture:
            self.skipTest('Actual Agg image is checked under the unmodified installed registry')
        fig = plot2d.plot_metamaterial(self.network, colormap=self.custom)
        fig.set_size_inches(5, 4)
        fig.set_dpi(80)
        fig.axes[0].title.set_fontsize(8)
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        self.assertEqual(rgba.shape, (320, 400, 4))
        self.assertGreater(np.unique(rgba.reshape(-1, 4), axis=0).shape[0], 20)
        output = BytesIO()
        fig.savefig(output, format='png')
        self.assertTrue(output.getvalue().startswith(b'\x89PNG\r\n\x1a\n'))


class Registry35Surface:
    """Expose documented 3.5 dict lookup, deliberately without get_cmap.

    Palette data still comes from installed Matplotlib. This is NOT an actual
    Matplotlib 3.5 runtime or a substitute for testing that supported version.
    """
    def __init__(self, registry):
        self.registry = registry

    def __getitem__(self, name):
        return self.registry[name]


class TestRegistry35SurfaceCompatibility(TestColormapCompatibility):
    """Repeat the color contracts against the limited registry API fixture."""
    registry_surface_fixture = True

    def setUp(self):
        super().setUp()
        registry_patch = patch.object(
            matplotlib, 'colormaps', Registry35Surface(matplotlib.colormaps)
        )
        registry_patch.start()
        self.addCleanup(registry_patch.stop)


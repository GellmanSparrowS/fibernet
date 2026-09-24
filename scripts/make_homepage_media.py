"""Render reproducible local homepage media from the shared science core.

Run from repo root: python scripts/make_homepage_media.py
Produces an animated GIF, a vector SVG keyframe, and a provenance manifest.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.collections import LineCollection
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'FiberScope'))

from fslab.engine2 import Engine2, Engine2Config
from fslab.percolation import compute_percolation
from fslab.simcache import ENGINE_SRC_HASH, ENGINE_VERSION
from fslab.structure import SPECTRUM_PRESETS, StructureFactory
from fibernet.gen.manufacturing import PlanarManufacturingConfig, compile_planar


class HomepageMedia:
    def __init__(self, output_dir=None, max_frames=18):
        self.output_dir = Path(output_dir or ROOT / 'docs' / 'media')
        self.max_frames = max_frames
        if not 2 <= max_frames <= 24:
            raise ValueError('max_frames must be between 2 and 24')

    def build(self):
        graph = StructureFactory(unit='kagome', grid_x=3, grid_y=3,
                                 n_pts_per_side=2, perturbation=0.25,
                                 topology='topnet26', seed=23).build()
        if graph.num_nodes > 600 or graph.num_edges > 800:
            raise MemoryError('homepage graph exceeds rendering budget')
        cfg = Engine2Config(target_stretch=1.4, n_increments=20,
                            num_steps=2000, use_contact=False,
                            use_bending=True)
        run = Engine2(graph, cfg).run()
        result = compute_percolation(run, alpha=0.05)
        frame_ids = np.unique(np.linspace(0, len(run.frames_xy) - 1,
                                          min(self.max_frames, len(run.frames_xy)),
                                          dtype=int))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        gif_path = self.output_dir / 'tensile_recruitment_kagome.gif'
        svg_path = self.output_dir / 'tensile_recruitment_kagome_final.svg'
        manifest_path = self.output_dir / 'tensile_recruitment_kagome.json'

        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                             'svg.fonttype': 'none'})
        fig, (network_ax, curve_ax) = plt.subplots(
            1, 2, figsize=(10, 4.8), gridspec_kw={'width_ratios': [1.15, 1]})
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18,
                            top=0.83, wspace=0.24)
        fig.text(0.5, 0.045,
                 'Threshold: 5% of frame max positive edge strain; contact off. '
                 'Colors do not measure total force flow.',
                 ha='center', fontsize=8.5, color='#52606D')
        total_span = np.ptp(run.frames_xy.reshape(-1, 2), axis=0)
        xlim = (float(run.frames_xy[..., 0].min() - 0.06 * total_span[0]),
                float(run.frames_xy[..., 0].max() + 0.06 * total_span[0]))
        ylim = (float(run.frames_xy[..., 1].min() - 0.08 * total_span[1]),
                float(run.frames_xy[..., 1].max() + 0.08 * total_span[1]))
        fractions = result.active_edges.mean(axis=1)

        def draw(frame):
            network_ax.clear()
            curve_ax.clear()
            xy = run.frames_xy[frame]
            segments = xy[np.asarray(run.edges, dtype=int)]
            inactive = ~result.active_edges[frame]
            recruited = result.active_edges[frame] & ~result.edge_in_spanning[frame]
            spanning = result.edge_in_spanning[frame]
            for mask, color, width, label in (
                    (inactive, '#AAB5C2', 1.2, 'Below threshold'),
                    (recruited, '#21698A', 2.3, 'Recruited'),
                    (spanning, '#D78732', 2.7, 'Spanning component')):
                if mask.any():
                    network_ax.add_collection(LineCollection(
                        segments[mask], colors=color, linewidths=width,
                        label=label))
            network_ax.scatter(xy[:, 0], xy[:, 1], s=6, color='#233646', zorder=3)
            network_ax.set(xlim=xlim, ylim=ylim, aspect='equal',
                           title='Positive-strain recruitment')
            network_ax.set_xticks([])
            network_ax.set_yticks([])
            network_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.04),
                              ncol=3, fontsize=8, frameon=False,
                              handlelength=1.3, columnspacing=1.0)
            network_ax.spines[:].set_visible(False)

            curve_ax.plot(run.strain_levels, fractions, color='#21698A', lw=2)
            curve_ax.scatter([run.strain_levels[frame]], [fractions[frame]],
                             s=70, color='#D78732', edgecolors='white', zorder=4)
            curve_ax.axvline(run.strain_levels[frame], color='#D78732',
                             lw=1, ls='--')
            curve_ax.set(xlim=(float(run.strain_levels.min()),
                                float(run.strain_levels.max())), ylim=(-0.04, 1.04),
                         xlabel='Engineering strain',
                         ylabel='Fraction of recruited edges',
                         title='Network recruitment over stretch')
            curve_ax.grid(axis='y', color='#E3E8ED', linewidth=0.8)
            for spine in ('top', 'right'):
                curve_ax.spines[spine].set_visible(False)
            fig.suptitle(f'Perturbed 3×3 kagome · frame {frame + 1}/{len(run.frames_xy)}',
                         fontsize=15, x=0.5, y=0.96)

        try:
            temp_gif = gif_path.with_suffix('.tmp.gif')
            writer = PillowWriter(fps=4)
            with writer.saving(fig, str(temp_gif), dpi=115):
                for frame in frame_ids:
                    draw(int(frame))
                    writer.grab_frame()
            os.replace(temp_gif, gif_path)
            draw(int(frame_ids[-1]))
            temp_svg = svg_path.with_suffix('.tmp.svg')
            fig.savefig(temp_svg, format='svg')
            os.replace(temp_svg, svg_path)
        finally:
            plt.close(fig)

        manifest = {
            'source': 'Engine2 raw trajectory and shared tensile recruitment API',
            'engine_version': ENGINE_VERSION,
            'engine_hash': ENGINE_SRC_HASH,
            'analysis_sha256': hashlib.sha256((ROOT / 'fibernet' / 'analysis' /
                                               'tensile_recruitment.py').read_bytes()).hexdigest(),
            'render_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'unit': 'kagome', 'grid': [3, 3], 'points_per_edge': 2,
            'perturbation': 0.25, 'topology': 'topnet26', 'seed': 23,
            'nodes': int(graph.num_nodes), 'edges': int(graph.num_edges),
            'target_stretch': 1.4, 'increments': 20, 'steps': 2000,
            'contact': False, 'bending': True, 'alpha': 0.05,
            'sampled_frame_indices': [int(f) for f in frame_ids],
            'total_frames': int(len(run.frames_xy)),
            'gif_sha256': hashlib.sha256(gif_path.read_bytes()).hexdigest(),
            'svg_sha256': hashlib.sha256(svg_path.read_bytes()).hexdigest(),
        }
        temporary = manifest_path.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(manifest, indent=2) + '\n',
                             encoding='utf-8')
        os.replace(temporary, manifest_path)
        self.build_threshold_gallery(run, graph, frame_ids)
        print('[homepage_media] wrote GIF, SVG and manifest')
        return manifest

    def build_threshold_gallery(self, run, graph, frame_ids):
        """Compare three analysis thresholds on the identical saved trajectory."""
        alphas = (0.02, 0.05, 0.15)
        analyses = [compute_percolation(run, alpha=a) for a in alphas]
        gif_path = self.output_dir / 'tensile_threshold_sensitivity.gif'
        svg_path = self.output_dir / 'tensile_threshold_sensitivity_final.svg'
        manifest_path = self.output_dir / 'tensile_threshold_sensitivity.json'
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                             'svg.fonttype': 'none'})
        fig, axes = plt.subplots(1, 3, figsize=(12, 5.0))
        fig.subplots_adjust(left=0.035, right=0.985, bottom=0.13,
                            top=0.82, wspace=0.12)
        span = np.ptp(run.frames_xy.reshape(-1, 2), axis=0)
        limits = [
            (float(run.frames_xy[..., axis].min() - 0.05 * span[axis]),
             float(run.frames_xy[..., axis].max() + 0.05 * span[axis]))
            for axis in range(2)
        ]
        fig.text(0.5, 0.04,
                 'Same perturbed kagome trajectory · positive axial strain only · '
                 'contact off · colors do not measure total force flow',
                 ha='center', fontsize=9, color='#52606D')

        def draw(frame):
            xy = run.frames_xy[frame]
            segments = xy[np.asarray(run.edges, dtype=int)]
            for ax, alpha, analysis in zip(axes, alphas, analyses):
                ax.clear()
                inactive = ~analysis.active_edges[frame]
                active = analysis.active_edges[frame] & ~analysis.edge_in_spanning[frame]
                spanning = analysis.edge_in_spanning[frame]
                for mask, color, width in (
                        (inactive, '#B8C2CB', 0.75),
                        (active, '#21698A', 1.5),
                        (spanning, '#D78732', 2.0)):
                    if mask.any():
                        ax.add_collection(LineCollection(
                            segments[mask], colors=color, linewidths=width))
                ax.set(xlim=limits[0], ylim=limits[1], aspect='equal',
                       title='Threshold α = %.2f\nRecruited %.0f%% · spanning %.0f%%' %
                       (alpha, 100 * analysis.active_edges[frame].mean(),
                        100 * analysis.backbone_frac[frame]))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[:].set_visible(False)
            fig.suptitle('Sensitivity of thresholded recruitment · strain %.3f' %
                         run.strain_levels[frame], fontsize=15, y=0.97)

        try:
            temporary_gif = gif_path.with_suffix('.tmp.gif')
            with PillowWriter(fps=4).saving(fig, str(temporary_gif), dpi=105) as writer:
                for frame in frame_ids:
                    draw(int(frame))
                    writer.grab_frame()
            os.replace(temporary_gif, gif_path)
            draw(int(frame_ids[-1]))
            temporary_svg = svg_path.with_suffix('.tmp.svg')
            fig.savefig(temporary_svg, format='svg')
            os.replace(temporary_svg, svg_path)
        finally:
            plt.close(fig)
        manifest = {
            'source': 'same Engine2 trajectory as tensile_recruitment_kagome',
            'engine_version': ENGINE_VERSION, 'engine_hash': ENGINE_SRC_HASH,
            'analysis_sha256': hashlib.sha256((ROOT / 'fibernet' / 'analysis' /
                                               'tensile_recruitment.py').read_bytes()).hexdigest(),
            'render_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'alphas': list(alphas), 'sampled_frame_indices': [int(f) for f in frame_ids],
            'nodes': graph.num_nodes, 'edges': graph.num_edges,
            'gif_sha256': hashlib.sha256(gif_path.read_bytes()).hexdigest(),
            'svg_sha256': hashlib.sha256(svg_path.read_bytes()).hexdigest(),
        }
        temporary_json = manifest_path.with_suffix('.tmp.json')
        temporary_json.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary_json, manifest_path)

    def build_generation_gallery(self):
        """Show one normalized profile across four APP manufacturing units."""
        units = ('square', 'hexagon', 'ring', 'kagome')
        if self.max_frames < 3:
            raise ValueError('generation gallery needs at least three frames')
        count = self.max_frames if self.max_frames % 2 else self.max_frames - 1
        amplitudes = 4.0 * np.sin(np.linspace(0, np.pi, count)) ** 2
        reference = np.asarray(SPECTRUM_PRESETS['swirl'], dtype=float)
        frames = []
        for amplitude in amplitudes:
            graphs = []
            for unit in units:
                graph = StructureFactory(
                    unit=unit, grid_x=2, grid_y=2, n_pts_per_side=3,
                    line_displacements=(reference * amplitude).tolist(),
                    topology='topnet26', seed=23).build()
                if graph.num_nodes > 1000 or graph.num_edges > 1200:
                    raise MemoryError('gallery graph exceeds node/edge budget')
                graphs.append(graph)
            frames.append(graphs)
        for unit_index in range(len(units)):
            first = frames[0][unit_index]
            for graphs in frames[1:]:
                current = graphs[unit_index]
                if (current.num_nodes != first.num_nodes
                        or not np.array_equal(current.edge_array(), first.edge_array())):
                    raise AssertionError('generation changed topology across profile frames')

        self.output_dir.mkdir(parents=True, exist_ok=True)
        gif_path = self.output_dir / 'spectrum_four_topologies.gif'
        svg_path = self.output_dir / 'spectrum_four_topologies_peak.svg'
        manifest_path = self.output_dir / 'spectrum_four_topologies.json'
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                             'svg.fonttype': 'none'})
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(left=0.04, right=0.96, bottom=0.11,
                            top=0.88, wspace=0.11, hspace=0.22)
        bounds = []
        for index in range(len(units)):
            positions = np.concatenate([graphs[index].node_positions()[:, :2]
                                        for graphs in frames])
            low = positions.min(axis=0)
            high = positions.max(axis=0)
            pad = max(float(np.ptp(positions, axis=0).max()) * 0.04, 0.2)
            bounds.append(((float(low[0] - pad), float(high[0] + pad)),
                           (float(low[1] - pad), float(high[1] + pad))))
        fig.text(0.5, 0.045,
                 'Geometry only · same dimensionless reference profile across units · '
                 'printability and mechanics not assessed',
                 ha='center', fontsize=9, color='#52606D')

        def draw(frame_index):
            for index, ax in enumerate(axs.flat):
                ax.clear()
                graph = frames[frame_index][index]
                xy = graph.node_positions()[:, :2]
                segments = xy[np.asarray(graph.edge_array(), dtype=int)]
                ax.add_collection(LineCollection(
                    segments, colors='#226783', linewidths=1.5))
                ax.scatter(xy[:, 0], xy[:, 1], s=2.5, color='#D18A36', zorder=3)
                ax.set(xlim=bounds[index][0], ylim=bounds[index][1],
                       aspect='equal', title=units[index].capitalize())
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[:].set_visible(False)
            fig.suptitle('One profile, four fiber-network topologies  ·  '
                         'amplitude %.2f×' % amplitudes[frame_index],
                         fontsize=15, y=0.96)

        try:
            temp_gif = gif_path.with_suffix('.tmp.gif')
            writer = PillowWriter(fps=4)
            with writer.saving(fig, str(temp_gif), dpi=110):
                for frame_index in range(len(frames)):
                    draw(frame_index)
                    writer.grab_frame()
            os.replace(temp_gif, gif_path)
            peak = int(np.argmax(amplitudes))
            draw(peak)
            temp_svg = svg_path.with_suffix('.tmp.svg')
            fig.savefig(temp_svg, format='svg')
            os.replace(temp_svg, svg_path)
        finally:
            plt.close(fig)

        manifest = {
            'source': 'FiberScope manufacturing topology and shared FiberNet spectrum',
            'engine_version': ENGINE_VERSION,
            'engine_hash': ENGINE_SRC_HASH,
            'units': list(units), 'grid': [2, 2], 'points_per_edge': 3,
            'topology': 'topnet26', 'seed': 23,
            'profile': 'swirl', 'base_profile': reference.tolist(),
            'amplitudes': [float(value) for value in amplitudes],
            'node_edge_counts': [[frames[0][i].num_nodes,
                                  frames[0][i].num_edges] for i in range(len(units))],
            'render_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'spectrum_sha256': hashlib.sha256((ROOT / 'fibernet' / 'gen' /
                                               'spectrum.py').read_bytes()).hexdigest(),
            'gif_sha256': hashlib.sha256(gif_path.read_bytes()).hexdigest(),
            'svg_sha256': hashlib.sha256(svg_path.read_bytes()).hexdigest(),
        }
        temporary = manifest_path.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(manifest, indent=2) + '\n',
                             encoding='utf-8')
        os.replace(temporary, manifest_path)
        print('[homepage_media] wrote generation GIF, SVG and manifest')
        return manifest

    def build_route_playback(self):
        """Animate edge-ordered Euler traversal from the independent library."""
        config = PlanarManufacturingConfig(unit='kagome', grid_x=2, grid_y=2,
                                           n_pts_per_side=2, seed=23)
        network = compile_planar(config)
        frame_counts = np.unique(np.linspace(0, len(network.edges),
                                             self.max_frames, dtype=int))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        gif_path = self.output_dir / 'continuous_route_kagome.gif'
        svg_path = self.output_dir / 'continuous_route_kagome_final.svg'
        manifest_path = self.output_dir / 'continuous_route_kagome.json'
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                             'svg.fonttype': 'none'})
        fig, ax = plt.subplots(figsize=(9, 5.4))
        fig.subplots_adjust(left=0.06, right=0.94, bottom=0.14, top=0.83)
        xy = network.positions[:, :2]
        segments = xy[network.edges]
        span = np.ptp(xy, axis=0)
        xlim = (float(xy[:, 0].min() - .05 * span[0]),
                float(xy[:, 0].max() + .05 * span[0]))
        ylim = (float(xy[:, 1].min() - .08 * span[1]),
                float(xy[:, 1].max() + .08 * span[1]))
        fig.text(0.5, .045,
                 'Topology-level closed traversal · each fiber edge visited once · '
                 'physical printing constraints not certified',
                 ha='center', fontsize=9, color='#52606D')

        def draw(count):
            ax.clear()
            visited = np.zeros(len(network.edges), dtype=bool)
            visited[network.route_edges[:count]] = True
            ax.add_collection(LineCollection(segments[~visited],
                                             colors='#CBD3DA', linewidths=.9))
            if count:
                ax.add_collection(LineCollection(segments[visited],
                                                 colors='#21698A', linewidths=1.6))
            head = int(network.route_nodes[count])
            ax.scatter([xy[head, 0]], [xy[head, 1]], s=85,
                       color='#D78732', edgecolors='white', zorder=5)
            ax.set(xlim=xlim, ylim=ylim, aspect='equal',
                   title='Continuous route · %d / %d fibers visited' %
                   (count, len(network.edges)))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            fig.suptitle('2×2 kagome · independent parallel fibers retained',
                         fontsize=15, y=.96)

        try:
            temporary_gif = gif_path.with_suffix('.tmp.gif')
            with PillowWriter(fps=4).saving(fig, str(temporary_gif), dpi=120) as writer:
                for count in frame_counts:
                    draw(int(count))
                    writer.grab_frame()
            os.replace(temporary_gif, gif_path)
            draw(int(frame_counts[-1]))
            temporary_svg = svg_path.with_suffix('.tmp.svg')
            fig.savefig(temporary_svg, format='svg')
            os.replace(temporary_svg, svg_path)
        finally:
            plt.close(fig)
        manifest = {
            'source': 'fibernet.gen.manufacturing.compile_planar',
            'engine_version': ENGINE_VERSION, 'engine_hash': ENGINE_SRC_HASH,
            'unit': config.unit, 'grid': [config.grid_x, config.grid_y],
            'points_per_edge': config.n_pts_per_side, 'seed': config.seed,
            'nodes': len(network.positions), 'edges': len(network.edges),
            'topology_id': network.topology_id,
            'sampled_edge_counts': [int(n) for n in frame_counts],
            'manufacturing_sha256': hashlib.sha256((ROOT / 'fibernet' / 'gen' /
                                                    'manufacturing.py').read_bytes()).hexdigest(),
            'render_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'gif_sha256': hashlib.sha256(gif_path.read_bytes()).hexdigest(),
            'svg_sha256': hashlib.sha256(svg_path.read_bytes()).hexdigest(),
        }
        temporary_json = manifest_path.with_suffix('.tmp.json')
        temporary_json.write_text(json.dumps(manifest, indent=2) + '\n',
                                  encoding='utf-8')
        os.replace(temporary_json, manifest_path)
        print('[homepage_media] wrote route GIF, SVG and manifest')
        return manifest


if __name__ == '__main__':
    media = HomepageMedia()
    media.build()
    media.build_generation_gallery()
    media.build_route_playback()

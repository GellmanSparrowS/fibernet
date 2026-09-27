"""Make an editable, source-backed x/y recruitment comparison for the local homepage.

Run: python scripts/make_directional_recruitment_media.py
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.collections import LineCollection
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FiberScope"))

from fslab.engine2 import Engine2, Engine2Config
from fslab.simcache import ENGINE_SRC_HASH, ENGINE_VERSION
from fslab.structure import StructureFactory
from fibernet.analysis.tensile_recruitment import compute_percolation


class DirectionalRecruitmentMedia:
    def __init__(self, output_dir=None, max_frames=18):
        self.output_dir = Path(output_dir or ROOT / "docs" / "media")
        self.max_frames = int(max_frames)
        if not 2 <= self.max_frames <= 21:
            raise ValueError("max_frames must be 2..21")

    @staticmethod
    def _case(direction):
        graph = StructureFactory(
            unit="hexagon", grid_x=3, grid_y=3, n_pts_per_side=1,
            perturbation=0.04, seed=11).build()
        if direction == "y":
            positions = np.asarray(graph.node_positions(), dtype=float)
            rotated = np.column_stack((positions[:, 1], -positions[:, 0]))
            graph.set_node_positions({i: point
                                      for i, point in enumerate(rotated)})
        run = Engine2(graph, Engine2Config(
            target_stretch=1.4, n_increments=20, num_steps=2000,
            use_contact=False, use_bending=True)).run()
        analysis = compute_percolation(run, alpha=0.05, min_active=0.05)
        xy = np.asarray(run.frames_xy)
        if direction == "y":
            xy = np.stack((-xy[..., 1], xy[..., 0]), axis=-1)
        return run, analysis, xy

    def run(self):
        cases = {direction: self._case(direction)
                 for direction in ("x", "y")}
        if cases["x"][1].perc_frame != 4 or cases["y"][1].perc_frame != 12:
            raise AssertionError("directional onset changed; review study checkpoint")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = "directional_recruitment_hexagon"
        gif = self.output_dir / (stem + ".gif")
        svg = self.output_dir / (stem + "_final.svg")
        manifest_path = self.output_dir / (stem + ".json")
        frames = np.unique(np.linspace(0, 20, self.max_frames, dtype=int))
        all_xy = np.concatenate([item[2].reshape(-1, 2)
                                 for item in cases.values()])
        span = np.ptp(all_xy, axis=0)
        bounds = [(float(all_xy[:, i].min() - 0.06 * span[i]),
                   float(all_xy[:, i].max() + 0.06 * span[i]))
                  for i in range(2)]
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "svg.fonttype": "none"})
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0),
                                 gridspec_kw={"height_ratios": [3, 1]})
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13,
                            top=0.88, hspace=0.29, wspace=0.18)
        fig.text(0.5, 0.91,
                 "Gray: below threshold   ·   Blue: recruited   ·   "
                 "Orange: spanning component",
                 ha="center", fontsize=9, color="#52606D")
        fig.text(0.5, 0.035,
                 "Engine2; positive axial edge strain >= 0.05; contact off. "
                 "Colors do not measure total force flow.",
                 ha="center", fontsize=9, color="#52606D")

        def draw(frame):
            for col, direction in enumerate(("x", "y")):
                run, recruited, coordinates = cases[direction]
                ax, curve = axes[0, col], axes[1, col]
                ax.clear()
                curve.clear()
                points = coordinates[frame]
                segments = points[np.asarray(run.edges, dtype=int)]
                active = recruited.active_edges[frame]
                spanning = recruited.edge_in_spanning[frame]
                for mask, color, width in (
                        (~active, "#B4BEC8", 1.0),
                        (active & ~spanning, "#277B9A", 2.0),
                        (spanning, "#DC8935", 2.7)):
                    if mask.any():
                        ax.add_collection(LineCollection(
                            segments[mask], colors=color, linewidths=width))
                ax.set(xlim=bounds[0], ylim=bounds[1], aspect="equal",
                       title=f"{direction.upper()} loading · first span "
                             f"at stretch {run.strain_levels[recruited.perc_frame]:.2f}")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[:].set_visible(False)
                curve.plot(run.strain_levels, recruited.backbone_frac,
                           color="#277B9A", lw=2)
                curve.scatter([run.strain_levels[frame]],
                              [recruited.backbone_frac[frame]], s=55,
                              color="#DC8935", zorder=3)
                curve.axvline(run.strain_levels[recruited.perc_frame],
                              color="#7F8E9C", lw=1, ls="--")
                curve.set(xlim=(1.0, 1.4), ylim=(-0.02, 1.02),
                          xlabel="Stretch ratio",
                          ylabel="Spanning edge fraction")
                curve.grid(axis="y", color="#E4E9EF", linewidth=.8)
                curve.spines[["top", "right"]].set_visible(False)
            fig.suptitle(
                f"Direction-dependent recruitment · 3×3 hexagon · "
                f"stretch {cases['x'][0].strain_levels[frame]:.2f}",
                fontsize=15, y=.97)

        try:
            temporary = gif.with_suffix(".tmp.gif")
            writer = PillowWriter(fps=4)
            with writer.saving(fig, str(temporary), dpi=120):
                for frame in frames:
                    draw(int(frame))
                    writer.grab_frame()
            os.replace(temporary, gif)
            draw(int(frames[-1]))
            temporary = svg.with_suffix(".tmp.svg")
            fig.savefig(temporary, format="svg")
            os.replace(temporary, svg)
        finally:
            plt.close(fig)
        manifest = {
            "source": "Engine2 trajectories and shared tensile recruitment API",
            "engine_version": ENGINE_VERSION, "engine_hash": ENGINE_SRC_HASH,
            "analysis_sha256": hashlib.sha256(
                (ROOT / "fibernet" / "analysis" /
                 "tensile_recruitment.py").read_bytes()).hexdigest(),
            "render_script": "scripts/make_directional_recruitment_media.py",
            "render_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "unit": "hexagon", "grid": [3, 3], "seed": 11,
            "directions": ["x", "y"], "alpha": 0.05,
            "min_active": 0.05, "contact": False,
            "onset_frames": {d: int(cases[d][1].perc_frame)
                             for d in ("x", "y")},
            "sampled_frame_indices": [int(i) for i in frames],
            "gif_sha256": hashlib.sha256(gif.read_bytes()).hexdigest(),
            "svg_sha256": hashlib.sha256(svg.read_bytes()).hexdigest(),
        }
        temporary = manifest_path.with_suffix(".tmp.json")
        try:
            temporary.write_text(json.dumps(manifest, indent=2) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
        print("[directional_media] GIF, SVG and manifest saved")
        return manifest


if __name__ == "__main__":
    DirectionalRecruitmentMedia().run()

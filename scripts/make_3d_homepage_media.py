"""Render a bounded, source-backed 3D topology gallery for the homepage.

Run: python scripts/make_3d_homepage_media.py
The GIF shows camera rotation only; graph geometry is fixed in every frame.
"""

import hashlib
import io
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fibernet.gen.pattern import pattern_3d


class ThreeDHomepageGallery:
    def __init__(self, root=None, frames=15):
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self.output = self.root / "docs" / "media"
        self.frames = int(frames)
        if not 8 <= self.frames <= 24:
            raise ValueError("frames must be between 8 and 24")
        self.units = ("octet", "diamond_3d", "gyroid")
        self.colors = ("#22c7c9", "#f4a261", "#a78bfa")

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def run(self):
        networks = []
        for unit in self.units:
            graph = pattern_3d(unit=unit, box=(2, 2, 2), grid=(2, 2, 2))
            points = np.asarray(graph.node_positions(), dtype=float)
            edges = np.asarray([(edge.node_i, edge.node_j)
                                for edge in graph.edges.values()], dtype=int)
            if points.shape[1] != 3 or not np.isfinite(points).all():
                raise AssertionError("invalid 3D graph")
            networks.append((points, edges))

        fig = plt.figure(figsize=(12.6, 4.2), dpi=100, facecolor="#081422")
        axes = []
        for index, ((points, edges), unit, color) in enumerate(
                zip(networks, self.units, self.colors)):
            ax = fig.add_subplot(1, 3, index + 1, projection="3d")
            ax.set_facecolor("#081422")
            ax.add_collection3d(Line3DCollection(points[edges], colors=color,
                                                  linewidths=1.0, alpha=0.76))
            center = points.mean(axis=0)
            span = max(np.ptp(points, axis=0)) * 0.62
            for setter, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
                setter(value - span, value + span)
            ax.set_box_aspect((1, 1, 1), zoom=0.85)
            ax.set_axis_off()
            ax.set_title(unit.replace("_3d", "").upper(), color="#f3f7fb",
                         fontsize=15, pad=8)
            ax.text2D(0.5, 0.03, "%d nodes · %d edges" %
                      (len(points), len(edges)), transform=ax.transAxes,
                      ha="center", color="#a9bbca", fontsize=9)
            axes.append(ax)
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.83,
                            wspace=0.01)
        self.output.mkdir(parents=True, exist_ok=True)
        gif_path = self.output / "three_dimensional_topologies.gif"
        svg_path = self.output / "three_dimensional_topologies_final.svg"
        images = []
        for frame in range(self.frames):
            angle = 25 + 360 * frame / self.frames
            for ax in axes:
                ax.view_init(elev=22, azim=angle)
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=100, facecolor=fig.get_facecolor())
            buffer.seek(0)
            images.append(Image.open(buffer).convert("RGB"))
        fig.savefig(svg_path, format="svg", facecolor=fig.get_facecolor(),
                    metadata={"Date": None})
        plt.close(fig)
        svg_path.write_text(
            "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
            encoding="utf-8",
        )
        temporary = gif_path.with_suffix(".gif.tmp")
        try:
            images[0].save(temporary, format="GIF", save_all=True,
                           append_images=images[1:], duration=130,
                           loop=0, optimize=True)
            os.replace(temporary, gif_path)
        finally:
            temporary.unlink(missing_ok=True)
        manifest = {
            "description": "Fixed 2x2x2 graphs; synchronized camera rotation only",
            "units": {unit: {"nodes": len(points), "edges": len(edges)}
                      for unit, (points, edges) in zip(self.units, networks)},
            "frames": self.frames,
            "source_sha256": {
                str(path.relative_to(self.root)).replace("\\", "/"): self.digest(path)
                for path in (self.root / "fibernet" / "gen" / "pattern.py",
                             self.root / "fibernet" / "core" / "structure_graph.py",
                             Path(__file__))},
            "gif_sha256": self.digest(gif_path),
            "svg_sha256": self.digest(svg_path),
        }
        target = self.output / "three_dimensional_topologies.json"
        temporary = target.with_suffix(".json.tmp")
        try:
            temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        print("[3d_media] %d frames; %s" % (self.frames, gif_path))


if __name__ == "__main__":
    ThreeDHomepageGallery().run()

"""Render a source-backed route-preserving deletion comparison for the local homepage.

Run: python scripts/make_intervention_media.py --case kagome:seed23:x
"""
import argparse
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "FiberScope"))

from benchmarks.constrained_cycle_intervention import (
    ConstrainedCycleIntervention, ConstrainedCycleConfig)
from benchmarks.percolation_cycle_intervention import CycleInterventionStudy
from fibernet.analysis import compute_percolation


MEDIA = ROOT / "docs" / "media"
STUDY = ROOT / "benchmarks" / "results" / "constrained_cycle_intervention.json"


class InterventionMedia:
    def __init__(self, output_dir=MEDIA, max_frames=15, case_key="ring:seed11:x"):
        self.output_dir = Path(output_dir)
        self.max_frames = int(max_frames)
        self.case_key = case_key
        if not 2 <= self.max_frames <= 21:
            raise ValueError("max_frames must be 2..21")

    @staticmethod
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def run(self):
        study_data = json.loads(STUDY.read_text(encoding="utf-8"))
        study = ConstrainedCycleIntervention(ConstrainedCycleConfig())
        if (study_data["analysis_hash"] != study.analysis_hash or
                study_data["config"] != json.loads(json.dumps(
                    study.config.__dict__))):
            raise ValueError("intervention results do not match current study")
        key = self.case_key
        if key not in study_data["cases"]:
            raise ValueError("case is absent from the verified study")
        unit, seed_text, direction = key.split(":")
        seed = int(seed_text.removeprefix("seed"))
        case = study_data["cases"][key]
        graph = study._graph(unit, seed, direction)
        source = [("Original", graph, case["baseline"])]
        for label, name in (("Low early strain", "low_early"),
                            ("Fixed random", "random"),
                            ("High late strain", "high_late")):
            candidate, _ = CycleInterventionStudy.compact_graph(
                graph, case["interventions"][name]["removed_ids"])
            source.append((label, candidate, case["interventions"][name]))
        panels = []
        for label, candidate, saved in source:
            if candidate.num_nodes > 1200 or candidate.num_edges > 1600:
                raise MemoryError("intervention media graph exceeds budget")
            run = study._simulate(candidate)
            if not np.isclose(float(run.force_curve[-1]),
                              saved["final_raw_reaction"], rtol=1e-8, atol=1e-6):
                raise AssertionError("rendered trajectory differs from saved study")
            recruitment = compute_percolation(
                run, alpha=study.config.alpha, min_active=study.config.min_active)
            panels.append((label, run, recruitment, saved))
        all_xy = np.concatenate([run.frames_xy.reshape(-1, 2)
                                 for _, run, _, _ in panels])
        lo, hi = all_xy.min(axis=0), all_xy.max(axis=0)
        span = np.maximum(hi - lo, 1e-9)
        bounds = [(float(lo[i] - 0.06 * span[i]),
                   float(hi[i] + 0.06 * span[i])) for i in range(2)]
        indices = np.unique(np.linspace(0, 20, self.max_frames, dtype=int))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = "route_preserving_intervention_" + unit
        gif = self.output_dir / (stem + ".gif")
        svg = self.output_dir / (stem + "_final.svg")
        manifest_path = self.output_dir / (stem + ".json")
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "svg.fonttype": "none"})
        fig, axes = plt.subplots(2, 4, figsize=(15.2, 6.4),
                                 gridspec_kw={"height_ratios": [3, 1]})
        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.16,
                            top=0.85, hspace=0.25, wspace=0.16)
        original_force = panels[0][1].force_curve[-1]
        maximum_force = max(float(np.max(run.force_curve))
                            for _, run, _, _ in panels)
        fig.text(0.5, 0.92,
                 "Gray: below strain threshold   ·   Blue: recruited   ·   "
                 "Orange: grip-spanning component", ha="center", fontsize=10,
                 color="#52606D")
        fig.text(0.5, 0.04,
                 f"Same removed length ({100 * case['selected_length_class']:.2f}%); "
                 "Euler route retained. "
                 "Reduced-model results; colors do not measure total force flow.",
                 ha="center", fontsize=10, color="#52606D")

        def draw(frame):
            for col, (label, run, recruited, saved) in enumerate(panels):
                ax, curve = axes[0, col], axes[1, col]
                ax.clear()
                curve.clear()
                xy = run.frames_xy[frame]
                segments = xy[np.asarray(run.edges, dtype=int)]
                active = recruited.active_edges[frame]
                spanning = recruited.edge_in_spanning[frame]
                for mask, color, width in ((~active, "#B4BEC8", .9),
                                           (active & ~spanning, "#277B9A", 1.7),
                                           (spanning, "#DC8935", 2.2)):
                    if mask.any():
                        ax.add_collection(LineCollection(
                            segments[mask], colors=color, linewidths=width))
                ratio = saved["final_raw_reaction"] / original_force
                ax.set(xlim=bounds[0], ylim=bounds[1], aspect="equal",
                       title=f"{label}\nfinal reaction / original {ratio:.3f}")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[:].set_visible(False)
                curve.plot(run.strain_levels, run.force_curve / original_force,
                           color="#277B9A", lw=2)
                curve.scatter([run.strain_levels[frame]],
                              [run.force_curve[frame] / original_force], s=45,
                              color="#DC8935", zorder=3)
                curve.set(xlim=(1.0, 1.4),
                          ylim=(-0.02, 1.08 * maximum_force / original_force),
                          xlabel="Stretch ratio",
                          ylabel="Raw reaction / original final" if col == 0 else "")
                curve.grid(axis="y", color="#E4E9EF", linewidth=.8)
                curve.spines[["top", "right"]].set_visible(False)
            fig.suptitle(f"Route-preserving cycle deletion · 3×3 perturbed {unit} · "
                         f"stretch {panels[0][1].strain_levels[frame]:.2f}",
                         fontsize=15, y=.99)

        try:
            temporary = gif.with_suffix(".tmp.gif")
            with PillowWriter(fps=4).saving(fig, str(temporary), dpi=120) as writer:
                for frame in indices:
                    draw(int(frame))
                    writer.grab_frame()
            os.replace(temporary, gif)
            draw(int(indices[-1]))
            temporary = svg.with_suffix(".tmp.svg")
            fig.savefig(temporary, format="svg")
            os.replace(temporary, svg)
        finally:
            plt.close(fig)
        manifest = {
            "case": key, "frame_indices": [int(x) for x in indices],
            "removed_length_fraction": float(case["selected_length_class"]),
            "final_reaction_ratios": {label: saved["final_raw_reaction"] /
                                      original_force for label, _, _, saved in panels},
            "source_results": str(STUDY.relative_to(ROOT)).replace("\\", "/"),
            "source_results_sha256": self.digest(STUDY),
            "render_script": "scripts/make_intervention_media.py",
            "render_script_sha256": self.digest(__file__),
            "gif_sha256": self.digest(gif), "svg_sha256": self.digest(svg),
        }
        temporary = manifest_path.with_suffix(".tmp.json")
        try:
            temporary.write_text(json.dumps(manifest, indent=2) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
        print("[intervention_media] GIF, SVG and manifest saved")
        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="ring:seed11:x")
    args = parser.parse_args()
    InterventionMedia(case_key=args.case).run()

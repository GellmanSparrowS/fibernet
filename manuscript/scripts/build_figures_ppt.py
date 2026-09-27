"""Build editable working figures for the FiberNet software Methods draft.

Run: python manuscript/scripts/build_figures_ppt.py
Only measured, source-backed panels are drawn. Fig. 5 awaits stronger evidence.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fibernet.gen import (PlanarManufacturingConfig, manufacturable_graph,
                          compile_surface, load_obj)
from fibernet.sim import ReducedBeamConfig, ReducedBeamSolver
from fibernet.analysis import compute_percolation


OUT = ROOT / "manuscript" / "FiberNet_Methods_Figures_Working.pptx"
NAVY = RGBColor(28, 43, 64)
BLUE = RGBColor(45, 115, 178)
ORANGE = RGBColor(214, 115, 44)
PALE = RGBColor(194, 204, 214)
MID = RGBColor(106, 122, 138)
WHITE = RGBColor(255, 255, 255)


class FigureDeckBuilder:
    def __init__(self, output=OUT):
        self.output = Path(output)
        self.prs = Presentation()
        self.prs.slide_width = Inches(13.333)
        self.prs.slide_height = Inches(7.5)
        self.provenance = {"status": "working scientific figures, not publication final",
                           "source_files": [], "slides": []}

    @staticmethod
    def text(slide, value, x, y, w, h, size=16, color=NAVY,
             bold=False, align=PP_ALIGN.LEFT):
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        frame = box.text_frame
        frame.clear()
        frame.word_wrap = True
        frame.margin_left = frame.margin_right = Inches(0.02)
        frame.margin_top = frame.margin_bottom = Inches(0.01)
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = frame.paragraphs[0]
        p.alignment = align
        run = p.add_run()
        run.text = str(value)
        run.font.name = "Microsoft YaHei"
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        return box

    @staticmethod
    def line(slide, x0, y0, x1, y1, color=PALE, width=0.8):
        shape = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(y0),
            Inches(x1), Inches(y1))
        shape.line.color.rgb = color
        shape.line.width = Pt(width)
        return shape

    def slide(self, title, number, footer):
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        background = slide.background.fill
        background.solid()
        background.fore_color.rgb = WHITE
        self.text(slide, title, 0.52, 0.26, 12.2, 0.56, size=25, bold=True)
        self.line(slide, 0.52, 0.89, 12.83, 0.89, NAVY, 1.1)
        self.text(slide, footer, 0.52, 7.10, 11.9, 0.24, size=8.5, color=MID)
        self.text(slide, f"{number:02d}", 12.38, 7.06, 0.42, 0.28,
                  size=10, color=MID, align=PP_ALIGN.RIGHT)
        return slide

    def network(self, slide, positions, edges, box, active=None,
                spanning=None, nodes=False):
        points = np.asarray(positions, float)
        segments = np.asarray(edges, int)
        if (points.ndim != 2 or points.shape[1] < 2 or
                segments.ndim != 2 or segments.shape[1] != 2 or
                len(points) > 1500 or len(segments) > 1800):
            raise ValueError("figure graph exceeds vector object budget")
        xy = points[:, :2]
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        span = np.maximum(hi - lo, 1e-9)
        x, y, w, h = box
        scale = min((w - 0.18) / span[0], (h - 0.18) / span[1])
        px = x + w / 2 + (xy[:, 0] - (lo[0] + hi[0]) / 2) * scale
        py = y + h / 2 - (xy[:, 1] - (lo[1] + hi[1]) / 2) * scale
        if active is None:
            active = np.ones(len(segments), bool)
        if spanning is None:
            spanning = np.zeros(len(segments), bool)
        for eid, (a, b) in enumerate(segments):
            color = ORANGE if spanning[eid] else (BLUE if active[eid] else PALE)
            width = 1.25 if spanning[eid] else (0.82 if active[eid] else 0.56)
            self.line(slide, px[a], py[a], px[b], py[b], color, width)
        if nodes:
            for xx, yy in zip(px, py):
                dot = slide.shapes.add_shape(MSO_SHAPE.OVAL,
                    Inches(xx - 0.015), Inches(yy - 0.015),
                    Inches(0.03), Inches(0.03))
                dot.fill.solid()
                dot.fill.fore_color.rgb = NAVY
                dot.line.fill.background()

    @staticmethod
    def _planar(unit, grid=2, seed=11, points=2, perturbation=0.2,
                profile=None):
        graph = manufacturable_graph(PlanarManufacturingConfig(
            unit=unit, grid_x=grid, grid_y=grid,
            n_pts_per_side=points, perturbation=perturbation, seed=seed,
            line_displacements=profile))
        return graph

    def figure_one(self):
        slide = self.slide("同一图对象连接 Python 与桌面工作流", 1,
            "来源：共享制造图与 OBJ 曲面示例；图形为可编辑矢量，截图为原始 APP 位图。")
        units = [("方形", "square"), ("六边形", "hexagon"),
                 ("圆环", "ring"), ("笼目", "kagome")]
        x0, widths = 0.56, 2.41
        for idx, (label, unit) in enumerate(units):
            graph = self._planar(unit, profile=[[0.12, -0.06],
                                                [-0.08, 0.14]])
            xx = x0 + idx * 2.58
            self.text(slide, label, xx, 1.1, widths, 0.38, 16, bold=True)
            self.network(slide, graph.node_positions(), graph.edge_array(),
                         (xx, 1.54, widths, 2.48), nodes=False)
            self.text(slide, f"{graph.num_nodes} 节点 · {graph.num_edges} 条独立边",
                      xx, 4.01, widths, 0.32, 10.5, MID)
            self.provenance["slides"].append({"slide": 1, "unit": unit,
                "nodes": graph.num_nodes, "edges": graph.num_edges})
        obj = ROOT / "FiberScope" / "assets" / "obj" / "demo_pyramid.obj"
        vertices, faces, obj_info = load_obj(obj, return_info=True,
                                             target_faces=18)
        curved = compile_surface(vertices, faces, PlanarManufacturingConfig(
            unit="kagome", grid_x=1, grid_y=1, n_pts_per_side=1, seed=23))
        xyz = np.asarray(curved.positions, float)
        projection = np.column_stack((xyz[:, 0] - 0.55 * xyz[:, 1],
                                      0.35 * xyz[:, 0] + 0.35 * xyz[:, 1] -
                                      0.8 * xyz[:, 2]))
        self.text(slide, "曲面路线", 10.88, 1.1, 1.95, 0.38, 16, bold=True)
        self.network(slide, projection, curved.edges, (10.68, 1.54, 2.08, 2.48))
        self.text(slide, f"{len(curved.positions)} 节点 · {len(curved.edges)} 边",
                  10.68, 4.01, 2.12, 0.32, 10.5, MID)
        self.provenance["source_files"].append(str(obj.relative_to(ROOT)))
        self.provenance["slides"].append({"slide": 1, "unit": "curved_kagome",
            "nodes": len(curved.positions), "edges": len(curved.edges),
            "obj_input": obj_info,
            "route_closed": bool(curved.route_nodes[0] == curved.route_nodes[-1])})
        self.line(slide, 0.63, 4.56, 12.70, 4.56, PALE, 0.7)
        self.text(slide, "Python API", 0.78, 4.85, 2.2, 0.41, 17, bold=True)
        self.text(slide, "共同的节点、边与路线数据", 3.35, 4.84, 4.2, 0.43,
                  17, color=BLUE, bold=True)
        self.text(slide, "FiberScope APP", 8.05, 4.85, 3.18, 0.41, 17, bold=True)
        self.line(slide, 2.95, 5.07, 3.25, 5.07, BLUE, 2)
        self.line(slide, 7.53, 5.07, 7.91, 5.07, BLUE, 2)
        self.text(slide, "同参数计算 · 同源结果 · 分别承担编程与交互入口",
                  1.0, 5.56, 10.8, 0.42, 13, MID,
                  align=PP_ALIGN.CENTER)
        screenshot = ROOT / "FiberScope" / "docs" / "screenshots" / "structure-en.png"
        self.provenance["source_files"].append(str(screenshot.relative_to(ROOT)))
        slide.shapes.add_picture(str(screenshot), Inches(9.94), Inches(5.27),
                                 width=Inches(2.70))
        return slide

    def figure_two(self):
        slide = self.slide("计算一致性与本地交付门槛", 2,
            "来源：PROGRESS.md 与完整测试记录；跨机器、Linux/macOS 尚未验收。")
        rows = [
            ("数值迁移", "40 组黄金数组", "最大相对偏差 0"),
            ("库端完整回归", "Python 3.10", "368 通过 / 19 跳过"),
            ("APP 完整回归", "Windows 本机", "25 / 25 套通过"),
            ("可安装库", "独立 wheel 目录", "标注与逆设计 API 实跑"),
            ("冻结 APP", "onedir 干净目录", "依赖 / 冒烟 / 可携带性 / 复制验收"),
        ]
        self.text(slide, "门槛", 0.72, 1.22, 2.54, 0.44, 15, bold=True)
        self.text(slide, "测试范围", 3.57, 1.22, 3.16, 0.44, 15, bold=True)
        self.text(slide, "当前观察", 7.10, 1.22, 5.46, 0.44, 15, bold=True)
        self.line(slide, 0.71, 1.72, 12.60, 1.72, NAVY, 1.0)
        for idx, (gate, scope, observation) in enumerate(rows):
            yy = 1.84 + idx * 0.82
            self.text(slide, gate, 0.72, yy, 2.58, 0.42, 14, bold=True)
            self.text(slide, scope, 3.57, yy, 3.19, 0.42, 13, MID)
            self.text(slide, observation, 7.10, yy, 5.42, 0.48, 13, BLUE)
            self.line(slide, 0.71, yy + 0.62, 12.60, yy + 0.62, PALE, 0.55)
        self.text(slide, "证据范围：软件数值与交付一致性", 0.76, 6.26,
                  6.4, 0.40, 17, bold=True)
        self.text(slide, "材料性能及跨平台泛化仍需独立验证", 7.03, 6.26,
                  5.44, 0.40, 13, MID)
        return slide

    def figure_three(self):
        slide = self.slide("加载方向改变阈值招募的贯通时序", 3,
            "3×3 六边形，种子 11，边上 2 个控制点，扰动 0.2；正向轴应变下限 0.10；颜色不代表完整力流。")
        cfg = ReducedBeamConfig(target_stretch=1.4, n_increments=20,
                                num_steps=2000, use_contact=False,
                                use_bending=True)
        panels = []
        for direction in ("x", "y"):
            graph = self._planar("hexagon", grid=3, seed=11)
            if direction == "y":
                xy = np.asarray(graph.node_positions(), float)
                rotated = np.column_stack((xy[:, 1], -xy[:, 0]))
                graph.set_node_positions({i: p for i, p in enumerate(rotated)})
            if graph.num_nodes > 1500 or graph.num_edges > 1800:
                raise MemoryError("figure simulation exceeds graph budget")
            run = ReducedBeamSolver(graph, cfg).run()
            rec = compute_percolation(run, alpha=0.05, min_active=0.10)
            panels.append((direction, run, rec))
            self.provenance["slides"].append({"slide": 3, "direction": direction,
                "nodes": graph.num_nodes, "edges": graph.num_edges,
                "first_spanning_frame": int(rec.perc_frame),
                "first_spanning_stretch": float(run.strain_levels[rec.perc_frame])
                if rec.perc_frame >= 0 else None})
        for row, (direction, run, rec) in enumerate(panels):
            yy = 1.38 + row * 2.43
            self.text(slide, "沿 x 拉伸" if direction == "x" else "沿 y 拉伸",
                      0.58, yy, 1.72, 0.35, 15, bold=True)
            for col, frame in enumerate((4, 12, 20)):
                xx = 2.10 + col * 3.43
                self.text(slide, f"伸长比 {float(run.strain_levels[frame]):.2f}",
                          xx, yy, 2.60, 0.34, 12.2, MID)
                self.network(slide, run.frames_xy[frame], run.edges,
                             (xx, yy + 0.37, 2.97, 1.83),
                             active=rec.active_edges[frame],
                             spanning=rec.edge_in_spanning[frame])
        self.line(slide, 0.70, 6.38, 12.65, 6.38, PALE, 0.7)
        self.text(slide, "灰：低于阈值", 0.78, 6.50, 2.48, 0.30, 11.5, MID)
        self.text(slide, "蓝：招募", 3.46, 6.50, 1.80, 0.30, 11.5, BLUE)
        self.text(slide, "橙：双端贯通分量", 5.32, 6.50, 3.48, 0.30,
                  11.5, ORANGE)
        first_x = float(panels[0][1].strain_levels[panels[0][2].perc_frame])
        first_y = float(panels[1][1].strain_levels[panels[1][2].perc_frame])
        self.text(slide, f"首次贯通：x {first_x:.2f}；y {first_y:.2f}", 9.02, 6.50,
                  3.62, 0.30, 11.5, NAVY, bold=True)
        return slide

    def figure_four(self):
        source = ROOT / "benchmarks" / "results" / "constrained_cycle_intervention.json"
        record = json.loads(source.read_text(encoding="utf-8"))
        cases = record["cases"]
        if len(cases) != 24 or any(case.get("status") != "complete" for case in cases.values()):
            raise ValueError("Fig. 4 requires all 24 completed intervention cases")
        strategies = (("low_early", "早期低应变", BLUE),
                      ("low_alignment", "静态几何", MID),
                      ("random", "固定随机", ORANGE),
                      ("high_late", "高后期应变", NAVY))
        units = (("square", "方形"), ("hexagon", "六边形"),
                 ("ring", "圆环"), ("kagome", "笼目"))
        slide = self.slide("等长度连续路线干预：早期规则没有稳定优势", 4,
            "24 个加载配置＝4 拓扑×3 基结构×2 方向；x/y 共享基结构。反力为简化模型原始末帧反力比。")
        self.text(slide, "每种拓扑六个加载配置的平均末帧反力 / 原图反力", 0.70,
                  1.10, 10.6, 0.34, 15, bold=True)
        for idx, (unit, label) in enumerate(units):
            subset = [cases[f"{unit}:seed{seed}:{direction}"]
                      for seed in (11, 23, 41) for direction in ("x", "y")]
            if any(case["max_material_class_spread"] > 1e-9 or
                   case["same_length_candidate_count"] < 5 for case in subset):
                raise ValueError("unmatched intervention budget or missing candidates")
            yy = 1.62 + idx * 1.22
            fraction = 100 * np.mean([case["selected_length_class"] for case in subset])
            self.text(slide, f"{label}  删除 {fraction:.2f}%", 0.70, yy, 2.55,
                      0.35, 13.5, bold=True)
            for row, (key, name, color) in enumerate(strategies):
                ybar = yy + 0.39 + row * 0.18
                ratios = [case["interventions"][key]["final_raw_reaction"] /
                          case["baseline"]["final_raw_reaction"] for case in subset]
                mean = float(np.mean(ratios))
                if not np.isfinite(mean) or not 0 <= mean <= 1.1:
                    raise ValueError("invalid Fig. 4 response ratio")
                self.line(slide, 3.55, ybar, 3.55 + 7.10 * mean, ybar,
                          color, 5.2)
                self.text(slide, f"{mean:.3f}", 10.80, ybar - 0.13,
                          0.95, 0.26, 10.5, color)
            self.line(slide, 0.70, yy + 1.10, 12.4, yy + 1.10, PALE, 0.5)
            self.provenance["slides"].append({"slide": 4, "unit": unit,
                "cases": len(subset), "removed_percent": fraction,
                "mean_reaction_ratio": {key: float(np.mean([
                    case["interventions"][key]["final_raw_reaction"] /
                    case["baseline"]["final_raw_reaction"] for case in subset]))
                    for key, _, _ in strategies}})
        for idx, (_, name, color) in enumerate(strategies):
            xx = 0.74 + idx * 2.93
            self.line(slide, xx, 6.65, xx + 0.35, 6.65, color, 5.2)
            self.text(slide, name, xx + 0.45, 6.49, 2.37, 0.34, 11, color)
        self.provenance["source_files"].append(str(source.relative_to(ROOT)))
        return slide

    def figure_four_models(self):
        source = ROOT / "benchmarks" / "results" / "independent_fem_cycle_intervention.json"
        record = json.loads(source.read_text(encoding="utf-8"))
        cases = record["cases"]
        if len(cases) != 24 or any(case.get("status") != "complete" for case in cases.values()):
            raise ValueError("cross-model figure requires 24 completed cases")
        slide = self.slide("同图同夹持删环复核：反力保持依赖力学模型", 5,
            "线性梁 FEM 与简化模型共同拉伸至 1.08；候选源自简化模型 1.40 轨迹。模拟结果，不是材料实验。")
        self.text(slide, "早期低应变单环删除后的末帧原始反力 / 各自未删图反力",
                  0.70, 1.11, 11.9, 0.40, 15, bold=True)
        units = (("square", "方形"), ("hexagon", "六边形"),
                 ("ring", "圆环"), ("kagome", "笼目"))
        colors = {11: BLUE, 23: ORANGE, 41: NAVY}
        for col, (unit, label) in enumerate(units):
            x0 = 0.70 + col * 3.10
            self.text(slide, label, x0, 1.72, 2.86, 0.36, 17, bold=True)
            values = []
            plot_top, plot_bottom = 2.45, 5.93
            def ordinate(ratio):
                if not np.isfinite(ratio) or ratio < .72 or ratio > 1.03:
                    raise ValueError("cross-model ratio outside shared figure scale")
                return plot_bottom - (ratio - .72) / .31 * (plot_bottom - plot_top)
            for tick in (.8, .9, 1.0):
                yy = ordinate(tick)
                self.line(slide, x0 + .34, yy, x0 + 2.70, yy, PALE, .6)
                self.text(slide, f"{tick:.1f}", x0, yy - .12, .32, .26, 9, MID)
            for seed in (11, 23, 41):
                for direction in ("x", "y"):
                    case = cases[f"{unit}:seed{seed}:{direction}"]
                    selected = case["interventions"]["low_early"]
                    reduced = (selected["reduced_raw_reaction"] /
                               case["baseline"]["reduced_raw_reaction"])
                    fem = (selected["fem_right_reaction"] /
                           case["baseline"]["fem_right_reaction"])
                    values.append((float(reduced), float(fem)))
                    xa, xb = x0 + .81, x0 + 2.37
                    ya, yb = ordinate(reduced), ordinate(fem)
                    self.line(slide, xa, ya, xb, yb, colors[seed], 1.1)
                    shape_type = MSO_SHAPE.OVAL if direction == "x" else MSO_SHAPE.DIAMOND
                    for xx, yy in ((xa, ya), (xb, yb)):
                        dot = slide.shapes.add_shape(shape_type,
                            Inches(xx - .036), Inches(yy - .036),
                            Inches(.072), Inches(.072))
                        dot.fill.solid()
                        dot.fill.fore_color.rgb = colors[seed]
                        dot.line.fill.background()
            mean_reduced = float(np.mean([value[0] for value in values]))
            mean_fem = float(np.mean([value[1] for value in values]))
            self.text(slide, f"{mean_reduced:.3f} → {mean_fem:.3f}",
                      x0, 2.08, 2.86, .32, 13.2, color=BLUE, bold=True)
            self.text(slide, "简化模型", x0 + .43, 6.04, 1.12, .30, 10, MID)
            self.text(slide, "梁 FEM", x0 + 2.07, 6.04, .72, .30, 10, MID)
            self.provenance["slides"].append({"slide": 5, "unit": unit,
                "cases": 6, "mean_reduced_ratio": mean_reduced,
                "mean_fem_ratio": mean_fem})
        self.line(slide, .74, 6.46, 12.52, 6.46, PALE, .7)
        self.text(slide, "颜色为几何种子 11 / 23 / 41；圆点为 x，菱形为 y。每对方向共享一个基结构。",
                  .76, 6.53, 11.8, .30, 10.5, MID)
        self.provenance["source_files"].append(str(source.relative_to(ROOT)))
        return slide

    def run(self):
        self.figure_one()
        self.figure_two()
        self.figure_three()
        self.figure_four()
        self.figure_four_models()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix(".tmp.pptx")
        try:
            self.prs.save(str(temporary))
            os.replace(temporary, self.output)
        finally:
            temporary.unlink(missing_ok=True)
        self.provenance["source_files"].extend([
            "fibernet/gen/manufacturing.py", "fibernet/sim/reduced_beam.py",
            "fibernet/analysis/tensile_recruitment.py",
            "fibernet/analysis/snapshot_features.py",
            "fibernet/analysis/width_contact.py",
            "fibernet/ml/beam_frame_fem.py",
            "fibernet/ml/beam_frame_fem_sparse.py",
            "manuscript/scripts/build_figures_ppt.py"])
        sources = sorted({Path(name).as_posix() for name in
                          self.provenance["source_files"]})
        self.provenance["source_files"] = sources
        self.provenance["source_sha256"] = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in sources}
        self.provenance["pptx_sha256"] = hashlib.sha256(
            self.output.read_bytes()).hexdigest()
        manifest = self.output.with_suffix(".json")
        temporary = manifest.with_suffix(".tmp.json")
        try:
            temporary.write_text(json.dumps(self.provenance, ensure_ascii=False,
                                            indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, manifest)
        finally:
            temporary.unlink(missing_ok=True)
        print("[figure_deck] saved 5 editable working figures")
        return self.output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    FigureDeckBuilder(args.output).run()

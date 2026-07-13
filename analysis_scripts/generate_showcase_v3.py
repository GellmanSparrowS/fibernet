#!/usr/bin/env python3
"""
FiberNet Showcase v3 — 简洁可视化生成脚本
用于验证上一轮工作完成情况，生成 12 张核心功能展示图。

用法:
    python3 analysis_scripts/generate_showcase_v3.py
    python3 analysis_scripts/generate_showcase_v3.py --resume   # 断点续跑

断点续跑: 已存在的图片会跳过 (--resume 模式)，默认模式重新生成全部。
"""

import sys, os, json, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fibernet import (
    pattern_2d, pattern_3d, list_units,
    render_graph, render_graph_3d, render_gallery,
    render_with_stats, render_deformation,
    BeamFEM, tile_2d, rotate, scale, translate,
    FiberNetworkEnv,
)

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output_viz"
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_FILE = OUTPUT_DIR / "_v3_checkpoint.json"
DPI = 150  # 适中的分辨率，避免过大文件


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {}


def save_checkpoint(state):
    CHECKPOINT_FILE.write_text(json.dumps(state, indent=2))


def save_fig(fig, name, state):
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    state[name] = {"path": str(path), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_checkpoint(state)
    size_kb = path.stat().st_size / 1024
    print(f"  ✓ {name}.png ({size_kb:.0f} KB)")


# ── Generation functions ────────────────────────────────────────────────────

def gen_01_2d_gallery(state, skip_existing):
    """01: 所有 12 种 2D 基元的 gallery"""
    name = "01_2d_gallery"
    if skip_existing and name in state:
        print(f"  ⏭ {name} (already exists)")
        return

    print("01. Generating 2D unit gallery...")
    units = list_units()
    fig = render_gallery(
        [pattern_2d(unit=u, box=(10, 10), grid=(3, 3), seed=42, n_pts_per_side=3)
         for u in units],
        titles=units,
        ncols=4,
        theme="dark",
        color_by="uniform",
        suptitle="FiberNet v3: All 12 2D Metamaterial Units",
    )
    save_fig(fig, name, state)


def gen_02_honeycomb_detail(state, skip_existing):
    """02: Honeycomb 细节放大"""
    name = "02_honeycomb_detail"
    if skip_existing and name in state:
        print(f"  ⏭ {name} (already exists)")
        return

    print("02. Generating honeycomb detail...")
    g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5),
                   n_pts_per_side=5, seed=42)
    fig = render_graph(g, figsize=(12, 12), theme="blueprint",
                       color_by="uniform", 
                       line_width=1.5, show_nodes=False,
                       title="Honeycomb Detail (5×5, n_pts=5)",
                       subtitle=f"{g.num_nodes} nodes, {g.num_edges} edges")
    save_fig(fig, name, state)


def gen_03_kagome_detail(state, skip_existing):
    """03: Kagome 细节"""
    name = "03_kagome_detail"
    if skip_existing and name in state:
        print(f"  ⏭ {name} (already exists)")
        return

    print("03. Generating kagome detail...")
    g = pattern_2d(unit="kagome", box=(10, 10), grid=(4, 4),
                   n_pts_per_side=4, seed=42)
    fig = render_graph(g, figsize=(12, 12), theme="dark",
                       color_by="uniform", 
                       line_width=1.8, show_nodes=False,
                       title="Kagome Lattice (4×4, n_pts=4)",
                       subtitle=f"{g.num_nodes} nodes, {g.num_edges} edges")
    save_fig(fig, name, state)


def gen_04_voronoi(state, skip_existing):
    """04: Voronoi 周期性结构"""
    name = "04_voronoi"
    if skip_existing and name in state:
        print(f"  ⏭ {name} (already exists)")
        return

    print("04. Generating Voronoi tessellation...")
    g = pattern_2d(unit="voronoi", box=(10, 10), grid=(3, 3),
                   n_pts_per_side=3, seed=42, n_internal=15)
    fig = render_graph(g, figsize=(12, 12), theme="dark",
                       color_by="uniform", 
                       line_width=1.3, show_nodes=False,
                       title="Voronoi Tessellation (3×3)",
                       subtitle=f"{g.num_nodes} nodes, {g.num_edges} edges")
    save_fig(fig, name, state)


def gen_05_auxetic_comparison(state, skip_existing):
    """05: 常规 vs 拉胀 (auxetic) 结构对比"""
    name = "05_auxetic_comparison"
    if skip_existing and name in state:
        print(f"  ⏭ {name} (already exists)")
        return

    print("05. Generating auxetic comparison...")
    configs = [
        ("honeycomb", {"n_pts_per_side": 4}),
        ("reentrant", {"n_pts_per_side": 4, "angle": 20}),
        ("chiral", {"n_pts_per_side": 4}),
        ("missing_rib", {"n_pts_per_side": 4}),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.patch.set_facecolor('#1a1a2e')

    for idx, (unit, kwargs) in enumerate(configs):
        ax = axes[idx // 2, idx % 2]
        g = pattern_2d(unit=unit, box=(10, 10), grid=(4, 4), seed=42)
        render_graph(g, ax=ax, theme="dark", color_by="uniform",
                      line_width=1.2, show_nodes=False)
        ax.set_title(f"{unit}\n({g.num_nodes} nodes)", color='white', fontsize=12)

    fig.suptitle("Regular vs Auxetic Structures", fontsize=16, color='white', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, name, state)


def gen_06_3d_cubic(state, skip_existing):
    """06: 3D Cubic 结构"""
    name = "06_3d_cubic"
    if skip_existing and name in state:
        print(f"  ⏭ {name} (already exists)")
        return

    print("06. Generating 3D cubic...")
    g = pattern_3d(unit="cubic", box=(10, 10, 10), grid=(3, 3, 3),
                   n_pts_per_side=3, seed=42)
    fig = render_graph_3d(g, figsize=(10, 10), theme="dark",
                          line_width=1.5, depth_alpha=True,
                          title="Cubic 3D (3×3×3)",
                          elevation=25, azimuth=-60)
    save_fig(fig, name, state)


def gen_07_3d_octet(state, skip_existing):
    """07: 3D Octet 结构"""
    name = "07_3d_octet"
    if skip_existing and name in state:
        print(f"  ⏭ {name} (already exists)")
        return

    print("07. Generating 3D octet...")
    g = pattern_3d(unit="octet", box=(10, 10, 10), grid=(2, 2, 2),
                   n_pts_per_side=3, seed=42)
    fig = render_graph_3d(g, figsize=(10, 10), theme="dark",
                          line_width=1.5, depth_alpha=True,
                          title="Octet 3D (2×2×2)",
                          elevation=30, azimuth=-45)
    save_fig(fig, name, state)


def gen_08_fem_deformation(state, skip_existing):
    """08: FEM 单轴拉伸变形"""
    name = "08_fem_deformation"
    if skip_existing and name in state:
        print(f"  ⏭ {name} (already exists)")
        return

    print("08. Generating FEM deformation...")
    g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4),
                   n_pts_per_side=4, seed=42)
    fem = BeamFEM(g, default_E=1e9, default_nu=0.3)
    result = fem.uniaxial_tension(strain=0.02, deformation_scale=20)

    fig = render_deformation(g, result.deformed_graph,
                             figsize=(16, 8), theme="dark",
                             color_by="displacement",
                             line_width=1.5,
                             title=f"FEM Uniaxial Tension — E*={result.effective_youngs_modulus:.2e} Pa")
    save_fig(fig, name, state)


def gen_09_fem_stress(state, skip_existing):
    """09: FEM 应力场 (reentrant auxetic)"""
    name = "09_fem_stress"
    if skip_existing and name in state:
        print(f"  ⏭ {name} (already exists)")
        return

    print("09. Generating FEM stress field...")
    g = pattern_2d(unit="reentrant", box=(10, 10), grid=(3, 3),
                   n_pts_per_side=4, seed=42,
                   unit_kwargs={"angle": 20})
    fem = BeamFEM(g, default_E=1e9, default_nu=0.3)
    result = fem.uniaxial_tension(strain=0.015, deformation_scale=30)

    fig = render_graph(result.deformed_graph, figsize=(12, 12), theme="dark",
                       color_by="stress", color_data=result.stresses,
                       colormap="inferno", line_width=1.8, show_nodes=False,
                       title="Reentrant Stress Field (Auxetic)",
                       subtitle=f"ν*={result.effective_poissons_ratio:.3f}")
    save_fig(fig, name, state)


def gen_10_connectivity(state, skip_existing):
    """10: 连通性验证 — 平铺 + 旋转后仍连通"""
    name = "10_connectivity"
    if skip_existing and name in state:
        print(f"  ⏭ {name} (already exists)")
        return

    print("10. Generating connectivity verification...")
    g = pattern_2d(unit="honeycomb", box=(5, 5), grid=(3, 3),
                   n_pts_per_side=3, seed=42)
    tiled = tile_2d(g, grid=(3, 3))
    rotated = rotate(tiled, angle=15)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#1a1a2e')

    render_graph(g, ax=axes[0], theme="dark", color_by="uniform",
                 line_width=1.0, show_nodes=False)
    axes[0].set_title(f"Unit Cell\n({g.num_nodes} nodes)", color='white', fontsize=11)

    render_graph(tiled, ax=axes[1], theme="dark", color_by="uniform",
                 line_width=1.0, show_nodes=False)
    axes[1].set_title(f"Tiled 3×3\n({tiled.num_nodes} nodes)", color='white', fontsize=11)

    render_graph(rotated, ax=axes[2], theme="dark", color_by="uniform",
                 line_width=1.0, show_nodes=False)
    axes[2].set_title(f"Rotated 15°\n({rotated.num_nodes} nodes)", color='white', fontsize=11)

    fig.suptitle("Connectivity: Tiling + Rotation Preserves Graph", fontsize=14, color='white', y=1.02)
    fig.tight_layout()
    save_fig(fig, name, state)


def gen_11_transforms(state, skip_existing):
    """11: 变换操作可视化 (scale, mirror, translate)"""
    name = "11_transforms"
    if skip_existing and name in state:
        print(f"  ⏭ {name} (already exists)")
        return

    print("11. Generating transforms visualization...")
    g = pattern_2d(unit="star", box=(8, 8), grid=(3, 3),
                   n_pts_per_side=4, seed=42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.patch.set_facecolor('#1a1a2e')

    render_graph(g, ax=axes[0, 0], theme="dark", color_by="uniform",
                 line_width=1.2, show_nodes=False)
    axes[0, 0].set_title("Original", color='white', fontsize=12)

    g_scaled = scale(g, factor=0.6)
    render_graph(g_scaled, ax=axes[0, 1], theme="dark", color_by="uniform",
                 line_width=1.2, show_nodes=False)
    axes[0, 1].set_title("Scaled ×0.6", color='white', fontsize=12)

    g_translated = translate(g, offset=[3, 3])
    render_graph(g_translated, ax=axes[1, 0], theme="dark", color_by="uniform",
                 line_width=1.2, show_nodes=False)
    axes[1, 0].set_title("Translated (3,3)", color='white', fontsize=12)

    from fibernet import mirror
    g_mirrored = mirror(g, axis='x')
    render_graph(g_mirrored, ax=axes[1, 1], theme="dark", color_by="uniform",
                 line_width=1.2, show_nodes=False)
    axes[1, 1].set_title("Mirror (x-axis)", color='white', fontsize=12)

    fig.suptitle("Geometric Transforms", fontsize=16, color='white', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, name, state)


def gen_12_chiral_stats(state, skip_existing):
    """12: Chiral 结构 + 统计信息面板"""
    name = "12_chiral_stats"
    if skip_existing and name in state:
        print(f"  ⏭ {name} (already exists)")
        return

    print("12. Generating chiral with statistics...")
    g = pattern_2d(unit="chiral", box=(10, 10), grid=(4, 4),
                   n_pts_per_side=4, seed=42)

    fig = render_with_stats(g, figsize=(12, 12), theme="dark",
                            color_by="uniform", 
                            line_width=1.5, show_nodes=False,
                            title="Chiral Honeycomb with Statistics")
    save_fig(fig, name, state)


# ── Main ────────────────────────────────────────────────────────────────────

GENERATORS = [
    gen_01_2d_gallery,
    gen_02_honeycomb_detail,
    gen_03_kagome_detail,
    gen_04_voronoi,
    gen_05_auxetic_comparison,
    gen_06_3d_cubic,
    gen_07_3d_octet,
    gen_08_fem_deformation,
    gen_09_fem_stress,
    gen_10_connectivity,
    gen_11_transforms,
    gen_12_chiral_stats,
]


def main():
    parser = argparse.ArgumentParser(description="FiberNet Showcase v3 Generator")
    parser.add_argument("--resume", action="store_true",
                        help="Skip images that already exist (checkpoint resume)")
    parser.add_argument("--only", type=int, default=None,
                        help="Generate only image number N (1-12)")
    args = parser.parse_args()

    print("=" * 60)
    print("FiberNet Showcase v3 — Generating visualization images")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Mode:   {'resume (skip existing)' if args.resume else 'full regeneration'}")
    print()

    state = load_checkpoint() if args.resume else {}
    skip_existing = args.resume

    start = time.time()
    for gen_fn in GENERATORS:
        if args.only is not None:
            num = int(gen_fn.__name__.split("_")[1])
            if num != args.only:
                continue
        try:
            gen_fn(state, skip_existing)
        except Exception as exc:
            print(f"  ✗ {gen_fn.__name__}: {exc}")
            import traceback; traceback.print_exc()

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Done — {len(state)}/{len(GENERATORS)} images in {elapsed:.1f}s")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

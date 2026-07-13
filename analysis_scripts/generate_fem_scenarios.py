#!/usr/bin/env python3
"""
FiberNet FEM Scenarios v2 — 经典力学模拟场景可视化 (修正版)

修正:
- deformation_scale 从 15 降到 5，避免视觉"爆炸"
- 断裂强度根据实际应力校准
- 所有非应力场图使用 uniform 单色

用法:
    python3 analysis_scripts/generate_fem_scenarios.py
    python3 analysis_scripts/generate_fem_scenarios.py --resume
    python3 analysis_scripts/generate_fem_scenarios.py --only 2
"""

import sys, os, json, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fibernet import pattern_2d, render_graph, render_deformation, BeamFEM

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output_viz" / "fem_scenarios"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUTPUT_DIR / "_checkpoint.json"
DPI = 150
DEFORM_SCALE = 5  # 适中的形变可视化比例


def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {}

def save_checkpoint(state):
    CHECKPOINT.write_text(json.dumps(state, indent=2))

def save_fig(fig, name, state):
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    state[name] = {"path": str(path), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_checkpoint(state)
    size_kb = path.stat().st_size / 1024
    print(f"  ✓ {name}.png ({size_kb:.0f} KB)")


# ── Scenario functions ─────────────────────────────────────────────────────

def scenario_01_uniaxial_tension(state, skip):
    """01: 单轴拉伸 (弹性区域) — 变形前后对比"""
    name = "01_uniaxial_tension"
    if skip and name in state:
        print(f"  ⏭ {name}"); return

    print("01. Uniaxial tension (elastic)...")
    g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5), seed=42)
    fem = BeamFEM(g, default_E=1e9, default_nu=0.3)
    result = fem.uniaxial_tension(strain=0.01, deformation_scale=DEFORM_SCALE)
    fem.save_result(result, str(OUTPUT_DIR / "uniaxial_tension.json"))

    fig = render_deformation(
        g, result.deformed_graph,
        figsize=(16, 8), theme="dark",
        line_width=1.5,
        title=f"Uniaxial Tension (ε=0.01) — E*={result.effective_youngs_modulus:.2e} Pa"
    )
    save_fig(fig, name, state)


def scenario_02_uniaxial_fracture(state, skip):
    """02: 单轴拉伸至断裂 — kagome (高应力集中)"""
    name = "02_uniaxial_fracture"
    if skip and name in state:
        print(f"  ⏭ {name}"); return

    print("02. Uniaxial tension to fracture...")
    # Use kagome: has higher stress at moderate strain
    g = pattern_2d(unit="kagome", box=(10, 10), grid=(4, 4), seed=42)
    fem = BeamFEM(g, default_E=1e9, default_nu=0.3)

    # Calibrate: at strain=0.01, max stress ~1e7, so strength=3e7 breaks at ~0.03
    result = fem.tensile_fracture(
        max_strain=0.05, n_steps=10,
        strength=3e7, deformation_scale=DEFORM_SCALE
    )
    fem.save_result(result, str(OUTPUT_DIR / "uniaxial_fracture.json"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#0a0a0f')

    # Left: final deformed state (uniform purple)
    if result.deformed_graph:
        render_graph(result.deformed_graph, ax=ax1, theme="dark",
                     color_by="uniform", line_width=1.5, show_nodes=False,
                     title="Final State")

    # Right: fracture history
    if result.history:
        strains = [h["strain"] for h in result.history]
        stresses = [h["stress_max"] for h in result.history]
        n_frac = [h["n_fractured"] for h in result.history]

        ax2_twin = ax2.twinx()
        ax2.plot(strains, stresses, 'o-', color='#b388ff', linewidth=2,
                 label="Max Stress", markersize=6)
        ax2_twin.plot(strains, n_frac, 's--', color='#ff6644', linewidth=2,
                      label="Fractured Edges", markersize=6)
        ax2.set_xlabel("Strain", color='#d0d0d0')
        ax2.set_ylabel("Max Stress (Pa)", color='#b388ff')
        ax2_twin.set_ylabel("Fractured Edges", color='#ff6644')
        ax2.set_facecolor('#0a0a0f')
        ax2.tick_params(colors='#d0d0d0')
        ax2_twin.tick_params(colors='#d0d0d0')
        ax2.set_title("Fracture Progression", color='white', fontsize=12)
        ax2.legend(loc='upper left', facecolor='#1a1a2a')
        ax2_twin.legend(loc='upper right', facecolor='#1a1a2a')

    fig.suptitle("Progressive Fracture (Kagome)",
                 color='white', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, name, state)


def scenario_03_biaxial(state, skip):
    """03: 双轴拉伸"""
    name = "03_biaxial"
    if skip and name in state:
        print(f"  ⏭ {name}"); return

    print("03. Biaxial tension...")
    g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5), seed=42)
    fem = BeamFEM(g, default_E=1e9, default_nu=0.3)
    result = fem.biaxial_tension(strain_x=0.01, strain_y=0.01, deformation_scale=DEFORM_SCALE)
    fem.save_result(result, str(OUTPUT_DIR / "biaxial.json"))

    fig = render_deformation(
        g, result.deformed_graph,
        figsize=(16, 8), theme="dark",
        line_width=1.5,
        title=f"Biaxial Tension (εx=εy=0.01) — E*={result.effective_youngs_modulus:.2e} Pa"
    )
    save_fig(fig, name, state)


def scenario_04_compression(state, skip):
    """04: 单轴压缩"""
    name = "04_compression"
    if skip and name in state:
        print(f"  ⏭ {name}"); return

    print("04. Uniaxial compression...")
    g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5), seed=42)
    fem = BeamFEM(g, default_E=1e9, default_nu=0.3)
    result = fem.compression(strain=0.01, deformation_scale=DEFORM_SCALE)
    fem.save_result(result, str(OUTPUT_DIR / "compression.json"))

    fig = render_deformation(
        g, result.deformed_graph,
        figsize=(16, 8), theme="dark",
        line_width=1.5,
        title=f"Compression (ε=-0.01) — E*={abs(result.effective_youngs_modulus):.2e} Pa"
    )
    save_fig(fig, name, state)


def scenario_05_shear(state, skip):
    """05: 剪切试验"""
    name = "05_shear"
    if skip and name in state:
        print(f"  ⏭ {name}"); return

    print("05. Shear test...")
    g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5), seed=42)
    fem = BeamFEM(g, default_E=1e9, default_nu=0.3)
    result = fem.shear_test(strain=0.01, deformation_scale=DEFORM_SCALE)
    fem.save_result(result, str(OUTPUT_DIR / "shear.json"))

    fig = render_deformation(
        g, result.deformed_graph,
        figsize=(16, 8), theme="dark",
        line_width=1.5,
        title=f"Shear Test (γ=0.01) — G*={result.effective_youngs_modulus:.2e} Pa"
    )
    save_fig(fig, name, state)


def scenario_06_stress_strain_curve(state, skip):
    """06: 应力-应变曲线 (多结构对比)"""
    name = "06_stress_strain_curve"
    if skip and name in state:
        print(f"  ⏭ {name}"); return

    print("06. Stress-strain curves...")
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#0a0a0f')

    structures = [
        ("honeycomb", {}),
        ("kagome", {}),
        ("reentrant", {"angle": 20}),
        ("chiral", {}),
    ]

    colors = ['#b388ff', '#7c4dff', '#ff6644', '#4caf50']
    all_data = {}

    for (unit, kwargs), color in zip(structures, colors):
        g = pattern_2d(unit=unit, box=(10, 10), grid=(4, 4),
                       seed=42, unit_kwargs=kwargs or None)
        fem = BeamFEM(g, default_E=1e9, default_nu=0.3)
        strains, stresses = fem.stress_strain_curve(max_strain=0.03, n_steps=8)
        ax.plot(strains * 100, stresses / 1e6, 'o-', color=color,
                linewidth=2, markersize=6, label=f"{unit}")
        all_data[unit] = {
            "strains": strains.tolist(),
            "stresses": stresses.tolist(),
        }

    ax.set_xlabel("Strain (%)", color='#d0d0d0', fontsize=12)
    ax.set_ylabel("Stress (MPa)", color='#d0d0d0', fontsize=12)
    ax.set_title("Stress-Strain Curves", color='white', fontsize=14, fontweight='bold')
    ax.set_facecolor('#0a0a0f')
    ax.tick_params(colors='#d0d0d0')
    ax.legend(loc='upper left', facecolor='#1a1a2a', fontsize=11)
    ax.grid(True, alpha=0.2, color='#1a1a2a')

    fig.tight_layout()
    save_fig(fig, name, state)

    # Save curve data as JSON for RL
    with open(OUTPUT_DIR / "stress_strain_curves.json", "w") as f:
        json.dump(all_data, f, indent=2)


# ── Main ────────────────────────────────────────────────────────────────────

SCENARIOS = [
    scenario_01_uniaxial_tension,
    scenario_02_uniaxial_fracture,
    scenario_03_biaxial,
    scenario_04_compression,
    scenario_05_shear,
    scenario_06_stress_strain_curve,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("FiberNet FEM Scenarios v2 — Clean Simulation Visualizations")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Mode:   {'resume' if args.resume else 'full'}")
    print(f"Deform scale: {DEFORM_SCALE}")
    print()

    state = load_checkpoint() if args.resume else {}
    skip = args.resume

    t0 = time.time()
    for fn in SCENARIOS:
        if args.only is not None:
            num = int(fn.__name__.split("_")[1])
            if num != args.only:
                continue
        try:
            fn(state, skip)
        except Exception as exc:
            print(f"  ✗ {fn.__name__}: {exc}")
            import traceback; traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done — {len(state)}/{len(SCENARIOS)} scenarios in {elapsed:.1f}s")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

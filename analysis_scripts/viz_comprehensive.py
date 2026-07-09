#!/usr/bin/env python3
"""
Pattern Engine v7 — Single Comprehensive Visualization
======================================================
All visualizations in ONE image, rows by theme.
- 5 columns per row, no empty slots
- All shapes have polyline complexity (no undeformed)
- Grid size shown in title
- Custom shapes with fit_to_box demonstrated
"""

import sys, os, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fibernet.viz.showcase_renderer import render_2d_grid
from analysis_scripts.pattern_engine_unified import pattern_2d, check_connectivity

OUT = Path('/home/codex/projects/codex_test/fibernet/output_viz')
os.makedirs(OUT, exist_ok=True)

C = '#00e87b'


def build():
    all_nets, all_titles = [], []

    # ── ROW 1: Custom shapes with boundary contact (5 shapes) ──
    custom_boundary = [
        ('L-shape 6pts', [(0,0),(10,0),(10,3),(3,3),(3,10),(0,10)], True),
        ('Cross 12pts', [(3,0),(7,0),(7,3),(10,3),(10,7),(7,7),(7,10),(3,10),(3,7),(0,7),(0,3),(3,3)], True),
        ('H-shape 12pts', [(0,0),(3,0),(3,3.5),(7,3.5),(7,0),(10,0),(10,10),(7,10),(7,6.5),(3,6.5),(3,10),(0,10)], True),
        ('T-shape 8pts', [(0,0),(10,0),(10,3),(6.5,3),(6.5,10),(3.5,10),(3.5,3),(0,3)], True),
        ('Diamond 4pts', [(5,0),(10,5),(5,10),(0,5)], True),
    ]
    for name, pts, closed in custom_boundary:
        net = pattern_2d(box=(10,10), points=pts, closed=closed, grid=(3,3),
                        mirror_x=True, mirror_y=True)
        all_nets.append(net)
        all_titles.append(f'[Custom] {name}\n3×3 mirror_xy')

    # ── ROW 2: Custom shapes with fit_to_box + open shapes (5 shapes) ──
    custom_fit = [
        ('Star fit', [(50,0),(65,35),(100,50),(65,65),(50,100),(35,65),(0,50),(35,35)], True),
        ('Arrow fit', [(0,40),(60,40),(60,0),(100,50),(60,100),(60,60),(0,60)], True),
        ('Spiral open', [(5+3*np.cos(t), 5+3*np.sin(t)) for t in np.linspace(0, 4*np.pi, 15)], False),
        ('Wave open', [(i, 5+3*np.sin(i*0.8)) for i in np.linspace(0, 10, 12)], False),
        ('Zigzag open', [(0,5),(5,10),(10,5),(15,10),(20,5)], False),
    ]
    for name, pts, closed in custom_fit:
        bmode = 'extend' if not closed else 'none'
        net = pattern_2d(box=(10,10), points=pts, closed=closed, fit_to_box=True, boundary_mode=bmode,
                        grid=(3,3), mirror_x=True, mirror_y=True)
        all_nets.append(net)
        all_titles.append(f'[Custom+fit] {name}\n3×3 mirror_xy')

    # ── ROW 3: Polygon presets with perturbation (5 configs) ──
    poly_configs = [
        ('square', 8, 0.15, 'Square n=8 pert=0.15'),
        ('square', 12, 0.2, 'Square n=12 pert=0.2'),
        ('triangle', 8, 0.15, 'Triangle n=8 pert=0.15'),
        ('triangle', 12, 0.2, 'Triangle n=12 pert=0.2'),
        ('hexagon', 8, 0.2, 'Hexagon n=8 pert=0.2'),
    ]
    for shape, n, pert, title in poly_configs:
        net = pattern_2d(polygon_type=shape, n_pts_per_side=n, perturbation=pert,
                        seed=42, grid=(3,3))
        all_nets.append(net)
        all_titles.append(f'[Polygon] {title}\n3×3')

    # ── ROW 4: Tiling transforms (5 configs) ──
    transforms = [
        ('square', 8, 0.2, True, False, 0, 'Sq +mirror_x'),
        ('square', 8, 0.2, False, True, 0, 'Sq +mirror_y'),
        ('square', 8, 0.2, True, True, 0, 'Sq +mirror_xy'),
        ('hexagon', 8, 0.2, True, True, 0, 'Hex +mirror_xy'),
        ('square', 8, 0.2, True, True, 45, 'Sq +mirror+rot45'),
    ]
    for shape, n, pert, mx, my, rot, title in transforms:
        net = pattern_2d(polygon_type=shape, n_pts_per_side=n, perturbation=pert,
                        seed=42, grid=(3,3), mirror_x=mx, mirror_y=my, rotation=rot)
        all_nets.append(net)
        all_titles.append(f'[Transform] {title}\n3×3')

    # ── ROW 5: Diversity (seed, pert, mirror combos) (5 configs) ──
    diversity = [
        ('square', 5, 0.15, 42, True, True, 'Sq n=5 p=0.15 s42'),
        ('square', 10, 0.2, 99, True, True, 'Sq n=10 p=0.2 s99'),
        ('square', 20, 0.1, 77, True, True, 'Sq n=20 p=0.1 s77'),
        ('triangle', 8, 0.2, 55, False, False, 'Tri n=8 p=0.2 s55'),
        ('hexagon', 8, 0.2, 123, True, True, 'Hex n=8 p=0.2 s123'),
    ]
    for shape, n, pert, seed, mx, my, title in diversity:
        net = pattern_2d(polygon_type=shape, n_pts_per_side=n, perturbation=pert,
                        seed=seed, grid=(3,3), mirror_x=mx, mirror_y=my)
        all_nets.append(net)
        all_titles.append(f'[Diversity] {title}\n3×3')

    # ── ROW 6: Grid sizes + boundary modes (5 configs) ──
    grid_and_boundary = [
        # Grid sizes
        lambda: pattern_2d(polygon_type='square', n_pts_per_side=5, perturbation=0.2,
                          seed=42, grid=(2,2)),
        lambda: pattern_2d(polygon_type='square', n_pts_per_side=5, perturbation=0.2,
                          seed=42, grid=(4,4)),
        lambda: pattern_2d(polygon_type='square', n_pts_per_side=5, perturbation=0.2,
                          seed=42, grid=(5,5)),
        # Boundary extend
        lambda: pattern_2d(box=(10,10), points=[(3,3),(7,3),(7,7),(3,7)], closed=True,
                          grid=(3,3), boundary_mode='extend'),
        # Custom interior with mirror
        lambda: pattern_2d(box=(10,10), points=[(2,2),(8,2),(8,8),(2,8)], closed=True,
                          grid=(3,3), boundary_mode='extend', mirror_x=True, mirror_y=True),
    ]
    grid_titles = [
        'Square 2×2\npert=0.2',
        'Square 4×4\npert=0.2',
        'Square 5×5\npert=0.2',
        'Interior 3×3\nboundary=extend',
        'Interior+mirror 3×3\nboundary=extend',
    ]
    for fn, title in zip(grid_and_boundary, grid_titles):
        net = fn()
        all_nets.append(net)
        all_titles.append(f'[Grid/Bound] {title}')

    # ── VERIFY ALL CONNECTED ──
    print("=== CONNECTIVITY CHECK ===")
    all_ok = True
    for i, (net, title) in enumerate(zip(all_nets, all_titles)):
        c = check_connectivity(net)
        if c != 1:
            all_ok = False
            print(f"  ❌ Row {(i//5)+1} [{i%5}]: {title} → {c} components")
    if all_ok:
        print(f"  ✅ All {len(all_nets)} structures connected")
    else:
        print(f"  ⚠️  Some disconnected")

    # ── RENDER ──
    n = len(all_nets)
    ncols = 5
    nrows = n // ncols
    print(f"\nTotal images: {n} ({nrows} rows × {ncols} cols)")
    out_path = OUT / 'comprehensive_api_demo.png'
    render_2d_grid(all_nets, all_titles, save_path=str(out_path), color=C)
    fsize = out_path.stat().st_size / 1024
    print(f"Saved: {out_path} ({fsize:.0f} KB)")
    return out_path


if __name__ == "__main__":
    print("=" * 60)
    print("PATTERN ENGINE v7 — COMPREHENSIVE VISUALIZATION")
    print("Single image, 6 rows × 5 cols, 6 themes")
    print("=" * 60)
    build()

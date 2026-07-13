#!/usr/bin/env python3
"""
FiberNet Showcase — publication-grade visualization suite.

Generates every figure in output_viz/showcase_final/ with:
  • checkpoint / resume (survives disconnections)
  • memory-safe operation (gc + explicit deletion)
  • pyvista for 3-D, matplotlib for 2-D
  • square canvas, no axes, dark theme, glow effect

Usage
-----
    python generate_showcase_final.py          # run all
    python generate_showcase_final.py --resume # resume from checkpoint
"""

import sys, os, gc, json, time, traceback, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import fibernet as fn


# ── config ────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / 'output_viz' / 'showcase_final'
CHECKPOINT = OUTPUT_DIR / 'checkpoint.json'
DPI = 180
PANEL = 4.0          # inches per panel
BG = '#0d0d0d'
FG2D = '#00e87b'
FG3D = '#00b4ff'
TITLE_C = '#ffffff'

# 3-D settings
USE_PYVISTA = True
TUBE_R = 0.004
PANEL_PX = 600


# ── checkpoint helpers ────────────────────────────────────────
def load_ckpt():
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {'done': [], 'failed': []}

def save_ckpt(ckpt):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT, 'w') as f:
        json.dump(ckpt, f, indent=2)

def run_step(name, func, ckpt):
    if name in ckpt['done']:
        print(f"  [skip] {name}")
        return True
    print(f"  [run]  {name} ...")
    t0 = time.time()
    try:
        func()
        ckpt['done'].append(name)
        save_ckpt(ckpt)
        print(f"  [ok]   {name}  ({time.time()-t0:.1f}s)")
        return True
    except Exception as e:
        ckpt['failed'].append({'step': name, 'error': str(e),
                               'tb': traceback.format_exc()})
        save_ckpt(ckpt)
        print(f"  [FAIL] {name}: {e}")
        return False


# ── safe generator wrapper ────────────────────────────────────
def safe_create(name, **kw):
    try:
        return fn.create(name, **kw)
    except Exception as e:
        print(f"    WARN {name}: {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  2-D rendering
# ══════════════════════════════════════════════════════════════

def _extract_2d(net):
    lines = []
    if net is None: return lines
    for f in net.fibers:
        pts = f.centerline[:, :2]
        if len(pts) >= 2:
            lines.append(pts.copy())
    return lines

def _norm_lines(lines, pad=0.05):
    if not lines: return lines
    all_pts = np.vstack(lines)
    mn, mx = all_pts.min(0), all_pts.max(0)
    span = max(mx - mn) or 1.0
    out = []
    for pts in lines:
        p = (pts - mn) / span
        off = (1.0 - (mx - mn) / span) / 2.0
        p = p + off[:2]
        p = p * (1 - 2*pad) + pad
        out.append(p)
    return out

def _lw(n):
    if n <= 40: return 1.5
    if n <= 150: return 0.9
    if n <= 500: return 0.55
    return 0.35

def draw_2d_grid(nets, titles, save_path, color=FG2D, glow=True):
    valid = [(n,t) for n,t in zip(nets,titles) if n is not None and n.num_fibers > 0]
    if not valid: return
    ns, ts = zip(*valid)
    n = len(ns); nc = min(5, n); nr = int(np.ceil(n/nc))
    fig, axes = plt.subplots(nr, nc, figsize=(PANEL*nc, PANEL*nr))
    if nr==1 and nc==1: axes = np.array([axes])
    flat = np.array(axes).flatten()
    fig.patch.set_facecolor(BG)

    for i in range(n):
        ax = flat[i]
        ax.set_facecolor(BG); ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_aspect('equal'); ax.axis('off')
        lines = _norm_lines(_extract_2d(ns[i]))
        if lines:
            lw = _lw(len(lines))
            if glow:
                ax.add_collection(LineCollection(lines, linewidths=lw*3.5,
                    colors=color, alpha=0.06, antialiased=True))
                ax.add_collection(LineCollection(lines, linewidths=lw*2.0,
                    colors=color, alpha=0.14, antialiased=True))
            ax.add_collection(LineCollection(lines, linewidths=lw,
                colors=color, alpha=0.92, antialiased=True))
        if ts[i]:
            ax.text(0.5, 1.04, ts[i], ha='center', va='bottom',
                    color=TITLE_C, fontsize=9, fontweight='bold',
                    transform=ax.transAxes)
    for j in range(n, len(flat)):
        flat[j].set_facecolor(BG); flat[j].axis('off')
    plt.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=DPI, facecolor=BG, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig); gc.collect()


# ══════════════════════════════════════════════════════════════
#  3-D rendering  (pyvista)
# ══════════════════════════════════════════════════════════════

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
except Exception:
    USE_PYVISTA = False

def _extract_3d(net):
    lines = []
    if net is None: return lines
    for f in net.fibers:
        pts = f.centerline
        if len(pts) >= 2:
            if pts.shape[1] == 2:
                pts = np.column_stack([pts, np.zeros(len(pts))])
            lines.append(pts.copy())
    return lines

def _norm_3d(lines, pad=0.06):
    if not lines: return lines
    all_pts = np.vstack(lines)
    mn, mx = all_pts.min(0), all_pts.max(0)
    span = max(mx - mn) or 1.0
    out = []
    for pts in lines:
        p = (pts - mn) / span
        off = (1.0 - (mx - mn) / span) / 2.0
        p = p + off
        p = p * (1 - 2*pad) + pad
        out.append(p)
    return out

def _render_3d_panel(net, color=FG3D):
    """Render one 3D network, return numpy image array."""
    if not USE_PYVISTA:
        return np.zeros((PANEL_PX, PANEL_PX, 3), dtype=np.uint8) + 13

    pl = pv.Plotter(off_screen=True, window_size=(PANEL_PX, PANEL_PX))
    pl.set_background(BG)

    lines = _norm_3d(_extract_3d(net))
    if lines:
        pts_list = []
        cells_list = []
        off = 0
        for seg in lines:
            n = len(seg)
            pts_list.append(seg)
            cells_list.append([n] + list(range(off, off+n)))
            off += n
        pts = np.vstack(pts_list)
        cells = []
        for c in cells_list:
            cells.extend(c)
        poly = pv.PolyData()
        poly.points = pts
        poly.lines = np.array(cells, dtype=np.int64)

        if poly.n_points > 1:
            tube = poly.tube(radius=TUBE_R, n_sides=6)
            pl.add_mesh(tube, color=color, smooth_shading=True,
                       specular=0.4, specular_power=30, ambient=0.15)

    pl.add_light(pv.Light(position=(5,5,5), intensity=0.85, light_type="scene light"))
    pl.add_light(pv.Light(position=(-3,-2,4), intensity=0.25, light_type="scene light"))
    pl.camera_position = 'iso'
    img = pl.screenshot(return_img=True)
    pl.close()
    return img

def draw_3d_grid(nets, titles, save_path, color=FG3D):
    valid = [(n,t) for n,t in zip(nets,titles) if n is not None and n.num_fibers > 0]
    if not valid: return
    ns, ts = zip(*valid)
    n = len(ns); nc = min(5, n); nr = int(np.ceil(n/nc))

    # Render panels
    imgs = []
    for net in ns:
        imgs.append(_render_3d_panel(net, color))
        del net
    gc.collect()

    fig, axes = plt.subplots(nr, nc, figsize=(PANEL*nc, PANEL*nr))
    if nr==1 and nc==1: axes = np.array([axes])
    flat = np.array(axes).flatten()
    fig.patch.set_facecolor(BG)

    for i in range(n):
        ax = flat[i]
        ax.imshow(imgs[i]); ax.axis('off'); ax.set_facecolor(BG)
        if ts[i]:
            ax.set_title(ts[i], color=TITLE_C, fontsize=9,
                        fontweight='bold', pad=6)
    for j in range(n, len(flat)):
        flat[j].set_facecolor(BG); flat[j].axis('off')
    plt.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=DPI, facecolor=BG, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig); gc.collect()


# ══════════════════════════════════════════════════════════════
#  CATEGORIES
# ══════════════════════════════════════════════════════════════

def cat01_lattice_2d_topologies(d):
    nets = [
        safe_create('lattice_2d', topology='square', cell_size=8, grid_size=(6,6)),
        safe_create('lattice_2d', topology='triangular', cell_size=8, grid_size=(6,6)),
        safe_create('lattice_2d', topology='honeycomb', cell_size=8, grid_size=(6,6)),
        safe_create('lattice_2d', topology='kagome', cell_size=8, grid_size=(6,6)),
    ]
    draw_2d_grid(nets, ['Square', 'Triangular', 'Honeycomb', 'Kagome'],
                 d / '01_lattice_2d_topologies.png')

def cat02_lattice_2d_perturbation(d):
    nets = [
        safe_create('lattice_2d', topology='honeycomb', cell_size=8, grid_size=(6,6), perturbation=0.0),
        safe_create('lattice_2d', topology='honeycomb', cell_size=8, grid_size=(6,6), perturbation=0.1, seed=42),
        safe_create('lattice_2d', topology='honeycomb', cell_size=8, grid_size=(6,6), perturbation=0.2, seed=42),
        safe_create('lattice_2d', topology='honeycomb', cell_size=8, grid_size=(6,6), perturbation=0.35, seed=42),
    ]
    draw_2d_grid(nets, ['Perfect', 'ε=0.1', 'ε=0.2', 'ε=0.35'],
                 d / '02_lattice_2d_perturbation.png')

def cat03_lattice_2d_cell_size(d):
    nets = [
        safe_create('lattice_2d', topology='honeycomb', cell_size=4, grid_size=(10,10)),
        safe_create('lattice_2d', topology='honeycomb', cell_size=8, grid_size=(6,6)),
        safe_create('lattice_2d', topology='honeycomb', cell_size=14, grid_size=(4,4)),
        safe_create('lattice_2d', topology='honeycomb', cell_size=25, grid_size=(2,2)),
    ]
    draw_2d_grid(nets, ['Fine', 'Medium', 'Coarse', 'Ultra-coarse'],
                 d / '03_lattice_2d_cell_size.png')

def cat04_metamaterial_2d_modes(d):
    nets = [
        safe_create('metamaterial_2d', mode='reentrant', angle=150, cell_size=8),
        safe_create('metamaterial_2d', mode='chiral', cell_size=8),
        safe_create('metamaterial_2d', mode='star', cell_size=8),
        safe_create('metamaterial_2d', mode='arrowhead', cell_size=8),
    ]
    draw_2d_grid(nets, ['Re-entrant', 'Chiral', 'Star', 'Arrowhead'],
                 d / '04_metamaterial_2d_modes.png')

def cat05_metamaterial_2d_angle(d):
    nets = [
        safe_create('metamaterial_2d', mode='reentrant', angle=110, cell_size=8),
        safe_create('metamaterial_2d', mode='reentrant', angle=130, cell_size=8),
        safe_create('metamaterial_2d', mode='reentrant', angle=150, cell_size=8),
        safe_create('metamaterial_2d', mode='reentrant', angle=165, cell_size=8),
    ]
    draw_2d_grid(nets, ['110°', '130°', '150°', '165°'],
                 d / '05_metamaterial_2d_angle.png')

def cat06_curved_random_2d(d):
    nets = [
        safe_create('curved_random_2d', num_fibers=80, curvature_type='sinusoidal',
                    curvature_amplitude=3.0, seed=42),
        safe_create('curved_random_2d', num_fibers=80, curvature_type='bezier',
                    curvature_amplitude=5.0, seed=42),
        safe_create('curved_random_2d', num_fibers=80, curvature_type='arc',
                    curvature_amplitude=4.0, seed=42),
        safe_create('curved_random_2d', num_fibers=80, curvature_type='random_walk',
                    curvature_amplitude=3.0, seed=42),
    ]
    draw_2d_grid(nets, ['Sinusoidal', 'Bézier', 'Arc', 'Random Walk'],
                 d / '06_curved_random_2d.png')

def cat07_curved_amplitude(d):
    nets = [
        safe_create('curved_random_2d', num_fibers=80, curvature_type='sinusoidal',
                    curvature_amplitude=0.5, seed=42),
        safe_create('curved_random_2d', num_fibers=80, curvature_type='sinusoidal',
                    curvature_amplitude=3.0, seed=42),
        safe_create('curved_random_2d', num_fibers=80, curvature_type='sinusoidal',
                    curvature_amplitude=8.0, seed=42),
        safe_create('curved_random_2d', num_fibers=80, curvature_type='sinusoidal',
                    curvature_amplitude=15.0, seed=42),
    ]
    draw_2d_grid(nets, ['Amp=0.5', 'Amp=3', 'Amp=8', 'Amp=15'],
                 d / '07_curved_amplitude.png')

def cat08_voronoi_2d(d):
    nets = [
        safe_create('voronoi_2d', num_seeds=20, seed=42),
        safe_create('voronoi_2d', num_seeds=50, seed=42),
        safe_create('voronoi_2d', num_seeds=120, seed=42),
        safe_create('voronoi_2d', num_seeds=50, regularity=0.8, seed=42),
    ]
    draw_2d_grid(nets, ['Sparse', 'Medium', 'Dense', 'Regular'],
                 d / '08_voronoi_2d.png')

def cat09_fractal(d):
    nets = [
        safe_create('sierpinski'),
        safe_create('koch_curve'),
        safe_create('fractal_tree'),
        safe_create('hilbert'),
    ]
    draw_2d_grid(nets, ['Sierpinski', 'Koch', 'Tree', 'Hilbert'],
                 d / '09_fractal.png')

def cat10_hierarchical(d):
    nets = [
        safe_create('hierarchical_lattice', levels=1, base_topology='triangular', cell_size=40),
        safe_create('hierarchical_lattice', levels=2, base_topology='triangular', cell_size=40),
        safe_create('hierarchical_lattice', levels=2, base_topology='honeycomb', cell_size=40),
        safe_create('hierarchical_lattice', levels=3, base_topology='triangular', cell_size=40),
    ]
    draw_2d_grid(nets, ['Tri L1', 'Tri L2', 'Honey L2', 'Tri L3'],
                 d / '10_hierarchical.png')

def cat11_lattice_3d(d):
    nets = [
        safe_create('lattice_3d', topology='cubic', cell_size=10, grid_size=(3,3,3)),
        safe_create('lattice_3d', topology='octet', cell_size=10, grid_size=(2,2,2)),
        safe_create('lattice_3d', topology='diamond', cell_size=10, grid_size=(2,2,2)),
    ]
    draw_3d_grid(nets, ['Cubic', 'Octet', 'Diamond'],
                 d / '11_lattice_3d.png')

def cat12_entangled_3d(d):
    nets = [
        safe_create('entangled_3d', num_fibers=40, seed=42),
        safe_create('entangled_3d', num_fibers=60, seed=42),
        safe_create('entangled_3d', num_fibers=80, seed=42),
        safe_create('entangled_3d', num_fibers=60, curvature=0.6, seed=42),
        safe_create('entangled_3d', num_fibers=60, curvature=0.9, seed=42),
    ]
    draw_3d_grid(nets, ['N=40', 'N=60', 'N=80', 'curv=0.6', 'curv=0.9'],
                 d / '12_entangled_3d.png')

def cat13_tpms(d):
    nets = [
        safe_create('tpms_sheet', kind='gyroid', resolution=12),
        safe_create('tpms_sheet', kind='primitive', resolution=12),
        safe_create('tpms_lattice', kind='gyroid', resolution=12),
        safe_create('tpms_lattice', kind='diamond', resolution=12),
    ]
    draw_3d_grid(nets, ['Gyroid Sheet', 'Primitive Sheet',
                        'Gyroid Lattice', 'Diamond Lattice'],
                 d / '13_tpms.png')

def cat14_tpms_gradient(d):
    nets = [
        safe_create('tpms_gradient', kind='gyroid', resolution=8),
        safe_create('tpms_gradient', kind='primitive', resolution=8),
    ]
    draw_3d_grid(nets, ['Gyroid Gradient', 'Primitive Gradient'],
                 d / '14_tpms_gradient.png')

def cat15_biomimetic(d):
    nets = [
        safe_create('biomimetic_network', network_type='collagen', num_fibers=60, seed=42),
        safe_create('biomimetic_network', network_type='fibrin', num_fibers=60, seed=42),
        safe_create('electrospun'),
        safe_create('meltblown'),
    ]
    draw_3d_grid(nets, ['Collagen', 'Fibrin', 'Electrospun', 'Meltblown'],
                 d / '15_biomimetic.png')

def cat16_voronoi_3d(d):
    nets = [
        safe_create('voronoi_3d', num_seeds=30, seed=42),
        safe_create('voronoi_3d', num_seeds=80, seed=42),
        safe_create('foam_like_3d'),
    ]
    draw_3d_grid(nets, ['Voronoi Sparse', 'Voronoi Dense', 'Foam'],
                 d / '16_voronoi_3d.png')

def cat17_random_2d(d):
    nets = [
        safe_create('random_2d', num_fibers=50, seed=42),
        safe_create('random_2d', num_fibers=150, seed=42),
        safe_create('random_walk', num_fibers=30, seed=42),
    ]
    draw_2d_grid(nets, ['Sparse Random', 'Dense Random', 'Random Walk'],
                 d / '17_random_2d.png')

def cat18_random_3d(d):
    nets = [
        safe_create('random_3d', num_fibers=50, seed=42),
        safe_create('random_3d', num_fibers=100, seed=42),
    ]
    draw_3d_grid(nets, ['Sparse', 'Dense'],
                 d / '18_random_3d.png')

def cat19_field_guided(d):
    nets = []
    titles = []
    for ft in ['uniform', 'radial', 'vortex']:
        try:
            from fibernet.gen.field_guided import FieldGuidedConfig, OrientationField
            cfg = FieldGuidedConfig(seed=42, fiber_count=80)
            field = OrientationField(canvas_size=100, field_type=ft, seed=42)
            net = fn.create('field_guided', config=cfg, field=field)
            nets.append(net)
            titles.append(ft.title())
        except Exception as e:
            print(f"    WARN field_guided/{ft}: {e}")
            nets.append(None)
            titles.append(ft.title())
    draw_2d_grid(nets, titles, d / '19_field_guided.png')

def cat20_gradient(d):
    nets = [
        safe_create('hierarchical_lattice', levels=2, base_topology='triangular',
                    cell_size=40, scaling_factor=0.2),
        safe_create('hierarchical_lattice', levels=2, base_topology='triangular',
                    cell_size=40, scaling_factor=0.35),
        safe_create('hierarchical_lattice', levels=2, base_topology='triangular',
                    cell_size=40, scaling_factor=0.5),
    ]
    draw_2d_grid(nets, ['scale=0.2', 'scale=0.35', 'scale=0.5'],
                 d / '20_hierarchical_scaling.png')


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

STEPS = [
    ('01_lattice_2d_topo',    cat01_lattice_2d_topologies),
    ('02_lattice_2d_pert',    cat02_lattice_2d_perturbation),
    ('03_lattice_2d_cell',    cat03_lattice_2d_cell_size),
    ('04_meta_2d_modes',      cat04_metamaterial_2d_modes),
    ('05_meta_2d_angle',      cat05_metamaterial_2d_angle),
    ('06_curved_2d_types',    cat06_curved_random_2d),
    ('07_curved_2d_amp',      cat07_curved_amplitude),
    ('08_voronoi_2d',         cat08_voronoi_2d),
    ('09_fractal',            cat09_fractal),
    ('10_hierarchical',       cat10_hierarchical),
    ('11_lattice_3d',         cat11_lattice_3d),
    ('12_entangled_3d',       cat12_entangled_3d),
    ('13_tpms',               cat13_tpms),
    ('14_tpms_gradient',      cat14_tpms_gradient),
    ('15_biomimetic',         cat15_biomimetic),
    ('16_voronoi_3d',         cat16_voronoi_3d),
    ('17_random_2d',          cat17_random_2d),
    ('18_random_3d',          cat18_random_3d),
    ('19_field_guided',       cat19_field_guided),
    ('20_hierarchical_param', cat20_gradient),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.resume and CHECKPOINT.exists():
        CHECKPOINT.unlink()  # fresh start
    ckpt = load_ckpt()
    print(f"Checkpoint: {len(ckpt['done'])} done, {len(ckpt['failed'])} failed")

    print(f"\n{'='*60}")
    print(f"FiberNet Showcase Final — {len(STEPS)} steps")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    for name, func in STEPS:
        run_step(name, lambda f=func: f(OUTPUT_DIR), ckpt)

    print(f"\n{'='*60}")
    print(f"Done: {len(ckpt['done'])}/{len(STEPS)} steps completed")
    if ckpt['failed']:
        print(f"Failed: {len(ckpt['failed'])}")
        for f in ckpt['failed']:
            print(f"  - {f['step']}: {f['error'][:100]}")
    print(f"{'='*60}")

    # List output files
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        sz = f.stat().st_size / 1024
        print(f"  {f.name} ({sz:.0f} KB)")

if __name__ == '__main__':
    main()

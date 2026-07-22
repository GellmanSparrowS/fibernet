"""
FiberNet Showcase Renderer — publication-grade 2D/3D visualization.

Design goals
------------
* Square canvas, no axes/ticks/frames
* Dark theme with subtle glow (2D) or tube rendering (3D)
* Grid layouts for parametric studies
* Consistent style across every output
* pyvista for 3D, matplotlib for 2D
"""

import gc
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    HAS_PV = True
except Exception:
    HAS_PV = False


# ── style tokens ─────────────────────────────────────────────
BG       = '#0d0d0d'
FG_MAIN  = '#00e87b'
FG_ALT   = '#00b4ff'
FG_WARM  = '#ff7744'
TITLE_C  = '#ffffff'
PANEL_BG = '#111111'


# ══════════════════════════════════════════════════════════════
#  2-D  rendering
# ══════════════════════════════════════════════════════════════

def _adaptive_lw(n_fibers: int) -> float:
    """Line-width that scales with density."""
    if n_fibers <= 30:
        return 1.6
    if n_fibers <= 150:
        return 1.0
    if n_fibers <= 500:
        return 0.6
    return 0.35


def _normalize_lines(lines: List[np.ndarray], pad: float = 0.04):
    """Normalize all line segments into [pad, 1-pad] square canvas."""
    if not lines:
        return lines
    all_pts = np.vstack(lines)
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    span = max(mx - mn)
    if span < 1e-12:
        span = 1.0
    out = []
    for pts in lines:
        p = (pts - mn) / span
        # center
        offset = (1.0 - (mx - mn) / span) / 2.0
        p = p + offset[:pts.shape[1]]
        p = p * (1 - 2 * pad) + pad
        out.append(p[:, :2])  # keep 2D
    return out


def render_2d_single(
    network,
    *,
    title: str = '',
    color: str = FG_MAIN,
    line_width: Optional[float] = None,
    glow: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 180,
    figsize: Tuple[float, float] = (6, 6),
):
    """Render one 2D network on a dark square canvas."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    lines = _extract_2d_lines(network)
    if lines:
        lines = _normalize_lines(lines)
        lw = line_width or _adaptive_lw(len(lines))

        if glow:
            # outer glow pass
            lc_glow = LineCollection(
                lines, linewidths=lw * 3.0, colors=color,
                alpha=0.08, antialiased=True,
            )
            ax.add_collection(lc_glow)
            # mid glow
            lc_mid = LineCollection(
                lines, linewidths=lw * 1.8, colors=color,
                alpha=0.18, antialiased=True,
            )
            ax.add_collection(lc_mid)

        # core pass
        lc = LineCollection(
            lines, linewidths=lw, colors=color,
            alpha=0.92, antialiased=True,
        )
        ax.add_collection(lc)

    if title:
        ax.text(0.5, 1.03, title, ha='center', va='bottom',
                color=TITLE_C, fontsize=10, fontweight='bold',
                transform=ax.transAxes, family='sans-serif')

    plt.tight_layout(pad=0.3)
    if save_path:
        fig.savefig(save_path, dpi=dpi, facecolor=BG, bbox_inches='tight',
                    pad_inches=0.05)
    plt.close(fig)
    gc.collect()


def render_2d_grid(
    networks: list,
    titles: List[str],
    *,
    save_path: str,
    color: str = FG_MAIN,
    glow: bool = True,
    dpi: int = 180,
    panel_size: float = 4.0,
):
    """Render a grid of 2D networks."""
    valid = [(n, t) for n, t in zip(networks, titles) if n is not None]
    if not valid:
        return
    nets, titles = zip(*valid)
    n = len(nets)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_size * ncols, panel_size * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes_flat = np.array(axes).flatten()
    fig.patch.set_facecolor(BG)

    for i in range(n):
        ax = axes_flat[i]
        ax.set_facecolor(BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')

        lines = _extract_2d_lines(nets[i])
        if lines:
            lines = _normalize_lines(lines)
            lw = _adaptive_lw(len(lines))
            if glow:
                lc_g = LineCollection(lines, linewidths=lw*3, colors=color,
                                      alpha=0.07, antialiased=True)
                ax.add_collection(lc_g)
                lc_m = LineCollection(lines, linewidths=lw*1.8, colors=color,
                                      alpha=0.15, antialiased=True)
                ax.add_collection(lc_m)
            lc = LineCollection(lines, linewidths=lw, colors=color,
                                alpha=0.92, antialiased=True)
            ax.add_collection(lc)

        if titles[i]:
            ax.text(0.5, 1.03, titles[i], ha='center', va='bottom',
                    color=TITLE_C, fontsize=9, fontweight='bold',
                    transform=ax.transAxes)

    # hide unused panels
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_facecolor(BG)
        axes_flat[j].axis('off')

    plt.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=dpi, facecolor=BG, bbox_inches='tight',
                pad_inches=0.05)
    plt.close(fig)
    gc.collect()


# ══════════════════════════════════════════════════════════════
#  3-D  rendering  (pyvista)
# ══════════════════════════════════════════════════════════════

def render_3d_single(
    network,
    *,
    title: str = '',
    color: str = FG_ALT,
    tube_radius: float = 0.003,
    save_path: Optional[str] = None,
    image_size: Tuple[int, int] = (1024, 1024),
    camera_pos: str = 'iso',
):
    """Render one 3D network with pyvista tube representation."""
    if not HAS_PV:
        # fallback to matplotlib 3D
        _render_3d_mpl(network, title=title, save_path=save_path)
        return

    plotter = pv.Plotter(off_screen=True, window_size=image_size)
    plotter.set_background(BG)

    lines = _extract_3d_lines(network)
    if lines:
        lines = _normalize_3d(lines)
        polydata = _lines_to_polydata(lines)
        if tube_radius > 0 and polydata.n_points > 0:
            tube = polydata.tube(radius=tube_radius, n_sides=6)
            plotter.add_mesh(tube, color=color, smooth_shading=True,
                           specular=0.3, specular_power=20)
        else:
            plotter.add_mesh(polydata, color=color, line_width=2)

    # lighting
    plotter.add_light(pv.Light(position=(5, 5, 5), intensity=0.8, light_type='scene light'))
    plotter.add_light(pv.Light(position=(-3, -3, 5), intensity=0.3, light_type='scene light'))

    _set_camera(plotter, camera_pos)

    if title:
        plotter.add_title(title, font_size=14, color='white',
                          font_family='arial')

    if save_path:
        plotter.screenshot(save_path, return_img=False)
    plotter.close()
    gc.collect()


def render_3d_grid(
    networks: list,
    titles: List[str],
    *,
    save_path: str,
    color: str = FG_ALT,
    tube_radius: float = 0.003,
    image_size: Tuple[int, int] = (1024, 1024),
    panel_size: int = 512,
):
    """Render a grid of 3D networks by compositing individual renders."""
    valid = [(n, t) for n, t in zip(networks, titles) if n is not None]
    if not valid:
        return

    nets, titles_v = zip(*valid)
    n = len(nets)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))

    # Render each panel individually then composite with matplotlib
    panel_images = []
    for i, (net, title) in enumerate(zip(nets, titles_v)):
        img = _render_3d_to_array(net, color=color, tube_radius=tube_radius,
                                  image_size=(panel_size, panel_size))
        panel_images.append((img, title))

    # Composite using matplotlib
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4*ncols, 4*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes_flat = np.array(axes).flatten()
    fig.patch.set_facecolor(BG)

    for i, (img, title) in enumerate(panel_images):
        ax = axes_flat[i]
        ax.imshow(img)
        ax.axis('off')
        ax.set_facecolor(BG)
        if title:
            ax.set_title(title, color=TITLE_C, fontsize=10,
                        fontweight='bold', pad=6)

    for j in range(len(panel_images), len(axes_flat)):
        axes_flat[j].set_facecolor(BG)
        axes_flat[j].axis('off')

    plt.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=180, facecolor=BG, bbox_inches='tight',
                pad_inches=0.05)
    plt.close(fig)
    gc.collect()


# ══════════════════════════════════════════════════════════════
#  helpers
# ══════════════════════════════════════════════════════════════

def _extract_2d_lines(network) -> List[np.ndarray]:
    """Extract 2D line segments from a FiberNetwork."""
    lines = []
    if network is None:
        return lines
    for f in network.fibers:
        pts = f.centerline
        if len(pts) >= 2:
            lines.append(pts[:, :2].copy())
    return lines


def _extract_3d_lines(network) -> List[np.ndarray]:
    """Extract 3D line segments from a FiberNetwork."""
    lines = []
    if network is None:
        return lines
    for f in network.fibers:
        pts = f.centerline
        if len(pts) >= 2:
            if pts.shape[1] == 2:
                pts = np.column_stack([pts, np.zeros(len(pts))])
            lines.append(pts.copy())
    return lines


def _normalize_3d(lines: List[np.ndarray], pad: float = 0.06):
    all_pts = np.vstack(lines)
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    span = max(mx - mn)
    if span < 1e-12:
        span = 1.0
    out = []
    for pts in lines:
        p = (pts - mn) / span
        offset = (1.0 - (mx - mn) / span) / 2.0
        p = p + offset
        p = p * (1 - 2*pad) + pad
        out.append(p)
    return out


def _lines_to_polydata(lines: List[np.ndarray]):
    """Convert list of polylines to a pyvista PolyData."""
    points = []
    line_cells = []
    offset = 0
    for pts in lines:
        n = len(pts)
        points.append(pts)
        cell = [n] + list(range(offset, offset + n))
        line_cells.append(cell)
        offset += n
    if not points:
        return pv.PolyData()
    points = np.vstack(points)
    cells = []
    for c in line_cells:
        cells.extend(c)
    poly = pv.PolyData()
    poly.points = points
    poly.lines = np.array(cells, dtype=np.int64)
    return poly


def _render_3d_to_array(network, *, color=FG_ALT, tube_radius=0.003,
                        image_size=(512, 512)):
    """Render a 3D network and return as numpy array."""
    if not HAS_PV:
        # black fallback
        return np.zeros((*image_size, 3), dtype=np.uint8)

    plotter = pv.Plotter(off_screen=True, window_size=image_size)
    plotter.set_background(BG)

    lines = _extract_3d_lines(network)
    if lines:
        lines = _normalize_3d(lines)
        polydata = _lines_to_polydata(lines)
        if tube_radius > 0 and polydata.n_points > 0:
            tube = polydata.tube(radius=tube_radius, n_sides=6)
            plotter.add_mesh(tube, color=color, smooth_shading=True,
                           specular=0.3, specular_power=20)
        else:
            plotter.add_mesh(polydata, color=color, line_width=2)

    plotter.add_light(pv.Light(position=(5, 5, 5), intensity=0.8, light_type='scene light'))
    plotter.add_light(pv.Light(position=(-3, -3, 5), intensity=0.3, light_type='scene light'))
    _set_camera(plotter, 'iso')

    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def _set_camera(plotter, mode='iso'):
    """Set camera for consistent 3D view."""
    if mode == 'iso':
        plotter.camera_position = 'iso'
    elif mode == 'top':
        plotter.camera_position = 'xy'
    elif mode == 'front':
        plotter.camera_position = 'xz'


def _render_3d_mpl(network, *, title='', save_path=None, color=FG_ALT):
    """Fallback 3D renderer using matplotlib."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(8, 8))
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BG)
    ax.axis('off')

    lines = _extract_3d_lines(network)
    if lines:
        lines = _normalize_3d(lines)
        lc = Line3DCollection(lines, linewidths=0.5, colors=color, alpha=0.7)
        ax.add_collection3d(lc)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

    if title:
        ax.set_title(title, color=TITLE_C, fontsize=10, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=180, facecolor=BG, bbox_inches='tight',
                    pad_inches=0.05)
    plt.close(fig)

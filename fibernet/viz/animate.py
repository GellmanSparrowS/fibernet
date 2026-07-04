"""
Animation utilities for fiber network simulations.

Provides:
- Deformation animations
- Dynamics trajectory animations
- GIF/MP4 export
"""

import numpy as np
from typing import Optional, List
from fibernet.core.network import FiberNetwork


def animate_deformation(
    network: FiberNetwork,
    displacement_steps: List[np.ndarray],
    save_path: str = "deformation.gif",
    fps: int = 10,
    scale_factor: float = 1.0,
):
    """Animate deformation process and save as GIF.
    
    Parameters
    ----------
    displacement_steps : list of np.ndarray
        Displacement vectors at each time step.
    save_path : str
        Output file path (.gif or .mp4).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    lines_data = []
    for fiber in network.fibers:
        lines_data.append(fiber.centerline[:, :2].copy())
    
    lc = LineCollection(lines_data, linewidths=1, colors='steelblue')
    ax.add_collection(lc)
    
    bb_min, bb_max = network.bounding_box()
    margin = 0.1 * (bb_max - bb_min)[:2]
    ax.set_xlim(bb_min[0] - margin[0], bb_max[0] + margin[0])
    ax.set_ylim(bb_min[1] - margin[1], bb_max[1] + margin[1])
    ax.set_aspect('equal')
    
    def update(frame):
        disp = displacement_steps[frame]
        new_lines = []
        offset = 0
        for fiber in network.fibers:
            n_pts = len(fiber.centerline)
            pts = fiber.centerline[:, :2].copy()
            for p_idx in range(n_pts):
                node_idx = offset + p_idx
                if node_idx * 6 + 1 < len(disp):
                    pts[p_idx, 0] += scale_factor * disp[node_idx * 6]
                    pts[p_idx, 1] += scale_factor * disp[node_idx * 6 + 1]
            new_lines.append(pts)
            offset += n_pts
        
        lc.set_segments(new_lines)
        ax.set_title(f"Step {frame + 1}/{len(displacement_steps)}")
        return [lc]
    
    anim = FuncAnimation(fig, update, frames=len(displacement_steps), interval=1000 // fps, blit=True)
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Animation saved to {save_path}")

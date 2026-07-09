"""
Field-guided fiber network generator.

Generates fiber networks guided by orientation fields, enabling:
- Multi-scale orientation analysis
- Field-guided network synthesis
- Biomimetic fiber alignment patterns

Based on the original implementation in Angle_deg.ipynb, adapted for FiberNet.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.core.material import Material


@dataclass
class FieldGuidedConfig:
    """Configuration for field-guided network generation."""
    fiber_count: int = 3600
    crosslink_degree: int = 3
    fiber_mode: str = "curved"  # "straight" or "curved"
    
    # Fiber length distribution
    fiber_length_mean: float = 135.0
    fiber_length_std: float = 35.0
    fiber_length_min: float = 40.0
    fiber_length_max: float = 260.0
    
    # Field guidance
    step_size: float = 1.2
    field_strength: float = 0.70
    angle_noise: float = 0.05
    start_valid_fraction: float = 0.70
    
    # Crosslinking
    crosslink_radius: float = 3.0
    
    # Geometry
    canvas_size: int = 1024
    fiber_radius: float = 0.05
    
    seed: int = 42


class OrientationField:
    """
    2D orientation field for guiding fiber placement.
    
    The field defines a preferred direction at each point in space,
    allowing generation of networks with controlled local alignment.
    
    Parameters
    ----------
    canvas_size : int
        Size of the square canvas.
    field_type : str
        Type of field: "uniform", "radial", "vortex", "gradient", "random_smooth"
    field_angle : float
        Base angle for uniform field (radians).
    gradient_strength : float
        Strength of gradient for gradient field.
    smoothing_sigma : float
        Gaussian smoothing sigma for random field.
    """
    
    def __init__(self, canvas_size: int = 512, field_type: str = "uniform",
                 field_angle: float = 0.0, gradient_strength: float = 0.5,
                 smoothing_sigma: float = 20.0, seed: int = 42):
        self.canvas_size = canvas_size
        self.field_type = field_type
        self.field_angle = field_angle
        self.gradient_strength = gradient_strength
        self.smoothing_sigma = smoothing_sigma
        self.rng = np.random.RandomState(seed)
        
        # Generate field
        self.field = self._generate_field()
    
    def _generate_field(self) -> np.ndarray:
        """Generate orientation field as angle map (H, W)."""
        size = self.canvas_size
        
        if self.field_type == "uniform":
            return np.full((size, size), self.field_angle, dtype=np.float32)
        
        elif self.field_type == "radial":
            y, x = np.mgrid[0:size, 0:size]
            cx, cy = size / 2, size / 2
            angles = np.arctan2(y - cy, x - cx)
            return angles.astype(np.float32)
        
        elif self.field_type == "vortex":
            y, x = np.mgrid[0:size, 0:size]
            cx, cy = size / 2, size / 2
            angles = np.arctan2(y - cy, x - cx) + np.pi / 2
            return angles.astype(np.float32)
        
        elif self.field_type == "gradient":
            y, x = np.mgrid[0:size, 0:size]
            angles = self.field_angle + self.gradient_strength * (x / size - 0.5) * np.pi
            return angles.astype(np.float32)
        
        elif self.field_type == "random_smooth":
            # Generate random angles and smooth
            raw = self.rng.uniform(-np.pi, np.pi, (size // 8 + 1, size // 8 + 1))
            from scipy.ndimage import zoom, gaussian_filter
            field = zoom(raw, (size / raw.shape[0], size / raw.shape[1]))
            field = gaussian_filter(field, sigma=self.smoothing_sigma / 8)
            return field[:size, :size].astype(np.float32)
        
        else:
            raise ValueError(f"Unknown field type: {self.field_type}")
    
    def get_angle(self, x: float, y: float) -> float:
        """Get field angle at position (x, y)."""
        ix = int(np.clip(x, 0, self.canvas_size - 1))
        iy = int(np.clip(y, 0, self.canvas_size - 1))
        return float(self.field[iy, ix])
    
    def visualize(self, ax=None, stride: int = 20, length: float = 15.0):
        """Visualize the orientation field."""
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        size = self.canvas_size
        y, x = np.ogrid[0:size:stride, 0:size:stride]
        angles = self.field[::stride, ::stride]
        
        dx = length * np.cos(angles)
        dy = length * np.sin(angles)
        
        ax.quiver(x, y, dx, dy, angles, cmap='hsv', scale=None, 
                  width=0.003, headwidth=1, headlength=0)
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect('equal')
        ax.set_title(f'Orientation Field: {self.field_type}')
        
        return ax


def field_guided_network(
    config: Optional[FieldGuidedConfig] = None,
    field: Optional[OrientationField] = None,
    material: Optional[Material] = None,
    box_size: Tuple[float, ...] = (100, 100),
) -> FiberNetwork:
    """
    Generate a fiber network guided by an orientation field.
    
    Parameters
    ----------
    config : FieldGuidedConfig, optional
        Generation configuration.
    field : OrientationField, optional
        Orientation field. Creates uniform field if None.
    material : Material, optional
        Fiber material.
    box_size : tuple
        Physical size of the network.
    
    Returns
    -------
    FiberNetwork
        Generated network with field-guided fiber placement.
    """
    if config is None:
        config = FieldGuidedConfig()
    if field is None:
        field = OrientationField(
            canvas_size=config.canvas_size,
            field_type="uniform",
            field_angle=0.0,
            seed=config.seed,
        )
    if material is None:
        material = Material(youngs_modulus=1e9, poissons_ratio=0.3)
    
    rng = np.random.RandomState(config.seed)
    
    # Scale factor: physical size / canvas size
    scale_x = box_size[0] / config.canvas_size
    scale_y = box_size[1] / config.canvas_size if len(box_size) > 1 else scale_x
    
    fibers = []
    
    # Valid starting region
    margin = config.canvas_size * (1 - config.start_valid_fraction) / 2
    
    for _ in range(config.fiber_count):
        # Random starting point in valid region
        x0 = rng.uniform(margin, config.canvas_size - margin)
        y0 = rng.uniform(margin, config.canvas_size - margin)
        
        # Sample fiber length
        length = rng.normal(config.fiber_length_mean, config.fiber_length_std)
        length = np.clip(length, config.fiber_length_min, config.fiber_length_max)
        
        # Grow fiber step by step, following the field
        points = [(x0, y0)]
        x, y = x0, y0
        total_length = 0.0
        
        while total_length < length:
            # Get field direction
            field_angle = field.get_angle(x, y)
            
            # Add noise
            angle = field_angle + rng.normal(0, config.angle_noise)
            
            # Blend with field
            if len(points) > 1:
                prev_angle = np.arctan2(points[-1][1] - points[-2][1],
                                         points[-1][0] - points[-2][0])
                angle = (1 - config.field_strength) * prev_angle + config.field_strength * angle
            
            # Step
            dx = config.step_size * np.cos(angle)
            dy = config.step_size * np.sin(angle)
            x += dx
            y += dy
            total_length += config.step_size
            
            # Check bounds
            if x < 0 or x >= config.canvas_size or y < 0 or y >= config.canvas_size:
                break
            
            points.append((x, y))
        
        if len(points) >= 2:
            # Convert to physical coordinates
            centerline = np.array([[p[0] * scale_x, p[1] * scale_y, 0.0] for p in points])
            fiber = Fiber(centerline=centerline, radius=config.fiber_radius, material=material)
            fibers.append(fiber)
    
    # Create network
    box_size_arr = np.array(list(box_size) + [1.0]) if len(box_size) == 2 else np.array(list(box_size))
    network = FiberNetwork(fibers=fibers, box_size=box_size_arr)
    
    # Add metadata
    network.metadata.update({
        'generator': 'field_guided',
        'field_type': field.field_type,
        'fiber_count': len(fibers),
        'field_strength': config.field_strength,
        'canvas_size': config.canvas_size,
    })
    
    # Auto-crosslink at fiber intersections
    network.auto_crosslink()
    # Auto-crosslink at fiber intersections
    network.auto_crosslink(threshold=2.0)
    
    # Bridge if still disconnected
    from collections import defaultdict
    adj = defaultdict(set)
    for cl in network.crosslinks:
        adj[cl.fiber_i].add(cl.fiber_j)
        adj[cl.fiber_j].add(cl.fiber_i)
    visited = set()
    n_comp = 0
    for s in range(network.num_fibers):
        if s not in visited:
            n_comp += 1
            q = [s]
            while q:
                n = q.pop(0)
                if n in visited: continue
                visited.add(n); q.extend(adj[n] - visited)
    if n_comp > 1:
        network.connect_components(max_gap=5.0)
    
    return network


def multi_scale_orientation_analysis(
    network: FiberNetwork,
    scales: Tuple[int, ...] = (9, 17, 33),
    sample_stride: int = 4,
) -> Dict[str, np.ndarray]:
    """
    Multi-scale orientation analysis of fiber network.
    
    Computes local orientation at multiple scales using structure tensor.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    scales : tuple of int
        Gaussian kernel sizes for multi-scale analysis.
    sample_stride : int
        Stride for sampling points.
    
    Returns
    -------
    dict with keys:
        'orientations': list of orientation angles
        'coherence': list of coherence values
        'dominant_angle': dominant orientation angle
        'nematic_order': nematic order parameter S
    """
    # Collect all fiber segments
    segments = []
    for fiber in network.fibers:
        pts = fiber.centerline
        for i in range(len(pts) - 1):
            dx = pts[i+1, 0] - pts[i, 0]
            dy = pts[i+1, 1] - pts[i, 1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 1e-10:
                angle = np.arctan2(dy, dx)
                segments.append({
                    'x': (pts[i, 0] + pts[i+1, 0]) / 2,
                    'y': (pts[i, 1] + pts[i+1, 1]) / 2,
                    'angle': angle,
                    'length': length,
                })
    
    if not segments:
        return {
            'orientations': np.array([]),
            'coherence': np.array([]),
            'dominant_angle': 0.0,
            'nematic_order': 0.0,
        }
    
    # Compute orientation statistics
    angles = np.array([s['angle'] for s in segments])
    lengths = np.array([s['length'] for s in segments])
    
    # Weighted orientation tensor
    cos2 = np.cos(2 * angles)
    sin2 = np.sin(2 * angles)
    
    S_xx = np.sum(lengths * cos2)
    S_xy = np.sum(lengths * sin2)
    
    # Nematic order parameter
    S = np.sqrt(S_xx**2 + S_xy**2) / np.sum(lengths) if np.sum(lengths) > 0 else 0
    
    # Dominant angle
    dominant = 0.5 * np.arctan2(S_xy, S_xx)
    
    # Coherence (how aligned)
    coherence = S
    
    return {
        'orientations': angles,
        'coherence': np.full_like(angles, coherence),
        'dominant_angle': dominant,
        'nematic_order': S,
        'mean_length': np.mean(lengths),
        'total_length': np.sum(lengths),
        'num_segments': len(segments),
    }

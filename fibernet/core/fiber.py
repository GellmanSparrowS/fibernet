"""
Fiber class - represents a single fiber with 3D geometry, cross-section, and material.

A fiber is defined by a centerline (3D curve), a cross-section profile, and a material.
Supports straight, curved (Bezier/spline), helical, and arbitrary parametric paths.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
from enum import Enum

from fibernet.core.material import Material


class CrossSection(Enum):
    """Supported cross-section shapes."""
    CIRCULAR = "circular"
    RECTANGULAR = "rectangular"
    ELLIPTICAL = "elliptical"
    HOLLOW_CIRCULAR = "hollow_circular"
    TRIANGULAR = "triangular"
    CUSTOM = "custom"


@dataclass
class Fiber:
    """Represents a single fiber in a fiber network.

    Parameters
    ----------
    centerline : np.ndarray
        Nx3 array of centerline points defining the fiber path.
    radius : float
        Cross-section radius (for circular) or characteristic dimension.
    material : Material
        Material properties of the fiber.
    cross_section : CrossSection
        Cross-section shape.
    cross_section_params : dict
        Additional cross-section parameters (e.g., width/height for rectangular).
    fiber_id : int
        Unique identifier.
    segments : int
        Number of discretization segments (for simulation).
    """
    centerline: np.ndarray
    radius: float = 1.0
    material: Material = field(default_factory=Material)
    cross_section: CrossSection = CrossSection.CIRCULAR
    cross_section_params: Dict[str, Any] = field(default_factory=dict)
    fiber_id: int = 0
    segments: int = 20

    def __post_init__(self):
        self.centerline = np.asarray(self.centerline, dtype=np.float64)
        if self.centerline.ndim == 1:
            if len(self.centerline) == 6:
                start, end = self.centerline[:3], self.centerline[3:]
                self.centerline = np.linspace(start, end, self.segments + 1)
            else:
                raise ValueError("centerline must be Nx3 array or 6-element vector [x1,y1,z1,x2,y2,z2]")
        if self.centerline.shape[1] != 3:
            raise ValueError("centerline must have shape (N, 3)")

    @property
    def start_point(self) -> np.ndarray:
        return self.centerline[0]

    @property
    def end_point(self) -> np.ndarray:
        return self.centerline[-1]

    @property
    def num_points(self) -> int:
        return len(self.centerline)

    @property
    def direction(self) -> np.ndarray:
        """Overall direction vector (normalized)."""
        d = self.end_point - self.start_point
        norm = np.linalg.norm(d)
        return d / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])

    @property
    def length(self) -> float:
        """Total arc length of the fiber."""
        diffs = np.diff(self.centerline, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    @property
    def cross_section_area(self) -> float:
        """Cross-sectional area."""
        if self.cross_section == CrossSection.CIRCULAR:
            return np.pi * self.radius**2
        elif self.cross_section == CrossSection.RECTANGULAR:
            w = self.cross_section_params.get("width", 2 * self.radius)
            h = self.cross_section_params.get("height", 2 * self.radius)
            return w * h
        elif self.cross_section == CrossSection.ELLIPTICAL:
            b = self.cross_section_params.get("semi_minor", self.radius * 0.5)
            return np.pi * self.radius * b
        elif self.cross_section == CrossSection.HOLLOW_CIRCULAR:
            r_inner = self.cross_section_params.get("inner_radius", self.radius * 0.7)
            return np.pi * (self.radius**2 - r_inner**2)
        else:
            return np.pi * self.radius**2

    @property
    def second_moment_area(self) -> Tuple[float, float]:
        """Second moment of area (I_y, I_z) for circular/rectangular sections."""
        if self.cross_section == CrossSection.CIRCULAR:
            I_val = np.pi * self.radius**4 / 4.0
            return I_val, I_val
        elif self.cross_section == CrossSection.RECTANGULAR:
            w = self.cross_section_params.get("width", 2 * self.radius)
            h = self.cross_section_params.get("height", 2 * self.radius)
            return w * h**3 / 12.0, h * w**3 / 12.0
        else:
            I_val = np.pi * self.radius**4 / 4.0
            return I_val, I_val

    @property
    def polar_moment(self) -> float:
        """Polar moment of area J."""
        Iy, Iz = self.second_moment_area
        return Iy + Iz

    def resample(self, num_points: int) -> "Fiber":
        """Resample the centerline to have exactly num_points points."""
        old_s = np.linspace(0, 1, len(self.centerline))
        new_s = np.linspace(0, 1, num_points)
        cs_x = CubicSpline(old_s, self.centerline[:, 0])
        cs_y = CubicSpline(old_s, self.centerline[:, 1])
        cs_z = CubicSpline(old_s, self.centerline[:, 2])
        new_centerline = np.column_stack([cs_x(new_s), cs_y(new_s), cs_z(new_s)])
        new_fiber = Fiber(
            centerline=new_centerline,
            radius=self.radius,
            material=self.material,
            cross_section=self.cross_section,
            cross_section_params=self.cross_section_params,
            fiber_id=self.fiber_id,
            segments=num_points - 1,
        )
        return new_fiber

    def local_tangents(self) -> np.ndarray:
        """Compute tangent vectors at each centerline point."""
        tangents = np.gradient(self.centerline, axis=0)
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return tangents / norms

    def local_normals(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute normal and binormal vectors along the fiber."""
        tangents = self.local_tangents()
        n = len(tangents)
        normals = np.zeros_like(tangents)
        binormals = np.zeros_like(tangents)
        
        t0 = tangents[0]
        if abs(t0[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        
        n0 = np.cross(t0, ref)
        n0 /= np.linalg.norm(n0)
        normals[0] = n0
        binormals[0] = np.cross(t0, n0)
        
        for i in range(1, n):
            n_i = np.cross(tangents[i], binormals[i - 1])
            norm = np.linalg.norm(n_i)
            if norm < 1e-12:
                normals[i] = normals[i - 1]
            else:
                normals[i] = n_i / norm
            binormals[i] = np.cross(tangents[i], normals[i])
        
        return normals, binormals

    def curvature(self) -> np.ndarray:
        """Compute curvature at each point along the centerline."""
        tangents = self.local_tangents()
        dt = np.diff(tangents, axis=0)
        
        kappa = np.zeros(len(self.centerline))
        if len(dt) > 0:
            kappa[1:] = np.linalg.norm(dt, axis=1)
            ds = np.linalg.norm(np.diff(self.centerline, axis=0), axis=1)
            ds_safe = np.maximum(ds, 1e-12)
            kappa[1:] /= ds_safe
        kappa[0] = kappa[1] if len(kappa) > 1 else 0.0
        return kappa

    def tortuosity(self) -> float:
        """Ratio of arc length to end-to-end distance."""
        end_dist = np.linalg.norm(self.end_point - self.start_point)
        if end_dist < 1e-12:
            return float('inf')
        return self.length / end_dist

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min_corner, max_corner) of the bounding box."""
        return self.centerline.min(axis=0), self.centerline.max(axis=0)

    def translate(self, offset: np.ndarray) -> "Fiber":
        """Translate fiber by offset vector. Returns new Fiber."""
        new_cl = self.centerline + np.asarray(offset)
        return Fiber(
            centerline=new_cl, radius=self.radius, material=self.material,
            cross_section=self.cross_section,
            cross_section_params=self.cross_section_params,
            fiber_id=self.fiber_id, segments=self.segments,
        )

    def rotate(self, rotation_matrix: np.ndarray, origin: Optional[np.ndarray] = None) -> "Fiber":
        """Rotate fiber around origin. Returns new Fiber."""
        R = np.asarray(rotation_matrix)
        o = np.asarray(origin) if origin is not None else np.zeros(3)
        centered = self.centerline - o
        new_cl = (R @ centered.T).T + o
        return Fiber(
            centerline=new_cl, radius=self.radius, material=self.material,
            cross_section=self.cross_section,
            cross_section_params=self.cross_section_params,
            fiber_id=self.fiber_id, segments=self.segments,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize fiber to dictionary."""
        return {
            "fiber_id": self.fiber_id,
            "centerline": self.centerline.tolist(),
            "radius": self.radius,
            "material": self.material.to_dict(),
            "cross_section": self.cross_section.value,
            "cross_section_params": self.cross_section_params,
            "segments": self.segments,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fiber":
        """Create Fiber from dictionary."""
        return cls(
            fiber_id=data["fiber_id"],
            centerline=np.array(data["centerline"]),
            radius=data["radius"],
            material=Material.from_dict(data["material"]),
            cross_section=CrossSection(data["cross_section"]),
            cross_section_params=data.get("cross_section_params", {}),
            segments=data.get("segments", 20),
        )

    def __repr__(self) -> str:
        return (
            f"Fiber(id={self.fiber_id}, L={self.length:.3f}, r={self.radius:.3f}, "
            f"pts={self.num_points}, mat='{self.material.name}')"
        )

    @classmethod
    def straight(cls, start: np.ndarray, end: np.ndarray, radius: float = 1.0,
                 material: Optional[Material] = None, segments: int = 20,
                 fiber_id: int = 0) -> "Fiber":
        """Create a straight fiber between two points."""
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        centerline = np.linspace(start, end, segments + 1)
        return cls(
            centerline=centerline, radius=radius,
            material=material or Material(),
            segments=segments, fiber_id=fiber_id,
        )

    @classmethod
    def helical(cls, axis_direction: np.ndarray, center: np.ndarray,
                helix_radius: float, pitch: float, num_turns: float,
                fiber_radius: float = 0.1, material: Optional[Material] = None,
                segments_per_turn: int = 50, fiber_id: int = 0) -> "Fiber":
        """Create a helical (chiral) fiber.
        
        Parameters
        ----------
        axis_direction : array-like
            Direction of the helix axis.
        center : array-like
            Center point at the start of the helix.
        helix_radius : float
            Radius of the helix from axis.
        pitch : float
            Distance along axis per full turn.
        num_turns : float
            Number of turns.
        fiber_radius : float
            Cross-section radius of the fiber.
        """
        axis = np.asarray(axis_direction, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        center = np.asarray(center, dtype=np.float64)
        
        if abs(axis[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(axis, ref)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        
        n_pts = int(num_turns * segments_per_turn) + 1
        t = np.linspace(0, num_turns * 2 * np.pi, n_pts)
        
        points = np.outer(t / (2 * np.pi) * pitch, axis)
        points += helix_radius * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
        points += center
        
        return cls(
            centerline=points, radius=fiber_radius,
            material=material or Material(),
            segments=n_pts - 1, fiber_id=fiber_id,
        )

    @classmethod
    def bezier(cls, control_points: np.ndarray, radius: float = 1.0,
               material: Optional[Material] = None, num_points: int = 50,
               fiber_id: int = 0) -> "Fiber":
        """Create a fiber following a Bezier curve.
        
        Parameters
        ----------
        control_points : np.ndarray
            Mx3 array of control points.
        num_points : int
            Number of points in the discretized curve.
        """
        from scipy.special import comb
        cp = np.asarray(control_points, dtype=np.float64)
        n = len(cp) - 1
        t = np.linspace(0, 1, num_points)
        
        points = np.zeros((num_points, 3))
        for i in range(n + 1):
            b = comb(n, i) * (t**i) * ((1 - t)**(n - i))
            points += np.outer(b, cp[i])
        
        return cls(
            centerline=points, radius=radius,
            material=material or Material(),
            segments=num_points - 1, fiber_id=fiber_id,
        )

    @classmethod
    def sine_wave(cls, start: np.ndarray, end: np.ndarray, amplitude: float,
                  num_waves: float, radius: float = 1.0,
                  material: Optional[Material] = None, num_points: int = 100,
                  fiber_id: int = 0) -> "Fiber":
        """Create a sinusoidal wavy fiber between two points."""
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        direction = end - start
        length = np.linalg.norm(direction)
        axis = direction / length
        
        if abs(axis[0]) < 0.9:
            perp = np.cross(axis, np.array([1.0, 0.0, 0.0]))
        else:
            perp = np.cross(axis, np.array([0.0, 1.0, 0.0]))
        perp /= np.linalg.norm(perp)
        
        t = np.linspace(0, 1, num_points)
        points = np.outer(1 - t, start) + np.outer(t, end)
        points += amplitude * np.sin(2 * np.pi * num_waves * t)[:, None] * perp[None, :]
        
        return cls(
            centerline=points, radius=radius,
            material=material or Material(),
            segments=num_points - 1, fiber_id=fiber_id,
        )

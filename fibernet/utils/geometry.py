"""
Geometric utility functions for fiber networks.

Provides:
- Distance computations
- Intersection detection
- Coordinate transforms
- Rotation matrices
"""

import numpy as np
from typing import Tuple, Optional


def rotation_matrix_x(angle: float) -> np.ndarray:
    """3x3 rotation matrix around x-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_y(angle: float) -> np.ndarray:
    """3x3 rotation matrix around y-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_matrix_z(angle: float) -> np.ndarray:
    """3x3 rotation matrix around z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_matrix_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """3x3 rotation matrix from axis-angle representation (Rodrigues)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def segment_distance(p1: np.ndarray, p2: np.ndarray,
                     p3: np.ndarray, p4: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Minimum distance between two line segments (p1-p2 and p3-p4).
    
    Returns (distance, closest_point_on_seg1, closest_point_on_seg2).
    """
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(r), p1, p3
    
    if a < 1e-12:
        s = 0.0
        t = np.clip(f / e, 0, 1)
    else:
        c = np.dot(d1, r)
        if e < 1e-12:
            t = 0.0
            s = np.clip(-c / a, 0, 1)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            
            if abs(denom) > 1e-12:
                s = np.clip((b * f - c * e) / denom, 0, 1)
            else:
                s = 0.0
            
            t = (b * s + f) / e
            
            if t < 0:
                t = 0
                s = np.clip(-c / a, 0, 1)
            elif t > 1:
                t = 1
                s = np.clip((b - c) / a, 0, 1)
    
    cp1 = p1 + s * d1
    cp2 = p3 + t * d2
    dist = np.linalg.norm(cp1 - cp2)
    
    return dist, cp1, cp2


def point_to_segment_distance(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
    """Distance from a point to a line segment."""
    d = seg_end - seg_start
    L2 = np.dot(d, d)
    if L2 < 1e-12:
        return np.linalg.norm(point - seg_start)
    t = np.clip(np.dot(point - seg_start, d) / L2, 0, 1)
    projection = seg_start + t * d
    return np.linalg.norm(point - projection)

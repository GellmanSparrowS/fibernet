"""
Input Validation Utilities

Provides validation functions for fiber network parameters.
"""

import numpy as np
from typing import Union, Tuple, Optional


def validate_positive(value: float, name: str = "value") -> float:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return float(value)


def validate_non_negative(value: float, name: str = "value") -> float:
    """Validate that a value is non-negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return float(value)


def validate_range(
    value: float,
    low: float,
    high: float,
    name: str = "value",
) -> float:
    """Validate that a value is within a range."""
    if value < low or value > high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
    return float(value)


def validate_integer(value: int, name: str = "value") -> int:
    """Validate that a value is a positive integer."""
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def validate_box_size(
    box_size: Union[Tuple[float, float], Tuple[float, float, float]],
    dimension: int = 2,
) -> np.ndarray:
    """Validate and normalize box size."""
    box_size = np.asarray(box_size, dtype=float)
    
    if dimension == 2:
        if len(box_size) == 2:
            box_size = np.append(box_size, 1.0)
        elif len(box_size) == 3:
            pass
        else:
            raise ValueError(f"2D box_size must have 2 or 3 elements, got {len(box_size)}")
    elif dimension == 3:
        if len(box_size) != 3:
            raise ValueError(f"3D box_size must have 3 elements, got {len(box_size)}")
    else:
        raise ValueError(f"dimension must be 2 or 3, got {dimension}")
    
    if np.any(box_size <= 0):
        raise ValueError(f"box_size must be positive, got {box_size}")
    
    return box_size


def validate_probability(value: float, name: str = "probability") -> float:
    """Validate that a value is a probability (0-1)."""
    return validate_range(value, 0.0, 1.0, name)


def validate_angle(value: float, name: str = "angle") -> float:
    """Validate angle in radians."""
    return float(value)


def validate_seed(seed: Optional[int] = None) -> Optional[int]:
    """Validate random seed."""
    if seed is not None:
        seed = int(seed)
    return seed


def validate_material_properties(
    youngs_modulus: float = None,
    density: float = None,
    poisson_ratio: float = None,
):
    """Validate material properties."""
    if youngs_modulus is not None:
        validate_positive(youngs_modulus, "youngs_modulus")
    
    if density is not None:
        validate_positive(density, "density")
    
    if poisson_ratio is not None:
        validate_range(poisson_ratio, -1.0, 0.5, "poisson_ratio")

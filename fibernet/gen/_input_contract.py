"""Scalar admission for ordinary FiberNetwork generators."""
import math
from numbers import Integral, Real


def nonnegative_count(value, name):
    """Accept integral counts, retaining NumPy scalar values and zero."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, not bool")
    return value


def positive_real(value, name):
    """Validate without changing the caller's arithmetic scalar type."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real scalar")
    try:
        valid = math.isfinite(value) and value > 0
    except (OverflowError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return value

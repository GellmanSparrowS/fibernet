"""Shared grid-count admission for native tiling and pattern entry points."""
from itertools import islice
from numbers import Integral

import numpy as np


def normalize_grid(grid, dimension):
    """Return nonnegative integer counts; zero axes produce an empty tiling.

    Python and NumPy integer scalars broadcast to every axis. Iterables must
    contain exactly ``dimension`` integer counts. Booleans are not counts.
    No float truncation or negative-count clamping is performed.
    """
    if isinstance(grid, (bool, np.bool_)):
        raise ValueError("grid counts must be nonnegative integers, not bool")
    if isinstance(grid, Integral):
        counts = (grid,) * dimension
    else:
        if isinstance(grid, (str, bytes, dict)):
            raise ValueError(f"grid must contain exactly {dimension} integer counts")
        try:
            counts = tuple(islice(iter(grid), dimension + 1))
        except TypeError as exc:
            raise ValueError(f"grid must be an integer or {dimension} integer counts") from exc
        if len(counts) != dimension:
            raise ValueError(f"grid must contain exactly {dimension} integer counts")
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 0
           for n in counts):
        raise ValueError("grid counts must be nonnegative integers, not bool")
    return tuple(int(n) for n in counts)

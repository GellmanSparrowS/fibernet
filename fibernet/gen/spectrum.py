"""Dimensionless reference-fiber displacement profiles for network design.

Longitudinal and normal offsets are fractions of each fiber's own length.
The functions retain the FiberScope interpolation and C4 edge order.
"""
import numpy as np


class FiberSpectrum:
    """Validated displacement profile for one oriented reference fiber."""

    def __init__(self, offsets=None):
        if offsets is None:
            values = np.empty((0, 2), dtype=float)
        else:
            values = np.asarray(offsets, dtype=float)
            if values.size == 0:
                values = np.empty((0, 2), dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError('offsets must be finite (n, 2) values')
        self.offsets = values.copy()

    def resample(self, points):
        """Interpolate at uniformly spaced interior positions of one fiber."""
        points = int(points)
        if points < 0:
            raise ValueError('points must be nonnegative')
        if points == 0:
            return []
        count = len(self.offsets)
        if count == 0:
            return [[0.0, 0.0] for _ in range(points)]
        source = (np.arange(count) + 1.0) / (count + 1.0)
        target = (np.arange(points) + 1.0) / (points + 1.0)
        x = np.interp(target, source, self.offsets[:, 0])
        y = np.interp(target, source, self.offsets[:, 1])
        return [[float(a), float(b)] for a, b in zip(x, y)]

    def fit(self, points):
        """Keep an exact-length profile or interpolate to the requested count."""
        points = int(points)
        if points < 0:
            raise ValueError('points must be nonnegative')
        if len(self.offsets) == points:
            return self.offsets.tolist()
        return self.resample(points)

    def rotated(self, length=10.0):
        """Return AB, BC, CD, DA offsets for a counterclockwise square cell."""
        length = float(length)
        if not np.isfinite(length) or length < 0:
            raise ValueError('length must be finite and nonnegative')
        ab = [(float(dx) * length, float(dy) * length)
              for dx, dy in self.offsets]
        bc = [(-dy, dx) for dx, dy in ab]
        cd = [(-dx, -dy) for dx, dy in ab]
        da = [(dy, -dx) for dx, dy in ab]
        return ab + bc + cd + da


def resample_spectrum(spec, points):
    """Resample a dimensionless reference-fiber profile."""
    if int(points) == 0:
        return []
    return FiberSpectrum(spec).resample(points)


def fit_spectrum(spec, points):
    """Return exactly ``points`` offsets, padding an empty profile with zero."""
    return FiberSpectrum(spec).fit(points)


def rotated_displacements(spectrum, length=10.0):
    """Replicate a square-cell profile with fourfold rotational symmetry."""
    return FiberSpectrum(spectrum).rotated(length)

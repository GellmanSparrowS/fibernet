# SPDX-License-Identifier: MIT
"""The example's cross-section integral also covers nonconstant P2 fields."""
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np


def test_integrates_signed_p2_flux_on_both_sides_of_section():
    root = Path(__file__).resolve().parents[1]
    section = runpy.run_path(str(root / "examples" / "feeder_repartition_stokes.py"))["_section_flux"]
    reference = np.array([[0., 1., 0.], [0., 2., 0.], [0., 1., 1.], [1., 1., 0.]])
    edges = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])
    points = np.r_[reference, reference[edges].mean(1)]
    both = np.r_[points - [0., 3., 0.], points]
    x, y, z = both.T
    u = np.c_[x*x + y*y + z, np.zeros_like(x), np.zeros_like(x)]
    field = SimpleNamespace(tetrahedra10=np.arange(20).reshape(2, 10), points_m=both, velocity_m_s=u)
    coordinate = .25
    a = 1 - coordinate
    y0 = np.array([-2., 1.])
    expected = a*a/2 * (coordinate**2 + y0*y0 + 2*y0*a/3 + a*a/6 + a/3)
    np.testing.assert_allclose(section(field, coordinate), expected, rtol=1e-14)

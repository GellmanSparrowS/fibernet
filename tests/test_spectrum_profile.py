"""Public reference-fiber displacement semantics."""
import numpy as np
import pytest

from fibernet.gen.spectrum import (
    FiberSpectrum, fit_spectrum, resample_spectrum, rotated_displacements,
)


def test_interpolation_and_fourfold_rotation():
    profile = FiberSpectrum([(0.1, 0.2), (-0.1, 0.3)])
    np.testing.assert_allclose(profile.resample(3),
                               [[0.1, 0.2], [0.0, 0.25], [-0.1, 0.3]],
                               atol=1e-14)
    assert profile.fit(2) == [[0.1, 0.2], [-0.1, 0.3]]
    assert rotated_displacements([(0.1, 0.2)], length=10.0) == [
        (1.0, 2.0), (-2.0, 1.0), (-1.0, -2.0), (2.0, -1.0)]
    assert fit_spectrum([], 3) == [[0.0, 0.0]] * 3
    assert resample_spectrum([(0.1, 0.2)], 0) == []




@pytest.mark.parametrize('offsets', [[[float('nan'), 0]], [[1, 2, 3]], [1, 2]])
def test_invalid_profile_rejected(offsets):
    with pytest.raises(ValueError):
        FiberSpectrum(offsets)

"""27 focused hand-built cast-contract cases; no generators or numerical solvers."""
import numpy as np
import pytest


def convert(values):
    from fibernet.ml.features import _float32_features
    return _float32_features(values, (len(values),), 'test features')


@pytest.mark.parametrize('case', ['minimum', 'minimum_above', 'maximum', 'negative_minimum', 'negative_maximum'])
def test_portable_normal_boundaries(case):
    minimum = np.float64(2.0 ** -126)
    maximum = np.float64(np.finfo(np.float32).max)
    values = {'minimum': minimum, 'minimum_above': np.nextafter(minimum, np.float64(np.inf)),
              'maximum': maximum, 'negative_minimum': -minimum, 'negative_maximum': -maximum}
    source = np.array([values[case]], dtype=np.float64)
    result = convert(source)
    assert result.shape == (1,) and result.dtype == np.float32
    np.testing.assert_array_equal(result.view(np.uint32), source.astype(np.float32).view(np.uint32))
    magnitude = result.view(np.uint32) & np.uint32(0x7fffffff)
    assert np.all(magnitude >= np.uint32(0x00800000))
    assert np.all(magnitude <= np.uint32(0x7f7fffff))


def test_both_signed_zeros_are_preserved():
    source = np.array([0, 0x8000000000000000], dtype=np.uint64).view(np.float64)
    np.testing.assert_array_equal(convert(source).view(np.uint32), np.array([0, 0x80000000], dtype=np.uint32))


@pytest.mark.parametrize('case', ['subnormal', 'below_minimum', 'rounds_to_zero', 'negative_subnormal', 'negative_below_minimum', 'negative_rounds_to_zero'])
def test_all_nonzero_below_normal_range_are_rejected(case):
    minimum = np.float64(2.0 ** -126)
    values = {'subnormal': 1e-40, 'below_minimum': np.nextafter(minimum, np.float64(0)), 'rounds_to_zero': 1e-50}
    negative = case.startswith('negative_')
    value = values[case.removeprefix('negative_')]
    source = np.array([-value if negative else value], dtype=np.float64)
    with pytest.raises(ValueError, match='below the portable float32 normal range; rescale'):
        convert(source)


@pytest.mark.parametrize('bits', [0x0000000000000001, 0x8000000000000001])
def test_stored_float64_subnormal_is_not_treated_as_zero(bits):
    # Construct storage directly; do not use floating arithmetic to classify it.
    source = np.array([bits], dtype=np.uint64).view(np.float64)
    with pytest.raises(ValueError, match='below the portable float32 normal range; rescale'):
        convert(source)


@pytest.mark.parametrize('sign', [1, -1])
def test_just_above_float32_maximum_is_rejected(sign):
    maximum = np.float64(np.finfo(np.float32).max)
    source = np.array([sign * np.nextafter(maximum, np.float64(np.inf))], dtype=np.float64)
    with pytest.raises(ValueError, match='exceeds float32 range; rescale'):
        convert(source)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_source_is_rejected(value):
    with pytest.raises(ValueError, match='not finite; rescale'):
        convert(np.array([value], dtype=np.float64))


@pytest.mark.parametrize('bits', [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x7f800000, 0x7fc00000])
def test_corrupted_cast_storage_is_rejected_without_fp_mode_changes(bits, monkeypatch):
    import fibernet.ml.features as module
    forced = np.array([bits], dtype=np.uint32).view(np.float32)
    class ForcedCast(np.ndarray):
        def astype(self, dtype, *args, **kwargs):
            assert dtype is np.float32
            return forced.copy()
    class LocalNumpyProxy:
        def __getattr__(self, name):
            return getattr(np, name)
        def asarray(self, values, dtype):
            return np.asarray(values, dtype=dtype).view(ForcedCast)
    # Only this module's NumPy binding is replaced; no global library/FP setting.
    monkeypatch.setattr(module, 'np', LocalNumpyProxy())
    with pytest.raises(ValueError, match='cast .*; rescale'):
        module._float32_features([1.0], (1,), 'test features')


@pytest.mark.parametrize('reject', [False, True])
def test_error_state_is_restored(reject):
    prior = np.geterr()
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        requested = np.geterr()
        if reject:
            with pytest.raises(ValueError, match='portable float32 normal range'):
                convert(np.array([1e-40], dtype=np.float64))
        else:
            convert(np.array([2.0 ** -126], dtype=np.float64))
        assert np.geterr() == requested
    assert np.geterr() == prior

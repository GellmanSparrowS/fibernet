"""Shared validation errors and public reexports preserve scalar contracts."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def validation():
    # Normal public package import, never source-file loading or package stubs.
    import fibernet.utils as public
    import fibernet.utils.validation as implementation
    for name in public.__all__:
        assert getattr(public, name) is getattr(implementation, name)
    return public


@pytest.mark.parametrize("classinfo, expected", [
    (int, "count must be of type int, got str"),
    ((int, float), "count must be of type int or float, got str"),
    ((int, (float, complex)), "count must be of type int or float or complex, got str"),
])
def test_rejected_type_message(validation, classinfo, expected):
    with pytest.raises(validation.ValidationError) as exc:
        validation.validate_type("1", "count", classinfo)
    assert str(exc.value) == expected


@pytest.mark.parametrize("value", ["1", [], np.int64(1), np.float32(1), np.bool_(True)])
def test_range_rejected_types_remain_rejected(validation, value):
    with pytest.raises(validation.ValidationError, match="count"):
        validation.validate_range(value, "count", 0, 2)


@pytest.mark.parametrize("value", [float("nan"), np.float64("nan")])
@pytest.mark.parametrize("method", ["range", "positive", "nonnegative"])
def test_nan_rejected(validation, value, method):
    with pytest.raises(validation.ValidationError, match="ratio.*NaN"):
        if method == "range":
            validation.validate_range(value, "ratio", 0, 1)
        else:
            validation.validate_positive(value, "ratio", allow_zero=method == "nonnegative")


@pytest.mark.parametrize("value", [0, 1, 0.5, False, True, np.float64(0.5)])
def test_existing_bounded_acceptance_and_none_return(validation, value):
    assert validation.validate_range(value, "ratio", 0, 1) is None


@pytest.mark.parametrize("value", [1, True, 10 ** 10000, float("inf")],
                         ids=["int", "bool", "huge-int", "positive-infinity"])
def test_positive_existing_domain_and_none_return(validation, value):
    assert validation.validate_positive(value, "size") is None


@pytest.mark.parametrize("value", [0, False])
def test_zero_policy_unchanged(validation, value):
    with pytest.raises(validation.ValidationError):
        validation.validate_positive(value, "size")
    assert validation.validate_positive(value, "size", allow_zero=True) is None


def test_one_sided_infinities_and_unbounded_large_int(validation):
    assert validation.validate_range(float("inf"), "x", min_val=0) is None
    assert validation.validate_range(float("-inf"), "x", max_val=0) is None
    assert validation.validate_range(10 ** 10000, "x") is None
    assert validation.validate_range(-(10 ** 10000), "x") is None


@pytest.mark.parametrize("value", [float("-inf"), float("inf"), -1, 2])
def test_bounded_rejection_unchanged(validation, value):
    with pytest.raises(validation.ValidationError):
        validation.validate_range(value, "ratio", 0, 1)


def test_fluent_aggregation_collects_type_and_nan_errors(validation):
    validator = validation.ParameterValidator()
    assert validator.check_range("1", "count", 0, 2) is validator
    assert validator.check_positive("1", "length") is validator
    assert validator.check_type("1", "direct", (int, float)) is validator
    assert validator.check_range(float("nan"), "ratio", 0, 1) is validator
    assert validator.check_positive(float("nan"), "positive") is validator
    assert len(validator.errors) == 5
    with pytest.raises(ValueError) as exc:
        validator.validate()
    message = str(exc.value)
    for name in ("count", "length", "direct", "ratio", "positive"):
        assert name in message


def test_fluent_success_unchanged(validation):
    validator = validation.ParameterValidator()
    assert validator.check_range(True, "count", 0, 1) is validator
    assert validator.check_positive(float("inf"), "size") is validator
    assert validator.validate() is None


def test_unbounded_nan_rejected(validation):
    with pytest.raises(validation.ValidationError, match="value.*NaN"):
        validation.validate_range(float("nan"), "value")


def test_none_allowance_unchanged(validation):
    assert validation.validate_type(None, "value", int, allow_none=True) is None
    with pytest.raises(validation.ValidationError, match="value cannot be None"):
        validation.validate_type(None, "value", int, allow_none=False)

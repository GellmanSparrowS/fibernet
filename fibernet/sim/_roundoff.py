"""Conservative binary64 budgets; not estimates of physical/solver error."""
import math
import struct
import sys


_MIN_SUBNORMAL = struct.unpack('>d', bytes.fromhex('0000000000000001'))[0]


class _SubnormalRoundoffError(ArithmeticError):
    """The current thread cannot preserve required subnormal arithmetic."""


def _require_gradual_underflow():
    # Check the current thread's arithmetic, not an import-time/cached mode.
    # DAZ can compare a stored nonzero subnormal equal to zero; FTZ can lose
    # its sum even when the comparison preserves the input.
    if not (_MIN_SUBNORMAL > 0.0 and _MIN_SUBNORMAL + _MIN_SUBNORMAL > 0.0):
        raise _SubnormalRoundoffError('subnormal roundoff arithmetic is unavailable')


def _magnitude_bits(value):
    bits = struct.unpack('>Q', struct.pack('>d', value))[0]
    magnitude = bits & 0x7fffffffffffffff
    if magnitude >= 0x7ff0000000000000:
        raise ArithmeticError('nonfinite roundoff budget')
    return bits, magnitude


def _require_gradual_for_subnormal(value):
    """Validate stored nonzero subnormals before sign/zero branch decisions."""
    _, magnitude = _magnitude_bits(value)
    if 0 < magnitude < 0x0010000000000000:
        _require_gradual_underflow()


def _nonnegative_values(values):
    result = []
    for value in values:
        value = float(value)
        bits, magnitude = _magnitude_bits(value)
        if bits >> 63 and magnitude:
            raise ArithmeticError('nonfinite roundoff budget')
        if 0 < magnitude < 0x0010000000000000:
            _require_gradual_underflow()
        result.append(value)
    return result


def upper_sum(values):
    """Upper bound a sum of nonnegative finite budget terms."""
    values = _nonnegative_values(values)
    value = math.fsum(values)
    result = math.nextafter(value, math.inf) if value else 0.0
    if not math.isfinite(result):
        raise ArithmeticError('roundoff budget overflow')
    return result


def upper_product(*values):
    """Round every nonnegative budget product upward, including underflow."""
    values = _nonnegative_values(values)
    if any(x == 0 for x in values):
        return 0.0
    result = 1.0
    for value in values:
        product = result * value
        # Positive normal operands can produce a subnormal, or a zero after
        # underflow. Only gradual underflow makes nextafter a valid bound.
        if product < sys.float_info.min:
            _require_gradual_underflow()
        result = math.nextafter(product, math.inf)
        if not math.isfinite(result):
            raise ArithmeticError('roundoff budget overflow')
    return result


def evaluation_ulp(value):
    """Full ulp, allowing fsum double rounding; exact zero operations get zero.

    Callers separately reject underflow-to-zero of nonzero products. Subtraction
    of equal stored floats is exact, so its zero result needs no absolute floor.
    """
    value = float(value)
    _, magnitude = _magnitude_bits(value)
    if not magnitude:
        return 0.0
    result = math.ulp(value)
    if result < sys.float_info.min:
        _require_gradual_underflow()
    return result

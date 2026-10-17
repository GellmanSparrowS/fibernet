"""Conservative binary64 budgets; not estimates of physical/solver error."""
import math


def upper_sum(values):
    """Upper bound a sum of nonnegative finite budget terms."""
    values = list(values)
    if any(x < 0 or not math.isfinite(x) for x in values):
        raise ArithmeticError('nonfinite roundoff budget')
    value = math.fsum(values)
    result = math.nextafter(value, math.inf) if value else 0.0
    if not math.isfinite(result):
        raise ArithmeticError('roundoff budget overflow')
    return result


def upper_product(*values):
    """Round every nonnegative budget product upward, including underflow."""
    if any(x < 0 or not math.isfinite(x) for x in values):
        raise ArithmeticError('nonfinite roundoff budget')
    if any(x == 0 for x in values):
        return 0.0
    result = 1.0
    for value in values:
        result = math.nextafter(result * value, math.inf)
        if not math.isfinite(result):
            raise ArithmeticError('roundoff budget overflow')
    return result


def evaluation_ulp(value):
    """Full ulp, allowing fsum double rounding; exact zero operations get zero.

    Callers separately reject underflow-to-zero of nonzero products. Subtraction
    of equal stored floats is exact, so its zero result needs no absolute floor.
    """
    return math.ulp(float(value)) if value != 0 else 0.0

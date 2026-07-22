"""
Custom exceptions for FiberNet.

Provides clear, actionable error messages for common issues.
"""


class FiberNetError(Exception):
    """Base exception for all FiberNet errors."""
    pass


class NetworkError(FiberNetError):
    """Error related to fiber network operations."""
    pass


class EmptyNetworkError(NetworkError):
    """Raised when an operation requires a non-empty network."""
    def __init__(self, operation: str = "operation"):
        super().__init__(
            f"Cannot perform {operation} on an empty network. "
            "Add fibers first using net.add_fiber() or a generator function."
        )


class DimensionError(NetworkError):
    """Raised when dimension mismatch occurs."""
    def __init__(self, expected: int, actual: int):
        super().__init__(
            f"Expected {expected}D network but got {actual}D. "
            "Use the appropriate generator or check dimension compatibility."
        )


class GeneratorError(FiberNetError):
    """Error in network generation."""
    pass


class InvalidParameterError(FiberNetError):
    """Raised when a parameter has an invalid value."""
    def __init__(self, name: str, value, reason: str = ""):
        msg = f"Invalid value for parameter '{name}': {value}"
        if reason:
            msg += f". {reason}"
        super().__init__(msg)


class SimulationError(FiberNetError):
    """Error during simulation."""
    pass


class SingularMatrixError(SimulationError):
    """Raised when FEM matrix is singular."""
    def __init__(self):
        super().__init__(
            "FEM stiffness matrix is singular. This usually means: "
            "(1) The network has disconnected components, "
            "(2) Insufficient boundary conditions, or "
            "(3) The network has zero-volume elements. "
            "Try increasing fiber density or checking network connectivity."
        )


class ConvergenceError(SimulationError):
    """Raised when nonlinear solver fails to converge."""
    def __init__(self, iterations: int, residual: float):
        super().__init__(
            f"Nonlinear solver did not converge after {iterations} iterations "
            f"(residual: {residual:.2e}). Try reducing the strain step size "
            "or using a different constitutive model."
        )


class IOError(FiberNetError):
    """Error in I/O operations."""
    pass


class FileFormatError(IOError):
    """Raised when file format is invalid or unsupported."""
    def __init__(self, path: str, expected_format: str = ""):
        msg = f"Invalid file format: {path}"
        if expected_format:
            msg += f". Expected {expected_format} format."
        super().__init__(msg)


class AnalysisError(FiberNetError):
    """Error during structural analysis."""
    pass


def check_nonempty(network, operation: str = "operation"):
    """Check that a network has fibers."""
    if network.num_fibers == 0:
        raise EmptyNetworkError(operation)


def check_positive(value, name: str, allow_zero: bool = False):
    """Check that a value is positive."""
    if allow_zero and value < 0:
        raise InvalidParameterError(name, value, "Must be non-negative.")
    elif not allow_zero and value <= 0:
        raise InvalidParameterError(name, value, "Must be positive.")


def check_range(value, low, high, name: str):
    """Check that a value is within a range."""
    if value < low or value > high:
        raise InvalidParameterError(
            name, value, f"Must be in range [{low}, {high}]."
        )


def check_integer(value, name: str, minimum: int = 1):
    """Check that a value is a positive integer."""
    if not isinstance(value, (int, float)) or int(value) != value:
        raise InvalidParameterError(name, value, "Must be an integer.")
    if value < minimum:
        raise InvalidParameterError(name, value, f"Must be >= {minimum}.")

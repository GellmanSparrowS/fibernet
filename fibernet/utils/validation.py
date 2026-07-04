"""
Parameter Validation Utilities

Provides tools for validating function parameters with clear error messages.

Features:
- Type checking
- Range validation
- Value constraints
- Custom validators
- Decorator-based validation

References:
- Python typing: https://docs.python.org/3/library/typing.html
"""

import numpy as np
from typing import Any, Callable, Optional, Tuple, Union, Type
from functools import wraps
import warnings


class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass


def validate_type(
    value: Any,
    name: str,
    expected_type: Type,
    allow_none: bool = False
) -> None:
    """
    Validate that a value has the expected type.
    
    Parameters
    ----------
    value : Any
        Value to validate
    name : str
        Parameter name (for error messages)
    expected_type : type
        Expected type
    allow_none : bool
        If True, allow None values
    
    Raises
    ------
    ValidationError
        If validation fails
    
    Examples
    --------
    >>> validate_type(42, 'count', int)
    >>> validate_type(None, 'optional', str, allow_none=True)
    """
    if value is None:
        if not allow_none:
            raise ValidationError(f"{name} cannot be None")
        return
    
    if not isinstance(value, expected_type):
        raise ValidationError(
            f"{name} must be of type {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )


def validate_range(
    value: Union[int, float],
    name: str,
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True
) -> None:
    """
    Validate that a numeric value is within a range.
    
    Parameters
    ----------
    value : int or float
        Value to validate
    name : str
        Parameter name
    min_val : int or float, optional
        Minimum value
    max_val : int or float, optional
        Maximum value
    min_inclusive : bool
        If True, min_val is inclusive (>=)
    max_inclusive : bool
        If True, max_val is inclusive (<=)
    
    Raises
    ------
    ValidationError
        If validation fails
    
    Examples
    --------
    >>> validate_range(5, 'count', min_val=1, max_val=10)
    >>> validate_range(0.5, 'ratio', min_val=0.0, max_val=1.0)
    """
    validate_type(value, name, (int, float))
    
    if min_val is not None:
        if min_inclusive:
            if value < min_val:
                raise ValidationError(
                    f"{name} must be >= {min_val}, got {value}"
                )
        else:
            if value <= min_val:
                raise ValidationError(
                    f"{name} must be > {min_val}, got {value}"
                )
    
    if max_val is not None:
        if max_inclusive:
            if value > max_val:
                raise ValidationError(
                    f"{name} must be <= {max_val}, got {value}"
                )
        else:
            if value >= max_val:
                raise ValidationError(
                    f"{name} must be < {max_val}, got {value}"
                )


def validate_positive(
    value: Union[int, float],
    name: str,
    allow_zero: bool = False
) -> None:
    """
    Validate that a value is positive.
    
    Parameters
    ----------
    value : int or float
        Value to validate
    name : str
        Parameter name
    allow_zero : bool
        If True, allow zero
    
    Raises
    ------
    ValidationError
        If validation fails
    
    Examples
    --------
    >>> validate_positive(5, 'count')
    >>> validate_positive(0, 'offset', allow_zero=True)
    """
    if allow_zero:
        validate_range(value, name, min_val=0.0)
    else:
        validate_range(value, name, min_val=0.0, min_inclusive=False)


def validate_array(
    value: Any,
    name: str,
    dtype: Optional[Type] = None,
    shape: Optional[Tuple[int, ...]] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> np.ndarray:
    """
    Validate and convert to numpy array.
    
    Parameters
    ----------
    value : Any
        Value to validate (list, tuple, or array)
    name : str
        Parameter name
    dtype : type, optional
        Expected data type
    shape : tuple, optional
        Expected shape
    min_length : int, optional
        Minimum length (for 1D arrays)
    max_length : int, optional
        Maximum length (for 1D arrays)
    
    Returns
    -------
    array : np.ndarray
        Validated numpy array
    
    Raises
    ------
    ValidationError
        If validation fails
    
    Examples
    --------
    >>> arr = validate_array([1, 2, 3], 'points', dtype=float)
    >>> arr = validate_array([[1, 2], [3, 4]], 'matrix', shape=(2, 2))
    """
    try:
        arr = np.asarray(value, dtype=dtype)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"{name} cannot be converted to array: {e}")
    
    if shape is not None:
        if arr.shape != shape:
            raise ValidationError(
                f"{name} must have shape {shape}, got {arr.shape}"
            )
    
    if min_length is not None:
        if len(arr) < min_length:
            raise ValidationError(
                f"{name} must have length >= {min_length}, got {len(arr)}"
            )
    
    if max_length is not None:
        if len(arr) > max_length:
            raise ValidationError(
                f"{name} must have length <= {max_length}, got {len(arr)}"
            )
    
    return arr


def validate_choices(
    value: Any,
    name: str,
    choices: list
) -> None:
    """
    Validate that a value is one of the allowed choices.
    
    Parameters
    ----------
    value : Any
        Value to validate
    name : str
        Parameter name
    choices : list
        List of allowed values
    
    Raises
    ------
    ValidationError
        If validation fails
    
    Examples
    --------
    >>> validate_choices('fast', 'method', ['fast', 'accurate', 'balanced'])
    """
    if value not in choices:
        raise ValidationError(
            f"{name} must be one of {choices}, got {value}"
        )


def validate_condition(
    condition: bool,
    message: str
) -> None:
    """
    Validate that a condition is True.
    
    Parameters
    ----------
    condition : bool
        Condition to check
    message : str
        Error message if condition is False
    
    Raises
    ------
    ValidationError
        If condition is False
    
    Examples
    --------
    >>> validate_condition(x > y, "x must be greater than y")
    """
    if not condition:
        raise ValidationError(message)


def validate_mutually_exclusive(
    params: dict,
    param_names: list,
    require_one: bool = False
) -> None:
    """
    Validate that parameters are mutually exclusive.
    
    Parameters
    ----------
    params : dict
        Dictionary of parameter values
    param_names : list
        List of parameter names to check
    require_one : bool
        If True, exactly one must be provided
    
    Raises
    ------
    ValidationError
        If validation fails
    
    Examples
    --------
    >>> validate_mutually_exclusive(
    ...     {'a': 1, 'b': None},
    ...     ['a', 'b'],
    ...     require_one=True
    ... )
    """
    provided = [name for name in param_names if params.get(name) is not None]
    
    if len(provided) > 1:
        raise ValidationError(
            f"Parameters {param_names} are mutually exclusive, "
            f"but multiple were provided: {provided}"
        )
    
    if require_one and len(provided) == 0:
        raise ValidationError(
            f"Exactly one of {param_names} must be provided"
        )


def with_validation(func: Callable) -> Callable:
    """
    Decorator to wrap function with validation error handling.
    
    Parameters
    ----------
    func : callable
        Function to wrap
    
    Returns
    -------
    wrapper : callable
        Wrapped function
    
    Examples
    --------
    >>> @with_validation
    ... def my_function(x, y):
    ...     validate_positive(x, 'x')
    ...     validate_positive(y, 'y')
    ...     return x + y
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            raise ValueError(f"Validation error in {func.__name__}: {e}")
    
    return wrapper


class ParameterValidator:
    """
    Fluent interface for parameter validation.
    
    Examples
    --------
    >>> validator = ParameterValidator()
    >>> validator.check_type(count, 'count', int) \\
    ...          .check_range(count, 'count', min_val=1) \\
    ...          .check_positive(length, 'length')
    """
    
    def __init__(self):
        self.errors = []
    
    def check_type(
        self,
        value: Any,
        name: str,
        expected_type: Type,
        allow_none: bool = False
    ) -> 'ParameterValidator':
        """Check type."""
        try:
            validate_type(value, name, expected_type, allow_none)
        except ValidationError as e:
            self.errors.append(str(e))
        return self
    
    def check_range(
        self,
        value: Union[int, float],
        name: str,
        min_val: Optional[Union[int, float]] = None,
        max_val: Optional[Union[int, float]] = None
    ) -> 'ParameterValidator':
        """Check range."""
        try:
            validate_range(value, name, min_val, max_val)
        except ValidationError as e:
            self.errors.append(str(e))
        return self
    
    def check_positive(
        self,
        value: Union[int, float],
        name: str,
        allow_zero: bool = False
    ) -> 'ParameterValidator':
        """Check positive."""
        try:
            validate_positive(value, name, allow_zero)
        except ValidationError as e:
            self.errors.append(str(e))
        return self
    
    def check_array(
        self,
        value: Any,
        name: str,
        dtype: Optional[Type] = None,
        shape: Optional[Tuple[int, ...]] = None
    ) -> 'ParameterValidator':
        """Check array."""
        try:
            validate_array(value, name, dtype, shape)
        except ValidationError as e:
            self.errors.append(str(e))
        return self
    
    def check_choices(
        self,
        value: Any,
        name: str,
        choices: list
    ) -> 'ParameterValidator':
        """Check choices."""
        try:
            validate_choices(value, name, choices)
        except ValidationError as e:
            self.errors.append(str(e))
        return self
    
    def check_condition(
        self,
        condition: bool,
        message: str
    ) -> 'ParameterValidator':
        """Check condition."""
        try:
            validate_condition(condition, message)
        except ValidationError as e:
            self.errors.append(str(e))
        return self
    
    def validate(self) -> None:
        """
        Validate all checks and raise errors if any.
        
        Raises
        ------
        ValueError
            If any validation failed
        """
        if self.errors:
            raise ValueError(
                "Validation failed:\n" + "\n".join(f"  - {e}" for e in self.errors)
            )



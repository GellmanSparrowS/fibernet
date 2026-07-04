"""Tests for parameter validation utilities."""

import pytest
import numpy as np
from fibernet.utils.validation import (
    ValidationError,
    validate_type,
    validate_range,
    validate_positive,
    validate_array,
    validate_choices,
    validate_condition,
    validate_mutually_exclusive,
    with_validation,
    ParameterValidator,
)


class TestValidateType:
    """Test type validation."""
    
    def test_valid_type(self):
        """Test valid type."""
        validate_type(42, 'count', int)
        validate_type('hello', 'name', str)
        validate_type([1, 2, 3], 'items', list)
    
    def test_invalid_type(self):
        """Test invalid type."""
        with pytest.raises(ValidationError):
            validate_type('hello', 'count', int)
    
    def test_allow_none(self):
        """Test None handling."""
        validate_type(None, 'optional', str, allow_none=True)
        
        with pytest.raises(ValidationError):
            validate_type(None, 'required', str, allow_none=False)


class TestValidateRange:
    """Test range validation."""
    
    def test_valid_range(self):
        """Test valid range."""
        validate_range(5, 'count', min_val=1, max_val=10)
        validate_range(0.5, 'ratio', min_val=0.0, max_val=1.0)
    
    def test_below_min(self):
        """Test below minimum."""
        with pytest.raises(ValidationError):
            validate_range(-1, 'count', min_val=0)
    
    def test_above_max(self):
        """Test above maximum."""
        with pytest.raises(ValidationError):
            validate_range(15, 'count', max_val=10)
    
    def test_inclusive_exclusive(self):
        """Test inclusive/exclusive bounds."""
        validate_range(5, 'x', min_val=5, min_inclusive=True)
        
        with pytest.raises(ValidationError):
            validate_range(5, 'x', min_val=5, min_inclusive=False)


class TestValidatePositive:
    """Test positive validation."""
    
    def test_positive(self):
        """Test positive values."""
        validate_positive(5, 'count')
        validate_positive(0.5, 'ratio')
    
    def test_zero(self):
        """Test zero handling."""
        validate_positive(0, 'offset', allow_zero=True)
        
        with pytest.raises(ValidationError):
            validate_positive(0, 'count', allow_zero=False)
    
    def test_negative(self):
        """Test negative values."""
        with pytest.raises(ValidationError):
            validate_positive(-5, 'count')


class TestValidateArray:
    """Test array validation."""
    
    def test_valid_array(self):
        """Test valid array."""
        arr = validate_array([1, 2, 3], 'points')
        assert isinstance(arr, np.ndarray)
    
    def test_with_dtype(self):
        """Test array with dtype."""
        arr = validate_array([1, 2, 3], 'values', dtype=float)
        assert arr.dtype == float
    
    def test_with_shape(self):
        """Test array with shape."""
        arr = validate_array([[1, 2], [3, 4]], 'matrix', shape=(2, 2))
        assert arr.shape == (2, 2)
        
        with pytest.raises(ValidationError):
            validate_array([1, 2, 3], 'matrix', shape=(2, 2))
    
    def test_with_length(self):
        """Test array with length constraints."""
        arr = validate_array([1, 2, 3], 'items', min_length=2, max_length=5)
        assert len(arr) == 3
        
        with pytest.raises(ValidationError):
            validate_array([1], 'items', min_length=2)


class TestValidateChoices:
    """Test choices validation."""
    
    def test_valid_choice(self):
        """Test valid choice."""
        validate_choices('fast', 'method', ['fast', 'accurate', 'balanced'])
    
    def test_invalid_choice(self):
        """Test invalid choice."""
        with pytest.raises(ValidationError):
            validate_choices('slow', 'method', ['fast', 'accurate'])


class TestValidateCondition:
    """Test condition validation."""
    
    def test_true_condition(self):
        """Test true condition."""
        validate_condition(True, "Should not fail")
        validate_condition(5 > 3, "5 should be greater than 3")
    
    def test_false_condition(self):
        """Test false condition."""
        with pytest.raises(ValidationError, match="x must be positive"):
            validate_condition(False, "x must be positive")


class TestValidateMutuallyExclusive:
    """Test mutually exclusive validation."""
    
    def test_one_provided(self):
        """Test one parameter provided."""
        validate_mutually_exclusive(
            {'a': 1, 'b': None},
            ['a', 'b']
        )
    
    def test_multiple_provided(self):
        """Test multiple parameters provided."""
        with pytest.raises(ValidationError):
            validate_mutually_exclusive(
                {'a': 1, 'b': 2},
                ['a', 'b']
            )
    
    def test_require_one(self):
        """Test require_one option."""
        validate_mutually_exclusive(
            {'a': 1, 'b': None},
            ['a', 'b'],
            require_one=True
        )
        
        with pytest.raises(ValidationError):
            validate_mutually_exclusive(
                {'a': None, 'b': None},
                ['a', 'b'],
                require_one=True
            )


class TestWithValidation:
    """Test validation decorator."""
    
    def test_decorator_success(self):
        """Test decorator with valid inputs."""
        @with_validation
        def add_positive(x, y):
            validate_positive(x, 'x')
            validate_positive(y, 'y')
            return x + y
        
        result = add_positive(3, 5)
        assert result == 8
    
    def test_decorator_failure(self):
        """Test decorator with invalid inputs."""
        @with_validation
        def add_positive(x, y):
            validate_positive(x, 'x')
            validate_positive(y, 'y')
            return x + y
        
        with pytest.raises(ValueError, match="Validation error"):
            add_positive(-1, 5)


class TestParameterValidator:
    """Test fluent validator."""
    
    def test_fluent_interface(self):
        """Test fluent interface."""
        validator = ParameterValidator()
        validator.check_type(42, 'count', int) \
                 .check_range(42, 'count', min_val=1) \
                 .check_positive(5, 'length')
        validator.validate()  # Should not raise
    
    def test_multiple_errors(self):
        """Test collecting multiple errors."""
        validator = ParameterValidator()
        validator.check_type('hello', 'count', int) \
                 .check_range(-5, 'offset', min_val=0) \
                 .check_positive(-1, 'length')
        
        with pytest.raises(ValueError, match="Validation failed"):
            validator.validate()
    
    def test_check_choices(self):
        """Test choices checking."""
        validator = ParameterValidator()
        validator.check_choices('fast', 'method', ['fast', 'slow'])
        validator.validate()
        
        validator = ParameterValidator()
        validator.check_choices('invalid', 'method', ['fast', 'slow'])
        
        with pytest.raises(ValueError):
            validator.validate()



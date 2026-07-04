"""
Utility modules for FiberNet.

Submodules:
- units: Unit system management and conversion
- geometry: Geometric utilities
- io_helpers: I/O helper functions
"""

from fibernet.utils.units import (
    UnitSystem, SI, CGS, MICRO, NANO, MOLECULAR,
    UnitConverter, convert_network_units,
)

__all__ = [
    "UnitSystem", "SI", "CGS", "MICRO", "NANO", "MOLECULAR",
    "UnitConverter", "convert_network_units",
]

from fibernet.utils.validation import (
    validate_positive, validate_non_negative, validate_range,
    validate_integer, validate_box_size, validate_probability,
    validate_angle, validate_seed, validate_material_properties,
)

# Parametric study and sensitivity analysis
from .parametric import (
    parametric_sweep,
    sensitivity_analysis,
    monte_carlo_analysis,
    correlation_matrix
)

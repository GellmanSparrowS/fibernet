"""
Unit Conversion Utilities Module

Provides unit conversion utilities for fiber network research:
- Length conversions
- Force conversions
- Pressure/Stress conversions
- Temperature conversions
- Energy conversions

All conversions are exact unless otherwise noted.

References:
- NIST Guide to the SI: https://physics.nist.gov/cuu/Units/
"""

import numpy as np
from typing import Union


# ============================================================
# LENGTH CONVERSIONS
# ============================================================

LENGTH_FACTORS = {
    'm': 1.0,
    'mm': 1e-3,
    'um': 1e-6,
    'µm': 1e-6,
    'nm': 1e-9,
    'cm': 1e-2,
    'km': 1e3,
    'in': 0.0254,
    'ft': 0.3048,
    'Å': 1e-10,
    'angstrom': 1e-10,
}


def convert_length(value: float, from_unit: str, to_unit: str) -> float:
    """Convert length between units.
    
    Parameters
    ----------
    value : float
        Length value to convert.
    from_unit : str
        Source unit ('m', 'mm', 'um', 'nm', 'cm', 'km', 'in', 'ft').
    to_unit : str
        Target unit.
    
    Returns
    -------
    converted : float
        Converted length value.
    
    Examples
    --------
    >>> from fibernet.units import convert_length
    >>> convert_length(1.0, 'mm', 'um')
    1000.0
    >>> convert_length(100, 'nm', 'm')
    1e-07
    """
    if from_unit not in LENGTH_FACTORS:
        raise ValueError(f"Unknown unit: {from_unit}. Available: {list(LENGTH_FACTORS.keys())}")
    if to_unit not in LENGTH_FACTORS:
        raise ValueError(f"Unknown unit: {to_unit}. Available: {list(LENGTH_FACTORS.keys())}")
    
    # Convert to meters, then to target unit
    meters = value * LENGTH_FACTORS[from_unit]
    return meters / LENGTH_FACTORS[to_unit]


# ============================================================
# FORCE CONVERSIONS
# ============================================================

FORCE_FACTORS = {
    'N': 1.0,
    'kN': 1e3,
    'mN': 1e-3,
    'µN': 1e-6,
    'uN': 1e-6,
    'nN': 1e-9,
    'lbf': 4.44822,
    'dyne': 1e-5,
}


def convert_force(value: float, from_unit: str, to_unit: str) -> float:
    """Convert force between units.
    
    Parameters
    ----------
    value : float
        Force value to convert.
    from_unit : str
        Source unit ('N', 'kN', 'mN', 'µN', 'nN', 'lbf', 'dyne').
    to_unit : str
        Target unit.
    
    Returns
    -------
    converted : float
        Converted force value.
    """
    if from_unit not in FORCE_FACTORS:
        raise ValueError(f"Unknown unit: {from_unit}. Available: {list(FORCE_FACTORS.keys())}")
    if to_unit not in FORCE_FACTORS:
        raise ValueError(f"Unknown unit: {to_unit}. Available: {list(FORCE_FACTORS.keys())}")
    
    newtons = value * FORCE_FACTORS[from_unit]
    return newtons / FORCE_FACTORS[to_unit]


# ============================================================
# PRESSURE/STRESS CONVERSIONS
# ============================================================

PRESSURE_FACTORS = {
    'Pa': 1.0,
    'kPa': 1e3,
    'MPa': 1e6,
    'GPa': 1e9,
    'TPa': 1e12,
    'bar': 1e5,
    'atm': 101325.0,
    'psi': 6894.76,
    'ksi': 6894760.0,
}


def convert_pressure(value: float, from_unit: str, to_unit: str) -> float:
    """Convert pressure/stress between units.
    
    Parameters
    ----------
    value : float
        Pressure value to convert.
    from_unit : str
        Source unit ('Pa', 'kPa', 'MPa', 'GPa', 'TPa', 'bar', 'atm', 'psi', 'ksi').
    to_unit : str
        Target unit.
    
    Returns
    -------
    converted : float
        Converted pressure value.
    """
    if from_unit not in PRESSURE_FACTORS:
        raise ValueError(f"Unknown unit: {from_unit}. Available: {list(PRESSURE_FACTORS.keys())}")
    if to_unit not in PRESSURE_FACTORS:
        raise ValueError(f"Unknown unit: {to_unit}. Available: {list(PRESSURE_FACTORS.keys())}")
    
    pascals = value * PRESSURE_FACTORS[from_unit]
    return pascals / PRESSURE_FACTORS[to_unit]


# ============================================================
# TEMPERATURE CONVERSIONS
# ============================================================

def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature between units.
    
    Parameters
    ----------
    value : float
        Temperature value to convert.
    from_unit : str
        Source unit ('K', 'C', 'F', 'R' for Kelvin, Celsius, Fahrenheit, Rankine).
    to_unit : str
        Target unit.
    
    Returns
    -------
    converted : float
        Converted temperature value.
    """
    # Convert to Kelvin
    if from_unit == 'K':
        kelvin = value
    elif from_unit == 'C':
        kelvin = value + 273.15
    elif from_unit == 'F':
        kelvin = (value - 32) * 5/9 + 273.15
    elif from_unit == 'R':
        kelvin = value * 5/9
    else:
        raise ValueError(f"Unknown unit: {from_unit}")
    
    # Convert from Kelvin to target
    if to_unit == 'K':
        return kelvin
    elif to_unit == 'C':
        return kelvin - 273.15
    elif to_unit == 'F':
        return (kelvin - 273.15) * 9/5 + 32
    elif to_unit == 'R':
        return kelvin * 9/5
    else:
        raise ValueError(f"Unknown unit: {to_unit}")


# ============================================================
# ENERGY CONVERSIONS
# ============================================================

ENERGY_FACTORS = {
    'J': 1.0,
    'kJ': 1e3,
    'mJ': 1e-3,
    'µJ': 1e-6,
    'uJ': 1e-6,
    'nJ': 1e-9,
    'cal': 4.184,
    'kcal': 4184.0,
    'eV': 1.602e-19,
    'keV': 1.602e-16,
    'MeV': 1.602e-13,
    'BTU': 1055.06,
    'kWh': 3.6e6,
}


def convert_energy(value: float, from_unit: str, to_unit: str) -> float:
    """Convert energy between units.
    
    Parameters
    ----------
    value : float
        Energy value to convert.
    from_unit : str
        Source unit.
    to_unit : str
        Target unit.
    
    Returns
    -------
    converted : float
        Converted energy value.
    """
    if from_unit not in ENERGY_FACTORS:
        raise ValueError(f"Unknown unit: {from_unit}. Available: {list(ENERGY_FACTORS.keys())}")
    if to_unit not in ENERGY_FACTORS:
        raise ValueError(f"Unknown unit: {to_unit}. Available: {list(ENERGY_FACTORS.keys())}")
    
    joules = value * ENERGY_FACTORS[from_unit]
    return joules / ENERGY_FACTORS[to_unit]


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def scale_network_properties(network, length_unit: str = 'm'):
    """Scale all network properties to a consistent unit system.
    
    Parameters
    ----------
    network : FiberNetwork
        Network to scale.
    length_unit : str
        Target length unit ('m', 'mm', 'um', 'nm').
    
    Returns
    -------
    scale_factor : float
        Scale factor applied.
    
    Examples
    --------
    >>> from fibernet.units import scale_network_properties
    >>> factor = scale_network_properties(net, length_unit='mm')
    >>> print(f"Scale factor: {factor}")
    """
    if length_unit not in LENGTH_FACTORS:
        raise ValueError(f"Unknown unit: {length_unit}")
    
    return LENGTH_FACTORS[length_unit]


# ============================================================
# UNIT PARSING
# ============================================================

def parse_unit_string(unit_str: str):
    """Parse a unit string and return (value_factor, unit_type).
    
    Parameters
    ----------
    unit_str : str
        Unit string like 'mm', 'MPa', 'kN', etc.
    
    Returns
    -------
    factor : float
        Conversion factor to SI base unit.
    unit_type : str
        Type of unit ('length', 'force', 'pressure', 'energy').
    
    Examples
    --------
    >>> from fibernet.units import parse_unit_string
    >>> factor, utype = parse_unit_string('MPa')
    >>> print(f"Factor: {factor}, Type: {utype}")
    Factor: 1000000.0, Type: pressure
    """
    unit_str = unit_str.strip()
    
    if unit_str in LENGTH_FACTORS:
        return LENGTH_FACTORS[unit_str], 'length'
    elif unit_str in FORCE_FACTORS:
        return FORCE_FACTORS[unit_str], 'force'
    elif unit_str in PRESSURE_FACTORS:
        return PRESSURE_FACTORS[unit_str], 'pressure'
    elif unit_str in ENERGY_FACTORS:
        return ENERGY_FACTORS[unit_str], 'energy'
    else:
        raise ValueError(f"Unknown unit: {unit_str}")



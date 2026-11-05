"""
Unit system management for FiberNet.

Provides consistent unit handling across the library.
Supports SI, CGS, and reduced (molecular) unit systems.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from numbers import Real


@dataclass
class UnitSystem:
    """Defines a unit system with base units.
    
    Attributes
    ----------
    name : str
        Unit system name.
    length : str
        Length unit (e.g., 'm', 'µm', 'nm', 'Å').
    mass : str
        Mass unit (e.g., 'kg', 'g', 'amu').
    time : str
        Time unit (e.g., 's', 'ps', 'fs').
    temperature : str
        Temperature unit (e.g., 'K').
    """
    name: str
    length: str = "m"
    mass: str = "kg"
    time: str = "s"
    temperature: str = "K"
    
    # Conversion factors to SI
    length_to_si: float = 1.0
    mass_to_si: float = 1.0
    time_to_si: float = 1.0
    temperature_to_si: float = 1.0  # K to K = 1
    
    # Derived units (computed)
    @property
    def force_to_si(self) -> float:
        return float(self.mass_to_si) * float(self.length_to_si) / float(self.time_to_si)**2
    
    @property
    def energy_to_si(self) -> float:
        return self.force_to_si * float(self.length_to_si)
    
    @property
    def stress_to_si(self) -> float:
        return self.force_to_si / float(self.length_to_si)**2
    
    @property
    def velocity_to_si(self) -> float:
        return float(self.length_to_si) / float(self.time_to_si)
    
    @property
    def viscosity_to_si(self) -> float:
        return self.stress_to_si * float(self.time_to_si)


# Pre-defined unit systems
SI = UnitSystem(
    name="SI",
    length="m", mass="kg", time="s", temperature="K",
    length_to_si=1.0, mass_to_si=1.0, time_to_si=1.0,
)

CGS = UnitSystem(
    name="CGS",
    length="cm", mass="g", time="s", temperature="K",
    length_to_si=1e-2, mass_to_si=1e-3, time_to_si=1.0,
)

MICRO = UnitSystem(
    name="Micro (µm·mg·ms)",
    length="µm", mass="mg", time="ms", temperature="K",
    length_to_si=1e-6, mass_to_si=1e-6, time_to_si=1e-3,
)

NANO = UnitSystem(
    name="Nano (nm·ag·ps)",
    length="nm", mass="ag", time="ps", temperature="K",
    length_to_si=1e-9, mass_to_si=1e-21, time_to_si=1e-12,
)

MOLECULAR = UnitSystem(
    name="Molecular (Å·amu·fs)",
    length="Å", mass="amu", time="fs", temperature="K",
    length_to_si=1e-10,
    mass_to_si=1.660539e-27,
    time_to_si=1e-15,
)


class UnitConverter:
    """Convert values between unit systems."""
    
    _systems: Dict[str, UnitSystem] = {
        "SI": SI, "CGS": CGS, "MICRO": MICRO,
        "NANO": NANO, "MOLECULAR": MOLECULAR,
    }
    
    @classmethod
    def get_system(cls, name: str) -> UnitSystem:
        """Look up a named system; unknown names are never interpreted as SI."""
        if not isinstance(name, str) or name.upper() not in cls._systems:
            raise ValueError(f"Unknown unit system: {name!r}")
        system = cls._systems[name.upper()]
        for field in ('length_to_si', 'mass_to_si', 'time_to_si', 'temperature_to_si'):
            factor = getattr(system, field)
            if isinstance(factor, (bool, np.bool_)) or not isinstance(factor, Real):
                raise ValueError(f"{field} must be finite and positive")
            try:
                factor = float(factor)
            except (ArithmeticError, TypeError, ValueError) as exc:
                raise ValueError(f"{field} must be finite and positive") from exc
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError(f"{field} must be finite and positive")
        return system

    @classmethod
    def convert(cls, value: float, quantity: str,
                from_system: str, to_system: str) -> float:
        """Convert a scalar or array for a supported physical quantity.

        Supported quantities: length, mass, time, force, energy, stress,
        velocity and viscosity. Unknown systems or quantities raise ValueError.
        """
        src = cls.get_system(from_system)
        dst = cls.get_system(to_system)
        quantities = ('length', 'mass', 'time', 'force', 'energy', 'stress',
                      'velocity', 'viscosity')
        if not isinstance(quantity, str) or quantity not in quantities:
            raise ValueError(f"Unknown physical quantity: {quantity!r}")
        attr = quantity + '_to_si'
        try:
            factor = float(getattr(src, attr)) / float(getattr(dst, attr))
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise ValueError("Unit conversion factors are outside the supported range") from exc
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Unit conversion factor must be finite and positive")
        return value * factor

    @classmethod
    def to_si(cls, value: float, quantity: str, from_system: str) -> float:
        return cls.convert(value, quantity, from_system, "SI")

    @classmethod
    def from_si(cls, value: float, quantity: str, to_system: str) -> float:
        return cls.convert(value, quantity, "SI", to_system)


def _label_metadata(value):
    """Whether a custom material value has no unclassified numeric content."""
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, dict):
        return all(isinstance(k, str) and _label_metadata(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_label_metadata(v) for v in value)
    return False


def _scaled(value, factor, field, *, allow_infinity=False):
    array = np.asarray(value)
    if array.dtype.kind not in 'iuf':
        raise ValueError(f"{field} must contain real numbers")
    with np.errstate(over='ignore', invalid='ignore'):
        result = array.astype(float) * factor
    valid = np.isfinite(result)
    if allow_infinity:
        valid = valid | (np.isposinf(array) & np.isposinf(result))
    if not np.all(valid):
        raise ValueError(f"{field} conversion requires finite values and results")
    return float(result) if result.ndim == 0 else result


def _check_declared_fields(value):
    """Do not relabel undeclared object attributes with a new unit system."""
    from dataclasses import fields
    unknown = set(vars(value)) - {field.name for field in fields(value)}
    if unknown:
        raise ValueError(f"Undeclared attributes cannot be converted: {sorted(unknown)}")


def convert_network_units(network, from_system: str, to_system: str):
    """Return a converted copy of a legacy FiberNetwork.

    Converts geometry, supported cross sections, crosslink force/stiffness and
    mechanical/thermal Material fields. Shared materials remain shared and are
    converted once. The input is unchanged, including when conversion fails.
    Network metadata is descriptive and copied unchanged except unit_system;
    this helper does not convert loads or results stored in arbitrary metadata.

    Electrical conductivity is unsupported because these systems define no
    electric-current base unit. Numeric Material.extra properties require an
    explicit dimensional contract and are rejected; descriptive labels survive.
    Custom/triangular sections and unknown section parameters are rejected.
    Subclasses, additional object attributes, and missing Material objects are
    rejected: their dimensions cannot be inferred. Supply an explicit Material
    rather than relying on this conversion function to invent one.
    """
    from copy import deepcopy
    from fibernet.core.network import FiberNetwork, Crosslink
    from fibernet.core.fiber import Fiber, CrossSection
    from fibernet.core.material import Material

    if type(network) is not FiberNetwork:
        raise TypeError("convert_network_units requires a FiberNetwork")
    _check_declared_fields(network)
    src = UnitConverter.get_system(from_system)
    dst = UnitConverter.get_system(to_system)
    try:
        length = float(src.length_to_si) / float(dst.length_to_si)
        mass = float(src.mass_to_si) / float(dst.mass_to_si)
        time = float(src.time_to_si) / float(dst.time_to_si)
        temperature = float(src.temperature_to_si) / float(dst.temperature_to_si)
        stress = mass / (length * time**2)
        force = mass * length / time**2
        factors = {
            'density': mass / length**3,
            'youngs_modulus': stress,
            'shear_modulus': stress,
            'yield_strength': stress,
            'tensile_strength': stress,
            'fracture_toughness': stress * np.sqrt(length),
            'thermal_conductivity': mass * length / (time**3 * temperature),
            'specific_heat': length**2 / (time**2 * temperature),
            'thermal_expansion': 1.0 / temperature,
            'stiffness_tensor': stress,
        }
        crosslink_stiffness = force / length
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise ValueError("Unit conversion factors are outside the supported range") from exc
    if any(not np.isfinite(x) or x <= 0 for x in
           (length, force, crosslink_stiffness, *factors.values())):
        raise ValueError("Unit conversion factors must be finite and positive")
    section_fields = {
        CrossSection.CIRCULAR: (),
        CrossSection.RECTANGULAR: ('width', 'height'),
        CrossSection.ELLIPTICAL: ('semi_minor',),
        CrossSection.HOLLOW_CIRCULAR: ('inner_radius',),
    }
    result = deepcopy(network)
    seen_fibers, seen_materials, seen_crosslinks = set(), set(), set()
    for fiber in result.fibers:
        if type(fiber) is not Fiber:
            raise TypeError("Only Fiber objects are supported")
        _check_declared_fields(fiber)
        if id(fiber) in seen_fibers:
            continue
        seen_fibers.add(id(fiber))
        allowed = section_fields.get(fiber.cross_section)
        if allowed is None or not set(fiber.cross_section_params).issubset(allowed):
            raise ValueError("Unsupported cross-section conversion")
        fiber.centerline = _scaled(fiber.centerline, length, 'centerline')
        fiber.radius = _scaled(fiber.radius, length, 'radius')
        fiber.cross_section_params = {
            key: _scaled(value, length, key)
            for key, value in fiber.cross_section_params.items()
        }
        mat = fiber.material
        if type(mat) is not Material:
            raise TypeError("Each fiber must have a Material")
        _check_declared_fields(mat)
        if id(mat) in seen_materials:
            continue
        seen_materials.add(id(mat))
        if mat.electrical_conductivity is not None:
            raise ValueError("Electrical conductivity conversion requires a current unit")
        if not _label_metadata(mat.extra):
            raise ValueError("Numeric Material.extra properties need declared dimensions")
        for field, factor in factors.items():
            value = getattr(mat, field)
            if value is not None:
                setattr(mat, field, _scaled(value, factor, field))
    for crosslink in result.crosslinks:
        if type(crosslink) is not Crosslink:
            raise TypeError("Only Crosslink objects are supported")
        _check_declared_fields(crosslink)
        if id(crosslink) in seen_crosslinks:
            continue
        seen_crosslinks.add(id(crosslink))
        crosslink.position = _scaled(crosslink.position, length, 'crosslink.position')
        crosslink.strength = _scaled(crosslink.strength, force, 'crosslink.strength',
                                    allow_infinity=True)
        crosslink.stiffness = _scaled(crosslink.stiffness, crosslink_stiffness,
                                     'crosslink.stiffness', allow_infinity=True)
    if result.box_size is not None:
        result.box_size = _scaled(result.box_size, length, 'box_size')
    result.metadata['unit_system'] = to_system.upper()
    return result

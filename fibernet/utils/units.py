"""
Unit system management for FiberNet.

Provides consistent unit handling across the library.
Supports SI, CGS, and reduced (molecular) unit systems.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


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
        return self.mass_to_si * self.length_to_si / self.time_to_si**2
    
    @property
    def energy_to_si(self) -> float:
        return self.force_to_si * self.length_to_si
    
    @property
    def stress_to_si(self) -> float:
        return self.force_to_si / self.length_to_si**2
    
    @property
    def velocity_to_si(self) -> float:
        return self.length_to_si / self.time_to_si
    
    @property
    def viscosity_to_si(self) -> float:
        return self.stress_to_si * self.time_to_si


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
        return cls._systems.get(name.upper(), SI)
    
    @classmethod
    def convert(cls, value: float, quantity: str,
                from_system: str, to_system: str) -> float:
        """Convert a value between unit systems.
        
        Parameters
        ----------
        value : float
            Value to convert.
        quantity : str
            Physical quantity: 'length', 'mass', 'time', 'force',
            'energy', 'stress', 'velocity', 'viscosity'.
        from_system : str
            Source unit system name.
        to_system : str
            Target unit system name.
        """
        src = cls.get_system(from_system)
        dst = cls.get_system(to_system)
        
        converters = {
            'length': ('length_to_si',),
            'mass': ('mass_to_si',),
            'time': ('time_to_si',),
            'force': ('force_to_si',),
            'energy': ('energy_to_si',),
            'stress': ('stress_to_si',),
            'velocity': ('velocity_to_si',),
            'viscosity': ('viscosity_to_si',),
        }
        
        if quantity not in converters:
            return value
        
        attr = converters[quantity][0]
        src_factor = getattr(src, attr)
        dst_factor = getattr(dst, attr)
        
        return value * src_factor / dst_factor
    
    @classmethod
    def to_si(cls, value: float, quantity: str, from_system: str) -> float:
        return cls.convert(value, quantity, from_system, "SI")
    
    @classmethod
    def from_si(cls, value: float, quantity: str, to_system: str) -> float:
        return cls.convert(value, quantity, "SI", to_system)


def convert_network_units(
    network,
    from_system: str,
    to_system: str,
):
    """Convert all physical quantities in a network between unit systems.
    
    Parameters
    ----------
    network : FiberNetwork
        Network to convert (modified in place).
    from_system : str
        Current unit system.
    to_system : str
        Target unit system.
    """
    from copy import deepcopy
    
    result = deepcopy(network)
    
    L_factor = UnitConverter.convert(1.0, 'length', from_system, to_system)
    
    for fiber in result.fibers:
        fiber.centerline *= L_factor
        fiber.radius *= L_factor
        
        mat = fiber.material
        if mat.youngs_modulus:
            mat.youngs_modulus = UnitConverter.convert(
                mat.youngs_modulus, 'stress', from_system, to_system
            )
        if mat.shear_modulus:
            mat.shear_modulus = UnitConverter.convert(
                mat.shear_modulus, 'stress', from_system, to_system
            )
        if mat.density:
            src = UnitConverter.get_system(from_system)
            dst = UnitConverter.get_system(to_system)
            density_factor = (src.mass_to_si / src.length_to_si**3) / (dst.mass_to_si / dst.length_to_si**3)
            mat.density *= density_factor
    
    if result.box_size is not None:
        result.box_size = result.box_size * L_factor
    
    result.metadata['unit_system'] = to_system
    return result

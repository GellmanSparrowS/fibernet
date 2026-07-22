"""
Material properties database for fiber network simulations.

Supports isotropic and anisotropic materials, with built-in database
of common fiber materials (polymers, metals, ceramics, biological).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Material:
    """Represents a material with mechanical, thermal, and electromagnetic properties.
    
    Parameters
    ----------
    name : str
        Material identifier name.
    density : float
        Mass density in kg/m^3.
    youngs_modulus : float
        Young's modulus in Pa.
    poissons_ratio : float
        Poisson's ratio (dimensionless).
    shear_modulus : float, optional
        Shear modulus in Pa. Computed from E and nu if not given.
    yield_strength : float, optional
        Yield strength in Pa.
    tensile_strength : float, optional
        Ultimate tensile strength in Pa.
    fracture_toughness : float, optional
        Fracture toughness K_IC in Pa*m^0.5.
    thermal_conductivity : float, optional
        Thermal conductivity in W/(m*K).
    specific_heat : float, optional
        Specific heat capacity in J/(kg*K).
    thermal_expansion : float, optional
        Coefficient of thermal expansion in 1/K.
    electrical_conductivity : float, optional
        Electrical conductivity in S/m.
    permittivity : float, optional
        Relative permittivity (dielectric constant).
    permeability : float, optional
        Relative magnetic permeability.
    stiffness_tensor : np.ndarray, optional
        Full 6x6 stiffness tensor (Voigt notation) for anisotropic materials.
    extra : dict
        Additional custom properties.
    """
    name: str = "generic"
    density: float = 1000.0
    youngs_modulus: float = 1e9
    poissons_ratio: float = 0.3
    shear_modulus: Optional[float] = None
    yield_strength: Optional[float] = None
    tensile_strength: Optional[float] = None
    fracture_toughness: Optional[float] = None
    thermal_conductivity: Optional[float] = None
    specific_heat: Optional[float] = None
    thermal_expansion: Optional[float] = None
    electrical_conductivity: Optional[float] = None
    permittivity: Optional[float] = None
    permeability: Optional[float] = None
    stiffness_tensor: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.shear_modulus is None:
            self.shear_modulus = self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))

    @property
    def bulk_modulus(self) -> float:
        """Bulk modulus in Pa."""
        return self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poissons_ratio))

    @property
    def is_anisotropic(self) -> bool:
        """Whether the material has a full stiffness tensor defined."""
        return self.stiffness_tensor is not None

    def get_lame_parameters(self) -> tuple:
        """Return Lame parameters (lambda, mu)."""
        lam = self.youngs_modulus * self.poissons_ratio / (
            (1.0 + self.poissons_ratio) * (1.0 - 2.0 * self.poissons_ratio)
        )
        mu = self.shear_modulus
        return lam, mu

    def to_dict(self) -> Dict[str, Any]:
        """Serialize material to dictionary."""
        data = {
            "name": self.name,
            "density": self.density,
            "youngs_modulus": self.youngs_modulus,
            "poissons_ratio": self.poissons_ratio,
            "shear_modulus": self.shear_modulus,
        }
        optional_fields = [
            "yield_strength", "tensile_strength", "fracture_toughness",
            "thermal_conductivity", "specific_heat", "thermal_expansion",
            "electrical_conductivity", "permittivity", "permeability",
        ]
        for f in optional_fields:
            val = getattr(self, f)
            if val is not None:
                data[f] = val
        if self.stiffness_tensor is not None:
            data["stiffness_tensor"] = self.stiffness_tensor.tolist()
        if self.extra:
            data["extra"] = self.extra
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Material":
        """Create Material from dictionary."""
        if "stiffness_tensor" in data and data["stiffness_tensor"] is not None:
            data["stiffness_tensor"] = np.array(data["stiffness_tensor"])
        return cls(**data)

    def __repr__(self) -> str:
        return (
            f"Material(name='{self.name}', E={self.youngs_modulus:.3e} Pa, "
            f"nu={self.poissons_ratio}, rho={self.density} kg/m^3)"
        )


# ============================================================
# Built-in Material Database
# ============================================================

MATERIAL_DB = {
    # --- Polymers ---
    "nylon": Material(
        name="nylon", density=1140.0, youngs_modulus=2.7e9, poissons_ratio=0.40,
        yield_strength=70e6, tensile_strength=85e6,
        thermal_conductivity=0.25, specific_heat=1700.0, thermal_expansion=80e-6,
    ),
    "pla": Material(
        name="pla", density=1240.0, youngs_modulus=3.5e9, poissons_ratio=0.36,
        yield_strength=60e6, tensile_strength=70e6,
        thermal_conductivity=0.13, specific_heat=1900.0, thermal_expansion=68e-6,
    ),
    "pva": Material(
        name="pva", density=1190.0, youngs_modulus=2.0e9, poissons_ratio=0.42,
        tensile_strength=40e6,
        thermal_conductivity=0.20, specific_heat=1500.0,
    ),
    "pdms": Material(
        name="pdms", density=970.0, youngs_modulus=1.8e6, poissons_ratio=0.49,
        tensile_strength=3.0e6,
        thermal_conductivity=0.15, specific_heat=1460.0, thermal_expansion=310e-6,
    ),
    "kevlar": Material(
        name="kevlar", density=1440.0, youngs_modulus=70e9, poissons_ratio=0.36,
        tensile_strength=3.6e9, fracture_toughness=50e6,
        thermal_conductivity=0.04, specific_heat=1420.0,
    ),
    "uHMWPE": Material(
        name="uHMWPE", density=970.0, youngs_modulus=120e9, poissons_ratio=0.46,
        tensile_strength=3.5e9,
        thermal_conductivity=0.45, specific_heat=1850.0,
    ),

    # --- Carbon-based ---
    "carbon_fiber": Material(
        name="carbon_fiber", density=1750.0, youngs_modulus=230e9, poissons_ratio=0.28,
        tensile_strength=4.0e9, fracture_toughness=25e6,
        thermal_conductivity=10.0, specific_heat=710.0, thermal_expansion=-0.5e-6,
        electrical_conductivity=1e4,
    ),
    "cnt": Material(
        name="cnt", density=1300.0, youngs_modulus=1e12, poissons_ratio=0.20,
        tensile_strength=63e9,
        thermal_conductivity=3000.0, specific_heat=700.0,
        electrical_conductivity=1e6,
    ),
    "graphene_sheet": Material(
        name="graphene_sheet", density=2200.0, youngs_modulus=1e12, poissons_ratio=0.17,
        tensile_strength=130e9,
        thermal_conductivity=5000.0, specific_heat=700.0,
        electrical_conductivity=1e7,
    ),

    # --- Metals ---
    "steel": Material(
        name="steel", density=7800.0, youngs_modulus=200e9, poissons_ratio=0.30,
        yield_strength=250e6, tensile_strength=400e6, fracture_toughness=50e6,
        thermal_conductivity=50.0, specific_heat=500.0, thermal_expansion=12e-6,
        electrical_conductivity=6e6, permeability=100.0,
    ),
    "aluminum": Material(
        name="aluminum", density=2700.0, youngs_modulus=70e9, poissons_ratio=0.33,
        yield_strength=40e6, tensile_strength=90e6,
        thermal_conductivity=237.0, specific_heat=897.0, thermal_expansion=23e-6,
        electrical_conductivity=3.5e7,
    ),
    "copper": Material(
        name="copper", density=8960.0, youngs_modulus=110e9, poissons_ratio=0.34,
        yield_strength=70e6, tensile_strength=220e6,
        thermal_conductivity=401.0, specific_heat=385.0, thermal_expansion=17e-6,
        electrical_conductivity=5.96e7,
    ),
    "titanium": Material(
        name="titanium", density=4500.0, youngs_modulus=116e9, poissons_ratio=0.32,
        yield_strength=880e6, tensile_strength=950e6,
        thermal_conductivity=21.9, specific_heat=523.0, thermal_expansion=8.6e-6,
        electrical_conductivity=2.34e6,
    ),

    # --- Ceramics ---
    "silica": Material(
        name="silica", density=2200.0, youngs_modulus=72e9, poissons_ratio=0.17,
        tensile_strength=50e6, fracture_toughness=0.8e6,
        thermal_conductivity=1.4, specific_heat=730.0, thermal_expansion=0.55e-6,
        permittivity=3.9,
    ),
    "alumina": Material(
        name="alumina", density=3950.0, youngs_modulus=370e9, poissons_ratio=0.22,
        tensile_strength=300e6, fracture_toughness=3.5e6,
        thermal_conductivity=30.0, specific_heat=880.0, thermal_expansion=8.0e-6,
        permittivity=9.8,
    ),

    # --- Biological ---
    "collagen": Material(
        name="collagen", density=1300.0, youngs_modulus=1.2e9, poissons_ratio=0.35,
        tensile_strength=120e6,
        thermal_conductivity=0.5, specific_heat=1600.0,
    ),
    "cellulose": Material(
        name="cellulose", density=1500.0, youngs_modulus=10e9, poissons_ratio=0.30,
        tensile_strength=300e6,
        thermal_conductivity=0.04, specific_heat=1300.0,
    ),
    "silk": Material(
        name="silk", density=1350.0, youngs_modulus=10e9, poissons_ratio=0.35,
        tensile_strength=1200e6, fracture_toughness=100e6,
        thermal_conductivity=0.05, specific_heat=1400.0,
    ),
    "fibrin": Material(
        name="fibrin", density=1100.0, youngs_modulus=1e6, poissons_ratio=0.45,
        tensile_strength=10e6,
        thermal_conductivity=0.5, specific_heat=3500.0,
    ),
    "actin": Material(
        name="actin", density=1300.0, youngs_modulus=2e9, poissons_ratio=0.38,
        tensile_strength=600e6,
        thermal_conductivity=0.5, specific_heat=1500.0,
    ),

    # --- Glass ---
    "glass_fiber": Material(
        name="glass_fiber", density=2500.0, youngs_modulus=76e9, poissons_ratio=0.22,
        tensile_strength=3.4e9,
        thermal_conductivity=1.0, specific_heat=840.0, thermal_expansion=5.4e-6,
        permittivity=4.7,
    ),
}


def get_material(name: str) -> Material:
    """Get a material from the built-in database.
    
    Parameters
    ----------
    name : str
        Material name (case-insensitive).
    
    Returns
    -------
    Material
        A copy of the material from the database.
    
    Raises
    ------
    KeyError
        If material name is not found.
    """
    key = name.lower().strip()
    if key not in MATERIAL_DB:
        available = ", ".join(sorted(MATERIAL_DB.keys()))
        raise KeyError(f"Material '{name}' not found. Available: {available}")
    return Material.from_dict(MATERIAL_DB[key].to_dict())


def list_materials() -> list:
    """List all available materials in the database."""
    return sorted(MATERIAL_DB.keys())

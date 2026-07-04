"""
Material Database Module

Provides a comprehensive database of pre-defined materials for fiber networks:
- Polymers (natural and synthetic)
- Metals
- Ceramics
- Biological materials
- Composites
- Carbon-based materials

All properties are at room temperature (293 K) unless otherwise specified.

References:
- Ashby, M.F., "Materials Selection in Mechanical Design", Butterworth-Heinemann, 2016
- Callister, W.D., "Materials Science and Engineering", Wiley, 2018
"""

from fibernet.core.material import Material


def _create_material_db():
    """Create the material database."""
    
    db = {}
    
    # ================================================================
    # POLYMERS
    # ================================================================
    
    db['polypropylene'] = Material(
        name='polypropylene',
        density=900.0,  # kg/m³
        youngs_modulus=1.5e9,  # Pa
        poissons_ratio=0.42,
        yield_strength=30e6,  # Pa
        tensile_strength=35e6,  # Pa
        fracture_toughness=5e6,  # Pa·√m
        thermal_conductivity=0.12,  # W/(m·K)
        specific_heat=1900.0,  # J/(kg·K)
        thermal_expansion=100e-6,  # 1/K
    )
    
    db['polyethylene_hdpe'] = Material(
        name='polyethylene_hdpe',
        density=950.0,
        youngs_modulus=1.0e9,
        poissons_ratio=0.42,
        yield_strength=25e6,
        tensile_strength=30e6,
        fracture_toughness=4e6,
        thermal_conductivity=0.46,
        specific_heat=1800.0,
        thermal_expansion=150e-6,
    )
    
    db['nylon_6'] = Material(
        name='nylon_6',
        density=1140.0,
        youngs_modulus=2.5e9,
        poissons_ratio=0.40,
        yield_strength=70e6,
        tensile_strength=80e6,
        fracture_toughness=6e6,
        thermal_conductivity=0.25,
        specific_heat=1600.0,
        thermal_expansion=80e-6,
    )
    
    db['kevlar'] = Material(
        name='kevlar',
        density=1440.0,
        youngs_modulus=70e9,
        poissons_ratio=0.36,
        yield_strength=2800e6,
        tensile_strength=3600e6,
        fracture_toughness=50e6,
        thermal_conductivity=0.04,
        specific_heat=1420.0,
        thermal_expansion=-2e-6,
    )
    
    db['pla'] = Material(
        name='pla',
        density=1240.0,
        youngs_modulus=3.5e9,
        poissons_ratio=0.36,
        yield_strength=60e6,
        tensile_strength=65e6,
        fracture_toughness=2e6,
        thermal_conductivity=0.13,
        specific_heat=1800.0,
        thermal_expansion=68e-6,
    )
    
    db['pva'] = Material(
        name='pva',
        density=1190.0,
        youngs_modulus=3.0e9,
        poissons_ratio=0.38,
        yield_strength=50e6,
        tensile_strength=60e6,
        fracture_toughness=3e6,
        thermal_conductivity=0.20,
        specific_heat=1500.0,
        thermal_expansion=70e-6,
    )
    
    # ================================================================
    # NATURAL FIBERS
    # ================================================================
    
    db['cotton'] = Material(
        name='cotton',
        density=1500.0,
        youngs_modulus=10e9,
        poissons_ratio=0.30,
        yield_strength=None,
        tensile_strength=500e6,
        fracture_toughness=None,
        thermal_conductivity=0.07,
        specific_heat=1300.0,
        thermal_expansion=6e-6,
    )
    
    db['silk'] = Material(
        name='silk',
        density=1300.0,
        youngs_modulus=15e9,
        poissons_ratio=0.33,
        yield_strength=None,
        tensile_strength=500e6,
        fracture_toughness=70e6,
        thermal_conductivity=0.04,
        specific_heat=1400.0,
        thermal_expansion=10e-6,
    )
    
    db['wool'] = Material(
        name='wool',
        density=1300.0,
        youngs_modulus=3e9,
        poissons_ratio=0.35,
        yield_strength=None,
        tensile_strength=200e6,
        fracture_toughness=None,
        thermal_conductivity=0.04,
        specific_heat=1360.0,
        thermal_expansion=25e-6,
    )
    
    db['cellulose'] = Material(
        name='cellulose',
        density=1500.0,
        youngs_modulus=120e9,
        poissons_ratio=0.30,
        yield_strength=None,
        tensile_strength=1000e6,
        fracture_toughness=None,
        thermal_conductivity=0.05,
        specific_heat=1300.0,
        thermal_expansion=2e-6,
    )
    
    # ================================================================
    # BIOLOGICAL MATERIALS
    # ================================================================
    
    db['collagen'] = Material(
        name='collagen',
        density=1300.0,
        youngs_modulus=1e9,
        poissons_ratio=0.35,
        yield_strength=None,
        tensile_strength=100e6,
        fracture_toughness=None,
        thermal_conductivity=0.20,
        specific_heat=1500.0,
        thermal_expansion=10e-6,
    )
    
    db['elastin'] = Material(
        name='elastin',
        density=1100.0,
        youngs_modulus=1e6,
        poissons_ratio=0.49,
        yield_strength=None,
        tensile_strength=2e6,
        fracture_toughness=None,
        thermal_conductivity=0.20,
        specific_heat=1500.0,
        thermal_expansion=10e-6,
    )
    
    db['fibrin'] = Material(
        name='fibrin',
        density=1100.0,
        youngs_modulus=5e6,
        poissons_ratio=0.45,
        yield_strength=None,
        tensile_strength=10e6,
        fracture_toughness=None,
        thermal_conductivity=0.20,
        specific_heat=1500.0,
        thermal_expansion=10e-6,
    )
    
    db['actin'] = Material(
        name='actin',
        density=1300.0,
        youngs_modulus=2e9,
        poissons_ratio=0.35,
        yield_strength=None,
        tensile_strength=100e6,
        fracture_toughness=None,
        thermal_conductivity=0.20,
        specific_heat=1500.0,
        thermal_expansion=10e-6,
    )
    
    # ================================================================
    # METALS
    # ================================================================
    
    db['steel'] = Material(
        name='steel',
        density=7850.0,
        youngs_modulus=210e9,
        poissons_ratio=0.28,
        yield_strength=250e6,
        tensile_strength=400e6,
        fracture_toughness=60e6,
        thermal_conductivity=50.0,
        specific_heat=500.0,
        thermal_expansion=12e-6,
    )
    
    db['aluminum'] = Material(
        name='aluminum',
        density=2700.0,
        youngs_modulus=69e9,
        poissons_ratio=0.33,
        yield_strength=40e6,
        tensile_strength=90e6,
        fracture_toughness=30e6,
        thermal_conductivity=237.0,
        specific_heat=900.0,
        thermal_expansion=23e-6,
    )
    
    db['titanium'] = Material(
        name='titanium',
        density=4500.0,
        youngs_modulus=110e9,
        poissons_ratio=0.32,
        yield_strength=800e6,
        tensile_strength=900e6,
        fracture_toughness=75e6,
        thermal_conductivity=21.9,
        specific_heat=520.0,
        thermal_expansion=8.6e-6,
    )
    
    db['copper'] = Material(
        name='copper',
        density=8960.0,
        youngs_modulus=110e9,
        poissons_ratio=0.34,
        yield_strength=70e6,
        tensile_strength=220e6,
        fracture_toughness=40e6,
        thermal_conductivity=401.0,
        specific_heat=385.0,
        thermal_expansion=16.5e-6,
        electrical_conductivity=5.96e7,  # S/m
    )
    
    db['tungsten'] = Material(
        name='tungsten',
        density=19300.0,
        youngs_modulus=411e9,
        poissons_ratio=0.28,
        yield_strength=750e6,
        tensile_strength=1500e6,
        fracture_toughness=80e6,
        thermal_conductivity=174.0,
        specific_heat=132.0,
        thermal_expansion=4.5e-6,
    )
    
    # ================================================================
    # CERAMICS
    # ================================================================
    
    db['alumina'] = Material(
        name='alumina',
        density=3950.0,
        youngs_modulus=390e9,
        poissons_ratio=0.22,
        yield_strength=None,
        tensile_strength=300e6,
        fracture_toughness=4e6,
        thermal_conductivity=30.0,
        specific_heat=880.0,
        thermal_expansion=8.1e-6,
    )
    
    db['silicon_carbide'] = Material(
        name='silicon_carbide',
        density=3200.0,
        youngs_modulus=410e9,
        poissons_ratio=0.14,
        yield_strength=None,
        tensile_strength=250e6,
        fracture_toughness=4e6,
        thermal_conductivity=120.0,
        specific_heat=750.0,
        thermal_expansion=4.0e-6,
    )
    
    db['glass'] = Material(
        name='glass',
        density=2500.0,
        youngs_modulus=70e9,
        poissons_ratio=0.22,
        yield_strength=None,
        tensile_strength=50e6,
        fracture_toughness=0.7e6,
        thermal_conductivity=1.0,
        specific_heat=840.0,
        thermal_expansion=9e-6,
    )
    
    # ================================================================
    # CARBON-BASED MATERIALS
    # ================================================================
    
    db['carbon_fiber'] = Material(
        name='carbon_fiber',
        density=1800.0,
        youngs_modulus=230e9,
        poissons_ratio=0.27,
        yield_strength=None,
        tensile_strength=3500e6,
        fracture_toughness=30e6,
        thermal_conductivity=10.0,
        specific_heat=710.0,
        thermal_expansion=-0.5e-6,
        electrical_conductivity=1e5,  # S/m (axial)
    )
    
    db['graphene'] = Material(
        name='graphene',
        density=2200.0,
        youngs_modulus=1e12,
        poissons_ratio=0.19,
        yield_strength=None,
        tensile_strength=130e9,
        fracture_toughness=None,
        thermal_conductivity=5000.0,
        specific_heat=710.0,
        thermal_expansion=-7e-6,
        electrical_conductivity=1e8,  # S/m
    )
    
    db['cnt'] = Material(
        name='cnt',
        density=1300.0,
        youngs_modulus=1e12,
        poissons_ratio=0.19,
        yield_strength=None,
        tensile_strength=50e9,
        fracture_toughness=None,
        thermal_conductivity=3500.0,
        specific_heat=710.0,
        thermal_expansion=-1e-6,
        electrical_conductivity=1e7,  # S/m
    )
    
    # ================================================================
    # GENERIC/DEFAULT MATERIALS
    # ================================================================
    
    db['generic_polymer'] = Material(
        name='generic_polymer',
        density=1200.0,
        youngs_modulus=2e9,
        poissons_ratio=0.35,
    )
    
    db['generic_metal'] = Material(
        name='generic_metal',
        density=7800.0,
        youngs_modulus=200e9,
        poissons_ratio=0.30,
        yield_strength=250e6,
    )
    
    db['generic_ceramic'] = Material(
        name='generic_ceramic',
        density=3500.0,
        youngs_modulus=300e9,
        poissons_ratio=0.25,
    )
    
    db['rubber'] = Material(
        name='rubber',
        density=1100.0,
        youngs_modulus=0.05e9,
        poissons_ratio=0.49,
        yield_strength=None,
        tensile_strength=20e6,
    )
    
    return db


# Global material database instance
_MATERIAL_DB = None


def get_material_database():
    """Get the material database (lazy initialization).
    
    Returns
    -------
    db : dict
        Dictionary mapping material names to Material objects.
    """
    global _MATERIAL_DB
    if _MATERIAL_DB is None:
        _MATERIAL_DB = _create_material_db()
    return _MATERIAL_DB


def get_material(name: str) -> Material:
    """Get a pre-defined material by name.
    
    Parameters
    ----------
    name : str
        Material name (case-insensitive).
    
    Returns
    -------
    material : Material
        Material object with pre-defined properties.
    
    Raises
    ------
    KeyError
        If material not found.
    
    Examples
    --------
    >>> from fibernet.materials import get_material
    >>> steel = get_material('steel')
    >>> print(f"Steel modulus: {steel.youngs_modulus/1e9:.1f} GPa")
    Steel modulus: 210.0 GPa
    """
    db = get_material_database()
    name_lower = name.lower().replace(' ', '_').replace('-', '_')
    
    if name_lower not in db:
        available = list(db.keys())
        raise KeyError(
            f"Material '{name}' not found. Available: {available}"
        )
    
    return db[name_lower]


def list_materials():
    """List all available pre-defined materials.
    
    Returns
    -------
    materials : list
        List of material names.
    
    Examples
    --------
    >>> from fibernet.materials import list_materials
    >>> materials = list_materials()
    >>> print(f"Available: {len(materials)} materials")
    """
    db = get_material_database()
    return sorted(db.keys())


def compare_materials(material_names):
    """Compare properties of multiple materials.
    
    Parameters
    ----------
    material_names : list of str
        List of material names.
    
    Returns
    -------
    comparison : dict
        Dictionary of property comparisons.
    
    Examples
    --------
    >>> from fibernet.materials import compare_materials
    >>> comp = compare_materials(['steel', 'aluminum', 'carbon_fiber'])
    >>> print(f"Stiffest: {comp['stiffest']}")
    """
    db = get_material_database()
    materials = {name: get_material(name) for name in material_names}
    
    # Find extremes
    youngs_moduli = {name: m.youngs_modulus for name, m in materials.items()}
    densities = {name: m.density for name, m in materials.items()}
    
    stiffest = max(youngs_moduli, key=youngs_moduli.get)
    lightest = min(densities, key=densities.get)
    
    # Specific stiffness (E/ρ)
    specific_stiffness = {
        name: m.youngs_modulus / m.density 
        for name, m in materials.items()
    }
    best_specific = max(specific_stiffness, key=specific_stiffness.get)
    
    return {
        'stiffest': stiffest,
        'lightest': lightest,
        'best_specific_stiffness': best_specific,
        'youngs_moduli': youngs_moduli,
        'densities': densities,
        'specific_stiffness': specific_stiffness,
    }


# Convenience function
def m(name: str) -> Material:
    """Shortcut for get_material.
    
    Parameters
    ----------
    name : str
        Material name.
    
    Returns
    -------
    material : Material
        Material object.
    
    Examples
    --------
    >>> from fibernet.materials import m
    >>> steel = m('steel')
    """
    return get_material(name)



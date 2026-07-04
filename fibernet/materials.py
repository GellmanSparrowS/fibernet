"""
Predefined material database for common fiber materials.

Provides ready-to-use Material objects for common fiber types used in research.
All properties are at room temperature (25°C) unless otherwise noted.

References:
- Callister, W.D. "Materials Science and Engineering"
- ASM Handbook
- Various manufacturer datasheets
"""

from .core.material import Material
from typing import Dict, Optional
import numpy as np


def carbon_fiber(grade: str = 'standard') -> Material:
    """
    Carbon fiber material.
    
    Parameters
    ----------
    grade : str
        'standard', 'intermediate', 'high_strength', 'high_modulus'
    
    Returns
    -------
    Material
        Carbon fiber material object
    
    Examples
    --------
    >>> mat = carbon_fiber('high_strength')
    >>> print(mat.youngs_modulus)
    """
    grades = {
        'standard': {
            'E': 230e9,  # Pa
            'sigma_y': 3.5e9,  # Pa
            'density': 1750,  # kg/m³
            'nu': 0.28,
        },
        'intermediate': {
            'E': 294e9,
            'sigma_y': 4.9e9,
            'density': 1800,
            'nu': 0.28,
        },
        'high_strength': {
            'E': 294e9,
            'sigma_y': 7.0e9,
            'density': 1800,
            'nu': 0.28,
        },
        'high_modulus': {
            'E': 588e9,
            'sigma_y': 3.9e9,
            'density': 1900,
            'nu': 0.28,
        },
    }
    
    props = grades.get(grade, grades['standard'])
    
    return Material(
        name=f'Carbon Fiber ({grade})',
        density=props['density'],
        youngs_modulus=props['E'],
        poissons_ratio=props['nu'],
        yield_strength=props['sigma_y'],
        tensile_strength=props['sigma_y'],  # Carbon fiber is brittle
        fracture_toughness=0.5e6,  # Pa·√m
        thermal_conductivity=10.0,  # W/(m·K) along fiber
        specific_heat=710,  # J/(kg·K)
        thermal_expansion=-0.5e-6,  # 1/K (negative along fiber)
        electrical_conductivity=1e5,  # S/m
    )


def glass_fiber(fiber_type: str = 'E-glass') -> Material:
    """
    Glass fiber material.
    
    Parameters
    ----------
    fiber_type : str
        'E-glass', 'S-glass', 'C-glass', 'AR-glass'
    
    Returns
    -------
    Material
        Glass fiber material object
    """
    types = {
        'E-glass': {
            'E': 72.4e9,
            'sigma_y': 3.45e9,
            'density': 2540,
            'nu': 0.22,
        },
        'S-glass': {
            'E': 85.5e9,
            'sigma_y': 4.58e9,
            'density': 2480,
            'nu': 0.22,
        },
        'C-glass': {
            'E': 69e9,
            'sigma_y': 3.1e9,
            'density': 2490,
            'nu': 0.22,
        },
        'AR-glass': {
            'E': 73e9,
            'sigma_y': 3.5e9,
            'density': 2700,
            'nu': 0.22,
        },
    }
    
    props = types.get(fiber_type, types['E-glass'])
    
    return Material(
        name=f'Glass Fiber ({fiber_type})',
        density=props['density'],
        youngs_modulus=props['E'],
        poissons_ratio=props['nu'],
        yield_strength=props['sigma_y'],
        tensile_strength=props['sigma_y'],
        fracture_toughness=0.8e6,
        thermal_conductivity=1.3,
        specific_heat=840,
        thermal_expansion=5.0e-6,
        electrical_conductivity=1e-12,  # Insulator
    )


def aramid_fiber(fiber_type: str = 'Kevlar-49') -> Material:
    """
    Aramid fiber material (Kevlar, Twaron, etc.).
    
    Parameters
    ----------
    fiber_type : str
        'Kevlar-29', 'Kevlar-49', 'Kevlar-129', 'Twaron'
    
    Returns
    -------
    Material
        Aramid fiber material object
    """
    types = {
        'Kevlar-29': {
            'E': 70e9,
            'sigma_y': 2.9e9,
            'density': 1440,
            'nu': 0.36,
        },
        'Kevlar-49': {
            'E': 131e9,
            'sigma_y': 3.6e9,
            'density': 1440,
            'nu': 0.36,
        },
        'Kevlar-129': {
            'E': 100e9,
            'sigma_y': 3.4e9,
            'density': 1440,
            'nu': 0.36,
        },
        'Twaron': {
            'E': 120e9,
            'sigma_y': 3.2e9,
            'density': 1440,
            'nu': 0.35,
        },
    }
    
    props = types.get(fiber_type, types['Kevlar-49'])
    
    return Material(
        name=f'Aramid Fiber ({fiber_type})',
        density=props['density'],
        youngs_modulus=props['E'],
        poissons_ratio=props['nu'],
        yield_strength=props['sigma_y'],
        tensile_strength=props['sigma_y'],
        fracture_toughness=3.0e6,  # High toughness
        thermal_conductivity=0.04,
        specific_heat=1420,
        thermal_expansion=-2.0e-6,
        electrical_conductivity=1e-14,
    )


def collagen_fiber() -> Material:
    """
    Type I collagen fiber (biological fibers).
    
    Based on literature values for tendon collagen.
    
    Returns
    -------
    Material
        Collagen fiber material object
    
    References
    ----------
    - Fratzl et al., J Struct Biol (2004)
    - Svensson et al., PNAS (2017)
    """
    return Material(
        name='Collagen Type I',
        density=1340,  # kg/m³
        youngs_modulus=1.2e9,  # Pa (fibril level)
        poissons_ratio=0.3,
        yield_strength=50e6,  # Pa
        tensile_strength=100e6,  # Pa
        fracture_toughness=1.0e6,
        thermal_conductivity=0.6,
        specific_heat=1500,
        thermal_expansion=50e-6,
        extra={
            'persistence_length': 14e-9,  # m
            'diameter_range': (50e-9, 500e-9),  # m
            'water_content': 0.6,  # fraction
        }
    )


def cellulose_nanofiber() -> Material:
    """
    Cellulose nanofiber (CNF/CNC).
    
    Returns
    -------
    Material
        Cellulose nanofiber material object
    
    References
    ----------
    - Dufresne, Prog Polym Sci (2013)
    - Moon et al., Adv Mater (2011)
    """
    return Material(
        name='Cellulose Nanofiber',
        density=1500,  # kg/m³
        youngs_modulus=150e9,  # Pa (crystalline cellulose)
        poissons_ratio=0.3,
        yield_strength=2.0e9,  # Pa
        tensile_strength=3.0e9,  # Pa
        fracture_toughness=0.5e6,
        thermal_conductivity=0.6,
        specific_heat=1200,
        thermal_expansion=1.0e-6,
        extra={
            'crystallinity': 0.8,
            'diameter_range': (5e-9, 50e-9),  # m
            'aspect_ratio_range': (50, 200),
        }
    )


def spider_silk(silk_type: str = 'dragline') -> Material:
    """
    Spider silk material.
    
    Parameters
    ----------
    silk_type : str
        'dragline', 'capture', 'minor_ampullate'
    
    Returns
    -------
    Material
        Spider silk material object
    
    References
    ----------
    - Gosline et al., J Exp Biol (1999)
    - Vollrath & Porter, Polymer (2006)
    """
    types = {
        'dragline': {
            'E': 10e9,
            'sigma_y': 1.2e9,
            'strain_failure': 0.30,
            'density': 1300,
        },
        'capture': {
            'E': 3e6,
            'sigma_y': 500e6,
            'strain_failure': 5.0,
            'density': 1300,
        },
        'minor_ampullate': {
            'E': 10e9,
            'sigma_y': 800e6,
            'strain_failure': 0.25,
            'density': 1300,
        },
    }
    
    props = types.get(silk_type, types['dragline'])
    
    return Material(
        name=f'Spider Silk ({silk_type})',
        density=props['density'],
        youngs_modulus=props['E'],
        poissons_ratio=0.4,
        yield_strength=props['sigma_y'] * 0.7,
        tensile_strength=props['sigma_y'],
        fracture_toughness=150e6,  # Extremely tough
        thermal_conductivity=0.2,
        specific_heat=1300,
        thermal_expansion=50e-6,
        extra={
            'strain_at_failure': props['strain_failure'],
            'toughness': 0.5 * props['sigma_y'] * props['strain_failure'],  # J/m³
        }
    )


def polymer_fiber(polymer: str = 'nylon') -> Material:
    """
    Synthetic polymer fiber.
    
    Parameters
    ----------
    polymer : str
        'nylon', 'polyester', 'polypropylene', 'UHMWPE'
    
    Returns
    -------
    Material
        Polymer fiber material object
    """
    types = {
        'nylon': {
            'E': 3.5e9,
            'sigma_y': 800e6,
            'density': 1140,
            'nu': 0.4,
        },
        'polyester': {
            'E': 14e9,
            'sigma_y': 700e6,
            'density': 1380,
            'nu': 0.38,
        },
        'polypropylene': {
            'E': 1.5e9,
            'sigma_y': 400e6,
            'density': 900,
            'nu': 0.42,
        },
        'UHMWPE': {
            'E': 100e9,
            'sigma_y': 3.0e9,
            'density': 970,
            'nu': 0.35,
        },
    }
    
    props = types.get(polymer, types['nylon'])
    
    return Material(
        name=f'Polymer Fiber ({polymer})',
        density=props['density'],
        youngs_modulus=props['E'],
        poissons_ratio=props['nu'],
        yield_strength=props['sigma_y'] * 0.8,
        tensile_strength=props['sigma_y'],
        fracture_toughness=5.0e6,
        thermal_conductivity=0.2,
        specific_heat=1500,
        thermal_expansion=100e-6,
    )


def metal_fiber(metal: str = 'steel') -> Material:
    """
    Metal fiber material.
    
    Parameters
    ----------
    metal : str
        'steel', 'aluminum', 'titanium', 'copper'
    
    Returns
    -------
    Material
        Metal fiber material object
    """
    types = {
        'steel': {
            'E': 210e9,
            'sigma_y': 2000e6,
            'density': 7850,
            'nu': 0.3,
        },
        'aluminum': {
            'E': 70e9,
            'sigma_y': 500e6,
            'density': 2700,
            'nu': 0.33,
        },
        'titanium': {
            'E': 110e9,
            'sigma_y': 1200e6,
            'density': 4500,
            'nu': 0.32,
        },
        'copper': {
            'E': 117e9,
            'sigma_y': 400e6,
            'density': 8960,
            'nu': 0.34,
        },
    }
    
    props = types.get(metal, types['steel'])
    
    return Material(
        name=f'Metal Fiber ({metal})',
        density=props['density'],
        youngs_modulus=props['E'],
        poissons_ratio=props['nu'],
        yield_strength=props['sigma_y'] * 0.9,
        tensile_strength=props['sigma_y'],
        fracture_toughness=50e6,
        thermal_conductivity=50.0,
        specific_heat=500,
        thermal_expansion=12e-6,
        electrical_conductivity=1e6,
    )


def basalt_fiber() -> Material:
    """
    Basalt fiber material.
    
    Returns
    -------
    Material
        Basalt fiber material object
    """
    return Material(
        name='Basalt Fiber',
        density=2700,
        youngs_modulus=89e9,
        poissons_ratio=0.22,
        yield_strength=4.0e9,
        tensile_strength=4.8e9,
        fracture_toughness=0.6e6,
        thermal_conductivity=0.4,
        specific_heat=800,
        thermal_expansion=8.0e-6,
        electrical_conductivity=1e-12,
    )


def silica_fiber() -> Material:
    """
    Fused silica (optical fiber) material.
    
    Returns
    -------
    Material
        Silica fiber material object
    """
    return Material(
        name='Silica Fiber',
        density=2200,
        youngs_modulus=72e9,
        poissons_ratio=0.17,
        yield_strength=5.0e9,
        tensile_strength=6.0e9,
        fracture_toughness=0.8e6,
        thermal_conductivity=1.4,
        specific_heat=700,
        thermal_expansion=0.5e-6,
        electrical_conductivity=1e-18,
        extra={
            'refractive_index': 1.46,
            'transmission_range': (0.2e-6, 2.0e-6),  # m
        }
    )


# Material registry for easy access
MATERIAL_DATABASE = {
    'carbon': carbon_fiber,
    'glass': glass_fiber,
    'aramid': aramid_fiber,
    'collagen': collagen_fiber,
    'cellulose': cellulose_nanofiber,
    'spider_silk': spider_silk,
    'polymer': polymer_fiber,
    'metal': metal_fiber,
    'basalt': basalt_fiber,
    'silica': silica_fiber,
}


def get_material(name: str, **kwargs) -> Material:
    """
    Get a predefined material by name.
    
    Parameters
    ----------
    name : str
        Material name (e.g., 'carbon', 'glass', 'collagen')
    **kwargs
        Additional arguments passed to material constructor
    
    Returns
    -------
    Material
        Material object
    
    Examples
    --------
    >>> mat = get_material('carbon', grade='high_strength')
    >>> mat = get_material('glass', fiber_type='S-glass')
    >>> mat = get_material('collagen')
    """
    name_lower = name.lower()
    
    if name_lower not in MATERIAL_DATABASE:
        available = ', '.join(MATERIAL_DATABASE.keys())
        raise ValueError(f"Unknown material '{name}'. Available: {available}")
    
    return MATERIAL_DATABASE[name_lower](**kwargs)


def list_materials() -> Dict[str, str]:
    """
    List all available predefined materials.
    
    Returns
    -------
    dict
        Dictionary of material names and descriptions
    """
    return {
        'carbon': 'Carbon fiber (standard, intermediate, high_strength, high_modulus)',
        'glass': 'Glass fiber (E-glass, S-glass, C-glass, AR-glass)',
        'aramid': 'Aramid fiber (Kevlar-29, Kevlar-49, Kevlar-129, Twaron)',
        'collagen': 'Type I collagen (biological)',
        'cellulose': 'Cellulose nanofiber (CNF/CNC)',
        'spider_silk': 'Spider silk (dragline, capture, minor_ampullate)',
        'polymer': 'Polymer fiber (nylon, polyester, polypropylene, UHMWPE)',
        'metal': 'Metal fiber (steel, aluminum, titanium, copper)',
        'basalt': 'Basalt fiber',
        'silica': 'Silica fiber (optical)',
    }

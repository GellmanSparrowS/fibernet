"""
Copy utilities for FiberNet core classes.

Provides deep copy functionality for safe object duplication.
"""

import numpy as np
from typing import Optional
from .fiber import Fiber
from .material import Material
from .network import FiberNetwork, Crosslink


def copy_fiber(fiber: Fiber, new_id: Optional[int] = None) -> Fiber:
    """
    Create a deep copy of a Fiber object.
    
    Parameters
    ----------
    fiber : Fiber
        The fiber to copy.
    new_id : int, optional
        New fiber ID. If None, uses the original ID.
    
    Returns
    -------
    Fiber
        Deep copy of the fiber.
    """
    return Fiber(
        centerline=fiber.centerline.copy(),
        radius=fiber.radius,
        material=fiber.material,  # Material is immutable, share reference
        fiber_id=new_id if new_id is not None else fiber.fiber_id
    )


def copy_material(material: Material) -> Material:
    """
    Create a deep copy of a Material object.
    
    Parameters
    ----------
    material : Material
        The material to copy.
    
    Returns
    -------
    Material
        Deep copy of the material.
    """
    return Material(
        name=material.name,
        density=material.density,
        youngs_modulus=material.youngs_modulus,
        poissons_ratio=material.poissons_ratio,
        shear_modulus=material.shear_modulus,
        yield_strength=material.yield_strength,
        tensile_strength=material.tensile_strength,
        fracture_toughness=material.fracture_toughness,
        thermal_conductivity=material.thermal_conductivity,
        specific_heat=material.specific_heat,
        thermal_expansion=material.thermal_expansion,
        electrical_conductivity=material.electrical_conductivity,
        permittivity=material.permittivity,
        permeability=material.permeability,
        stiffness_tensor=material.stiffness_tensor,
        extra=material.extra.copy() if material.extra else None
    )


def copy_network(network: FiberNetwork) -> FiberNetwork:
    """
    Create a deep copy of a FiberNetwork object.
    
    Parameters
    ----------
    network : FiberNetwork
        The network to copy.
    
    Returns
    -------
    FiberNetwork
        Deep copy of the network with all fibers and crosslinks duplicated.
    """
    new_net = FiberNetwork()
    
    # Copy fibers with new IDs
    for i, fiber in enumerate(network.fibers):
        new_fiber = copy_fiber(fiber, new_id=i)
        new_net.add_fiber(new_fiber)
    
    # Copy crosslinks
    for cl in network.crosslinks:
        new_cl = Crosslink(
            fiber_i=cl.fiber_i,
            fiber_j=cl.fiber_j,
            param_i=cl.param_i,
            param_j=cl.param_j,
            position=cl.position.copy(),
            crosslink_type=cl.crosslink_type,
            strength=cl.strength,
            stiffness=cl.stiffness
        )
        new_net.crosslinks.append(new_cl)
    
    # Copy metadata
    new_net.metadata = network.metadata.copy() if network.metadata else {}
    
    return new_net


# Add copy methods to classes
def _fiber_copy(self, new_id=None):
    return copy_fiber(self, new_id)

def _material_copy(self):
    return copy_material(self)

def _network_copy(self):
    return copy_network(self)


# Monkey-patch the classes
Fiber.copy = _fiber_copy
Material.copy = _material_copy
FiberNetwork.copy = _network_copy

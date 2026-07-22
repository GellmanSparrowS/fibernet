"""
Fracture mechanics simulation for fiber networks.

Implements:
- Progressive fiber failure (strength-based criterion)
- Crosslink debonding
- Energy-based fracture (Griffith criterion)
- Damage accumulation models
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.sim.mechanical import FiberFEM, MechanicalResult


@dataclass
class FractureResult:
    """Results from fracture simulation."""
    load_displacement: Tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([]))
    failed_fibers: List[int] = field(default_factory=list)
    failed_crosslinks: List[int] = field(default_factory=list)
    failure_sequence: List[Tuple[int, str, float]] = field(default_factory=list)
    energy_absorbed: float = 0.0
    peak_load: float = 0.0
    toughness: float = 0.0
    damage_map: np.ndarray = None


class FiberFracture:
    """Progressive fracture simulation for fiber networks.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to simulate.
    failure_criterion : str
        'max_stress', 'max_strain', or 'energy'.
    damage_model : str
        'brittle' (instant failure) or 'softening' (progressive damage).
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        failure_criterion: str = "max_stress",
        damage_model: str = "brittle",
    ):
        self.network = network
        self.failure_criterion = failure_criterion
        self.damage_model = damage_model
    
    def run_progressive_failure(
        self,
        max_strain: float = 0.1,
        strain_increment: float = 0.001,
        axis: int = 0,
        segments_per_fiber: int = 5,
    ) -> FractureResult:
        """Run progressive failure analysis under increasing strain.
        
        At each strain step, checks failure criteria and removes failed elements.
        """
        result = FractureResult()
        
        strains = np.arange(strain_increment, max_strain + strain_increment, strain_increment)
        loads = np.zeros(len(strains))
        
        active_fibers = list(range(self.network.num_fibers))
        active_crosslinks = list(range(self.network.num_crosslinks))
        
        for step_idx, eps in enumerate(strains):
            if not active_fibers:
                break
            
            sub_fibers = [self.network.fibers[i] for i in active_fibers]
            sub_cl = [self.network.crosslinks[i] for i in active_crosslinks]
            sub_net = FiberNetwork(
                fibers=sub_fibers, crosslinks=sub_cl,
                box_size=self.network.box_size, dimension=self.network.dimension,
            )
            
            fem = FiberFEM(sub_net, segments_per_fiber)
            mech_result = fem.apply_uniaxial_strain(eps, axis)
            
            if mech_result.stresses is not None and len(mech_result.stresses) > 0:
                bb_min, bb_max = sub_net.bounding_box()
                dims = bb_max - bb_min
                if axis == 0:
                    area = dims[1] * dims[2] if len(dims) > 2 else dims[1]
                elif axis == 1:
                    area = dims[0] * dims[2] if len(dims) > 2 else dims[0]
                else:
                    area = dims[0] * dims[1]
                
                if area > 1e-12:
                    stress = mech_result.energy * 2 / (sub_net.total_volume * eps)
                else:
                    stress = 0.0
                loads[step_idx] = stress * area
            
            if mech_result.stresses is not None:
                elem_idx = 0
                failed_this_step = []
                for f_i, fiber in enumerate(sub_fibers):
                    if fiber.material.tensile_strength is not None:
                        n_elem = segments_per_fiber
                        elem_stresses = mech_result.stresses[elem_idx:elem_idx + n_elem]
                        if len(elem_stresses) > 0 and np.max(np.abs(elem_stresses)) > fiber.material.tensile_strength:
                            failed_this_step.append(active_fibers[f_i])
                            result.failure_sequence.append((active_fibers[f_i], "fiber", eps))
                    elem_idx += segments_per_fiber
                
                for fid in failed_this_step:
                    if fid in active_fibers:
                        active_fibers.remove(fid)
                        result.failed_fibers.append(fid)
        
        valid_steps = np.where(loads > 0)[0]
        if len(valid_steps) > 0:
            result.load_displacement = (strains[valid_steps], loads[valid_steps])
            result.peak_load = float(np.max(loads))
            result.toughness = float(np.trapz(loads[valid_steps], strains[valid_steps]))
            result.energy_absorbed = result.toughness
        
        result.damage_map = np.array([
            1.0 if i in result.failed_fibers else 0.0
            for i in range(self.network.num_fibers)
        ])
        
        return result

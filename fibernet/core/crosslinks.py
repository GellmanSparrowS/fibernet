"""
Advanced crosslink models for fiber networks.

Provides different crosslink mechanical behaviors:
- RigidCrosslink: Perfectly rigid connections
- SpringCrosslink: Linear spring connections
- BreakableCrosslink: Force-dependent rupture
- FrictionCrosslink: Friction-based sliding
- BondedCrosslink: Covalent-like with bending resistance
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class CrosslinkState:
    """State of a crosslink during simulation."""
    force: np.ndarray = None
    displacement: np.ndarray = None
    broken: bool = False
    slip_displacement: float = 0.0
    num_cycles: int = 0


class CrosslinkModel(ABC):
    """Base class for crosslink models."""
    
    @abstractmethod
    def compute_force(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        state: CrosslinkState,
    ) -> Tuple[np.ndarray, CrosslinkState]:
        """Compute crosslink force and update state."""
        pass
    
    @abstractmethod
    def energy(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute crosslink potential energy."""
        pass


class RigidCrosslink(CrosslinkModel):
    """Rigid crosslink (infinite stiffness constraint).
    
    Enforces p1 == p2 exactly.
    """
    
    def __init__(self):
        pass
    
    def compute_force(self, p1, p2, state):
        dr = np.asarray(p2) - np.asarray(p1)
        # Return constraint force (Lagrange multiplier approach)
        if state.force is None:
            force = np.zeros(3)
        else:
            force = state.force
        return force, CrosslinkState(force=force, displacement=dr)
    
    def energy(self, p1, p2):
        return 0.0


class SpringCrosslink(CrosslinkModel):
    """Linear spring crosslink.
    
    F = k * (|r| - r0) * r_hat
    
    Parameters
    ----------
    stiffness : float
        Spring stiffness (N/m).
    rest_length : float
        Rest length. Default 0 (point connection).
    """
    
    def __init__(self, stiffness: float = 1e6, rest_length: float = 0.0):
        self.k = stiffness
        self.r0 = rest_length
    
    def compute_force(self, p1, p2, state):
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        dr = p2 - p1
        r = np.linalg.norm(dr)
        
        if r < 1e-15:
            force = np.zeros(3)
        else:
            r_hat = dr / r
            force = self.k * (r - self.r0) * r_hat
        
        new_state = CrosslinkState(force=force, displacement=dr)
        return force, new_state
    
    def energy(self, p1, p2):
        dr = np.asarray(p2) - np.asarray(p1)
        r = np.linalg.norm(dr)
        return 0.5 * self.k * (r - self.r0)**2


class BreakableCrosslink(CrosslinkModel):
    """Breakable crosslink with force-dependent rupture.
    
    Follows Bell-like bond model:
    - Below critical force: behaves as spring
    - Above critical force: bond breaks irreversibly
    
    Parameters
    ----------
    stiffness : float
        Spring stiffness.
    critical_force : float
        Force at which bond breaks.
    rest_length : float
        Rest length.
    """
    
    def __init__(self, stiffness: float = 1e6, critical_force: float = 1e3, rest_length: float = 0.0):
        self.k = stiffness
        self.f_crit = critical_force
        self.r0 = rest_length
    
    def compute_force(self, p1, p2, state):
        if state.broken:
            return np.zeros(3), state
        
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        dr = p2 - p1
        r = np.linalg.norm(dr)
        
        if r < 1e-15:
            force = np.zeros(3)
        else:
            r_hat = dr / r
            force_mag = self.k * (r - self.r0)
            force = force_mag * r_hat
        
        f_norm = np.linalg.norm(force)
        broken = f_norm > self.f_crit
        
        new_state = CrosslinkState(
            force=force, displacement=dr, broken=broken,
        )
        return force, new_state
    
    def energy(self, p1, p2):
        dr = np.asarray(p2) - np.asarray(p1)
        r = np.linalg.norm(dr)
        return 0.5 * self.k * (r - self.r0)**2


class FrictionCrosslink(CrosslinkModel):
    """Friction-based crosslink with slip.
    
    - Below friction force: elastic spring
    - Above friction force: slips with constant friction force
    
    Parameters
    ----------
    stiffness : float
        Spring stiffness.
    friction_force : float
        Maximum friction force.
    """
    
    def __init__(self, stiffness: float = 1e6, friction_force: float = 100.0):
        self.k = stiffness
        self.f_friction = friction_force
    
    def compute_force(self, p1, p2, state):
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        
        dr = p2 - p1
        slip = state.slip_displacement
        
        elastic_disp = dr - slip * (dr / (np.linalg.norm(dr) + 1e-15))
        r_elastic = np.linalg.norm(elastic_disp)
        
        if r_elastic < 1e-15:
            force = np.zeros(3)
        else:
            force_mag = min(self.k * r_elastic, self.f_friction)
            force = force_mag * elastic_disp / r_elastic
        
        # Update slip if friction is exceeded
        if self.k * r_elastic > self.f_friction:
            slip_new = np.linalg.norm(dr) - self.f_friction / self.k
        else:
            slip_new = slip
        
        new_state = CrosslinkState(
            force=force, displacement=dr,
            slip_displacement=slip_new,
        )
        return force, new_state
    
    def energy(self, p1, p2):
        dr = np.asarray(p2) - np.asarray(p1)
        r = np.linalg.norm(dr)
        return min(0.5 * self.k * r**2, self.f_friction * r)


class BondedCrosslink(CrosslinkModel):
    """Bonded crosslink with stretching and bending resistance.
    
    Models covalent-like bonds with:
    - Stretching energy: 0.5 * k_s * (r - r0)²
    - Bending energy: 0.5 * k_b * theta² (requires neighbor info)
    
    Parameters
    ----------
    stretch_stiffness : float
        Stretching stiffness.
    bend_stiffness : float
        Bending stiffness.
    rest_length : float
        Rest length.
    """
    
    def __init__(self, stretch_stiffness: float = 1e6, bend_stiffness: float = 1e4, rest_length: float = 0.0):
        self.k_s = stretch_stiffness
        self.k_b = bend_stiffness
        self.r0 = rest_length
    
    def compute_force(self, p1, p2, state):
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        dr = p2 - p1
        r = np.linalg.norm(dr)
        
        if r < 1e-15:
            force = np.zeros(3)
        else:
            r_hat = dr / r
            force = self.k_s * (r - self.r0) * r_hat
        
        new_state = CrosslinkState(force=force, displacement=dr)
        return force, new_state
    
    def energy(self, p1, p2):
        dr = np.asarray(p2) - np.asarray(p1)
        r = np.linalg.norm(dr)
        return 0.5 * self.k_s * (r - self.r0)**2
    
    def bending_energy(self, p1, p2, p3):
        """Compute bending energy for three-point angle."""
        v1 = np.asarray(p1) - np.asarray(p2)
        v2 = np.asarray(p3) - np.asarray(p2)
        
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        
        if n1 < 1e-15 or n2 < 1e-15:
            return 0.0
        
        cos_theta = np.dot(v1, v2) / (n1 * n2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # Bending energy around rest angle of pi (straight)
        return 0.5 * self.k_b * (theta - np.pi)**2

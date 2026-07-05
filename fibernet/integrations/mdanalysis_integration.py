"""
MDAnalysis Integration for FiberNet

Provides integration with MDAnalysis for trajectory analysis:
- Convert FiberNetwork to MDAnalysis Universe
- RMSD calculation for fiber deformation
- Radius of gyration for network compactness
- End-to-end distance distribution
- Contact analysis between fibers

MDAnalysis is GPL v2 licensed: https://github.com/MDAnalysis/mdanalysis
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

from fibernet.core.network import FiberNetwork


@dataclass
class MDAnalysisResult:
    """Result of MDAnalysis trajectory analysis."""
    rmsd: float = 0.0
    radius_of_gyration: float = 0.0
    end_to_end_distances: np.ndarray = None
    num_frames: int = 0
    contact_matrix: np.ndarray = None
    
    def to_dict(self) -> Dict:
        return {
            'rmsd': self.rmsd,
            'radius_of_gyration': self.radius_of_gyration,
            'num_frames': self.num_frames,
            'e2e_mean': float(np.mean(self.end_to_end_distances)) if self.end_to_end_distances is not None else 0.0,
            'e2e_std': float(np.std(self.end_to_end_distances)) if self.end_to_end_distances is not None else 0.0,
        }


class MDAnalysisBridge:
    """Bridge between FiberNetwork and MDAnalysis.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    """
    
    def __init__(self, network: FiberNetwork):
        self.network = network
        try:
            import MDAnalysis as mda
            from MDAnalysis.topology.base import Topology
            self.mda = mda
        except ImportError:
            raise ImportError(
                "MDAnalysis required. Install with: pip install MDAnalysis"
            )
    
    def to_universe(self):
        """Convert FiberNetwork to MDAnalysis Universe.
        
        Returns
        -------
        u : MDAnalysis.Universe
            MDAnalysis Universe with fibers as residues.
        """
        # Collect all coordinates and fiber assignments
        coords_list = []
        resids_list = []
        resnames_list = []
        
        for i, fiber in enumerate(self.network.fibers):
            cl = fiber.centerline
            n_points = len(cl)
            coords_list.append(cl)
            resids_list.extend([i] * n_points)
            resnames_list.extend([f"FIB{i}"] * n_points)
        
        all_coords = np.vstack(coords_list)
        n_atoms = len(all_coords)
        
        # Build universe from coordinates
        u = self.mda.Universe.empty(
            n_atoms,
            n_residues=len(self.network.fibers),
            atom_resindex=resids_list,
            trajectory=True,
        )
        
        u.add_TopologyAttr('name', ['C'] * n_atoms)
        u.add_TopologyAttr('type', ['C'] * n_atoms)
        u.add_TopologyAttr('resname', [f"FIB{i}" for i in range(len(self.network.fibers))])
        u.add_TopologyAttr('resid', list(range(1, len(self.network.fibers) + 1)))
        
        u.atoms.positions = all_coords
        
        return u
    
    def analyze_structure(self, u=None) -> MDAnalysisResult:
        """Analyze fiber network structure.
        
        Parameters
        ----------
        u : MDAnalysis.Universe, optional
            Pre-computed Universe.
        
        Returns
        -------
        result : MDAnalysisResult
            Analysis results.
        """
        if u is None:
            u = self.to_universe()
        
        positions = u.atoms.positions
        n_residues = len(u.residues)
        
        # RMSD (from mean positions as reference)
        mean_pos = positions.mean(axis=0)
        rmsd = float(np.sqrt(np.mean(np.sum((positions - mean_pos)**2, axis=1))))
        
        # Radius of gyration
        com = positions.mean(axis=0)
        distances_from_com = np.linalg.norm(positions - com, axis=1)
        rg = float(np.sqrt(np.mean(distances_from_com**2)))
        
        # End-to-end distances per fiber
        e2e = []
        for res in u.residues:
            pos = res.atoms.positions
            if len(pos) >= 2:
                dist = np.linalg.norm(pos[-1] - pos[0])
                e2e.append(dist)
        
        return MDAnalysisResult(
            rmsd=rmsd,
            radius_of_gyration=rg,
            end_to_end_distances=np.array(e2e),
            num_frames=1,
        )
    
    def compute_contacts(self, u=None, cutoff: float = 5.0) -> np.ndarray:
        """Compute contact matrix between fibers.
        
        Parameters
        ----------
        u : MDAnalysis.Universe, optional
            Pre-computed Universe.
        cutoff : float
            Contact distance cutoff.
        
        Returns
        -------
        contacts : np.ndarray
            Contact matrix (num_fibers, num_fibers).
        """
        if u is None:
            u = self.to_universe()
        
        centers = []
        for res in u.residues:
            centers.append(res.atoms.center_of_mass())
        centers = np.array(centers)
        
        n = len(centers)
        contacts = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < cutoff:
                    contacts[i, j] = 1.0
                    contacts[j, i] = 1.0
        
        return contacts
    
    def compute_msd(self, positions_frames: List[np.ndarray], dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean squared displacement from trajectory frames.
        
        Parameters
        ----------
        positions_frames : list of ndarray
            List of position arrays for each frame.
        dt : float
            Time step between frames.
        
        Returns
        -------
        times : np.ndarray
            Time values.
        msd : np.ndarray
            MSD values.
        """
        n_frames = len(positions_frames)
        if n_frames < 2:
            return np.array([0.0]), np.array([0.0])
        
        ref = positions_frames[0]
        times = []
        msd_values = []
        
        for lag in range(1, n_frames):
            if lag < len(positions_frames):
                disp = positions_frames[lag] - ref
                msd_val = np.mean(np.sum(disp**2, axis=1))
                times.append(lag * dt)
                msd_values.append(msd_val)
        
        return np.array(times), np.array(msd_values)


def analyze_fiber_dynamics(network: FiberNetwork, **kwargs) -> MDAnalysisResult:
    """Convenience function for fiber dynamics analysis.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    
    Returns
    -------
    result : MDAnalysisResult
        Analysis results.
    """
    bridge = MDAnalysisBridge(network)
    return bridge.analyze_structure(**kwargs)

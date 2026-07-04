"""
Stress-strain curve extraction and analysis.

Provides tools for generating stress-strain curves from mechanical simulations
and analyzing key mechanical properties.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field
from ..core.network import FiberNetwork
from ..sim.mechanical import MechanicalResult


@dataclass
class StressStrainCurve:
    """
    Container for stress-strain curve data.
    
    Attributes
    ----------
    strain : np.ndarray
        Strain values (dimensionless)
    stress : np.ndarray
        Stress values (Pa)
    energy : np.ndarray
        Strain energy density (J/m³)
    metadata : dict
        Additional metadata (loading direction, temperature, etc.)
    """
    strain: np.ndarray
    stress: np.ndarray
    energy: np.ndarray
    metadata: Dict = field(default_factory=dict)
    
    @property
    def youngs_modulus(self) -> float:
        """Calculate Young's modulus from initial linear region."""
        # Use first 10% of curve for linear fit
        n_points = max(3, len(self.strain) // 10)
        strain_linear = self.strain[:n_points]
        stress_linear = self.stress[:n_points]
        
        if len(strain_linear) < 2:
            return 0.0
        
        # Linear regression
        A = np.vstack([strain_linear, np.ones(len(strain_linear))]).T
        slope, intercept = np.linalg.lstsq(A, stress_linear, rcond=None)[0]
        return slope
    
    @property
    def yield_strength(self) -> float:
        """Calculate 0.2% offset yield strength."""
        if len(self.strain) < 3:
            return 0.0
        
        # 0.2% offset line
        offset_strain = 0.002
        E = self.youngs_modulus
        offset_stress = E * (self.strain - offset_strain)
        
        # Find intersection
        for i in range(1, len(self.strain)):
            if self.stress[i] < offset_stress[i]:
                # Linear interpolation
                if i > 0:
                    frac = (self.stress[i-1] - offset_stress[i-1]) / \
                           (offset_stress[i] - offset_stress[i] - self.stress[i] + self.stress[i-1])
                    return self.stress[i-1] + frac * (self.stress[i] - self.stress[i-1])
                break
        
        return self.stress[-1]
    
    @property
    def ultimate_strength(self) -> float:
        """Maximum stress in the curve."""
        return np.max(self.stress) if len(self.stress) > 0 else 0.0
    
    @property
    def fracture_strain(self) -> float:
        """Strain at fracture (last point)."""
        return self.strain[-1] if len(self.strain) > 0 else 0.0
    
    @property
    def toughness(self) -> float:
        """Area under the curve (energy to fracture)."""
        return np.trapezoid(self.stress, self.strain)
    
    @property
    def resilience(self) -> float:
        """Area under elastic region (energy to yield)."""
        # Find yield point
        yield_strain = self.yield_strength / self.youngs_modulus if self.youngs_modulus > 0 else 0
        
        # Integrate up to yield
        mask = self.strain <= yield_strain
        if np.any(mask):
            return np.trapezoid(self.stress[mask], self.strain[mask])
        return 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with properties."""
        return {
            'strain': self.strain,
            'stress': self.stress,
            'energy': self.energy,
            'youngs_modulus': self.youngs_modulus,
            'yield_strength': self.yield_strength,
            'ultimate_strength': self.ultimate_strength,
            'fracture_strain': self.fracture_strain,
            'toughness': self.toughness,
            'resilience': self.resilience,
            'metadata': self.metadata,
        }
    
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        try:
            import pandas as pd
            return pd.DataFrame({
                'strain': self.strain,
                'stress': self.stress,
                'energy': self.energy,
            })
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")
    
    def plot(self, ax=None, **kwargs):
        """
        Plot stress-strain curve.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        **kwargs
            Passed to matplotlib plot
        
        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(self.strain, self.stress / 1e6, **kwargs)
        ax.set_xlabel('Strain')
        ax.set_ylabel('Stress (MPa)')
        ax.set_title('Stress-Strain Curve')
        ax.grid(True, alpha=0.3)
        
        # Add key properties as text
        text = f'E = {self.youngs_modulus/1e9:.1f} GPa\n'
        text += f'σ_y = {self.yield_strength/1e6:.0f} MPa\n'
        text += f'σ_u = {self.ultimate_strength/1e6:.0f} MPa\n'
        text += f'ε_f = {self.fracture_strain:.3f}'
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return ax


def extract_stress_strain(
    network: FiberNetwork,
    strain_range: Tuple[float, float] = (0.0, 0.1),
    num_steps: int = 20,
    axis: int = 0,
    segments_per_fiber: int = 5,
    **kwargs
) -> StressStrainCurve:
    """
    Extract stress-strain curve by running incremental strain simulation.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to test
    strain_range : tuple
        (min_strain, max_strain)
    num_steps : int
        Number of strain increments
    axis : int
        Loading axis (0=x, 1=y, 2=z)
    segments_per_fiber : int
        FEM discretization
    **kwargs
        Additional arguments passed to FiberFEM
    
    Returns
    -------
    StressStrainCurve
        Stress-strain curve data
    
    Examples
    --------
    >>> net = gen.square_lattice_2d(spacing=5, grid_size=(10, 10))
    >>> curve = extract_stress_strain(net, strain_range=(0, 0.05), num_steps=10)
    >>> print(f"Young's modulus: {curve.youngs_modulus/1e9:.1f} GPa")
    >>> curve.plot()
    """
    from ..sim.mechanical import FiberFEM
    
    strain_values = np.linspace(strain_range[0], strain_range[1], num_steps)
    stress_values = []
    energy_values = []
    
    # Calculate reference volume
    bbox = network.bounding_box()
    if network.dimension == 2:
        volume = (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])
    else:
        volume = (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1]) * (bbox[1][2] - bbox[0][2])
    
    for strain in strain_values:
        try:
            fem = FiberFEM(network, segments_per_fiber=segments_per_fiber)
            result = fem.apply_uniaxial_strain(strain=strain, axis=axis, **kwargs)
            
            # Calculate stress from energy
            if result.energy is not None and volume > 0:
                stress = 2 * result.energy / volume  # σ = dU/dε ≈ 2U/ε for linear
                stress_values.append(stress)
                energy_values.append(result.energy / volume)
            else:
                stress_values.append(0.0)
                energy_values.append(0.0)
        except Exception as e:
            # If simulation fails, use previous values
            if stress_values:
                stress_values.append(stress_values[-1])
                energy_values.append(energy_values[-1])
            else:
                stress_values.append(0.0)
                energy_values.append(0.0)
    
    return StressStrainCurve(
        strain=strain_values,
        stress=np.array(stress_values),
        energy=np.array(energy_values),
        metadata={
            'axis': axis,
            'segments_per_fiber': segments_per_fiber,
            'volume': volume,
            'num_fibers': network.num_fibers,
        }
    )


def compare_curves(
    curves: List[StressStrainCurve],
    labels: List[str],
    ax=None,
    **kwargs
):
    """
    Compare multiple stress-strain curves on one plot.
    
    Parameters
    ----------
    curves : list of StressStrainCurve
        Curves to compare
    labels : list of str
        Labels for each curve
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    **kwargs
        Passed to matplotlib plot
    
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    for curve, label in zip(curves, labels):
        ax.plot(curve.strain, curve.stress / 1e6, label=label, **kwargs)
    
    ax.set_xlabel('Strain')
    ax.set_ylabel('Stress (MPa)')
    ax.set_title('Stress-Strain Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax

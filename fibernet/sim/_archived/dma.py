"""
Dynamic Mechanical Analysis (DMA) simulations.

Provides frequency and temperature sweep simulations for viscoelastic
material characterization, commonly used in polymer and biological fiber research.

Key outputs:
- Storage modulus (E'): Elastic response
- Loss modulus (E''): Viscous response
- Loss tangent (tan δ): Damping factor
- Complex modulus (|E*|): Overall stiffness
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field
from .viscoelastic import GeneralizedMaxwell


@dataclass
class DMAResult:
    """
    Container for DMA simulation results.
    
    Attributes
    ----------
    frequency : np.ndarray
        Angular frequency (rad/s) or frequency (Hz)
    temperature : np.ndarray
        Temperature values (K)
    storage_modulus : np.ndarray
        E' - elastic response (Pa)
    loss_modulus : np.ndarray
        E'' - viscous response (Pa)
    loss_tangent : np.ndarray
        tan(δ) = E''/E' - damping factor
    complex_modulus : np.ndarray
        |E*| = sqrt(E'^2 + E''^2)
    phase_angle : np.ndarray
        δ = arctan(E''/E') (radians)
    metadata : dict
        Simulation parameters and metadata
    """
    frequency: np.ndarray
    temperature: np.ndarray
    storage_modulus: np.ndarray
    loss_modulus: np.ndarray
    loss_tangent: np.ndarray
    complex_modulus: np.ndarray
    phase_angle: np.ndarray
    metadata: Dict = field(default_factory=dict)
    
    @property
    def glass_transition_temperature(self) -> Optional[float]:
        """
        Estimate glass transition temperature (Tg) from tan(δ) peak.
        
        Returns
        -------
        float or None
            Temperature at peak tan(δ), or None if not applicable
        """
        if len(self.temperature) > 1 and len(self.loss_tangent) > 0:
            # Find peak in tan(δ) vs temperature
            peak_idx = np.argmax(self.loss_tangent)
            return float(self.temperature[peak_idx])
        return None
    
    @property
    def crossover_frequency(self) -> Optional[float]:
        """
        Find frequency where E' = E'' (crossover point).
        
        Returns
        -------
        float or None
            Crossover frequency, or None if not found
        """
        if len(self.frequency) < 2:
            return None
        
        # Find where storage_modulus crosses loss_modulus
        diff = self.storage_modulus - self.loss_modulus
        for i in range(len(diff) - 1):
            if diff[i] * diff[i+1] < 0:  # Sign change
                # Linear interpolation
                frac = abs(diff[i]) / (abs(diff[i]) + abs(diff[i+1]))
                return float(self.frequency[i] + frac * (self.frequency[i+1] - self.frequency[i]))
        
        return None
    
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        try:
            import pandas as pd
            return pd.DataFrame({
                'frequency': self.frequency,
                'temperature': self.temperature,
                'storage_modulus': self.storage_modulus,
                'loss_modulus': self.loss_modulus,
                'loss_tangent': self.loss_tangent,
                'complex_modulus': self.complex_modulus,
                'phase_angle': self.phase_angle,
            })
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")
    
    def plot(self, plot_type: str = 'modulus', ax=None, **kwargs):
        """
        Plot DMA results.
        
        Parameters
        ----------
        plot_type : str
            'modulus' (E', E''), 'tangent' (tan δ), 'complex' (|E*|),
            'cole_cole' (E'' vs E')
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
        
        x_data = self.frequency if len(self.frequency) > 1 else self.temperature
        x_label = 'Frequency (rad/s)' if len(self.frequency) > 1 else 'Temperature (K)'
        
        if plot_type == 'modulus':
            ax.plot(x_data, self.storage_modulus / 1e6, label="E' (Storage)", **kwargs)
            ax.plot(x_data, self.loss_modulus / 1e6, label="E'' (Loss)", **kwargs)
            ax.set_ylabel('Modulus (MPa)')
            ax.legend()
        
        elif plot_type == 'tangent':
            ax.plot(x_data, self.loss_tangent, label='tan(δ)', **kwargs)
            ax.set_ylabel('Loss Tangent (tan δ)')
            ax.legend()
            
            # Mark Tg or crossover
            if self.glass_transition_temperature is not None and len(self.temperature) > 1:
                ax.axvline(self.glass_transition_temperature, color='red', 
                          linestyle='--', alpha=0.5, 
                          label=f'Tg ≈ {self.glass_transition_temperature:.1f} K')
                ax.legend()
        
        elif plot_type == 'complex':
            ax.plot(x_data, self.complex_modulus / 1e6, label='|E*|', **kwargs)
            ax.set_ylabel('Complex Modulus (MPa)')
            ax.legend()
        
        elif plot_type == 'cole_cole':
            ax.plot(self.storage_modulus / 1e6, self.loss_modulus / 1e6, **kwargs)
            ax.set_xlabel("E' (Storage Modulus, MPa)")
            ax.set_ylabel("E'' (Loss Modulus, MPa)")
            ax.set_title('Cole-Cole Plot')
        
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.3)
        
        return ax


def frequency_sweep(
    model: GeneralizedMaxwell,
    freq_range: Tuple[float, float] = (0.1, 100),
    num_points: int = 50,
    temperature: float = 298.15,
    use_log_scale: bool = True,
) -> DMAResult:
    """
    Perform frequency sweep DMA simulation.
    
    Parameters
    ----------
    model : GeneralizedMaxwell
        Viscoelastic model (Prony series)
    freq_range : tuple
        (omega_min, omega_max) in rad/s
    num_points : int
        Number of frequency points
    temperature : float
        Temperature (K) - for metadata
    use_log_scale : bool
        Use logarithmic frequency spacing
    
    Returns
    -------
    DMAResult
        DMA results
    
    Examples
    --------
    >>> model = GeneralizedMaxwell(E_inf=1e9, E_i=[5e8, 3e8], tau_i=[0.1, 1.0])
    >>> result = frequency_sweep(model, freq_range=(0.1, 100), num_points=50)
    >>> result.plot('modulus')
    >>> print(f"Crossover frequency: {result.crossover_frequency:.2f} rad/s")
    """
    if use_log_scale:
        omega = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), num_points)
    else:
        omega = np.linspace(freq_range[0], freq_range[1], num_points)
    
    # Calculate E'(ω) and E''(ω) from Prony series
    # E'(ω) = E_inf + Σ E_i * (ωτ_i)^2 / (1 + (ωτ_i)^2)
    # E''(ω) = Σ E_i * ωτ_i / (1 + (ωτ_i)^2)
    
    E_prime = np.full_like(omega, model.E_inf)
    E_double_prime = np.zeros_like(omega)
    
    for E_i, tau_i in zip(model.E_i, model.tau_i):
        omega_tau = omega * tau_i
        E_prime += E_i * omega_tau**2 / (1 + omega_tau**2)
        E_double_prime += E_i * omega_tau / (1 + omega_tau**2)
    
    # Calculate derived quantities
    loss_tangent = E_double_prime / E_prime
    complex_modulus = np.sqrt(E_prime**2 + E_double_prime**2)
    phase_angle = np.arctan2(E_double_prime, E_prime)
    
    return DMAResult(
        frequency=omega,
        temperature=np.full_like(omega, temperature),
        storage_modulus=E_prime,
        loss_modulus=E_double_prime,
        loss_tangent=loss_tangent,
        complex_modulus=complex_modulus,
        phase_angle=phase_angle,
        metadata={
            'sweep_type': 'frequency',
            'temperature': temperature,
            'model': 'GeneralizedMaxwell',
            'E_inf': model.E_inf,
            'E_i': model.E_i.tolist(),
            'tau_i': model.tau_i.tolist(),
        }
    )


def temperature_sweep(
    model: GeneralizedMaxwell,
    temp_range: Tuple[float, float] = (200, 400),
    num_points: int = 50,
    frequency: float = 1.0,
    activation_energy: float = 50e3,
    reference_temp: float = 298.15,
) -> DMAResult:
    """
    Perform temperature sweep DMA simulation.
    
    Uses Arrhenius time-temperature superposition to shift relaxation times.
    
    Parameters
    ----------
    model : GeneralizedMaxwell
        Viscoelastic model
    temp_range : tuple
        (T_min, T_max) in K
    num_points : int
        Number of temperature points
    frequency : float
        Angular frequency (rad/s)
    activation_energy : float
        Activation energy for shift factor (J/mol)
    reference_temp : float
        Reference temperature for shift factor (K)
    
    Returns
    -------
    DMAResult
        DMA results
    
    Examples
    --------
    >>> model = GeneralizedMaxwell(E_inf=1e9, E_i=[5e8, 3e8], tau_i=[0.1, 1.0])
    >>> result = temperature_sweep(model, temp_range=(250, 350), frequency=1.0)
    >>> result.plot('tangent')
    >>> print(f"Glass transition: {result.glass_transition_temperature:.1f} K")
    """
    R = 8.314  # Gas constant (J/(mol·K))
    temperature = np.linspace(temp_range[0], temp_range[1], num_points)
    
    # Arrhenius shift factor: a_T = exp(Ea/R * (1/T - 1/T_ref))
    log_a_T = (activation_energy / R) * (1/temperature - 1/reference_temp)
    a_T = np.exp(log_a_T)
    
    # Calculate E'(T) and E''(T) with shifted relaxation times
    E_prime = np.full_like(temperature, model.E_inf)
    E_double_prime = np.zeros_like(temperature)
    
    omega = np.full_like(temperature, frequency)
    
    for E_i, tau_i in zip(model.E_i, model.tau_i):
        # Shift relaxation time: τ_shifted = τ * a_T
        tau_shifted = tau_i * a_T
        omega_tau = omega * tau_shifted
        E_prime += E_i * omega_tau**2 / (1 + omega_tau**2)
        E_double_prime += E_i * omega_tau / (1 + omega_tau**2)
    
    # Calculate derived quantities
    loss_tangent = E_double_prime / E_prime
    complex_modulus = np.sqrt(E_prime**2 + E_double_prime**2)
    phase_angle = np.arctan2(E_double_prime, E_prime)
    
    return DMAResult(
        frequency=np.full_like(temperature, frequency),
        temperature=temperature,
        storage_modulus=E_prime,
        loss_modulus=E_double_prime,
        loss_tangent=loss_tangent,
        complex_modulus=complex_modulus,
        phase_angle=phase_angle,
        metadata={
            'sweep_type': 'temperature',
            'frequency': frequency,
            'activation_energy': activation_energy,
            'reference_temp': reference_temp,
            'model': 'GeneralizedMaxwell',
            'E_inf': model.E_inf,
            'E_i': model.E_i.tolist(),
            'tau_i': model.tau_i.tolist(),
        }
    )


def master_curve(
    model: GeneralizedMaxwell,
    reference_temp: float = 298.15,
    temperatures: List[float] = [280, 290, 300, 310, 320],
    freq_range: Tuple[float, float] = (0.01, 1000),
    num_points: int = 50,
    activation_energy: float = 50e3,
) -> Dict[float, DMAResult]:
    """
    Generate master curve using time-temperature superposition.
    
    Parameters
    ----------
    model : GeneralizedMaxwell
        Viscoelastic model
    reference_temp : float
        Reference temperature (K)
    temperatures : list
        List of temperatures to shift (K)
    freq_range : tuple
        Frequency range for each curve
    num_points : int
        Number of frequency points per curve
    activation_energy : float
        Activation energy for shift factor (J/mol)
    
    Returns
    -------
    dict
        Dictionary mapping temperature to DMAResult
    
    Examples
    --------
    >>> model = GeneralizedMaxwell(E_inf=1e9, E_i=[5e8, 3e8], tau_i=[0.1, 1.0])
    >>> curves = master_curve(model, reference_temp=298, temperatures=[280, 298, 320])
    >>> for T, result in curves.items():
    ...     result.plot('modulus')
    """
    R = 8.314  # Gas constant
    curves = {}
    
    for T in temperatures:
        # Calculate shift factor
        log_a_T = (activation_energy / R) * (1/T - 1/reference_temp)
        a_T = np.exp(log_a_T)
        
        # Shift frequency range
        shifted_freq_range = (freq_range[0] * a_T, freq_range[1] * a_T)
        
        # Generate frequency sweep at this temperature
        result = frequency_sweep(
            model,
            freq_range=shifted_freq_range,
            num_points=num_points,
            temperature=T,
        )
        
        # Shift back to reference frequency
        result.frequency = result.frequency / a_T
        result.metadata['shift_factor'] = a_T
        result.metadata['log_shift_factor'] = log_a_T
        
        curves[T] = result
    
    return curves

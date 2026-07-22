"""
Example 9: Creep Analysis of Fiber Networks

Demonstrates:
- Creep compliance curves
- Creep-recovery testing
- Burger's model fitting
- Time-temperature superposition
- Stress relaxation

Usage:
    python examples/09_creep_analysis.py
"""

import numpy as np
from fibernet import gen
from fibernet.sim.creep import CreepAnalyzer, analyze_creep

print("=" * 60)
print("Example 9: Creep Analysis of Fiber Networks")
print("=" * 60)

# Generate network
print("\n[1/5] Generating random fiber network...")
net = gen.random_straight_2d(
    num_fibers=60,
    fiber_length=10.0,
    box_size=(50.0, 50.0),
    seed=42,
)
print(f"  Fibers: {net.num_fibers}")
print(f"  Crosslinks: {net.num_crosslinks}")

# Initialize creep analyzer
print("\n[2/5] Initializing creep analyzer...")
creep = CreepAnalyzer(
    net,
    viscosity=1e12,  # Pa·s
    temperature=293.15,  # K
)
print(f"  Instantaneous modulus: {creep.E0:.2e} Pa")
print(f"  Viscosity: {creep.eta:.2e} Pa·s")

# Creep test
print("\n[3/5] Running creep test (1 hour)...")
creep_result = creep.creep_test(
    stress=1e6,
    duration=3600,
    num_points=100,
)

print(f"  Applied stress: {creep_result.applied_stress:.2e} Pa")
print(f"  Instantaneous strain: {creep_result.instantaneous_strain:.4f}")
print(f"  Total strain (1h): {creep_result.total_strain:.4f}")
print(f"  Steady-state creep rate: {creep_result.steady_state_rate:.2e} 1/s")

# Creep-recovery
print("\n[4/5] Running creep-recovery test...")
creep_phase, recovery_phase = creep.creep_recovery(
    stress=1e6,
    creep_duration=3600,
    recovery_duration=1800,
)

print(f"  Creep strain (1h): {creep_phase.total_strain:.4f}")
print(f"  Recovered strain: {recovery_phase.recovery_strain:.4f}")
print(f"  Permanent strain: {recovery_phase.total_strain:.4f}")
recovery_pct = recovery_phase.recovery_strain / creep_phase.total_strain * 100
print(f"  Recovery percentage: {recovery_pct:.1f}%")

# Stress relaxation
print("\n[5/5] Simulating stress relaxation...")
relax_data = creep.stress_relaxation(
    initial_strain=0.01,
    duration=3600,
)

print(f"  Initial stress: {relax_data['initial_stress']:.2e} Pa")
print(f"  Final stress (1h): {relax_data['final_stress']:.2e} Pa")
print(f"  Relaxation time: {relax_data['relaxation_time']:.2e} s")
relax_pct = (1 - relax_data['final_stress'] / relax_data['initial_stress']) * 100
print(f"  Stress relaxation: {relax_pct:.1f}%")

# Time-temperature superposition
print("\n[BONUS] Time-temperature superposition (WLF)...")
tts_data = creep.time_temperature_superposition(
    reference_temperature=293.15,
    temperatures=[253.15, 273.15, 293.15, 313.15, 333.15],
    shift_factor='wlf',
)

print("  Temperature (K)  |  Shift Factor a_T")
print("  " + "-" * 45)
for T, aT in zip(tts_data['temperatures'], tts_data['shift_factors']):
    print(f"  {T:>16.1f}  |  {aT:>18.2e}")

print("\n" + "=" * 60)
print("Creep analysis complete!")
print("\nKey insights:")
print("  - Creep = instantaneous elastic + time-dependent viscous")
print("  - Recovery shows reversible vs permanent deformation")
print("  - Higher temperatures accelerate creep (time-temperature shift)")

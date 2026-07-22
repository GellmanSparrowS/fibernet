"""
Example 8: Fatigue Analysis of Fiber Networks

Demonstrates:
- S-N curve generation (Basquin equation)
- Cyclic loading simulation
- Fatigue life prediction with Goodman correction
- Variable amplitude loading (Miner's rule)
- Stiffness degradation tracking

Usage:
    python examples/08_fatigue_analysis.py
"""

import numpy as np
from fibernet import gen
from fibernet.sim.fatigue import FatigueAnalyzer, analyze_fatigue

print("=" * 60)
print("Example 8: Fatigue Analysis of Fiber Networks")
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

# Initialize fatigue analyzer
print("\n[2/5] Initializing fatigue analyzer...")
fatigue = FatigueAnalyzer(
    net,
    fatigue_strength_exponent=-0.12,  # Basquin exponent
)
print(f"  Ultimate tensile strength: {fatigue.uts:.2e} Pa")
print(f"  Fatigue strength coefficient: {fatigue.sigma_f:.2e} Pa")
print(f"  Endurance limit: {fatigue.endurance:.2e} Pa")

# Generate S-N curve
print("\n[3/5] Generating S-N curve...")
sn_result = fatigue.generate_sn_curve(num_points=10)

print("  Stress Amplitude (Pa)  |  Cycles to Failure")
print("  " + "-" * 50)
for point in sn_result.sn_curve:
    print(f"  {point.stress_amplitude:>20.2e}  |  {point.cycles_to_failure:>18d}")

# Predict fatigue life with mean stress
print("\n[4/5] Predicting fatigue life with mean stress correction...")
stress_amplitudes = [1e7, 5e6, 2e6]
mean_stresses = [0, 5e6, 1e7]

for sigma_a, sigma_m in zip(stress_amplitudes, mean_stresses):
    Nf = fatigue.predict_life(sigma_a, mean_stress=sigma_m)
    print(f"  σ_a={sigma_a:.1e}, σ_m={sigma_m:.1e}: Nf = {Nf:,d} cycles")

# Cyclic loading simulation
print("\n[5/5] Simulating cyclic loading (500 cycles)...")
cyclic_result = fatigue.cyclic_loading(
    stress_amplitude=5e6,
    mean_stress=2e6,
    num_cycles=500,
)

print(f"  Max stress: {cyclic_result.max_stress:.2e} Pa")
print(f"  Min stress: {cyclic_result.min_stress:.2e} Pa")
print(f"  Stress ratio R: {cyclic_result.stress_ratio:.2f}")
print(f"  Cycles completed: {cyclic_result.num_cycles}")
print(f"  Residual stiffness: {cyclic_result.residual_stiffness:.3f}")
print(f"  Damage per cycle: {cyclic_result.damage_per_cycle:.2e}")

# Variable amplitude loading
print("\n[BONUS] Variable amplitude loading (spectrum)...")
load_spectrum = [
    (8e6, 100),   # High stress, few cycles
    (5e6, 500),   # Medium stress, moderate cycles
    (2e6, 5000),  # Low stress, many cycles
]
va_result = fatigue.variable_amplitude_loading(load_spectrum)

print(f"  Total damage (Miner's rule): {va_result.damage_accumulation:.4f}")
print(f"  Equivalent life: {va_result.cycles_to_failure:,d} cycles")
print(f"  Failed: {va_result.is_failed}")

print("\n" + "=" * 60)
print("Fatigue analysis complete!")
print("\nKey insights:")
print("  - S-N curve shows log-linear relationship")
print("  - Mean stress reduces fatigue life (Goodman effect)")
print("  - Damage accumulates linearly (Miner's rule)")

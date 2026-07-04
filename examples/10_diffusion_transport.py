"""
Example 10: Diffusion and Transport Through Fiber Networks

Demonstrates:
- Effective diffusion coefficient
- Tortuosity analysis
- Concentration profile simulation
- Filtration performance
- Multiple tortuosity models

Usage:
    python examples/10_diffusion_transport.py
"""

import numpy as np
from fibernet import gen
from fibernet.sim.diffusion import DiffusionAnalyzer, analyze_diffusion

print("=" * 60)
print("Example 10: Diffusion and Transport Analysis")
print("=" * 60)

# Generate network
print("\n[1/5] Generating random fiber network...")
net = gen.random_straight_2d(
    num_fibers=80,
    fiber_length=10.0,
    box_size=(50.0, 50.0),
    seed=42,
)
print(f"  Fibers: {net.num_fibers}")
print(f"  Crosslinks: {net.num_crosslinks}")

# Initialize diffusion analyzer
print("\n[2/5] Initializing diffusion analyzer...")
diff = DiffusionAnalyzer(
    net,
    molecular_diffusion=1e-9,  # m²/s (typical for small molecules)
)
print(f"  Molecular diffusion: {diff.D_mol:.2e} m²/s")
print(f"  Network porosity: {diff.porosity:.3f}")

# Effective diffusion
print("\n[3/5] Computing effective diffusion coefficient...")
diff_result = diff.compute_effective_diffusion()

print(f"  Effective diffusion: {diff_result.effective_diffusion_coefficient:.2e} m²/s")
print(f"  Tortuosity factor: {diff_result.tortuosity:.3f}")
print(f"  Reduction factor: {diff_result.effective_diffusion_coefficient / diff.D_mol:.3f}")

# Compare tortuosity models
print("\n[4/5] Comparing tortuosity models...")
models = ['bruggeman', 'comiti', 'weissberg']
for model in models:
    tau = diff.compute_tortuosity(method=model)
    print(f"  {model:12s}: τ = {tau:.3f}")

# Concentration profile simulation
print("\n[5/5] Simulating concentration profile (1 hour)...")
conc_result = diff.simulate_concentration_profile(
    initial_concentration=1.0,
    duration=3600,
    num_time_steps=100,
    num_space_steps=50,
)

print(f"  Breakthrough time: {conc_result.breakthrough_time:.1f} s")
print(f"  Concentration at midpoint (t=1800s): "
      f"{conc_result.concentration_profile[50, 25]:.3f}")
print(f"  Concentration at end (t=3600s): "
      f"{conc_result.concentration_profile[99, 49]:.3f}")

# Filtration analysis
print("\n[BONUS] Filtration performance analysis...")
filtration = diff.filtration_analysis(
    particle_size=1e-6,  # 1 μm particles
    flow_velocity=0.01,  # 1 cm/s
    fluid_viscosity=1e-3,  # Water
)

print(f"  Capture efficiency: {filtration.capture_efficiency * 100:.1f}%")
print(f"  Pressure drop: {filtration.pressure_drop:.2f} Pa")
print(f"  Flow velocity: {filtration.filtration_velocity:.3f} m/s")

print("\n" + "=" * 60)
print("Diffusion analysis complete!")
print("\nKey insights:")
print("  - Fibers reduce effective diffusion (tortuosity effect)")
print("  - Lower porosity = higher tortuosity = slower diffusion")
print("  - Filtration efficiency depends on particle/fiber size ratio")
print("  - Pressure drop increases with lower porosity")

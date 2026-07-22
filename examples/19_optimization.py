"""
Example 19: Energy Minimization and Parameter Optimization

Demonstrates SciPy-based optimization capabilities:
- Energy minimization of fiber networks
- Parameter optimization for target properties
- Multiple optimization methods (L-BFGS-B, CG, Powell, etc.)
- Global optimization with differential evolution

Applications:
- Finding equilibrium configurations
- Optimizing network parameters for desired properties
- Structure-property relationship studies

Usage:
    python examples/19_optimization.py
"""

import numpy as np
from fibernet import gen
from fibernet.sim.optimization import (
    EnergyMinimizer,
    ParameterOptimizer,
    OptimizationResult,
)
from fibernet.core.material import Material

print("=" * 70)
print("  FiberNet - Optimization Examples")
print("=" * 70)

# Define material properties
material = Material(
    name="polymer_fiber",
    youngs_modulus=1e9,  # 1 GPa
    density=1000.0,  # kg/m³
    poissons_ratio=0.35,
)

# 1. Energy Minimization
print("\n[1/4] Energy Minimization (Geometry Optimization)")
print("-" * 70)

# Create a slightly deformed network
net = gen.random_straight_3d(
    num_fibers=20,
    fiber_length=10.0,
    box_size=(30, 30, 30),
    seed=42
)

# Add small perturbations to simulate deformation
for fiber in net.fibers:
    perturbation = np.random.randn(*fiber.centerline.shape) * 0.05
    fiber.centerline += perturbation

print(f"  Initial network: {len(net.fibers)} fibers")
print(f"  Perturbation added to simulate non-equilibrium state")

# Create energy minimizer
minimizer = EnergyMinimizer(net)
print(f"  Optimization variables: {minimizer.num_vars}")

# Minimize energy using L-BFGS-B
result = minimizer.minimize(
    method='L-BFGS-B',
    max_iterations=500,
    tolerance=1e-6
)

print(f"\n  Optimization Results:")
print(f"    Success: {result.success}")
print(f"    Iterations: {result.num_iterations}")
print(f"    Function evaluations: {result.num_function_evals}")
print(f"    Final energy: {result.final_value:.6e}")

# Update network with optimized positions
minimizer.update_network(result)
print(f"  Network updated with optimized configuration")

# 2. Compare Different Methods
print("\n[2/4] Comparing Optimization Methods")
print("-" * 70)

methods = ['L-BFGS-B', 'CG', 'Powell', 'Nelder-Mead']
results = {}

for method in methods:
    # Create fresh network for each test
    net_test = gen.random_straight_3d(
        num_fibers=10,
        fiber_length=8.0,
        box_size=(20, 20, 20),
        seed=42
    )
    
    # Add perturbation
    for fiber in net_test.fibers:
        perturbation = np.random.randn(*fiber.centerline.shape) * 0.03
        fiber.centerline += perturbation
    
    # Minimize
    minimizer = EnergyMinimizer(net_test)
    result = minimizer.minimize(
        method=method,
        max_iterations=200
    )
    
    results[method] = result
    print(f"  {method:15s}: {result.num_iterations:4d} iters, "
          f"energy = {result.final_value:.6e}")

# 3. Parameter Optimization
print("\n[3/4] Parameter Optimization")
print("-" * 70)

print("  Objective: Find parameters that maximize fiber density")

def density_objective(params):
    """Objective function: maximize fiber density (minimize negative density)"""
    num_fibers = int(params[0])
    fiber_length = params[1]
    
    # Generate network
    net = gen.random_straight_3d(
        num_fibers=num_fibers,
        fiber_length=fiber_length,
        box_size=(30, 30, 30),
        seed=42
    )
    
    # Calculate total fiber length (proxy for density)
    total_length = sum(fiber.length for fiber in net.fibers)
    
    # Return negative for minimization (we want to maximize)
    return -total_length

# Create optimizer
optimizer = ParameterOptimizer(density_objective)

# Optimize with bounds
result_param = optimizer.optimize(
    bounds=[(10, 50), (5.0, 20.0)],  # [num_fibers, fiber_length]
    method='L-BFGS-B',
    max_iterations=50
)

print(f"\n  Optimization Results:")
print(f"    Success: {result_param.success}")
print(f"    Optimal num_fibers: {result_param.final_params[0]:.1f}")
print(f"    Optimal fiber_length: {result_param.final_params[1]:.2f}")
print(f"    Max density (negative objective): {-result_param.final_value:.2f}")

# 4. Global Optimization
print("\n[4/4] Global Optimization (Differential Evolution)")
print("-" * 70)

print("  Using differential evolution for global search")
print("  (Slower but more robust for complex landscapes)")

def complex_objective(params):
    """Complex objective with multiple local minima"""
    num_fibers = int(params[0])
    fiber_length = params[1]
    
    net = gen.random_straight_3d(
        num_fibers=num_fibers,
        fiber_length=fiber_length,
        box_size=(30, 30, 30),
        seed=42
    )
    
    # Target: specific combination of properties
    target_fibers = 30
    target_length = 12.0
    
    # Penalize deviation from targets
    penalty = (len(net.fibers) - target_fibers)**2 + \
              (fiber_length - target_length)**2
    
    # Add oscillatory term to create local minima
    oscillation = 0.5 * np.sin(num_fibers / 5.0) * np.cos(fiber_length / 3.0)
    
    return penalty + oscillation

optimizer_global = ParameterOptimizer(complex_objective)

result_global = optimizer_global.optimize_global(
    bounds=[(10, 50), (5.0, 20.0)],
    max_iterations=100,
    seed=42
)

print(f"\n  Global Optimization Results:")
print(f"    Success: {result_global.success}")
print(f"    Best num_fibers: {result_global.final_params[0]:.1f}")
print(f"    Best fiber_length: {result_global.final_params[1]:.2f}")
print(f"    Best objective: {result_global.final_value:.4f}")

print("\n" + "=" * 70)
print("  Optimization Examples Complete!")
print("=" * 70)
print("""
Optimization Methods Available:
  Local Methods (fast, may find local minima):
    - L-BFGS-B: Limited-memory BFGS with bounds (recommended)
    - CG: Conjugate gradient
    - BFGS: Quasi-Newton method
    - Nelder-Mead: Simplex method (derivative-free)
    - Powell: Conjugate direction method (derivative-free)
  
  Global Methods (slower, more robust):
    - Differential Evolution: Evolutionary algorithm
    - Basin Hopping: Stochastic global optimization

Applications:
  - Find equilibrium configurations
  - Optimize network parameters for target properties
  - Study structure-property relationships
  - Design networks with specific characteristics

References:
  - SciPy Optimization: https://docs.scipy.org/doc/scipy/reference/optimize.html
  - Nocedal & Wright, "Numerical Optimization", Springer, 2006
""")

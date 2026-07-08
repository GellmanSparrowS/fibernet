# FiberNet FEM Implementation Guide

## Overview

FiberNet implements a **custom finite element method (FEM)** solver for fiber network mechanics, built from scratch using NumPy and SciPy. It does **not** depend on any open-source Python FEM library (such as FEniCS, SfePy, PyFEM, or GetFEM).

This document describes the mathematical formulation, implementation details, and validation approach.

---

## Mathematical Foundation

### Element Type: 3D Euler-Bernoulli Beam

Each fiber is discretized into Euler-Bernoulli beam elements. Each element has:
- **2 nodes** (end points)
- **6 DOF per node**: [u_x, u_y, u_z, θ_x, θ_y, θ_z]
- **12 DOF total** per element

The beam theory assumes:
1. Plane sections remain plane
2. Shear deformation is neglected (valid for slender beams, L/r > 10)
3. Small deformations and rotations
4. Linear elastic material behavior

### Local Stiffness Matrix

The 12×12 local stiffness matrix combines three deformation modes:

#### Axial Deformation (DOFs 0, 6)
```
k_axial = EA/L * [[1, -1], [-1, 1]]
```

#### Bending (DOFs 1, 5, 7, 11 for y-z plane; DOFs 2, 4, 8, 10 for x-z plane)
```
k_bend = EI/L³ * [[12, 6L, -12, 6L],
                  [6L, 4L², -6L, 2L²],
                  [-12, -6L, 12, -6L],
                  [6L, 2L², -6L, 4L²]]
```

#### Torsion (DOFs 3, 9)
```
k_torsion = GJ/L * [[1, -1], [-1, 1]]
```

Where:
- E = Young's modulus
- G = Shear modulus = E / (2(1+ν))
- A = Cross-sectional area = πr²
- I = Second moment of area = πr⁴/4
- J = Polar moment = πr⁴/2
- L = Element length

### Coordinate Transformation

Local stiffness is transformed to global coordinates via:

```
K_global = T^T · K_local · T
```

Where T is the 12×12 transformation matrix constructed from the element direction vector and a reference vector to define the local coordinate system.

### Global Assembly

The global stiffness matrix is assembled using scipy.sparse:

```python
K = lil_matrix((num_dof, num_dof))
for elem, (ni, nj) in zip(elements, connectivity):
    k_g = elem.stiffness_global
    # Scatter k_g into K at appropriate DOF locations
K = K.tocsr()  # Convert to CSR for efficient solving
```

### Boundary Conditions

Three types of boundary conditions are supported:

1. **Fixed nodes**: All 6 DOFs constrained to zero
2. **Fixed DOFs**: Specific DOF indices constrained to zero
3. **Prescribed displacements**: Specific DOFs with non-zero displacement values

For prescribed displacements, the load vector is modified:
```
F_modified = F - K[:, prescribed_dofs] · u_prescribed
```

### Linear System Solution

The reduced system is solved using scipy.sparse.linalg.spsolve:

```python
K_free = K[np.ix_(free_dofs, free_dofs)]
F_free = F[free_dofs]
u_free = spsolve(K_free, F_free)
```

**Robustness features:**
- Tikhonov regularization for near-singular systems
- Isolated DOF detection and stabilization
- Fallback to dense least-squares solver if sparse solver fails

---

## Post-processing

### Strain and Stress

For each element, axial strain and stress are computed:

```
ε_axial = (u_local[6] - u_local[0]) / L
σ_axial = E · ε_axial
```

### Effective Modulus

The effective Young's modulus is computed from strain energy:

```
E_eff = 2U / (V · ε²)
```

Where:
- U = Total strain energy = ½ u^T K u
- V = Network volume (bounding box)
- ε = Applied strain

### Poisson's Ratio

Computed from transverse strain response:

```
ν = -ε_transverse / ε_axial
```

---

## Nonlinear Extensions

The `sim/nonlinear.py` module extends the linear FEM with:

### Constitutive Models

1. **Linear Elastic**: σ = Eε
2. **Bilinear Plasticity**: Elastic-perfectly plastic or with hardening
3. **Hyperelastic**: Neo-Hookean, Mooney-Rivlin, Arruda-Boyce
4. **Viscoelastic**: Maxwell, Kelvin-Voigt, Standard Linear Solid

### Incremental Loading

For nonlinear analysis, load is applied incrementally:

```python
for strain_step in np.linspace(0, max_strain, num_steps):
    # Apply incremental displacement
    # Solve linearized system
    # Update stress state
    # Check convergence (Newton-Raphson)
```

### Damage and Failure

The `sim/incremental_fem.py` module implements:
- Isotropic damage evolution
- Fiber failure criteria
- Progressive network degradation

---

## Validation

The `sim/validation.py` module provides analytical benchmarks:

### 1. Cantilever Beam (Euler-Bernoulli)

Analytical solution:
```
δ_tip = PL³ / (3EI)
θ_tip = PL² / (2EI)
M_max = PL
```

### 2. Gibson-Ashby Cellular Solids

For 2D honeycomb (bending-dominated):
```
E*/E_s ∝ (ρ*/ρ_s)³
```

For 3D open-cell foam (stretching-dominated):
```
E*/E_s ∝ (ρ*/ρ_s)²
```

### 3. Patch Test

Uniform strain should be exactly reproduced by linear elements (machine precision).

### 4. Convergence Study

h-refinement: effective modulus should converge as segments_per_fiber increases.

---

## Performance

### Computational Complexity

- **Assembly**: O(N_elem × 144) = O(N_elem)
- **Solving**: O(N_dof^1.5) for 2D, O(N_dof^2) for 3D (sparse direct solver)
- **Memory**: O(N_elem × 144) for stiffness matrices

### Typical Performance (CPU)

| Network Size | Elements | DOF | Solve Time |
|--------------|----------|-----|------------|
| Small (2D)   | ~500     | ~3000 | <0.1 s |
| Medium (2D)  | ~2000    | ~12000 | ~1 s |
| Large (3D)   | ~10000   | ~60000 | ~10 s |

### GPU Acceleration (Optional)

The `sim/accelerated.py` module provides Taichi-based GPU acceleration for large-scale problems. However, **all core FEM functionality works on CPU only** with no GPU dependencies.

---

## Comparison with Other FEM Libraries

| Feature | FiberNet | FEniCS | SfePy | PyFEM |
|---------|----------|--------|-------|-------|
| Beam elements | ✅ Native | ❌ Requires custom | ⚠️ Limited | ✅ |
| Fiber networks | ✅ Native | ❌ | ❌ | ❌ |
| Lightweight | ✅ (NumPy+SciPy) | ❌ (heavy deps) | ⚠️ | ⚠️ |
| GPU acceleration | ✅ (Taichi) | ❌ | ❌ | ❌ |
| Domain-specific | ✅ Materials | ❌ General | ❌ General | ❌ General |
| Installation | `pip install fibernet` | Complex | Moderate | Moderate |

**Key advantages of FiberNet:**
1. Purpose-built for fiber network mechanics
2. Zero heavy dependencies (only NumPy + SciPy)
3. Integrated with network generation and analysis
4. CPU and GPU backends
5. Academic-focused with validation tools

---

## References

1. Gibson, L. J., & Ashby, M. F. (1997). *Cellular Solids: Structure and Properties* (2nd ed.). Cambridge University Press.
2. Timoshenko, S. P., & Gere, J. M. (1961). *Theory of Elastic Stability*. McGraw-Hill.
3. Bathe, K. J. (1996). *Finite Element Procedures*. Prentice Hall.
4. Zienkiewicz, O. C., & Taylor, R. L. (2000). *The Finite Element Method* (5th ed.). Butterworth-Heinemann.

---

## Code Structure

```
fibernet/sim/
├── mechanical.py          # Core FEM (FiberFEM, BeamElement)
├── nonlinear.py           # Nonlinear constitutive models
├── incremental_fem.py     # Incremental loading with damage
├── validation.py          # Analytical benchmarks
├── buckling_analysis.py   # Eigenvalue buckling
├── dynamics.py            # Time integration
├── fracture_mechanics.py  # Crack propagation
├── accelerated.py         # Taichi GPU backend
└── ...
```

---

## Example Usage

```python
import fibernet as fn
from fibernet.sim.mechanical import FiberFEM
from fibernet.sim.validation import validate_cantilever_beam

# Generate a network
net = fn.create("random_2d", num_fibers=100, fiber_length=10.0)

# Create FEM solver
fem = FiberFEM(net, segments_per_fiber=5)

# Apply uniaxial strain
result = fem.apply_uniaxial_strain(strain=0.01, axis=0)

# Compute effective modulus
E_eff = fem.effective_modulus(strain=0.001, axis=0)
print(f"Effective modulus: {E_eff:.2e} Pa")

# Validate against analytical solution
validation = validate_cantilever_beam()
print(validation.summary())
```

---

## Citation

If you use FiberNet's FEM implementation in your research, please cite:

```bibtex
@software{fibernet2025,
  title     = {FiberNet: A Comprehensive Python Toolkit for Fiber Network
               Generation, Simulation, and Analysis},
  author    = {FiberNet Contributors},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/GellmanSparrowS/fibernet},
  version   = {1.24.0}
}
```

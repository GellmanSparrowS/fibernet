"""
Nonlinear mechanics example - demonstrating advanced constitutive models.

Shows:
1. Hyperelastic models (Neo-Hookean, Mooney-Rivlin, Arruda-Boyce)
2. Plasticity models (Bilinear, Power-law)
3. Viscoelastic models (Maxwell, Kelvin-Voigt, SLS)
4. Full stress-strain curves with yielding
5. I/O export to LAMMPS and VTK
"""
import sys
sys.path.insert(0, '/home/codex/projects/codex_test/fibernet')

import numpy as np
from fibernet import gen
from fibernet.sim.nonlinear import (
    NonlinearFEM,
    LinearElastic, BilinearPlasticity, PowerLawHardening,
    HyperelasticNeoHookean, HyperelasticMooneyRivlin, ArrudaBoyce,
    KelvinVoigtModel, MaxwellModel, StandardLinearSolid,
)
from fibernet.io import to_lammps, to_vtk, to_gmsh
from fibernet.utils.units import UnitConverter, MICRO


def main():
    print("=" * 70)
    print("FiberNet Nonlinear Mechanics Example")
    print("=" * 70)
    
    # Generate test network
    print("\n[1] Generating fiber network...")
    net = gen.square_lattice_2d(spacing=5, grid_size=(4, 4))
    print(f"  Network: {net.num_fibers} fibers, {net.num_crosslinks} crosslinks")
    
    # ============================================================
    # Constitutive Models
    # ============================================================
    print("\n[2] Constitutive Models")
    print("-" * 70)
    
    # Linear elastic
    print("\n  Linear Elastic (E = 1 GPa):")
    model = LinearElastic(E=1e9)
    strains = np.linspace(0, 0.01, 50)
    stresses = np.array([model.stress(e) for e in strains])
    print(f"    Stress at 1% strain: {model.stress(0.01):.2e} Pa")
    
    # Bilinear plasticity
    print("\n  Bilinear Plasticity (E=1 GPa, sigma_y=5 MPa, Et=50 MPa):")
    plastic = BilinearPlasticity(E=1e9, sigma_y=5e6, Et=5e7)
    stresses_plastic = np.array([plastic.stress(e) for e in strains])
    print(f"    Yield strain: {5e6/1e9:.4f}")
    print(f"    Stress at 1% strain: {plastic.stress(0.01):.2e} Pa")
    
    # Neo-Hookean
    print("\n  Neo-Hookean (G = 1 MPa):")
    neo_hook = HyperelasticNeoHookean(G=1e6)
    large_strains = np.linspace(0, 0.5, 50)
    stresses_nh = np.array([neo_hook.stress(e) for e in large_strains])
    print(f"    Stress at 50% strain: {neo_hook.stress(0.5):.2e} Pa")
    
    # Mooney-Rivlin
    print("\n  Mooney-Rivlin (C1=0.5 MPa, C2=0.25 MPa):")
    mr = HyperelasticMooneyRivlin(C1=5e5, C2=2.5e5)
    stresses_mr = np.array([mr.stress(e) for e in large_strains])
    print(f"    Stress at 50% strain: {mr.stress(0.5):.2e} Pa")
    
    # Arruda-Boyce
    print("\n  Arruda-Boyce (N=100, nkT=1 MPa):")
    ab = ArrudaBoyce(n_chain=100, nkT=1e6)
    stresses_ab = np.array([ab.stress(e) for e in large_strains])
    print(f"    Stress at 50% strain: {ab.stress(0.5):.2e} Pa")
    
    # ============================================================
    # Full Stress-Strain Curves
    # ============================================================
    print("\n[3] Full Stress-Strain Curves")
    print("-" * 70)
    
    # Linear elastic
    print("\n  Linear elastic stress-strain curve:")
    fem_linear = NonlinearFEM(net, constitutive_model=LinearElastic(E=1e9), segments_per_fiber=5)
    eps, sigma, energy = fem_linear.stress_strain_curve(axis=0, max_strain=0.01, num_steps=20)
    print(f"    Points: {len(eps)}")
    print(f"    Max stress: {sigma[-1]:.2e} Pa at strain {eps[-1]:.4f}")
    print(f"    Strain energy: {energy[-1]:.2e} J")
    
    # With plasticity
    print("\n  Plasticity stress-strain curve:")
    fem_plastic = NonlinearFEM(net, constitutive_model=BilinearPlasticity(E=1e9, sigma_y=5e6, Et=5e7), segments_per_fiber=5)
    eps_p, sigma_p, energy_p = fem_plastic.stress_strain_curve(axis=0, max_strain=0.01, num_steps=20)
    print(f"    Max stress: {sigma_p[-1]:.2e} Pa at strain {eps_p[-1]:.4f}")
    
    # ============================================================
    # Viscoelastic Loading
    # ============================================================
    print("\n[4] Viscoelastic Loading")
    print("-" * 70)
    
    fem = NonlinearFEM(net, segments_per_fiber=5)
    
    # Kelvin-Voigt
    print("\n  Kelvin-Voigt model (E=1 GPa, eta=1 MPa.s):")
    kv = KelvinVoigtModel(E=1e9, eta=1e6)
    t_kv, sigma_kv, eps_kv = fem.viscoelastic_loading(
        visco_model=kv, axis=0, strain_rate=1e-3, max_strain=0.005, dt=1e-4,
    )
    print(f"    Time points: {len(t_kv)}")
    print(f"    Final stress: {sigma_kv[-1]:.2e} Pa at strain {eps_kv[-1]:.4f}")
    
    # Standard Linear Solid
    print("\n  Standard Linear Solid (E1=1 GPa, E2=0.5 GPa, eta=1 MPa.s):")
    sls = StandardLinearSolid(E1=1e9, E2=5e8, eta=1e6)
    t_sls, sigma_sls, eps_sls = fem.viscoelastic_loading(
        visco_model=sls, axis=0, strain_rate=1e-3, max_strain=0.005, dt=1e-4,
    )
    print(f"    Final stress: {sigma_sls[-1]:.2e} Pa at strain {eps_sls[-1]:.4f}")
    
    # ============================================================
    # I/O Export
    # ============================================================
    print("\n[5] I/O Export")
    print("-" * 70)
    
    # LAMMPS
    lammps_file = "/tmp/fibernet_network.lammps"
    to_lammps(net, lammps_file, bead_spacing=2.0)
    print(f"  LAMMPS data: {lammps_file}")
    
    # VTK
    vtk_file = "/tmp/fibernet_network.vtk"
    to_vtk(net, vtk_file)
    print(f"  VTK data: {vtk_file}")
    
    # GMSH
    gmsh_file = "/tmp/fibernet_mesh.msh"
    to_gmsh(net, gmsh_file, segments_per_fiber=5)
    print(f"  GMSH mesh: {gmsh_file}")
    
    # ============================================================
    # Unit Conversion
    # ============================================================
    print("\n[6] Unit Conversion")
    print("-" * 70)
    
    print(f"  1 Pa (SI) = {UnitConverter.convert(1.0, 'stress', 'SI', 'CGS'):.2f} dyne/cm² (CGS)")
    print(f"  1 N (SI) = {UnitConverter.convert(1.0, 'force', 'SI', 'CGS'):.2e} dyne (CGS)")
    print(f"  1 m (SI) = {UnitConverter.convert(1.0, 'length', 'SI', 'NANO'):.2e} nm (Nano)")
    print(f"  1 m (SI) = {UnitConverter.convert(1.0, 'length', 'SI', 'MOLECULAR'):.2e} Å (Molecular)")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("Constitutive models available:")
    print("  - Linear Elastic")
    print("  - Bilinear Plasticity")
    print("  - Power-law Hardening (Ramberg-Osgood)")
    print("  - Neo-Hookean Hyperelastic")
    print("  - Mooney-Rivlin Hyperelastic")
    print("  - Arruda-Boyce 8-chain model")
    print("\nViscoelastic models:")
    print("  - Maxwell (spring + dashpot in series)")
    print("  - Kelvin-Voigt (spring + dashpot in parallel)")
    print("  - Standard Linear Solid (Zener)")
    print("\nI/O formats: LAMMPS, VTK, GMSH, PDB, XYZ")
    print("\nUnit systems: SI, CGS, Micro, Nano, Molecular")
    print("=" * 70)

if __name__ == "__main__":
    main()

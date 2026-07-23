#!/usr/bin/env python3
"""
FiberNet v4 — Large Deformation FEM Test Suite
================================================
Tests deformed structures (n_pts_per_side=5, perturbation=±0.40) using
BeamFrameFEM (beam frame finite element method) under large stretch
and compression with different fiber thicknesses.

Key analysis:
- Deformation propagation (boundary vs interior displacement)
- Axial vs bending stress distribution
- Force-displacement for different radii
- 3D complex structure tests
- Comprehensive visualization (edges colored by stress, not points)

Checkpoint/resume: Each simulation saves JSON; re-run skips completed.
Memory guard: gc.collect() between runs.

Usage:
    cd fibernet && source .venv/bin/activate
    python scripts/deformation_test/run_large_deformation_fem.py
"""

import os, sys, json, time, gc, traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import re

def parse_task_key(key):
    """Parse task key to extract unit, radius, and test name.
    Format: {dim}_{unit}_r{radius:.3f}_{test_name}
    """
    m = re.match(r'(\d+d)_(.+)_r(\d+\.\d+)_(.+)', key)
    if m:
        dim_str, unit, radius_str, test_name = m.groups()
        return {
            'dim': dim_str,
            'unit': unit,
            'radius': float(radius_str),
            'test_name': test_name
        }
    return None



# ── Paths ──
SCRIPT_DIR = Path(__file__).parent
FIBERNET_ROOT = SCRIPT_DIR.parent.parent  # fibernet package root
OUTPUT_DIR = FIBERNET_ROOT / "output_data" / "deformation_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR = OUTPUT_DIR / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint_fem.json"

# ── Physical Parameters ──
# 10cm × 10cm structure: box_per_cell × grid = total_size
BOX_PER_CELL_2D = (2.5, 2.5)
GRID_2D = (4, 4)
BOX_PER_CELL_3D = (3.33, 3.33, 3.33)
GRID_3D = (3, 3, 3)

# Deformation: 边上5个点, ±0.40
N_PTS_PER_SIDE = 5
PERTURBATION = 0.40
SEED = 42

# FEM material
E_MODULUS = 1e9    # 1 GPa (typical polymer)
NU = 0.3           # Poisson's ratio

# Boundary: 10% each side fixed (rigid plate)
PCT_BOUNDARY = 0.10

# Test configs
RADII_2D = [0.02, 0.05, 0.10, 0.20]
RADII_3D = [0.05, 0.10]

# Stretch targets: ratio of final length / original length
STRETCH_TARGETS = {
    "stretch_2x": 2.0,     # 10cm → 20cm
    "stretch_1.5x": 1.5,   # 10cm → 15cm
    "compress_0.5x": 0.5,  # 10cm → 5cm
    "compress_0.7x": 0.7,  # 10cm → 7cm
}
STRETCH_TARGETS_3D = {
    "stretch_2x": 2.0,
    "compress_0.5x": 0.5,
}

UNITS_2D = ["honeycomb", "reentrant", "kagome", "triangle", "square", "diamond", "chiral", "star"]
UNITS_3D = ["octet", "diamond_3d", "reentrant_3d", "cubic", "fcc", "bcc"]

MAX_NODES = 15000

# Nonlinear threshold: use nonlinear solver for |stretch-1| > 0.3
NONLINEAR_THRESHOLD = 0.3


# ═══════════════════════════════════════════════════════════
# CHECKPOINT
# ═══════════════════════════════════════════════════════════

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": []}

def save_checkpoint(ckpt):
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(ckpt, f, indent=2)
    tmp.rename(CHECKPOINT_FILE)

def make_task_key(prefix, unit, radius, test_name):
    return f"{prefix}_{unit}_r{radius:.3f}_{test_name}"


# ═══════════════════════════════════════════════════════════
# STRUCTURE GENERATION
# ═══════════════════════════════════════════════════════════

def generate_2d_structure(unit, radius):
    from fibernet import pattern_2d
    g = pattern_2d(
        unit=unit, box=BOX_PER_CELL_2D, grid=GRID_2D,
        n_pts_per_side=N_PTS_PER_SIDE, perturbation=PERTURBATION,
        radius=radius, seed=SEED,
    )
    return g

def generate_3d_structure(unit, radius):
    from fibernet import pattern_3d
    g = pattern_3d(
        unit=unit, box=BOX_PER_CELL_3D, grid=GRID_3D,
        n_pts_per_side=N_PTS_PER_SIDE, radius=radius, seed=SEED,
    )
    return g

def graph_to_fem_input(graph, dim=2):
    """Convert StructureGraph to BeamFrameFEM input format."""
    from fibernet.sim.accelerated import _graph_to_arrays
    pos, elements, _, _ = _graph_to_arrays(graph)
    edge_index = elements.T.astype(np.int64)
    node_pos = pos[:, :dim] if dim == 2 else pos
    return edge_index, node_pos


# ═══════════════════════════════════════════════════════════
# FEM SIMULATION
# ═══════════════════════════════════════════════════════════

def run_fem_2d(graph, target_stretch, radius, E=E_MODULUS, nu=NU,
               pct=PCT_BOUNDARY, label=""):
    """
    Run 2D beam frame FEM with prescribed displacement.
    
    - Left 10%: fully fixed (ux=uy=θ=0)
    - Right 10%: prescribed (ux=target_disp, uy=0)
    - Uses nonlinear solver for large deformation
    """
    from fibernet.sim.accelerated import _graph_to_arrays, _get_boundary_indices
    from fibernet.ml.beam_frame_fem import BeamFrameFEM
    
    pos, elements, _, _ = _graph_to_arrays(graph)
    edge_index = elements.T.astype(np.int64)
    node_pos = pos[:, :2]  # 2D
    radii = np.full(edge_index.shape[1], radius)
    
    if graph.num_nodes > MAX_NODES:
        print(f"  ⚠ {label}: {graph.num_nodes} nodes > {MAX_NODES}, skipping")
        return None
    
    boundaries = _get_boundary_indices(pos, pct=pct)
    left_nodes = boundaries.get("left", [])
    right_nodes = boundaries.get("right", [])
    
    if not left_nodes or not right_nodes:
        print(f"  ⚠ {label}: No boundary nodes")
        return None
    
    # Calculate prescribed displacement
    x_range = pos[:, 0].max() - pos[:, 0].min()
    target_disp = x_range * (target_stretch - 1.0)
    
    prescribed = {}
    for ni in right_nodes:
        prescribed[ni] = (target_disp, 0.0)
    
    solver = BeamFrameFEM(E=E, nu=nu)
    
    # Choose solver based on deformation magnitude
    strain_mag = abs(target_stretch - 1.0)
    use_nonlinear = strain_mag > NONLINEAR_THRESHOLD
    
    try:
        t0 = time.time()
        if use_nonlinear:
            res = solver.solve_2d_nonlinear(
                edge_index, node_pos, radii,
                prescribed_disp=prescribed,
                fixed_nodes=left_nodes,
                n_steps=10,
                tol=1e-6,
                max_iter=20,
            )
        else:
            res = solver.solve_2d(
                edge_index, node_pos, radii,
                fixed_nodes=left_nodes,
                prescribed_disp=prescribed,
            )
        elapsed = time.time() - t0
    except Exception as e:
        print(f"  ✗ {label}: FEM failed: {e}")
        traceback.print_exc()
        return None
    
    # Extract results
    u = res['u']  # (n_nodes, 3) — ux, uy, θz
    deformed_pos = node_pos + u[:, :2]
    
    # Edge data
    edge_list = res['edge_list']
    sigma_axial = res['sigma_axial']   # per unique edge
    sigma_bending = res['sigma_bending']
    sigma_total = res['sigma_total']
    moments = res['moments']           # (n_edges, 2) — moment at each end
    edge_forces = res['edge_forces']   # (n_edges, 3) — [axial_N, shear_V, moment_avg]
    reactions = res['reactions']       # (n_nodes, 3) — reaction forces
    
    # Calculate edge strains from displacement
    orig_lengths = np.linalg.norm(
        node_pos[elements[:, 1]] - node_pos[elements[:, 0]], axis=1
    )
    def_lengths = np.linalg.norm(
        deformed_pos[elements[:, 1]] - deformed_pos[elements[:, 0]], axis=1
    )
    edge_strains = (def_lengths - orig_lengths) / orig_lengths
    
    # Calculate total force from edge forces (works for both linear and nonlinear)
    # For each boundary node, sum the forces from connected edges
    node_edges = {}
    for idx, e in enumerate(edge_list):
        i_node, j_node = int(edge_index[0, e]), int(edge_index[1, e])
        node_edges.setdefault(i_node, []).append((idx, j_node))
        node_edges.setdefault(j_node, []).append((idx, i_node))
    
    total_force_left = 0.0
    for ni in left_nodes:
        ni_int = int(ni)
        if ni_int in node_edges:
            for edge_idx, other_node in node_edges[ni_int]:
                i_node = int(edge_index[0, edge_list[edge_idx]])
                j_node = int(edge_index[1, edge_list[edge_idx]])
                dx = node_pos[j_node, 0] - node_pos[i_node, 0]
                dy = node_pos[j_node, 1] - node_pos[i_node, 1]
                L = np.sqrt(dx**2 + dy**2)
                if L > 1e-12:
                    cx, cy = dx/L, dy/L
                    N = edge_forces[edge_idx, 0]  # axial force
                    V = edge_forces[edge_idx, 1]  # shear force
                    if ni_int == i_node:
                        fx = -N * cx + V * cy
                    else:
                        fx = N * cx - V * cy
                    total_force_left += fx
    
    total_force_right = 0.0
    for ni in right_nodes:
        ni_int = int(ni)
        if ni_int in node_edges:
            for edge_idx, other_node in node_edges[ni_int]:
                i_node = int(edge_index[0, edge_list[edge_idx]])
                j_node = int(edge_index[1, edge_list[edge_idx]])
                dx = node_pos[j_node, 0] - node_pos[i_node, 0]
                dy = node_pos[j_node, 1] - node_pos[i_node, 1]
                L = np.sqrt(dx**2 + dy**2)
                if L > 1e-12:
                    cx, cy = dx/L, dy/L
                    N = edge_forces[edge_idx, 0]
                    V = edge_forces[edge_idx, 1]
                    if ni_int == i_node:
                        fx = -N * cx + V * cy
                    else:
                        fx = N * cx - V * cy
                    total_force_right += fx
    
    # Use average of left and right for stability
    total_force = (abs(total_force_left) + abs(total_force_right)) / 2.0
    
    # Deformation propagation analysis
    all_disps = np.linalg.norm(u[:, :2], axis=1)
    boundary_set = set(left_nodes) | set(right_nodes)
    interior_set = set(range(graph.num_nodes)) - boundary_set
    interior_list = sorted(interior_set)
    boundary_list = sorted(boundary_set)
    
    interior_disps = all_disps[interior_list] if interior_list else np.array([0.0])
    boundary_disps = all_disps[boundary_list] if boundary_list else np.array([0.0])
    
    max_boundary = boundary_disps.max() if len(boundary_disps) > 0 else 1e-10
    max_interior = interior_disps.max() if len(interior_disps) > 0 else 0.0
    propagation_ratio = max_interior / max_boundary if max_boundary > 1e-10 else 0.0
    
    threshold = max_boundary * 0.1
    interior_affected = float(np.sum(interior_disps > threshold) / len(interior_disps)) if len(interior_disps) > 0 else 0.0
    
    # Strain propagation: compare interior vs boundary edge strains
    interior_node_set = set(interior_list)
    interior_edge_mask = np.array([
        (elements[e, 0] in interior_node_set and elements[e, 1] in interior_node_set)
        for e in range(len(elements))
    ])
    boundary_edge_mask = np.array([
        (elements[e, 0] in boundary_set or elements[e, 1] in boundary_set)
        for e in range(len(elements))
    ])
    
    mean_interior_strain = float(np.mean(np.abs(edge_strains[interior_edge_mask]))) if interior_edge_mask.any() else 0.0
    mean_boundary_strain = float(np.mean(np.abs(edge_strains[boundary_edge_mask]))) if boundary_edge_mask.any() else 0.0
    strain_propagation = mean_interior_strain / mean_boundary_strain if mean_boundary_strain > 1e-10 else 0.0
    
    # Bending vs axial stress ratio (for beam behavior analysis)
    max_axial = float(np.max(np.abs(sigma_axial))) if len(sigma_axial) > 0 else 0.0
    max_bending = float(np.max(np.abs(sigma_bending))) if len(sigma_bending) > 0 else 0.0
    bending_dominance = max_bending / (max_axial + 1e-10)
    
    return {
        "label": label,
        "target_stretch": target_stretch,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "total_force_kN": float(total_force / 1000.0),
        "total_force_x_left_kN": float(abs(total_force_left) / 1000.0),
        "total_force_x_right_kN": float(abs(total_force_right) / 1000.0),
        "max_axial_stress_MPa": float(max_axial / 1e6),
        "max_bending_stress_MPa": float(max_bending / 1e6),
        "max_total_stress_MPa": float(np.max(sigma_total) / 1e6),
        "bending_dominance": float(bending_dominance),
        "max_strain": float(np.max(np.abs(edge_strains))),
        "mean_strain": float(np.mean(np.abs(edge_strains))),
        "max_displacement": float(all_disps.max()),
        "mean_displacement": float(all_disps.mean()),
        "interior_mean_disp": float(interior_disps.mean()),
        "interior_max_disp": float(max_interior),
        "boundary_mean_disp": float(boundary_disps.mean()),
        "boundary_max_disp": float(max_boundary),
        "propagation_ratio": float(propagation_ratio),
        "strain_propagation": float(strain_propagation),
        "interior_affected_pct": interior_affected,
        "mean_interior_strain": mean_interior_strain,
        "mean_boundary_strain": mean_boundary_strain,
        "use_nonlinear": use_nonlinear,
        "elapsed_seconds": elapsed,
        # Arrays for visualization (edges, not points)
        "deformed_positions": deformed_pos.tolist(),
        "original_positions": node_pos.tolist(),
        "displacements": u[:, :2].tolist(),
        "edge_strains": edge_strains.tolist(),
        "sigma_axial": sigma_axial.tolist(),
        "sigma_bending": sigma_bending.tolist(),
        "sigma_total": sigma_total.tolist(),
        "elements": [(int(i), int(j)) for i, j in elements],
        "edge_list": [int(e) for e in edge_list],
        "left_nodes": [int(x) for x in left_nodes],
        "right_nodes": [int(x) for x in right_nodes],
        "reactions": reactions.tolist() if reactions is not None else None,
    }


def run_fem_3d(graph, target_stretch, radius, E=E_MODULUS, nu=NU,
               pct=PCT_BOUNDARY, label=""):
    """Run 3D beam frame FEM with prescribed displacement."""
    from fibernet.sim.accelerated import _graph_to_arrays, _get_boundary_indices
    from fibernet.ml.beam_frame_fem import BeamFrameFEM
    
    pos, elements, _, _ = _graph_to_arrays(graph)
    edge_index = elements.T.astype(np.int64)
    node_pos = pos  # Full 3D
    radii = np.full(edge_index.shape[1], radius)
    
    if graph.num_nodes > MAX_NODES:
        print(f"  ⚠ {label}: {graph.num_nodes} nodes > {MAX_NODES}, skipping")
        return None
    
    boundaries = _get_boundary_indices(pos, pct=pct)
    left_nodes = boundaries.get("left", [])
    right_nodes = boundaries.get("right", [])
    
    if not left_nodes or not right_nodes:
        print(f"  ⚠ {label}: No boundary nodes")
        return None
    
    x_range = pos[:, 0].max() - pos[:, 0].min()
    target_disp = x_range * (target_stretch - 1.0)
    
    prescribed = {}
    for ni in right_nodes:
        prescribed[ni] = (target_disp, 0.0, 0.0)
    
    solver = BeamFrameFEM(E=E, nu=nu)
    
    try:
        t0 = time.time()
        res = solver.solve_3d(
            edge_index, node_pos, radii,
            fixed_nodes=left_nodes,
            prescribed_disp=prescribed,
        )
        elapsed = time.time() - t0
    except Exception as e:
        print(f"  ✗ {label}: FEM 3D failed: {e}")
        traceback.print_exc()
        return None
    
    # Extract results
    u = res['u']  # (n_nodes, 6) — ux, uy, uz, θx, θy, θz
    deformed_pos = node_pos + u[:, :3]
    
    sigma_axial = res['sigma_axial']
    sigma_bending = res['sigma_bending']
    sigma_total = res['sigma_total']
    reactions = res.get('reactions')
    
    # Edge strains
    orig_lengths = np.linalg.norm(
        node_pos[elements[:, 1]] - node_pos[elements[:, 0]], axis=1
    )
    def_lengths = np.linalg.norm(
        deformed_pos[elements[:, 1]] - deformed_pos[elements[:, 0]], axis=1
    )
    edge_strains = (def_lengths - orig_lengths) / orig_lengths
    
    # Calculate total force from stress * area for boundary edges
    # For each edge connected to left boundary, F = σ_axial * A * cos(θ)
    edge_list = res.get('edge_list', list(range(len(sigma_axial))))
    total_force = 0.0
    for idx, e in enumerate(edge_list):
        i_node, j_node = int(edge_index[0, e]), int(edge_index[1, e])
        if i_node in left_nodes or j_node in left_nodes:
            r = radii[e]
            A = np.pi * r**2
            dx = node_pos[j_node, 0] - node_pos[i_node, 0]
            dy = node_pos[j_node, 1] - node_pos[i_node, 1] if node_pos.shape[1] > 1 else 0.0
            dz = node_pos[j_node, 2] - node_pos[i_node, 2] if node_pos.shape[1] > 2 else 0.0
            L = np.sqrt(dx**2 + dy**2 + dz**2)
            if L > 1e-12:
                cx = dx / L
                # F = σ * A * cos(θ) where θ is angle with x-axis
                total_force += abs(sigma_axial[idx]) * A * abs(cx)
    
    # Propagation
    all_disps = np.linalg.norm(u[:, :3], axis=1)
    boundary_set = set(left_nodes) | set(right_nodes)
    interior_set = set(range(graph.num_nodes)) - boundary_set
    interior_list = sorted(interior_set)
    boundary_list = sorted(boundary_set)
    
    interior_disps = all_disps[interior_list] if interior_list else np.array([0.0])
    boundary_disps = all_disps[boundary_list] if boundary_list else np.array([0.0])
    
    max_boundary = boundary_disps.max() if len(boundary_disps) > 0 else 1e-10
    max_interior = interior_disps.max() if len(interior_disps) > 0 else 0.0
    propagation_ratio = max_interior / max_boundary if max_boundary > 1e-10 else 0.0
    
    threshold = max_boundary * 0.1
    interior_affected = float(np.sum(interior_disps > threshold) / len(interior_disps)) if len(interior_disps) > 0 else 0.0
    
    # Strain propagation
    interior_node_set = set(interior_list)
    interior_edge_mask = np.array([
        (elements[e, 0] in interior_node_set and elements[e, 1] in interior_node_set)
        for e in range(len(elements))
    ])
    boundary_edge_mask = np.array([
        (elements[e, 0] in boundary_set or elements[e, 1] in boundary_set)
        for e in range(len(elements))
    ])
    
    mean_interior_strain = float(np.mean(np.abs(edge_strains[interior_edge_mask]))) if interior_edge_mask.any() else 0.0
    mean_boundary_strain = float(np.mean(np.abs(edge_strains[boundary_edge_mask]))) if boundary_edge_mask.any() else 0.0
    strain_propagation = mean_interior_strain / mean_boundary_strain if mean_boundary_strain > 1e-10 else 0.0
    
    max_axial = float(np.max(np.abs(sigma_axial))) if len(sigma_axial) > 0 else 0.0
    max_bending = float(np.max(np.abs(sigma_bending))) if len(sigma_bending) > 0 else 0.0
    bending_dominance = max_bending / (max_axial + 1e-10)
    
    return {
        "label": label,
        "target_stretch": target_stretch,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "total_force_kN": float(total_force / 1000.0),
        "max_axial_stress_MPa": float(max_axial / 1e6),
        "max_bending_stress_MPa": float(max_bending / 1e6),
        "max_total_stress_MPa": float(np.max(sigma_total) / 1e6),
        "bending_dominance": float(bending_dominance),
        "max_strain": float(np.max(np.abs(edge_strains))),
        "mean_strain": float(np.mean(np.abs(edge_strains))),
        "max_displacement": float(all_disps.max()),
        "mean_displacement": float(all_disps.mean()),
        "interior_mean_disp": float(interior_disps.mean()),
        "interior_max_disp": float(max_interior),
        "boundary_mean_disp": float(boundary_disps.mean()),
        "boundary_max_disp": float(max_boundary),
        "propagation_ratio": float(propagation_ratio),
        "strain_propagation": float(strain_propagation),
        "interior_affected_pct": interior_affected,
        "mean_interior_strain": mean_interior_strain,
        "mean_boundary_strain": mean_boundary_strain,
        "use_nonlinear": False,
        "elapsed_seconds": elapsed,
        "deformed_positions": deformed_pos.tolist(),
        "original_positions": node_pos.tolist(),
        "displacements": u[:, :3].tolist(),
        "edge_strains": edge_strains.tolist(),
        "sigma_axial": sigma_axial.tolist(),
        "sigma_bending": sigma_bending.tolist(),
        "sigma_total": sigma_total.tolist(),
        "elements": [(int(i), int(j)) for i, j in elements],
        "edge_list": [int(e) for e in res.get('edge_list', [])],
        "left_nodes": [int(x) for x in left_nodes],
        "right_nodes": [int(x) for x in right_nodes],
        "reactions": reactions.tolist() if reactions is not None else None,
    }


# ═══════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════

def analyze_results(all_results):
    """Comprehensive deformation propagation analysis."""
    summary = {}
    
    for key, res in all_results.items():
        if res is None:
            continue
        
        # Parse key: format is "{dim}_{unit}_r{radius:.3f}_{test_name}"
        # Use regex for robust parsing (handles unit names containing 'r')
        import re
        m = re.match(r'(\d+d)_(.+)_r(\d+\.\d+)_(.+)', key)
        if m:
            dim_str, unit, radius_str, test_name = m.groups()
            radius = float(radius_str)
            is_3d = dim_str == "3d"
        else:
            # Fallback: find last "_r" followed by float
            idx = key.rfind("_r")
            prefix_unit = key[:idx]
            rest = key[idx+2:]
            radius = float(rest.split("_")[0])
            test_name = "_".join(rest.split("_")[1:])
            is_3d = prefix_unit.startswith("3d")
            unit = prefix_unit.replace("2d_", "").replace("3d_", "")
        
        prop = res["propagation_ratio"]
        strain_prop = res["strain_propagation"]
        interior_affected = res["interior_affected_pct"]
        
        if prop > 0.5:
            disp_class = "FULL"
        elif prop > 0.2:
            disp_class = "PARTIAL"
        else:
            disp_class = "LOCALIZED"
        
        if strain_prop > 0.5:
            strain_class = "FULL"
        elif strain_prop > 0.2:
            strain_class = "PARTIAL"
        else:
            strain_class = "LOCALIZED"
        
        if unit not in summary:
            summary[unit] = []
        summary[unit].append({
            "key": key,
            "is_3d": is_3d,
            "test": test_name,
            "radius": radius,
            "propagation_ratio": prop,
            "strain_propagation": strain_prop,
            "interior_affected_pct": interior_affected,
            "disp_classification": disp_class,
            "strain_classification": strain_class,
            "max_strain": res["max_strain"],
            "total_force_kN": res["total_force_kN"],
            "max_axial_MPa": res["max_axial_stress_MPa"],
            "max_bending_MPa": res["max_bending_stress_MPa"],
            "bending_dominance": res["bending_dominance"],
        })
    
    return summary


def print_analysis(summary):
    print("\n" + "=" * 70)
    print("DEFORMATION PROPAGATION ANALYSIS (FEM Beam Frame)")
    print("=" * 70)
    
    for unit in sorted(summary.keys()):
        tests = summary[unit]
        print(f"\n{'─' * 70}")
        print(f"  {unit}")
        print(f"{'─' * 70}")
        print(f"  {'Test':20s} {'r':5s} {'DispProp':8s} {'StrnProp':8s} "
              f"{'Int%':6s} {'Force':8s} {'BendDom':8s} {'Class':10s}")
        for t in sorted(tests, key=lambda x: (x["radius"], x["test"])):
            print(f"  {t['test']:20s} {t['radius']:.2f}  "
                  f"{t['propagation_ratio']:8.3f} {t['strain_propagation']:8.3f} "
                  f"{t['interior_affected_pct']:5.1%} "
                  f"{t['total_force_kN']:7.1f}kN "
                  f"{t['bending_dominance']:8.2f} "
                  f"{t['disp_classification']:10s}")
    
    print(f"\n{'─' * 70}")
    print("SUMMARY")
    print(f"{'─' * 70}")
    
    all_localized = sum(1 for u, ts in summary.items() for t in ts if t["disp_classification"] == "LOCALIZED")
    all_partial = sum(1 for u, ts in summary.items() for t in ts if t["disp_classification"] == "PARTIAL")
    all_full = sum(1 for u, ts in summary.items() for t in ts if t["disp_classification"] == "FULL")
    total = all_localized + all_partial + all_full
    
    if total > 0:
        print(f"  Total tests: {total}")
        print(f"  FULL propagation:      {all_full:3d} ({all_full/total:.0%})")
        print(f"  PARTIAL propagation:   {all_partial:3d} ({all_partial/total:.0%})")
        print(f"  LOCALIZED (no spread): {all_localized:3d} ({all_localized/total:.0%})")
    
    print(f"\n  Radius effect on propagation:")
    radius_props = {}
    for unit, tests in summary.items():
        for t in tests:
            r = t["radius"]
            if r not in radius_props:
                radius_props[r] = []
            radius_props[r].append(t["propagation_ratio"])
    for r in sorted(radius_props.keys()):
        vals = radius_props[r]
        print(f"    r={r:.2f}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")
    
    print(f"\n  Bending vs Axial dominance:")
    for unit, tests in summary.items():
        avg_bd = np.mean([t["bending_dominance"] for t in tests])
        dom = "BENDING-dominated" if avg_bd > 2.0 else "MIXED" if avg_bd > 0.5 else "AXIAL-dominated"
        print(f"    {unit:15s}: avg bending_dominance={avg_bd:.2f} → {dom}")


# ═══════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════

def generate_visualization(all_results, summary):
    """Comprehensive visualization — edges colored by stress, deformed coordinates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    results_2d = {k: v for k, v in all_results.items() if v and not k.startswith("3d_")}
    results_3d = {k: v for k, v in all_results.items() if v and k.startswith("3d_")}
    
    units_2d = sorted(set(parse_task_key(k)["unit"] for k in results_2d if parse_task_key(k)))
    test_names = sorted(set(parse_task_key(k)["test_name"] for k in results_2d if parse_task_key(k)))
    viz_radii = [0.02, 0.10]  # thin + thick
    
    n_units_2d = len(units_2d)
    n_cols = max(len(test_names) * len(viz_radii), 6)
    n_rows = n_units_2d + 4  # structures + 4 analysis rows
    
    fig = plt.figure(figsize=(3.5 * n_cols, 3.5 * n_rows + 2))
    
    fig.suptitle(
        f"FiberNet v4 — Large Deformation FEM Analysis (BeamFrameFEM)\n"
        f"n_pts_per_side={N_PTS_PER_SIDE}, perturbation=±{PERTURBATION}, "
        f"E={E_MODULUS:.0e} Pa, boundary={PCT_BOUNDARY*100:.0f}% each side\n"
        f"{len([v for v in all_results.values() if v])} FEM simulations across "
        f"{len(units_2d)} 2D units + {len(set(k.split('_r')[0].replace('3d_','') for k in results_3d))} 3D units",
        fontsize=12, fontweight="bold", y=0.99)
    
    plot_idx = 1
    
    # ── 2D Structure Panels ──
    for unit in units_2d:
        for test_name in test_names:
            for radius in viz_radii:
                key = f"2d_{unit}_r{radius:.3f}_{test_name}"
                res = all_results.get(key)
                if res is None:
                    continue
                
                ax = fig.add_subplot(n_rows, n_cols, plot_idx)
                plot_idx += 1
                
                def_pos = np.array(res["deformed_positions"])
                elements = res["elements"]
                sigma_total = np.array(res["sigma_total"])
                
                # Color by total stress (normalized)
                if len(sigma_total) > 0:
                    s_max = np.percentile(np.abs(sigma_total), 95)
                    if s_max < 1e-6:
                        s_max = 1.0
                    norm = Normalize(vmin=-s_max, vmax=s_max)
                else:
                    norm = Normalize(vmin=-1, vmax=1)
                    s_max = 1.0
                cmap = cm.coolwarm
                
                # Build unique edge segments from elements + edge_list mapping
                edge_list = res.get("edge_list", list(range(len(sigma_total))))
                segments = []
                colors = []
                
                # Map sigma_total (per unique edge) to all edges
                for e_idx, (i, j) in enumerate(elements):
                    if 0 <= i < len(def_pos) and 0 <= j < len(def_pos):
                        # Find sigma for this edge
                        if e_idx < len(sigma_total):
                            s = sigma_total[e_idx]
                        else:
                            s = 0.0
                        segments.append([def_pos[i], def_pos[j]])
                        colors.append(cmap(norm(np.clip(s, -s_max, s_max))))
                
                if segments:
                    lc = LineCollection(segments, colors=colors, linewidths=0.5)
                    ax.add_collection(lc)
                    ax.autoscale()
                
                ax.set_title(f"{unit} r={radius:.2f} {test_name}\n"
                           f"F={res['total_force_kN']:.1f}kN "
                           f"σ_ax={res['max_axial_stress_MPa']:.1f}MPa "
                           f"σ_bend={res['max_bending_stress_MPa']:.1f}MPa\n"
                           f"prop={res['propagation_ratio']:.2f} "
                           f"B/A={res['bending_dominance']:.1f}",
                           fontsize=5.5, pad=2)
                ax.set_aspect("equal")
                ax.axis("off")
    
    # ── Analysis Row 1: Force-Stretch ──
    ax_force = fig.add_subplot(n_rows, n_cols, plot_idx)
    plot_idx += 1
    for radius in RADII_2D:
        forces_by_stretch = {}
        for key, res in all_results.items():
            if res is None or key.startswith("3d_"):
                continue
            # Parse radius from key
            import re
            m = re.search(r'_r(\d+\.\d+)_', key)
            if m:
                r = float(m.group(1))
            else:
                continue
            if abs(r - radius) < 0.001:
                stretch = res["target_stretch"]
                forces_by_stretch.setdefault(stretch, []).append(res["total_force_kN"])
        if forces_by_stretch:
            stretches = sorted(forces_by_stretch.keys())
            avg = [np.mean(forces_by_stretch[s]) for s in stretches]
            std = [np.std(forces_by_stretch[s]) for s in stretches]
            ax_force.errorbar(stretches, avg, yerr=std, marker="o", markersize=4,
                            label=f"r={radius:.2f}", capsize=3, linewidth=1.5)
    ax_force.set_xlabel("Stretch Ratio (L/L₀)")
    ax_force.set_ylabel("Force (kN)")
    ax_force.set_title("Force-Stretch (avg across units)")
    ax_force.legend(fontsize=7)
    ax_force.grid(True, alpha=0.3)
    ax_force.axvline(1.0, color="gray", ls="--", alpha=0.5)
    
    # ── Analysis Row 1: Propagation bars ──
    ax_prop = fig.add_subplot(n_rows, n_cols, plot_idx)
    plot_idx += 1
    color_map = {"FULL": "#2ecc71", "PARTIAL": "#f39c12", "LOCALIZED": "#e74c3c"}
    prop_labels, prop_values, prop_colors = [], [], []
    for unit in sorted(summary.keys()):
        for t in summary[unit]:
            prop_labels.append(f"{unit}\nr={t['radius']:.2f}\n{t['test']}")
            prop_values.append(t["propagation_ratio"])
            prop_colors.append(color_map.get(t["disp_classification"], "#95a5a6"))
    if prop_values:
        ax_prop.barh(range(len(prop_values)), prop_values, color=prop_colors, height=0.7)
        ax_prop.set_yticks(range(len(prop_values)))
        ax_prop.set_yticklabels(prop_labels, fontsize=4)
        ax_prop.set_xlabel("Propagation Ratio")
        ax_prop.set_title("Deformation Propagation\n(interior_max / boundary_max)")
        ax_prop.axvline(0.5, color="green", ls="--", alpha=0.5)
        ax_prop.axvline(0.2, color="orange", ls="--", alpha=0.5)
    
    # ── Analysis Row 2: Bending dominance heatmap ──
    ax_bend = fig.add_subplot(n_rows, n_cols, plot_idx)
    plot_idx += 1
    all_units = sorted(summary.keys())
    heat_data = np.zeros((len(all_units), len(RADII_2D)))
    for ui, unit in enumerate(all_units):
        for t in summary[unit]:
            ri = min(range(len(RADII_2D)), key=lambda i: abs(RADII_2D[i] - t["radius"]))
            if "stretch_2x" in t["test"]:
                heat_data[ui, ri] = min(t["bending_dominance"], 10.0)
    if heat_data.any():
        im = ax_bend.imshow(heat_data, aspect="auto", cmap="viridis", vmin=0, vmax=10)
        ax_bend.set_xticks(range(len(RADII_2D)))
        ax_bend.set_xticklabels([f"{r:.2f}" for r in RADII_2D], fontsize=7)
        ax_bend.set_yticks(range(len(all_units)))
        ax_bend.set_yticklabels(all_units, fontsize=6)
        ax_bend.set_xlabel("Fiber Radius")
        ax_bend.set_ylabel("Unit Type")
        ax_bend.set_title("Bending/Axial Dominance\n(stretch_2x, capped at 10)")
        plt.colorbar(im, ax=ax_bend, shrink=0.8)
    
    # ── Analysis Row 2: Radius effect ──
    ax_reff = fig.add_subplot(n_rows, n_cols, plot_idx)
    plot_idx += 1
    radius_data = {}
    for key, res in all_results.items():
        if res is None or key.startswith("3d_"):
            continue
        parsed = parse_task_key(key)
        if not parsed:
            continue
        r = parsed["radius"]
        radius_data.setdefault(r, {"prop": [], "force": []})
        radius_data[r]["prop"].append(res["propagation_ratio"])
        radius_data[r]["force"].append(res["total_force_kN"])
    if radius_data:
        radii_sorted = sorted(radius_data.keys())
        props = [np.mean(radius_data[r]["prop"]) for r in radii_sorted]
        props_std = [np.std(radius_data[r]["prop"]) for r in radii_sorted]
        forces = [np.mean(radius_data[r]["force"]) for r in radii_sorted]
        x_pos = np.arange(len(radii_sorted))
        ax2 = ax_reff.twinx()
        ax_reff.bar(x_pos - 0.2, props, 0.4, yerr=props_std,
                   label="Propagation ratio", color="#3498db", alpha=0.8, capsize=3)
        ax2.bar(x_pos + 0.2, forces, 0.4, label="Force (kN)", color="#e74c3c", alpha=0.8)
        ax_reff.set_xticks(x_pos)
        ax_reff.set_xticklabels([f"{r:.2f}" for r in radii_sorted], fontsize=7)
        ax_reff.set_xlabel("Fiber Radius")
        ax_reff.set_ylabel("Propagation Ratio", color="#3498db")
        ax2.set_ylabel("Force (kN)", color="#e74c3c")
        ax_reff.set_title("Radius Effect")
        lines1, labels1 = ax_reff.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_reff.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
        ax_reff.grid(True, alpha=0.3, axis="y")
    
    # ── 3D Structure Panels ──
    for key in sorted(results_3d.keys()):
        res = results_3d[key]
        ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection="3d")
        plot_idx += 1
        
        def_pos = np.array(res["deformed_positions"])
        elements = res["elements"]
        sigma_total = np.array(res["sigma_total"])
        
        s_max = np.percentile(np.abs(sigma_total), 95) if len(sigma_total) > 0 else 1.0
        if s_max < 1e-6:
            s_max = 1.0
        norm = Normalize(vmin=-s_max, vmax=s_max)
        cmap = cm.coolwarm
        
        for e_idx, (i, j) in enumerate(elements):
            if 0 <= i < len(def_pos) and 0 <= j < len(def_pos):
                s = sigma_total[e_idx] if e_idx < len(sigma_total) else 0.0
                color = cmap(norm(np.clip(s, -s_max, s_max)))
                ax.plot3D(
                    [def_pos[i, 0], def_pos[j, 0]],
                    [def_pos[i, 1], def_pos[j, 1]],
                    [def_pos[i, 2], def_pos[j, 2]],
                    color=color, linewidth=0.4, alpha=0.7,
                )
        
        ax.set_title(f"3D: {key}\nF={res['total_force_kN']:.1f}kN "
                    f"prop={res['propagation_ratio']:.2f}", fontsize=6)
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_path = VIZ_DIR / "large_deformation_fem_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n✓ Visualization saved: {out_path}")
    return str(out_path)


def generate_report(all_results, summary):
    report = {
        "timestamp": datetime.now().isoformat(),
        "method": "BeamFrameFEM (beam frame finite element)",
        "parameters": {
            "n_pts_per_side": N_PTS_PER_SIDE,
            "perturbation": PERTURBATION,
            "E_modulus": E_MODULUS,
            "nu": NU,
            "pct_boundary": PCT_BOUNDARY,
            "box_2d": list(BOX_PER_CELL_2D),
            "grid_2d": list(GRID_2D),
            "radii_2d": RADII_2D,
            "stretch_targets": STRETCH_TARGETS,
        },
        "total_simulations": len([v for v in all_results.values() if v]),
        "failed_simulations": len([v for v in all_results.values() if v is None]),
        "summary": {},
        "key_findings": [],
    }
    
    for unit, tests in summary.items():
        report["summary"][unit] = {
            "n_tests": len(tests),
            "full_disp": sum(1 for t in tests if t["disp_classification"] == "FULL"),
            "partial_disp": sum(1 for t in tests if t["disp_classification"] == "PARTIAL"),
            "localized_disp": sum(1 for t in tests if t["disp_classification"] == "LOCALIZED"),
            "avg_propagation": float(np.mean([t["propagation_ratio"] for t in tests])),
            "avg_bending_dominance": float(np.mean([t["bending_dominance"] for t in tests])),
            "avg_force_kN": float(np.mean([t["total_force_kN"] for t in tests])),
        }
    
    findings = []
    for unit, data in report["summary"].items():
        if data["localized_disp"] > data["full_disp"] + data["partial_disp"]:
            findings.append(f"⚠ {unit}: Deformation LOCALIZES — does not propagate to interior")
        elif data["full_disp"] > data["localized_disp"]:
            findings.append(f"✓ {unit}: Deformation FULLY propagates through structure")
        else:
            findings.append(f"~ {unit}: Mixed propagation behavior")
        if data["avg_bending_dominance"] > 2.0:
            findings.append(f"  → {unit}: BENDING-dominated response (beam behavior)")
        else:
            findings.append(f"  → {unit}: AXIAL-dominated response (truss-like)")
    
    report["key_findings"] = findings
    
    report_path = OUTPUT_DIR / "analysis_report_fem.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"✓ Report saved: {report_path}")
    
    return report


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("FiberNet v4 — Large Deformation FEM Test Suite")
    print("=" * 60)
    print(f"Method: BeamFrameFEM (beam frame FEM)")
    print(f"E={E_MODULUS:.0e} Pa, ν={NU}")
    print(f"n_pts={N_PTS_PER_SIDE}, pert=±{PERTURBATION}")
    print(f"2D: {len(UNITS_2D)} units × {len(RADII_2D)} radii × {len(STRETCH_TARGETS)} tests = "
          f"{len(UNITS_2D) * len(RADII_2D) * len(STRETCH_TARGETS)} sims")
    print(f"3D: {len(UNITS_3D)} units × {len(RADII_3D)} radii × {len(STRETCH_TARGETS_3D)} tests = "
          f"{len(UNITS_3D) * len(RADII_3D) * len(STRETCH_TARGETS_3D)} sims")
    print(f"Boundary: {PCT_BOUNDARY*100:.0f}% each side (rigid plate)")
    print(f"Nonlinear solver: |stretch-1| > {NONLINEAR_THRESHOLD}")
    print()
    
    ckpt = load_checkpoint()
    completed = set(ckpt.get("completed", []))
    print(f"Checkpoint: {len(completed)} tasks completed")
    
    all_results = {}
    
    # ═══ PHASE 1: 2D Tests ═══
    print("\n" + "=" * 50)
    print("PHASE 1: 2D FEM Large Deformation Tests")
    print("=" * 50)
    
    for unit in UNITS_2D:
        for radius in RADII_2D:
            for test_name, target_stretch in STRETCH_TARGETS.items():
                task_key = make_task_key("2d", unit, radius, test_name)
                
                if task_key in completed:
                    save_path = OUTPUT_DIR / f"{task_key}.json"
                    if save_path.exists():
                        with open(save_path) as f:
                            all_results[task_key] = json.load(f)
                        print(f"  [skip] {task_key}")
                        continue
                    else:
                        completed.discard(task_key)
                
                nl_tag = " [NL]" if abs(target_stretch - 1.0) > NONLINEAR_THRESHOLD else ""
                print(f"\n[2D] {unit} r={radius:.2f} {test_name} (stretch={target_stretch:.1f}x){nl_tag}")
                
                try:
                    g = generate_2d_structure(unit, radius)
                    from fibernet.sim.accelerated import _graph_to_arrays
                    pos, _, _, _ = _graph_to_arrays(g)
                    x_range = pos[:, 0].max() - pos[:, 0].min()
                    print(f"  Nodes={g.num_nodes}, Edges={g.num_edges}, x_range={x_range:.2f}")
                    
                    if g.num_nodes > MAX_NODES:
                        print(f"  ⚠ Too many nodes, skipping")
                        all_results[task_key] = None
                        ckpt["completed"].append(task_key)
                        save_checkpoint(ckpt)
                        continue
                    
                    result = run_fem_2d(g, target_stretch, radius, label=task_key)
                    
                    if result:
                        print(f"  ✓ F={result['total_force_kN']:.1f}kN, "
                              f"σ_ax={result['max_axial_stress_MPa']:.1f}MPa, "
                              f"σ_bend={result['max_bending_stress_MPa']:.1f}MPa, "
                              f"prop={result['propagation_ratio']:.3f}, "
                              f"int%={result['interior_affected_pct']:.1%}, "
                              f"B/A={result['bending_dominance']:.2f}, "
                              f"{result['elapsed_seconds']:.1f}s")
                        
                        save_path = OUTPUT_DIR / f"{task_key}.json"
                        tmp_path = save_path.with_suffix(".tmp")
                        with open(tmp_path, "w") as f:
                            json.dump(result, f, indent=2)
                        tmp_path.rename(save_path)
                    
                    all_results[task_key] = result
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    traceback.print_exc()
                    all_results[task_key] = None
                
                ckpt["completed"].append(task_key)
                save_checkpoint(ckpt)
                gc.collect()
    
    # ═══ PHASE 2: 3D Tests ═══
    print("\n" + "=" * 50)
    print("PHASE 2: 3D FEM Large Deformation Tests")
    print("=" * 50)
    
    for unit in UNITS_3D:
        for radius in RADII_3D:
            for test_name, target_stretch in STRETCH_TARGETS_3D.items():
                task_key = make_task_key("3d", unit, radius, test_name)
                
                if task_key in completed:
                    save_path = OUTPUT_DIR / f"{task_key}.json"
                    if save_path.exists():
                        with open(save_path) as f:
                            all_results[task_key] = json.load(f)
                        print(f"  [skip] {task_key}")
                        continue
                    else:
                        completed.discard(task_key)
                
                print(f"\n[3D] {unit} r={radius:.2f} {test_name} (stretch={target_stretch:.1f}x)")
                
                try:
                    g = generate_3d_structure(unit, radius)
                    from fibernet.sim.accelerated import _graph_to_arrays
                    pos, _, _, _ = _graph_to_arrays(g)
                    print(f"  Nodes={g.num_nodes}, Edges={g.num_edges}")
                    
                    if g.num_nodes > MAX_NODES:
                        print(f"  ⚠ Too many nodes ({g.num_nodes}), skipping")
                        all_results[task_key] = None
                        ckpt["completed"].append(task_key)
                        save_checkpoint(ckpt)
                        continue
                    
                    result = run_fem_3d(g, target_stretch, radius, label=task_key)
                    
                    if result:
                        print(f"  ✓ F={result['total_force_kN']:.1f}kN, "
                              f"prop={result['propagation_ratio']:.3f}, "
                              f"{result['elapsed_seconds']:.1f}s")
                        
                        save_path = OUTPUT_DIR / f"{task_key}.json"
                        tmp_path = save_path.with_suffix(".tmp")
                        with open(tmp_path, "w") as f:
                            json.dump(result, f, indent=2)
                        tmp_path.rename(save_path)
                    
                    all_results[task_key] = result
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    traceback.print_exc()
                    all_results[task_key] = None
                
                ckpt["completed"].append(task_key)
                save_checkpoint(ckpt)
                gc.collect()
    
    # ═══ PHASE 3: Analysis ═══
    print("\n" + "=" * 50)
    print("PHASE 3: Analysis")
    print("=" * 50)
    
    summary = analyze_results(all_results)
    print_analysis(summary)
    
    # ═══ PHASE 4: Visualization ═══
    print("\n" + "=" * 50)
    print("PHASE 4: Visualization")
    print("=" * 50)
    
    viz_path = generate_visualization(all_results, summary)
    
    # ═══ PHASE 5: Report ═══
    print("\n" + "=" * 50)
    print("PHASE 5: Report")
    print("=" * 50)
    
    report = generate_report(all_results, summary)
    
    print("\n" + "=" * 60)
    print("ALL FEM TESTS COMPLETE")
    print("=" * 60)
    print(f"Results:   {OUTPUT_DIR}")
    print(f"Viz:       {viz_path}")
    print(f"Report:    {OUTPUT_DIR / 'analysis_report_fem.json'}")
    
    print("\nKey Findings:")
    for f in report["key_findings"]:
        print(f"  {f}")
    
    return all_results, summary, report


if __name__ == "__main__":
    main()

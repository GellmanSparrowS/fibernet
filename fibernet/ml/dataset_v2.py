"""
ML dataset generation pipeline for FiberNet.

Generates a labeled dataset of structures with mechanical properties computed
via FEM. Features are extracted from graph topology and geometry.

Design
------
- **Deterministic**: Parameter sweep with full control over structure params.
- **Checkpoint/resume**: Long generations save progress incrementally.
- **Memory-safe**: Batch processing with optional batch_size limit.
- **Standardized output**: numpy arrays + JSON metadata.

Feature Extraction
------------------
For each structure:
- Topological: node count, edge count, connectivity, degree distribution
- Geometric: total length, mean edge length, bounding box, density
- Material: radius distribution, E distribution

Labels (from FEM):
- E* (effective Young's modulus)
- ν* (effective Poisson's ratio)
- G* (effective shear modulus)
- Strain energy
- Max stress/strain

Examples
--------
>>> from fibernet.ml.dataset_v2 import generate_dataset
>>> ds = generate_dataset(
...     units=["honeycomb", "square", "reentrant"],
...     grid_range=[(3,3), (5,5), (7,7)],
...     radius_range=[0.05, 0.1, 0.2],
...     save_dir="datasets/honeycomb_sweep",
... )
>>> print(f"Generated {ds['n_samples']} samples")
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fibernet.core.structure_graph import StructureGraph
from fibernet.gen.pattern import pattern_2d, list_units
from fibernet.sim.fem import BeamFEM


# ======================================================================
# Feature extraction
# ======================================================================

def extract_features(graph: StructureGraph) -> Dict[str, float]:
    """Extract numerical features from a StructureGraph.

    Returns
    -------
    dict
        Feature name → value.
    """
    features = {}

    # Topological
    features["n_nodes"] = graph.num_nodes
    features["n_edges"] = graph.num_edges
    features["density"] = graph.num_edges / max(graph.num_nodes, 1)
    features["n_components"] = len(graph.connected_components())

    # Degree statistics
    degrees = [graph.degree(nid) for nid in graph.nodes]
    if degrees:
        features["mean_degree"] = np.mean(degrees)
        features["max_degree"] = float(np.max(degrees))
        features["min_degree"] = float(np.min(degrees))
        features["std_degree"] = float(np.std(degrees))

    # Geometric
    edge_lengths = graph.edge_lengths()
    if len(edge_lengths) > 0:
        features["total_length"] = float(edge_lengths.sum())
        features["mean_edge_length"] = float(edge_lengths.mean())
        features["std_edge_length"] = float(edge_lengths.std())
        features["min_edge_length"] = float(edge_lengths.min())
        features["max_edge_length"] = float(edge_lengths.max())

    bb_min, bb_max = graph.bounding_box()
    span = bb_max - bb_min
    features["bbox_width"] = float(span[0])
    features["bbox_height"] = float(span[1])
    area = max(span[0] * span[1], 1e-12)
    features["length_density"] = features.get("total_length", 0) / area

    # Radius statistics
    radii = np.array([e.radius for e in graph.edges.values()])
    if len(radii) > 0:
        features["mean_radius"] = float(radii.mean())
        features["std_radius"] = float(radii.std())

    return features


# ======================================================================
# Dataset generation
# ======================================================================

def generate_dataset(
    *,
    units: Optional[List[str]] = None,
    grid_range: Optional[List[Tuple[int, int]]] = None,
    radius_range: Optional[List[float]] = None,
    box: Tuple[float, float] = (10.0, 10.0),
    n_internal: int = 4,
    default_E: float = 1e9,
    applied_strain: float = 0.01,
    save_dir: Optional[str] = None,
    checkpoint_file: Optional[str] = None,
    batch_size: int = 50,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Generate a labeled dataset of structures with mechanical properties.

    Parameters
    ----------
    units : list of str, optional
        Unit types to generate. Defaults to all built-in units.
    grid_range : list of (nx, ny), optional
        Grid sizes to sweep. Defaults to [(3,3), (5,5)].
    radius_range : list of float, optional
        Beam radii to sweep. Defaults to [0.05, 0.1, 0.2].
    box : (w, h)
        Unit cell dimensions.
    n_internal : int
        Internal points per edge.
    default_E : float
        Young's modulus (Pa).
    applied_strain : float
        Uniaxial strain for mechanical testing.
    save_dir : str, optional
        Directory to save dataset files.
    checkpoint_file : str, optional
        Path for checkpoint JSON (for resume support).
    batch_size : int
        Save checkpoint every N samples.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Dataset with keys: features, labels, metadata, n_samples.
    """
    units = units or list_units()
    grid_range = grid_range or [(3, 3), (5, 5)]
    radius_range = radius_range or [0.05, 0.1, 0.2]

    # Initialize or resume
    all_features = []
    all_labels = []
    all_metadata = []
    start_idx = 0

    if checkpoint_file and os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            ckpt = json.load(f)
        all_features = ckpt.get("features", [])
        all_labels = ckpt.get("labels", [])
        all_metadata = ckpt.get("metadata", [])
        start_idx = len(all_features)
        if verbose:
            print(f"Resuming from checkpoint: {start_idx} samples")

    # Generate parameter combinations
    params = []
    for unit_name in units:
        for grid in grid_range:
            for radius in radius_range:
                params.append({
                    "unit": unit_name,
                    "grid": list(grid),
                    "radius": radius,
                    "box": list(box),
                })

    total = len(params)
    if verbose:
        print(f"Total parameter combinations: {total}")
        print(f"Starting from index: {start_idx}")

    t0 = time.time()
    errors = []

    for idx, param in enumerate(params[start_idx:], start=start_idx):
        try:
            # Generate structure
            g = pattern_2d(
                unit=param["unit"],
                box=tuple(param["box"]),
                grid=tuple(param["grid"]),
                radius=param["radius"],
                n_internal=n_internal,
            )

            # Run FEM
            fem = BeamFEM(g, default_E=default_E, default_radius=param["radius"])
            result = fem.uniaxial_tension(strain=applied_strain)

            # Extract features
            feat = extract_features(g)
            feat["unit_type"] = param["unit"]

            # Labels
            label = {
                "E_star": result.effective_youngs_modulus,
                "nu_star": result.effective_poissons_ratio,
                "strain_energy": result.strain_energy,
                "max_stress": float(np.max(np.abs(result.stresses))) if len(result.stresses) > 0 else 0,
                "max_strain": float(np.max(np.abs(result.strains))) if len(result.strains) > 0 else 0,
            }

            all_features.append(feat)
            all_labels.append(label)
            all_metadata.append({
                **param,
                "n_nodes": g.num_nodes,
                "n_edges": g.num_edges,
                "connected": g.is_connected(),
                "fem_solve_time": result.solve_time,
            })

            if verbose and (idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1 - start_idx) / elapsed if elapsed > 0 else 0
                eta = (total - idx - 1) / rate if rate > 0 else 0
                print(f"  [{idx+1}/{total}] {param['unit']} grid={param['grid']} "
                      f"r={param['radius']:.2f} → E*={label['E_star']:.2e} "
                      f"({rate:.1f}/s, ETA {eta:.0f}s)")

        except Exception as exc:
            errors.append({"index": idx, "param": param, "error": str(exc)})
            if verbose:
                print(f"  [{idx+1}/{total}] ERROR: {param['unit']} grid={param['grid']}: {exc}")

        # Checkpoint
        if checkpoint_file and (idx + 1) % batch_size == 0:
            _save_checkpoint(checkpoint_file, all_features, all_labels, all_metadata, errors)

    # Final save
    if checkpoint_file:
        _save_checkpoint(checkpoint_file, all_features, all_labels, all_metadata, errors)

    # Build dataset
    dataset = {
        "n_samples": len(all_features),
        "features": all_features,
        "labels": all_labels,
        "metadata": all_metadata,
        "errors": errors,
    }

    if save_dir:
        _save_dataset(save_dir, dataset)

    if verbose:
        elapsed = time.time() - t0
        print(f"\nDataset generation complete: {len(all_features)} samples in {elapsed:.1f}s")
        if errors:
            print(f"  Errors: {len(errors)}")

    return dataset


def _save_checkpoint(path: str, features, labels, metadata, errors):
    """Save checkpoint for resume."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "features": features,
            "labels": labels,
            "metadata": metadata,
            "errors": errors,
        }, f, indent=1)


def _save_dataset(save_dir: str, dataset: dict):
    """Save full dataset to directory."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save metadata + labels as JSON
    with open(save_path / "dataset.json", "w") as f:
        json.dump({
            "n_samples": dataset["n_samples"],
            "features": dataset["features"],
            "labels": dataset["labels"],
            "metadata": dataset["metadata"],
            "errors": dataset["errors"],
        }, f, indent=1)

    # Save feature matrix as numpy
    feature_names = sorted(set().union(*(f.keys() for f in dataset["features"] if f)))
    feature_names = [n for n in feature_names if n != "unit_type"]
    feature_matrix = np.zeros((dataset["n_samples"], len(feature_names)))
    for i, feat in enumerate(dataset["features"]):
        for j, name in enumerate(feature_names):
            feature_matrix[i, j] = feat.get(name, 0.0)

    np.save(save_path / "features.npy", feature_matrix)
    with open(save_path / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # Save label matrix
    label_names = sorted(set().union(*(l.keys() for l in dataset["labels"] if l)))
    label_matrix = np.zeros((dataset["n_samples"], len(label_names)))
    for i, label in enumerate(dataset["labels"]):
        for j, name in enumerate(label_names):
            label_matrix[i, j] = label.get(name, 0.0)

    np.save(save_path / "labels.npy", label_matrix)
    with open(save_path / "label_names.json", "w") as f:
        json.dump(label_names, f)

    # Save unit type as categorical
    unit_types = [f.get("unit_type", "unknown") for f in dataset["features"]]
    with open(save_path / "unit_types.json", "w") as f:
        json.dump(unit_types, f)

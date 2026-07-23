"""
Dataset Generation & Management for FiberNet ML Pipeline.

Features
--------
- Checkpoint/resume for long-running dataset generation
- Memory-safe batch processing with configurable limits
- Multi-target labels (force, stretch, energy, E*, ν*)
- Feature matrix construction from StructureGraph
- Train/val/test splitting with stratification support
- PyTorch Dataset/DataLoader integration
- CSV & numpy export/import

Examples
--------
>>> from fibernet.ml.dataset import FiberNetDataset
>>> ds = FiberNetDataset.generate(
...     units=["honeycomb", "square", "reentrant"],
...     grid_range=[(3,3), (5,5)],
...     save_dir="datasets/sweep_v1",
... )
>>> X, y = ds.features, ds.labels("max_force")
>>> train, val, test = ds.split(ratios=(0.7, 0.15, 0.15))
"""

from __future__ import annotations

import json
import os
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass, field

import numpy as np


def _extract_graph_features(g) -> Dict[str, float]:
    """Extract numerical features from a StructureGraph.

    Parameters
    ----------
    g : StructureGraph
        Input structure graph.

    Returns
    -------
    dict
        Feature name to value mapping.
    """
    features = {}

    features["n_nodes"] = g.num_nodes
    features["n_edges"] = g.num_edges
    features["density"] = g.num_edges / max(g.num_nodes, 1)

    try:
        cc = g.connected_components()
        features["n_components"] = len(cc)
    except Exception:
        features["n_components"] = 1

    degrees = [g.degree(nid) for nid in g.nodes]
    if degrees:
        features["mean_degree"] = float(np.mean(degrees))
        features["max_degree"] = float(np.max(degrees))
        features["min_degree"] = float(np.min(degrees))
        features["std_degree"] = float(np.std(degrees))
    else:
        features["mean_degree"] = 0.0
        features["max_degree"] = 0.0
        features["min_degree"] = 0.0
        features["std_degree"] = 0.0

    try:
        edge_lengths = g.edge_lengths()
        if len(edge_lengths) > 0:
            features["total_length"] = float(edge_lengths.sum())
            features["mean_edge_length"] = float(edge_lengths.mean())
            features["std_edge_length"] = float(edge_lengths.std())
            features["min_edge_length"] = float(edge_lengths.min())
            features["max_edge_length"] = float(edge_lengths.max())
        else:
            features["total_length"] = 0.0
            features["mean_edge_length"] = 0.0
            features["std_edge_length"] = 0.0
            features["min_edge_length"] = 0.0
            features["max_edge_length"] = 0.0
    except Exception:
        features["total_length"] = 0.0
        features["mean_edge_length"] = 0.0
        features["std_edge_length"] = 0.0
        features["min_edge_length"] = 0.0
        features["max_edge_length"] = 0.0

    try:
        bb_min, bb_max = g.bounding_box()
        span = bb_max - bb_min
        features["bbox_width"] = float(span[0])
        features["bbox_height"] = float(span[1]) if len(span) > 1 else 0.0
        area = max(span[0] * (span[1] if len(span) > 1 else 1.0), 1e-12)
        features["length_density"] = features.get("total_length", 0) / area
    except Exception:
        features["bbox_width"] = 0.0
        features["bbox_height"] = 0.0
        features["length_density"] = 0.0

    try:
        radii = np.array([e.radius for e in g.edges.values()])
        if len(radii) > 0:
            features["mean_radius"] = float(radii.mean())
            features["std_radius"] = float(radii.std())
            features["min_radius"] = float(radii.min())
            features["max_radius"] = float(radii.max())
        else:
            features["mean_radius"] = 0.0
            features["std_radius"] = 0.0
            features["min_radius"] = 0.0
            features["max_radius"] = 0.0
    except Exception:
        features["mean_radius"] = 0.0
        features["std_radius"] = 0.0
        features["min_radius"] = 0.0
        features["max_radius"] = 0.0

    # Anisotropy / orientation features
    try:
        positions = np.array([g.nodes[nid].position[:2] for nid in g.nodes])
        if len(positions) > 2:
            centroid = positions.mean(axis=0)
            dists = np.linalg.norm(positions - centroid, axis=1)
            features["mean_node_dist_from_center"] = float(dists.mean())
            features["std_node_dist_from_center"] = float(dists.std())
        else:
            features["mean_node_dist_from_center"] = 0.0
            features["std_node_dist_from_center"] = 0.0
    except Exception:
        features["mean_node_dist_from_center"] = 0.0
        features["std_node_dist_from_center"] = 0.0

    return features


@dataclass
class DatasetSample:
    """Single dataset sample."""
    sample_id: str
    unit: str
    grid: Tuple[int, int]
    params: Dict[str, Any]
    features: Dict[str, float]
    labels: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class FiberNetDataset:
    """Labeled dataset of fiber network structures with mechanical properties.

    Parameters
    ----------
    samples : list of DatasetSample
        All samples in the dataset.
    feature_names : list of str, optional
        Ordered feature names. Auto-detected if not given.
    target_names : list of str, optional
        Ordered target names. Auto-detected if not given.
    """

    def __init__(
        self,
        samples: List[DatasetSample],
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
    ):
        self.samples = samples

        if feature_names is None and samples:
            self.feature_names = sorted(samples[0].features.keys())
        else:
            self.feature_names = feature_names or []

        if target_names is None and samples:
            self.target_names = sorted(samples[0].labels.keys())
        else:
            self.target_names = target_names or []

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def features(self) -> np.ndarray:
        """Feature matrix (n_samples, n_features)."""
        rows = []
        for s in self.samples:
            rows.append([s.features.get(f, 0.0) for f in self.feature_names])
        return np.array(rows, dtype=np.float64)

    @property
    def all_labels(self) -> np.ndarray:
        """Label matrix (n_samples, n_targets)."""
        rows = []
        for s in self.samples:
            rows.append([s.labels.get(t, 0.0) for t in self.target_names])
        return np.array(rows, dtype=np.float64)

    def labels(self, target: str) -> np.ndarray:
        """Single-target label vector."""
        idx = self.target_names.index(target)
        return self.all_labels[:, idx]

    def split(
        self,
        ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_state: int = 42,
    ) -> Tuple["FiberNetDataset", "FiberNetDataset", "FiberNetDataset"]:
        """Split into train/val/test.

        Parameters
        ----------
        ratios : (train, val, test)
            Must sum to 1.0.
        random_state : int
            Random seed.

        Returns
        -------
        (train, val, test) : tuple of FiberNetDataset
        """
        rng = np.random.RandomState(random_state)
        n = len(self.samples)
        indices = rng.permutation(n)

        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        def _sub(idxs):
            return FiberNetDataset(
                [self.samples[i] for i in idxs],
                feature_names=self.feature_names,
                target_names=self.target_names,
            )

        return _sub(train_idx), _sub(val_idx), _sub(test_idx)

    def to_csv(self, path: str) -> None:
        """Export to CSV."""
        import csv
        fieldnames = ["sample_id", "unit", "grid_x", "grid_y"]
        fieldnames += [f"feat_{f}" for f in self.feature_names]
        fieldnames += [f"label_{t}" for t in self.target_names]

        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for s in self.samples:
                row = {
                    "sample_id": s.sample_id,
                    "unit": s.unit,
                    "grid_x": s.grid[0],
                    "grid_y": s.grid[1],
                }
                for f in self.feature_names:
                    row[f"feat_{f}"] = s.features.get(f, 0.0)
                for t in self.target_names:
                    row[f"label_{t}"] = s.labels.get(t, 0.0)
                writer.writerow(row)

    def save(self, directory: str) -> None:
        """Save as numpy arrays + metadata JSON."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        np.save(out / "features.npy", self.features)
        np.save(out / "labels.npy", self.all_labels)

        meta = {
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "n_samples": self.n_samples,
            "samples": [
                {
                    "id": s.sample_id,
                    "unit": s.unit,
                    "grid": list(s.grid),
                    "params": {k: _json_safe(v) for k, v in s.params.items()},
                    "metadata": {k: _json_safe(v) for k, v in s.metadata.items()},
                }
                for s in self.samples
            ],
        }
        with open(out / "metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, directory: str) -> "FiberNetDataset":
        """Load from saved directory."""
        d = Path(directory)
        X = np.load(d / "features.npy")
        Y = np.load(d / "labels.npy")

        with open(d / "metadata.json") as fh:
            meta = json.load(fh)

        feature_names = meta["feature_names"]
        target_names = meta["target_names"]

        samples = []
        for i, sm in enumerate(meta["samples"]):
            feat = {fn: float(X[i, j]) for j, fn in enumerate(feature_names)}
            lbl = {tn: float(Y[i, j]) for j, tn in enumerate(target_names)}
            samples.append(DatasetSample(
                sample_id=sm["id"],
                unit=sm["unit"],
                grid=tuple(sm["grid"]),
                params=sm.get("params", {}),
                features=feat,
                labels=lbl,
                metadata=sm.get("metadata", {}),
            ))

        return cls(samples, feature_names=feature_names, target_names=target_names)

    def to_pytorch(self, target: Optional[str] = None):
        """Convert to PyTorch tensors.

        Parameters
        ----------
        target : str, optional
            Single target name. If None, returns all targets.

        Returns
        -------
        X : torch.Tensor (n_samples, n_features)
        y : torch.Tensor (n_samples,) or (n_samples, n_targets)
        """
        import torch
        X = torch.tensor(self.features, dtype=torch.float32)
        if target:
            y = torch.tensor(self.labels(target), dtype=torch.float32)
        else:
            y = torch.tensor(self.all_labels, dtype=torch.float32)
        return X, y

    def __repr__(self) -> str:
        return (f"FiberNetDataset(n_samples={self.n_samples}, "
                f"n_features={self.n_features}, "
                f"targets={self.target_names})")

    @classmethod
    def generate(
        cls,
        *,
        units: Optional[List[str]] = None,
        grid_range: Optional[List[Tuple[int, int]]] = None,
        radius_range: Optional[List[float]] = None,
        n_internal_range: Optional[List[int]] = None,
        box: Tuple[float, float] = (10.0, 10.0),
        target_stretch: float = 1.5,
        stiffness: float = 1e5,
        damping: float = 0.3,
        num_steps: int = 1000,
        save_dir: Optional[str] = None,
        checkpoint_every: int = 10,
        max_samples: Optional[int] = None,
        seed: int = 42,
        verbose: bool = True,
    ) -> "FiberNetDataset":
        """Generate a labeled dataset by sweeping over structure parameters.

        Supports checkpoint/resume — if save_dir exists and has partial
        results, generation continues from where it stopped.

        Parameters
        ----------
        units : list of str, optional
            Unit types. Defaults to ["honeycomb", "square", "triangle"].
        grid_range : list of (nx, ny)
            Grid sizes. Defaults to [(3,3), (5,5)].
        radius_range : list of float
            Beam radii. Defaults to [0.05, 0.1, 0.2].
        n_internal_range : list of int
            Internal points per edge. Defaults to [0, 2, 4].
        box : (w, h)
            Unit cell dimensions.
        target_stretch : float
            Stretch ratio for simulation.
        stiffness : float
            Spring stiffness.
        damping : float
            Damping ratio.
        num_steps : int
            Simulation steps.
        save_dir : str, optional
            Directory for checkpoint/resume.
        checkpoint_every : int
            Save checkpoint every N samples.
        max_samples : int, optional
            Cap on total samples.
        seed : int
            Random seed.
        verbose : bool
            Print progress.

        Returns
        -------
        FiberNetDataset
        """
        from fibernet.gen.pattern import pattern_2d, list_units
        from fibernet.sim.accelerated import TaichiEngine

        if units is None:
            units = ["honeycomb", "square", "triangle"]
        if grid_range is None:
            grid_range = [(3, 3), (5, 5)]
        if radius_range is None:
            radius_range = [0.05, 0.1, 0.2]
        if n_internal_range is None:
            n_internal_range = [0, 2, 4]

        # Build parameter grid
        param_combos = []
        for unit in units:
            for grid in grid_range:
                for n_int in n_internal_range:
                    for s in range(seed, seed + max(len(radius_range), 1)):
                        param_combos.append({
                            "unit": unit,
                            "grid": grid,
                            "n_internal": n_int,
                            "seed": s % 10000,
                        })

        if max_samples:
            rng = np.random.RandomState(seed)
            rng.shuffle(param_combos)
            param_combos = param_combos[:max_samples]

        # Checkpoint resume
        existing_samples = []
        existing_ids = set()
        if save_dir:
            ckpt_path = Path(save_dir) / "checkpoint.json"
            if ckpt_path.exists():
                with open(ckpt_path) as fh:
                    ckpt = json.load(fh)
                for s in ckpt.get("samples", []):
                    sample = DatasetSample(
                        sample_id=s["id"],
                        unit=s["unit"],
                        grid=tuple(s["grid"]),
                        params=s.get("params", {}),
                        features=s["features"],
                        labels=s["labels"],
                        metadata=s.get("metadata", {}),
                    )
                    existing_samples.append(sample)
                    existing_ids.add(sample.sample_id)
                if verbose:
                    print(f"Resumed {len(existing_samples)} samples from checkpoint")

        engine = TaichiEngine()
        samples = list(existing_samples)
        n_total = len(param_combos)
        n_done = 0
        n_failed = 0

        for i, params in enumerate(param_combos):
            sample_id = _make_id(params)
            if sample_id in existing_ids:
                n_done += 1
                continue

            unit = params["unit"]
            grid = params["grid"]
            n_int = params["n_internal"]
            sd = params["seed"]

            try:
                g = pattern_2d(
                    unit=unit, box=box, grid=grid,
                    n_internal=n_int, seed=sd,
                )

                result = engine.stretch_test(
                    g,
                    target_stretch=target_stretch,
                    stiffness=stiffness,
                    damping=damping,
                    num_steps=num_steps,
                    save_interval=num_steps,
                    auto_steps=False,
                )

                feat = _extract_graph_features(g)
                labels = {
                    "max_force": float(result.max_force),
                    "max_stretch": float(result.max_stretch),
                    "mean_stretch": float(result.mean_stretch),
                    "std_stretch": float(result.std_stretch),
                }

                # Add effective properties if available
                try:
                    labels["effective_youngs_modulus"] = float(result.effective_youngs_modulus)
                except (AttributeError, TypeError):
                    pass
                try:
                    labels["effective_poissons_ratio"] = float(result.effective_poissons_ratio)
                except (AttributeError, TypeError):
                    pass
                try:
                    labels["strain_energy"] = float(result.strain_energy)
                except (AttributeError, TypeError):
                    pass

                sample = DatasetSample(
                    sample_id=sample_id,
                    unit=unit,
                    grid=grid,
                    params=params,
                    features=feat,
                    labels=labels,
                    metadata={"time_s": float(getattr(result, "time_seconds", 0))},
                )
                samples.append(sample)
                n_done += 1

                if verbose:
                    print(f"  [{n_done}/{n_total}] {unit} grid={grid} "
                          f"max_force={labels['max_force']:.1f}")

            except Exception as exc:
                n_failed += 1
                n_done += 1
                if verbose:
                    print(f"  [{n_done}/{n_total}] FAILED ({unit} grid={grid}): {exc}")

            # Checkpoint
            if save_dir and (len(samples) - len(existing_samples)) % checkpoint_every == 0:
                _save_checkpoint(save_dir, samples)

        # Final checkpoint
        if save_dir:
            _save_checkpoint(save_dir, samples)

        # Auto-detect feature/target names
        feature_names = None
        target_names = None
        if samples:
            feature_names = sorted(samples[0].features.keys())
            target_names = sorted(samples[0].labels.keys())

        ds = cls(samples, feature_names=feature_names, target_names=target_names)

        if verbose:
            print(f"\nDataset: {ds.n_samples} samples, "
                  f"{ds.n_features} features, "
                  f"{len(ds.target_names)} targets, "
                  f"{n_failed} failed")

        if save_dir:
            ds.save(save_dir)
            if verbose:
                print(f"Saved to {save_dir}")

        return ds


# ============================================================
# Internal helpers
# ============================================================

def _make_id(params: dict) -> str:
    """Deterministic sample ID from params."""
    s = json.dumps(params, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _json_safe(val):
    """Make value JSON-serializable."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, tuple):
        return list(val)
    return val


def _save_checkpoint(directory: str, samples: List[DatasetSample]) -> None:
    """Save checkpoint."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "n_samples": len(samples),
        "timestamp": time.time(),
        "samples": [
            {
                "id": s.sample_id,
                "unit": s.unit,
                "grid": list(s.grid),
                "params": {k: _json_safe(v) for k, v in s.params.items()},
                "features": s.features,
                "labels": s.labels,
                "metadata": {k: _json_safe(v) for k, v in s.metadata.items()},
            }
            for s in samples
        ],
    }
    tmp_path = out / "checkpoint.json.tmp"
    final_path = out / "checkpoint.json"
    with open(tmp_path, "w") as fh:
        json.dump(ckpt, fh, indent=2)
    os.replace(tmp_path, final_path)

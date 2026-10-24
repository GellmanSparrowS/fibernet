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
...     n_internal_range=[0],
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
import logging
import shutil
import stat
import tempfile
import warnings
from pathlib import Path
from numbers import Integral, Real
from fractions import Fraction
from collections.abc import Mapping
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


def _source_group_split_indices(n_samples, ratios, random_state, groups):
    """Validated three-way indices using the existing sklearn group splitter."""
    if (not isinstance(groups, (list, tuple)) or len(groups) != n_samples
            or any(not isinstance(group, str) or not group.strip() for group in groups)):
        raise ValueError("groups must be a list/tuple of nonempty string labels, one per Dataset row")
    groups = tuple(groups)
    n_groups = len(set(groups))
    if n_groups < 3:
        raise ValueError("Group splitting requires at least three distinct source groups")
    if (not isinstance(ratios, (list, tuple)) or len(ratios) != 3
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                   for value in ratios)):
        raise ValueError("Group ratios must be three positive finite fractions summing to 1")
    try:
        fractions = tuple(float(value) for value in ratios)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("Group ratios must be three positive finite fractions summing to 1") from exc
    if (not all(np.isfinite(value) and 0 < value < 1 for value in fractions)
            or not np.isclose(sum(fractions), 1.0, rtol=0.0, atol=1e-12)):
        raise ValueError("Group ratios must be three positive finite fractions summing to 1")
    if (isinstance(random_state, (bool, np.bool_))
            or not isinstance(random_state, Integral)
            or not 0 <= random_state <= 2**32 - 1):
        raise ValueError("Group random_state must be an integer in [0, 2**32 - 1]")
    # Exact arithmetic over the validated floats' shortest decimal spellings
    # avoids adding a group for intermediate roundoff (e.g. 2.0000000000000004).
    weights = tuple(Fraction(str(value)) for value in fractions)
    def ceiling(value):
        return -(-value.numerator // value.denominator)
    n_test = ceiling(n_groups * weights[2])
    n_remaining = n_groups - n_test
    n_val = ceiling(n_remaining * weights[1] / (weights[0] + weights[1]))
    if n_test < 1 or n_val < 1 or n_remaining - n_val < 1:
        raise ValueError("Group ratios require more source groups for nonempty train/validation/test")

    try:
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError as exc:
        raise ImportError("Group dataset splitting requires optional scikit-learn; row mode does not") from exc
    # NumPy string coercion can merge distinct labels such as "source" and
    # "source\0". Sorted exact string identities give deterministic integer
    # codes, independent of row order, without exposing strings to the backend.
    code_by_group = {group: code for code, group in enumerate(sorted(set(groups)))}
    group_codes = np.asarray([code_by_group[group] for group in groups], dtype=np.int64)
    rng = np.random.RandomState(int(random_state))
    indices = np.arange(n_samples)
    development, test = next(GroupShuffleSplit(
        n_splits=1, test_size=n_test, random_state=rng).split(indices, groups=group_codes))
    local_train, local_val = next(GroupShuffleSplit(
        n_splits=1, test_size=n_val, random_state=rng).split(
            development, groups=group_codes[development]))
    train, val = development[local_train], development[local_val]
    parts = (train, val, test)
    rows = [int(index) for part in parts for index in part]
    if any(len(part) == 0 for part in parts) or sorted(rows) != list(range(n_samples)):
        raise ValueError("Group split must exactly partition all rows into three nonempty sets")
    sets = [{groups[index] for index in part} for part in parts]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Source groups overlap across train/validation/test")
    if tuple(len(part) for part in sets) != (n_remaining - n_val, n_val, n_test):
        raise ValueError("Group split does not match the requested group counts")
    return parts


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

    def _selected_matrix(self, attribute: str, names: Sequence[str]) -> np.ndarray:
        """Validate the current selection without inventing missing values."""
        kind = "feature" if attribute == "features" else "target"
        if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
            raise ValueError(f"Selected {kind} names must be a sequence of names")
        seen = set()
        for column, name in enumerate(names):
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"Invalid selected {kind} name at column {column}: {name!r}; "
                    "expected a nonempty string")
            if name in seen:
                raise ValueError(f"Duplicate selected {kind} name: {name!r}")
            seen.add(name)

        values = np.empty((self.n_samples, len(names)), dtype=np.float64)
        for row, sample in enumerate(self.samples):
            mapping = getattr(sample, attribute)
            if names and not isinstance(mapping, Mapping):
                raise ValueError(
                    f"sample {sample.sample_id!r} (row {row}): selected {kind} "
                    "values must be a mapping")
            for column, name in enumerate(names):
                context = f"sample {sample.sample_id!r} (row {row}), {kind} {name!r}"
                if name not in mapping:
                    raise ValueError(f"{context}: missing selected value")
                value = mapping[name]
                if value is None:
                    raise ValueError(f"{context}: selected value is None")
                # Preserve binary features represented by Python or NumPy bool.
                if not isinstance(value, (Real, np.bool_)):
                    raise ValueError(f"{context}: selected value must be a real numeric scalar")
                try:
                    value = float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"{context}: selected value must be finite") from exc
                if not np.isfinite(value):
                    raise ValueError(f"{context}: selected value must be finite")
                values[row, column] = value
        return values

    @property
    def features(self) -> np.ndarray:
        """Finite feature matrix (n_samples, n_features); missing values fail."""
        return self._selected_matrix("features", self.feature_names)

    @property
    def all_labels(self) -> np.ndarray:
        """Finite label matrix (n_samples, n_targets); missing values fail."""
        return self._selected_matrix("labels", self.target_names)

    def labels(self, target: str) -> np.ndarray:
        """Single-target label vector."""
        idx = self.target_names.index(target)
        return self.all_labels[:, idx]

    def split(
        self,
        ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_state: int = 42,
        *,
        method: str = "row",
        groups: Optional[Sequence[str]] = None,
    ) -> Tuple["FiberNetDataset", "FiberNetDataset", "FiberNetDataset"]:
        """Split into train/validation/test with explicit row or source-group intent.

        Parameters
        ----------
        ratios : (train, val, test)
            Requested fractions, summing to 1.0. Group mode requires three
            positive finite fractions and rounds by group count, not row count:
            reserve ceil(n_groups * test), then ceil(remaining_groups *
            val / (train + val)), using exact arithmetic on the validated floats'
            shortest decimal spellings. No near-integer snapping is performed.
            Infeasible nonempty partitions are rejected.
        random_state : int
            Random seed. Group mode requires an integer in [0, 2**32 - 1].
        method : {"row", "group"}, keyword-only
            "row" preserves the historical row-random behavior, including its
            rounding and validation behavior; it is not source-group-safe.
            "group" uses optional scikit-learn GroupShuffleSplit twice and
            preserves input row order inside each returned partition.
        groups : list or tuple of str, keyword-only
            Required only for group mode: one nonempty source label per row.
            All related material/scale/geometry variants must share their parent
            source label. Labels are caller assertions, not inferred or verified
            from metadata; unique sample IDs do not establish independence.
            Exact string identities are preserved, with no trimming or Unicode
            normalization. Sorted labels are encoded as integers for splitting.

        Returns
        -------
        (train, val, test) : tuple of FiberNetDataset
            Samples and their metadata are retained as in the historical method.
            No exposure history is reset: calling a partition "test" does not
            make previously evaluated data untouched.
        """
        if not isinstance(method, str) or method not in ("row", "group"):
            raise ValueError("method must be 'row' or 'group'")
        if method == "group":
            train_idx, val_idx, test_idx = _source_group_split_indices(
                len(self.samples), ratios, random_state, groups)
        else:
            if groups is not None:
                raise ValueError("groups requires method='group'; row mode is not group-safe")
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
        """Stage a complete CSV before replacing a regular destination file.

        A write or replacement exception preserves the previous destination.
        Existing ordinary permission bits are retained; new files are private.
        This does not promise crash durability or concurrent-writer safety.
        """
        import csv
        # Check both matrices before creating a temporary file.
        self.features
        self.all_labels
        out = Path(path)
        mode = _dataset_file_mode(out)
        fieldnames = ["sample_id", "unit", "grid_x", "grid_y"]
        fieldnames += [f"feat_{f}" for f in self.feature_names]
        fieldnames += [f"label_{t}" for t in self.target_names]

        fd, temporary = tempfile.mkstemp(prefix=f".{out.name}.save-", dir=out.parent)
        temporary = Path(temporary)
        try:
            with os.fdopen(fd, "w", newline="") as fh:
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
                        row[f"feat_{f}"] = s.features[f]
                    for t in self.target_names:
                        row[f"label_{t}"] = s.labels[t]
                    writer.writerow(row)
            os.chmod(temporary, 0o600 if mode is None else mode)
            os.replace(temporary, out)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except Exception as exc:
                _warn_dataset_cleanup(temporary, exc)

    def save(self, directory: str) -> None:
        """Save the legacy NPY/JSON format with staged writes and rollback.

        Ordinary exceptions preserve the old destination when rollback succeeds.
        If recovery also fails, a RuntimeError identifies retained originals and
        the destination must not be loaded until recovered. Unrelated files and
        checkpoints are left alone. No crash-atomic multi-file or concurrent
        reader/writer guarantee is provided.
        """
        X, Y = self.features, self.all_labels
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
        # Preserve the existing conversion/JSON semantics, but fail before writes.
        metadata = json.dumps(meta, indent=2)
        _save_dataset_files(Path(directory), X, Y, metadata)

    @classmethod
    def load(cls, directory: str) -> "FiberNetDataset":
        """Load a structurally consistent, finite real legacy NPY/JSON dataset.

        Historical zero-row ``(0,)`` arrays remain supported. Validation does
        not establish common-generation provenance or scientific validity.
        """
        d = Path(directory)
        with open(d / "metadata.json") as fh:
            meta = json.load(fh)

        if not isinstance(meta, dict):
            raise ValueError("metadata.json: expected an object")
        for key in ("feature_names", "target_names", "n_samples", "samples"):
            if key not in meta:
                raise ValueError(f"metadata.json: missing required field {key!r}")

        feature_names = meta["feature_names"]
        target_names = meta["target_names"]
        for kind, names in (("feature", feature_names), ("target", target_names)):
            if not isinstance(names, list):
                raise ValueError(f"metadata.json: {kind} names must be an array")
            seen = set()
            for column, name in enumerate(names):
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        f"metadata.json: invalid {kind} name at column {column}: "
                        f"{name!r}; expected a nonempty string")
                if name in seen:
                    raise ValueError(
                        f"metadata.json: duplicate {kind} name {name!r} at column {column}")
                seen.add(name)

        n_samples = meta["n_samples"]
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 0:
            raise ValueError("metadata.json: n_samples must be a nonnegative integer (not bool)")
        records = meta["samples"]
        if not isinstance(records, list):
            raise ValueError("metadata.json: samples must be an array")
        if len(records) != n_samples:
            raise ValueError(
                f"metadata.json: n_samples={n_samples} does not match samples length {len(records)}")
        for row, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"metadata.json: samples[{row}] must be an object")
            for key in ("id", "unit", "grid"):
                if key not in record:
                    raise ValueError(
                        f"metadata.json: samples[{row}] is missing required field {key!r}")
            if not isinstance(record["grid"], list):
                raise ValueError(f"metadata.json: samples[{row}].grid must be an array")

        X = _load_dataset_matrix(d / "features.npy", (n_samples, len(feature_names)))
        Y = _load_dataset_matrix(d / "labels.npy", (n_samples, len(target_names)))

        samples = []
        for i, sm in enumerate(records):
            feat = {
                fn: _load_dataset_value(X[i, j], "features.npy", sm["id"], i, "feature", fn)
                for j, fn in enumerate(feature_names)
            }
            lbl = {
                tn: _load_dataset_value(Y[i, j], "labels.npy", sm["id"], i, "target", tn)
                for j, tn in enumerate(target_names)
            }
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
        geometry_policy: str = "reject_internal_points",
    ) -> "FiberNetDataset":
        """Generate a labeled dataset by sweeping over structure parameters.

        Supports checkpoint/resume — if save_dir exists and has partial
        results with the exact versioned run binding, generation continues from
        where it stopped. Legacy/unbound checkpoints and changed sampling plans
        are rejected. Sampling v2 uses one unchanged base seed across radii;
        historical radius-count/seed sweeps are not silently migrated.

        Parameters
        ----------
        units : list of str, optional
            Unit types. Defaults to ["honeycomb", "square", "triangle"].
        grid_range : list of (nx, ny)
            Grid sizes. Defaults to [(3,3), (5,5)].
        radius_range : list of float
            Sampling v2: explicit Cartesian radius dimension; default [0.05, 0.1, 0.2].
        n_internal_range : list of int
            Internal points per edge. Defaults to [0, 2, 4].
            This default is unchanged: strict geometry acceptance rejects a
            selected built-in plan containing nonzero internal-point counts.
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
        geometry_policy : {"reject_internal_points", "endpoint_chord"}
            Strict by default: reject nonempty edge-owned internal points before
            solving. Explicit endpoint_chord ignores those points mechanically;
            it does not preserve or establish curved-beam mechanics. Actual
            intermediate graph nodes connected by straight edges remain eligible.
            The policy and helper implementation are bound into checkpoints.

        Returns
        -------
        FiberNetDataset
        """
        from fibernet.gen import pattern
        from . import _generate_contract as contract
        from fibernet.sim._taichi_geometry import preflight_taichi_geometry, GeometryPolicyError

        declaration, solver, param_combos = contract.make_plan(
            units=units, grid_range=grid_range, radius_range=radius_range,
            n_internal_range=n_internal_range, box=box, target_stretch=target_stretch,
            stiffness=stiffness, damping=damping, num_steps=num_steps,
            max_samples=max_samples, seed=seed, checkpoint_every=checkpoint_every,
            geometry_policy=geometry_policy)
        if any(u.lower() not in pattern._UNIT_FACTORIES for u in declaration['units']):
            raise ValueError('unknown generation unit')
        run_binding = None
        existing_samples = []
        existing_ids = set()
        checkpoint_loaded = False
        if save_dir is not None:
            recipe = contract.implementation_recipe(pattern, declaration['units'])
            run_binding = contract.binding(declaration, solver, param_combos, recipe)
            ckpt_path = Path(save_dir) / "checkpoint.json"
            _dataset_file_mode(ckpt_path)
            if ckpt_path.exists():
                rows = contract.read_checkpoint(ckpt_path, run_binding, param_combos)
                checkpoint_loaded = True
                for row in rows:
                    sample = DatasetSample(
                        sample_id=row['id'], unit=row['unit'], grid=tuple(row['grid']),
                        params=row['params'], features=row['features'], labels=row['labels'],
                        metadata=row.get('metadata', {}))
                    existing_samples.append(sample)
                    existing_ids.add(sample.sample_id)
                if verbose:
                    print(f"Resumed {len(existing_samples)} samples from checkpoint")

            elif any((Path(save_dir) / name).exists()
                     for name in ('features.npy', 'labels.npy', 'metadata.json')):
                raise ValueError('existing Dataset has no bound checkpoint; use a new save_dir')
        # Preserve checkpoint mismatch precedence, but reject an incompatible
        # selected built-in plan before any graph, provider, or output writes.
        contract.validate_plan_geometry(pattern, param_combos, solver['geometry_policy'])
        complete_resume = checkpoint_loaded and all(
            contract.sample_id(p) in existing_ids for p in param_combos)

        # Optional engine is imported/constructed only for a genuinely pending job.
        engine = None
        samples = list(existing_samples)
        n_total = len(param_combos)
        n_done = 0
        n_failed = 0

        for i, params in enumerate(param_combos):
            sample_id = contract.sample_id(params)
            if sample_id in existing_ids:
                n_done += 1
                continue

            unit = params["unit"]
            grid = params["grid"]
            n_int = params["n_internal"]
            sd = params["seed"]

            try:
                g = pattern.pattern_2d(
                    unit=unit, box=params['box'], grid=grid, radius=params['radius'],
                    n_internal=n_int, seed=sd,
                )
                geometry_record = preflight_taichi_geometry(
                    g, geometry_policy=solver['geometry_policy'])
                if engine is None:
                    from fibernet.sim.accelerated import TaichiEngine
                    engine = TaichiEngine()

                result = engine.stretch_test(
                    g,
                    target_stretch=solver['target_stretch'],
                    stiffness=solver['stiffness'],
                    damping=solver['damping'],
                    num_steps=solver['num_steps'],
                    save_interval=solver['save_interval'],
                    auto_steps=False,
                    geometry_policy=solver['geometry_policy'],
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
                    metadata={"time_s": float(getattr(result, "time_seconds", 0)),
                              "taichi_geometry": geometry_record},
                )
                # Reject nonfinite accepted rows before they enter a checkpoint.
                contract.validate_rows([{
                    'id': sample_id, 'unit': unit, 'grid': list(grid), 'params': params,
                    'features': feat, 'labels': labels, 'metadata': sample.metadata,
                }], param_combos)
                samples.append(sample)
                existing_ids.add(sample_id)
                n_done += 1

                if verbose:
                    print(f"  [{n_done}/{n_total}] {unit} grid={grid} "
                          f"max_force={labels['max_force']:.1f}")

            except GeometryPolicyError:
                # Geometry-policy rejection must not become a silently partial
                # dataset, including when a registered custom factory is used.
                raise
            except Exception as exc:
                n_failed += 1
                n_done += 1
                if verbose:
                    print(f"  [{n_done}/{n_total}] FAILED ({unit} grid={grid}): {exc}")

            # Checkpoint
            if save_dir is not None and (len(samples) - len(existing_samples)) % checkpoint_every == 0:
                _save_checkpoint(save_dir, samples, run_binding)

        # Final checkpoint
        if save_dir is not None and not complete_resume:
            _save_checkpoint(save_dir, samples, run_binding)

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

        if save_dir is not None and not complete_resume:
            ds.save(save_dir)
            if verbose:
                print(f"Saved to {save_dir}")

        return ds


# ============================================================
# Internal helpers
# ============================================================


def _load_dataset_matrix(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    """Validate a saved matrix without coercing or copying its data."""
    values = np.load(path, allow_pickle=False)
    if isinstance(values, np.lib.npyio.NpzFile):
        values.close()
        raise ValueError(f"{path.name}: expected a single NPY array, got an NPZ archive")
    if not isinstance(values, np.ndarray):
        raise ValueError(f"{path.name}: expected a single NPY array")
    if values.dtype.kind not in "biuf":
        raise ValueError(
            f"{path.name}: expected a real numeric or boolean dtype, got {values.dtype}")
    # The original writer used np.array([]), even with declared column names.
    legacy_empty = shape[0] == 0 and values.shape == (0,)
    if values.shape != shape and not legacy_empty:
        raise ValueError(f"{path.name}: expected shape {shape}, got {values.shape}")
    return values


def _load_dataset_value(value, filename: str, sample_id, row: int,
                        kind: str, name: str) -> float:
    """Check representability after the loader's existing float conversion."""
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{filename}: sample {sample_id!r} (row {row}), {kind} {name!r}: "
            "value must be finite after float conversion") from exc
    if not np.isfinite(converted):
        raise ValueError(
            f"{filename}: sample {sample_id!r} (row {row}), {kind} {name!r}: "
            "value must be finite after float conversion")
    return converted

_DATASET_FILES = ("features.npy", "labels.npy", "metadata.json")


class _DatasetSaveRecoveryError(RuntimeError):
    """A publication exception was followed by an incomplete rollback."""

    def __init__(self, destination, recovery_directory, original, rollback_errors):
        self.destination = str(destination)
        self.recovery_directory = str(recovery_directory)
        self.original_error = original
        self.rollback_errors = rollback_errors
        super().__init__(
            f"Dataset save failed and rollback was incomplete. Do not load "
            f"{destination} until recovered. Original files and recovery.json "
            f"are retained at {recovery_directory}. Original error: {original!r}; "
            f"rollback errors: {rollback_errors!r}")


def _dataset_file_mode(path):
    """Inspect only; reject unsafe managed targets before staging/publication."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Dataset destination must be a regular file, not a "
                         f"symlink or nonregular target: {path}")
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o7000:
        raise ValueError(f"Dataset destination has unsupported special mode bits: {path}")
    return mode


def _dataset_private_file(path):
    """Create an owned private workspace file without touching existing paths."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)


def _dataset_file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _warn_dataset_cleanup(path, error):
    message = (f"Dataset persistence cleanup failed; private temporary data remain "
               f"at {path}: {error!r}. The save/rollback outcome is unchanged.")
    try:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    except Exception:
        # Treating warnings as errors must not turn a committed save into failure.
        try:
            logging.getLogger(__name__).warning(message)
        except Exception:
            pass  # A custom logging handler must not replace the save outcome.


def _save_dataset_files(out, X, Y, metadata):
    """Publish this class's three owned files; no general transaction API."""
    try:
        info = out.lstat()
    except FileNotFoundError:
        existed = False
    else:
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Dataset destination must be a real directory, not "
                             f"a symlink or non-directory: {out}")
        existed = True
    modes = {name: _dataset_file_mode(out / name) for name in _DATASET_FILES}
    out.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=f".{out.name}.save-", dir=out.parent))
    new, original = workspace / "new", workspace / "original"
    published = []
    created = False
    retain = False
    try:
        new.mkdir()
        original.mkdir()
        for name in _DATASET_FILES:
            _dataset_private_file(new / name)
        np.save(new / "features.npy", X)
        np.save(new / "labels.npy", Y)
        with open(new / "metadata.json", "w") as fh:
            fh.write(metadata)
        for name in _DATASET_FILES:
            # Only generated private files receive chmod, never original targets.
            os.chmod(new / name, 0o600 if modes[name] is None else modes[name])
            if modes[name] is not None:
                _dataset_private_file(original / name)
                shutil.copyfile(out / name, original / name)
        recovery = {
            "destination": str(out.resolve()),
            "destination_existed": existed,
            "managed_files": list(_DATASET_FILES),
            "original_present": {name: modes[name] is not None for name in _DATASET_FILES},
            "original_modes": modes,
            "original_sha256": {name: _dataset_file_hash(original / name)
                                for name in _DATASET_FILES if modes[name] is not None},
        }
        with open(workspace / "recovery.json", "w") as fh:
            json.dump(recovery, fh, indent=2)
        # Keep originals across interruptions once publication can begin. This
        # also covers interruption between os.replace and recording its result,
        # or an interruption during rollback. It is not crash recovery.
        retain = True
        if not existed:
            out.mkdir()
            created = True
        for name in _DATASET_FILES:
            os.replace(new / name, out / name)
            published.append(name)
        retain = False
    except Exception as error:
        rollback_errors = []
        for name in reversed(published):
            try:
                if modes[name] is None:
                    (out / name).unlink()
                else:
                    restore = workspace / ("restore-" + name)
                    _dataset_private_file(restore)
                    shutil.copyfile(original / name, restore)
                    os.chmod(restore, modes[name])
                    os.replace(restore, out / name)
            except Exception as rollback_error:
                rollback_errors.append((name, repr(rollback_error)))
        if created and not rollback_errors:
            try:
                out.rmdir()  # Never recursively delete the destination.
            except Exception as rollback_error:
                rollback_errors.append(("directory", repr(rollback_error)))
        if rollback_errors:
            retain = True
            raise _DatasetSaveRecoveryError(out, workspace, error,
                                            rollback_errors) from error
        retain = False
        raise
    finally:
        if not retain:
            try:
                shutil.rmtree(workspace)  # Only the workspace this call created.
            except Exception as cleanup_error:
                _warn_dataset_cleanup(workspace, cleanup_error)


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


def _save_checkpoint(directory: str, samples: List[DatasetSample], run_binding: dict) -> None:
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
    from . import _generate_contract as contract
    contract.validate_rows(ckpt['samples'], run_binding['ordered_plan'])
    ckpt.update(checkpoint_schema=contract.SCHEMA, run_binding=run_binding,
                run_binding_sha256=contract.digest(run_binding),
                samples_digest=contract.digest(ckpt['samples']))
    payload = json.dumps(ckpt, indent=2, allow_nan=False)
    final_path = out / "checkpoint.json"
    mode = _dataset_file_mode(final_path)
    fd, name = tempfile.mkstemp(prefix='.checkpoint-', suffix='.tmp', dir=out)
    os.close(fd)
    tmp_path = Path(name)
    try:
        tmp_path.write_text(payload)
        os.chmod(tmp_path, 0o600 if mode is None else mode)
        os.replace(tmp_path, final_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            # Cleanup cannot mask the primary failure or reverse a committed save.
            try:
                logging.getLogger(__name__).warning('checkpoint cleanup failed: %s', exc)
            except Exception:
                pass

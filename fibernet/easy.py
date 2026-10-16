"""
FiberNet Easy API — 一行代码完成常见任务

让高中生也能轻松使用 FiberNet：
- show(graph) → 可视化并保存
- simulate(graph, mode) → FEM 模拟
- batch_simulate(configs) → 批量模拟输出 CSV
- train_model(csv_path) → ML 训练
- train_rl(env_config) → RL 训练

Examples
--------
>>> from fibernet import pattern_2d, show, simulate, batch_simulate
>>> g = pattern_2d("honeycomb", box=(10,10), grid=(4,4))
>>> show(g)  # 一行出图
>>> result = simulate(g, mode="stretch", strain=2.0)  # 一行模拟
>>> batch_simulate([{"unit": "honeycomb"}, {"unit": "kagome"}], "results.csv")  # 一行批量
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from fibernet.core.structure_graph import StructureGraph
from fibernet.gen.pattern import pattern_2d, pattern_3d
from fibernet.sim.accelerated import TaichiEngine, SimResult
from fibernet.viz.render import render_graph, render_deformation


def show(
    graph: StructureGraph,
    *,
    theme: str = "dark",
    save_path: Optional[str] = None,
    title: str = "",
    **kwargs,
) -> None:
    """一行出图：可视化 StructureGraph。

    Parameters
    ----------
    graph : StructureGraph
        要可视化的结构。
    theme : str
        主题: "dark" (紫黑), "light" (白底), "blueprint", "publication"。
    save_path : str, optional
        保存路径，默认不保存。
    title : str
        标题。

    Examples
    --------
    >>> from fibernet import pattern_2d, show
    >>> g = pattern_2d("honeycomb", box=(10,10), grid=(4,4))
    >>> show(g)
    >>> show(g, theme="light", save_path="honeycomb.png")
    """
    fig = render_graph(graph, theme=theme, title=title, **kwargs)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")
    return fig


def simulate(
    graph: StructureGraph,
    mode: str = "stretch",
    strain: float = 2.0,
    backend: str = "spring",
    save_path: Optional[str] = None,
    **kwargs,
) -> SimResult:
    """Run a stretch simulation with the spring or beam-frame FEM backend.

    Parameters
    ----------
    graph : StructureGraph
        要模拟的结构。
    mode : str
        模拟模式: "stretch" (位移控制拉伸) 或 "dynamics" (自定义动力学)
    strain : float
        Target stretch ratio for both backends (1.01 = 1% extension).
    backend : str
        "spring" or "fem". The FEM backend accepts E, nu, pct, and nonlinear.
        Its dim defaults to graph.dimension; an explicit dim=2 or dim=3
        overrides this selection.
    save_path : str, optional
        JSON 保存路径。

    Returns
    -------
    SimResult
        模拟结果。

    Examples
    --------
    >>> from fibernet import pattern_2d, simulate
    >>> g = pattern_2d("honeycomb", box=(10,10), grid=(4,4))
    >>> r = simulate(g, mode="stretch", strain=2.0)  # 质点弹簧拉伸2倍
    >>> r = simulate(g, mode="stretch", strain=2.0, backend="spring")  # 质点弹簧拉 2 倍
    """
    if backend == "spring":
        engine = TaichiEngine()
        if mode == "stretch":
            result = engine.stretch_test(graph, target_stretch=strain, **kwargs)
        elif mode == "dynamics":
            result = engine.dynamics(graph, **kwargs)
        else:
            raise ValueError(f"Unknown spring mode '{mode}'. Use: stretch, dynamics")
    elif backend == "fem":
        if mode != "stretch":
            raise ValueError("The FEM backend supports only mode='stretch'")
        from fibernet.ml.beam_frame_fem import BeamFrameFEM
        dim = kwargs.setdefault("dim", graph.dimension)
        if not isinstance(dim, (int, np.integer)) or dim not in (2, 3):
            raise ValueError("The FEM dimension must be 2 or 3")
        E = kwargs.pop("E", 1e9)
        nu = kwargs.pop("nu", 0.3)
        fem = BeamFrameFEM(E=E, nu=nu)
        raw = fem.stretch_test(graph, target_stretch=strain, **kwargs)
        result = fem.to_sim_result(raw, graph)
    else:
        raise ValueError("backend must be 'spring' or 'fem'")


    if save_path:
        result.save(save_path)
        print(f"Saved to {save_path}")

    return result


def batch_simulate(
    configs: List[Dict[str, Any]],
    output: str = "results.csv",
    mode: str = "stretch",
    strain: float = 2.0,
) -> str:
    """Run independent FEM stretch cases and write their measured outputs to CSV.

    Parameters
    ----------
    configs : list of dict
        每个 dict 包含: unit, box, grid, seed 等参数。
    output : str
        CSV 输出路径。
    mode : str
        模拟模式。
    strain : float
        施加应变。

    Returns
    -------
    str
        CSV 文件路径。

    Examples
    --------
    >>> from fibernet import batch_simulate
    >>> configs = [
    ...     {"unit": "honeycomb", "grid": (4,4)},
    ...     {"unit": "kagome", "grid": (4,4)},
    ...     {"unit": "reentrant", "grid": (3,3)},
    ... ]
    >>> batch_simulate(configs, "results.csv")
    """
    import csv

    rows = []
    for i, cfg in enumerate(configs):
        unit = cfg.get("unit", "honeycomb")
        box = cfg.get("box", (10, 10))
        grid = cfg.get("grid", (4, 4))
        seed = cfg.get("seed", 42)

        try:
            g = pattern_2d(unit=unit, box=box, grid=grid, seed=seed, **cfg.get("kwargs", {}))
            result = simulate(g, mode=mode, strain=strain, backend="fem")

            row = {
                "id": i,
                "unit": unit,
                "grid_x": grid[0] if isinstance(grid, (tuple, list)) else grid,
                "grid_y": grid[1] if isinstance(grid, (tuple, list)) else grid,
                "n_nodes": g.num_nodes,
                "n_edges": g.num_edges,
                "max_force": result.max_force,
                "max_displacement": result.max_displacement,
                "energy": result.energy,
                "mode": mode,
                "strain": strain,
            }
            rows.append(row)
            print(f"{i+1}/{len(configs)}: {unit} (max_force={row['max_force']:.2e})")

        except Exception as e:
            print(f"{i+1}/{len(configs)}: {unit} failed: {e}")

    if rows:
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} results to {output}")
    else:
        raise RuntimeError('All batch simulations failed; no CSV was written')

    return output


def train_model(
    csv_path,
    target: str = "max_force",
    features: Optional[List[str]] = None,
    model_type: str = "rf",
    test_size: float = 0.2,
    save_path: Optional[str] = None,
    *,
    split=None,
    groups=None,
    metadata_keys=None,
) -> Dict[str, Any]:
    """Train a sklearn regressor from a CSV or an explicit FiberNetDataset.

    Native datasets use raw names and require explicit ``split='row'`` or
    ``split='group', groups=[...]``. Optional ``metadata_keys`` snapshots selected
    per-row JSON-native assertions; it does not verify units or provenance.
    Native results add target and row/split provenance in ``training_data``.
    CSV defaults, column rules, metrics and saved envelope remain unchanged.

    Parameters
    ----------
    csv_path : str, path-like, or FiberNetDataset
        CSV path or explicit native Dataset. The historical keyword is retained.
    target : str
        Exact CSV column or raw native target name; no target is inferred.
    features : list of str, optional
        Nonempty unique selected names. CSV selections reject the target and
        every label_ column; only complete Dataset CSVs default to feat_ columns.
        Native selections default to stored feature_names in order and reject
        every declared target name, without applying CSV prefix rules.
    model_type : str
        模型: "rf" (随机森林), "lr" (线性回归), "gb" (梯度提升)。
    test_size : float
        Held-out row fraction for CSV/native row mode, or group fraction for
        native group mode. Group sizes determine the realized row fraction.
    save_path : str, optional
        Model artifact path. A complete pickle is staged before replacement;
        ordinary save failures preserve previous bytes or destination absence.
        Existing targets must be regular files without special permission bits.
        Parent directories must exist. No crash/concurrent-writer guarantee.
    split : {"row", "group"}, keyword-only
        Required explicitly for native data; omitted for CSV compatibility.
    groups : list or tuple of str, keyword-only
        Caller-supplied origin labels aligned to native rows, required only for
        group mode. Disjointness is checked; scientific origin is not verified.
    metadata_keys : list of str, optional, keyword-only
        Existing per-row native metadata keys to snapshot as finite JSON-native
        assertions. None/[] retains no metadata. Opaque unselected data is ignored.

    Returns
    -------
    dict
        包含 R2, RMSE 等指标。

    Examples
    --------
    >>> from fibernet import train_model
    >>> metrics = train_model("results.csv", target="max_force",
    ...                       features=["grid_x", "grid_y", "strain"])
    >>> print(f"R² = {metrics['r2']:.3f}")
    """
    from fibernet.ml.dataset import FiberNetDataset

    if isinstance(csv_path, FiberNetDataset):
        return _train_native_dataset(
            csv_path, target=target, features=features, model_type=model_type,
            test_size=test_size, save_path=save_path, split=split, groups=groups,
            metadata_keys=metadata_keys,
        )
    if split is not None or groups is not None or metadata_keys is not None:
        raise ValueError("split, groups and metadata_keys are native FiberNetDataset options")
    import os
    if not isinstance(csv_path, (str, bytes, os.PathLike)):
        raise ValueError("csv_path must be a CSV path or an explicit FiberNetDataset")

    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
    except ImportError:
        raise ImportError("需要安装 pandas 和 scikit-learn: pip install pandas scikit-learn")

    import csv
    from itertools import dropwhile

    # Validate the raw header before pandas can silently rename duplicates.
    # Match read_csv's default comma/quote and initial blank-line behavior.
    # Retain each record's raw text to distinguish blank physical lines from
    # quoted empty/whitespace fields; quoted multiline values stay one record.
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        raw_lines = []

        def tracked_lines():
            for line in dropwhile(lambda line: not line.strip(), handle):
                raw_lines.append(line)
                yield line

        try:
            reader = csv.reader(tracked_lines(), strict=True)
            header = next(reader, [])
            raw_lines.clear()
            for record_number, row in enumerate(reader, start=2):
                blank = not "".join(raw_lines).strip()
                raw_lines.clear()
                if blank:
                    continue
                if len(row) != len(header):
                    raise ValueError(
                        f"CSV record {record_number} has {len(row)} fields; "
                        f"header has {len(header)}. Fix the CSV row width before training."
                    )
        except csv.Error as exc:
            raise ValueError(f"Malformed CSV record near physical line {reader.line_num}: {exc}") from exc
    if not header or any(not name.strip() for name in header):
        raise ValueError("CSV header must contain nonempty column names")
    seen = set()
    duplicates = []
    for name in header:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"CSV header contains duplicate column names: {duplicates}")

    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a nonempty column name")
    if target not in header:
        labels = [name for name in header if name.startswith("label_")]
        raise ValueError(
            f"Missing target column {target!r}; pass target='existing_column' explicitly. "
            f"Available label_ columns: {labels}. No target is inferred."
        )

    if features is None:
        structural = {"sample_id", "unit", "grid_x", "grid_y"}
        feature_columns = [name for name in header
                           if name.startswith("feat_") and name[5:].strip()]
        label_columns = [name for name in header
                         if name.startswith("label_") and name[6:].strip()]
        recognized = (
            structural.issubset(header) and feature_columns and label_columns
            and set(header) == structural | set(feature_columns) | set(label_columns)
            and target in label_columns
        )
        if not recognized:
            raise ValueError(
                "Automatic feature selection requires a Dataset CSV with sample_id, "
                "unit, grid_x, grid_y, feat_ and label_ columns, and a label_ target. "
                "For other CSVs, pass features=['input_column', ...] explicitly."
            )
        features = feature_columns
    if not isinstance(features, list):
        raise ValueError("features must be a nonempty list of column names")
    if not features or any(not isinstance(name, str) or not name.strip() for name in features):
        raise ValueError("features must contain nonempty string column names")
    if len(set(features)) != len(features):
        raise ValueError("features contains duplicate column names")
    forbidden = [name for name in features if name == target or name.startswith("label_")]
    if forbidden:
        raise ValueError(f"features must not contain the target or any reserved label_ column: {forbidden}")
    missing = [name for name in features if name not in header]
    if missing:
        raise ValueError(f"features contains missing columns: {missing}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    invalid = []
    for name in features + [target]:
        column = df[name]
        if (not pd.api.types.is_numeric_dtype(column.dtype)
                or pd.api.types.is_complex_dtype(column.dtype)
                or not np.isfinite(column.to_numpy()).all()):
            invalid.append(name)
    if invalid:
        raise ValueError(
            f"Selected columns must contain only numeric, real-valued, finite values "
            f"(no missing values): {invalid}"
        )

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "lr":
        model = LinearRegression()
    elif model_type == "gb":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use: rf, lr, gb")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metrics = {"r2": r2, "rmse": rmse, "n_samples": len(df), "features": features}
    print(f"✓ R²={r2:.3f}, RMSE={rmse:.2e}, samples={len(df)}")

    if save_path:
        _save_model_artifact(save_path, {"model": model, "features": features,
                                      "metrics": metrics, "artifact_version": 1,
                                      "target": target})
        print(f"✓ Model saved to {save_path}")

    return metrics


def _train_native_dataset(dataset, *, target, features, model_type, test_size,
                          save_path, split, groups, metadata_keys):
    """Native convenience boundary; source assertions are not physical checks."""
    from collections.abc import Mapping
    from numbers import Real
    from fibernet.ml.dataset import FiberNetDataset

    def names(value, context, *, allow_empty=False):
        if (not isinstance(value, list) or (not value and not allow_empty)
                or any(not isinstance(name, str) or not name.strip() for name in value)
                or len(set(value)) != len(value)):
            raise ValueError(f"{context} must be a {'possibly empty' if allow_empty else 'nonempty'} "
                             "list of unique nonempty string names")
        return list(value)

    declared_features = names(dataset.feature_names, "Dataset feature_names")
    declared_targets = names(dataset.target_names, "Dataset target_names")
    if not isinstance(target, str) or not target.strip() or target not in declared_targets:
        raise ValueError(f"Missing or invalid native target {target!r}; available raw targets: "
                         f"{declared_targets}. Pass target explicitly; no target is inferred.")
    selected = declared_features if features is None else names(features, "features")
    missing = [name for name in selected if name not in declared_features]
    if missing:
        raise ValueError(f"features contains missing native names: {missing}; use raw Dataset names")
    forbidden = [name for name in selected if name in declared_targets]
    if forbidden:
        raise ValueError(f"features must not contain any declared target name: {forbidden}")
    if not isinstance(model_type, str) or model_type not in ("rf", "lr", "gb"):
        raise ValueError(f"Unknown model_type {model_type!r}. Use: rf, lr, gb")
    if not isinstance(split, str) or split not in ("row", "group"):
        raise ValueError("Native training requires explicit split='row' or split='group', groups=[...]")
    if isinstance(test_size, (bool, np.bool_)) or not isinstance(test_size, Real):
        raise ValueError("Native test_size must be a finite fraction strictly between 0 and 1")
    try:
        test_size = float(test_size)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("Native test_size must be a finite fraction strictly between 0 and 1") from exc
    if not np.isfinite(test_size) or not 0 < test_size < 1:
        raise ValueError("Native test_size must be a finite fraction strictly between 0 and 1")
    if split == "row" and groups is not None:
        raise ValueError("groups must be omitted for split='row'")
    if split == "group":
        if (not isinstance(groups, (list, tuple)) or len(groups) != dataset.n_samples
                or any(not isinstance(group, str) or not group.strip() for group in groups)):
            raise ValueError("groups must be a list/tuple of nonempty string labels, one per Dataset row")
        groups = list(groups)
        if len(set(groups)) < 2:
            raise ValueError("Group splitting requires at least two distinct caller-supplied groups")

    keys = [] if metadata_keys is None else names(metadata_keys, "metadata_keys", allow_empty=True)

    def snapshot_json(value, context, active):
        # Exact JSON-native types avoid lossy coercion (e.g. tuple to list or
        # non-string dictionary keys) and do not serialize arbitrary objects.
        if value is None or type(value) in (bool, str, int):
            return value
        if type(value) is float:
            if np.isfinite(value):
                return value
            raise ValueError(f"{context}: selected metadata must contain finite floats")
        if type(value) not in (list, dict):
            raise ValueError(f"{context}: selected metadata must contain only JSON-native values")
        identity = id(value)
        if identity in active:
            raise ValueError(f"{context}: selected metadata must not contain cycles")
        active.add(identity)
        try:
            if type(value) is list:
                return [snapshot_json(item, f"{context}[{index}]", active)
                        for index, item in enumerate(value)]
            if any(type(key) is not str for key in value):
                raise ValueError(f"{context}: selected metadata dictionary keys must be strings")
            return {key: snapshot_json(item, f"{context}[{key!r}]", active)
                    for key, item in value.items()}
        finally:
            active.remove(identity)

    rows = []
    for index, sample in enumerate(dataset.samples):
        sample_id = getattr(sample, "sample_id", None)
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"Dataset row {index}: sample_id must be a nonempty string")
        row = {"row_index": index, "sample_id": sample_id}
        if keys:
            context = f"sample {sample_id!r} (row {index})"
            metadata = getattr(sample, "metadata", None)
            if not isinstance(metadata, Mapping):
                raise ValueError(f"{context}: selected metadata must be a mapping")
            missing_keys = [key for key in keys if key not in metadata]
            if missing_keys:
                raise ValueError(f"{context}: missing selected metadata keys: {missing_keys}")
            try:
                row["metadata"] = {key: snapshot_json(metadata[key], f"{context}, metadata {key!r}", set())
                                   for key in keys}
            except RecursionError as exc:
                raise ValueError(f"{context}: selected metadata nesting is too deep") from exc
        rows.append(row)

    # A temporary view validates only the requested target, without mutating
    # the caller or accidentally validating unrelated unavailable labels.
    view = FiberNetDataset(dataset.samples, feature_names=selected, target_names=[target])
    X, y = view.features, view.labels(target)
    if dataset.n_samples < 3:
        raise ValueError("Native training requires at least one train row and two test rows for R2")

    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split, GroupShuffleSplit
        from sklearn.metrics import r2_score, mean_squared_error
    except ImportError as exc:
        raise ImportError("Native training requires scikit-learn: pip install scikit-learn") from exc

    indices = np.arange(dataset.n_samples)
    try:
        if split == "row":
            train, test = train_test_split(indices, test_size=test_size, random_state=42)
        else:
            train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
                               .split(indices, groups=groups))
    except ValueError as exc:
        raise ValueError(f"Native {split} split cannot create a usable holdout: {exc}") from exc
    if len(train) < 1 or len(test) < 2:
        raise ValueError("Native split requires at least one train row and two test rows for R2")
    split_record = {"method": split, "random_state": 42,
                    "train_rows": train.tolist(), "test_rows": test.tolist()}
    if split == "group":
        train_groups = {groups[index] for index in train}
        test_groups = {groups[index] for index in test}
        if train_groups & test_groups:
            raise ValueError("Native group split must not overlap caller-supplied groups")
        split_record["groups"] = groups
    training_data = {"source_kind": "FiberNetDataset", "rows": rows,
                     "metadata_keys": keys, "split": split_record}
    # Check representability before fitting/saving; this is not schema or unit validation.
    json.dumps(training_data, allow_nan=False)

    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "lr":
        model = LinearRegression()
    else:
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X[train], y[train])
    predictions = model.predict(X[test])
    r2 = float(r2_score(y[test], predictions))
    rmse = float(np.sqrt(mean_squared_error(y[test], predictions)))
    metrics = {"r2": r2, "rmse": rmse, "n_samples": dataset.n_samples,
               "features": selected, "target": target, "training_data": training_data}
    print(f"✓ R²={r2:.3f}, RMSE={rmse:.2e}, samples={dataset.n_samples}")
    if save_path:
        _save_model_artifact(save_path, {"model": model, "features": selected,
                                      "metrics": metrics, "artifact_version": 1,
                                      "target": target})
        print(f"✓ Model saved to {save_path}")
    return metrics


def _save_model_artifact(path, artifact):
    """Publish one fully closed pickle; preserve old bytes on ordinary failure.

    Only ordinary rwx bits survive replacement, not inode/owner/ACL metadata.
    No parent creation, crash durability, or concurrent-writer guarantees.
    """
    import logging
    import os
    import pickle
    import stat
    import tempfile
    import warnings

    out = Path(os.fsdecode(path))
    try:
        info = out.lstat()
    except FileNotFoundError:
        mode = 0o600
    else:
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Model destination must be a regular file, not a "
                             f"symlink or nonregular target: {out}")
        mode = stat.S_IMODE(info.st_mode)
        if mode & 0o7000:
            raise ValueError(f"Model destination has unsupported special mode bits: {out}")

    fd, temporary = tempfile.mkstemp(prefix=f".{out.name}.save-", dir=out.parent)
    try:
        handle = os.fdopen(fd, "wb")
        fd = None  # The handle now owns this descriptor, including on close failure.
        with handle:
            pickle.dump(artifact, handle)
        # Never chmod the original target or unrelated files.
        os.chmod(temporary, mode)
        os.replace(temporary, out)
    finally:
        cleanup_errors = []
        if fd is not None:  # fdopen itself failed before assuming ownership.
            try:
                os.close(fd)
            except Exception as exc:
                cleanup_errors.append(exc)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass  # Successful replacement already consumed the stage.
        except Exception as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            message = (f"Model artifact cleanup failed; private temporary data may "
                       f"remain at {temporary}: {cleanup_errors!r}. "
                       "The save outcome is unchanged.")
            try:
                warnings.warn(message, RuntimeWarning, stacklevel=3)
            except Exception:
                # Warnings-as-errors or a broken handler cannot mask the outcome.
                try:
                    logging.getLogger(__name__).warning(message)
                except Exception:
                    pass


def predict_named(artifact, data, *, extra_columns="error"):
    """Predict from named rows using an already-loaded, trusted artifact dict.

    Accept a DataFrame, a mapping, or a nonempty list/tuple of mappings. Stored
    feature order is authoritative; extra columns reject unless explicitly
    ignored. Selected values must be finite real numbers (including booleans).
    No deserialization, imputation, text coercion, or unit/schema checks occur.
    Return JSON-ready predictions, target, features and named-column validation.
    Legacy unversioned artifacts may omit target; envelope version 1 requires it.
    """
    from collections.abc import Mapping
    from numbers import Real

    if not isinstance(artifact, dict):
        raise ValueError("artifact must be an already-loaded trusted dictionary")
    if not isinstance(extra_columns, str) or extra_columns not in ("error", "ignore"):
        raise ValueError("extra_columns must be 'error' or 'ignore'")

    def names_valid(names):
        return (all(isinstance(name, str) and name.strip() for name in names)
                and len(set(names)) == len(names))

    features = artifact.get("features")
    if not isinstance(features, list) or not features or not names_valid(features):
        raise ValueError("artifact features must be a nonempty unique list of nonempty string names")
    features = list(features)
    if "artifact_version" in artifact:
        version = artifact["artifact_version"]
        if type(version) is not int or version != 1:
            raise ValueError("Unsupported artifact_version; expected integer 1")
        if "target" not in artifact or artifact["target"] is None:
            raise ValueError("artifact version 1 requires a target name")
    target = artifact.get("target")
    if target is not None and (not isinstance(target, str) or not target.strip()
                               or target in features):
        raise ValueError("artifact target must be a nonempty name distinct from features")
    model = artifact.get("model")
    if not callable(getattr(model, "predict", None)):
        raise ValueError("artifact model must have callable predict")

    if isinstance(data, Mapping):
        records = [data]
    elif isinstance(data, (list, tuple)):
        records = data
    else:
        # pandas is optional for named records and is never imported at module load.
        try:
            import pandas as pd
        except ImportError:
            pd = None
        if pd is None or not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a DataFrame or named mapping records; arrays are unsupported")
        columns = list(data.columns)
        if not names_valid(columns):
            raise ValueError("input column names must be unique nonempty strings")
        records = data.to_dict(orient="records")
    if not records:
        raise ValueError("data must contain at least one row")

    def finite_real(value):
        if np.ma.is_masked(value) or not isinstance(value, (Real, np.bool_)):
            raise ValueError("Selected values and predictions must be finite real numbers")
        try:
            converted = float(value)
        except (OverflowError, ValueError, TypeError) as exc:
            raise ValueError("Numeric values must be representable as finite real numbers") from exc
        if not np.isfinite(converted):
            raise ValueError("Selected values and predictions must be finite real numbers")
        return converted

    rows = []
    selected = set(features)
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Every input row must be a named mapping")
        names = list(record.keys())
        if not names_valid(names):
            raise ValueError("input column names must be unique nonempty strings")
        missing = selected - set(names)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        extras = set(names) - selected
        if extras and extra_columns == "error":
            raise ValueError(f"Unexpected extra columns: {sorted(extras)}")
        rows.append([finite_real(record[name]) for name in features])
    matrix = np.asarray(rows, dtype=float)
    raw_output = model.predict(matrix)
    # np.asarray discards MaskedArray masks, which could expose missing values
    # as apparently valid predictions. Reject active masks before conversion.
    if np.ma.is_masked(raw_output):
        raise ValueError("model.predict must not return masked predictions")
    output = np.asarray(raw_output)
    if output.shape != (len(rows),):
        raise ValueError("model.predict must return a single-target vector of shape (n_rows,)")
    predictions = [finite_real(value) for value in output]
    return {"predictions": predictions, "target": target, "features": features,
            "validation": "named_columns_only"}


def train_rl(
    env_config: Optional[Dict] = None,
    n_episodes: int = 100,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """一行 RL 训练：训练 RL agent 优化结构。

    Parameters
    ----------
    env_config : dict, optional
        环境配置，默认使用标准配置。
    n_episodes : int
        训练回合数。
    save_path : str, optional
        保存路径。

    Returns
    -------
    dict
        训练统计。

    Examples
    --------
    >>> from fibernet import train_rl
    >>> stats = train_rl(n_episodes=100)
    >>> print(f"Final reward: {stats['final_reward']:.2f}")
    """
    from fibernet.sim.rl_env import FiberNetworkEnv

    env = FiberNetworkEnv(**(env_config or {}))

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        if (ep + 1) % 10 == 0:
            avg = np.mean(rewards[-10:])
            print(f"  Episode {ep+1}/{n_episodes}: avg_reward={avg:.2f}")

    env.close()

    stats = {
        "n_episodes": n_episodes,
        "final_reward": rewards[-1],
        "mean_reward": np.mean(rewards),
        "max_reward": np.max(rewards),
    }

    if save_path:
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Stats saved to {save_path}")

    print(f"✓ Mean reward: {stats['mean_reward']:.2f}, Max: {stats['max_reward']:.2f}")
    return stats

def batch_simulate_from_json(
    json_dir: str,
    output_dir: str = "simulation_results",
    mode: str = "stretch",
    target_stretch: float = 1.5,
    stiffness: float = 1e5,
    damping: float = 0.3,
    num_steps: int = 1000,
    save_interval: int = 200,
    backend: str = "spring",
) -> str:
    """Batch simulate structures from JSON files.
    
    Reads all .json files in json_dir, simulates each, and saves results.
    
    Parameters
    ----------
    json_dir : str
        Directory containing StructureGraph JSON files.
    output_dir : str
        Directory to save results (CSV + individual JSON results).
    mode : str
        Simulation mode: "stretch".
    target_stretch : float
        Target stretch ratio for stretch mode.
    stiffness : float
        Spring stiffness for spring backend.
    damping : float
        Damping ratio.
    num_steps : int
        Number of simulation steps.
    save_interval : int
        Save trajectory every N steps.
    backend : str
        Simulation backend: "spring" (TaichiEngine).
    
    Returns
    -------
    str
        Path to the output CSV file.
    
    Examples
    --------
    >>> from fibernet import batch_simulate_from_json
    >>> csv_path = batch_simulate_from_json(
    ...     "my_structures/",
    ...     output_dir="results/",
    ...     mode="stretch",
    ...     target_stretch=1.5,
    ... )
    >>> print(f"Results saved to {csv_path}")
    
    Notes
    -----
    - Automatically parses JSON files in various formats (StructureGraph, networkx, custom)
    - Saves trajectory data for visualization
    - Supports checkpoint resume (skips already simulated files)
    - Output CSV uses a union schema: filename, n_nodes, n_edges, max_force,
      max_stretch, mean_stretch, std_stretch, time_seconds, success, error,
      plus any existing columns. Missing values are blank.
    - Resume retains all existing rows, including failures, and skips their files.
      Existing CSV success values must be "True" or "False".
    - Each CSV checkpoint is replaced atomically after serialization succeeds.
      Malformed existing CSV files raise ValueError and are left unchanged.
    """
    import csv
    import os
    import tempfile
    from pathlib import Path

    if backend != "spring":
        raise ValueError(f"Unsupported backend: {backend!r}; expected 'spring'")
    if mode != "stretch":
        raise ValueError(f"Unsupported mode: {mode!r}; expected 'stretch'")

    json_path = Path(json_dir)
    output_path = Path(output_dir)
    csv_file = output_path / "simulation_results.csv"
    fieldnames = [
        "filename", "n_nodes", "n_edges", "max_force", "max_stretch",
        "mean_stretch", "std_stretch", "time_seconds", "success", "error",
    ]
    results = []
    existing = set()
    if csv_file.exists():
        try:
            with csv_file.open(newline="", encoding="utf-8") as stream:
                reader = csv.reader(stream, strict=True)
                header = next(reader, [])
                if (not header or any(not name.strip() for name in header)
                        or len(set(header)) != len(header)
                        or not {"filename", "success"}.issubset(header)):
                    raise ValueError("Invalid existing CSV header")
                fieldnames.extend(name for name in header if name not in fieldnames)
                for values in reader:
                    if len(values) != len(header):
                        raise ValueError("Invalid existing CSV row width")
                    row = dict(zip(header, values))
                    fname = row["filename"]
                    if not fname.strip() or fname in existing:
                        raise ValueError("Invalid or duplicate filename in existing CSV")
                    if row["success"] not in ("True", "False"):
                        raise ValueError("Invalid success value in existing CSV")
                    existing.add(fname)
                    results.append(row)
        except (csv.Error, UnicodeError) as exc:
            raise ValueError(f"Invalid existing CSV: {exc}") from exc

    json_files = sorted(json_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return ""

    print(f"Found {len(json_files)} JSON files in {json_dir}")
    pending = []
    for json_file in json_files:
        if json_file.name in existing:
            print(f"  [{json_file.name}] Already simulated, skipping")
        else:
            pending.append(json_file)
    if not pending:
        return str(csv_file)

    engine = TaichiEngine()
    output_path.mkdir(parents=True, exist_ok=True)

    def write_checkpoint():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", newline="", encoding="utf-8", dir=output_path,
                prefix=".simulation_results.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, csv_file)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    for json_file in pending:
        fname = json_file.name
        try:
            g = StructureGraph.load_json(str(json_file))
            r = engine.stretch_test(
                g,
                target_stretch=target_stretch,
                stiffness=stiffness,
                damping=damping,
                num_steps=num_steps,
                save_interval=save_interval,
                auto_steps=False,
            )
            row = {
                "filename": fname,
                "n_nodes": g.num_nodes,
                "n_edges": g.num_edges,
                "max_force": float(r.max_force),
                "max_stretch": float(r.max_stretch),
                "mean_stretch": float(r.mean_stretch),
                "std_stretch": float(r.std_stretch),
                "time_seconds": float(r.time_seconds),
                "success": True,
            }
            r.save(str(output_path / f"{json_file.stem}_result.json"), detailed=True)
            print(f"  [{fname}] max_force={r.max_force:.0f}, stretch={r.max_stretch:.3f}")
        except Exception as e:
            row = {"filename": fname, "success": False, "error": str(e)}
            print(f"  [{fname}] FAILED: {e}")
        results.append(row)
        # Persistence failures propagate; they are not simulation failures.
        write_checkpoint()

    ok = sum(row["success"] in (True, "True") for row in results)
    print(f"\n✓ Simulated {ok}/{len(results)} successfully")
    print(f"  Results: {csv_file}")
    return str(csv_file)

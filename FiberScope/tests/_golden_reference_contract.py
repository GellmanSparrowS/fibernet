"""Pinned historical golden comparison; this module imports no product code.

This benchmark deliberately requires the exact historical reference and audited
current source even when version labels happen to match. A future approved source
or golden update requires deliberate review/update of the test contract; there is
no generic same-version bypass. Admission only permits numerical comparison and
never declares numerical equivalence or cache/checkpoint interchangeability.
"""
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import time


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _normalized(data):
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def load_contract():
    return json.loads(Path(__file__).with_name("golden_reference_contract.json").read_text())


def validate_files(root, contract):
    """Read-only, fail closed before importing or executing the numerical driver."""
    root = Path(root).resolve()
    reference = root / "FiberScope/tests/golden_engine.npz"
    _require(reference.is_file(), "golden reference missing; comparison cannot skip")
    raw = reference.read_bytes()
    _require(_sha(raw) == contract["reference_sha256"], "golden reference bytes changed")
    _require(len(contract["source_sha256"]) == 14, "expected 14 source inputs")
    for relative, digest in contract["source_sha256"].items():
        path = root / relative
        _require(path.is_file() and _sha(path.read_bytes()) == digest,
                 "current source mismatch: " + relative)
    driver = root / "FiberScope/scripts/make_golden_engine.py"
    _require(driver.is_file() and _sha(_normalized(driver.read_bytes())) ==
             contract["driver_normalized_sha256"], "golden driver changed")
    return raw


def validate_identity(contract, metadata, version, source_hash, actual_hash, cases):
    _require(metadata == contract["reference_metadata"], "reference metadata changed")
    _require(metadata["engine_version"] == contract["reference_version"],
             "unknown reference version")
    _require(version == contract["current_version"], "unknown current version")
    _require(source_hash == contract["current_engine_hash"] and actual_hash == source_hash,
             "current declared/actual source identity mismatch")
    # JSON normalizes only list/tuple representation, never numerical values.
    case_records = json.loads(json.dumps(cases))
    _require(case_records == contract["cases"], "golden cases changed")
    names = [case[0] for case in case_records]
    _require(len(names) == 4 and len(set(names)) == 4, "duplicate or missing golden cases")
    _require(len(contract["arrays"]) == 40, "expected 40 golden arrays")


def compare_arrays(current, reference, expected_shapes, output_report=None):
    """Compare all arrays in the completed case; retain diagnostics before raising."""
    import numpy as np
    report = output_report if output_report is not None else {}
    _require(set(current) == set(reference) == set(expected_shapes), "array key mismatch")
    failures = []
    for key, shape in expected_shapes.items():
        try:
            got, ref = np.asarray(current[key], float), np.asarray(reference[key], float)
            report[key] = {"current_shape": list(got.shape), "reference_shape": list(ref.shape)}
            _require(list(got.shape) == list(ref.shape) == shape, "array shape mismatch: " + key)
            _require(np.isfinite(got).all() and np.isfinite(ref).all(), "nonfinite array: " + key)
            _require(got.size > 0, "empty array: " + key)
            scale = max(float(np.abs(ref).max()), 1e-12)
            deviation = float(np.abs(got - ref).max()) / scale
            report[key]["relative_error"] = deviation
            _require(deviation < 1e-9, "golden mismatch %s: rel dev %.3e" % (key, deviation))
        except ValueError as error:
            report.setdefault(key, {})["error"] = str(error)
            failures.append(str(error))
    _require(not failures, "; ".join(failures))
    return report


def run_comparison(root, output_dir=None):
    """Execute only the four pinned current cases; never regenerate the reference."""
    import numpy as np
    root = Path(root).resolve()
    contract = load_contract()
    raw = validate_files(root, contract)
    output = Path(output_dir).resolve() if output_dir is not None else None
    if output is not None:
        _require(not output.exists(), "comparison output already exists; refusing replay")
        output.mkdir(parents=True)
    report = {"status": "started", "reference_sha256": _sha(raw), "cases": []}
    try:
        from fslab import simcache
        _require(Path(simcache.__file__).resolve() == root / "FiberScope/fslab/simcache.py",
                 "wrong simcache import origin")
        driver_dir = root / "FiberScope/scripts"
        sys.path.insert(0, str(driver_dir))
        try:
            driver = importlib.import_module("make_golden_engine")
        finally:
            sys.path.pop(0)
        _require(Path(driver.__file__).resolve() == driver_dir / "make_golden_engine.py",
                 "wrong golden driver import origin")
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            metadata = json.loads(str(saved["meta"]))
            validate_identity(contract, metadata, simcache.ENGINE_VERSION,
                              simcache.ENGINE_SRC_HASH, simcache.engine_src_hash(), driver.CASES)
            _require(set(saved.files) == set(contract["arrays"]) | {"meta"},
                     "reference keys changed")
            report.update(current_version=simcache.ENGINE_VERSION,
                          current_engine_hash=simcache.ENGINE_SRC_HASH)
            for case in driver.CASES:
                started = time.monotonic()
                result, solver_seconds = driver.run_case(case)
                name = case[0]
                arrays = {name + "_force": result.force_curve,
                          name + "_strain": result.strain_levels,
                          name + "_ccounts": result.contact_counts,
                          name + "_final": result.frames_xy[-1],
                          name + "_estrain": result.edge_strain[-1]}
                arrays.update({name + "_e_" + key: np.asarray(value, float)
                               for key, value in result.energies.items()})
                # Preserve the current result exactly once, before comparisons can fail.
                if output is not None:
                    with (output / (name + "_current.npz")).open("xb") as target:
                        np.savez_compressed(target, **arrays)
                shapes = {key: value for key, value in contract["arrays"].items()
                          if key.startswith(name + "_")}
                item = {"name": name, "solver_seconds": solver_seconds,
                        "wall_seconds": time.monotonic() - started}
                report["cases"].append(item)
                item["arrays"] = {}
                compare_arrays(arrays, {key: saved[key] for key in shapes}, shapes, item["arrays"])
        _require(sum(len(case["arrays"]) for case in report["cases"]) == 40,
                 "not all 40 arrays compared")
        validate_files(root, contract)
        report["status"] = "passed"
        return report
    except BaseException as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        if output is not None:
            with (output / "comparison_report.json").open("x") as target:
                json.dump(report, target, indent=2)
                target.write("\n")

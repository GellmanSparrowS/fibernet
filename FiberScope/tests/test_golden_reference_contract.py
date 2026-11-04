"""Synthetic contract checks only: never import or run the product solver."""
import copy
import importlib.util
from pathlib import Path
import shutil

import pytest


@pytest.fixture(scope="module")
def contract_module():
    path = Path(__file__).with_name("_golden_reference_contract.py")
    spec = importlib.util.spec_from_file_location("_tested_golden_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(module, **overrides):
    c = module.load_contract()
    values = dict(metadata=copy.deepcopy(c["reference_metadata"]),
                  version=c["current_version"], source_hash=c["current_engine_hash"],
                  actual_hash=c["current_engine_hash"], cases=copy.deepcopy(c["cases"]))
    values.update(overrides)
    return c, values


def test_exact_identity_admits_comparison_not_result(contract_module):
    c, values = identity(contract_module)
    assert contract_module.validate_identity(c, **values) is None


@pytest.mark.parametrize("field,value", [
    ("version", "e2v15"), ("version", "e2v13"),
    ("source_hash", "unreviewed"), ("actual_hash", "unreviewed"),
    ("actual_hash", None),
])
def test_unknown_identity_refused(contract_module, field, value):
    c, values = identity(contract_module, **{field: value})
    with pytest.raises(ValueError):
        contract_module.validate_identity(c, **values)


@pytest.mark.parametrize("mutation", ["reference_version", "case_parameter", "case_duplicate", "case_missing"])
def test_metadata_and_cases_refused(contract_module, mutation):
    c, values = identity(contract_module)
    if mutation == "reference_version":
        values["metadata"]["engine_version"] = "e2v14"
    elif mutation == "case_parameter":
        values["cases"][0][5]["num_steps"] += 1
    elif mutation == "case_duplicate":
        values["cases"][1] = values["cases"][0]
    else:
        values["cases"].pop()
    with pytest.raises(ValueError):
        contract_module.validate_identity(c, **values)


@pytest.fixture
def pinned_tree(contract_module, tmp_path):
    root = Path(__file__).resolve().parents[2]
    c = contract_module.load_contract()
    for relative in list(c["source_sha256"]) + [
            "FiberScope/tests/golden_engine.npz", "FiberScope/scripts/make_golden_engine.py"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    return tmp_path, c


def test_exact_files_and_driver_crlf_admitted(contract_module, pinned_tree):
    root, c = pinned_tree
    raw = contract_module.validate_files(root, c)
    assert raw
    driver = root / "FiberScope/scripts/make_golden_engine.py"
    driver.write_bytes(contract_module._normalized(driver.read_bytes()).replace(b"\n", b"\r\n"))
    assert contract_module.validate_files(root, c) == raw


@pytest.mark.parametrize("mutation", ["missing_reference", "reference_bytes", "source_bytes", "driver_body"])
def test_file_changes_refused(contract_module, pinned_tree, mutation):
    root, c = pinned_tree
    if mutation == "missing_reference":
        (root / "FiberScope/tests/golden_engine.npz").unlink()
    else:
        relative = {"reference_bytes": "FiberScope/tests/golden_engine.npz",
                    "source_bytes": next(iter(c["source_sha256"])),
                    "driver_body": "FiberScope/scripts/make_golden_engine.py"}[mutation]
        target = root / relative
        target.write_bytes(target.read_bytes() + b"# changed")
    with pytest.raises(ValueError):
        contract_module.validate_files(root, c)


def test_numeric_comparison_required_and_original_tolerance(contract_module):
    expected = {"x": [2]}
    assert contract_module.compare_arrays({"x": [1.0, 1.0]}, {"x": [1.0, 1.0]}, expected)["x"]["relative_error"] == 0
    with pytest.raises(ValueError, match="golden mismatch"):
        contract_module.compare_arrays({"x": [1.0, 1.0 + 2e-9]}, {"x": [1.0, 1.0]}, expected)


@pytest.mark.parametrize("current,reference,shapes", [
    ({"x": [float("nan")]}, {"x": [1.0]}, {"x": [1]}),
    ({"x": [1.0]}, {"x": [float("inf")]}, {"x": [1]}),
    ({"x": [1.0]}, {"x": [1.0, 1.0]}, {"x": [2]}),
    ({"y": [1.0]}, {"x": [1.0]}, {"x": [1]}),
    ({"x": []}, {"x": []}, {"x": [0]}),
])
def test_bad_numeric_contract_refused(contract_module, current, reference, shapes):
    with pytest.raises(ValueError):
        contract_module.compare_arrays(current, reference, shapes)


def test_failure_retains_all_completed_case_array_diagnostics(contract_module):
    report = {}
    with pytest.raises(ValueError):
        contract_module.compare_arrays({"x": [2.0], "y": [1.0]},
                                       {"x": [1.0], "y": [1.0]},
                                       {"x": [1], "y": [1]}, report)
    assert report["x"]["relative_error"] == 1.0
    assert report["y"]["relative_error"] == 0.0
    assert "error" in report["x"]

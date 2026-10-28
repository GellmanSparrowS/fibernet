"""Batch CSV orchestration unit tests with a deterministic stub solver.

These tests validate persistence and resume behavior, not simulation physics.
"""
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import fibernet.easy as easy


@pytest.fixture
def batch(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    output = tmp_path / "output"
    calls = []
    failures = set()
    save_failures = set()

    class Result:
        max_force = 12.5
        max_stretch = 1.5
        mean_stretch = 1.2
        std_stretch = 0.1
        time_seconds = 0.25

        def __init__(self, name):
            self.name = name

        def save(self, path, detailed):
            assert detailed is True
            if self.name in save_failures:
                raise OSError("injected result save failure")
            Path(path).write_text("{}")

    class Engine:
        def __init__(self):
            calls.append("init")

        def stretch_test(self, graph, **kwargs):
            calls.append(graph.name)
            if graph.name in failures:
                raise RuntimeError("injected solver failure")
            return Result(graph.name)

    monkeypatch.setattr(easy, "TaichiEngine", Engine)
    monkeypatch.setattr(
        easy.StructureGraph, "from_dict",
        lambda data: SimpleNamespace(
            name=data["name"], num_nodes=4, num_edges=3,
            edges={eid: SimpleNamespace(internal_points=None) for eid in (9, 2, 17)}),
    )

    def add(*names):
        for name in names:
            (inputs / name).write_text(json.dumps({"name": name}))

    def run(**kwargs):
        return Path(easy.batch_simulate_from_json(str(inputs), str(output), **kwargs))

    return SimpleNamespace(inputs=inputs, output=output, calls=calls,
                           failures=failures, save_failures=save_failures,
                           add=add, run=run,
                           csv=output / "simulation_results.csv")


def read_rows(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert None not in reader.fieldnames
        assert all(None not in row and None not in row.values() for row in rows)
        return rows


def test_resume_retains_old_rows_and_custom_columns(batch):
    batch.add("a.json")
    path = batch.run()
    old = read_rows(path)[0]
    old["legacy_note"] = "kept, verbatim\nincluding newline"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(old))
        writer.writeheader()
        writer.writerow(old)
    batch.add("b.json")
    batch.run()
    rows = read_rows(path)
    assert [row["filename"] for row in rows] == ["a.json", "b.json"]
    assert rows[0] == old
    assert rows[1]["legacy_note"] == ""
    assert batch.calls.count("a.json") == 1


def test_completed_resume_preserves_bytes_without_engine(batch, monkeypatch):
    batch.add("a.json")
    path = batch.run()
    before = path.read_bytes()

    def forbidden():
        pytest.fail("completed resume must not construct optional solver")

    monkeypatch.setattr(easy, "TaichiEngine", forbidden)
    assert batch.run() == path
    assert path.read_bytes() == before


@pytest.mark.parametrize("failed", ["a.json", "b.json"])
def test_mixed_success_failure_uses_complete_schema(batch, failed):
    batch.add("a.json", "b.json")
    batch.failures.add(failed)
    rows = read_rows(batch.run())
    assert [r["filename"] for r in rows] == ["a.json", "b.json"]
    for row in rows:
        if row["filename"] == failed:
            assert row["success"] == "False"
            assert row["error"] == "injected solver failure"
            assert row["max_force"] == ""
        else:
            assert row["success"] == "True"
            assert row["error"] == ""
            assert row["max_force"] == "12.5"


def test_result_save_failure_has_exactly_one_outcome(batch):
    batch.add("a.json")
    batch.save_failures.add("a.json")
    rows = read_rows(batch.run())
    assert len(rows) == 1
    assert rows[0]["success"] == "False"
    assert rows[0]["error"] == "injected result save failure"


@pytest.mark.parametrize("contents", [
    "", "filename,success,success\na.json,True,True\n",
    "filename,,success\na.json,,True\n", "success\nTrue\n",
    "filename\na.json\n", "filename,success\na.json\n",
    "filename,success\na.json,True,extra\n",
    "filename,success\n,True\n",
    "filename,success\na.json,True\na.json,False\n",
    "filename,success\na.json,maybe\n",
    'filename,success\n"a.json,True\n',
])
def test_malformed_existing_csv_is_not_clobbered(batch, contents):
    batch.add("b.json")
    batch.output.mkdir()
    batch.csv.write_text(contents)
    before = batch.csv.read_bytes()
    with pytest.raises(ValueError, match="CSV"):
        batch.run()
    assert batch.csv.read_bytes() == before
    assert batch.calls == []


@pytest.mark.parametrize("failure", ["serialize", "fsync", "replace"])
def test_atomic_write_failure_preserves_original_bytes(batch, monkeypatch, failure):
    batch.add("a.json")
    path = batch.run()
    before = path.read_bytes()
    batch.add("b.json")
    if failure == "serialize":
        def broken_write(self, rows):
            self.writerow(rows[0])
            raise OSError("injected serialization failure")
        monkeypatch.setattr(csv.DictWriter, "writerows", broken_write)
    elif failure == "fsync":
        def broken_fsync(fd):
            raise OSError("injected flush failure")
        monkeypatch.setattr(os, "fsync", broken_fsync)
    else:
        def broken_replace(src, dst):
            assert Path(src).parent == path.parent
            raise OSError("injected replacement failure")
        monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="injected"):
        batch.run()
    assert path.read_bytes() == before
    assert sorted(p.name for p in batch.output.iterdir()) == [
        "a_result.json", "b_result.json", "simulation_results.csv"
    ]


@pytest.mark.parametrize("kwargs", [{"backend": "other"}, {"mode": "dynamics"}])
def test_invalid_options_fail_before_mutating_files(batch, kwargs):
    batch.add("a.json")
    with pytest.raises(ValueError):
        batch.run(**kwargs)
    assert not batch.output.exists()
    assert batch.calls == []


def test_resume_keeps_failed_rows_and_does_not_retry_them(batch):
    batch.add("a.json")
    batch.failures.add("a.json")
    path = batch.run()
    failed = read_rows(path)[0]
    batch.failures.clear()
    batch.add("b.json")
    rows = read_rows(batch.run())
    assert rows[0] == failed
    assert rows[1]["success"] == "True"
    assert batch.calls.count("a.json") == 1

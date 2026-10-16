"""Batch experiment identity contracts using native graphs and a stub engine.

These are loading, dispatch and persistence tests, not simulation-physics tests.
The digest expectations are computed independently of the implementation helpers.
"""
import builtins
import copy
import csv
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fibernet as fn
import fibernet.easy as easy
from fibernet.core.material import Material
from fibernet.core.structure_graph import StructureGraph


SOURCES = {
    "fibernet/easy.py",
    "fibernet/sim/accelerated.py",
    "fibernet/core/structure_graph.py",
    "fibernet/core/material.py",
}
DEFAULT_PARAMETERS = {
    "backend": "spring", "mode": "stretch", "target_stretch": 1.5,
    "stiffness": 1e5, "damping": 0.3, "num_steps": 1000,
    "save_interval": 200, "auto_steps": False,
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def object_digest(value):
    return digest(canonical(value).encode("utf-8"))


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert len(reader.fieldnames) == len(set(reader.fieldnames))
        assert all(None not in row and None not in row.values() for row in rows)
        return rows


def write_rows(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if fields is None else fields
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def graph_bytes(name="a.json", *, ensure_ascii=True):
    graph = StructureGraph(dimension=2, box_size=(2, 2))
    graph.metadata.update(name=name, label="纤维 café")
    a = graph.add_node([0, 0])
    b = graph.add_node([1, 1])
    graph.add_edge(a, b, radius=0.2,
                   material=Material(name="fixture", youngs_modulus=1e8))
    return json.dumps(graph.to_dict(), ensure_ascii=ensure_ascii,
                      sort_keys=True).encode("utf-8")


@pytest.fixture
def batch(tmp_path, monkeypatch):
    state = SimpleNamespace(
        inputs=tmp_path / "inputs", output=tmp_path / "output",
        calls=[], solves=[], failures=set(), save_failures=set(),
        on_init=None, on_solve=None,
    )
    state.inputs.mkdir()
    state.csv = state.output / "simulation_results.csv"

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
            if self.name in state.save_failures:
                raise OSError("injected result save failure")
            Path(path).write_text(canonical({"name": self.name}), encoding="utf-8")

    class Engine:
        def __init__(self):
            state.calls.append("init")
            if state.on_init is not None:
                state.on_init()

        def stretch_test(self, graph, **kwargs):
            assert isinstance(graph, StructureGraph)
            name = graph.metadata["name"]
            state.calls.append(name)
            state.solves.append((graph.to_dict(), kwargs))
            if state.on_solve is not None:
                state.on_solve(name)
            if name in state.failures:
                raise RuntimeError("injected solver failure")
            return Result(name)

    monkeypatch.setattr(easy, "TaichiEngine", Engine)

    def add(name="a.json", raw=None):
        path = state.inputs / name
        path.write_bytes(graph_bytes(name) if raw is None else raw)
        return path

    def run(**kwargs):
        return fn.batch_simulate_from_json(str(state.inputs), str(state.output), **kwargs)

    state.add = add
    state.run = run
    return state


def snapshot_output(batch):
    if not batch.output.exists():
        return None
    return {str(path.relative_to(batch.output)): path.read_bytes()
            for path in batch.output.rglob("*") if path.is_file()}


def reject_without_work(batch, **kwargs):
    before = snapshot_output(batch)
    batch.calls.clear()
    with pytest.raises(ValueError) as error:
        batch.run(**kwargs)
    assert batch.calls == []
    assert snapshot_output(batch) == before
    return str(error.value)


def assert_binding(row, raw, parameters=DEFAULT_PARAMETERS):
    binding = json.loads(row["run_binding"])
    assert set(binding) == {
        "schema", "input_sha256", "parameters", "parameters_sha256",
        "implementation", "implementation_sha256",
    }
    assert row["run_binding"] == canonical(binding)
    assert binding["schema"] == "fibernet-batch-run-1"
    assert binding["input_sha256"] == digest(raw)
    assert binding["parameters"] == parameters
    assert binding["parameters_sha256"] == object_digest(parameters)
    implementation = binding["implementation"]
    assert set(implementation) == {"recipe", "sources", "runtime"}
    assert implementation["recipe"] == "fibernet-batch-source-1"
    assert set(implementation["sources"]) == SOURCES
    root = Path(easy.__file__).resolve().parent.parent
    for relative_path, expected in implementation["sources"].items():
        assert expected == digest((root / relative_path).read_bytes())
    runtime = implementation["runtime"]
    assert set(runtime) == {"python", "numpy", "taichi"}
    assert runtime["python"] == platform.python_version()
    assert runtime["numpy"] == importlib.metadata.version("numpy")
    try:
        taichi_version = importlib.metadata.version("taichi")
    except importlib.metadata.PackageNotFoundError:
        taichi_version = None
    assert runtime["taichi"] == taichi_version
    assert binding["implementation_sha256"] == object_digest(implementation)
    return binding


def test_public_signature_and_native_graph_dispatch_are_unchanged(batch):
    assert fn.batch_simulate_from_json is easy.batch_simulate_from_json
    signature = inspect.signature(fn.batch_simulate_from_json)
    assert list(signature.parameters) == [
        "json_dir", "output_dir", "mode", "target_stretch", "stiffness",
        "damping", "num_steps", "save_interval", "backend",
    ]
    assert signature.parameters["num_steps"].default == 1000
    assert signature.parameters["save_interval"].default == 200
    path = batch.add()
    assert Path(batch.run()) == batch.csv
    row = read_rows(batch.csv)[0]
    assert row["success"] == "True"
    assert row["n_nodes"] == "2" and row["n_edges"] == "1"
    assert row["max_force"] == "12.5"
    assert_binding(row, path.read_bytes())
    expected = StructureGraph.load_json(str(path)).to_dict()
    assert batch.solves[0][0] == expected
    assert batch.solves[0][1] == {
        key: value for key, value in DEFAULT_PARAMETERS.items()
        if key not in {"backend", "mode"}
    }


@pytest.mark.parametrize("failure", ["load", "solve", "save"])
def test_each_failure_outcome_has_truthful_binding_and_is_not_retried(batch, failure):
    path = batch.add(raw=b'{"broken":' if failure == "load" else None)
    if failure == "solve":
        batch.failures.add(path.name)
    elif failure == "save":
        batch.save_failures.add(path.name)
    batch.run()
    rows = read_rows(batch.csv)
    assert len(rows) == 1 and rows[0]["success"] == "False"
    assert rows[0]["error"]
    assert_binding(rows[0], path.read_bytes())
    before = snapshot_output(batch)
    batch.calls.clear()
    batch.failures.clear()
    batch.save_failures.clear()
    batch.run()
    assert batch.calls == []
    assert snapshot_output(batch) == before


@pytest.mark.parametrize("kwargs", [
    {"target_stretch": 1.6}, {"stiffness": 2e5}, {"damping": 0.4},
    {"num_steps": 1001}, {"save_interval": 201},
])
def test_each_effective_parameter_change_rejects_existing_outcomes(batch, kwargs):
    batch.add()
    batch.run()
    message = reject_without_work(batch, **kwargs)
    assert "a.json" in message
    assert "parameter" in message.lower()
    assert "fresh" in message.lower()


@pytest.mark.parametrize("kwargs", [{"backend": "fem"}, {"mode": "dynamics"}])
def test_unsupported_backend_or_mode_rejects_before_fresh_or_hit_work(batch, kwargs):
    batch.add()
    reject_without_work(batch, **kwargs)
    assert not batch.output.exists()
    batch.run()
    reject_without_work(batch, **kwargs)


def test_none_steps_and_numpy_scalars_have_effective_builtin_identity(batch):
    path = batch.add()
    batch.run(num_steps=None, stiffness=np.float64(1e5),
              target_stretch=np.float32(1.5), damping=np.float64(0.3),
              save_interval=np.int64(200))
    expected = dict(DEFAULT_PARAMETERS, num_steps=5000)
    assert_binding(read_rows(batch.csv)[0], path.read_bytes(), expected)
    before = snapshot_output(batch)
    batch.calls.clear()
    batch.run(num_steps=np.int64(5000), stiffness=100000,
              target_stretch=1.5, damping=0.3, save_interval=200)
    assert batch.calls == []
    assert snapshot_output(batch) == before


@pytest.mark.parametrize("name", ["target_stretch", "stiffness", "damping"])
@pytest.mark.parametrize("value", [True, np.bool_(False), float("nan"),
                                    float("inf"), -float("inf"), "1.5", 1 + 0j])
def test_invalid_physical_scalar_fails_before_a_stale_hit(batch, name, value):
    batch.add()
    batch.run()
    reject_without_work(batch, **{name: value})


@pytest.mark.parametrize("name", ["num_steps", "save_interval"])
@pytest.mark.parametrize("value", [True, np.bool_(True), 0, -1, 1.5,
                                    np.float64(2), "2", float("nan")])
def test_step_counts_are_positive_nonboolean_integers(batch, name, value):
    batch.add()
    batch.run()
    reject_without_work(batch, **{name: value})


@pytest.mark.parametrize("change", ["geometry", "material", "whitespace", "same_size"])
def test_exact_input_bytes_bind_geometry_material_and_formatting(batch, change):
    path = batch.add()
    batch.run()
    before = path.read_bytes()
    stat = path.stat()
    if change == "whitespace":
        after = before + b"\n"
    elif change == "same_size":
        after = before.replace(b'"radius": 0.2', b'"radius": 0.3')
        assert len(after) == len(before) and after != before
    else:
        document = json.loads(before)
        if change == "geometry":
            document["nodes"][1]["position"][0] = 1.25
        else:
            document["edges"][0]["material"]["youngs_modulus"] *= 2
        after = json.dumps(document).encode("utf-8")
    path.write_bytes(after)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    message = reject_without_work(batch)
    assert "a.json" in message and "input" in message.lower()
    assert "fresh" in message.lower()


@pytest.mark.parametrize("component", sorted(SOURCES) + ["python", "numpy", "taichi"])
def test_source_and_runtime_changes_invalidate_all_hits(batch, monkeypatch, component):
    batch.add()
    batch.run()
    implementation = copy.deepcopy(easy._batch_implementation_binding())
    if component in SOURCES:
        previous = implementation["sources"][component]
        implementation["sources"][component] = ("0" if previous[0] != "0" else "1") + previous[1:]
    else:
        implementation["runtime"][component] = "changed-version"
    monkeypatch.setattr(easy, "_batch_implementation_binding", lambda: implementation)
    message = reject_without_work(batch)
    assert "implementation" in message.lower()
    assert "fresh" in message.lower()


@pytest.mark.parametrize("existing", [False, True])
def test_unavailable_source_fails_before_engine_or_output(batch, monkeypatch, existing):
    batch.add()
    if existing:
        batch.run()
    before = snapshot_output(batch)
    batch.calls.clear()
    source = Path(easy.__file__).resolve()
    original = Path.open

    def unavailable(path, *args, **kwargs):
        if path.resolve() == source:
            raise OSError("injected missing source bytes")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unavailable)
    with pytest.raises((ValueError, OSError), match="source|unverified|implementation"):
        batch.run()
    assert batch.calls == []
    assert snapshot_output(batch) == before


def test_runtime_identity_does_not_import_taichi_and_handles_absence(monkeypatch):
    original_import = builtins.__import__
    original_version = importlib.metadata.version

    def no_taichi_import(name, *args, **kwargs):
        if name == "taichi" or name.startswith("taichi."):
            pytest.fail("binding must not import or initialize Taichi")
        return original_import(name, *args, **kwargs)

    def version(name):
        if name == "taichi":
            raise importlib.metadata.PackageNotFoundError(name)
        return original_version(name)

    monkeypatch.setattr(builtins, "__import__", no_taichi_import)
    monkeypatch.setattr(importlib.metadata, "version", version)
    implementation = easy._batch_implementation_binding()
    assert implementation["runtime"]["taichi"] is None


def corrupt_binding(binding, corruption):
    binding = copy.deepcopy(binding)
    if corruption == "unknown_schema":
        binding["schema"] = "fibernet-batch-run-999"
    elif corruption == "missing_field":
        del binding["parameters_sha256"]
    elif corruption == "short_input_digest":
        binding["input_sha256"] = "a" * 63
    elif corruption == "nonhex_input_digest":
        binding["input_sha256"] = "z" * 64
    elif corruption == "wrong_parameter_digest":
        binding["parameters_sha256"] = "0" * 64
    elif corruption == "wrong_implementation_digest":
        binding["implementation_sha256"] = "0" * 64
    elif corruption == "missing_parameter":
        del binding["parameters"]["auto_steps"]
        binding["parameters_sha256"] = object_digest(binding["parameters"])
    elif corruption == "changed_parameters":
        binding["parameters"]["stiffness"] = 2e5
        binding["parameters_sha256"] = object_digest(binding["parameters"])
    elif corruption == "missing_source":
        del binding["implementation"]["sources"]["fibernet/easy.py"]
        binding["implementation_sha256"] = object_digest(binding["implementation"])
    elif corruption == "unknown_recipe":
        binding["implementation"]["recipe"] = "fibernet-batch-source-999"
        binding["implementation_sha256"] = object_digest(binding["implementation"])
    elif corruption == "changed_implementation":
        binding["implementation"]["runtime"]["numpy"] = "different"
        binding["implementation_sha256"] = object_digest(binding["implementation"])
    return canonical(binding)


@pytest.mark.parametrize("corruption", [
    "blank", "not_json", "list", "partial", "unknown_schema", "missing_field",
    "short_input_digest", "nonhex_input_digest", "wrong_parameter_digest",
    "wrong_implementation_digest", "missing_parameter", "changed_parameters",
    "missing_source", "unknown_recipe", "changed_implementation",
])
@pytest.mark.parametrize("disjoint", [False, True])
def test_every_retained_row_must_have_a_valid_matching_binding(batch, corruption, disjoint):
    old_path = batch.add()
    batch.run()
    rows = read_rows(batch.csv)
    invalid_text = {"blank": "", "not_json": "not JSON", "list": "[]", "partial": "{}"}
    rows[0]["run_binding"] = invalid_text.get(corruption) or corrupt_binding(
        json.loads(rows[0]["run_binding"]), corruption)
    if corruption == "blank":
        rows[0]["run_binding"] = ""
    write_rows(batch.csv, rows)
    if disjoint:
        old_path.unlink()
        batch.add("b.json")
    message = reject_without_work(batch)
    assert "a.json" in message
    assert "fresh" in message.lower()


@pytest.mark.parametrize("current", ["same", "disjoint", "none"])
@pytest.mark.parametrize("collision", [False, True])
def test_nonempty_legacy_csv_and_reserved_column_collision_are_not_migrated(batch, current, collision):
    if current == "same":
        batch.add("a.json")
    elif current == "disjoint":
        batch.add("b.json")
    row = {"filename": "a.json", "success": "True", "note": "legacy"}
    if collision:
        row["run_binding"] = "my old custom value"
    write_rows(batch.csv, [row])
    message = reject_without_work(batch)
    assert "fresh" in message.lower()


def test_mixed_bound_and_unbound_history_rejects_before_new_file(batch):
    batch.add("a.json")
    batch.run()
    rows = read_rows(batch.csv)
    rows.append({"filename": "old.json", "success": "False", "run_binding": ""})
    write_rows(batch.csv, rows)
    batch.add("b.json")
    reject_without_work(batch)


@pytest.mark.parametrize("corruption", ["duplicate_key", "nonfinite", "wrong_type"])
def test_reserved_binding_json_rejects_ambiguous_or_nonfinite_values(batch, corruption):
    batch.add()
    batch.run()
    rows = read_rows(batch.csv)
    if corruption == "duplicate_key":
        rows[0]["run_binding"] = rows[0]["run_binding"].replace(
            '"schema":"fibernet-batch-run-1"',
            '"schema":"fibernet-batch-run-1","schema":"fibernet-batch-run-1"',
        )
    elif corruption == "nonfinite":
        rows[0]["run_binding"] = rows[0]["run_binding"].replace(
            '"target_stretch":1.5', '"target_stretch":NaN',
        )
    else:
        binding = json.loads(rows[0]["run_binding"])
        binding["parameters"]["num_steps"] = True
        binding["parameters_sha256"] = object_digest(binding["parameters"])
        rows[0]["run_binding"] = canonical(binding)
    write_rows(batch.csv, rows)
    reject_without_work(batch)


def test_empty_legacy_header_is_upgraded_and_custom_columns_are_preserved(batch):
    path = batch.add()
    write_rows(batch.csv, [], fields=["filename", "success", "custom_note"])
    batch.run()
    row = read_rows(batch.csv)[0]
    assert row["custom_note"] == ""
    assert_binding(row, path.read_bytes())


def test_partial_resume_retains_absent_history_and_multiline_custom_values(batch):
    old_path = batch.add("a.json")
    batch.failures.add("a.json")
    batch.run()
    rows = read_rows(batch.csv)
    rows[0]["note"] = 'kept, with "quotes"\nand a new line'
    write_rows(batch.csv, rows)
    old = rows[0].copy()
    old_path.unlink()
    new_path = batch.add("b.json")
    batch.calls.clear()
    batch.run()
    resumed = read_rows(batch.csv)
    assert resumed[0] == old
    assert resumed[1]["filename"] == "b.json" and resumed[1]["note"] == ""
    assert_binding(resumed[1], new_path.read_bytes())
    assert batch.calls == ["init", "b.json"]


@pytest.mark.parametrize("existing", [False, True])
def test_no_current_inputs_preserves_valid_history_without_engine(batch, existing):
    if existing:
        path = batch.add()
        batch.run()
        path.unlink()
    before = snapshot_output(batch)
    batch.calls.clear()
    assert batch.run() == ""
    assert batch.calls == []
    assert snapshot_output(batch) == before


def test_no_current_inputs_still_require_matching_historical_parameters(batch):
    path = batch.add()
    batch.run()
    path.unlink()
    reject_without_work(batch, stiffness=2e5)


def test_late_sorted_overlap_mismatch_rejects_before_early_pending_file(batch):
    old_path = batch.add("z.json")
    batch.run()
    batch.add("a.json")
    old_path.write_bytes(old_path.read_bytes() + b"\n")
    reject_without_work(batch)
    assert not (batch.output / "a_result.json").exists()


def test_preflight_hashes_every_current_file_before_engine(batch, monkeypatch):
    for name in ["a.json", "b.json", "z.json"]:
        batch.add(name)
    events = []
    original = easy._batch_input_sha256

    def recorded(path):
        if Path(path).parent == batch.inputs:
            events.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(easy, "_batch_input_sha256", recorded)
    batch.on_init = lambda: events.append("init")
    batch.run()
    assert events[:4] == ["a.json", "b.json", "z.json", "init"]


def test_input_hash_preflight_uses_bounded_stream_reads(tmp_path, monkeypatch):
    raw = b"native input bytes\n" * 140000
    path = tmp_path / "large.json"
    path.write_bytes(raw)
    original_open = Path.open
    requested_sizes = []

    class BoundedReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size=-1):
            assert 0 < size < len(raw), "preflight must stream bounded byte chunks"
            requested_sizes.append(size)
            return self.stream.read(size)

    def tracked(candidate, *args, **kwargs):
        stream = original_open(candidate, *args, **kwargs)
        return BoundedReader(stream) if candidate == path else stream

    monkeypatch.setattr(Path, "open", tracked)
    assert easy._batch_input_sha256(path) == digest(raw)
    assert len(requested_sizes) > 1


def test_hash_read_failure_aborts_whole_preflight_without_failure_row(batch, monkeypatch):
    batch.add("a.json")
    batch.add("z.json")
    original = easy._batch_input_sha256

    def unreadable(path):
        if Path(path).name == "z.json":
            raise PermissionError("injected unreadable input")
        return original(path)

    monkeypatch.setattr(easy, "_batch_input_sha256", unreadable)
    with pytest.raises((OSError, ValueError), match="unreadable|input"):
        batch.run()
    assert batch.calls == []
    assert not batch.output.exists()


def test_pending_input_changed_after_preflight_is_not_a_bound_failure(batch):
    path = batch.add()
    batch.on_init = lambda: path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="chang|input|digest"):
        batch.run()
    assert batch.solves == []
    assert not batch.csv.exists()
    assert not (batch.output / "a_result.json").exists()


def test_later_capture_mutation_keeps_only_already_completed_checkpoint(batch):
    first = batch.add("a.json")
    later = batch.add("b.json")

    def mutate_after_first(name):
        if name == "a.json":
            later.write_bytes(later.read_bytes() + b"\n")

    batch.on_solve = mutate_after_first
    with pytest.raises(ValueError, match="chang|input|digest"):
        batch.run()
    rows = read_rows(batch.csv)
    assert [row["filename"] for row in rows] == ["a.json"]
    assert_binding(rows[0], first.read_bytes())
    assert not (batch.output / "b_result.json").exists()
    assert batch.calls == ["init", "a.json"]


def test_capture_read_failure_propagates_instead_of_authenticating_failure(batch, monkeypatch):
    path = batch.add()
    original = Path.open
    state = {"capturing": False}
    batch.on_init = lambda: state.update(capturing=True)

    def unreadable(candidate, *args, **kwargs):
        if candidate == path and state["capturing"]:
            raise PermissionError("injected capture read failure")
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unreadable)
    with pytest.raises((OSError, ValueError), match="capture|input"):
        batch.run()
    assert batch.solves == []
    assert not batch.csv.exists()


def test_graph_is_parsed_from_exact_captured_bytes_without_reopening_path(batch, monkeypatch):
    path = batch.add()
    raw = path.read_bytes()
    original = easy._batch_graph_from_bytes
    captured = []

    def change_path_during_parse(payload):
        captured.append(payload)
        path.write_bytes(b"this path now contains invalid JSON")
        return original(payload)

    monkeypatch.setattr(easy, "_batch_graph_from_bytes", change_path_during_parse)
    batch.run()
    assert captured == [raw]
    row = read_rows(batch.csv)[0]
    assert row["success"] == "True"
    assert_binding(row, raw)
    assert batch.solves[0][0] == StructureGraph.from_dict(json.loads(raw)).to_dict()


@pytest.mark.parametrize("encoding", ["utf8", "utf8_nonascii", "crlf", "utf8_bom",
                                        "utf16", "utf32", "invalid_utf8", "malformed_json"])
def test_captured_decoder_matches_historical_default_text_loader(batch, encoding):
    raw = graph_bytes(ensure_ascii=encoding != "utf8_nonascii")
    if encoding == "crlf":
        raw = json.dumps(json.loads(raw), indent=2).replace("\n", "\r\n").encode("utf-8")
    elif encoding == "utf8_bom":
        raw = b"\xef\xbb\xbf" + raw
    elif encoding in {"utf16", "utf32"}:
        raw = raw.decode("utf-8").encode(encoding)
    elif encoding == "invalid_utf8":
        # Keep a complete graph, including the stub engine's required name.
        # 0xff is invalid UTF-8 but valid cp1252: decoding success must not
        # accidentally turn into an unrelated engine KeyError.
        data = json.loads(raw)
        data["metadata"]["label"] = "ENCODING_SENTINEL"
        raw = json.dumps(data).encode("ascii").replace(b"ENCODING_SENTINEL", b"\xff")
    elif encoding == "malformed_json":
        raw = b'{"nodes": [}'
    path = batch.add(raw=raw)
    try:
        expected = StructureGraph.load_json(str(path))
    except (ValueError, UnicodeError) as error:
        with pytest.raises(type(error)):
            easy._batch_graph_from_bytes(raw)
        expected_success = "False"
    else:
        actual = easy._batch_graph_from_bytes(raw)
        assert actual.to_dict() == expected.to_dict()
        expected_success = "True"
    batch.run()
    row = read_rows(batch.csv)[0]
    assert row["success"] == expected_success
    assert_binding(row, raw)


@pytest.mark.parametrize("artifact", ["missing", "corrupt", "intact"])
def test_exact_csv_hit_needs_no_engine_or_result_artifact_validation(batch, artifact):
    batch.add()
    batch.run()
    result = batch.output / "a_result.json"
    if artifact == "missing":
        result.unlink()
    elif artifact == "corrupt":
        result.write_bytes(b"not a result JSON")
    before = snapshot_output(batch)
    batch.calls.clear()
    assert Path(batch.run()) == batch.csv
    assert batch.calls == []
    assert snapshot_output(batch) == before


def test_same_input_bytes_in_a_different_directory_can_reuse_outcome(batch, tmp_path):
    path = batch.add()
    raw = path.read_bytes()
    batch.run()
    other = tmp_path / "copied_inputs"
    other.mkdir()
    (other / path.name).write_bytes(raw)
    before = snapshot_output(batch)
    batch.calls.clear()
    assert Path(fn.batch_simulate_from_json(str(other), str(batch.output))) == batch.csv
    assert batch.calls == []
    assert snapshot_output(batch) == before


@pytest.mark.parametrize("text_encoding,accepted", [("utf-8", False), ("cp1252", True)])
def test_invalid_utf8_graph_against_explicit_disk_decoder(batch, monkeypatch,
                                                         text_encoding, accepted):
    """Exercise both decoding policies without changing the process locale.

    Explicit cp1252 injection is a unit test, not a Windows execution claim.
    The independent oracle reads a real disk file with an explicit codec.
    """
    data = json.loads(graph_bytes())
    data["metadata"]["label"] = "ENCODING_SENTINEL"
    raw = json.dumps(data).encode("ascii").replace(b"ENCODING_SENTINEL", b"\xff")
    path = batch.add(raw=raw)
    wrapper = io.TextIOWrapper

    def codec_wrapper(buffer, *args, **kwargs):
        kwargs.setdefault("encoding", text_encoding)
        return wrapper(buffer, *args, **kwargs)

    monkeypatch.setattr(io, "TextIOWrapper", codec_wrapper)
    if accepted:
        with path.open(encoding=text_encoding) as stream:
            expected = StructureGraph.from_dict(json.load(stream))
        assert expected.metadata["name"] == "a.json"
        assert expected.metadata["label"] == "ÿ"
        assert easy._batch_graph_from_bytes(raw).to_dict() == expected.to_dict()
    else:
        with pytest.raises(UnicodeDecodeError):
            with path.open(encoding=text_encoding) as stream:
                json.load(stream)
        with pytest.raises(UnicodeDecodeError):
            easy._batch_graph_from_bytes(raw)
    batch.run()
    row = read_rows(batch.csv)[0]
    assert row["success"] == str(accepted)
    assert len(batch.solves) == int(accepted)
    if accepted:
        assert batch.solves[0][0] == expected.to_dict()
    assert_binding(row, raw)

"""No-engine lower-bound admission tests; no native physics claims."""

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations import lammps_backend as backend
from test_lammps_backend_contract import fixture, prepared

RUNTIME = backend.RuntimeSpec("cli_serial_static", "/fixture/python", "/fixture/lmp")
MIB = 1024**2


@pytest.fixture(autouse=True)
def forbid_launch():
    with patch(
        "subprocess.Popen", side_effect=AssertionError("native launch forbidden")
    ):
        yield


def chain(nodes=256, steps=100000):
    graph = StructureGraph(dimension=3)
    for i in range(nodes):
        graph.add_node([i * 0.0001, 0, 0])
    for i in range(nodes - 1):
        graph.add_edge(i, i + 1, n_internal=0)
    options = dict(
        length_unit="m",
        maximum_bond_length_m=0.1,
        masses_kg={i: 1.0 for i in graph.nodes},
        bonds={i: backend.HarmonicBond(1.0, 0.0001) for i in graph.edges},
        run=backend.RunSpec(1e-6, steps, 1, ((-1, 1),) * 3),
    )
    return graph, options


@pytest.mark.parametrize("budget", [8 * MIB, 16 * MIB])
def test_impossible_maximum_rejected_without_runtime_or_files(tmp_path, budget):
    graph, options = chain()
    before = json.dumps(graph.to_dict(), sort_keys=True)
    p = backend.prepare_harmonic_beads(graph, **options)
    limits = backend.ResourceLimits(output_bytes=budget, file_size_bytes=budget)
    target = tmp_path / "never-created"
    with (
        patch.object(backend.RuntimeSpec, "checked") as checked,
        patch.object(Path, "mkdir", side_effect=AssertionError("directory creation")),
        patch.object(Path, "write_text", side_effect=AssertionError("artifact write")),
    ):
        with pytest.raises(backend.LAMMPSContractError, match="563205632 bytes"):
            backend.LAMMPSTask(p, RUNTIME, target, limits)
    checked.assert_not_called()
    assert not target.exists()
    assert json.dumps(graph.to_dict(), sort_keys=True) == before


@pytest.mark.parametrize("field", ["output_bytes", "file_size_bytes"])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_budget_boundary_is_strict_and_checks_each_limit(tmp_path, field, delta):
    # 25 frames, including the initial state, times two 22-byte atom rows.
    p = prepared(run=backend.RunSpec(1e-6, 24, 1, ((-0.01, 0.01),) * 3))
    floor = 1100
    limits = replace(backend.ResourceLimits(), **{field: floor + delta})
    target = tmp_path / "never-created"
    with patch.object(backend.RuntimeSpec, "checked", return_value=RUNTIME) as checked:
        if delta < 0:
            with pytest.raises(backend.LAMMPSContractError, match="1100 bytes"):
                backend.LAMMPSTask(p, RUNTIME, target, limits)
            checked.assert_not_called()
        else:
            task = backend.LAMMPSTask(p, RUNTIME, target, limits)
            checked.assert_called_once_with()
            assert task.state == backend.TaskState.CREATED
            assert task.output_forecast is None
            assert task.prepared is p
    assert not target.exists()


@pytest.mark.parametrize("nodes,steps", [(2, 0), (2, 100000), (64, 100), (256, 0)])
def test_small_and_run_zero_admitted_without_preparation_changes(
    tmp_path, nodes, steps
):
    graph, options = chain(nodes, steps)
    before = json.dumps(graph.to_dict(), sort_keys=True)
    p = backend.prepare_harmonic_beads(graph, **options)
    identity = p.identity
    with patch.object(backend.RuntimeSpec, "checked", return_value=RUNTIME):
        task = backend.LAMMPSTask(p, RUNTIME, tmp_path / "not-started")
    assert task.state == backend.TaskState.CREATED
    assert task.output_forecast is None
    assert task.prepared is p
    assert p.identity == identity
    assert not task.workdir.exists()
    assert json.dumps(graph.to_dict(), sort_keys=True) == before


def test_floor_is_loose_and_admission_does_not_promise_fit(tmp_path):
    p = prepared(run=backend.RunSpec(1e-6, 24, 1, ((-0.01, 0.01),) * 3))
    dump, log = fixture(p)
    atom_rows = [
        line
        for line in dump.splitlines()
        if len(line.split()) == 11 and not line.startswith("ITEM:")
    ]
    assert len(atom_rows) == 2 * 25
    assert all(len(row.encode("utf-8")) + 1 >= 22 for row in atom_rows)
    assert len(dump.encode("utf-8")) > 1100
    assert len((dump + log).encode("utf-8")) > 1100
    limits = backend.ResourceLimits(output_bytes=1100, file_size_bytes=1100)
    with patch.object(backend.RuntimeSpec, "checked", return_value=RUNTIME):
        task = backend.LAMMPSTask(p, RUNTIME, tmp_path / "not-started", limits)
    assert task.output_forecast is None


def test_public_dispatch_rejects_before_native_runtime_validation(tmp_path):
    import fibernet

    graph, options = chain()
    with patch.object(backend.RuntimeSpec, "checked") as checked:
        with pytest.raises(backend.LAMMPSContractError, match="563205632 bytes"):
            fibernet.simulate(
                graph,
                mode="dynamics",
                backend="lammps",
                runtime=RUNTIME,
                workdir=tmp_path / "not-started",
                limits=backend.ResourceLimits(),
                **options,
            )
    checked.assert_not_called()
    assert not (tmp_path / "not-started").exists()


def test_tampered_schedule_cannot_lower_the_bound(tmp_path):
    graph, options = chain()
    p = backend.prepare_harmonic_beads(graph, **options)
    with patch.object(backend.RuntimeSpec, "checked") as checked:
        for altered in (replace(p, steps=0), replace(p, sample_every=100000)):
            with pytest.raises(
                backend.LAMMPSContractError, match="deterministic compiler"
            ):
                backend.LAMMPSTask(altered, RUNTIME, tmp_path / "not-started")
    checked.assert_not_called()
    assert not (tmp_path / "not-started").exists()

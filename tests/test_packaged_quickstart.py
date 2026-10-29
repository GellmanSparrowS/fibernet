"""Exercise the installed example entry point and the tutorial code blocks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np

import fibernet as fn


def test_packaged_example_from_another_directory(tmp_path):
    output = tmp_path / "example"
    env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(tmp_path / "mpl"))
    completed = subprocess.run(
        [sys.executable, "-m", "fibernet.examples.basic_usage",
         "--output-dir", str(output)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    graph = fn.StructureGraph.load_json(str(output / "network.json"))
    result = fn.SimResult.load(str(output / "fem_result.json"))
    assert (graph.num_nodes, graph.num_edges) == (18, 22)
    assert summary["nodes"] == result.n_nodes == graph.num_nodes
    assert summary["physical_edges"] == result.n_edges == graph.num_edges
    assert summary["connected_components"] == 1
    assert summary["max_displacement_m"] > 0
    assert summary["max_member_axial_force_N"] > 0
    assert np.isfinite(result.displacements).all()
    assert set(result.metadata["node_ids"]) == set(graph.nodes)
    assert set(result.metadata["edge_ids"]) == set(graph.edges)
    assert (output / "network.png").stat().st_size > 1000


def test_tutorial_python_blocks(tmp_path, monkeypatch):
    document = Path(__file__).resolve().parents[1] / "docs" / "tutorials.rst"
    lines = document.read_text(encoding="utf-8").splitlines()
    blocks = []
    for index, line in enumerate(lines):
        if line.strip() == ".. code-block:: python":
            body = []
            for following in lines[index + 1:]:
                if following and not following.startswith(" "):
                    break
                body.append(following)
            blocks.append(textwrap.dedent("\n".join(body)))
    assert len(blocks) == 3
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for index, block in enumerate(blocks):
        exec(compile(block, "tutorials.rst:block%d" % index, "exec"), namespace)
    assert namespace["topology"].n_components == 1
    assert namespace["features"].shape == (14,)
    assert np.isfinite(namespace["features"]).all()
    assert (tmp_path / "fem_result.json").is_file()

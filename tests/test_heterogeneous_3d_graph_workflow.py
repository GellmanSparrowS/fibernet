"""Check the public spatial graph example and its file handoff."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import fibernet as fn
import numpy as np

PATH = Path(__file__).parents[1] / "examples/heterogeneous_3d_graph_workflow.py"
main = runpy.run_path(str(PATH))["main"]


class HeterogeneousGraphWorkflowTest(unittest.TestCase):
    def test_cli_saved_materials_and_displacements(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "mixed"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                main(["--output", str(output)])
            report = json.loads(stdout.getvalue())
            graph = fn.StructureGraph.load_json(output / "graph.json")
            result = fn.SimResult.load(output / "result.json")
            self.assertEqual((len(graph.nodes), len(graph.edges)), (12, 20))
            np.testing.assert_allclose(report["axial_cut_force_N"], 0.128679635091, rtol=1e-7)
            np.testing.assert_allclose(report["beam_strain_energy_J"], 2.57359270182e-7, rtol=1e-9)
            meta = result.metadata["material_contract"]
            self.assertEqual((meta["source"], meta["unit_system"]), ("graph", "SI"))
            self.assertEqual(meta["edge_ids"], result.metadata["edge_ids"])
            self.assertEqual(set(meta["edge_ids"]), set(graph.edges))
            self.assertEqual(dict(zip(meta["edge_ids"], meta["youngs_modulus"])), {
                eid: edge.material.youngs_modulus for eid, edge in graph.edges.items()
            })
            self.assertEqual(meta["youngs_modulus"].count(1e9), 4)

            middle_ids = {nid for nid, node in graph.nodes.items() if node.position[0] == 0.002}
            self.assertEqual(len(middle_ids), 4)
            deformed = dict(zip(result.metadata["node_ids"], result.deformed_positions))
            for nid in middle_ids:
                np.testing.assert_allclose(
                    deformed[nid] - graph.nodes[nid].position, [3.2e-6, 0.0, 0.0],
                    rtol=0, atol=5e-13,
                )

    def test_cli_rejects_existing_directory_and_missing_parent_before_solver(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(fn, "simulate") as solve:
            output = Path(tmp) / "mixed"
            output.mkdir()
            old = output / "graph.json"
            old.write_bytes(b"previous output is unchanged")
            with self.assertRaisesRegex(SystemExit, "already exists"):
                main(["--output", str(output)])
            self.assertEqual(list(output.iterdir()), [old])
            self.assertEqual(old.read_bytes(), b"previous output is unchanged")

            absent = Path(tmp) / "not-created"
            with self.assertRaisesRegex(SystemExit, "parent must exist"):
                main(["--output", str(absent / "mixed")])
            self.assertFalse(absent.exists())
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as usage:
                main([])
            self.assertEqual(usage.exception.code, 2)
            solve.assert_not_called()

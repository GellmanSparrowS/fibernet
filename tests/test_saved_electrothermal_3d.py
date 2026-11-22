"""Self-contained admission and exact-frame tests for the optional saved 3D bridge."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from fibernet import Material, StructureGraph
from fibernet.sim import electrothermal, saved_electrothermal, thermoelastic_3d
from fibernet.sim import saved_electrothermal_3d as bridge


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class SavedElectrothermal3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate once from a small noncoplanar welded star. Stable IDs are
        # deliberately noncontiguous so they cannot be mistaken for row numbers.
        ids = (10, 37, 59, 83)
        graph = StructureGraph.from_dict({
            "dimension": 3,
            "nodes": [{"id": nid, "position": xyz} for nid, xyz in zip(
                ids, ([0., 0., 0.], [-.012, 0., 0.], [0., -.01, 0.], [0., 0., -.015]))],
            "edges": [
                {"id": eid, "node_i": nid, "node_j": 10, "radius": .001,
                 "material": Material(youngs_modulus=E, poissons_ratio=.3).to_dict()}
                for eid, nid, E in ((101, 37, 1e9), (203, 59, 2e9), (307, 83, 3e9))
            ],
        })
        edge_ids = tuple(graph.edges)
        weights = {e: {graph.edges[e].node_i: .5, graph.edges[e].node_j: .5} for e in edge_ids}
        law = electrothermal.TemperatureConductanceLaw(
            {101: .3, 203: .45, 307: .5}, dict.fromkeys(edge_ids, .01),
            dict.fromkeys(edge_ids, 300.), weights, (296., 320.),
        )
        result = electrothermal.solve_electrothermal(
            graph, law, prescribed_voltage_V={37: 0., 59: 1., 83: .25},
            capacity_J_per_K=dict.fromkeys(ids, 1.0),
            conductance_W_per_K=dict.fromkeys(edge_ids, .1),
            initial_temperature_K={10: 302., 37: 301., 59: 303., 83: 300.},
            times_s=(0., .125, .25), endpoint_heat_weights=weights, length_unit="m",
        )
        if not result.completed or len(result.frames) != 3:
            raise AssertionError("synthetic source must have exactly three accepted frames")
        cls.graph_record = graph.to_dict()
        cls.archive_bytes = _canonical(saved_electrothermal.export_saved_electrothermal(
            graph, result, length_unit="m"))
        cls.byte_pin = hashlib.sha256(cls.archive_bytes).hexdigest()
        cls.controls = {
            "length_unit": "m",
            "stress_free_temperature_K": 300.,
            # Signed mechanical expansion is independent of the positive
            # electrical law coefficients even though both have units 1/K.
            "expansion_alpha_per_K_by_edge_id": {101: -1e-5, 203: 0., 307: 2e-5},
            "constraints_by_node_id": {nid: (0.,) * 6 for nid in ids if nid != 10},
            "applicability_limits": dict.fromkeys((
                "endpoint_thermal_strain", "nodal_rotation", "transverse_slope_bound",
                "total_extreme_fiber_strain", "endpoint_elastic_extreme_fiber_strain",
                "twist_shear"), 1e-3),
        }

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.graph = StructureGraph.from_dict(self.graph_record)
        self.path = Path(temp.name) / "complete.json"
        self.path.write_bytes(self.archive_bytes)

    def solve(self, graph=None, **kwargs):
        args = dict(self.controls, frame_index=1, expected_file_sha256=self.byte_pin)
        args.update(kwargs)
        return bridge.solve_saved_electrothermal_frame_3d(
            self.graph if graph is None else graph, self.path, **args)

    def test_preopen_byte_bound_and_bad_pin_predecode(self):
        # Close before opening; NamedTemporaryFile reopen isn't portable to Windows.
        with self.path.open("r+b") as stream:
            stream.truncate(bridge._MAX_ARCHIVE_BYTES + 1)
        with patch.object(bridge.os, "open", side_effect=AssertionError("opened oversized archive")), \
             patch.object(saved_electrothermal, "_decode", side_effect=AssertionError("decoded oversized archive")):
            with self.assertRaisesRegex(ValueError, "4 MiB"):
                self.solve()
        self.path.write_bytes(self.archive_bytes)
        with patch.object(saved_electrothermal, "_decode", side_effect=AssertionError("decoded wrong exact bytes")):
            with self.assertRaisesRegex(ValueError, "expected SHA256"):
                self.solve(expected_file_sha256="0" * 64)

    def test_material_only_mutation_and_rehashed_unselected_frame_reject(self):
        changed = self.graph.to_dict()
        changed["edges"][0]["material"]["youngs_modulus"] *= 1.1
        changed_graph = StructureGraph.from_dict(changed)
        with patch.object(thermoelastic_3d, "solve_thermoelastic_network_snapshot_3d",
                          side_effect=AssertionError("invalid archive reached mechanics")):
            with self.assertRaisesRegex(ValueError, "differs from captured graph"):
                self.solve(graph=changed_graph)
            changed = json.loads(self.archive_bytes)
            # Keep frame 1 untouched and regenerate the payload checksum. That
            # satisfies a checksum but not the all-frame consistency contract.
            changed["source"]["frames"][2]["temperature_K"][0] += .25
            del changed["content_sha256"]
            changed["content_sha256"] = hashlib.sha256(_canonical(changed)).hexdigest()
            self.path.write_bytes(_canonical(changed))
            with self.assertRaisesRegex(ValueError, "conductance differs"):
                self.solve(expected_file_sha256=None)

    def test_exact_middle_frame_matches_direct_owned_output_and_old_guard_is_retained(self):
        data = json.loads(self.archive_bytes)
        source = data["source"]
        frame = source["frames"][1]
        expected = dict(zip(source["node_ids"], frame["temperature_K"]))
        solve_direct = thermoelastic_3d.solve_thermoelastic_network_snapshot_3d
        direct = solve_direct(self.graph, temperature_K_by_node_id=expected, **self.controls)
        with patch.object(electrothermal, "solve_electrothermal",
                          side_effect=AssertionError("source was solved again")), \
             patch.object(thermoelastic_3d, "solve_thermoelastic_network_snapshot_3d", wraps=solve_direct) as owner:
            result = self.solve()
        self.assertEqual(owner.call_count, 1)
        self.assertIsNot(owner.call_args.args[0], self.graph)
        self.assertEqual(owner.call_args.kwargs["expansion_alpha_per_K_by_edge_id"],
                         self.controls["expansion_alpha_per_K_by_edge_id"])
        self.assertEqual(result.snapshot, direct)
        self.assertEqual(result.receipt.frame_index, 1)
        self.assertEqual(result.receipt.frame_time_s, .125)
        self.assertEqual({nid: struct.pack(">d", t)
                          for nid, t in result.receipt.temperatures_K_by_node_id},
                         {nid: struct.pack(">d", t) for nid, t in expected.items()})
        self.assertEqual(result.receipt.file_sha256, self.byte_pin)
        self.assertTrue(result.receipt.caller_byte_pin_checked)
        self.assertEqual(result.receipt.content_sha256, data["content_sha256"])
        self.assertEqual(result.receipt.graph_sha256, data["graph_sha256"])
        self.assertEqual(result.receipt.reported_native_input_sha256, source["input_sha256"])
        self.assertEqual(result.receipt.selected_electrical_input_sha256,
                         frame["electrical"]["provenance"]["input_sha256"])
        with self.assertRaises(AttributeError):
            result.receipt.frame_index = 2
        with self.assertRaisesRegex(ValueError, "requires dimension=2"):
            saved_electrothermal.prepare_saved_electrothermal_temperature(
                self.graph, data, length_unit="m", frame_index=1)


if __name__ == "__main__":
    unittest.main()

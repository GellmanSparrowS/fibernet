"""Synthetic-only R10 interchange tests; no native solver or benchmark data."""
import copy
import os
import sys
import unittest

_OPTIONAL = {"gmsh", "shapely", "numba", "llvmlite", "psutil", "tkinter",
             "fibernet_continuum"}
_IMPORT_ENVIRONMENT = dict(os.environ)
_PRELOADED_OPTIONAL = _OPTIONAL.intersection(sys.modules)


class _DenyOptionalImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _OPTIONAL:
            raise AssertionError("R10 adapter eagerly imported optional backend: " + fullname)
        return None


_guard = _DenyOptionalImports()
sys.meta_path.insert(0, _guard)
try:
    from fibernet.core.material import Material
    from fibernet.io.r10_geometry import (
        SCHEMA, import_r10_geometry, export_r10_geometry, validate_r10_geometry,
    )
finally:
    sys.meta_path.remove(_guard)
_AFTER_IMPORT_ENVIRONMENT = dict(os.environ)
_AFTER_OPTIONAL = _OPTIONAL.intersection(sys.modules)


def fixture():
    # Sparse IDs, parallel source fibres, and coincident distinct nodes.
    source = dict(schema=SCHEMA, case_id="synthetic",
                  nodes_xy_mm=[[0., 0.], [1., 1.], [0., 0.]],
                  node_ids=[40, 2, 91], edges=[[0, 1], [0, 1], [2, 1]],
                  edge_ids=[88, 5, 101], parent_fiber_ids=[[1], [2], [3]],
                  source_edge_ids=[[7], [8], [9]],
                  source_named_path_ids=[["a"], ["b"], ["c"]],
                  outline=dict(exterior=[[-1., -1.], [2., -1.], [2., 2.],
                                         [-1., 2.], [-1., -1.]], holes=[]),
                  grips=dict(bottom=dict(segments_xy_mm=[[[-1., -1.], [2., -1.]]]),
                             top=dict(segments_xy_mm=[[[-1., 2.], [2., 2.]]])),
                  material=dict(C10_MPa=6., C01_MPa=3., D1_per_MPa=.495),
                  load_axis="Y", reference_thickness_mm=1., nominal_gauge_mm=3.,
                  self_contact=False, custom_extension={"provenance": ["synthetic"]})
    props = {eid: dict(radius=.2 + i * .01, segments=2 + i,
                       material=Material(name="explicit-" + str(eid),
                                         youngs_modulus=1e6 + i * 1e5,
                                         extra={"annotation_only": True}))
             for i, eid in enumerate(source["edge_ids"])}
    return source, props


class R10GeometryTests(unittest.TestCase):
    def test_import_does_not_load_optional_backends_or_mutate_environment(self):
        # Fresh isolated interpreter required to establish the stronger claim.
        self.assertFalse(_PRELOADED_OPTIONAL, "run in fresh isolated interpreter")
        self.assertFalse(_AFTER_OPTIONAL)
        self.assertEqual(_AFTER_IMPORT_ENVIRONMENT, _IMPORT_ENVIRONMENT)

    def test_round_trip_identity_and_annotations(self):
        source, props = fixture()
        graph, binding = import_r10_geometry(source, edge_properties=props, length_unit="mm")
        self.assertEqual(set(graph.nodes), {40, 2, 91})
        self.assertEqual(set(graph.edges), {88, 5, 101})
        self.assertEqual(graph.num_nodes, 3)
        self.assertEqual(graph.num_edges, 3)
        self.assertEqual((graph.edges[88].node_i, graph.edges[88].node_j), (40, 2))
        self.assertEqual((graph.edges[5].node_i, graph.edges[5].node_j), (40, 2))
        self.assertEqual(graph.edges[101].material.name, "explicit-101")
        self.assertEqual(export_r10_geometry(graph, binding, length_unit="mm"), source)
        # Full graph reimport requires retained annotations, never default Material.
        retained = {int(k): dict(v, material=Material.from_dict(v["material"]))
                    for k, v in binding["edge_properties"].items()}
        again, _ = import_r10_geometry(source, edge_properties=retained, length_unit="mm")
        self.assertEqual(graph.to_dict(), again.to_dict())
        self.assertEqual(binding["status"], "geometry_only_not_solver_ready")

    def test_detached_inputs_and_outputs(self):
        source, props = fixture()
        before = copy.deepcopy(source)
        graph, binding = import_r10_geometry(source, edge_properties=props, length_unit="mm")
        source["nodes_xy_mm"][0][0] = 100
        props[88]["material"].name = "changed"
        self.assertEqual(graph.nodes[40].position[0], 0)
        self.assertEqual(graph.edges[88].material.name, "explicit-88")
        out = export_r10_geometry(graph, binding, length_unit="mm")
        self.assertEqual(out, before)
        out["custom_extension"]["provenance"].append("changed")
        self.assertEqual(binding["source"]["custom_extension"]["provenance"], ["synthetic"])

    def test_invalid_source_inputs(self):
        modifications = [
            lambda s: s.update(schema="unknown"),
            lambda s: s.update(node_ids=[40, 40, 91]),
            lambda s: s.update(edge_ids=[88, True, 101]),
            lambda s: s.update(node_ids=[40, 2 ** 63 - 1, 91]),
            lambda s: s.update(edge_ids=[88, 10 ** 400, 101]),
            lambda s: s["edges"].__setitem__(0, [40, 2]),
            lambda s: s["edges"].__setitem__(0, [-1, 1]),
            lambda s: s["edges"].__setitem__(0, [True, 1]),
            lambda s: s["edges"].__setitem__(0, [0, 2]),
            lambda s: s["nodes_xy_mm"].__setitem__(0, [0, 0, 1]),
            lambda s: s["nodes_xy_mm"].__setitem__(0, [float("nan"), 0]),
            lambda s: s["nodes_xy_mm"].__setitem__(0, [False, 0]),
            lambda s: s["nodes_xy_mm"].__setitem__(0, [10 ** 400, 0]),
            lambda s: s.update(parent_fiber_ids=[]),
            lambda s: s.update(material=dict(C10_MPa=6, C01_MPa=3, D1_per_MPa=0)),
            lambda s: s.update(reference_thickness_mm=True),
            lambda s: s.update(self_contact=0),
            lambda s: s.update(nominal_gauge_mm=0),
            lambda s: s.pop("outline"),
            lambda s: s["outline"].update(exterior=[[0, 0], [1, 0], [1, 1], [2, 2]]),
            lambda s: s["grips"].pop("top"),
            lambda s: s.update(custom_extension={"nan": float("inf")}),
        ]
        for modify in modifications:
            with self.subTest(modification=modifications.index(modify)):
                source, _ = fixture()
                modify(source)
                with self.assertRaises(ValueError):
                    validate_r10_geometry(source, length_unit="mm")

    def test_explicit_units_and_beam_properties(self):
        source, props = fixture()
        for unit in (None, "m", "um", "MPa"):
            with self.subTest(unit=unit), self.assertRaises(ValueError):
                import_r10_geometry(source, edge_properties=props, length_unit=unit)
        with self.assertRaises(TypeError):
            import_r10_geometry(source, edge_properties=props)
        for bad in ({}, {88: {}}, {**props, True: props[88]}):
            with self.assertRaises(ValueError):
                import_r10_geometry(source, edge_properties=bad, length_unit="mm")
        for field, bad in (("radius", -1), ("radius", True), ("segments", 1.5),
                           ("segments", 0), ("material", {})):
            source, props = fixture()
            props[88][field] = bad
            with self.assertRaises(ValueError):
                import_r10_geometry(source, edge_properties=props, length_unit="mm")

    def test_changed_graph_and_domain_rejected(self):
        changes = [lambda g: g.set_node_position(40, [.1, 0]),
                   lambda g: setattr(g.edges[88], "radius", .7),
                   lambda g: setattr(g.edges[88].material, "name", "changed"),
                   lambda g: setattr(g.edges[88], "segments", 19),
                   lambda g: g.remove_edge(88),
                   lambda g: g.nodes[40].metadata.update(hidden="change"),
                   lambda g: g.nodes[40].position.__setitem__(2, 1),
                   lambda g: setattr(g.edges[88], "internal_points", [[.2, .2, 0]])]
        for change in changes:
            source, props = fixture()
            graph, binding = import_r10_geometry(source, edge_properties=props, length_unit="mm")
            change(graph)
            with self.assertRaises(ValueError):
                export_r10_geometry(graph, binding, length_unit="mm")
        for section in ("source", "edge_properties"):
            source, props = fixture()
            graph, binding = import_r10_geometry(source, edge_properties=props, length_unit="mm")
            if section == "source":
                binding[section]["outline"]["exterior"][0][0] = -.5
            else:
                binding[section]["88"]["radius"] = 4.
            with self.assertRaises(ValueError):
                export_r10_geometry(graph, binding, length_unit="mm")

    def test_holes_extensions_and_environment_unchanged(self):
        source, props = fixture()
        source["outline"]["holes"] = [[[.1, .1], [.2, .1], [.2, .2], [.1, .1]]]
        env = dict(os.environ)
        graph, binding = import_r10_geometry(source, edge_properties=props, length_unit="mm")
        self.assertEqual(export_r10_geometry(graph, binding, length_unit="mm"), source)
        self.assertEqual(dict(os.environ), env)

    def test_malformed_binding_has_explicit_validation_error(self):
        for bad in (None, [], {"88": {}}, {"88": {"radius": 1, "segments": 1,
                                                   "material": []}}):
            source, props = fixture()
            graph, binding = import_r10_geometry(source, edge_properties=props, length_unit="mm")
            binding["edge_properties"] = bad
            with self.assertRaises(ValueError):
                export_r10_geometry(graph, binding, length_unit="mm")


if __name__ == "__main__":
    unittest.main()

"""Provider-free checks of the *production* SimResult JSON save/load path.

The class fields, to_dict, atomic save, load, and mechanical-envelope validator
are AST-extracted, not reimplemented. Known profile/material imports are replaced
with explicit inert validation collaborators; nothing is added to sys.modules.
Array conversion uses a small owned-list double, not NumPy. These tests prove
metadata transport and ordinary file I/O only: not material validation, numeric
equivalence, a provider solve, binary64 ledger behavior, or geometry authenticity.
The historical material fixture is read only and is never rewritten in place.
"""

import ast
import copy
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
import warnings

from .support import ROOT, isolated_builtins


def _shape(value):
    if not isinstance(value, list):
        return ()
    shapes = [_shape(item) for item in value]
    if shapes and any(shape != shapes[0] for shape in shapes):
        raise ValueError("ragged test array")
    return (len(value),) + (shapes[0] if shapes else ())


class _Array:
    """Owned JSON-list array, deliberately without numerical operations."""

    def __init__(self, value):
        self.values = copy.deepcopy(value.tolist() if isinstance(value, _Array) else value)
        self.shape = _shape(self.values)
        self.ndim = len(self.shape)

    def tolist(self):
        return copy.deepcopy(self.values)

    def __len__(self):
        return len(self.values)


class _Scalar:
    """An uninstantiated np.generic stand-in for save's default converter."""


def _finite_array(value, name, shape=None):
    """Minimal JSON fixture validation, not the production NumPy validator."""
    value = value.tolist() if isinstance(value, _Array) else copy.deepcopy(value)

    def finite(item):
        if isinstance(item, list):
            for child in item:
                finite(child)
        elif type(item) not in (int, float) or not math.isfinite(item):
            raise ValueError(name + " must contain finite numbers")

    finite(value)
    array = _Array(value)
    if shape is not None and array.shape != shape:
        raise ValueError(name + " has wrong shape")
    return array


class _InertImports(ast.NodeTransformer):
    """Replace only named dependency imports, preserving method control flow."""

    replacements = {
        (0, "fibernet.integrations", "_harmonic_profile"): "_profile",
        (0, "fibernet.integrations", "_cg_network_profile"): "_network_profile",
        (1, "_taichi_viscoelastic", "validate_result"): "_material_validate",
        (1, "_taichi_viscoelastic", "populate_result"): "_material_populate",
    }

    def __init__(self):
        self.replaced = set()

    def visit_ImportFrom(self, node):
        keys = [(node.level, node.module, alias.name) for alias in node.names]
        if not any(key in self.replacements for key in keys):
            return node
        if not all(key in self.replacements for key in keys):
            raise AssertionError("Unreviewed import mixed with a validation seam")
        self.replaced.update(keys)
        return [ast.copy_location(ast.Assign(
            targets=[ast.Name(id=alias.asname or alias.name, ctx=ast.Store())],
            value=ast.Name(id=self.replacements[key], ctx=ast.Load())), node)
                for alias, key in zip(node.names, keys)]


def _result_class():
    """Keep original class field defaults and the three production methods."""
    path = ROOT / "fibernet/sim/accelerated.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    original = next(node for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == "SimResult")
    selected = copy.deepcopy(original)
    selected.body = [node for node in selected.body
                     if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     or node.name in {"to_dict", "save", "load"}]
    mechanical = copy.deepcopy(next(node for node in tree.body
                                    if isinstance(node, ast.FunctionDef)
                                    and node.name == "_validate_mechanical_json"))
    imports = _InertImports()
    selected = imports.visit(selected)
    if imports.replaced != set(imports.replacements):
        raise AssertionError("Production dependency seams changed; review this harness")
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        mechanical, selected,
    ], type_ignores=[])
    calls = []

    def select(profile):
        calls.append(("profile.select", profile))
        if profile is not None:
            raise AssertionError("This harness only tests non-profile mechanical results")
        return None

    def material_validate(result):
        calls.append(("material.validate", copy.deepcopy(result.metadata["native_axial_viscoelastic"])))

    def material_populate(result, record):
        # Real derived numeric material population is intentionally not run.
        calls.append(("material.populate", copy.deepcopy(record)))

    def mass_inputs(record, n_nodes, node_ids):
        calls.append(("mass.validate", n_nodes, copy.deepcopy(node_ids)))
        return copy.deepcopy(record)

    def spring_provenance(record, n_edges, edge_ids, **options):
        calls.append(("spring.validate", n_edges, copy.deepcopy(edge_ids), dict(options)))
        return copy.deepcopy(record)

    namespace = dict(
        __name__=__name__, dataclass=dataclass, field=field,
        __builtins__=isolated_builtins(),
        np=SimpleNamespace(ndarray=_Array, generic=_Scalar, array=_Array,
                           asarray=_Array, isfinite=math.isfinite),
        json=json, os=os, stat=stat, tempfile=tempfile, warnings=warnings, Path=Path,
        _finite_real_array=_finite_array,
        _SPRING_PROVENANCE="spring_coefficient_provenance",
        _validate_spring_provenance=spring_provenance,
        _validate_mass_inputs=mass_inputs,
        _profile=SimpleNamespace(select=select),
        _network_profile=SimpleNamespace(NAMES=()),
        _material_validate=material_validate, _material_populate=material_populate,
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["SimResult"], calls


def _geometry(policy="endpoint_chord", ignored=(91, 7)):
    # A JSON-native transport fixture, not a claim that a real solve occurred.
    return dict(schema="taichi_endpoint_chord/1", policy=policy,
                representation="endpoint_chord",
                ignored_internal_point_edge_ids=list(ignored))


class ResultGeometryRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.Result, self.calls = _result_class()
        self.temp = tempfile.TemporaryDirectory(prefix="geometry-result-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def result(self, geometry=True):
        metadata = dict(node_ids=[83, 17], edge_ids=[91, 7], stiffness=4.0,
                        damping=0.3, dt=0.01, num_steps=2,
                        spring_coefficient_provenance={
                            "schema": "spring_coefficient_provenance/1",
                            "law": "geometric_rest_axial/1", "quantity_kind": "force",
                            "simulation": {"source": "stiffness", "representation": "scalar",
                                           "edge_count": 2, "availability": "available", "value": 4.0},
                            "current_metrics": {"source": "simulation"},
                        })
        if geometry:
            metadata["taichi_geometry"] = _geometry()
        return self.Result(
            mode="dynamics", energy=1.25, time_seconds=0.1, n_nodes=2, n_edges=2,
            max_force=0.5, max_stretch=1.2, mean_stretch=1.1, std_stretch=0.1,
            max_displacement=0.25, metadata=metadata,
            displacements=_Array([[0.0, 0.0], [0.25, 0.0]]),
            deformed_positions=_Array([[0.0, 0.0], [1.25, 0.0]]),
            edge_forces=_Array([0.5, 0.25]), edge_stretches=_Array([1.2, 1.0]),
            positions_trajectory=[_Array([[0.0, 0.0], [1.0, 0.0]]),
                                  _Array([[0.0, 0.0], [1.25, 0.0]])],
            history=[{"step": 2, "max_stretch": 1.2}],
        )

    def round_trip(self, result, detailed):
        path = self.directory / ("detailed.json" if detailed else "summary.json")
        before = copy.deepcopy(result.metadata)
        result.save(path, detailed=detailed)
        self.assertTrue(path.is_file())
        # Inspect the file actually committed by production save, then use load.
        payload = json.loads(path.read_text(encoding="utf-8"))
        restored = self.Result.load(path)
        self.assertEqual(result.metadata, before)
        self.assertEqual(restored.to_dict(detailed=detailed), payload)
        self.assertEqual(list(self.directory.glob(".*.result-*")), [])
        return restored, payload

    def test_summary_and_detailed_keep_geometry_identity_and_spring_metadata(self):
        for detailed in (False, True):
            for policy, ignored in (("reject_internal_points", ()), ("endpoint_chord", (91, 7))):
                with self.subTest(detailed=detailed, policy=policy):
                    result = self.result()
                    result.metadata["taichi_geometry"] = _geometry(policy, ignored)
                    restored, payload = self.round_trip(result, detailed)
                    self.assertEqual(restored.metadata, result.metadata)
                    self.assertEqual(payload["taichi_geometry"], result.metadata["taichi_geometry"])
                    self.assertEqual(restored.energy, 1.25)
                    self.assertEqual(restored.metadata["node_ids"], [83, 17])
                    self.assertEqual(restored.metadata["edge_ids"], [91, 7])
                    self.assertNotIn("metadata", payload)  # Existing flat export format.
                    for name in ("edge_forces", "edge_stretches", "displacements", "deformed_positions"):
                        if detailed:
                            self.assertEqual(getattr(restored, name).tolist(), getattr(result, name).tolist())
                        else:
                            self.assertIsNone(getattr(restored, name))
                            self.assertNotIn(name, payload)
                    self.assertEqual(len(restored.positions_trajectory), 2 if detailed else 0)
                    self.assertEqual(restored.history, result.history if detailed else [])
                    # The loaded nested record owns the JSON decode, not the input.
                    restored.metadata["taichi_geometry"]["ignored_internal_point_edge_ids"].append(999)
                    self.assertEqual(result.metadata["taichi_geometry"], _geometry(policy, ignored))
        self.assertTrue(any(call[0] == "spring.validate" for call in self.calls))

    def test_legacy_summary_and_detailed_load_without_geometry_fabrication(self):
        for detailed in (False, True):
            with self.subTest(detailed=detailed):
                result = self.result(geometry=False)
                restored, payload = self.round_trip(result, detailed)
                self.assertNotIn("taichi_geometry", payload)
                self.assertNotIn("taichi_geometry", restored.metadata)
                self.assertIsNone(restored.dynamics_state)
                self.assertIsNone(restored.algorithmic_energy_ledger)
                self.assertEqual(restored.metadata, result.metadata)
                self.assertEqual(restored.energy, result.energy)

    def test_existing_material_fixture_load_stays_legacy_and_read_only(self):
        fixture = ROOT / "tests/fixtures/dynamic_viscoelastic_generalized_summary.json"
        original = fixture.read_bytes()
        payload = json.loads(original)
        result = self.Result.load(fixture)
        self.assertNotIn("taichi_geometry", result.metadata)
        for detailed in (False, True):
            with self.subTest(detailed=detailed):
                restored, encoded = self.round_trip(result, detailed)
                self.assertNotIn("taichi_geometry", encoded)
                for key in ("node_ids", "edge_ids", "native_axial_viscoelastic", "native_axial_inputs"):
                    self.assertEqual(restored.metadata[key], payload[key])
                self.assertEqual(restored.energy, payload["energy"])
        self.assertEqual(fixture.read_bytes(), original)
        self.assertTrue(any(call[0] == "material.validate" for call in self.calls))
        self.assertTrue(any(call[0] == "material.populate" for call in self.calls))

    def test_new_geometry_record_coexists_with_material_and_its_energy_history(self):
        fixture = ROOT / "tests/fixtures/dynamic_viscoelastic_generalized_summary.json"
        template = json.loads(fixture.read_text(encoding="utf-8"))
        # Construct a synthetic current result. Do not retrofit the legacy file.
        known = set(self.Result.__dataclass_fields__) - {"metadata"}
        metadata = {key: copy.deepcopy(value) for key, value in template.items() if key not in known}
        metadata["taichi_geometry"] = _geometry("reject_internal_points", ())
        result = self.Result(**{key: value for key, value in template.items() if key in known}, metadata=metadata)
        for detailed in (False, True):
            with self.subTest(detailed=detailed):
                restored, payload = self.round_trip(result, detailed)
                self.assertEqual(restored.metadata, metadata)
                self.assertEqual(payload["native_axial_viscoelastic"], template["native_axial_viscoelastic"])
                self.assertEqual(restored.energy, template["energy"])
                self.assertEqual(payload["taichi_geometry"], _geometry("reject_internal_points", ()))

    def test_geometry_metadata_remains_opaque_not_authenticated_by_loader(self):
        result = self.result()
        opaque = {"schema": "unverified_fixture/99", "opaque": [None, True, "not validated"]}
        result.metadata["taichi_geometry"] = opaque
        restored, payload = self.round_trip(result, detailed=False)
        self.assertEqual(payload["taichi_geometry"], opaque)
        self.assertEqual(restored.metadata["taichi_geometry"], opaque)


if __name__ == "__main__":
    unittest.main()

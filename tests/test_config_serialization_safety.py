"""Standard-library tests for config serialization failure safety.

The controlled YAML provider tests the call boundary, not PyYAML formatting.
"""
import builtins
import copy
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "fibernet" / "utils" / "config.py"
STAMP = "2020-01-02T03:04:05"


class FixedClock:
    @staticmethod
    def now():
        return datetime.fromisoformat(STAMP)


class ConfigSerializationSafetyTests(unittest.TestCase):
    def load_config(self, provider=None):
        name = "_fibernet_config_serialization_safety_regression"
        spec = importlib.util.spec_from_file_location(name, SOURCE)
        module = importlib.util.module_from_spec(spec)
        # Load the real shared parser helper without importing the package facade.
        helper_name = "fibernet._json"
        helper_spec = importlib.util.spec_from_file_location(helper_name, SOURCE.parents[1] / "_json.py")
        helper = importlib.util.module_from_spec(helper_spec)
        original_import = builtins.__import__

        def controlled_import(requested, globals=None, locals=None, fromlist=(), level=0):
            if requested == "yaml" and level == 0:
                if provider is None:
                    raise ModuleNotFoundError("No module named 'yaml'", name="yaml")
                return provider
            return original_import(requested, globals, locals, fromlist, level)

        with patch.dict(sys.modules, {name: module, helper_name: helper}), \
                patch.object(sys, "dont_write_bytecode", True), \
                patch("builtins.__import__", new=controlled_import):
            helper_spec.loader.exec_module(helper)
            spec.loader.exec_module(module)
        return module

    def test_json_serialization_failure_preserves_existing_file(self):
        module = self.load_config()
        config = module.ExperimentConfig(network={"unsupported": {1}})
        original = copy.deepcopy(config.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.json"
            target.write_bytes(b"original config bytes\n")
            with self.assertRaises(TypeError):
                config.to_json(target)
            self.assertEqual(target.read_bytes(), b"original config bytes\n")
            self.assertEqual(list(Path(directory).iterdir()), [target])
        self.assertEqual(config.to_dict(), original)

    def test_json_serialization_failure_leaves_new_destination_absent(self):
        module = self.load_config()
        config = module.ExperimentConfig(network={"unsupported": {1}})
        original = copy.deepcopy(config.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new-parent" / "config.json"
            with self.assertRaises(TypeError):
                config.to_json(target, include_metadata=False)
            self.assertFalse(target.exists())
            self.assertFalse(target.parent.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(config.to_dict(), original)

    def test_successful_json_roundtrip_and_metadata_flags(self):
        module = self.load_config()
        config = module.ExperimentConfig(experiment={"name": "example"}, network={"n": 3})
        original = copy.deepcopy(config.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            for include in (True, False):
                with self.subTest(include_metadata=include):
                    target = Path(directory) / str(include) / "config.json"
                    expected = copy.deepcopy(original)
                    if include:
                        expected["metadata"]["saved_at"] = STAMP
                    else:
                        expected.pop("metadata")
                    with patch.object(module, "datetime", FixedClock):
                        self.assertIsNone(config.to_json(target, include_metadata=include))
                    self.assertEqual(target.read_text(), json.dumps(expected, indent=2))
                    self.assertEqual(json.loads(target.read_text()), expected)
                    restored = module.ExperimentConfig.from_json(target)
                    self.assertEqual(restored.experiment, original["experiment"])
                    self.assertEqual(restored.network, original["network"])
                    self.assertEqual(restored.metadata["source_file"], str(target.absolute()))
                    self.assertEqual("saved_at" in restored.metadata, include)
                    self.assertEqual(config.to_dict(), original)

    def yaml_failure_case(self, existing):
        provider = types.ModuleType("yaml")
        failure = ValueError("controlled YAML serialization failure")
        calls = []

        def dump(data, stream=None, **kwargs):
            calls.append((stream, kwargs))
            # Shows why the old streaming implementation could destroy a target.
            if stream is not None:
                stream.write("partial invalid YAML")
            raise failure

        provider.dump = dump
        module = self.load_config(provider)
        config = module.ExperimentConfig(experiment={"name": "example"})
        original = copy.deepcopy(config.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            target = (Path(directory) / "config.yaml" if existing else
                      Path(directory) / "new-parent" / "config.yaml")
            if existing:
                target.write_bytes(b"original YAML bytes\n")
            with self.assertRaises(ValueError) as caught:
                config.to_yaml(target)
            self.assertIs(caught.exception, failure)
            if existing:
                self.assertEqual(target.read_bytes(), b"original YAML bytes\n")
            else:
                self.assertFalse(target.exists())
                self.assertFalse(target.parent.exists())
        self.assertEqual(calls, [(None, {"default_flow_style": False, "sort_keys": False})])
        self.assertEqual(config.to_dict(), original)

    def test_yaml_dump_failure_preserves_existing_file(self):
        self.yaml_failure_case(existing=True)

    def test_yaml_dump_failure_leaves_new_destination_absent(self):
        self.yaml_failure_case(existing=False)

    def test_yaml_success_stub_format_kwargs_and_metadata_flags(self):
        provider = types.ModuleType("yaml")
        calls = []

        def dump(data, stream=None, **kwargs):
            calls.append((copy.deepcopy(data), stream, kwargs))
            return "controlled YAML payload\n"

        provider.dump = dump
        module = self.load_config(provider)
        config = module.ExperimentConfig(experiment={"name": "example"})
        original = copy.deepcopy(config.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            for include in (True, False):
                with self.subTest(include_metadata=include):
                    target = Path(directory) / str(include) / "config.yaml"
                    expected = copy.deepcopy(original)
                    if include:
                        expected["metadata"]["saved_at"] = STAMP
                    else:
                        expected.pop("metadata")
                    with patch.object(module, "datetime", FixedClock):
                        self.assertIsNone(config.to_yaml(target, include_metadata=include))
                    self.assertEqual(target.read_text(), "controlled YAML payload\n")
                    self.assertEqual(calls[-1], (expected, None,
                        {"default_flow_style": False, "sort_keys": False}))
                    self.assertEqual(config.to_dict(), original)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()

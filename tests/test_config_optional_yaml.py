"""Standard-library-only optional YAML import boundary regression tests."""
import builtins
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "fibernet" / "utils" / "config.py"


class ConfigOptionalYamlTests(unittest.TestCase):
    def load_config(self, yaml_outcome):
        """Execute the real owner, not the package facade or a rewritten excerpt."""
        name = "_fibernet_config_optional_yaml_regression"
        spec = importlib.util.spec_from_file_location(name, SOURCE)
        module = importlib.util.module_from_spec(spec)
        # Load the real shared parser helper without importing the package facade.
        helper_name = "fibernet._json"
        helper_spec = importlib.util.spec_from_file_location(helper_name, SOURCE.parents[1] / "_json.py")
        helper = importlib.util.module_from_spec(helper_spec)
        original_import = builtins.__import__
        yaml_calls = []

        def controlled_import(requested, globals=None, locals=None, fromlist=(), level=0):
            if requested == "yaml" and level == 0:
                yaml_calls.append(requested)
                if isinstance(yaml_outcome, BaseException):
                    raise yaml_outcome
                return yaml_outcome
            return original_import(requested, globals, locals, fromlist, level)

        # Dataclasses resolves annotations through sys.modules. Both patches are
        # scoped and restored even when importing the owner raises an exception.
        with patch.dict(sys.modules, {name: module, helper_name: helper}), \
                patch.object(sys, "dont_write_bytecode", True), \
                patch("builtins.__import__", new=controlled_import):
            helper_spec.loader.exec_module(helper)
            try:
                spec.loader.exec_module(module)
            finally:
                self.assertEqual(yaml_calls, ["yaml"])
        return module

    def test_successful_yaml_provider_is_preserved(self):
        provider = types.ModuleType("yaml")
        module = self.load_config(provider)
        self.assertTrue(module.HAS_YAML)
        self.assertIs(module.yaml, provider)
        config = module.ExperimentConfig.from_dict({"experiment": {"name": "example"}})
        self.assertEqual(config.to_dict()["experiment"], {"name": "example"})

    def test_missing_top_level_yaml_remains_optional(self):
        absent = ModuleNotFoundError("No module named 'yaml'", name="yaml")
        module = self.load_config(absent)
        self.assertFalse(module.HAS_YAML)
        self.assertIsNone(module.yaml)
        self.assertEqual(module.ExperimentConfig.from_dict({}).to_dict()["network"], {})
        # Dependency check precedes file opening, so no filesystem fixture needed.
        with self.assertRaisesRegex(ImportError, "PyYAML is required"):
            module.ExperimentConfig.from_yaml("unused-not-opened.yaml")

    def test_transitive_missing_dependency_propagates_same_exception(self):
        broken = ModuleNotFoundError("No module named 'yaml_private_dependency'",
                                     name="yaml_private_dependency")
        with self.assertRaises(ModuleNotFoundError) as caught:
            self.load_config(broken)
        self.assertIs(caught.exception, broken)
        self.assertEqual(caught.exception.name, "yaml_private_dependency")

    def test_provider_internal_import_error_propagates_same_exception(self):
        broken = ImportError("provider internal import failure", name="yaml")
        with self.assertRaises(ImportError) as caught:
            self.load_config(broken)
        self.assertIs(caught.exception, broken)


if __name__ == "__main__":
    unittest.main()

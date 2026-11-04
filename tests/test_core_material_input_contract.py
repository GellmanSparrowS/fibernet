"""Direct core material contracts, runnable with the stdlib unittest runner."""
import copy
import json
import unittest

import numpy as np

from fibernet import Material
from fibernet.core.material import MATERIAL_DB, get_material, list_materials


class CoreMaterialInputContractTests(unittest.TestCase):
    def test_every_core_catalog_name_is_retrievable_without_catalog_changes(self):
        names = list_materials()
        before = {name: value.to_dict() for name, value in MATERIAL_DB.items()}
        for name in names:
            with self.subTest(name=name):
                result = get_material(name)
                self.assertIsNot(result, MATERIAL_DB[name])
                self.assertEqual(result.to_dict(), before[name])
        self.assertEqual(list_materials(), names)
        self.assertEqual({name: value.to_dict() for name, value in MATERIAL_DB.items()}, before)

    def test_core_catalog_mixed_case_name_lookup(self):
        for name in ('uHMWPE', 'UHMWPE', 'uhmwpe', ' uHmWpE '):
            with self.subTest(name=name):
                result = get_material(name)
                self.assertEqual(result.name, 'uHMWPE')
                self.assertEqual(result.to_dict(), MATERIAL_DB['uHMWPE'].to_dict())

    def test_core_catalog_existing_normalization_and_missing_name(self):
        self.assertEqual(get_material(' STEEL ').to_dict(), MATERIAL_DB['steel'].to_dict())
        with self.assertRaisesRegex(KeyError, 'not found'):
            get_material('not_a_material')

    def test_direct_material_tensor_deserialization_preserves_payload(self):
        for kind in ('json_list', 'ndarray'):
            with self.subTest(tensor_input=kind):
                tensor = np.eye(6).tolist() if kind == 'json_list' else np.eye(6)
                extra = {'nested': {'tag': 'legacy'}}
                payload = {'name': 'tensor', 'stiffness_tensor': tensor, 'extra': extra}
                before_tensor = copy.deepcopy(tensor)
                serialized = json.dumps(payload, allow_nan=False) if kind == 'json_list' else None
                material = Material.from_dict(payload)
                self.assertIs(payload['stiffness_tensor'], tensor)
                self.assertIs(payload['extra'], extra)
                self.assertIs(material.extra, extra)  # Preserve existing shallow metadata policy.
                self.assertIsNot(material.stiffness_tensor, tensor)
                np.testing.assert_array_equal(tensor, before_tensor)
                np.testing.assert_array_equal(material.stiffness_tensor, np.eye(6))
                if kind == 'json_list':
                    self.assertEqual(json.dumps(payload, allow_nan=False), serialized)
                else:
                    self.assertFalse(np.shares_memory(material.stiffness_tensor, tensor))
                material.stiffness_tensor[0, 0] = 7
                np.testing.assert_array_equal(tensor, before_tensor)

    def test_direct_material_constructor_failure_preserves_payload(self):
        tensor = np.eye(6).tolist()
        payload = {'stiffness_tensor': tensor, 'unknown_material_field': 1}
        before = copy.deepcopy(payload)
        serialized = json.dumps(payload, allow_nan=False)
        with self.assertRaises(TypeError):
            Material.from_dict(payload)
        self.assertIs(payload['stiffness_tensor'], tensor)
        self.assertEqual(payload, before)
        self.assertEqual(json.dumps(payload, allow_nan=False), serialized)

    def test_direct_material_optional_tensor_and_computed_shear_unchanged(self):
        cases = ({}, {'stiffness_tensor': None},
                 {'youngs_modulus': 100., 'poissons_ratio': .25})
        for index, payload in enumerate(cases):
            with self.subTest(case=index):
                before = copy.deepcopy(payload)
                material = Material.from_dict(payload)
                self.assertEqual(payload, before)
                self.assertIsNone(material.stiffness_tensor)
                self.assertEqual(material.shear_modulus,
                                 material.youngs_modulus / (2 * (1 + material.poissons_ratio)))


if __name__ == '__main__':
    unittest.main()

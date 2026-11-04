"""Real h5py I/O plus explicitly injected filesystem commit failures.

h5py is required: a missing/broken installation must fail collection, not skip.
Schema checks cover the existing writer schema, not fields it never serialized.
"""

import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from fibernet.core.fiber import CrossSection, Fiber
from fibernet.core.material import Material
from fibernet.core.network import Crosslink, FiberNetwork


class StringPath:
    def __init__(self, path):
        self.path = path

    def __fspath__(self):
        return str(self.path)


class Hdf5SafeSaveIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='fibernet-hdf5-test-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def network(self, box=True):
        material = Material(name='test-material', density=1234.0,
                            youngs_modulus=2.5e9, poissons_ratio=0.25,
                            extra={'label': 'preserved', 'values': [1, None, True]})
        fibers = [
            Fiber(np.array([[0., 0., 0.], [1., 0., 0.]]),
                  radius=0.2, material=material, fiber_id=7, segments=3),
            Fiber(np.array([[1., 0., 0.], [1., 1., 0.]]),
                  radius=0.3, material=material, fiber_id=11, segments=5,
                  cross_section=CrossSection.ELLIPTICAL,
                  cross_section_params={'semi_minor': 0.1}),
        ]
        return FiberNetwork(
            fibers=fibers,
            crosslinks=[Crosslink(0, 1, 1.0, 0.0, np.array([1., 0., 0.]),
                                 crosslink_type='pinned', strength=8.0, stiffness=9.0)],
            dimension=2, periodic=True,
            box_size=np.array([2., 3., 1.]) if box else None,
            metadata={'source': 'tiny fixture', 'revision': 1},
        )

    def assert_no_temporary(self):
        self.assertEqual(list(self.root.glob('.fibernet-*.h5.tmp')), [])

    def assert_schema(self, path, net):
        with h5py.File(path, 'r') as hf:
            self.assertEqual(set(hf.keys()), {'fibers', 'crosslinks'})
            expected_attrs = {'version', 'dimension', 'periodic', 'metadata'}
            if net.box_size is not None:
                expected_attrs.add('box_size')
            self.assertEqual(set(hf.attrs), expected_attrs)
            self.assertEqual(hf.attrs['version'], '0.1.0')
            self.assertEqual(hf.attrs['dimension'], net.dimension)
            self.assertEqual(hf.attrs['periodic'], net.periodic)
            self.assertEqual(json.loads(hf.attrs['metadata']), net.metadata)
            if net.box_size is not None:
                np.testing.assert_array_equal(hf.attrs['box_size'], net.box_size)
            self.assertEqual(set(hf['fibers']), {'fiber_0', 'fiber_1'})
            for index, fiber in enumerate(net.fibers):
                group = hf['fibers'][f'fiber_{index}']
                self.assertEqual(set(group), {'centerline'})
                self.assertEqual(set(group.attrs), {
                    'radius', 'fiber_id', 'cross_section', 'segments', 'material'})
                np.testing.assert_array_equal(group['centerline'][...], fiber.centerline)
                self.assertEqual(group['centerline'].dtype, fiber.centerline.dtype)
                self.assertEqual(group.attrs['radius'], fiber.radius)
                self.assertEqual(group.attrs['fiber_id'], fiber.fiber_id)
                self.assertEqual(group.attrs['cross_section'], fiber.cross_section.value)
                self.assertEqual(group.attrs['segments'], fiber.segments)
                self.assertEqual(json.loads(group.attrs['material']), fiber.material.to_dict())
            self.assertEqual(set(hf['crosslinks']), {'cl_0'})
            group = hf['crosslinks']['cl_0']
            crosslink = net.crosslinks[0]
            self.assertEqual(set(group), {'position'})
            self.assertEqual(set(group.attrs), {
                'fiber_i', 'fiber_j', 'param_i', 'param_j', 'crosslink_type'})
            np.testing.assert_array_equal(group['position'][...], crosslink.position)
            self.assertEqual(group['position'].dtype, crosslink.position.dtype)
            for name in ('fiber_i', 'fiber_j', 'param_i', 'param_j', 'crosslink_type'):
                self.assertEqual(group.attrs[name], getattr(crosslink, name))

    def test_new_and_existing_full_schema_readback(self):
        for box in (False, True):
            with self.subTest(box=box):
                net = self.network(box)
                target = self.root / f'network-{box}.h5'
                original_record = net.to_dict()
                original_material = net.fibers[0].material
                net.save_hdf5(str(target))
                self.assert_schema(target, net)
                self.assertEqual(net.to_dict(), original_record)
                self.assertIs(net.fibers[0].material, original_material)
                self.assertIs(net.fibers[1].material, original_material)
                net.metadata['revision'] = 2
                net.save_hdf5(target)
                self.assert_schema(target, net)
                self.assert_no_temporary()

    def test_custom_pathlike_success(self):
        target = self.root / 'pathlike.h5'
        net = self.network()
        net.save_hdf5(StringPath(target))
        self.assert_schema(target, net)
        self.assert_no_temporary()

    def test_json_preflight_preserves_old_or_absent_destination(self):
        for failure in ('metadata', 'material'):
            for existing in (False, True):
                with self.subTest(failure=failure, existing=existing):
                    target = self.root / f'{failure}-{existing}.h5'
                    old = b'old output, do not truncate'
                    if existing:
                        target.write_bytes(old)
                    net = self.network()
                    if failure == 'metadata':
                        net.metadata['bad'] = object()
                        error = TypeError
                    else:
                        net.fibers[1].material = None
                        error = AttributeError
                    with self.assertRaises(error):
                        net.save_hdf5(target)
                    if existing:
                        self.assertEqual(target.read_bytes(), old)
                    else:
                        self.assertFalse(target.exists())
                    self.assert_no_temporary()

    def test_real_late_dataset_failure_preserves_destination(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                target = self.root / f'late-{existing}.h5'
                old = b'old output'
                if existing:
                    target.write_bytes(old)
                net = self.network()
                # The first fiber writes normally; h5py rejects the second
                # dataset's arbitrary object dtype after opening the temp file.
                net.fibers[1].centerline = np.full((2, 3), object(), dtype=object)
                with self.assertRaises(TypeError):
                    net.save_hdf5(target)
                if existing:
                    self.assertEqual(target.read_bytes(), old)
                else:
                    self.assertFalse(target.exists())
                self.assert_no_temporary()

    def test_injected_replace_failure_after_real_hdf5_write(self):
        target = self.root / 'replace.h5'
        target.write_bytes(b'old output')
        net = self.network()
        failure = OSError('injected replace rejection')

        def reject(source, destination):
            self.assertEqual(Path(destination), target)
            self.assert_schema(source, net)  # Successfully closed real HDF5.
            raise failure

        with patch('os.replace', side_effect=reject) as replace:
            with self.assertRaises(OSError) as caught:
                net.save_hdf5(target)
        self.assertIs(caught.exception, failure)
        replace.assert_called_once()
        self.assertEqual(target.read_bytes(), b'old output')
        self.assert_no_temporary()

    def test_injected_chmod_failure_prevents_replace(self):
        target = self.root / 'chmod.h5'
        target.write_bytes(b'old output')
        failure = PermissionError('injected chmod rejection')
        with patch('os.chmod', side_effect=failure) as chmod, patch('os.replace') as replace:
            with self.assertRaises(PermissionError) as caught:
                self.network().save_hdf5(target)
        self.assertIs(caught.exception, failure)
        chmod.assert_called_once()
        replace.assert_not_called()
        self.assertEqual(target.read_bytes(), b'old output')
        self.assert_no_temporary()

    @unittest.skipUnless(os.name == 'posix', 'POSIX link/mode policy test')
    def test_symlink_target_is_rejected_without_changing_referent(self):
        referent = self.root / 'referent.h5'
        referent.write_bytes(b'referent')
        target = self.root / 'symlink.h5'
        target.symlink_to(referent)
        with self.assertRaises(ValueError):
            self.network().save_hdf5(target)
        self.assertTrue(target.is_symlink())
        self.assertEqual(referent.read_bytes(), b'referent')
        self.assert_no_temporary()

    @unittest.skipUnless(os.name == 'posix', 'POSIX link/mode policy test')
    def test_hardlink_target_is_rejected_without_changing_either_name(self):
        target = self.root / 'hardlink.h5'
        alias = self.root / 'alias.h5'
        target.write_bytes(b'linked')
        os.link(target, alias)
        with self.assertRaises(ValueError):
            self.network().save_hdf5(target)
        self.assertEqual(target.read_bytes(), b'linked')
        self.assertEqual(alias.read_bytes(), b'linked')
        self.assertEqual(target.stat().st_ino, alias.stat().st_ino)
        self.assert_no_temporary()

    @unittest.skipUnless(os.name == 'posix', 'POSIX mode-bit policy, not ACL authorization')
    def test_readonly_mode_policy_and_existing_mode_preservation(self):
        target = self.root / 'readonly.h5'
        target.write_bytes(b'readonly')
        target.chmod(0o444)
        try:
            with self.assertRaises(PermissionError):
                self.network().save_hdf5(target)
            self.assertEqual(target.read_bytes(), b'readonly')
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o444)
        finally:
            target.chmod(0o600)
        target.chmod(0o640)
        net = self.network()
        net.save_hdf5(target)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
        self.assert_schema(target, net)
        self.assert_no_temporary()

    def test_directory_target_and_missing_parent_are_not_modified(self):
        directory = self.root / 'destination'
        directory.mkdir()
        marker = directory / 'marker'
        marker.write_bytes(b'preserve')
        with self.assertRaises(ValueError):
            self.network().save_hdf5(directory)
        self.assertEqual(marker.read_bytes(), b'preserve')
        with self.assertRaises(FileNotFoundError):
            self.network().save_hdf5(self.root / 'missing' / 'file.h5')
        self.assertFalse((self.root / 'missing').exists())
        self.assert_no_temporary()


if __name__ == '__main__':
    unittest.main()

"""Batch output must not overwrite source structures."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fibernet.easy as easy


class BatchInputOutputCollisionTests(unittest.TestCase):
    def test_result_file_cannot_alias_any_input(self):
        for alias in ('same_directory', 'symlink', 'hardlink'):
            with self.subTest(alias=alias), TemporaryDirectory() as temp:
                root = Path(temp)
                inputs = root / 'inputs'
                output = inputs if alias == 'same_directory' else root / 'output'
                inputs.mkdir()
                output.mkdir(exist_ok=True)
                a = inputs / 'a.json'
                b = inputs / ('a_result.json' if alias == 'same_directory' else 'b.json')
                a.write_bytes(b'original a')
                b.write_bytes(b'original b')
                target = output / 'a_result.json'
                try:
                    if alias == 'symlink':
                        target.symlink_to(b)
                    elif alias == 'hardlink':
                        os.link(b, target)
                except (OSError, NotImplementedError) as exc:
                    self.skipTest(f'{alias} links unavailable: {exc}')
                before = {p: p.read_bytes() for p in (a, b)}
                with patch.object(easy, '_taichi_engine_class',
                                  side_effect=AssertionError('engine constructed')):
                    with self.assertRaisesRegex(ValueError, 'output.*input'):
                        easy.batch_simulate_from_json(str(inputs), str(output))
                self.assertEqual({p: p.read_bytes() for p in before}, before)
                self.assertFalse((output / 'simulation_results.csv').exists())


    def test_csv_cannot_alias_input_symlink(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inputs, output = root / 'inputs', root / 'output'
            inputs.mkdir(); output.mkdir()
            csv_path = output / 'simulation_results.csv'
            csv_path.write_bytes(b'filename,success\n')
            (inputs / 'b.json').write_bytes(b'graph')
            try:
                (inputs / 'a.json').symlink_to(csv_path)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f'symlinks unavailable: {exc}')
            with patch.object(easy, '_taichi_engine_class', side_effect=AssertionError('engine constructed')):
                with self.assertRaisesRegex(ValueError, 'output.*input'):
                    easy.batch_simulate_from_json(str(inputs), str(output))
            self.assertEqual(csv_path.read_bytes(), b'filename,success\n')

    def test_input_and_output_parent_symlinks(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inputs, real_output, output = root / 'inputs', root / 'real_output', root / 'output'
            inputs.mkdir(); real_output.mkdir()
            (inputs / 'a.json').write_bytes(b'graph')
            target = real_output / 'a_result.json'
            target.write_bytes(b'original input')
            try:
                (inputs / 'b.json').symlink_to(target)
                output.symlink_to(real_output, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f'symlinks unavailable: {exc}')
            with patch.object(easy, '_taichi_engine_class', side_effect=AssertionError('engine constructed')):
                with self.assertRaisesRegex(ValueError, 'output.*input'):
                    easy.batch_simulate_from_json(str(inputs), str(output))
            self.assertEqual(target.read_bytes(), b'original input')

    def test_separate_directories_still_run_and_resume(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / 'inputs'
            inputs.mkdir()
            (inputs / 'a.json').write_bytes(b'graph')
            output = root / 'output'
            calls = []
            class Result:
                max_force = 2.0
                max_stretch = 1.5
                mean_stretch = 1.2
                std_stretch = 0.1
                time_seconds = 0.01
                def save(self, path, detailed):
                    Path(path).write_text('{}')
            class Engine:
                def stretch_test(self, graph, **kwargs):
                    calls.append(graph)
                    return Result()
            graph = SimpleNamespace(num_nodes=2, num_edges=1)
            with patch.object(easy, '_taichi_engine_class', return_value=Engine), \
                 patch.object(easy, '_batch_graph_from_bytes', return_value=graph), \
                 patch.object(easy, '_preflight_taichi_geometry'):
                first = easy.batch_simulate_from_json(str(inputs), str(output))
                second = easy.batch_simulate_from_json(str(inputs), str(output))
            self.assertEqual(first, second)
            self.assertEqual(calls, [graph])
            self.assertTrue((output / 'a_result.json').exists())
            self.assertEqual((inputs / 'a.json').read_bytes(), b'graph')


if __name__ == '__main__':
    unittest.main()

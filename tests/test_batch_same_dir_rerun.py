"""Same-directory result files must not silently become batch inputs."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fibernet.easy as easy


class SameDirectoryRerunTests(unittest.TestCase):
    def test_result_input_collision_on_rerun(self):
        for already_recorded in (False, True):
            for replaced_by_user in (False, True):
                with self.subTest(already_recorded=already_recorded,
                                  replaced_by_user=replaced_by_user), TemporaryDirectory() as temp:
                    root = Path(temp)
                    (root / 'a.json').write_bytes(b'graph')
                    calls = []
                    class Result:
                        max_force = 2.0
                        max_stretch = 1.5
                        mean_stretch = 1.2
                        std_stretch = 0.1
                        time_seconds = 0.01
                        def save(self, path, detailed):
                            Path(path).write_text('{"result":true}')
                    class Engine:
                        def __init__(self):
                            calls.append('init')
                        def stretch_test(self, graph, **kwargs):
                            calls.append('solve')
                            return Result()
                    graph = SimpleNamespace(num_nodes=2, num_edges=1)
                    with patch.object(easy, '_taichi_engine_class', return_value=Engine), \
                         patch.object(easy, '_batch_graph_from_bytes', return_value=graph), \
                         patch.object(easy, '_preflight_taichi_geometry'):
                        easy.batch_simulate_from_json(str(root), str(root))
                        result = root / 'a_result.json'
                        if replaced_by_user:
                            result.write_bytes(b'a legitimate user structure')
                        csv_path = root / 'simulation_results.csv'
                        if already_recorded:
                            # Model a historical run which already recorded this
                            # ambiguous file, so there will be no pending cases.
                            with csv_path.open(newline='', encoding='utf-8') as f:
                                reader = csv.DictReader(f)
                                fields = reader.fieldnames
                                rows = list(reader)
                            params = easy._batch_parameters('stretch', 'spring', 1.5, 1e5,
                                                           0.3, 1000, 200)
                            binding = easy._batch_binding(easy._batch_input_sha256(result), params,
                                                          easy._batch_implementation_binding())
                            rows.append({'filename': result.name, 'success': 'False',
                                         'run_binding': easy._batch_canonical_json(binding)})
                            with csv_path.open('w', newline='', encoding='utf-8') as f:
                                writer = csv.DictWriter(f, fieldnames=fields)
                                writer.writeheader(); writer.writerows(rows)
                        before = {p: p.read_bytes() for p in (root/'a.json', result, csv_path)}
                        with self.assertRaisesRegex(ValueError, 'separate intended input structures'):
                            easy.batch_simulate_from_json(str(root), str(root))
                        self.assertEqual(calls, ['init', 'solve'])
                        self.assertEqual({p: p.read_bytes() for p in before}, before)


if __name__ == '__main__':
    unittest.main()

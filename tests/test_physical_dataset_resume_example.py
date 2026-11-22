"""Resume semantics of the public physical-label example (no model or solver)."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from examples import physical_dataset_workflow as example


class PhysicalDatasetResumeExampleTests(unittest.TestCase):
    def test_completed_checkpoint_above_target_retains_all_rows(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            checkpoint = folder / "physical_labels.npz"
            np.savez_compressed(checkpoint, X=np.ones((3, 14)), Y=np.ones((3, 3)))
            before = checkpoint.read_bytes()

            class CompletedDataset:
                config = {"engine_source": "synthetic-test-only"}
                next_seed = 26

                def __init__(self, *, path, **kwargs):
                    self.path = path

                def generate(self, n):
                    if n != 2:
                        raise AssertionError("the requested minimum changed")
                    with np.load(self.path, allow_pickle=False) as saved:
                        return saved["X"], saved["Y"]

            with patch.object(example, "PlanarPhysicalDataset", CompletedDataset):
                summary = example.PhysicalDatasetWorkflow(folder, samples=2).run()
            self.assertEqual(summary["samples"], 3)
            self.assertEqual(summary["next_seed"], 26)
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_short_or_malformed_results_do_not_replace_summary(self):
        cases = [
            ((1, 14), (1, 3), "did not reach"),
            ((2, 13), (2, 3), "misaligned"),
            ((2, 14), (3, 3), "misaligned"),
            ((2, 14), (2, 2), "misaligned"),
            ((14,), (2, 3), "misaligned"),
        ]
        for xshape, yshape, message in cases:
            with self.subTest(xshape=xshape, yshape=yshape), TemporaryDirectory() as directory:
                summary = Path(directory) / "summary.json"
                summary.write_bytes(b"prior summary")
                with patch.object(example, "PlanarPhysicalDataset") as dataset:
                    dataset.return_value.generate.return_value = np.ones(xshape), np.ones(yshape)
                    with self.assertRaisesRegex(RuntimeError, message):
                        example.PhysicalDatasetWorkflow(directory, samples=2).run()
                self.assertEqual(summary.read_bytes(), b"prior summary")

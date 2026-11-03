"""Successful ensemble members retain the seeds that generated them."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

from fibernet.utils.ensemble import generate_ensemble


class TestEnsembleSeedAlignment(unittest.TestCase):
    def test_seed_alignment_after_generation_failures(self):
        for parallel in (False, True):
            for failed in (set(), {42}, {43}, {42, 43, 44}):
                with self.subTest(parallel=parallel, failed=failed):
                    def generate(seed):
                        if seed in failed:
                            raise ValueError("deliberate generation failure")
                        fiber = SimpleNamespace(length=1.0, radius=0.5)
                        return SimpleNamespace(seed=seed, num_fibers=1,
                                               num_crosslinks=0, fibers=[fiber])

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="Failed to generate network")
                        ensemble = generate_ensemble(
                            generate, num_networks=3, base_seed=42,
                            parallel=parallel, show_progress=False,
                        )
                    expected = [seed for seed in (42, 43, 44) if seed not in failed]
                    self.assertEqual(ensemble.seeds, expected)
                    self.assertEqual([net.seed for net in ensemble.networks], expected)
                    self.assertEqual(ensemble.metadata["num_generated"], len(expected))

    def test_parallel_seed_alignment_in_reverse_completion_order(self):
        def generate(seed):
            if seed == 43:
                raise ValueError("deliberate generation failure")
            fiber = SimpleNamespace(length=1.0, radius=0.5)
            return SimpleNamespace(seed=seed, num_fibers=1,
                                   num_crosslinks=0, fibers=[fiber])

        # Consume submitted futures in reverse order, independent of timing.
        with patch("fibernet.utils.ensemble.as_completed",
                   side_effect=lambda futures: iter(reversed(list(futures)))):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Failed to generate network")
                ensemble = generate_ensemble(generate, num_networks=3, base_seed=42,
                                             parallel=True, show_progress=False)
        self.assertEqual(ensemble.seeds, [42, 44])
        self.assertEqual([net.seed for net in ensemble.networks], [42, 44])


if __name__ == "__main__":
    unittest.main()

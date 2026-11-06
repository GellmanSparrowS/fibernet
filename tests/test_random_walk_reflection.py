"""Random-walk step endpoints remain inside a reflected cubic box."""
import unittest
import numpy as np
from fibernet.gen.disordered import random_walk_fibers


def reference(seed, steps, step_size, box_size):
    """Small reference using repeated single-wall reflections."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0, box_size, size=3)
    points = [pos.copy()]
    for _ in range(steps):
        direction = rng.randn(3)
        direction /= np.linalg.norm(direction) + 1e-12
        pos = pos + direction * step_size
        for d in range(3):
            while pos[d] < 0 or pos[d] > box_size:
                if pos[d] < 0:
                    pos[d] = -pos[d]
                else:
                    pos[d] = 2 * box_size - pos[d]
        points.append(pos.copy())
    return np.array(points)


class TestRandomWalkReflection(unittest.TestCase):
    def test_multiple_wall_crossings(self):
        for seed in (0, 42, 123):
            with self.subTest(seed=seed):
                net = random_walk_fibers(num_fibers=1, num_steps=3, step_size=10,
                                         box_size=1, seed=seed, ensure_connected=False)
                points = net.fibers[0].centerline
                self.assertEqual(points.shape, (4, 3))
                self.assertTrue(np.all((points >= 0) & (points <= 1)))
                np.testing.assert_allclose(points, reference(seed, 3, 10, 1),
                                           rtol=0, atol=1e-12)
                repeat = random_walk_fibers(num_fibers=1, num_steps=3, step_size=10,
                                            box_size=1, seed=seed, ensure_connected=False)
                np.testing.assert_array_equal(points, repeat.fibers[0].centerline)

    def test_single_wall_reflection_retains_existing_results(self):
        net = random_walk_fibers(num_fibers=1, num_steps=8, step_size=0.75,
                                 box_size=1, seed=42, ensure_connected=False)
        np.testing.assert_array_equal(net.fibers[0].centerline, reference(42, 8, 0.75, 1))


if __name__ == '__main__':
    unittest.main()

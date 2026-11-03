"""Small source-only pore scale regressions; no generator or solver."""
import unittest
import numpy as np
from fibernet.analysis.snapshot_features import SnapshotFeatureExtractor


class SnapshotPoreScaleTests(unittest.TestCase):
    def check_scaled(self, points, edges, expected_count, area, perim, coverage, rect=None):
        ex = SnapshotFeatureExtractor()
        for s in (1., 1e-3, 1e3):
            r = None if rect is None else np.asarray(rect)*s
            f = ex.extract(np.asarray(points)*s, edges, rect=r)
            self.assertEqual(f['pore_count'], expected_count)
            self.assertAlmostEqual(f['pore_area_mean'] / (s*s), area, places=11)
            self.assertAlmostEqual(f['pore_perim_mean'] / s, perim, places=11)
            self.assertAlmostEqual(f['porosity'], coverage, places=11)
            self.assertTrue(np.allclose(np.asarray(f['pore_areas']) / (s*s), area))

    def test_square_and_region(self):
        p=[[0,0],[1,0],[1,1],[0,1]]; e=[[0,1],[1,2],[2,3],[3,0]]
        self.check_scaled(p,e,1,1.,4.,1.)
        self.check_scaled(p,e,1,1.,4.,.25,rect=[0,0,2,2])
        f=SnapshotFeatureExtractor().extract(p,e,rect=['0','0','2','2'])
        self.assertAlmostEqual(f['porosity'],.25)

    def test_crossing_split_and_adjacency(self):
        p=[[0,0],[1,0],[1,1],[0,1]];e=[[0,1],[1,2],[2,3],[3,0],[0,2],[1,3]]
        self.check_scaled(p,e,4,.25,1.+2**.5,1.)
        for s in (1.,1e-3,1e3):
            f=SnapshotFeatureExtractor().extract(np.asarray(p)*s,e)
            self.assertAlmostEqual(f['largest_pore_ratio'],.25)
            self.assertEqual(f['pore_neighbor_mean'],2.)

    def test_t_junction_splits_faces(self):
        p=[[0,0],[1,0],[1,1],[0,1],[.5,0],[.5,1]]
        e=[[0,1],[1,2],[2,3],[3,0],[4,5]]
        self.check_scaled(p,e,2,.5,3.,1.)

    def test_distant_isolated_node_does_not_erase_face(self):
        p=[[0,0],[1,0],[1,1],[0,1],[1e9,1e9]]
        e=[[0,1],[1,2],[2,3],[3,0]]
        f=SnapshotFeatureExtractor().extract(p,e)
        self.assertEqual(f['pore_count'],1)
        self.assertEqual(f['pore_area_mean'],1.)

    def test_contact_keeps_original_input_units(self):
        p=np.array([[0,0],[1,0],[1,1],[0,1],[.5,-.2],[.5,1.2],[-.2,.5],[1.2,.5]])
        e=[[0,1],[1,2],[2,3],[3,0],[4,5],[6,7]]
        from fibernet.analysis import ContactConfig
        f=SnapshotFeatureExtractor(include_contact=True,contact_config=ContactConfig(
            width=.16,resolution=64,max_pixels=200000)).extract(p,e)
        self.assertEqual(len(f),54)
        self.assertEqual((f['pore_count'],f['contact_pair_count']),(4,5))
        self.assertAlmostEqual(f['pore_area_mean'],.25)
        self.assertAlmostEqual(f['contact_overlap_area'],.13416780045351476,places=12)

    def test_empty_and_zero_extent(self):
        ex=SnapshotFeatureExtractor()
        for p,e in [(np.empty((0,2)),[]), ([[0,0],[0,0]],[[0,1]])]:
            f=ex.extract(p,e)
            self.assertEqual(f['pore_count'],0)
            self.assertEqual(f['pore_area_mean'],0.)

    def test_unrepresentable_extent_is_explicit(self):
        with np.errstate(all='ignore'):
            with self.assertRaisesRegex(ValueError,'not representable'):
                SnapshotFeatureExtractor().extract([[-1e308,0],[1e308,0]], [[0,1]])
            for scale in (1e-200,1e200):
                with self.subTest(scale=scale), self.assertRaisesRegex(ValueError,'area scale'):
                    SnapshotFeatureExtractor().extract([[0,0],[scale,0]], [[0,1]])

if __name__ == '__main__': unittest.main()

"""Snapshot open-chain endpoint geometry on small independent fixtures."""
import unittest
import numpy as np
from fibernet.analysis.snapshot_features import SnapshotFeatureExtractor, _chain_straightness

class SnapshotChainStraightness(unittest.TestCase):
    def test_saved_graph_corner_slice(self):
        # Coordinates from saved synthetic triangular graph node IDs 1, 0, 2.
        positions=np.array([[0.01,0.],[0.,0.],[0.005,0.008660254037844387]])
        values=_chain_straightness(positions,np.array([[0,1],[1,2]]))
        np.testing.assert_allclose(values,[.5],rtol=1e-12)

    def test_open_chain_endpoints_and_public_metric(self):
        pos=np.array([[0.,0.],[1.,0.],[1.,1.]])
        e=np.array([[0,1],[1,2]])
        result=SnapshotFeatureExtractor().extract(pos,e)
        np.testing.assert_allclose(result['segment_straightness'],[2**.5/2])
        self.assertAlmostEqual(result['straightness_mean'],2**.5/2)
        pos=np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.]])
        np.testing.assert_allclose(_chain_straightness(pos,np.array([[0,1],[1,2],[2,3]])),[1.])

    def test_branch_boundaries_are_included_once(self):
        pos=np.array([[0.,0.],[1.,0.],[2.,0.],[0.,1.],[0.,2.],[0.,-1.]])
        e=np.array([[0,1],[1,2],[0,3],[3,4],[0,5]])
        np.testing.assert_allclose(sorted(_chain_straightness(pos,e)),[1.,1.])

    def test_parallel_loop_does_not_hide_open_chain(self):
        pos=np.array([[0.,0.],[1.,0.],[2.,0.],[0.,1.]])
        edges=np.array([[0,1],[1,2],[0,3],[0,3]])
        np.testing.assert_allclose(_chain_straightness(pos,edges),[1.])
        np.testing.assert_allclose(_chain_straightness(pos,edges[::-1,::-1]),[1.])

    def test_coincident_distinct_endpoints_and_length_cutoff(self):
        edges=np.array([[0,1],[1,2]])
        closed_coordinates=np.array([[0.,0.],[1.,0.],[0.,0.]])
        np.testing.assert_allclose(_chain_straightness(closed_coordinates,edges),[0.])
        self.assertEqual(_chain_straightness(np.zeros((3,2)),edges).size,0)
        tiny=np.array([[0.,0.],[.5e-12,0.],[1e-12,0.]])
        self.assertEqual(_chain_straightness(tiny,edges).size,0)

    def test_closed_and_single_edge_are_excluded(self):
        pos=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
        self.assertEqual(_chain_straightness(pos,np.array([[0,1],[1,2],[2,3],[3,0]])).size,0)
        self.assertEqual(_chain_straightness(pos,np.array([[0,1]])).size,0)
        self.assertEqual(_chain_straightness(pos,np.array([[0,1],[0,1]])).size,0)
        # A loop returning to a degree-3 junction is not an open chain.
        self.assertEqual(_chain_straightness(pos,np.array([[0,1],[1,2],[2,0],[0,3]])).size,0)

if __name__=='__main__': unittest.main()

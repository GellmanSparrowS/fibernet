import tempfile
import unittest
from pathlib import Path
import numpy as np
from fibernet.core.structure_graph import StructureGraph

class InternalPointsSave(unittest.TestCase):
    def graph(self):
        g=StructureGraph();g.add_node([0,0]);g.add_node([2,0])
        g.add_edge(0,1,internal_points=np.ma.array([[1.,1.,0.]],mask=False))
        return g

    def test_actual_mask_rejected_preserving_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'graph.json';g=self.graph();g.save_json(path)
            before=path.read_bytes()
            g.edges[0].internal_points.mask[0,1]=True
            with self.assertRaisesRegex(ValueError,'internal_points.*masked'):
                g.to_dict()
            with self.assertRaisesRegex(ValueError,'internal_points.*masked'):
                g.save_json(path)
            self.assertEqual(path.read_bytes(),before)
            np.testing.assert_array_equal(StructureGraph.load_json(path).edges[0].internal_points,[[1,1,0]])
            new=Path(directory)/'new.json'
            with self.assertRaisesRegex(ValueError,'internal_points.*masked'):g.save_json(new)
            self.assertFalse(new.exists())
            self.assertTrue(g.edges[0].internal_points.mask[0,1])

    def test_unmasked_samples_roundtrip(self):
        for points in (np.array([[1.,1.,0.]]),np.ma.array([[1.,1.,0.]],mask=False),np.empty((0,3))):
            with self.subTest(shape=points.shape),tempfile.TemporaryDirectory() as directory:
                g=self.graph();g.edges[0].internal_points=points
                path=Path(directory)/'graph.json';g.save_json(path)
                np.testing.assert_array_equal(StructureGraph.load_json(path).edges[0].internal_points,points)

    def test_duplicate_json_key_protection_remains_active(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'graph.json';g=self.graph();g.save_json(path)
            before=path.read_bytes();g.metadata.update({1:'integer','1':'string'})
            with self.assertRaisesRegex(ValueError,'duplicate JSON object key'):g.save_json(path)
            self.assertEqual(path.read_bytes(),before)

if __name__=='__main__':unittest.main()

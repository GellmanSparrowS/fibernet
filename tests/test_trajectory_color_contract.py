"""Trajectory color semantics only; synthetic geometry, no numerical solve."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fibernet.core.structure_graph import StructureGraph
from fibernet.viz.render import render_trajectory_3d

class TrajectoryColorContract(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def graph(self, curved=False):
        g=StructureGraph(dimension=3)
        g.add_node([0,0,0]);g.add_node([1,1,1])
        g.add_edge(0,1,internal_points=np.array([[0,2,0]]) if curved else None)
        return g

    def test_stretch_rejects_straight_and_curved_before_output(self):
        for curved in (False,True):
            with self.subTest(curved=curved), tempfile.TemporaryDirectory() as directory:
                g=self.graph(curved);p=Path(directory)/'keep.png';p.write_bytes(b'preserve')
                output=Path(directory)/'frames'
                before=plt.get_fignums()
                with self.assertRaisesRegex(NotImplementedError,'per-frame edge stretch'):
                    render_trajectory_3d(g,[g.node_positions()+[3,4,0]],color_by='stretch',save_path=p,save_dir=output)
                self.assertEqual(plt.get_fignums(),before)
                self.assertEqual(p.read_bytes(),b'preserve')
                self.assertFalse(output.exists())

    def test_unknown_mode_rejects_even_with_no_frames(self):
        g=self.graph()
        for mode in ('strain','unknown',None):
            with self.subTest(mode=mode):
                before=plt.get_fignums()
                with self.assertRaisesRegex(ValueError,'color_by'):
                    render_trajectory_3d(g,[],color_by=mode)
                self.assertEqual(plt.get_fignums(),before)

    def test_rejected_modes_do_not_consume_trajectory(self):
        for mode,error in (('stretch',NotImplementedError),('unknown',ValueError)):
            with self.subTest(mode=mode):
                consumed=[]
                def frames():
                    consumed.append(True)
                    yield self.graph().node_positions()
                with self.assertRaises(error):
                    render_trajectory_3d(self.graph(),frames(),color_by=mode)
                self.assertEqual(consumed,[])

    def test_displacement_default_and_explicit_keep_actual_colors(self):
        g=self.graph();origin=g.node_positions()
        for frame,expected in ((origin+[3,4,0],1.0),(origin*2,0.5)):
            with self.subTest(expected=expected):
                figures=[render_trajectory_3d(g,[frame]),render_trajectory_3d(g,[frame],color_by='displacement')]
                for figure in figures:
                    np.testing.assert_allclose(figure.axes[0].collections[0].get_colors()[0],plt.cm.viridis(expected))
                    self.assertEqual(figure.axes[-1].get_ylabel(),'Displacement')
                for figure in figures:plt.close(figure)

    def test_explicit_node_map_preserved(self):
        g=self.graph();frame=g.node_positions()+[3,4,2]
        figure=render_trajectory_3d(g,[frame[::-1]],node_ids=[1,0])
        np.testing.assert_allclose(figure.axes[0].collections[0]._segments3d[0],frame)

    def test_final_edge_stretches_are_not_reused_for_every_frame(self):
        g=self.graph();origin=g.node_positions()
        result=SimpleNamespace(metadata={},positions_trajectory=[origin,origin*2],deformed_positions=origin*2,displacements=origin,edge_forces=None,edge_stretches=np.array([2.]))
        with self.assertRaisesRegex(NotImplementedError,'per-frame edge stretch'):
            render_trajectory_3d(g,sim_result=result,color_by='stretch')
        # Existing result-derived displacement route remains usable.
        figure=render_trajectory_3d(g,sim_result=result)
        self.assertEqual(figure.axes[-1].get_ylabel(),'Displacement')

if __name__=='__main__':unittest.main()

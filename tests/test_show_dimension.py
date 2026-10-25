"""Headless dimension-aware rendering and single-save regressions."""
import sys,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import fibernet as fn
from fibernet.viz.render import render_graph

def graph(dim):
 g=fn.StructureGraph(dimension=dim)
 a=g.add_node([0.,0.,0.]);b=g.add_node([1.,1.,2.] if dim==3 else [1.,1.,0.]);g.add_edge(a,b);return g
class ShowTests(unittest.TestCase):
 def setUp(self):
  self.output=tempfile.TemporaryDirectory();self.addCleanup(self.output.cleanup)
 def tearDown(self):plt.close('all')
 def test_3d_preserves_z_and_camera(self):
  f=fn.show(graph(3),theme='light',elevation=30,azimuth=42,title='3D: z is retained')
  self.assertIsInstance(f,Figure);a=f.axes[0];self.assertEqual(a.name,'3d');self.assertEqual(a.elev,30);self.assertEqual(a.azim,42)
  lines=[c for c in a.collections if isinstance(c,Line3DCollection)];self.assertEqual(len(lines),1)
  seg=np.asarray(lines[0]._segments3d);np.testing.assert_array_equal(seg[0],[[0.,0.,0.],[1.,1.,2.]])
  self.assertLess(a.get_zlim()[0],0.);self.assertGreater(a.get_zlim()[1],2.)
 def test_2d_existing_geometry_and_options(self):
  g=graph(2);actual=fn.show(g,theme='light',color_by='length',show_nodes=True);expected=render_graph(g,theme='light',color_by='length',show_nodes=True,dpi=150)
  self.assertEqual(actual.axes[0].name,'rectilinear');self.assertEqual(len(actual.axes[0].collections),len(expected.axes[0].collections))
  a=[c for c in actual.axes[0].collections if isinstance(c,LineCollection)][0];b=[c for c in expected.axes[0].collections if isinstance(c,LineCollection)][0]
  np.testing.assert_array_equal(a.get_segments(),b.get_segments());np.testing.assert_array_equal(actual.axes[0].get_xlim(),expected.axes[0].get_xlim())
 def test_nested_2d_and_3d_save_exactly_once(self):
  original=Figure.savefig;calls=[]
  def spy(fig,*args,**kwargs):calls.append((args,kwargs));return original(fig,*args,**kwargs)
  with patch.object(Figure,'savefig',spy):
   for dim in (2,3):
    dest=Path(self.output.name)/f'nested/{dim}d/network.png';self.assertFalse(dest.exists());f=fn.show(graph(dim),theme='light',save_path=dest,dpi=100)
    self.assertIsInstance(f,Figure);self.assertGreater(dest.stat().st_size,1000);self.assertEqual(dest.read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
  self.assertEqual(len(calls),2);self.assertTrue(all(k['dpi']==100 for a,k in calls))
 def test_incompatible_options_actionable_no_figure(self):
  before=plt.get_fignums()
  with self.assertRaisesRegex(TypeError,'render_graph_3d.*3D.*color_by.*explicit 2-D XY projection'):fn.show(graph(3),color_by='length')
  with self.assertRaisesRegex(TypeError,'render_graph.*2D.*elevation'):fn.show(graph(2),elevation=30)
  self.assertEqual(plt.get_fignums(),before)
 def test_explicit_2d_projection_still_available(self):
  f=fn.render_graph(graph(3),theme='light');self.assertEqual(f.axes[0].name,'rectilinear')
 def test_annotation_and_discoverability(self):
  self.assertIn('Figure',str(fn.show.__annotations__['return']));self.assertIn('Renderer-specific',fn.show.__doc__);self.assertIn('show',dir(fn))
if __name__=='__main__':unittest.main()

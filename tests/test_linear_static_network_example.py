"""Orchestration/analytical/saved-render tests; no physical solve is performed."""
import ast
import copy
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fibernet.examples import linear_static_network as example


def saved_fields():
    positions=[[.025*i,.025*j] for j in range(9) for i in range(9)]
    edges=[[9*j+i,9*j+i+1] for j in range(9) for i in range(8)]
    edges += [[9*j+i,9*(j+1)+i] for j in range(8) for i in range(9)]
    edges += [[9*j+i+1,9*(j+1)+i] for j in range(8) for i in range(8)]
    return dict(node_ids=list(range(81)),edge_ids=list(range(208)),positions_m=positions,
                endpoints_by_id=edges,displacement=[[0.,0.,0.] for _ in positions],
                loads=[[0.,0.,0.] for _ in positions],reactions=[[0.,0.,0.] for _ in positions],
                constraint_mask=[[i%9==0]*3 for i in range(81)],radii_m=[.0005]*208,
                total_axial_strain=[0.]*208,applicability={},E_Pa=1e9,profile='thermoelastic_2d_network/1',numerical_regularization=False,stress_free_temperature_K=300.,temperature_K=[300.]*81,expansion_alpha_per_K=[0.]*208,thermal_axial_strain=[0.]*208,
                units=dict(length='m',rotation='rad',force='N',moment='N*m',stress='Pa',temperature='K',alpha='1/K'))


class StaticExampleTests(unittest.TestCase):
    def test_module_import_has_no_top_level_generation_or_solver_import(self):
        tree=ast.parse(Path(example.__file__).read_text(encoding='utf-8'))
        names=[n.module for n in tree.body if isinstance(n,ast.ImportFrom)]
        self.assertEqual(names,['pathlib'])
        self.assertFalse(any(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) for n in tree.body))

    def test_existing_output_refused_before_work(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(example,'_run') as worker:
            path=Path(directory);(path/'keep').write_text('original')
            with self.assertRaises(FileExistsError):example.run(path)
            worker.assert_not_called();self.assertEqual((path/'keep').read_text(),'original')

    def test_failure_preserves_partial_output_and_original_exception(self):
        error=KeyboardInterrupt('original')
        def fail(out):
            (out/'partial').write_text('keep')
            (out/'failure.json').write_text('previous')
            raise error
        with tempfile.TemporaryDirectory() as directory, patch.object(example,'_run',side_effect=fail):
            out=Path(directory)/'new'
            with self.assertRaises(KeyboardInterrupt) as caught:example.run(out)
            self.assertIs(caught.exception,error)
            self.assertEqual((out/'failure.json').read_text(),'previous')
            self.assertEqual((out/'partial').read_text(),'keep')

    def test_exactly_two_inert_solver_calls_and_graph_load(self):
        from fibernet.gen.pattern import pattern_2d
        from fibernet.sim import thermoelastic
        fields=saved_fields()
        with tempfile.TemporaryDirectory() as directory, patch('fibernet.gen.pattern.pattern_2d',wraps=pattern_2d) as generation, patch.object(thermoelastic,'solve_thermoelastic_network_snapshot',return_value=SimpleNamespace(to_dict=lambda:copy.deepcopy(fields))) as solve, patch.object(example,'draw_from_saved') as draw:
            out=Path(directory)/'run';summary=example.run(out)
            self.assertEqual(generation.call_count,1);self.assertEqual(solve.call_count,2)
            first,second=solve.call_args_list
            self.assertEqual(second.kwargs['loads'],{n:[2*v for v in row] for n,row in first.kwargs['loads'].items()})
            self.assertEqual(first.kwargs['E_Pa'],1e9)
            self.assertEqual(set(first.kwargs['expansion_alpha_per_K'].values()),{0.})
            self.assertEqual(summary['solve_count'],2)
            self.assertTrue(all(c['pass'] for c in summary['checks']))
            draw.assert_called_once_with(out,out/'figures')
            self.assertEqual(json.loads((out/'graph.json').read_text())['dimension'],2)
            with (out/'nodal_F1.csv').open(newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual([int(row['node_id']) for row in rows],fields['node_ids'])
            self.assertEqual(len(rows),81)
            self.assertIn('reaction_moment_Nm',rows[0])

    def test_analytical_axial_energy(self):
        r=dict(node_ids=[7,31],edge_ids=[99],positions_m=[[0.,0.],[2.,0.]],endpoints_by_id=[[7,31]],radii_m=[.1],displacement=[[0.,0.,0.],[.01,0.,0.]])
        energy=example._edge_energies(r)[0]
        self.assertAlmostEqual(energy['axial_J'],.5*1e9*math.pi*.1**2*.01**2/2)
        self.assertEqual(energy['bending_J'],0.)

    def test_analytical_constant_curvature_and_rigid_motion(self):
        length=2.;curvature=.003
        r=dict(node_ids=[7,31],edge_ids=[99],positions_m=[[0.,0.],[length,0.]],endpoints_by_id=[[7,31]],radii_m=[.1],displacement=[[0.,0.,0.],[0.,curvature*length**2/2,curvature*length]])
        e=example._edge_energies(r)[0]
        self.assertAlmostEqual(e['bending_J'],.5*1e9*math.pi*.1**4/4*curvature**2*length)
        r['displacement']=[[.02,.03,.001],[.02,.03+.001*length,.001]]
        self.assertLess(abs(example._edge_energies(r)[0]['bending_J']),1e-24)

    def test_saved_renderer_no_generation_or_solve_and_no_overwrite(self):
        import builtins
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.startswith(('fibernet.gen','fibernet.sim','fibernet.ml')):
                raise AssertionError('renderer attempted physics import '+name)
            return original(name,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);raw=json.dumps(saved_fields());(path/'snapshot_F1.json').write_text(raw)
            with patch('builtins.__import__',side_effect=guarded):example.draw_from_saved(path,path/'figures')
            self.assertGreater((path/'figures/static_workflow_A4.png').stat().st_size,1000)
            self.assertEqual((path/'snapshot_F1.json').read_text(),raw)
            with self.assertRaises(FileExistsError):example.draw_from_saved(path,path/'figures')

    def test_invalid_saved_fields_fail_without_regeneration(self):
        r=saved_fields();r['units']['length']='mm'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/'snapshot_F1.json').write_text(json.dumps(r))
            with self.assertRaises(ValueError):example.draw_from_saved(path,path/'newfig')
            self.assertFalse((path/'newfig').exists())

    def test_cli_requires_explicit_new_plot_directory(self):
        with patch.object(example,'run') as solve, patch.object(example,'draw_from_saved',return_value={}) as draw:
            with self.assertRaises(SystemExit):example.main(['--output-dir','saved','--plot-only'])
            solve.assert_not_called();draw.assert_not_called()
            example.main(['--output-dir','saved','--plot-only','--figure-dir','new'])
            draw.assert_called_once_with('saved','new')

    def test_failed_render_has_no_pass_summary(self):
        from fibernet.sim import thermoelastic
        error=RuntimeError('render failure')
        with tempfile.TemporaryDirectory() as directory, patch.object(thermoelastic,'solve_thermoelastic_network_snapshot',return_value=SimpleNamespace(to_dict=saved_fields)), patch.object(example,'draw_from_saved',side_effect=error):
            out=Path(directory)/'new'
            with self.assertRaises(RuntimeError) as caught:example.run(out)
            self.assertIs(caught.exception,error)
            self.assertFalse((out/'summary.json').exists())
            self.assertTrue((out/'snapshot_F2.json').exists())
            self.assertTrue((out/'failure.json').exists())

    def test_renderer_closes_figure_and_restores_style_on_failure(self):
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        user_figure=plt.figure();user_axes=user_figure.add_subplot();line,=user_axes.plot([0,1],[1,2])
        error=OSError('save failure');before=list(plt.get_fignums());style=dict(matplotlib.rcParams);backend=matplotlib.get_backend()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/'snapshot_F1.json').write_text(json.dumps(saved_fields()))
            with patch.object(Figure,'savefig',side_effect=error):
                with self.assertRaises(OSError) as caught:example.draw_from_saved(path,path/'new')
            self.assertIs(caught.exception,error)
            self.assertEqual(plt.get_fignums(),before)
            self.assertEqual(dict(matplotlib.rcParams),style)
            self.assertEqual(matplotlib.get_backend(),backend)
            self.assertIs(user_axes.lines[0],line)
        plt.close(user_figure)

    def test_other_material_and_thermal_case_rejected(self):
        for name,value in [('E_Pa',2e9),('radii_m',[.001]*208),('temperature_K',[301.]*81),('profile','other'),('numerical_regularization',True)]:
            with self.subTest(field=name):
                fields=saved_fields();fields[name]=value
                with self.assertRaises(ValueError):example._validate_saved(fields)

    def test_render_does_not_change_user_backend_figures_or_style(self):
        import matplotlib
        import matplotlib.pyplot as plt
        fig=plt.figure();axis=fig.add_subplot();line,=axis.plot([0,1],[2,3])
        backend=matplotlib.get_backend();numbers=plt.get_fignums();style=dict(matplotlib.rcParams)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory);(path/'snapshot_F1.json').write_text(json.dumps(saved_fields()))
                with patch.object(matplotlib,'use',side_effect=AssertionError('global backend change')), patch.object(plt,'close',side_effect=AssertionError('user figure close')):
                    example.draw_from_saved(path,path/'new')
                self.assertEqual(matplotlib.get_backend(),backend)
                self.assertEqual(plt.get_fignums(),numbers)
                self.assertIs(axis.lines[0],line)
                self.assertEqual(dict(matplotlib.rcParams),style)
        finally:
            plt.close(fig)

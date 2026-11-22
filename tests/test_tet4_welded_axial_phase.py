"""Contract tests for the opt-in matched-grip Tet4 diagnostic."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'tet4_static_welded_junction.py'


class WeldedAxialPhaseTests(unittest.TestCase):
    def setUp(self):
        self.main = runpy.run_path(str(EXAMPLE))['main']
        self.scope = self.main.__globals__

    def cli(self, *args):
        out = io.StringIO()
        with patch.object(sys, 'argv', [str(EXAMPLE), *args]), contextlib.redirect_stdout(out):
            self.main()
        return out.getvalue()

    def reject_solve(self, *args, **kwargs):
        self.fail('The solver was called before it was permitted')

    def test_preflight_checks_both_tiers_and_does_not_solve(self):
        with patch.dict(self.scope, {'solve_static_tet4': self.reject_solve}):
            for tier, cap_nodes, cap_area, cap_triangles, cells in (
                ('quick', 21, .75, 24, (2064, 2160)),
                ('reference', 37, .84375, 48, (5424, 5352)),
            ):
                with self.subTest(tier=tier):
                    result = json.loads(self.cli('--tier', tier, '--axial-phase', '--preflight'))
                    self.assertEqual(result['cap_x_mm'], [-1.5, 1.5])
                    self.assertEqual(result['stroke_m'], 3.291840836e-9)
                    self.assertEqual(len(result['cases']), 2)
                    for case, phase, tets in zip(result['cases'], (0., .5), cells):
                        self.assertEqual(case['axial_phase_cells'], phase)
                        self.assertEqual(case['tetrahedra'], tets)
                        self.assertEqual(case['fixed_cap_nodes'], cap_nodes)
                        self.assertEqual(case['loaded_cap_nodes'], cap_nodes)
                        self.assertEqual(case['cap_area_mm2'], cap_area)
                        self.assertEqual(case['cap_triangles'], [cap_triangles]*2)
                        self.assertLessEqual(case['factor_entries'], case['factor_entry_limit'])
                        self.assertNotIn('actuator_force_x_N', case)

    def test_vtu_conflict_is_rejected_before_solving(self):
        err = io.StringIO()
        with patch.dict(self.scope, {'solve_static_tet4': self.reject_solve}), contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as exc:
                self.cli('--axial-phase', '--write-vtu-dir', '.')
        self.assertEqual(exc.exception.code, 2)
        self.assertIn('--axial-phase cannot be combined with --write-vtu-dir', err.getvalue())

    def test_help_distinguishes_preflight_modes(self):
        out = io.StringIO()
        with patch.object(sys, 'argv', [str(EXAMPLE), '--help']), contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as exc:
                self.main()
        self.assertEqual(exc.exception.code, 0)
        help_text = ' '.join(out.getvalue().split())
        self.assertIn('print the factor budget to stderr, then solve', help_text)
        self.assertIn('emit preflight JSON to stdout without solving', help_text)

    def test_nan_cannot_be_written_as_json(self):
        with patch.dict(self.scope, {'run_axial_phase': lambda *a, **kw: {'x': float('nan')}}):
            with self.assertRaisesRegex(ValueError, 'Out of range float values'):
                self.cli('--axial-phase')

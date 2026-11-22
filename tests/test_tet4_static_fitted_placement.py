"""Input, no-solve preflight, and refusal tests for the fitted example."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from examples import tet4_static_fitted_placement as example


def test_fitted_preflight_uses_public_budget_without_solving(monkeypatch, tmp_path, capsys):
    def no_solve(*args, **kwargs):
        pytest.fail('preflight must not solve')
    monkeypatch.setattr(example, 'solve_static_tet4', no_solve)
    monkeypatch.chdir(tmp_path)
    example.main(['--preflight'])
    report = json.loads(capsys.readouterr().out)
    assert report['solve_performed'] is False and 'results' not in report
    plan = report['tet4_preflight']
    assert (plan['points'], plan['tetrahedra']) == (694, 2241)
    assert plan['within_factor_budget'] and 0 < plan['factor_entries'] < plan['factor_entry_limit']
    assert report['fixed_cap_nodes'] == report['loaded_cap_nodes'] == 62
    assert report['total_force_N'] == .001
    a, b = (report['placements'][k] for k in ('junction', 'arm'))
    assert a['stiffened_volume_mm3'] == pytest.approx(.17959438003021647, abs=1e-8)
    assert b['stiffened_volume_mm3'] == pytest.approx(a['stiffened_volume_mm3'], abs=1e-8)
    assert (a['cells_with_nonzero_fraction'], b['cells_with_nonzero_fraction']) == (174, 300)


def test_fitted_rejects_mismatched_frozen_input(monkeypatch, tmp_path, capsys):
    changed = tmp_path / 'data.json'
    changed.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(example, 'FIXTURE', changed)
    with pytest.raises(SystemExit, match='2'):
        example.main([])
    assert 'frozen input SHA256' in capsys.readouterr().err


def test_fitted_rejects_failed_budget_before_default_solve(monkeypatch, capsys):
    monkeypatch.setattr(example, 'inspect_static_tet4_admission',
                        lambda *a, **k: SimpleNamespace(within_factor_budget=False))
    def no_solve(*args, **kwargs):
        pytest.fail('failed budget must prevent a solve')
    monkeypatch.setattr(example, 'solve_static_tet4', no_solve)
    with pytest.raises(SystemExit, match='2'):
        example.main([])
    assert 'factor-entry budget' in capsys.readouterr().err


@pytest.mark.parametrize(('axis', 'column'), [('y', 1), ('z', 2)])
def test_fitted_transverse_preflight_without_solving(monkeypatch, capsys, axis, column):
    def no_solve(*args, **kwargs):
        pytest.fail('preflight must not solve')
    monkeypatch.setattr(example, 'solve_static_tet4', no_solve)
    original_prepare = example._prepare

    def checked_prepare(*, load_axis):
        prepared = original_prepare(load_axis=load_axis)
        bundles, holds, loads, force, _ = prepared
        mesh = bundles['uniform']
        rows = {int(node_id): row for row, node_id in enumerate(mesh.node_ids)}
        held_rows = [rows[i] for i in holds]
        loaded_rows = [rows[i] for i in loads]
        np.testing.assert_array_equal(mesh.points[held_rows, column], -1.5)
        np.testing.assert_array_equal(mesh.points[loaded_rows, column], 1.5)
        np.testing.assert_array_equal(np.delete(force, column, axis=1), 0.)
        np.testing.assert_allclose(force[:, column].sum(), .001, rtol=0, atol=2e-18)
        return prepared

    monkeypatch.setattr(example, '_prepare', checked_prepare)
    argv = ['--load-axis', axis, '--preflight']
    example.main(argv)
    report = json.loads(capsys.readouterr().out)
    assert report['load_axis'] == axis
    assert report['solve_performed'] is False and 'results' not in report
    assert report['fixed_cap_nodes'] == report['loaded_cap_nodes'] == 15
    assert report['loaded_cap_area_m2'] == pytest.approx(7.347315653655915e-7)
    assert report['traction_Pa'] == pytest.approx(1361.0412933632638)
    assert report['total_force_N'] == .001
    assert report['tet4_preflight']['within_factor_budget']
    assert report['tet4_preflight']['free_dofs'] == 2037
    assert report['placements']['arm']['center_mm'] == [.82, 0., 0.]


def test_fitted_refuses_unknown_load_axis_before_preparing(monkeypatch, capsys):
    def no_prepare(*args, **kwargs):
        pytest.fail('invalid axis must not prepare or solve')
    monkeypatch.setattr(example, '_prepare', no_prepare)
    with pytest.raises(SystemExit, match='2'):
        example.main(['--load-axis=-x'])
    text = capsys.readouterr().err
    assert '--load-axis' in text and 'invalid choice' in text


def test_fitted_sensitivity_reuses_three_solutions_and_normalizes_to_uniform(monkeypatch, capsys):
    # Two springs in series, under unit force, with r=3: scaling the first
    # stiffness by 3 or the second by 2 gives the analytic energies below.
    case_energies = {'uniform': (2., 3.), 'junction': (2. / 3., 3.), 'arm': (2., 1.5)}
    fractions = {'junction': np.array([1., 0.]), 'arm': np.array([0., .5])}
    force = np.array([[1., 0., 0.]])
    node_ids, cell_ids = np.array([41]), np.array([57, 19])
    bundles = {name: SimpleNamespace(case=name, node_ids=node_ids, cell_ids=cell_ids,
                                     points=np.zeros((1, 3))) for name in case_energies}

    def prepare(*, load_axis, fractions_by_placement=None):
        if fractions_by_placement is not None:
            fractions_by_placement.update(fractions)
        return bundles, {}, {}, force, {'total_force_N': 1., 'fully_stiffened_E_Pa': 3.,
                                       'base_E_Pa': 1.}

    calls = []

    def solve(bundle, *, fixed_dofs_by_id, forces_by_id, return_cell_fields):
        assert return_cell_fields
        calls.append(bundle.case)
        energy = case_energies[bundle.case]
        total = sum(energy)
        stress, strain = np.zeros((2, 3, 3)), np.zeros((2, 3, 3))
        stress[:, 0, 0] = 2. * np.asarray(energy)
        strain[:, 0, 0] = 1.
        fields = SimpleNamespace(cell_ids=cell_ids, volume_m3=np.ones(2),
                                 stress_Pa=stress, strain=strain)
        return SimpleNamespace(cell_fields=fields, node_ids=node_ids,
                               displacement_m=2 * total * force, reaction_N=-force,
                               strain_energy_J=total, half_external_work_J=total)

    monkeypatch.setattr(example, '_prepare', prepare)
    monkeypatch.setattr(example, 'solve_static_tet4', solve)
    example.main([])
    plain = json.loads(capsys.readouterr().out)
    assert calls == list(case_energies)
    calls.clear()

    example.main(['--compliance-sensitivity'])
    with_sensitivity = json.loads(capsys.readouterr().out)
    slopes = with_sensitivity.pop('compliance_sensitivity')
    assert calls == list(case_energies)
    assert with_sensitivity == plain
    assert (slopes['initial_modulus_ratio'], slopes['final_modulus_ratio']) == (1., 3.)
    keys = ['initial_normalized_slope_per_r', 'average_normalized_slope_per_r',
            'final_normalized_slope_per_r']
    for name, expected in [('junction', (2. / 5., 2. / 15., 2. / 45.)),
                           ('arm', (3. / 10., 3. / 20., 3. / 40.))]:
        np.testing.assert_allclose([slopes['placements'][name][k] for k in keys],
                                   expected, rtol=1e-14, atol=0.)


def test_fitted_sensitivity_cannot_run_with_preflight(monkeypatch, capsys):
    monkeypatch.setattr(example, '_prepare', lambda **_: pytest.fail('must reject before prepare'))
    with pytest.raises(SystemExit, match='2'):
        example.main(['--preflight', '--compliance-sensitivity'])
    assert 'not allowed with argument --preflight' in capsys.readouterr().err


def test_fixed_cap_uses_same_read_only_weights_and_only_x_end_caps():
    bases = {}
    bundles, holds, loads, x, _ = example._prepare(force_bases=bases)
    assert list(bases) == ['x', 'y', 'z']
    mesh = bundles['uniform']
    row_by_id = {int(node_id): i for i, node_id in enumerate(mesh.node_ids)}
    np.testing.assert_array_equal(mesh.points[[row_by_id[i] for i in holds], 0], -1.5)
    for column, (faces_loads, force) in enumerate(bases.values()):
        assert set(faces_loads) == set(loads) and not force.flags.writeable
        np.testing.assert_array_equal(mesh.points[[row_by_id[i] for i in faces_loads], 0], 1.5)
        np.testing.assert_array_equal(force[:, column], x[:, 0])
        np.testing.assert_array_equal(np.delete(force, column, axis=1), 0.)


def _work_fixture(force_N=2.):
    uniform = np.array([[4., 1., 0.], [1., 4., 0.], [0., 0., 1.]])
    difference = np.array([[.6, .2, 0.], [.2, .6, 0.], [0., 0., .02]])
    center = .9 * uniform
    return {name: force_N**2 * C for name, C in
            zip(('uniform', 'junction', 'arm'), (uniform, center, center - difference))}


def test_fixed_cap_nonzero_cross_work_and_generalized_extrema(monkeypatch):
    works = _work_fixture()
    F0 = 2.
    bases = {key: (i, F0 * row[None, :])
             for i, (key, row) in enumerate(zip('xyz', np.eye(3)))}
    bundles = {name: SimpleNamespace(node_ids=np.array([29])) for name in works}
    called = []

    def solve(b, holds, j, force, force_N, *, displacements):
        assert b is bundles and force_N == F0
        called.append(j)
        for name, w in works.items():
            displacements[name] = w[:, j][None, :] / F0
        return {name: {'half_external_work_J': w[j, j] / 2} for name, w in works.items()}

    monkeypatch.setattr(example, '_solve', solve)
    report = example._fixed_cap_compliance(bundles, {}, bases, F0)
    assert called == [0, 1, 2]
    assert report['ranking'] == 'offset_lower_for_all_directions'
    # The two coupled xy eigenvector directions and the z axis give the extrema.
    np.testing.assert_allclose(report['offset_advantage_extrema_percentage_points'], [2., 16.])
    assert report['placements']['uniform']['work_matrix_J'][0][1] == 4.
    assert report['placements']['uniform']['compliance_matrix_m_per_N'][0][1] == 1.


def test_fixed_cap_bad_reciprocity_and_uncertified_ranking():
    work = _work_fixture()
    work['uniform'][0, 1] += .1
    with pytest.raises(ValueError, match='reciprocity'):
        example._fixed_cap_summary(work, 2.)
    work = _work_fixture()
    work['arm'][2, 2] = work['junction'][2, 2] - 1e-10
    report = example._fixed_cap_summary(work, 2.)
    assert 0 < report['offset_advantage_extrema_percentage_points'][0] < .000001
    assert report['ranking'] == 'inconclusive_at_numerical_margin'


@pytest.mark.parametrize('other', [['--preflight'], ['--compliance-sensitivity'], ['--load-axis', 'x']])
def test_fixed_cap_cli_rejects_other_modes_before_prepare(monkeypatch, capsys, other):
    monkeypatch.setattr(example, '_prepare', lambda **_: pytest.fail('must reject before prepare'))
    with pytest.raises(SystemExit, match='2'):
        example.main(['--fixed-cap-compliance', *other])
    assert ('cannot be combined' if other[0] == '--load-axis' else 'not allowed') in capsys.readouterr().err


@pytest.mark.parametrize('extra', [
    ['--preflight'], ['--compliance-sensitivity'], ['--fixed-cap-compliance'],
    ['--load-axis', 'x'], ['--load-axis', 'z'],
])
def test_stroke_cli_rejects_other_controls_before_preparing(monkeypatch, capsys, extra):
    monkeypatch.setattr(example, '_prepare', lambda **_: pytest.fail('must reject before prepare'))
    with pytest.raises(SystemExit) as e:
        example.main(['--fixed-cap-stroke-m=1e-9', *extra])
    assert e.value.code == 2


@pytest.mark.parametrize('stroke', ['0', '-0', 'nan', 'inf', '-inf'])
def test_stroke_requires_finite_nonzero_m(monkeypatch, capsys, stroke):
    monkeypatch.setattr(example, '_prepare', lambda **_: pytest.fail('must reject before prepare'))
    with pytest.raises(SystemExit) as e:
        example.main(['--fixed-cap-stroke-m='+stroke])
    assert e.value.code == 2 and 'finite nonzero metres' in capsys.readouterr().err


def test_negative_stroke_prepares_guided_x_cap_without_any_traction(monkeypatch, capsys):
    monkeypatch.setattr(example, '_cap_loads', lambda *_: pytest.fail('must not construct nodal traction'))
    seen = []
    real = example.inspect_static_tet4_admission
    def capture(b, *, fixed_dofs_by_id, prescribed_displacements_m_by_id):
        seen.append(prescribed_displacements_m_by_id)
        return real(b, fixed_dofs_by_id=fixed_dofs_by_id,
                    prescribed_displacements_m_by_id=prescribed_displacements_m_by_id)
    monkeypatch.setattr(example, 'inspect_static_tet4_admission', capture)
    def verify(bundles, holds, motion, stroke):
        b = bundles['uniform']
        p = dict(zip(map(int, b.node_ids), b.points))
        assert motion is seen[0] and len(motion) == len(holds) == 62
        assert all(p[n][0] == 1.5 and v == {'x': -1e-9} for n, v in motion.items())
        assert all(p[n][0] == -1.5 and v == ('x','y','z') for n, v in holds.items())
        assert stroke == -1e-9 and list(bundles) == ['uniform', 'junction', 'arm']
        return dict(probe=True)
    monkeypatch.setattr(example, '_fixed_cap_stroke', verify)
    example.main(['--fixed-cap-stroke-m=-1e-9'])
    out = json.loads(capsys.readouterr().out)
    assert out['solve_performed'] and out['results']=={'probe': True}
    assert out['stroke_m']==-1e-9 and out['tet4_preflight']['factor_entries']==1514430
    assert not {'loaded_cap_nodes','traction_Pa','total_force_N','fixed_cap_compliance'} & out.keys()

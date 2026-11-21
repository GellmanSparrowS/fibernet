"""Input, no-solve preflight, and refusal tests for the fitted example."""
import json
from types import SimpleNamespace

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

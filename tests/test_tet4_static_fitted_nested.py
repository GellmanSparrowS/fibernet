"""Check the frozen nested example without running mechanics."""
import json
from hashlib import sha256

import numpy as np
import pytest
from examples import tet4_static_fitted_nested as example
from examples.tet4_static_fitted_placement import _prepare


def test_nested_geometry_ids_and_materials():
    grip = {}
    old, holds, *_ = _prepare(stroke_control=(example.STROKE_M, grip))
    b = old['junction']
    x, t, ids, cells, parent, edges = example._nested(b, holds, grip)
    assert (len(x), len(t), len(edges)) == (766, 2534, 72)
    np.testing.assert_array_equal(ids[:694], b.node_ids)
    assert set(b.cell_ids) <= set(cells) and len(set(cells)) == len(cells)
    np.testing.assert_array_equal(x[:694], b.points)
    np.testing.assert_array_equal(x[694:], b.points[edges].mean(axis=1))
    assert sha256(t.astype('<i8').tobytes()).hexdigest() == '33ac1ee9dbc58022a91fb2a565a5b21f8c3eefb86c66a0b866034bf6b9e4abc7'
    assert len(example._faces(t)) == 1142
    fine_vol = example._volume_mm3(x, t)
    old_vol = example._volume_mm3(b.points, b.tetrahedra)
    for name in ('junction', 'arm'):
        old_E = np.array([old[name].materials[int(i)]['young_modulus_pa']
                          for i in old[name].material_ids])
        np.testing.assert_allclose(old_E[parent] @ fine_vol, old_E @ old_vol, rtol=1e-12)


def test_nested_default_reports_budget_without_solve(monkeypatch, capsys):
    monkeypatch.setattr(example, 'solve_static_tet4', lambda *a, **k: pytest.fail('unexpected solve'))
    example.main([])
    out = json.loads(capsys.readouterr().out)
    assert out['solve_performed'] is False and 'results' not in out
    assert out['preflight']['coarse']['tetrahedra'] == 2241
    assert out['preflight']['fine']['tetrahedra'] == 2534
    assert out['admitted_with_margin'] == all(
        plan['within_factor_budget'] and
        0 < max(plan['factor_entries'], plan['padded_factor_entries']) <= .8 * plan['factor_entry_limit']
        for plan in out['preflight'].values())


def test_nested_rejects_budget_before_opted_in_solves(monkeypatch):
    from dataclasses import replace
    good = example.inspect_static_tet4_admission
    calls = []
    def inspect(*a, **k):
        plan = good(*a, **k)
        calls.append(plan.points)
        return replace(plan, factor_entries=plan.factor_entry_limit) if plan.points == 766 else plan
    monkeypatch.setattr(example, 'inspect_static_tet4_admission', inspect)
    monkeypatch.setattr(example, 'solve_static_tet4', lambda *a, **k: pytest.fail('unexpected solve'))
    with pytest.raises(SystemExit) as exc:
        example.main(['--solve'])
    assert exc.value.code == 2
    assert calls == [694, 766]

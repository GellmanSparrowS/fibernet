"""Mechanical JSON scalar admission; no solver or optional runtime execution."""
import json
import sys

import pytest

from fibernet.sim.accelerated import SimResult
from fibernet.integrations import _harmonic_profile as hp

FIELDS = (
    'energy', 'time_seconds', 'max_force', 'max_stretch',
    'mean_stretch', 'std_stretch', 'max_displacement',
)


def payload():
    return dict(mode='spring', energy=1.25, n_nodes=0, n_edges=0)


def load_payload(tmp_path, data):
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return SimResult.load(path)


@pytest.mark.parametrize('field', FIELDS)
@pytest.mark.parametrize('value', [0, -1, 0.125, 2**64, -(2**64), 2**64 + 1,
                                  10**308, -(10**308), sys.float_info.max],
                         ids=['zero', 'negative', 'fraction', 'uint64-over',
                              'negative-large', 'not-exact-binary64', 'large-finite',
                              'negative-large-finite', 'max-float'])
def test_finite_scalars_keep_original_value_and_type(tmp_path, field, value):
    data = payload()
    data[field] = value
    result = load_payload(tmp_path, data)
    assert getattr(result, field) == value
    assert type(getattr(result, field)) is type(value)
    assert result.to_dict()[field] == value


@pytest.mark.parametrize('field', FIELDS)
@pytest.mark.parametrize('value', [10**400, -(10**400), True, False, None, '1',
                                  [], {}, float('nan'), float('inf'), -float('inf')],
                         ids=['overflow', 'negative-overflow', 'true', 'false',
                              'null', 'string', 'list', 'object', 'nan', 'inf', 'neg-inf'])
def test_invalid_scalars_name_field_before_backend_admission(tmp_path, monkeypatch, field, value):
    def backend_must_not_run(*args, **kwargs):
        raise AssertionError('backend result guard reached before scalar rejection')

    monkeypatch.setattr(hp, 'result_guard', backend_must_not_run)
    data = payload()
    data.update(backend='lammps')
    data[field] = value
    with pytest.raises(ValueError) as exc:
        load_payload(tmp_path, data)
    assert str(exc.value) == field + ' must be a finite real number'


def test_legacy_minimal_mechanical_summary_defaults(tmp_path):
    result = load_payload(tmp_path, payload())
    assert result.mode == 'spring'
    assert result.energy == 1.25
    assert result.n_nodes == result.n_edges == 0
    for field in FIELDS[1:]:
        assert getattr(result, field) == 0
    assert result.displacements is None
    assert result.edge_forces is None
    assert result.history == []

"""Saved-byte replay contracts, with every numerical provider disabled."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from benchmarks.saved_electrothermal_fixtures import fixture_bytes

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/saved_electrothermal_v1'
spec = importlib.util.spec_from_file_location(
    '_saved_electrothermal_replay', ROOT / 'benchmarks/saved_electrothermal_replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


@pytest.fixture(autouse=True)
def no_physics(monkeypatch):
    import fibernet.sim.electrothermal as et
    import fibernet.sim.linear_transport as lt
    import fibernet.sim.transient_thermal as tt
    import fibernet.sim.joule_heating as jh
    import fibernet.sim._thermoelastic as mechanics
    def forbidden(*args, **kwargs):
        pytest.fail('saved-byte replay must never execute a numerical provider')
    for module, name in ((et, 'solve_electrothermal'), (et, 'solve_linear_transport'),
                         (lt, 'solve_linear_transport'), (et, '_solve_thermal_interval'),
                         (tt, '_solve_thermal_interval'), (tt, '_solve_transient_thermal'),
                         (tt, 'solve_transient_thermal'), (jh, 'run_joule_heating'),
                         (mechanics, 'solve_thermoelastic_snapshot'),
                         (mechanics, 'solve_thermoelastic_network_snapshot')):
        monkeypatch.setattr(module, name, forbidden)


@pytest.mark.parametrize('name,nodes,edges,expected', [
    ('axial_chain.json', 3, 2, '137d1709e51a2402ea752d26fe8b87bf8a19bf93803d6cc78de070845cbe5bdc'),
    ('planar_network.json', 25, 56, 'affb33723da4d67e09f1b74c7b5cc20102f1f68600b92340be3c006591d6a895'),
])
def test_exact_saved_fixture_replay(name, nodes, edges, expected, tmp_path):
    path = tmp_path / name
    original = fixture_bytes(name)
    path.write_bytes(original)
    assert hashlib.sha256(original).hexdigest() == expected
    report = replay.replay_saved_electrothermal(path, frame_indices=[0, 5, 10], length_unit='m')
    assert report['saved_envelope_file_sha256'] == expected
    assert (report['node_count'], report['edge_count']) == (nodes, edges)
    assert report['source_completed'] is True
    assert report['source_frame_count'] == 11
    assert [f['time_s'] for f in report['frames']] == [0., .5, 1.]
    assert all(f['exact_binary64_transfer'] for f in report['frames'])
    assert path.read_bytes() == original


@pytest.mark.parametrize('indices', [[], [True], [-1], [0, 0], [1.], None, '0', [11]])
def test_replay_requires_existing_explicit_frames(indices, tmp_path):
    path = tmp_path / 'axial_chain.json'
    path.write_bytes(fixture_bytes('axial_chain.json'))
    with pytest.raises(ValueError):
        replay.replay_saved_electrothermal(path,
                                           frame_indices=indices, length_unit='m')


def test_cli_reports_only_saved_binding(capsys):
    assert replay.main(['--fixture', 'planar_network.json', '--frames', '0', '5', '10',
                        '--length-unit', 'm']) == 0
    out = json.loads(capsys.readouterr().out)
    assert out['schema'] == 'fibernet.saved_electrothermal_replay/1'
    assert 'mechanical' not in out
    assert 'duration' not in out


def test_replay_rejects_corrupt_saved_content(tmp_path):
    data = json.loads(fixture_bytes('planar_network.json'))
    data['source']['frames'][5]['temperature_K'][0] += 1.
    path = tmp_path / 'corrupt.json'
    path.write_text(json.dumps(data, allow_nan=False))
    with pytest.raises(ValueError):
        replay.replay_saved_electrothermal(path, frame_indices=[5], length_unit='m')


def test_native_module_matches_saved_fixture_implementation():
    raw = (ROOT / 'fibernet/sim/saved_electrothermal.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '113afa9374e9f63d6376055f139e0ffef14a82f071446790705bbd5924375386'


def test_replay_import_is_stdlib_only():
    import builtins
    from unittest.mock import patch
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.startswith(('numpy', 'scipy', 'torch', 'taichi', 'fibernet'))
        return original(name, *args, **kwargs)
    spec = importlib.util.spec_from_file_location(
        '_saved_replay_import_only', ROOT / 'benchmarks/saved_electrothermal_replay.py')
    module = importlib.util.module_from_spec(spec)
    with patch.object(builtins, '__import__', side_effect=guarded):
        spec.loader.exec_module(module)


def test_plain_json_cli_still_uses_original_reader(tmp_path, capsys):
    path = tmp_path / 'user-saved.json'
    path.write_bytes(fixture_bytes('axial_chain.json'))
    assert replay.main([str(path), '--frames', '0', '10', '--length-unit', 'm']) == 0
    report = json.loads(capsys.readouterr().out)
    assert 'bundled_fixture' not in report
    assert report['saved_envelope_file_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_generic_json_path_does_not_accept_encoded_fixture():
    with pytest.raises(ValueError):
        replay.replay_saved_electrothermal(
            FIXTURES / 'axial_chain.json.gz.b64', frame_indices=[0], length_unit='m')


@pytest.mark.parametrize('arguments', [
    ['--frames', '0', '--length-unit', 'm'],
    ['ordinary.json', '--fixture', 'axial_chain.json', '--frames', '0', '--length-unit', 'm'],
])
def test_cli_requires_exactly_one_input_route(arguments):
    with pytest.raises(SystemExit) as caught:
        replay.main(arguments)
    assert caught.value.code == 2


def test_bundled_replay_preserves_decoded_identity():
    report = replay.replay_bundled_fixture('planar_network.json', frame_indices=[0, 5, 10], length_unit='m')
    assert report['bundled_fixture'] == 'planar_network.json'
    assert report['saved_envelope_file_sha256'] == 'affb33723da4d67e09f1b74c7b5cc20102f1f68600b92340be3c006591d6a895'
    assert report['saved_canonical_content_sha256'] == '5ea01acf284957aed0eb3695de03529f6666b667d41263dd1e3a0eee3304b81d'


@pytest.mark.parametrize('raises', [False, True])
def test_bundled_temporary_json_is_removed_on_success_and_error(tmp_path, monkeypatch, raises):
    import tempfile
    original = tempfile.TemporaryDirectory
    observed = []
    def temporary(*args, **kwargs):
        return original(*args, dir=tmp_path, **kwargs)
    def read_owned(path, **kwargs):
        observed.append(path)
        assert path.is_file()
        assert path.read_bytes() == fixture_bytes('axial_chain.json')
        if raises:
            raise ValueError('synthetic reader failure')
        return {'saved_envelope_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(tempfile, 'TemporaryDirectory', temporary)
    monkeypatch.setattr(replay, 'replay_saved_electrothermal', read_owned)
    if raises:
        with pytest.raises(ValueError, match='synthetic reader failure'):
            replay.replay_bundled_fixture('axial_chain.json', frame_indices=[0], length_unit='m')
    else:
        assert replay.replay_bundled_fixture('axial_chain.json', frame_indices=[0], length_unit='m')['bundled_fixture'] == 'axial_chain.json'
    assert len(observed) == 1
    assert not observed[0].parent.exists()
    assert not list(tmp_path.iterdir())

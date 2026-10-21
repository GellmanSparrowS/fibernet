"""Named-column consumer contract; no physical-schema claims."""
import copy
import json
import pickle

import numpy as np
import pytest

import fibernet
from fibernet.easy import predict_named, train_model


class Recorder:
    def __init__(self, output=None):
        self.calls = []
        self.output = output

    def predict(self, matrix):
        self.calls.append(matrix.copy())
        return matrix[:, 0] + 10 * matrix[:, 1] if self.output is None else self.output


def artifact(**overrides):
    return dict(model=Recorder(), features=['b', 'a'], metrics={}, **overrides)


def test_root_exact_callable_and_json_ready_order():
    assert fibernet.predict_named is predict_named
    saved = artifact()
    rows = [{'a': 2, 'b': 3}, {'b': 4, 'a': 5}]
    before = copy.deepcopy(rows)
    result = predict_named(saved, rows)
    assert result == dict(predictions=[23., 54.], target=None, features=['b', 'a'],
                          validation='named_columns_only')
    json.dumps(result, allow_nan=False)
    assert rows == before
    result['features'].reverse()
    assert saved['features'] == ['b', 'a']
    assert predict_named(saved, rows[0])['predictions'] == [23.]
    assert predict_named(saved, tuple(rows))['predictions'] == [23., 54.]


@pytest.mark.parametrize('dtype', ['Int64', 'Float64', 'UInt64', 'boolean'])
def test_nullable_pandas_valid_and_missing(dtype):
    pd = pytest.importorskip('pandas')
    frame = pd.DataFrame({'a': pd.Series([True, False], dtype=dtype),
                          'b': pd.Series([False, True], dtype=dtype)})
    before = frame.copy(deep=True)
    saved = artifact()
    assert predict_named(saved, frame)['predictions'] == [10., 1.]
    pd.testing.assert_frame_equal(frame, before)
    frame.loc[0, 'a'] = pd.NA
    count = len(saved['model'].calls)
    with pytest.raises(ValueError):
        predict_named(saved, frame)
    assert len(saved['model'].calls) == count


def test_bool_numpy_scalars_and_ignored_heterogeneous_extras():
    saved = artifact()
    rows = [{'a': True, 'b': np.bool_(False), 'id': object()},
            {'b': np.int64(2), 'a': np.float32(3), 'label': None}]
    with pytest.raises(ValueError, match='extra'):
        predict_named(saved, rows)
    assert not saved['model'].calls
    assert predict_named(saved, rows, extra_columns='ignore')['predictions'] == [10., 32.]


@pytest.mark.parametrize('value', [None, np.nan, np.inf, -np.inf, '2', 1+0j, [], {}, 10**400])
def test_invalid_selected_values_never_predict(value):
    saved = artifact()
    with pytest.raises(ValueError):
        predict_named(saved, [{'a': 1, 'b': 2}, {'a': value, 'b': 3}])
    assert not saved['model'].calls


@pytest.mark.parametrize('data', [[], (), {}, [{'a': 1}], [{'a': 1, 'b': 2}, {'b': 1}],
                                  np.array([[1, 2]]), [1, 2], 'file.csv', b'bytes',
                                  {'a': 1, 'b': 2, '': 1}, {'a': 1, 'b': 2, 3: 4}])
def test_invalid_data_never_predict(data):
    saved = artifact()
    with pytest.raises(ValueError):
        predict_named(saved, data, extra_columns='ignore')
    assert not saved['model'].calls


@pytest.mark.parametrize('columns', [['a', 'a'], ['a', ''], ['a', 1]])
def test_invalid_dataframe_names(columns):
    pd = pytest.importorskip('pandas')
    saved = artifact()
    with pytest.raises(ValueError):
        predict_named(saved, pd.DataFrame([[1, 2]], columns=columns))
    assert not saved['model'].calls


def test_empty_dataframe_and_shuffled_full_table():
    pd = pytest.importorskip('pandas')
    saved = artifact()
    with pytest.raises(ValueError):
        predict_named(saved, pd.DataFrame(columns=['a', 'b']))
    frame = pd.DataFrame({'label': ['text', None], 'a': [2, 3], 'b': [4, 5]})
    result = predict_named(saved, frame, extra_columns='ignore')
    assert result['predictions'] == [24., 35.]
    np.testing.assert_array_equal(saved['model'].calls[0], frame[['b', 'a']].to_numpy())


@pytest.mark.parametrize('features', [None, (), [], ['a', 'a'], [''], [' '], [1], ['a', None]])
def test_invalid_stored_features(features):
    saved = artifact()
    saved['features'] = features
    with pytest.raises(ValueError):
        predict_named(saved, {'a': 1, 'b': 2})
    assert not saved['model'].calls


@pytest.mark.parametrize('version', [None, True, False, 0, 2, -1, 1., '1', [], np.int64(1)])
def test_invalid_version_never_predict(version):
    saved = artifact(artifact_version=version, target='y')
    with pytest.raises(ValueError, match='artifact_version'):
        predict_named(saved, {'a': 1, 'b': 2})
    assert not saved['model'].calls


@pytest.mark.parametrize('metadata', [{'artifact_version': 1}, {'artifact_version': 1, 'target': None},
                                    {'target': ''}, {'target': ' '}, {'target': 1}, {'target': 'a'}])
def test_invalid_target_never_predict(metadata):
    saved = artifact(**metadata)
    with pytest.raises(ValueError, match='target'):
        predict_named(saved, {'a': 1, 'b': 2})
    assert not saved['model'].calls


def test_legacy_and_versioned_target():
    for metadata in ({'target': 'y'}, {'target': 'y', 'artifact_version': 1}):
        assert predict_named(artifact(**metadata), {'a': 1, 'b': 2})['target'] == 'y'
    for bad in ('model.pkl', b'pickle', None, []):
        with pytest.raises(ValueError, match='dictionary'):
            predict_named(bad, {'a': 1})
    for policy in ('drop', None, [], True):
        saved = artifact()
        with pytest.raises(ValueError, match='extra_columns'):
            predict_named(saved, {'a': 1, 'b': 2}, extra_columns=policy)
        assert not saved['model'].calls
    saved = artifact()
    saved['model'] = object()
    with pytest.raises(ValueError, match='predict'):
        predict_named(saved, {'a': 1, 'b': 2})


@pytest.mark.parametrize('output', [1., [[1.]], [], [1., 2.], [np.nan], [np.inf],
                                   [1+0j], ['1'], [None]])
def test_reject_output_shapes_and_nonfinite(output):
    saved = artifact()
    saved['model'] = Recorder(output)
    with pytest.raises(ValueError):
        predict_named(saved, {'a': 1, 'b': 2})
    assert len(saved['model'].calls) == 1


@pytest.mark.parametrize('model_type', ['lr', 'rf', 'gb'])
def test_real_saved_artifacts_equivalence(tmp_path, model_type):
    pd = pytest.importorskip('pandas')
    pytest.importorskip('sklearn')
    frame = pd.DataFrame({'a': np.arange(16), 'b': np.arange(16)**2,
                          'custom_target': 3*np.arange(16)**2 + 2*np.arange(16)})
    csv = tmp_path / 'data.csv'
    frame.to_csv(csv, index=False)
    path = tmp_path / 'new.pkl'
    metrics = train_model(csv, target='custom_target', features=['b', 'a'],
                          model_type=model_type, save_path=path)
    # Only deserialize artifacts produced locally in this test.
    with path.open('rb') as handle:
        saved = pickle.load(handle)
    assert set(saved) == {'model', 'features', 'metrics', 'target', 'artifact_version'}
    assert saved['target'] == 'custom_target' and saved['artifact_version'] == 1
    assert saved['metrics'] == metrics
    expected = saved['model'].predict(frame[['b', 'a']].to_numpy())
    result = predict_named(saved, frame, extra_columns='ignore')
    np.testing.assert_array_equal(result['predictions'], expected)
    # Serialize the historical envelope and verify its trusted reload remains usable.
    old_path = tmp_path / 'old.pkl'
    with old_path.open('wb') as handle:
        pickle.dump({key: saved[key] for key in ('model', 'features', 'metrics')}, handle)
    with old_path.open('rb') as handle:
        legacy = pickle.load(handle)
    assert predict_named(legacy, frame, extra_columns='ignore')['target'] is None
    np.testing.assert_array_equal(predict_named(legacy, frame, extra_columns='ignore')['predictions'], expected)


def test_observable_duplicate_mapping_keys_reject_before_predict():
    from collections.abc import Mapping

    class DuplicateMapping(Mapping):
        def __iter__(self):
            return iter(['a', 'a', 'b'])

        def __len__(self):
            return 3

        def __getitem__(self, key):
            return 1

    saved = artifact()
    with pytest.raises(ValueError, match='unique'):
        predict_named(saved, DuplicateMapping())
    assert not saved['model'].calls


def test_mapping_prediction_without_pandas_import():
    import subprocess
    import sys
    from pathlib import Path

    code = '''
import sys
import importlib.abc
class BlockPandas(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pandas' or fullname.startswith('pandas.'):
            raise ImportError('pandas intentionally unavailable')
sys.meta_path.insert(0, BlockPandas())
import fibernet
from fibernet.easy import predict_named
assert fibernet.predict_named is predict_named
class Model:
    def predict(self, matrix):
        return matrix[:, 0] * 2
assert predict_named({'model': Model(), 'features': ['x']}, {'x': 3})['predictions'] == [6.0]
assert 'pandas' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True,
                   cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)


@pytest.mark.parametrize('output', [np.ma.array([123.], mask=[True]),
                                   np.ma.array([1., 123.], mask=[False, True])])
def test_masked_predictions_reject_before_array_conversion(output):
    saved = artifact()
    saved['model'] = Recorder(output)
    rows = [{'a': 1, 'b': 2}] * len(output)
    with pytest.raises(ValueError, match='masked predictions'):
        predict_named(saved, rows)
    assert len(saved['model'].calls) == 1


def test_unmasked_prediction_array_retains_vector_contract():
    saved = artifact()
    saved['model'] = Recorder(np.ma.array([123.], mask=[False]))
    assert predict_named(saved, {'a': 1, 'b': 2})['predictions'] == [123.]
    saved['model'] = Recorder(np.ma.array([[123.]], mask=False))
    with pytest.raises(ValueError, match='shape'):
        predict_named(saved, {'a': 1, 'b': 2})


@pytest.mark.parametrize('value', [np.ma.masked, np.ma.array(123., mask=True),
                                  np.ma.array([123.], mask=[True])[0]])
def test_masked_selected_input_rejects_before_predict(value):
    saved = artifact()
    with pytest.raises(ValueError, match='finite real'):
        predict_named(saved, [{'a': 1, 'b': 2}, {'a': value, 'b': 2}])
    assert not saved['model'].calls


def test_masked_dataframe_selected_input_rejects_before_predict():
    pd = pytest.importorskip('pandas')
    saved = artifact()
    frame = pd.DataFrame({'a': np.ma.array([123.], mask=[True]), 'b': [2.]})
    with pytest.raises(ValueError, match='finite real'):
        predict_named(saved, frame)
    assert not saved['model'].calls

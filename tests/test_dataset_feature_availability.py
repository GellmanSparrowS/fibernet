"""Synthetic checked selections; no model training or inference."""
import copy
import json

import numpy as np
import pytest

from fibernet.analysis.graph_features import GraphFeatureExtractor as C
from fibernet.ml.dataset import DatasetSample, FiberNetDataset


def contract():
    # Explicitly synthetic metadata for consumer validation, not claimed runtime evidence.
    statuses = {n: {'state': 'unknown', 'reason': 'outside_audited_scope'} for n in C.FEATURE_COLS}
    for n in C.AUDITED_NAMES:
        statuses[n] = {'state': 'computed', 'reason': 'input_recorded' if n in C.CONTACT_CONFIGURATION else 'evaluated'}
    return dict(schema=C.STATUS_SCHEMA, feature_schema=C.VALUE_SCHEMA, definition_id=C.DEFINITION_ID,
                geometry=C.GEOMETRY_SCHEMA, feature_names=list(C.FEATURE_COLS),
                audited_names=list(C.AUDITED_NAMES), parameters=dict(canvas_size=64, thick=5,
                    edge_margin=.12, top_k=3, area_thresh=.005, connectivity=8),
                dependencies={n: {'available': True, 'version': 'synthetic-test'}
                              for n in ('numpy', 'networkx', 'cv2', 'scipy')}, status=statuses)


def dataset(n=2):
    rows = [DatasetSample(f'row-{i}', 'synthetic', (1, 1), {},
                          {name: float(i) for name in C.FEATURE_COLS}, {'target': float(i)},
                          {'feature_contract': contract()}) for i in range(n)]
    return FiberNetDataset(rows, feature_names=list(C.FEATURE_COLS), target_names=['target'])


def metadata(ds, row=1):
    return ds.samples[row].metadata['feature_contract']


def test_selected_finite_subset_and_reordered_columns_without_mutation():
    ds = dataset()
    ds.samples[1].features['mesh_median_area'] = 7.
    before = list(ds.feature_names)
    np.testing.assert_equal(ds.feature_matrix_checked(['contact_nodes', 'mesh_median_area']), [[0., 0.], [1., 7.]])
    assert ds.feature_names == before
    # Unknown unselected structural values do not pretend to have been audited.
    assert ds.features.shape == (2, 94)
    with pytest.raises(ValueError, match='n_node.*unknown/outside_audited_scope'):
        ds.feature_matrix_checked()


@pytest.mark.parametrize('name,value', [('canvas_size', 128), ('thick', 7), ('edge_margin', .2),
                                       ('top_k', 4), ('area_thresh', .01), ('connectivity', 4)])
def test_all_six_parameter_alignment_even_small_subset(name, value):
    ds = dataset()
    metadata(ds)['parameters'][name] = value
    with pytest.raises(ValueError, match='row-1.*incompatible feature space'):
        ds.feature_matrix_checked(['contact_nodes'])


@pytest.mark.parametrize('field', ['schema', 'feature_schema', 'definition_id', 'geometry'])
def test_foreign_schema_or_geometry_rejected_even_all_rows_agree(field):
    ds = dataset()
    for row in ds.samples:
        row.metadata['feature_contract'][field] = 'foreign'
    with pytest.raises(ValueError, match='schema.unsupported'):
        ds.feature_matrix_checked(['contact_nodes'], allow_reasons=['domain.no_pores'])


@pytest.mark.parametrize('mutation', ['missing_status', 'extra_status', 'name_order', 'audited_names',
                                     'missing_parameter', 'extra_parameter', 'missing_schema',
                                     'bad_pair', 'upgrade_unknown', 'wrong_feature_reason', 'bad_dependency'])
def test_malformed_v1_not_repaired_to_unknown(mutation):
    ds = dataset()
    m = metadata(ds)
    if mutation == 'missing_status': del m['status']['mesh_cv_area']
    elif mutation == 'extra_status': m['status']['unexpected'] = {'state': 'computed', 'reason': 'evaluated'}
    elif mutation == 'name_order': m['feature_names'].reverse()
    elif mutation == 'audited_names': m['audited_names'].pop()
    elif mutation == 'missing_parameter': del m['parameters']['thick']
    elif mutation == 'extra_parameter': m['parameters']['other'] = 0
    elif mutation == 'missing_schema': del m['schema']
    elif mutation == 'bad_pair': m['status']['mesh_cv_area'] = {'state': 'computed', 'reason': 'domain.no_overlap_pairs'}
    elif mutation == 'upgrade_unknown': m['status']['n_node'] = {'state': 'computed', 'reason': 'evaluated'}
    elif mutation == 'wrong_feature_reason': m['status']['contact_nodes'] = {'state': 'not_applicable', 'reason': 'domain.no_pores'}
    elif mutation == 'bad_dependency': m['dependencies']['cv2']['available'] = 'yes'
    with pytest.raises(ValueError, match='row-1.*contract.malformed'):
        ds.feature_matrix_checked(['contact_nodes'], allow_reasons=['domain.no_pores'])


@pytest.mark.parametrize('name,value', [('canvas_size', True), ('canvas_size', 10), ('thick', 0),
    ('top_k', -1), ('connectivity', 6), ('edge_margin', np.nan), ('edge_margin', .5),
    ('area_thresh', np.inf), ('area_thresh', -1.), ('edge_margin', 10**1000), ('area_thresh', 10**1000)])
def test_checked_parameter_regime(name, value):
    ds = dataset()
    metadata(ds)['parameters'][name] = value
    with pytest.raises(ValueError, match='parameter'):
        ds.feature_matrix_checked(['contact_nodes'])


def test_legacy_no_contract_still_readable_but_checked_rejects():
    ds = dataset()
    ds.samples[0].metadata = {}
    assert ds.features.shape == (2, 94)
    with pytest.raises(ValueError, match='legacy.status_absent'):
        ds.feature_matrix_checked(['contact_nodes'])


@pytest.mark.parametrize('reason', ['outside_audited_scope', 'legacy.status_absent', 'dependency_missing.cv2',
                                  'dependency_missing.scipy', 'stat.nonfinite', 'render_empty_edge', 'invented'])
def test_policy_never_bypasses_unknown_dependency_or_nonfinite(reason):
    with pytest.raises(ValueError, match='forbidden or unknown'):
        dataset().feature_matrix_checked(['contact_nodes'], allow_reasons=[reason])


def test_eligible_finite_convention_requires_explicit_reason():
    ds = dataset()
    for row in ds.samples:
        row.metadata['feature_contract']['status']['contact_overlap_pair_size_mean'] = {
            'state': 'not_applicable', 'reason': 'domain.no_overlap_pairs'}
        row.features['contact_overlap_pair_size_mean'] = 0.
    with pytest.raises(ValueError, match='not_applicable/domain.no_overlap_pairs'):
        ds.feature_matrix_checked(['contact_overlap_pair_size_mean'])
    np.testing.assert_equal(ds.feature_matrix_checked(['contact_overlap_pair_size_mean'],
        allow_reasons=['domain.no_overlap_pairs']), [[0.], [0.]])


@pytest.mark.parametrize('value', [None, np.nan, np.inf, -np.inf, '0', complex(0, 1), [0]])
def test_policy_does_not_bypass_numeric_guard(value):
    ds = dataset()
    name = 'contact_overlap_pair_size_mean'
    for row in ds.samples:
        row.metadata['feature_contract']['status'][name] = {'state': 'not_applicable', 'reason': 'domain.no_overlap_pairs'}
    ds.samples[1].features[name] = value
    with pytest.raises(ValueError, match='row-1.*contact_overlap_pair_size_mean'):
        ds.feature_matrix_checked([name], allow_reasons=['domain.no_overlap_pairs'])


def test_missing_selected_value_not_defaulted():
    ds = dataset()
    del ds.samples[1].features['contact_nodes']
    with pytest.raises(ValueError, match='row-1.*contact_nodes.*missing'):
        ds.feature_matrix_checked(['contact_nodes'], allow_reasons=['domain.no_pores'])


@pytest.mark.parametrize('names', [['contact_nodes', 'contact_nodes'], ['not94'], 'contact_nodes', [None]])
def test_bad_selection(names):
    with pytest.raises(ValueError):
        dataset().feature_matrix_checked(names)


def test_status_unavailable_rejected_even_finite():
    ds = dataset()
    metadata(ds)['status']['mesh_cv_area'] = {'state': 'unavailable', 'reason': 'dependency_missing.cv2'}
    with pytest.raises(ValueError, match='unavailable/dependency_missing.cv2'):
        ds.feature_matrix_checked(['mesh_cv_area'])


def test_metadata_only_json_and_explicit_projection_save_load_split(tmp_path):
    ds = dataset(8)
    # Select an explicit reviewed finite projection, no mutation of original.
    projected = FiberNetDataset(ds.samples, ['contact_nodes', 'mesh_median_area'], ['target'])
    json.dumps(projected.samples[0].metadata['feature_contract'], allow_nan=False)
    projected.save(str(tmp_path / 'selected'))
    loaded = FiberNetDataset.load(str(tmp_path / 'selected'))
    np.testing.assert_equal(projected.feature_matrix_checked(), loaded.feature_matrix_checked())
    assert loaded.samples[0].metadata == projected.samples[0].metadata
    parts = loaded.split(ratios=(.5, .25, .25), random_state=11)
    assert sum(p.n_samples for p in parts) == 8
    for part in parts:
        assert part.feature_matrix_checked().shape == (part.n_samples, 2)
    assert ds.feature_names == list(C.FEATURE_COLS)


def test_new_full_nonfinite_value_cannot_be_saved_or_json_coerced(tmp_path):
    ds = dataset()
    ds.samples[0].features['pore_area_skew'] = np.nan
    ds.samples[0].metadata['feature_contract']['status']['pore_area_skew'] = {
        'state': 'fallback', 'reason': 'stat.nonfinite'}
    json.dumps(ds.samples[0].metadata['feature_contract'], allow_nan=False)
    with pytest.raises(ValueError):
        json.dumps(ds.samples[0].features, allow_nan=False)
    with pytest.raises(ValueError, match='finite'):
        ds.save(str(tmp_path / 'must_not_exist'))
    assert not (tmp_path / 'must_not_exist').exists()
    with pytest.raises(ValueError, match='fallback/stat.nonfinite'):
        ds.feature_matrix_checked(['pore_area_skew'])
    np.testing.assert_equal(ds.feature_matrix_checked(['contact_nodes']), [[0.], [1.]])


def test_old_alphabetic_default_unchanged():
    ds = dataset()
    old = FiberNetDataset(ds.samples)
    assert old.feature_names == sorted(C.FEATURE_COLS)

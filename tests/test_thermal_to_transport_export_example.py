"""Actual small installed-API workflow and persisted field/geometry checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('thermal_export_example',
    Path(__file__).parents[1]/'examples/thermal_to_transport_export.py')
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


def test_one_way_saved_field_workflow_and_equivalent_units(tmp_path):
    meshio = pytest.importorskip('meshio')
    outputs = {}
    for unit, scale in [('m', 1.), ('mm', .001)]:
        out = tmp_path/unit
        summary = example.run(out, length_unit=unit)
        assert summary['workflow'] == 'sequential_one_way'
        assert summary['feedback'] is False
        assert (summary['node_count'], summary['edge_count'], summary['stl_triangles']) == (16,24,1152)
        for name, digest in summary['files_sha256'].items():
            assert hashlib.sha256((out/name).read_bytes()).hexdigest() == digest
        mechanical_bundle = json.loads((out/'mechanical.json').read_text())
        assert mechanical_bundle['feedback_to_electrical_or_thermal'] is False
        mechanical = mechanical_bundle['mechanical']
        thermal = json.loads((out/'thermal.json').read_text())
        electrical = json.loads((out/'electrical.json').read_text())
        binding_bytes = (out/'transport_binding.json').read_bytes()
        binding = json.loads(binding_bytes)
        assert electrical['provenance']['model_metadata']['deformed_transport_preparation'] == {
            'schema': binding['schema'],
            'input_sha256': binding['input_sha256'],
            'report_file_sha256': hashlib.sha256(binding_bytes).hexdigest(),
        }
        assert thermal['edge_ids'] != mechanical['edge_ids']  # actual reordered rows
        temperature_by_id = dict(zip(thermal['node_ids'],thermal['temperature_K'][3]))
        np.testing.assert_array_equal(mechanical['temperature_K'],
                                      [temperature_by_id[n] for n in mechanical['node_ids']])
        mesh = meshio.read(out/'vtu/network.vtu')
        np.testing.assert_array_equal(mesh.point_data['node_id'], mechanical['node_ids'])
        np.testing.assert_array_equal(mesh.point_data['translation_m'][:,:2],
                                      np.array(mechanical['displacement'])[:,:2])
        assert np.all(mesh.point_data['translation_m'][:,2] == 0.)
        assert np.max(abs(np.array(mechanical['displacement'])[:,2])) > 0.  # nonzero rotations were not used as z
        projection = json.loads((out/'fiber_projection.json').read_text())
        assert not projection['omitted_edge_ids']
        assert set(r['edge_id'] for r in projection['edges']) == set(mechanical['edge_ids'])
        declaration = json.loads((out/'stl_geometry.json').read_text())
        assert declaration['coordinate_unit'] == unit
        assert declaration['stl_embeds_units_or_stable_ids'] is False
        raw = (out/'network.stl').read_bytes()
        count = struct.unpack_from('<I',raw,80)[0]
        dtype = np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
        vertices = np.frombuffer(raw,dtype=dtype,offset=84,count=count)['vertices'].astype(float)*scale
        outputs[unit] = mechanical,electrical,vertices
    for key in ('node_ids','edge_ids','displacement','edge_forces','reactions','sigma_total'):
        np.testing.assert_array_equal(outputs['m'][0][key], outputs['mm'][0][key])
    for key in ('voltage_V','current_A','conductance_S','edge_power_W'):
        np.testing.assert_allclose(outputs['m'][1][key],outputs['mm'][1][key],rtol=2e-12,atol=1e-14)
    np.testing.assert_allclose(outputs['m'][2],outputs['mm'][2],rtol=2e-7,atol=1e-12)


def test_existing_output_unchanged(tmp_path):
    pytest.importorskip('meshio')
    sentinel=tmp_path/'keep.txt'
    sentinel.write_text('existing output')
    with pytest.raises(FileExistsError):
        example.run(tmp_path)
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text() == 'existing output'


def test_invalid_unit_before_output(tmp_path):
    pytest.importorskip('meshio')
    out=tmp_path/'unwritten'
    with pytest.raises(ValueError, match='length_unit'):
        example.run(out,length_unit='cm')
    assert not out.exists()

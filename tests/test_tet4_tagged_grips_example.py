"""Executable smoke tests for imported named grips above the banded limits."""
import json
import numpy as np
import pytest

pytest.importorskip("meshio")
pytest.importorskip("pyamg")

from examples.tet4_tagged_grip_iterative import make_fixture, prepare, solve
from fibernet.io.tet4_meshio import read_tet4_vtu_bundle


def test_prepared_mesh_above_banded_cap_can_solve_and_save(tmp_path):
    source = tmp_path/"mesh.msh"
    make_fixture(source, refinement=2)
    path, grip_hash = prepare(source,tmp_path,length_unit="mm",axis="x",stroke_m=-1.5e-6)
    b = read_tet4_vtu_bundle(path)
    assert len(b.tetrahedra) == 8448
    assert set(b.material_ids) == {7,19}
    result = solve(path,tmp_path/"grips.json",tmp_path,prepared_grip_sha256=grip_hash)
    assert result["reaction_N"] < 0
    assert result["strain_energy_J"] > 0 and result["free_residual_inf_N"] < 1e-9
    assert json.loads((tmp_path/"summary.json").read_text()) == result
    with np.load(tmp_path/"nodal-fields.npz",allow_pickle=False) as fields:
        np.testing.assert_array_equal(fields["node_ids"],b.node_ids)
        assert min(fields["node_ids"])>2**53
        assert fields["reaction_N"].shape == fields["displacement_m"].shape == b.points.shape


def test_edited_grip_setting_is_refused_without_emitting_results(tmp_path):
    source = tmp_path/"mesh.msh"
    make_fixture(source, refinement=1)
    path, grip_hash = prepare(source,tmp_path,length_unit="mm",axis="x",stroke_m=1.5e-6)
    grip = tmp_path/"grips.json"
    payload = json.loads(grip.read_text())
    payload["axis"] = "y"
    grip.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="settings no longer match"):
        solve(path,grip,tmp_path,prepared_grip_sha256=grip_hash)
    assert not (tmp_path/"nodal-fields.npz").exists()


def test_sidecar_from_another_bundle_is_refused(tmp_path):
    source = tmp_path/"mesh.msh"
    first,second = tmp_path/"first",tmp_path/"second"
    first.mkdir()
    second.mkdir()
    make_fixture(source, refinement=1)
    _, grip_hash = prepare(source,first,length_unit="mm",axis="x",stroke_m=1.5e-6)
    path, _ = prepare(source,second,length_unit="mm",axis="x",stroke_m=1.5e-6,
                      moduli_Pa=(3e9,4e9))
    with pytest.raises(ValueError,match="sidecar does not match"):
        solve(path,first/"grips.json",second,prepared_grip_sha256=grip_hash)
    assert not (second/"nodal-fields.npz").exists()

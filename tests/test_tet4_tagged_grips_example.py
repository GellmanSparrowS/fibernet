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


def test_custom_external_gmsh_group_names_reproduce_same_inputs(tmp_path):
    source = tmp_path/"original.msh"
    make_fixture(source, refinement=1)
    materials = ("Soft phase", "Stiff phase")
    grips = ("Fixed end", "Driven end")
    display = source.read_text()
    for old, new in zip(("material_low","material_high","hold_face","pull_face"), materials + grips):
        display=display.replace('"'+old+'"','"'+new+'"')
    custom=tmp_path/"custom.msh"
    custom.write_text(display)
    original_out, custom_out = tmp_path/"baseline", tmp_path/"user"
    original_out.mkdir()
    custom_out.mkdir()
    kwargs=dict(length_unit="mm",axis="x",stroke_m=1.5e-6,moduli_Pa=(3e9,7e9))
    expected,x=prepare(source,original_out,**kwargs)
    with pytest.raises(ValueError,match="Missing named physical group"):
        prepare(custom,custom_out,**kwargs)
    with pytest.raises(ValueError,match="distinct names"):
        prepare(custom,custom_out,material_groups=(materials[0],materials[0]),**kwargs)
    actual,y=prepare(custom,custom_out,material_groups=materials,grip_groups=grips,**kwargs)
    a,b=read_tet4_vtu_bundle(expected),read_tet4_vtu_bundle(actual)
    assert a.materials == b.materials
    assert b.materials[7]["young_modulus_pa"]==3e9
    assert b.materials[19]["young_modulus_pa"]==7e9
    for key in ("node_ids","cell_ids","points","tetrahedra","material_ids"):
        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    wanted=solve(expected,original_out/"grips.json",original_out,prepared_grip_sha256=x)
    got=solve(actual,custom_out/"grips.json",custom_out,prepared_grip_sha256=y)
    for key in ("reaction_N","strain_energy_J"):
        np.testing.assert_allclose(wanted[key],got[key],rtol=2e-10)

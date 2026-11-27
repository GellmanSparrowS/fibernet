"""Executable smoke tests for imported named grips above the banded limits."""
import json
import numpy as np
import pytest

pytest.importorskip("meshio")
pytest.importorskip("pyamg")

from fibernet.examples.tet4_tagged_grip_iterative import make_fixture, prepare, solve
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


def test_display_vtu_reuses_solve_and_keeps_si_and_exact_ids(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import meshio
    from fibernet.examples import tet4_tagged_grip_iterative as example
    source=tmp_path/"mesh.msh"
    make_fixture(source,refinement=2)
    path,h=prepare(source,tmp_path,length_unit="mm",axis="x",stroke_m=-1.5e-6)
    original=example.solve_static_tet4
    calls=[]
    def record(*a,**k):
        calls.append(1)
        return original(*a,**k)
    monkeypatch.setattr(example,"solve_static_tet4",record)
    out=solve(path,tmp_path/"grips.json",tmp_path,prepared_grip_sha256=h,write_vtu=True)
    assert calls==[1] and out["deformation_vtu"]=="deformation.vtu"
    v=meshio.read(tmp_path/out["deformation_vtu"])
    b=read_tet4_vtu_bundle(path)
    np.testing.assert_array_equal(v.points,b.points*1e-3)
    np.testing.assert_array_equal(v.get_cells_type("tetra"),b.tetrahedra)
    keys=((v.point_data["node_id"],b.node_ids),
          (v.get_cell_data("cell_id","tetra"),b.cell_ids),
          (v.get_cell_data("material_id","tetra"),b.material_ids))
    for lhs,rhs in keys:
        assert lhs.dtype==np.uint64
        np.testing.assert_array_equal(lhs,rhs)
    assert min(v.point_data["node_id"])>2**53
    assert min(v.get_cell_data("cell_id","tetra"))>2**53
    with np.load(tmp_path/"nodal-fields.npz",allow_pickle=False) as fields:
        for k in ("displacement_m","reaction_N"):
            np.testing.assert_array_equal(v.point_data[k],fields[k])
        moved=np.flatnonzero(fields["displacement_m"][:,0] == -1.5e-6)
        assert len(moved)>0
        xyz=v.points[moved]
        warped=xyz+v.point_data["displacement_m"][moved]
        np.testing.assert_allclose(warped[:,0]-xyz[:,0],-1.5e-6,rtol=1e-12)
    result=SimpleNamespace(node_ids=v.point_data["node_id"],
                           displacement_m=v.point_data["displacement_m"],
                           reaction_N=v.point_data["reaction_N"])
    with pytest.raises(FileExistsError,match="already exists"):
        example._write_display_vtu(tmp_path/out["deformation_vtu"],b,result)
    assert calls==[1]


def test_single_physical_volume_matches_two_equal_moduli(tmp_path):
    import meshio
    source=tmp_path/"matched.msh"
    make_fixture(source,refinement=1)
    raw=meshio.read(source)
    raw.field_data.pop("material_low")
    raw.field_data.pop("material_high")
    raw.field_data["Solid bulk"]=np.array([55,3])
    for i,block in enumerate(raw.cells):
        if block.type=="tetra":
            raw.cell_data["gmsh:physical"][i]=np.full(len(block.data),55,dtype=int)
    one=tmp_path/"homogeneous.msh"
    meshio.write(one,raw,file_format="gmsh22",binary=False)
    x,y=tmp_path/"single",tmp_path/"double"
    x.mkdir(); y.mkdir()
    common=dict(length_unit="mm",axis="x",stroke_m=1.5e-6)
    with pytest.raises(ValueError,match="declared material tags"):
        prepare(source,x,material_groups=("material_low",),moduli_Pa=(2.e9,),**common)
    with pytest.raises(ValueError,match="Missing named physical group"):
        prepare(one,x,material_groups=("hold_face",),moduli_Pa=(2.e9,),**common)
    a,s1=prepare(one,x,material_groups=("Solid bulk",),moduli_Pa=(2.e9,),**common)
    b,s2=prepare(source,y,moduli_Pa=(2.e9,2.e9),**common)
    am=read_tet4_vtu_bundle(a); bm=read_tet4_vtu_bundle(b)
    assert set(am.material_ids)=={55} and set(bm.material_ids)=={7,19}
    for k in ("points","tetrahedra"):
        np.testing.assert_array_equal(getattr(am,k),getattr(bm,k))
    u=solve(a,x/"grips.json",x,prepared_grip_sha256=s1,write_vtu=True)
    v=solve(b,y/"grips.json",y,prepared_grip_sha256=s2)
    for k in ("reaction_N","strain_energy_J"):
        np.testing.assert_allclose(u[k],v[k],rtol=2e-10)
    with np.load(x/"nodal-fields.npz",allow_pickle=False) as a,np.load(y/"nodal-fields.npz",allow_pickle=False) as b:
        np.testing.assert_allclose(a["displacement_m"],b["displacement_m"],rtol=2e-10,atol=2e-17)
        np.testing.assert_allclose(a["reaction_N"],b["reaction_N"],rtol=2e-10,atol=2e-13)
    display=meshio.read(x/"deformation.vtu")
    assert set(display.get_cell_data("material_id","tetra"))=={55}


@pytest.mark.parametrize("flags",[
    ["--material-group","Solid"],
    ["--young-modulus-gpa","2"],
    ["--material-group","Solid","--young-modulus-gpa","2","--material-low-gpa","1"],
])
def test_single_material_flags_fail_before_creating_output(tmp_path,monkeypatch,flags):
    import sys
    from fibernet.examples import tet4_tagged_grip_iterative as example
    output=tmp_path/"must-not-exist"
    monkeypatch.setattr(sys,"argv",["tagged","--input","not-read.msh","--length-unit","mm",
                                  "--output",str(output),*flags])
    with pytest.raises(SystemExit,match="2"):
        example.main()
    assert not output.exists()


@pytest.mark.parametrize("tags,named_secondary",[
    ("55 56", True), ("56 55", True), ("55 56", False),
])
def test_single_material_gmsh41_refuses_secondary_volume_tag(tmp_path,tags,named_secondary):
    physical='4\n' if named_secondary else '3\n'
    physical+='2 31 "hold_face"\n2 37 "pull_face"\n3 55 "Solid bulk"\n'
    if named_secondary:
        physical+='3 56 "Other volume"\n'
    source=tmp_path/"overlap.msh"
    source.write_text(f"""$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
{physical}$EndPhysicalNames
$Entities
0 0 2 1
31 0 0 0 1 1 0 1 31 0
37 0 0 0 1 0 1 1 37 0
1 0 0 0 1 1 1 2 {tags} 0
$EndEntities
$Nodes
1 4 1 4
3 1 0 4
1
2
3
4
0 0 0
1 0 0
0 1 0
0 0 1
$EndNodes
$Elements
3 3 1 3
2 31 2 1
1 1 2 3
2 37 2 1
2 1 2 4
3 1 4 1
3 1 2 3 4
$EndElements
""")
    with pytest.raises(ValueError,match="exactly one declared physical tag"):
        prepare(source,tmp_path,length_unit="mm",axis="x",stroke_m=1.5e-6,
                material_groups=("Solid bulk",),moduli_Pa=(2e9,))
    assert not (tmp_path/"tagged-solid.zip").exists()


def test_material_json_changes_properties_without_changing_network(tmp_path, monkeypatch):
    import sys
    import meshio
    from fibernet.examples import tet4_tagged_grip_iterative as example

    mesh = tmp_path / "three-regions.msh"
    make_fixture(mesh, refinement=1)
    raw = meshio.read(mesh)
    raw.field_data["Bond region"] = np.array([21, 3])
    for i, block in enumerate(raw.cells):
        if block.type == "tetra":
            x = raw.points[block.data, 0].mean(axis=1)
            ids = np.select([x < 1, x > 2], [7, 19], default=21)
            raw.cell_data["gmsh:physical"][i] = ids
            raw.cell_data["gmsh:geometrical"][i] = ids
    meshio.write(mesh, raw, file_format="gmsh22", binary=False)
    materials = {
        "material_low": {"young_modulus_pa": 1e9, "poisson_ratio": .24},
        "Bond region": {"young_modulus_pa": 2e9, "poisson_ratio": .32},
        "material_high": {"young_modulus_pa": 5e9, "poisson_ratio": .39},
    }
    config = tmp_path / "materials.json"
    records = []
    for scale in (1, 3):
        output = tmp_path / f"run-{scale}"
        data = {name: dict(p, young_modulus_pa=p["young_modulus_pa"] * scale)
                for name, p in materials.items()}
        config.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", [
            "tagged", "--input", str(mesh), "--length-unit", "mm", "--output", str(output),
            "--materials-json", str(config), "--write-vtu",
        ])
        example.main()
        b = read_tet4_vtu_bundle(output / "tagged-solid.zip")
        for mid, name in enumerate(("material_low", "material_high", "Bond region")):
            assert b.materials[(7, 19, 21)[mid]] == data[name]
        vtu = meshio.read(output / "deformation.vtu")
        np.testing.assert_array_equal(vtu.get_cell_data("material_id", "tetra"), b.material_ids)
        summary = json.loads((output / "summary.json").read_text())
        with np.load(output / "nodal-fields.npz", allow_pickle=False) as fields:
            assert summary["free_residual_inf_N"] < 1e-9
            records.append((b, summary, dict(fields)))
    (a, one, f_one), (b, three, f_three) = records
    for key in ("points", "tetrahedra", "node_ids", "cell_ids", "material_ids"):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    np.testing.assert_allclose(f_one["displacement_m"], f_three["displacement_m"],
                               atol=1e-16, rtol=1e-9)
    np.testing.assert_allclose(f_one["reaction_N"] * 3, f_three["reaction_N"],
                               atol=1e-11, rtol=1e-9)
    for key in ("reaction_N", "strain_energy_J"):
        np.testing.assert_allclose(one[key] * 3, three[key], rtol=1e-9)


@pytest.mark.parametrize("payload,message", [
    ('{"V":{"young_modulus_pa":1e9,"poisson_ratio":0.3},"V":{}}', "Duplicate"),
    ('{"V":{"young_modulus_pa":1e9,"poisson_ratio":0.3,"density":1}}', "needs"),
    ('{"V":{"young_modulus_pa":true,"poisson_ratio":0.3}}', "Young modulus"),
    ('{"V":{"young_modulus_pa":1e9,"poisson_ratio":0.50}}', "Poisson ratio"),
    ('{"V":{"young_modulus_pa":1e9,"poisson_ratio":NaN}}', "Poisson ratio"),
    ('{}', "named material volumes"),
    (' ' * 16385, "at most 16 KiB"),
])
def test_bad_material_config_refused_before_creating_output(tmp_path, monkeypatch, capsys, payload, message):
    import sys
    from fibernet.examples import tet4_tagged_grip_iterative as example

    config = tmp_path / "materials.json"
    config.write_text(payload, encoding="utf-8")
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(sys, "argv", [
        "tagged", "--input", "not-read.msh", "--length-unit", "mm",
        "--output", str(output), "--materials-json", str(config),
    ])
    with pytest.raises(SystemExit, match="2"):
        example.main()
    assert message in capsys.readouterr().err
    assert not output.exists()


def test_material_json_refuses_mixed_legacy_flags(tmp_path, monkeypatch, capsys):
    import sys
    from fibernet.examples import tet4_tagged_grip_iterative as example

    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(sys, "argv", [
        "tagged", "--input", "not-read.msh", "--length-unit", "mm",
        "--output", str(output), "--materials-json", "not-read.json", "--poisson-ratio", ".3",
    ])
    with pytest.raises(SystemExit, match="2"):
        example.main()
    assert "do not mix" in capsys.readouterr().err
    assert not output.exists()


def test_multiple_materials_refuse_an_unnamed_secondary_gmsh41_tag(tmp_path):
    source = tmp_path / "ambiguous.msh"
    source.write_text("""$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
4
2 31 "hold_face"
2 37 "pull_face"
3 55 "Lower"
3 56 "Upper"
$EndPhysicalNames
$Entities
0 0 2 2
31 0 0 0 1 1 0 1 31 0
37 0 0 0 1 0 1 1 37 0
1 0 0 0 1 1 1 2 55 99 0
2 0 0 0 1 1 1 1 56 0
$EndEntities
$Nodes
1 5 1 5
3 1 0 5
1
2
3
4
5
0 0 0
1 0 0
0 1 0
0 0 1
1 1 1
$EndNodes
$Elements
4 4 1 4
2 31 2 1
1 1 2 3
2 37 2 1
2 1 2 4
3 1 4 1
3 1 2 3 4
3 2 4 1
4 5 2 3 4
$EndElements
""", encoding="utf-8")
    config = {name: {"young_modulus_pa": 2e9, "poisson_ratio": .31}
              for name in ("Lower", "Upper")}
    with pytest.raises(ValueError, match="exactly one declared physical tag"):
        prepare(source, tmp_path, length_unit="mm", axis="x", stroke_m=1.5e-6,
                material_properties=config)
    assert not (tmp_path / "tagged-solid.zip").exists()


def test_local_material_sensitivity_uses_only_the_one_grip_solve(tmp_path, monkeypatch):
    from fibernet.examples import tet4_tagged_grip_iterative as example
    src = tmp_path / "network.msh"
    make_fixture(src, refinement=1)
    common = dict(length_unit="mm", axis="x", stroke_m=-1.5e-6)
    config = dict(material_low={"young_modulus_pa": 1e9, "poisson_ratio": .28},
                  material_high={"young_modulus_pa": 2e9, "poisson_ratio": .34})
    a = tmp_path / "nominal"
    a.mkdir()
    path, grip = prepare(src, a, material_properties=config, **common)
    calls = []
    native_solver = example.solve_static_tet4

    def record(*a, **kwargs):
        calls.append(1)
        return native_solver(*a, **kwargs)

    monkeypatch.setattr(example, "solve_static_tet4", record)
    nominal = solve(path, a / "grips.json", a, prepared_grip_sha256=grip,
                    material_sensitivities=True)
    assert calls == [1]
    g = nominal["material_sensitivities"]
    assert set(g) == {"7", "19"}
    dlog = "d_reaction_d_ln_young_modulus_N"
    dnu = "d_reaction_d_poisson_ratio_N"
    assert all(g[mid][dlog] < 0 for mid in g)
    np.testing.assert_allclose(sum(v[dlog] for v in g.values()), nominal["reaction_N"], rtol=1e-9)
    for param, analytical in (("young_modulus_pa", dlog), ("poisson_ratio", dnu)):
        signed = []
        step = 1e-4
        for sign in (-1, 1):
            dst = tmp_path / f"{param}-{sign}"
            dst.mkdir()
            value = config["material_high"][param]
            changed = value + sign * step if analytical == dnu else value * np.exp(sign * step)
            props = dict(config, material_high=dict(config["material_high"], **{param: float(changed)}))
            path, grip = prepare(src, dst, material_properties=props, **common)
            result = solve(path, dst / "grips.json", dst, prepared_grip_sha256=grip)
            assert "material_sensitivities" not in result
            signed.append(result["reaction_N"])
        np.testing.assert_allclose((signed[1] - signed[0]) / (2 * step), g["19"][analytical], rtol=2e-6)

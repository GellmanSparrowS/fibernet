"""Validate the opt-in complete P2 field through native solve and VTU readers."""

from dataclasses import replace
from itertools import product
from pathlib import Path
import json
from tempfile import TemporaryDirectory

import meshio
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_TETRA
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

from fibernet.examples.tet4_periodic_pore import _voxel_bundle
from fibernet.sim.tet4_open_stokes import (
    solve_open_pressure_stokes,
    write_open_stokes_velocity_vtu,
)


def main():
    source = _voxel_bundle(product(range(12), range(4), range(4)))
    bundle = replace(
        source, node_ids=2**63 + 11*np.arange(len(source.points),dtype=np.uint64),
        cell_ids=2**63 + 71*np.arange(len(source.tetrahedra),dtype=np.uint64),
    )
    bc = dict(pressure_drop_Pa=.0035, viscosity_Pa_s=.001, solver="direct")
    short_ids = solve_open_pressure_stokes(source, include_quadratic_velocity=True, **bc).quadratic_velocity
    assert short_ids.source_node_ids.dtype == short_ids.source_cell_ids.dtype == np.uint64
    np.testing.assert_array_equal(short_ids.source_node_ids[:len(source.points),0],source.node_ids)
    np.testing.assert_array_equal(short_ids.source_cell_ids,source.cell_ids)
    normal = solve_open_pressure_stokes(bundle, **bc)
    result = solve_open_pressure_stokes(bundle, include_quadratic_velocity=True, **bc)
    assert normal.quadratic_velocity is None
    field = result.quadratic_velocity
    n = len(bundle.points)
    assert len(field.points_m) > n
    np.testing.assert_array_equal(normal.velocity_at_vertices_m_s,result.velocity_at_vertices_m_s)
    f = field.source_node_ids
    assert f.dtype == field.source_cell_ids.dtype == np.uint64
    np.testing.assert_array_equal(f[:n], np.repeat(bundle.node_ids[:,None],2,axis=1))
    assert (f[n:,0] < f[n:,1]).all()
    np.testing.assert_array_equal(field.source_cell_ids,bundle.cell_ids)
    assert (np.linalg.det(field.points_m[field.tetrahedra10[:,1:4]]-
                          field.points_m[field.tetrahedra10[:,:1]]) > 0).all()
    # Remap reversed points, cells and element orientation by raw source IDs.
    p=np.arange(n)[::-1]
    c=np.arange(len(bundle.tetrahedra))[::-1]
    reverse=replace(bundle, points=bundle.points[p],node_ids=bundle.node_ids[p],
                    tetrahedra=np.argsort(p)[bundle.tetrahedra[c]][:,[1,0,2,3]],cell_ids=bundle.cell_ids[c])
    v=solve_open_pressure_stokes(reverse, **bc, include_quadratic_velocity=True).quadratic_velocity
    g=v.source_node_ids
    a,b=np.lexsort((f[:,1],f[:,0])),np.lexsort((g[:,1],g[:,0]))
    np.testing.assert_array_equal(f[a],g[b])
    np.testing.assert_allclose(field.velocity_m_s[a],v.velocity_m_s[b],rtol=2e-8,atol=5e-14)
    with TemporaryDirectory() as directory:
        short_path=Path(directory)/"normal-ids.vtu"
        write_open_stokes_velocity_vtu(short_path,short_ids)
        small=meshio.read(short_path)
        np.testing.assert_array_equal(small.point_data["source_node_ids"],short_ids.source_node_ids)
        np.testing.assert_array_equal(small.get_cell_data("source_cell_id","tetra10"),source.cell_ids)
        path=Path(directory)/"pressure-flow.vtu"
        write_open_stokes_velocity_vtu(path,field)
        original=path.read_bytes()
        try:
            write_open_stokes_velocity_vtu(path,field)
        except FileExistsError:
            assert path.read_bytes() == original
        else:
            raise AssertionError("Open Stokes writer must not overwrite a prior VTU")
        sample=meshio.read(path)
        np.testing.assert_array_equal(sample.get_cells_type("tetra10"),field.tetrahedra10)
        np.testing.assert_array_equal(sample.point_data["source_node_ids"],f)
        np.testing.assert_array_equal(sample.get_cell_data("source_cell_id","tetra10"),field.source_cell_ids)
        reader=vtkXMLUnstructuredGridReader(); reader.SetFileName(str(path)); reader.Update()
        vtk=reader.GetOutput()
        assert vtk.GetCellType(0)==VTK_QUADRATIC_TETRA
        np.testing.assert_array_equal(vtk_to_numpy(vtk.GetPointData().GetArray("source_node_ids")),f)
        np.testing.assert_array_equal(vtk_to_numpy(vtk.GetPointData().GetArray("velocity_m_s")),field.velocity_m_s)
        np.testing.assert_array_equal(vtk_to_numpy(vtk.GetCellData().GetArray("source_cell_id")),field.source_cell_ids)
    print(json.dumps({"cells":len(bundle.tetrahedra),"vtk_quadratic_points":len(field.points_m),
                      "original_uint64_ids":True,"remapped_equivalence":True}))


if __name__ == "__main__":
    main()

"""Exercise installable-library solid exports without FiberScope imports."""
import struct
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest

from fibernet.gen.manufacturing import PlanarManufacturingConfig, compile_planar
from fibernet.gen.solid_export import (PrintSettings, build_solid,
                                       export_solid, export_route)


def test_library_stl_3mf_and_route(tmp_path):
    pytest.importorskip('manifold3d')
    network = compile_planar(PlanarManufacturingConfig(
        unit='square', grid_x=1, grid_y=1, n_pts_per_side=0))
    solid = build_solid(network, PrintSettings(width=30, depth=30,
                                                height=1, diameter=1))
    assert solid.validate() > 0
    assert np.allclose(np.ptp(solid.vertices, axis=0), [30, 30, 1], atol=1e-4)
    stl = tmp_path / 'network.stl'
    package = tmp_path / 'network.3mf'
    route = tmp_path / 'route.csv'
    export_solid(stl, solid)
    export_solid(package, solid)
    export_route(route, network, solid.centers)
    raw = stl.read_bytes()
    assert struct.unpack('<I', raw[80:84])[0] == len(solid.faces)
    assert len(raw) == 84 + 50 * len(solid.faces)
    with zipfile.ZipFile(package) as archive:
        model = ET.fromstring(archive.read('3D/3dmodel.model'))
        assert model.attrib['unit'] == 'millimeter'
        assert len(model.findall('.//{*}triangle')) == len(solid.faces)
    assert len(route.read_text().splitlines()) == len(network.edges) + 2

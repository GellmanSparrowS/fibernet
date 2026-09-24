"""Generate a bounded fiber solid and export STL, 3MF, and route CSV.

Run: python -m examples.solid_export_workflow --output-dir demo_solid
Install the optional manufacturing extra before running this example.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zipfile

from fibernet.gen.manufacturing import PlanarManufacturingConfig, compile_planar
from fibernet.gen.solid_export import (PrintSettings, build_solid,
                                       export_route, export_solid)


class SolidExportWorkflow:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        network = compile_planar(PlanarManufacturingConfig(
            unit='square', grid_x=1, grid_y=1, n_pts_per_side=0))
        settings = PrintSettings(width=30, depth=30, height=1, diameter=1)
        solid = build_solid(network, settings)
        stl = self.output_dir / 'network.stl'
        package = self.output_dir / 'network.3mf'
        route = self.output_dir / 'route.csv'
        export_solid(stl, solid)
        export_solid(package, solid)
        export_route(route, network, solid.centers)
        with stl.open('rb') as stream:
            stream.seek(80)
            triangles = struct.unpack('<I', stream.read(4))[0]
        with zipfile.ZipFile(package) as archive:
            model = ET.fromstring(archive.read('3D/3dmodel.model'))
            if model.attrib.get('unit') != 'millimeter':
                raise AssertionError('3MF unit is not millimeter')
        if triangles != len(solid.faces):
            raise AssertionError('STL triangle count changed')
        summary = {
            'vertices': len(solid.vertices),
            'triangles': triangles,
            'volume_mm3': solid.validate(),
            'route_edges': len(network.route_edges),
            'unit': 'millimeter',
            'scope': 'mesh and route output; printer-specific feasibility unverified',
        }
        destination = self.output_dir / 'summary.json'
        temporary = destination.with_suffix('.tmp.json')
        try:
            temporary.write_text(json.dumps(summary, indent=2) + '\n',
                                 encoding='utf-8')
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        print('[solid_export_workflow] STL, 3MF and route CSV saved')
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='demo_solid')
    args = parser.parse_args()
    SolidExportWorkflow(args.output_dir).run()

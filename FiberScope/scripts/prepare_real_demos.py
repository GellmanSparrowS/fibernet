"""Download CC0 Khronos geometry and make bounded OBJ demos.

Run: python scripts/prepare_real_demos.py --triangles 500
Requires trimesh and fast-simplification (development only). Downloads are
bounded and cached; completed assets and provenance are atomic.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fslab.search_algorithms import atomic_json
from fslab.storage import atomic_replace
from fslab.surface_mapping import load_obj, reduce_triangles


class DemoBuilder:
    models = ('FlightHelmet', 'SheenChair', 'WaterBottle')
    base = 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/'

    def __init__(self, triangles=700):
        self.triangles = triangles
        self.cache = Path('_cache/model_sources23')
        self.cache.mkdir(parents=True, exist_ok=True)
        self.output = Path('assets/obj')
        self.licenses = Path('assets/licenses/model_sources')
        self.licenses.mkdir(parents=True, exist_ok=True)

    def download(self, url, path):
        if path.exists():
            return path.read_bytes()
        request = urllib.request.Request(url, headers={'User-Agent': 'FiberScope-demo-preparation/2.3'})
        temporary = path.with_suffix(path.suffix + '.tmp')
        source = None
        for attempt in range(4):
            try:
                source = urllib.request.urlopen(request, timeout=45)
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(1.)
        with source, temporary.open('wb') as dest:
            total = 0
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > 64 * 1024 * 1024:
                    raise MemoryError('source download exceeds 64 MB')
                dest.write(block)
        atomic_replace(temporary, path)
        return path.read_bytes()

    def run(self):
        import trimesh
        import fast_simplification
        records = []
        for name in self.models:
            folder = self.cache / name
            folder.mkdir(exist_ok=True)
            url = self.base + name + '/glTF/' + name + '.gltf'
            raw = self.download(url, folder / (name + '.gltf'))
            license_data = self.download(self.base + name + '/README.md', self.licenses / (name + '.md'))
            assert b'Creative Commons Zero' in license_data, 'review model license before importing'
            document = json.loads(raw)
            for buffer in document['buffers']:
                uri = buffer['uri']
                if Path(uri).name != uri:
                    raise ValueError('unexpected buffer path')
                self.download(self.base + name + '/glTF/' + uri, folder / uri)
            for key in ('images', 'textures', 'materials', 'samplers', 'extensionsUsed', 'extensionsRequired', 'extensions'):
                document.pop(key, None)
            for mesh in document['meshes']:
                for primitive in mesh['primitives']:
                    primitive.pop('material', None)
                    primitive.pop('extensions', None)
            scene = trimesh.load(io.BytesIO(json.dumps(document).encode()), file_type='gltf',
                                 resolver=trimesh.resolvers.FilePathResolver(str(folder)), process=False)
            mesh = scene.dump(concatenate=True)
            mesh.merge_vertices(merge_tex=True, merge_norm=True)
            original = len(mesh.faces)
            # Union thin accessories into a watertight display envelope before
            # strong decimation; this retains a recognizable continuous shell.
            pitch = float(np.ptp(mesh.vertices, axis=0).max()) / 64.
            voxels = mesh.voxelized(pitch).fill()
            if voxels.matrix.size > 1000000:
                raise MemoryError('voxel envelope exceeds one million cells')
            from skimage.measure import marching_cubes
            vertices, faces, _, _ = marching_cubes(np.pad(voxels.matrix.astype(np.float32), 1), .5)
            vertices = trimesh.transform_points(vertices-1, voxels.transform)
            vertices, faces = reduce_triangles(vertices, faces, target=self.triangles)
            reduced = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            triangle_path = folder / 'reduced_triangles.obj'
            reduced.export(str(triangle_path))
            vertices, quads = load_obj(triangle_path)
            path = self.output / ('real_' + name.lower() + '.obj')
            temporary = path.with_suffix('.obj.tmp')
            with temporary.open('w', encoding='utf-8') as stream:
                stream.write('# CC0 Khronos ' + name + '; reduced and converted to conforming quads\n')
                for vertex in vertices:
                    stream.write('v %.9g %.9g %.9g\n' % tuple(vertex))
                for face in quads:
                    stream.write('f %d %d %d %d\n' % tuple(i+1 for i in face))
            atomic_replace(temporary, path)
            records.append(dict(name=name, source=url, license='CC0-1.0',
                                license_url=self.base + name + '/README.md',
                                source_sha256=hashlib.sha256(raw).hexdigest(),
                                original_triangles=original, reduced_triangles=len(faces), quads=len(quads),
                                output=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                method='64-cell voxel union envelope, quadric edge collapse, shared edge midpoint and face-center corner quads'))
            print(name, original, 'triangles ->', len(quads), 'quads', flush=True)
        atomic_json(self.output / 'real_models.json', records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--triangles', type=int, default=700)
    args = parser.parse_args()
    DemoBuilder(args.triangles).run()

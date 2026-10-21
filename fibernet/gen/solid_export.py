"""Millimetre fiber solids, bounded meshing, atomic STL/3MF and route exports."""
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import struct
import zipfile
import numpy as np
import os
import stat
import tempfile
import warnings
import sys
from contextlib import contextmanager

atomic_replace = os.replace


def _export_file_mode(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('export destination must be a regular file, not a symlink or nonregular target')
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o7000:
        raise ValueError('export destination has unsupported special mode bits')
    return mode


@contextmanager
def _owned_export_file(path):
    """Stage only in an exclusively created sibling; never touch path+'.tmp'."""
    mode = _export_file_mode(path)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.export-', dir=path.parent)
    temporary = Path(name)
    try:
        os.close(fd)
        yield temporary
        os.chmod(temporary, 0o600 if mode is None else mode)
        atomic_replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except Exception as exc:
            # A warning filter must not replace a write exception or undo success.
            try:
                warnings.warn(f'Export cleanup failed for owned temporary {temporary}: {exc!r}; '
                              'the export outcome is unchanged', RuntimeWarning, stacklevel=3)
            except Exception:
                pass


@dataclass
class PrintSettings:
    width: float = 100.
    depth: float = 100.
    height: float = 2.
    diameter: float = 2.
    curved: bool = False
    max_voxels: int = 8000000
    max_height: float = 250.
    up_axis: str = 'z'

    def validate(self):
        if self.up_axis not in ('y','z'):
            raise ValueError('up_axis must be y or z')
        if not all(np.isfinite(v) and .1 <= v <= 1000 for v in (self.width, self.depth, self.height, self.diameter)):
            raise ValueError('dimensions must be finite millimetres in 0.1..1000')
        if self.curved and not (np.isfinite(self.max_height) and self.diameter < self.max_height <= 250.):
            raise ValueError('curved height limit must exceed diameter and be at most 250 mm')
        if self.width <= self.diameter or self.depth <= self.diameter:
            raise ValueError('width and depth must exceed the fiber diameter')
        if not 100000 <= self.max_voxels <= 16000000:
            raise ValueError('voxel budget must be 100000..16000000')

    def coordinates(self, network):
        self.validate()
        points = network.positions.copy()
        if self.curved and self.up_axis == 'y':
            points = points[:,[0,2,1]] * np.array([1.,-1.,1.])
        low, span = points.min(0), np.ptp(points, axis=0)
        points -= low
        if not self.curved and np.any(span[:2] < 1e-9):
            raise ValueError('planar fabrication requires nonzero width and depth; edit the cell geometry')
        if self.curved:
            scale = min((self.width-self.diameter)/max(span[0], 1e-9),
                        (self.depth-self.diameter)/max(span[1], 1e-9),
                        (self.max_height-self.diameter)/max(span[2],1e-9),
                        (250.-self.diameter)/max(float(span.max()),1e-9))
            points *= scale
            points += self.diameter/2
        else:
            for axis, extent in enumerate((self.width, self.depth)):
                if span[axis] > 1e-9:
                    points[:, axis] *= (extent-self.diameter)/span[axis]
                points[:, axis] += self.diameter/2
            points[:, 2] = self.height/2
        return points


def _coordinate_layout(dtype):
    """Qualify stored real coordinates without silently narrowing their values."""
    dtype = np.dtype(dtype)
    if dtype.kind in 'iu' and dtype.itemsize in (1, 2, 4, 8):
        return None
    if dtype.kind == 'f' and dtype.itemsize in (2, 4, 8):
        info = np.finfo(dtype)
        layout = {2: (5, 10, 15), 4: (8, 23, 127), 8: (11, 52, 1023)}[dtype.itemsize]
        exponent_bits, fraction_bits, bias = layout
        byteorder = sys.byteorder if dtype.byteorder in ('=', '|') else (
            'little' if dtype.byteorder == '<' else 'big')
        # Names such as longdouble do not identify a storage representation.
        if ((info.bits, info.iexp, info.nmant, info.maxexp, info.minexp) ==
                (dtype.itemsize * 8, exponent_bits, fraction_bits, bias + 1, 1 - bias)
                and int.from_bytes(np.array(1, dtype=dtype).tobytes(), byteorder) == bias << fraction_bits
                and int.from_bytes(np.array(-0., dtype=dtype).tobytes(), byteorder) ==
                1 << (dtype.itemsize * 8 - 1)):
            return layout
    raise ValueError('solid coordinates require integer storage up to 64 bits or '
                     'qualified IEEE binary16, binary32 or binary64; unsupported dtype: '
                     + str(dtype))


def _first_zero_area_face(vertices, faces):
    """Return the first exactly collinear face in the stored coordinate values.

    Only nine scalars are decoded per face. Python integer subtraction and cross
    products avoid floating overflow, underflow and FTZ/DAZ arithmetic. This does
    not certify the later floating volume calculation or pre-rounded geometry.
    """
    dtype = vertices.dtype
    layout = _coordinate_layout(dtype)
    width = dtype.itemsize
    byteorder = sys.byteorder if dtype.byteorder in ('=', '|') else (
        'little' if dtype.byteorder == '<' else 'big')
    for row, face in enumerate(faces):
        raw = vertices[face].tobytes(order='C')
        if layout is None:
            coordinates = [int.from_bytes(raw[i:i + width], byteorder, signed=dtype.kind == 'i')
                           for i in range(0, 9 * width, width)]
        else:
            exponent_bits, fraction_bits, bias = layout
            values = []
            for i in range(0, 9 * width, width):
                bits = int.from_bytes(raw[i:i + width], byteorder)
                exponent = (bits >> fraction_bits) & ((1 << exponent_bits) - 1)
                coefficient = bits & ((1 << fraction_bits) - 1)
                if exponent == (1 << exponent_bits) - 1:
                    raise ValueError('solid vertices must be finite (N, 3) coordinates')
                if exponent:
                    coefficient |= 1 << fraction_bits
                if bits >> (width * 8 - 1):
                    coefficient = -coefficient
                values.append((coefficient, (exponent or 1) - bias - fraction_bits))
            nonzero_exponents = [exponent for coefficient, exponent in values if coefficient]
            if not nonzero_exponents:
                return row
            minimum = min(nonzero_exponents)
            coordinates = [coefficient << (exponent - minimum) if coefficient else 0
                           for coefficient, exponent in values]
        u = [coordinates[i + 3] - coordinates[i] for i in range(3)]
        v = [coordinates[i + 6] - coordinates[i] for i in range(3)]
        if (u[1] * v[2] == u[2] * v[1] and
                u[2] * v[0] == u[0] * v[2] and
                u[0] * v[1] == u[1] * v[0]):
            return row
    return None


@dataclass
class FiberSolid:
    vertices: np.ndarray
    faces: np.ndarray
    centers: np.ndarray
    resolution: float
    settings: PrintSettings

    def validate(self):
        """Validate stored triangles, closure and the existing signed-volume rule.

        Coordinates support real integer storage through 64 bits and qualified
        IEEE binary16/32/64. Wider or non-real dtypes are explicitly refused.
        Exactly zero-area triangles are rejected without a size tolerance.
        """
        vertices = np.asarray(self.vertices)
        faces = np.asarray(self.faces)
        _coordinate_layout(vertices.dtype)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or
                not len(vertices) or not np.isfinite(vertices).all()):
            raise ValueError('solid vertices must be finite (N, 3) coordinates')
        if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or
                not np.issubdtype(faces.dtype, np.integer) or
                np.any(faces < 0) or np.any(faces >= len(vertices))):
            raise ValueError('solid faces must contain valid integer vertex indices')
        if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
            raise ValueError('solid faces must contain three distinct vertices')
        zero_area = _first_zero_area_face(vertices, faces)
        if zero_area is not None:
            raise ValueError(f'solid face {zero_area} has exactly zero area')
        self.vertices, self.faces = vertices, faces
        edges = np.sort(np.concatenate([self.faces[:, [0,1]], self.faces[:, [1,2]], self.faces[:, [2,0]]]), axis=1)
        _, count = np.unique(edges, axis=0, return_counts=True)
        if np.any(count != 2):
            raise ValueError('solid is not a closed two-manifold mesh')
        v = self.vertices[self.faces]
        volume = float(np.einsum('ij,ij->i', v[:, 0], np.cross(v[:, 1], v[:, 2])).sum()/6)
        if abs(volume) < 1e-9:
            raise ValueError('solid has no volume')
        if volume < 0:
            self.faces = self.faces[:, [0,2,1]]
        return abs(volume)


def build_solid(network, settings=None, progress=None, stop_cb=None):
    from .tube_solid import tube_mesh
    settings = settings or PrintSettings()
    points = settings.coordinates(network)
    metric = points.copy()
    zscale = 1. if settings.curved else settings.diameter/settings.height
    metric[:,2] *= zscale
    vertices,faces = tube_mesh(metric,network.edges,settings.diameter/2,progress,stop_cb)
    vertices[:,2] /= zscale
    low = vertices.min(0)
    vertices -= low
    points -= low
    solid = FiberSolid(vertices,faces,points,settings.diameter/2*(1-np.cos(np.pi/16)),settings)
    solid.validate()
    if progress: progress(100)
    return solid


def build_voxel_solid(network, settings=None, progress=None, stop_cb=None):
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes
    settings = settings or PrintSettings()
    points = settings.coordinates(network)
    radius = settings.diameter/2
    metric = points.copy()
    zscale = 1. if settings.curved else settings.diameter/settings.height
    metric[:, 2] *= zscale
    # Exact duplicate centerlines contribute material once, while the path retains both identities.
    segments = metric[network.edges]
    lengths = np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1)
    counts = np.maximum(1, np.ceil(lengths/(radius*.4)).astype(int))
    if counts.sum()+len(counts) > 1500000:
        raise MemoryError('fiber sampling budget exceeded; reduce network density or increase diameter')
    samples = np.concatenate([a+(b-a)*np.linspace(0.,1.,int(n)+1)[:, None]
                              for (a,b), n in zip(segments, counts)])
    samples = np.unique(np.round(samples, 8), axis=0)
    tree = cKDTree(samples)
    step = radius/4
    span = np.ptp(metric, axis=0)+2*radius
    while np.prod(np.ceil(span/step).astype(np.int64)+5) > settings.max_voxels:
        step *= 1.08
    if step > radius*.8:
        raise MemoryError('fiber too thin for the bounded mesh resolution; increase diameter or reduce dimensions')
    low = metric.min(0)-radius-2*step
    shape = np.ceil(span/step).astype(int)+5
    field = np.empty(tuple(shape), np.float32)
    yz = np.stack(np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing='ij'), axis=-1).reshape(-1,2)
    for i in range(shape[0]):
        if stop_cb and stop_cb():
            raise InterruptedError('solid export cancelled')
        q = np.column_stack([np.full(len(yz), i), yz])*step+low
        distance = tree.query(q, workers=1)[0]
        field[i] = (radius-distance).reshape(shape[1:])
        if progress and i % 8 == 0:
            progress(int(85*i/shape[0]))
    vertices, faces, _, _ = marching_cubes(field, 0., spacing=(step,)*3, allow_degenerate=False)
    vertices += low
    vertices[:, 2] /= zscale
    # Calibrate the planar outer envelope; apply exactly the same affine map to the path.
    if not settings.curved:
        low, span = vertices.min(0), np.ptp(vertices, axis=0)
        scale = np.array([settings.width, settings.depth, settings.height])/span
        vertices = (vertices-low)*scale
        points = (points-low)*scale
    else:
        low = vertices.min(0)
        vertices -= low
        points -= low
    if len(faces) > 1500000:
        raise MemoryError('solid face budget exceeded; reduce density or dimensions')
    solid = FiberSolid(vertices, np.asarray(faces, np.int32), points, step, settings)
    solid.validate()
    if progress:
        progress(100)
    return solid


def _binary_stl_vertices(solid):
    """Check the exact binary32 triangle soup before creating an output file.

    STL has no vertex indices. Equal serialized coordinates are therefore
    welded for validation only; emitted coordinates and face order are unchanged.
    Promotion to binary64 is exact and avoids binary32 volume arithmetic overflow.
    """
    with np.errstate(over='ignore', invalid='ignore'):
        packed = solid.vertices.astype('<f4')
    if not np.isfinite(packed).all():
        raise ValueError('binary STL coordinates overflow or are not finite')
    vertices, inverse = np.unique(packed, axis=0, return_inverse=True)
    faces = inverse[solid.faces]
    checked = FiberSolid(vertices.astype(np.float64), faces, solid.centers,
                         solid.resolution, solid.settings)
    try:
        volume = checked.validate()
    except ValueError as exc:
        raise ValueError(f'binary STL serialization is invalid: {exc}') from exc
    if not np.isfinite(volume):
        raise ValueError('binary STL serialized volume is not finite')
    if not np.array_equal(checked.faces, faces):
        raise ValueError('binary STL serialization reverses solid orientation')
    return packed


def export_solid(path, solid):
    """Write closed geometry atomically. 3MF declares millimetres explicitly.

    Binary STL additionally validates exact serialized coordinates after welding
    equal positions. Refusal preserves any existing destination. This is not a
    self-intersection or manufacturing suitability certificate. 3MF decimal
    coordinates roundtrip stored binary64 values through a binary64 XML parser;
    downstream readers may use lower precision. Integer storage beyond binary64
    exactness is not covered by that coordinate-fidelity guarantee.
    """
    path = Path(path)
    if path.suffix.lower() not in ('.stl', '.3mf'):
        raise ValueError('solid export requires .stl or .3mf')
    solid.validate()
    packed = _binary_stl_vertices(solid) if path.suffix.lower() == '.stl' else None
    with _owned_export_file(path) as temporary:
        if path.suffix.lower() == '.stl':
            dtype = np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
            with temporary.open('wb') as stream:
                stream.write(b'FiberScope; coordinates in millimetres'.ljust(80,b' '))
                stream.write(struct.pack('<I',len(solid.faces)))
                for start in range(0,len(solid.faces),65536):
                    triangles=packed[solid.faces[start:start+65536]]
                    # Calculate normals from the exact emitted positions in binary64.
                    # Binary32 cross products can overflow for finite STL coordinates.
                    normal_vertices=triangles.astype(np.float64)
                    normal=np.cross(normal_vertices[:,1]-normal_vertices[:,0],
                                    normal_vertices[:,2]-normal_vertices[:,0])
                    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-15)
                    if not np.isfinite(normal).all():
                        raise ValueError('binary STL normals are not finite')
                    data=np.zeros(len(triangles),dtype=dtype)
                    data['normal'],data['vertices']=normal,triangles
                    stream.write(data.tobytes())
        elif path.suffix.lower() == '.3mf':
            with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('[Content_Types].xml','<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
                archive.writestr('_rels/.rels','<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
                with archive.open('3D/3dmodel.model','w') as stream:
                    stream.write(b'<?xml version="1.0" encoding="UTF-8"?><model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources><object id="1" type="model"><mesh><vertices>')
                    for x,y,z in solid.vertices:
                        stream.write(('<vertex x="%.17g" y="%.17g" z="%.17g"/>'%(x,y,z)).encode())
                    stream.write(b'</vertices><triangles>')
                    for a,b,c in solid.faces:
                        stream.write(('<triangle v1="%d" v2="%d" v3="%d"/>'%(a,b,c)).encode())
                    stream.write(b'</triangles></mesh></object></resources><build><item objectid="1"/></build></model>')


def export_route(path, network, centers):
    """Write finite millimetre centers in a verified closed all-edge route."""
    centers = np.asarray(centers, dtype=float)
    nodes = np.asarray(network.route_nodes)
    route = np.asarray(network.route_edges)
    edges = np.asarray(network.edges)
    if (centers.shape != (len(network.positions), 3) or
            not np.isfinite(centers).all()):
        raise ValueError('route centers must be finite (N, 3) coordinates')
    if (edges.ndim != 2 or edges.shape[1] != 2 or not len(edges) or
            not np.issubdtype(edges.dtype, np.integer) or
            np.any(edges < 0) or np.any(edges >= len(centers))):
        raise ValueError('route edges must contain valid integer node indices')
    if (nodes.ndim != 1 or route.ndim != 1 or
            not np.issubdtype(nodes.dtype, np.integer) or
            not np.issubdtype(route.dtype, np.integer) or
            len(nodes) != len(edges) + 1 or len(route) != len(edges) or
            np.any(nodes < 0) or np.any(nodes >= len(centers)) or
            not np.array_equal(np.sort(route), np.arange(len(edges)))):
        raise ValueError('route must cover every independent edge exactly once')
    steps = np.column_stack((nodes[:-1], nodes[1:]))
    if (nodes[0] != nodes[-1] or
            not np.array_equal(np.sort(steps, axis=1),
                               np.sort(edges[route], axis=1))):
        raise ValueError('route must be closed and follow its edge identities')
    path = Path(path)
    with _owned_export_file(path) as temporary:
        with temporary.open('w',encoding='utf-8',newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['step','incoming_edge_id','node_id','x_mm','y_mm','z_mm'])
            for step,node in enumerate(network.route_nodes):
                writer.writerow([step,-1 if step == 0 else int(network.route_edges[step-1]),int(node),*centers[node]])

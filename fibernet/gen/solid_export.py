"""Millimetre fiber solids, bounded meshing, atomic STL/3MF and route exports."""
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import struct
import zipfile
import numpy as np
import os

atomic_replace = os.replace


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


@dataclass
class FiberSolid:
    vertices: np.ndarray
    faces: np.ndarray
    centers: np.ndarray
    resolution: float
    settings: PrintSettings

    def validate(self):
        if not len(self.faces) or not np.isfinite(self.vertices).all():
            raise ValueError('empty or non-finite solid')
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


def export_solid(path, solid):
    """Write closed geometry atomically. 3MF declares millimetres explicitly."""
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    solid.validate()
    try:
        if path.suffix.lower() == '.stl':
            dtype = np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
            with temporary.open('wb') as stream:
                stream.write(b'FiberScope; coordinates in millimetres'.ljust(80,b' '))
                stream.write(struct.pack('<I',len(solid.faces)))
                for start in range(0,len(solid.faces),65536):
                    triangles=solid.vertices[solid.faces[start:start+65536]].astype('<f4')
                    normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
                    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-15)
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
                        stream.write(('<vertex x="%.9g" y="%.9g" z="%.9g"/>'%(x,y,z)).encode())
                    stream.write(b'</vertices><triangles>')
                    for a,b,c in solid.faces:
                        stream.write(('<triangle v1="%d" v2="%d" v3="%d"/>'%(a,b,c)).encode())
                    stream.write(b'</triangles></mesh></object></resources><build><item objectid="1"/></build></model>')
        else:
            raise ValueError('solid export requires .stl or .3mf')
        atomic_replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def export_route(path, network, centers):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    try:
        with temporary.open('w',encoding='utf-8',newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['step','incoming_edge_id','node_id','x_mm','y_mm','z_mm'])
            for step,node in enumerate(network.route_nodes):
                writer.writerow([step,-1 if step == 0 else int(network.route_edges[step-1]),int(node),*centers[node]])
        atomic_replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)

"""Operator-only registration and offline conversion for strict Tet4 JSON."""
import argparse
import os
from pathlib import Path
import re

from .safety import canonical, read_at, sha
from .tet4_contract import MAX_SOURCE, parse_mesh, require

SOURCE_DIRECTORY='tet4-sources'

def fixture_filename(name):
    require(type(name) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name) is not None,
            'invalid Tet4 source name')
    return name+'.tet4.json'

def load_operator_sources(operator_fd, entries, reserved_names):
    """Read only pinned basenames inside the literal directory on this fd."""
    require(isinstance(entries,list) and 1<=len(entries)<=2,'invalid Tet4 sources list')
    specs=[]
    names=set(reserved_names)
    for item in entries:
        require(isinstance(item,dict) and set(item)=={'name','sha256'},'invalid Tet4 source fields')
        name,digest=item['name'],item['sha256']
        filename=fixture_filename(name)
        require(name not in names and type(digest) is str and re.fullmatch('[0-9a-f]{64}',digest) is not None,
                'duplicate Tet4 name or invalid digest')
        names.add(name)
        specs.append((name,digest,filename))
    fd=os.open(SOURCE_DIRECTORY,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=operator_fd)
    try:
        out=[]
        for name,digest,filename in specs:
            raw=read_at(fd,filename,MAX_SOURCE,single_link=True)
            mesh=parse_mesh(raw,digest)
            out.append((name,raw,digest,mesh))
        return out
    finally: os.close(fd)

def export_snapshot(b,name,directory):
    """Serialize an already prepared native Tet4Bundle without optional IO."""
    target=Path(directory)/fixture_filename(name)
    data=dict(schema='fibernet.tet4_mesh.v1',unit=b.length_unit,
              points=b.points.tolist(),tetrahedra=b.tetrahedra.tolist(),
              node_ids=list(map(str,b.node_ids)),cell_ids=list(map(str,b.cell_ids)),
              material_ids=list(map(str,b.material_ids)),
              materials={str(k):v for k,v in b.materials.items()})
    raw=canonical(data)
    digest=sha(raw)
    parse_mesh(raw,digest)
    with target.open('xb') as f: f.write(raw)
    return dict(name=name,sha256=digest)

def export_bundle(bundle_path,name,directory):
    """Offline only; native ZIP input optional Meshio, never server dispatch."""
    from fibernet.io.tet4_meshio import read_tet4_vtu_bundle
    return export_snapshot(read_tet4_vtu_bundle(bundle_path),name,directory)

def main():
    parser=argparse.ArgumentParser(description='Convert a trusted strict FiberNet Tet4 VTU input bundle offline')
    parser.add_argument('bundle')
    parser.add_argument('name')
    parser.add_argument('--directory',default=SOURCE_DIRECTORY)
    args=parser.parse_args()
    print(canonical(export_bundle(args.bundle,args.name,args.directory)).decode())

if __name__=='__main__':main()

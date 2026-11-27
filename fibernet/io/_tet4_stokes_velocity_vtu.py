"""Preflight the specific inline Meshio VTU schema before constructing a VTK grid."""

import base64
import os
import stat
from pathlib import Path
from xml.etree import ElementTree
import zlib

import numpy as np


_DTYPE = {"Float64":"f8", "Float32":"f4", "Int64":"i8", "Int32":"i4", "UInt64":"u8", "UInt8":"u1"}
def _blocks(raw, expected, endian):
    if raw is None:
        raise ValueError("Missing inline VTU payload")
    text=raw.strip()
    try:
        header=np.frombuffer(base64.b64decode(text[:16],validate=True),dtype=endian+"u4")
    except Exception as exc:
        raise ValueError("Invalid VTU compression header") from exc
    if len(header)!=3:
        raise ValueError("Incomplete VTU compression header")
    n,block,last=map(int,header)
    if not 1<=n<=512 or block!=32768 or not 0<last<=block or ((n-1)*block+last)!=expected:
        raise ValueError("Unexpected VTU block sizes for declared array")
    hdr_chars=4*((4*(3+n)+2)//3)
    try:
        full=np.frombuffer(base64.b64decode(text[:hdr_chars],validate=True),dtype=endian+"u4")
        payload=base64.b64decode(text[hdr_chars:],validate=True)
    except Exception as exc:
        raise ValueError("Invalid inline compressed VTU payload") from exc
    if len(full)!=3+n or (full[:3]!=header).any():
        raise ValueError("Inconsistent VTU block count")
    sizes=list(map(int,full[3:]))
    if sum(sizes)!=len(payload) or any(not 0<s<=block+128 for s in sizes):
        raise ValueError("Invalid VTU compressed-array extents")
    offset=0; pieces=[]
    for j,size in enumerate(sizes):
        target=last if j==n-1 else block
        unit=zlib.decompressobj()
        try:
            piece=unit.decompress(payload[offset:offset+size],target+1)
        except zlib.error as exc:
            raise ValueError("Invalid compressed VTU array") from exc
        if len(piece)!=target or not unit.eof or unit.unconsumed_tail or unit.unused_data:
            raise ValueError("Incomplete or excessive VTU decompression")
        pieces.append(piece);offset+=size
    return b"".join(pieces)


def read_own_quadratic_vtu(path):
    path=Path(path).resolve(strict=True)
    if path.suffix.lower()!=".vtu":
        raise ValueError("Expected a locally generated FiberNet .vtu file")
    if not path.is_file():
        raise ValueError("Expected a regular, locally generated FiberNet VTU file")
    flags=os.O_RDONLY | getattr(os,"O_BINARY",0) | getattr(os,"O_NONBLOCK",0)
    with os.fdopen(os.open(path,flags),"rb") as ioh:
        # Verify the opened descriptor as well as the resolved path.
        if not stat.S_ISREG(os.fstat(ioh.fileno()).st_mode):
            raise ValueError("Expected a regular, locally generated FiberNet VTU file")
        data=ioh.read(16*1024**2+1)
    if not 0<len(data)<=16*1024**2 or b"\x00" in data or b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise ValueError("Unsupported VTU document size or XML structure")
    try:
        root=ElementTree.fromstring(data)
    except ElementTree.ParseError as exc:
        raise ValueError("Invalid VTU XML") from exc
    a=root.attrib
    if (root.tag!="VTKFile" or a.get("type")!="UnstructuredGrid"
            or a.get("version")!="0.1" or a.get("header_type","UInt32")!="UInt32"
            or a.get("compressor")!="vtkZLibDataCompressor" or a.get("byte_order") not in ("LittleEndian","BigEndian")
            or len(root)!=1 or root[0].tag!="UnstructuredGrid" or len(root[0])!=1 or root[0][0].tag!="Piece"):
        raise ValueError("Expected FiberNet's single-piece inline-compressed quadratic VTU format")
    endian="<" if a["byte_order"]=="LittleEndian" else ">"
    e=root[0][0]
    n,m=(int(e.attrib.get(i,"0")) for i in ("NumberOfPoints","NumberOfCells"))
    if not 10<=n<=125000//3 or not 1<=m<=22000:
        raise ValueError("VTU point/cell dimensions exceed the admitted range")
    sections={"Points": {"Points":(n,3,("Float32","Float64"))},
      "Cells":{"connectivity":(m*10,1,("Int32","Int64")),"offsets":(m,1,("Int32","Int64")),
               "types":(m,1,("UInt8","Int32","Int64"))},
      "PointData":{"velocity_m_s":(n,3,("Float32","Float64")),"source_node_ids":(n,2,("UInt64",))},
      "CellData":{"source_cell_id":(m,1,("UInt64",))}}
    if len(e)!=4 or {s.tag for s in e}!=set(sections):
        raise ValueError("Unexpected VTU sections")
    result={}
    for group in e:
        g=sections[group.tag]
        if len(group)!=len(g) or {f.get("Name") for f in group}!=set(g):
            raise ValueError("Unexpected VTU arrays")
        for f in group:
            shape,components,allowed=g[f.get("Name")]
            if (len(f) or f.tag!="DataArray" or f.get("format")!="binary" or f.get("type") not in allowed
                    or f.get("NumberOfComponents","1")!=str(components)):
                raise ValueError("Unexpected VTU array layout")
            dtype=np.dtype(endian+_DTYPE[f.get("type")])
            raw=_blocks(f.text,shape*components*dtype.itemsize,endian)
            v=np.frombuffer(raw,dtype).reshape((shape,components)) if components>1 else np.frombuffer(raw,dtype)
            result[f.get("Name")]=v.astype(np.dtype(_DTYPE[f.get("type")]),copy=False)
    cells=result["connectivity"].reshape(m,10)
    if (np.min(cells)<0 or np.max(cells)>=n or np.any(result["types"]!=24)
            or not np.array_equal(result["offsets"],np.arange(1,m+1)*10)):
        raise ValueError("Invalid tetra10 connectivity or offsets")
    pts=result["Points"];v=result["velocity_m_s"]
    if not np.isfinite(pts).all() or not np.isfinite(v).all():
        raise ValueError("Non-finite point or velocity array")
    return pts,cells,v

"""
VTK file export for visualization in Paraview/VisIt.

Exports fiber networks as VTK polydata with fiber segments as lines.
Coordinates use round-trip float64 text precision; legacy point/cell attribute
formatting is unchanged. Legacy attributes use Float32; XML attributes retain
Float64, Int64, UInt64 or boolean UInt8 values with ASCII token names. This does not
encode units or welded crosslinks.
"""

from array import array as typed_array
from collections.abc import Mapping, Sequence
import operator
from xml.sax.saxutils import quoteattr

import numpy as np
from fibernet.core.network import FiberNetwork


def _vtp_values(data):
    """Preserve integer sequence values before NumPy's mixed-type promotion."""
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray, memoryview, typed_array, np.ndarray)):
        return np.asarray(data)
    raw = np.asarray(data, dtype=object)
    values = list(raw.flat)
    if not values:
        return np.empty(raw.shape, dtype=np.float64)
    def integer(value):
        return (isinstance(value, (bool, np.bool_, int, np.integer))
                and not isinstance(value, np.timedelta64))
    if all(integer(value) for value in values):
        if all(isinstance(value, (bool, np.bool_)) for value in values):
            return np.asarray(data, dtype=bool)
        ints = [int(value) if isinstance(value, (bool, np.bool_)) else operator.index(value)
                for value in values]
        low, high = min(ints), max(ints)
        if low >= 0 and high <= np.iinfo(np.uint64).max and (
                high > np.iinfo(np.int64).max
                or any(isinstance(value, np.unsignedinteger) for value in values)):
            dtype = np.uint64
        elif np.iinfo(np.int64).min <= low and high <= np.iinfo(np.int64).max:
            dtype = np.int64
        else:
            raise ValueError('VTP integer values must fit a common signed or unsigned 64-bit type')
        return np.asarray(ints, dtype=dtype).reshape(raw.shape)
    floating_values = []
    for value in values:
        if integer(value):
            number = int(value) if isinstance(value, (bool, np.bool_)) else operator.index(value)
            try:
                floating = float(number)
            except OverflowError as exc:
                raise ValueError('VTP mixed floating data cannot represent this integer exactly') from exc
            if not np.isfinite(floating) or int(floating) != number:
                raise ValueError('VTP mixed floating data cannot represent this integer exactly')
            floating_values.append(floating)
        elif isinstance(value, (float, np.floating)):
            if isinstance(value, np.floating) and value.dtype.itemsize > 8:
                raise ValueError('VTP floating fields support at most Float64 precision')
            floating_values.append(float(value))
        else:
            raise ValueError('VTP fields must contain real numeric scalar values')
    return np.asarray(floating_values, dtype=np.float64).reshape(raw.shape)


def _write_vtp_attributes(stream, tag, arrays):
    if not arrays:
        return
    active = {}
    for name, array in arrays.items():
        active.setdefault('Vectors' if array.ndim == 2 else 'Scalars', name)
    suffix = ''.join(' ' + kind + '=' + quoteattr(str(name)) for kind, name in active.items())
    stream.write(f'      <{tag}{suffix}>\n')
    for name, array in arrays.items():
        vtk_type = {'b': 'UInt8', 'i': 'Int64', 'u': 'UInt64', 'f': 'Float64'}[array.dtype.kind]
        components = 3 if array.ndim == 2 else 1
        stream.write(f'        <DataArray type="{vtk_type}" Name={quoteattr(str(name))} '
                     f'NumberOfComponents="{components}" format="ascii">\n')
        for row in array.reshape(-1, components):
            if array.dtype.kind == 'f':
                tokens = [f'{float(value):.17g}' for value in row]
            elif array.dtype.kind == 'b':
                tokens = [str(int(value)) for value in row]
            else:
                tokens = [str(operator.index(value)) for value in row]
            stream.write('          ' + ' '.join(tokens) + '\n')
        stream.write('        </DataArray>\n')
    stream.write(f'      </{tag}>\n')


def _has_masked_value(value, *, sequences=False):
    if np.ma.isMaskedArray(value) and np.ma.is_masked(value):
        return True
    if isinstance(value, (list, tuple)) or (sequences and isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray, memoryview, typed_array, np.ndarray))):
        return any(_has_masked_value(item, sequences=sequences) for item in value)
    if isinstance(value, np.ndarray) and value.dtype.kind == 'O':
        return any(_has_masked_value(item, sequences=sequences) for item in value.flat)
    return False

def _vtk_attributes(values, count, location, vector=False, *, float_bits=32):
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError(f'{location} data must be a mapping')
    arrays = {}
    for name, data in values.items():
        if float_bits == 64 and isinstance(name, str):
            name = str.__str__(name)
        if (not isinstance(name, str) or not 1 <= len(name) <= 255
                or '%' in name or any(ord(char) < 33 or ord(char) > 126 for char in name)):
            raise ValueError(f'{location} data names must be printable ASCII tokens of 1-255 characters without percent signs')
        if _has_masked_value(data, sequences=float_bits == 64):
            raise ValueError(f'{location} data {name!r} contains masked values')
        array = np.asarray(data) if float_bits == 32 else _vtp_values(data)
        shapes = ((count,), (count, 3)) if vector else ((count,),)
        if array.shape not in shapes:
            expected = '(N,) or (N, 3)' if vector else '(M,)'
            raise ValueError(f'{location} data {name!r} must have shape {expected}')
        if array.dtype.kind not in 'biuf' or not np.isfinite(array).all():
            raise ValueError(f'{location} data {name!r} must contain finite real numeric values')
        # Finite float16/32 and every NumPy integer fit this bound.
        # Only compare wider floats, avoiding NumPy scalar coercion of
        # the float32 limit into float16 (which can warn or overflow).
        if float_bits == 64 and array.dtype.itemsize > 8:
            raise ValueError(f'{location} data {name!r} exceeds supported 64-bit numeric precision')
        if float_bits == 32 and array.dtype.kind == 'f' and array.dtype.itemsize > 4:
            maximum = float(np.finfo(np.float32).max)
            if np.any(array < -maximum) or np.any(array > maximum):
                raise ValueError(f'{location} data {name!r} exceeds the maximum finite float32 magnitude')
        arrays[name] = array
    return arrays


def to_vtk(
    network: FiberNetwork,
    filename: str,
    point_data: dict = None,
    cell_data: dict = None,
) -> str:
    """Export fiber network to VTK legacy format.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to export.
    filename : str
        Output .vtk file path.
    point_data : dict, optional
        Arrays shaped (N,) for scalars or (N, 3) for vectors, where N is
        the total number of centerline points in fiber/point order. Points
        belonging to different fibers remain separate, even if coincident.
        Values must be finite real numeric data (including boolean indicators),
        with magnitude at most the largest finite float32. Names must be
        printable ASCII tokens of 1-255 characters, without whitespace or %.
        Masked values
        are rejected rather than exported as underlying data.
    cell_data : dict, optional
        Scalar arrays shaped (M,), where M is the total number of line
        segments, in fiber/segment order. Vector cell data is not supported.
        The same value and name restrictions as point_data apply.
        Masked values are rejected.
    
    Returns
    -------
    str
        Output filename.
    """
    # Collect all points and cells
    points = []
    cells = []
    
    point_id = 0
    for fiber in network.fibers:
        pts = fiber.centerline
        n_pts = len(pts)
        
        for pt in pts:
            points.append(pt)
        
        for i in range(n_pts - 1):
            cells.append([point_id + i, point_id + i + 1])
        
        point_id += n_pts
    
    # Validate array shapes before opening the destination, so a mismatched
    # point or cell count cannot silently produce a malformed file.
    point_arrays = _vtk_attributes(point_data, len(points), 'point', vector=True)
    cell_arrays = _vtk_attributes(cell_data, len(cells), 'cell')

    # Write VTK file
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"FiberNet network: {network.num_fibers} fibers\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        
        # Points
        f.write(f"POINTS {len(points)} double\n")
        for pt in points:
            f.write(f"{pt[0]:.17g} {pt[1]:.17g} {pt[2]:.17g}\n")
        
        # Lines (cells)
        total_ints = sum(2 + 1 for _ in cells)  # n_pts + point_ids
        f.write(f"\nLINES {len(cells)} {total_ints}\n")
        for cell in cells:
            f.write(f"2 {cell[0]} {cell[1]}\n")
        
        # Point data
        if point_arrays:
            f.write(f"\nPOINT_DATA {len(points)}\n")
            for name, data in point_arrays.items():
                if data.ndim == 2:  # Three-component vector
                    f.write(f"VECTORS {name} float\n")
                    for val in data:
                        f.write(f"{val[0]:.6e} {val[1]:.6e} {val[2]:.6e}\n")
                else:  # Scalar
                    f.write(f"SCALARS {name} float 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for val in data:
                        f.write(f"{val:.6e}\n")
        
        # Cell data
        if cell_arrays:
            f.write(f"\nCELL_DATA {len(cells)}\n")
            for name, data in cell_arrays.items():
                f.write(f"SCALARS {name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for val in data:
                    f.write(f"{val:.6e}\n")
    
    return filename


def to_vtk_xml(
    network: FiberNetwork,
    filename: str,
    *,
    point_data: dict = None,
    cell_data: dict = None,
) -> str:
    """Export one polyline per fiber to VTK XML (.vtp).

    Point fields are scalars (N,) or vectors (N, 3) in fiber/sample order.
    Cell fields are scalars (F,), one per fiber/polyline, not per segment.
    Floating fields use Float64, integer fields Int64/UInt64 and boolean
    indicators UInt8. Extended precision and ambiguous mixed integer/float
    sequences are rejected. The legacy VTK name/mask/finite checks apply.
    Field validation finishes before the destination is opened.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    filename : str
        Output .vtp file path.
    """
    # Collect geometry
    points = []
    connectivity = []
    offsets = []
    
    offset = 0
    for fiber in network.fibers:
        pts = fiber.centerline
        n_pts = len(pts)
        
        for pt in pts:
            points.append(pt)
        
        conn = list(range(offset, offset + n_pts))
        connectivity.extend(conn)
        offsets.append(offset + n_pts)
        offset += n_pts
    
    point_arrays = _vtk_attributes(point_data, len(points), 'point', vector=True, float_bits=64)
    cell_arrays = _vtk_attributes(cell_data, len(offsets), 'cell', float_bits=64)

    # Write XML
    with open(filename, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="1.0">\n')
        f.write('  <PolyData>\n')
        f.write(f'    <Piece NumberOfPoints="{len(points)}" NumberOfLines="{len(offsets)}">\n')
        
        # Points
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for pt in points:
            f.write(f'          {pt[0]:.17g} {pt[1]:.17g} {pt[2]:.17g}\n')
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')
        
        # Lines
        f.write('      <Lines>\n')
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write('          ' + ' '.join(map(str, connectivity)) + '\n')
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write('          ' + ' '.join(map(str, offsets)) + '\n')
        f.write('        </DataArray>\n')
        f.write('      </Lines>\n')
        
        _write_vtp_attributes(f, 'PointData', point_arrays)
        _write_vtp_attributes(f, 'CellData', cell_arrays)

        f.write('    </Piece>\n')
        f.write('  </PolyData>\n')
        f.write('</VTKFile>\n')
    
    return filename

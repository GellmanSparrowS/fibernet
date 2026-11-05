"""
VTK file export for visualization in Paraview/VisIt.

Exports fiber networks as VTK polydata with fiber segments as lines.
Coordinates use round-trip float64 text precision; legacy point/cell attribute
formatting is unchanged. This does not encode units or welded crosslinks.
"""

import numpy as np
from fibernet.core.network import FiberNetwork


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
        Masked values are rejected rather than exported as underlying data.
    cell_data : dict, optional
        Scalar arrays shaped (M,), where M is the total number of line
        segments, in fiber/segment order. Vector cell data is not supported.
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
    def has_masked_value(value):
        if np.ma.isMaskedArray(value) and np.ma.is_masked(value):
            return True
        if isinstance(value, (list, tuple)):
            return any(has_masked_value(item) for item in value)
        if isinstance(value, np.ndarray) and value.dtype.kind == 'O':
            return any(has_masked_value(item) for item in value.flat)
        return False

    point_arrays = {}
    for name, data in (point_data or {}).items():
        if has_masked_value(data):
            raise ValueError(f'point data {name!r} contains masked values')
        array = np.asarray(data)
        if array.shape not in ((len(points),), (len(points), 3)):
            raise ValueError(f'point data {name!r} must have shape (N,) or (N, 3)')
        point_arrays[name] = array
    cell_arrays = {}
    for name, data in (cell_data or {}).items():
        if has_masked_value(data):
            raise ValueError(f'cell data {name!r} contains masked values')
        array = np.asarray(data)
        if array.shape != (len(cells),):
            raise ValueError(f'cell data {name!r} must have shape (M,)')
        cell_arrays[name] = array

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
) -> str:
    """Export to VTK XML format (.vtp).
    
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
        
        f.write('    </Piece>\n')
        f.write('  </PolyData>\n')
        f.write('</VTKFile>\n')
    
    return filename

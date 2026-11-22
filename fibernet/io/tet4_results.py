"""Optional one-way VTU visualization of the bounded static Tet4 solve."""
from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle, _meshio
from fibernet.sim.tet4 import (
    Tet4StaticResult, _LENGTH_TO_M, _solve_static_validated, _validate_static_mesh,
)

__all__ = ['solve_static_tet4_to_vtu']

# At the existing solver size cap, the fixed arrays fit below this cap even
# with uncompressed, base64-encoded VTU data and XML overhead.
_MAX_RESULT_VTU_BYTES = 4 * 1024**2


def solve_static_tet4_to_vtu(path, bundle: Tet4Bundle, *, forces_by_id: Mapping,
                            fixed_dofs_by_id: Mapping,
                            prescribed_displacements_m_by_id: Mapping | None = None
                            ) -> Tet4StaticResult:
    """Solve and write a new .vtu diagnostic; return the result with cell fields.

    The optional prescribed displacement mapping uses the same signed metres
    and stable IDs as the numerical solver.
    Requires optional Meshio. Points are undeformed coordinates in metres.
    Displacement is PointData and stress/strain are CellData, flattened to
    nine row-major components in global axes. IDs are exact UInt64. This
    is not an exchange bundle and cannot be loaded by its strict v1 reader.

    The target is never overwritten. The destination parent must support
    same-filesystem atomic hard links; otherwise writing fails explicitly.
    Use a trusted parent directory that remains stable during the export.
    """
    target = Path(path)
    if target.suffix.lower() != '.vtu':
        raise ValueError('Tet4 result destination must have a .vtu suffix')
    parent = target.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(f'Tet4 result destination parent is not a directory: {parent}')
    # Pin ordinary relative paths for the operation, not the final basename.
    target = parent / target.name
    if os.path.lexists(target):
        raise FileExistsError(f'Tet4 result destination already exists: {target}')
    try:
        meshio = _meshio()
    except ImportError as exc:
        raise ImportError('Tet4 result export requires optional meshio; install meshio==5.3.5') from exc
    mesh = _validate_static_mesh(bundle)
    result = _solve_static_validated(mesh, forces_by_id, fixed_dofs_by_id, True,
                                     prescribed_displacements_m_by_id)
    fields = result.cell_fields
    vtu_mesh = meshio.Mesh(
        mesh.points * _LENGTH_TO_M[mesh.length_unit], [('tetra', mesh.tetrahedra)],
        point_data={'node_id': np.asarray(result.node_ids, dtype=np.uint64),
                    'displacement_m': result.displacement_m},
        cell_data={'cell_id': [np.asarray(fields.cell_ids, dtype=np.uint64)],
                   'material_id': [np.asarray(fields.material_ids, dtype=np.uint64)],
                   'strain': [fields.strain.reshape(-1, 9)],
                   'stress_Pa': [fields.stress_Pa.reshape(-1, 9)],
                   'volume_m3': [fields.volume_m3]},
    )
    with TemporaryDirectory(prefix='.tet4-results-', dir=target.parent) as staging:
        vtupath = Path(staging) / 'results.vtu'
        meshio.vtu.write(vtupath, vtu_mesh, binary=True, compression=None)
        if not 0 < vtupath.stat().st_size <= _MAX_RESULT_VTU_BYTES:
            raise ValueError('Tet4 result VTU is empty or exceeds its 4 MiB size limit')
        try:
            os.link(vtupath, target)
        except FileExistsError:
            raise
        except OSError as exc:
            raise OSError('Tet4 result atomic publication requires a writable destination '
                          f'filesystem supporting hard links: {exc}') from exc
    return result

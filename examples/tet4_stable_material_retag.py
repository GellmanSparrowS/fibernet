"""Example-only same-mesh material assignment by explicit stable cell IDs.

Use a validated Tet4Bundle from read_tet4_vtu_bundle, not a remeshed sibling.
This example does not infer phase geometry or promote row numbers to IDs.
"""
from collections.abc import Mapping
from hashlib import sha256
from types import MappingProxyType

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle, _float64, _row_count, _validate

_MAX_CELLS = 8000  # native static solve envelope
_MAX_NODES = 1800
_U64_MAX = 2**64 - 1


def _id(value):
    if (isinstance(value, (bool, np.bool_, np.timedelta64))
            or not isinstance(value, (int, np.integer))
            or not 0 <= int(value) <= _U64_MAX):
        raise ValueError('expected exact nonnegative uint64 ID')
    return int(value)


def _ids(values, size):
    if len(values) != size:
        raise ValueError('ID array length does not match the mesh')
    result = np.array([_id(value) for value in values], dtype=np.uint64)
    if np.unique(result).size != size:
        raise ValueError('duplicate stable mesh ID')
    return result


def _owned(bundle):
    if not isinstance(bundle, Tet4Bundle):
        raise TypeError('expected a Tet4Bundle')
    if not (1 <= _row_count(bundle.points, 'points') <= _MAX_NODES
            and 1 <= _row_count(bundle.tetrahedra, 'tetrahedra') <= _MAX_CELLS):
        raise ValueError('mesh exceeds native static example envelope')
    # Reuse the interchange/native normalizer before hashing or coercing.
    return _validate(bundle.points, bundle.tetrahedra, bundle.node_ids, bundle.cell_ids,
                     bundle.material_ids, bundle.materials, bundle.length_unit)


def _fingerprint(n, c, p, t, unit):
    no = np.argsort(n)
    co = np.argsort(c)
    h = sha256(b'FiberNet example: stable Tet4 mesh identity v1\0' + unit.encode('ascii') + b'\0')
    for a, dtype in (([len(n), len(c)], '<u8'), (n[no], '<u8'), (p[no], '<f8'),
                     (c[co], '<u8'), (n[t[co]], '<u8')):
        h.update(np.asarray(a, dtype=dtype).tobytes(order='C'))
    return h.hexdigest()


def mesh_identity(bundle):
    """Row-order-independent identity; exact coordinate and local vertex order."""
    mesh = _owned(bundle)
    return _fingerprint(mesh.node_ids, mesh.cell_ids, mesh.points, mesh.tetrahedra, mesh.length_unit)


def reassign_cells(bundle, *, expected_mesh_identity, cell_ids, material_id, material=None):
    """Overwrite a nonempty exact selection; leave unselected properties alone.

    New IDs require explicit isotropic properties. An existing target reuses
    its existing properties and must not specify a replacement. Remove old
    definitions only if no output cells reference them. Native preflight still
    decides topology/constraints/budget; this helper is not a solver.
    """
    bundle = _owned(bundle)
    n, c, p, t = bundle.node_ids, bundle.cell_ids, bundle.points, bundle.tetrahedra
    if type(expected_mesh_identity) is not str or expected_mesh_identity != _fingerprint(n, c, p, t, bundle.length_unit):
        raise ValueError('selection mesh identity mismatch (do not reuse after remeshing)')
    if (np.ma.isMaskedArray(cell_ids) or not isinstance(cell_ids, (list, tuple, np.ndarray))
            or (isinstance(cell_ids, np.ndarray) and cell_ids.ndim != 1)
            or not 1 <= len(cell_ids) <= len(c)):
        raise ValueError('provide a nonempty bounded explicit cell ID sequence')
    selected = _ids(cell_ids, len(cell_ids))
    # Signed and unsigned NumPy membership can promote IDs to float64.
    # Compare exact Python integers instead, including above 2**53.
    selected_ids = set(map(int, selected))
    mask = np.fromiter((int(cell_id) in selected_ids for cell_id in c),
                       dtype=bool, count=len(c))
    if int(mask.sum()) != len(selected):
        raise ValueError('selection contains cell IDs absent from this mesh')
    mid = _id(material_id)
    if mid in bundle.materials:
        if material is not None:
            raise ValueError('existing target retains its original properties')
        entry = bundle.materials[mid]
    else:
        if not isinstance(material, Mapping) or set(material) != {'young_modulus_pa','poisson_ratio'}:
            raise ValueError('new target needs young_modulus_pa and poisson_ratio')
        entry = material
    if np.ma.isMaskedArray(entry):
        raise ValueError('masked material data are unsupported')
    entry = {k: _float64(entry[k], k) for k in ('young_modulus_pa', 'poisson_ratio')}
    if not (entry['young_modulus_pa'] > 0 and -1 < entry['poisson_ratio'] < .5):
        raise ValueError('material requires finite E > 0 Pa and -1 < nu < 0.5')
    source = bundle.material_ids.astype(np.uint64, copy=True)
    source[mask] = mid
    retained = {k: MappingProxyType(dict(entry if k == mid else bundle.materials[k]))
                for k in map(int, np.unique(source))}
    def lock(a):
        return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)
    return Tet4Bundle(*(lock(a) for a in (p, t, n, c, source)), MappingProxyType(retained), bundle.length_unit)

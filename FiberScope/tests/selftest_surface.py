"""Verify topology mapping, seam connectivity, assets and safe import budgets."""
import sys
import tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fslab.surface_mapping import map_cells, MappingConfig, load_obj, front_basis
from fslab.structure import all_unit_keys
from fslab.cell_rules import graph_health


def main():
    V = np.array([[0,0,0],[1,0,0],[2,0,0],[0,1,0],[1,1,0],[2,1,0]], float)
    F = [[0,1,4,3], [1,2,5,4]]
    counts = set()
    for unit in all_unit_keys():
        P, E = map_cells(V, F, np.zeros((2,2)), unit)
        health = graph_health(P, E)
        assert health['components'] == 1, (unit, health)
        assert P[:, 0].min() < 0 and P[:, 0].max() > 2
        counts.add(len(E))
        # Base quad corners are not automatically included as scaffold fibers.
        assert not any(np.allclose(p, V[0]) for p in P)
        assert np.isfinite(P).all() and E.max() < len(P)
    assert len(counts) > 4
    for path in (Path(__file__).resolve().parents[1]/'assets'/'obj').glob('*.obj'):
        v, f = load_obj(path)
        p, e = map_cells(v, f, [[0.,0.]], 'triangle')
        assert len(e) and np.isfinite(p).all()
        basis = front_basis(v)
        assert np.allclose(basis @ basis.T, np.eye(3))
        print('[surface] %s mapped: %d segments' % (path.name, len(e)))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'triangle.obj'
        path.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n')
        v, f = load_obj(path)
        assert len(f) == 3 and all(len(face) == 4 for face in f)
    try:
        map_cells(V,F,[[0.,0.]],config=MappingConfig(max_points=2))
        raise AssertionError('budget ignored')
    except MemoryError:
        pass
    print('[surface] all types, seam connectivity, overscale, no scaffold and import PASS')


if __name__ == '__main__':
    main()

"""Validate real demos and triangle import in the actual surface widget."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PySide6.QtWidgets import QApplication
from fslab.structure import all_unit_keys
from fslab.surface_mapping import load_obj, map_cells
from fslab.search_algorithms import atomic_json
from studio.surface_tab import SurfaceTab


def main():
    app = QApplication.instance() or QApplication([])
    tab = SurfaceTab(mode='light')
    tab.resize(1366, 850)
    tab.show()
    tab.chk_follow.setChecked(False)
    records = []
    folder = Path('_cache/qa23')
    folder.mkdir(parents=True, exist_ok=True)
    try:
        for name in ('real_flighthelmet.obj', 'real_sheenchair.obj', 'real_waterbottle.obj'):
            path = Path('assets/obj') / name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            v, f = load_obj(path)
            for unit in all_unit_keys():
                p, e = map_cells(v, f, np.zeros((5, 2)), unit)
                assert np.isfinite(p).all() and len(e) and e.max() < len(p)
                records.append(dict(model=name, unit=unit, quads=len(f), points=len(p), edges=len(e)))
            index = next(i for i, p in enumerate(tab._obj_files) if Path(p).name == name)
            tab.obj_combo.setCurrentIndex(index)
            tab._debounce.stop()
            tab._recompute()
            assert tab._mapping_error is None, tab._mapping_error
            for _ in range(4):
                app.processEvents()
            tab.grab().save(str(folder / (name + '.png')))
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        with tempfile.TemporaryDirectory(prefix='triangle_import_') as directory:
            for asset in Path('assets/obj').glob('real_*.obj'):
                v, f = load_obj(asset)
                path = Path(directory)/asset.name
                with path.open('w', encoding='utf-8') as stream:
                    for vertex in v:
                        stream.write('v %.9g %.9g %.9g\n' % tuple(vertex))
                    for a, b, c, d in f:
                        stream.write('f %d %d %d\nf %d %d %d\n' % (a+1,b+1,c+1,a+1,c+1,d+1))
                v, f, info = load_obj(path, return_info=True, target_faces=1500)
                assert info['converted'] and info['reduced'] and all(len(face) == 4 for face in f)
                p, e = map_cells(v, f, np.zeros((5, 2)), 'square')
                assert np.isfinite(p).all() and len(e)
        atomic_json('docs/validation/surface23.json', dict(passed=True, cases=records, triangle_imports=3))
        print('Surface actual GUI, %d topology/asset pairs, triangle imports PASS' % len(records))
    finally:
        tab.close()


if __name__ == '__main__':
    main()

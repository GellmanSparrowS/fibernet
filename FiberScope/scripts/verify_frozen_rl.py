"""Test the EXE's actual bundled worker using a local optional training Python."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
from PyInstaller.archive.readers import CArchiveReader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FrozenRLCheck:
    def run(self, executable):
        from fslab.rl_process import run_external
        from fslab.structure import StructureFactory
        from fslab.learning import Regressor
        from fslab.mlmodel import gen_dataset
        from fslab.model_inverse import PredictedStructure
        X, Y = gen_dataset('hexagon', n=6)
        model = Regressor('ridge')
        model.train(X, Y)
        archive = CArchiveReader(str(executable))
        (ROOT/'_tmp').mkdir(exist_ok=True)
        old = {key: os.environ.get(key) for key in ('FIBERSCOPE_TRAINING_DIR', 'FIBERSCOPE_TRAIN_PYTHON')}
        frozen, meipass = getattr(sys, 'frozen', None), getattr(sys, '_MEIPASS', None)
        with tempfile.TemporaryDirectory(prefix='frozen_rl_', dir=ROOT/'_tmp') as directory:
            root = Path(directory)
            count = 0
            for name in archive.toc:
                relative = Path(name.replace('\\', '/'))
                if relative.parts[0] != 'training_runtime':
                    continue
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('invalid bundle entry')
                path = root/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extract(name))
                count += 1
            assert count > 10, 'training source missing from bundle'
            try:
                os.environ['FIBERSCOPE_TRAINING_DIR'] = str(root/'jobs')
                os.environ['FIBERSCOPE_TRAIN_PYTHON'] = sys.executable
                sys.frozen, sys._MEIPASS = True, str(root)
                updates = []
                def progress(record, run):
                    assert run is not None, 'bundled worker omitted live geometry'
                    updates.append(record.eval_id)
                result = run_external(StructureFactory(grid_x=1, grid_y=1, n_pts_per_side=1),
                                      'J', 'PPO', {}, 3, 220, callback=progress)
                assert len(result['records']) == 3 and result['best_run'] is not None
                assert updates == [1, 2, 3]
                predicted = run_external(StructureFactory(unit='hexagon', grid_x=1, grid_y=1),
                    'max_peak', 'PPO', {}, 3, 220, model=model)
                assert isinstance(predicted['best_preview'], PredictedStructure)
                assert predicted['best_run'] is None
                print('[frozen-rl] bundled source=%d files; external PPO + actual mechanics + learned model + replay PASS' % count)
            finally:
                for key, value in old.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                for key, value in (('frozen', frozen), ('_MEIPASS', meipass)):
                    if value is None:
                        delattr(sys, key)
                    else:
                        setattr(sys, key, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exe', nargs='?', default=str(ROOT/'dist'/'FiberScope.exe'))
    FrozenRLCheck().run(parser.parse_args().exe)

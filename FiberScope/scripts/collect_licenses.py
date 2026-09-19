"""Collect installed distribution license files into bundled assets."""
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil


class LicenseInventory:
    packages = ('numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl',
                'PySide6', 'PySide6_Essentials', 'shiboken6', 'pyqtgraph',
                'stable-baselines3', 'gymnasium', 'torch', 'fast-simplification', 'scikit-image', 'lazy-loader', 'packaging',
                'networkx', 'imageio', 'tifffile', 'pillow', 'manifold3d')

    def __init__(self, root):
        self.root = Path(root)

    def collect(self):
        output = self.root / 'assets' / 'licenses'
        output.mkdir(parents=True, exist_ok=True)
        inventory = {}
        for name in self.packages:
            distribution = metadata.distribution(name)
            paths = []
            for relative in distribution.files or []:
                path = Path(distribution.locate_file(relative))
                if not path.is_file() or path.stat().st_size > 2000000:
                    continue
                if ('licenses' in [part.lower() for part in Path(relative).parts]
                        or path.name.lower().startswith(('license', 'copying', 'notice'))):
                    target = output / name / str(relative).replace('..', '_').replace(':', '_')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
                    paths.append(target.relative_to(output).as_posix())
            inventory[name] = dict(version=distribution.version,
                                   license=distribution.metadata.get('License-Expression') or distribution.metadata.get('License'),
                                   files=paths)
        (output/'inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
        print('[licenses] %d installed distributions inventoried' % len(inventory))


if __name__ == '__main__':
    LicenseInventory(Path(__file__).resolve().parents[1]).collect()

"""Check the shared generator in sibling and monorepo source layouts."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from fslab.fibernet_bridge import FIBERNET_ROOT, ensure_fibernet

    app_root = Path(__file__).resolve().parents[1]
    parent = app_root.parent
    expected = (Path(os.environ['FIBERNET_ROOT']) if 'FIBERNET_ROOT' in os.environ
                else parent if (parent / 'fibernet' / 'core').is_dir()
                else parent / 'fibernet')
    assert Path(FIBERNET_ROOT).resolve() == expected.resolve()
    graph = ensure_fibernet().pattern_2d(unit='square', box=(10, 10), grid=(1, 1))
    assert graph.num_nodes > 0 and graph.num_edges > 0
    print('[selftest_bridge] PASS')


if __name__ == '__main__':
    main()

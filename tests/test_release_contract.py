"""Check that the public package reports its packaged version."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_version_matches_package_metadata():
    import fibernet
    from fibernet.version import __version__

    project = (Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text(
        encoding='utf-8')
    declared = re.search(r'^version = "([^"]+)"', project, re.MULTILINE)
    assert declared is not None
    assert fibernet.__version__ == __version__ == declared.group(1)


if __name__ == '__main__':
    test_version_matches_package_metadata()
    print('[test_release_contract] PASS')

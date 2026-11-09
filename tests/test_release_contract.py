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


def test_release_installation_guides_match_package_version():
    import fibernet

    root = Path(__file__).resolve().parents[1]
    version = fibernet.__version__
    for name in ('README.md', 'README_CN.md', 'PYPI_README.md'):
        text = (root / name).read_text(encoding='utf-8')
        assert 'fibernet==' + version in text
    pypi_readme = (root / 'PYPI_README.md').read_text(encoding='utf-8')
    assert pypi_readme.startswith('# FiberNet ' + version + '\n')
    for extra in ('ml', 'manufacturing', 'fracture'):
        assert 'fibernet[' + extra + ']==' + version in pypi_readme
    assert (root / 'docs' / ('RELEASE_' + version.replace('.', '_') + '.md')).is_file()

"""Homepage public channel stays usable independently of source releases."""
import ast
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('homepage_install_version_check', ROOT/'scripts/check_bilingual_homepage.py')
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def _config(path, version='4.4.3', quote='"'):
    (path/'pyproject.toml').write_text('[project]\nname = "fibernet"\nversion = '+quote+version+quote+'\n[tool.example]\nversion = "0.0.1"\n', encoding='utf-8')


def _commands(version='4.4.3'):
    common='python -m pip install fibernet\n[PyPI](https://pypi.org/project/fibernet/#generate-analyze-simulate-and-save)\n'
    return ('Development checkout: '+version+'\npython -m pip install -e .\n'+common,
            '开发版源码：'+version+'\n'+common)


@pytest.mark.parametrize('version', ['4.4.3','5.0.0','4.5.0rc1','4.4.3.post1'])
@pytest.mark.parametrize('quote', ['"', "'"])
def test_source_version_can_advance_without_unpublished_public_pin(tmp_path, version, quote):
    _config(tmp_path, version, quote)
    CHECK.check_install_commands(*_commands(version), root=tmp_path)


@pytest.mark.parametrize('language', [0,1])
@pytest.mark.parametrize('extra', [
    'python -m pip install fibernet==4.4.3',
    'python -m pip install --upgrade fibernet==4.4.20',
    "python -m pip install 'fibernet==4.4.2'",
    'python -m pip install "fibernet[ml]==4.4.3"',
    'python -m pip install fibernet>=5',
    'python -m pip install "fibernet @ https://example.invalid/fibernet.whl"',
    'python -m pip install fibernet @ https://example.invalid/fibernet.whl',
])
def test_public_homepage_cannot_acquire_version_constraints(tmp_path, language, extra):
    _config(tmp_path)
    commands=list(_commands());commands[language]+='\n'+extra+'\n'
    with pytest.raises(ValueError, match='public installation must use unpinned fibernet'):
        CHECK.check_install_commands(*commands, root=tmp_path)


@pytest.mark.parametrize('language', [0,1])
@pytest.mark.parametrize('suffix', ['4.4.3/','?version=4.4.3'])
def test_version_specific_or_malformed_project_page_is_rejected(tmp_path, language, suffix):
    _config(tmp_path)
    commands=list(_commands())
    # Versioned URL is wrong even if that version happens to exist today.
    commands[language]+='\n[bad](https://pypi.org/project/fibernet/'+suffix+')\n'
    with pytest.raises(ValueError, match='stable PyPI project page'):
        CHECK.check_install_commands(*commands, root=tmp_path)


@pytest.mark.parametrize('language', [0,1])
def test_missing_public_command_or_source_label_does_not_pass(tmp_path, language):
    _config(tmp_path)
    for bad in ['', _commands()[language].replace('4.4.3','4.4.2'),
                _commands()[language].replace('python -m pip install fibernet\n','')]:
        commands=list(_commands());commands[language]=bad
        with pytest.raises(ValueError):CHECK.check_install_commands(*commands, root=tmp_path)


@pytest.mark.parametrize('editable', ['', 'python -m pip install -e X\n'])
def test_english_editable_source_installation_still_required(tmp_path, editable):
    _config(tmp_path);english,chinese=_commands()
    english=english.replace('python -m pip install -e .\n',editable)
    with pytest.raises(ValueError,match='English editable source installation'):
        CHECK.check_install_commands(english,chinese,root=tmp_path)


@pytest.mark.parametrize('config', ['[tool.other]\nversion = "4.4.3"\n', '[project]\nname = "fibernet"\n', '[project]\nversion = "4.4.3\'\n'])
def test_source_version_must_be_literal_project_version(tmp_path, config):
    (tmp_path/'pyproject.toml').write_text(config,encoding='utf-8')
    with pytest.raises(ValueError,match='Missing literal project.version'):
        CHECK.check_install_commands(*_commands(),root=tmp_path)


def test_quote_and_option_do_not_constrain_unpinned_optional_extras(tmp_path):
    _config(tmp_path);commands=list(_commands())
    commands[0]+='python -m pip install --upgrade "fibernet[ml]"\n'
    CHECK.check_install_commands(*commands,root=tmp_path)


def test_checker_parses_on_python39_without_package_or_toml_import():
    tree=ast.parse((ROOT/'scripts/check_bilingual_homepage.py').read_text(encoding='utf-8'),feature_version=(3,9))
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
    assert not any(x and (x in ('tomllib','tomli') or x.startswith('fibernet')) for x in imports)


def test_real_homepages_accept_next_unpublished_source_version(tmp_path):
    homepage=(ROOT/'README.md').read_text(encoding='utf-8')
    standalone=(ROOT/'README_CN.md').read_text(encoding='utf-8')
    english,chinese=CHECK.check_locale_structure(homepage,standalone)
    _config(tmp_path,'9.9.0')
    current=re.search(r'(?m)^version = "([^"]+)"', (ROOT/'pyproject.toml').read_text(encoding='utf-8')).group(1)
    english=english.replace('Development checkout: '+current,'Development checkout: 9.9.0')
    chinese=chinese.replace('开发版源码：'+current,'开发版源码：9.9.0')
    CHECK.check_install_commands(english,chinese,root=tmp_path)
    assert 'python -m pip install fibernet==9.9.0' not in english+chinese


def test_homepage_source_reads_ignore_windows_default_encoding(monkeypatch, tmp_path):
    original_read_text = Path.read_text

    def windows_read_text(path, *args, **kwargs):
        if not args and kwargs.get('encoding') is None:
            kwargs['encoding'] = 'cp1252'
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', windows_read_text)
    # Same source bytes and implicit decoding boundary as the Windows failure.
    with pytest.raises(UnicodeDecodeError):
        (ROOT/'scripts/check_bilingual_homepage.py').read_text()
    test_checker_parses_on_python39_without_package_or_toml_import()
    test_real_homepages_accept_next_unpublished_source_version(tmp_path)

"""Version pins follow package configuration in both homepage languages."""
import ast
import importlib.util
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('homepage_install_version_check', ROOT/'scripts/check_bilingual_homepage.py')
CHECK=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def _config(path,version='4.4.2',quote='"'):
    (path/'pyproject.toml').write_text('[project]\nname = "fibernet"\nversion = '+quote+version+quote+'\n[tool.example]\nversion = "0.0.1"\n',encoding='utf-8')


def _commands(version):
    pin='python -m pip install fibernet=='+version
    return pin+'\npython -m pip install -e .\n',pin+'\n'


@pytest.mark.parametrize('version',['4.4.2','4.4.3','5.0.0','4.5.0rc1','4.4.2.post1'])
@pytest.mark.parametrize('quote',['"',"'"])
def test_current_and_future_package_version_is_authoritative(tmp_path,version,quote):
    _config(tmp_path,version,quote)
    CHECK.check_install_commands(*_commands(version),root=tmp_path)


@pytest.mark.parametrize('language',[0,1])
@pytest.mark.parametrize('wrong',['4.4.1','4.2.0','4.4.20','4.4.3'])
def test_stale_wrong_and_prefix_matching_versions_are_rejected(tmp_path,language,wrong):
    _config(tmp_path)
    commands=list(_commands('4.4.2'));commands[language]=commands[language].replace('4.4.2',wrong)
    with pytest.raises(ValueError,match='pinned installation must use fibernet==4.4.2'):
        CHECK.check_install_commands(*commands,root=tmp_path)


@pytest.mark.parametrize('language',[0,1])
def test_missing_or_additional_stale_pin_is_rejected(tmp_path,language):
    _config(tmp_path)
    commands=list(_commands('4.4.2'))
    commands[language]+='python -m pip install fibernet==4.4.1\n'
    with pytest.raises(ValueError,match='pinned installation'):
        CHECK.check_install_commands(*commands,root=tmp_path)
    commands[language]=''
    with pytest.raises(ValueError,match='pinned installation'):
        CHECK.check_install_commands(*commands,root=tmp_path)


@pytest.mark.parametrize('editable',['','python -m pip install -e X\n'])
def test_english_editable_source_installation_is_still_required(tmp_path,editable):
    _config(tmp_path)
    english,chinese=_commands('4.4.2')
    english=english.replace('python -m pip install -e .\n',editable)
    with pytest.raises(ValueError,match='English editable source installation'):
        CHECK.check_install_commands(english,chinese,root=tmp_path)


@pytest.mark.parametrize('config',[
    '[tool.other]\nversion = "4.4.2"\n',
    '[project]\nname = "fibernet"\n[tool.other]\nversion = "4.4.2"\n',
    '[project]\nversion = "4.4.2\'\n',
])
def test_nonproject_or_malformed_version_cannot_supply_a_pin(tmp_path,config):
    (tmp_path/'pyproject.toml').write_text(config,encoding='utf-8')
    with pytest.raises(ValueError,match='Missing literal project.version'):
        CHECK.check_install_commands(*_commands('4.4.2'),root=tmp_path)


def test_checker_parses_on_python39_grammar_without_tomllib_dependency():
    source=(ROOT/'scripts/check_bilingual_homepage.py').read_text(encoding='utf-8')
    tree=ast.parse(source,feature_version=(3,9))
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
    assert not any(x and (x in ('tomllib','tomli') or x.startswith('fibernet')) for x in imports)


@pytest.mark.parametrize('language',[0,1])
@pytest.mark.parametrize('extra',[
    'python -m pip install --upgrade fibernet==4.4.20',
    "python -m pip install 'fibernet==4.4.20'",
    'python -m pip install "fibernet[ml]==4.4.1"',
])
def test_extra_pin_cannot_hide_behind_shell_options_or_quotes(tmp_path,language,extra):
    _config(tmp_path)
    commands=list(_commands('4.4.2'));commands[language]+=extra+'\n'
    with pytest.raises(ValueError,match='pinned installation'):
        CHECK.check_install_commands(*commands,root=tmp_path)

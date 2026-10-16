"""The dev aggregate follows published wheel tags; explicit accel stays pinned."""
from pathlib import Path
import re

from packaging.requirements import Requirement
import pytest

_TEXT = (Path(__file__).parents[1] / 'pyproject.toml').read_text(encoding='utf-8')
_DEV = _TEXT.split('dev = [', 1)[1].split('\ndocs = [', 1)[0]
_TAICHI = Requirement(re.search(r'"(taichi[^\"]+)"', _DEV).group(1))


@pytest.mark.parametrize('system,machine,version,expected', [
    ('darwin', 'arm64', '3.9', False),
    ('darwin', 'arm64', '3.10', True),
    ('darwin', 'arm64', '3.12', True),
    ('darwin', 'x86_64', '3.12', False),
    ('linux', 'x86_64', '3.9', True),
    ('linux', 'x86_64', '3.12', True),
    ('linux', 'aarch64', '3.12', False),
    ('win32', 'AMD64', '3.9', True),
    ('win32', 'AMD64', '3.12', True),
    ('linux', 'x86_64', '3.14', False),
])
def test_dev_taichi_uses_published_wheel_platforms(system, machine, version, expected):
    assert str(_TAICHI.specifier) == '==1.7.4'
    assert _TAICHI.marker.evaluate({
        'implementation_name': 'cpython', 'sys_platform': system,
        'platform_machine': machine, 'python_version': version,
    }) is expected


def test_explicit_accel_request_is_not_silently_empty():
    assert 'accel = ["taichi==1.7.4"]' in _TEXT

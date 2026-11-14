"""Portable public API/unsupported-platform contracts; no native execution."""
import ast
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

import fibernet.integrations as api
from fibernet.integrations import lammps_saved as reader


def test_public_exports_are_the_single_offline_implementation():
    names = ('SavedLAMMPSRead', 'LAMMPSSavedReadLimits', 'LAMMPSSavedReadError',
             'LAMMPSCompilerCompatibilityError', 'read_saved_lammps_run')
    for name in names:
        assert name in api.__all__
        assert getattr(api, name) is getattr(reader, name)
    assert issubclass(reader.LAMMPSSavedReadError, api.LAMMPSContractError)
    assert issubclass(reader.LAMMPSCompilerCompatibilityError, reader.LAMMPSSavedReadError)


def test_python39_grammar_for_public_reader_and_tests():
    root = Path(__file__).resolve().parents[1]
    paths = [root/'fibernet/integrations/lammps_saved.py',
             root/'fibernet/integrations/__init__.py']
    paths += sorted((root/'tests').glob('test_lammps_saved_read*.py'))
    for path in paths:
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 9))


def test_unsupported_platform_rejects_before_paths_or_runtime():
    with patch.object(reader.os, 'name', 'nt'), \
         patch.object(reader.os, 'open', side_effect=AssertionError('no file open')), \
         patch.object(reader.Path, 'resolve', side_effect=AssertionError('no path resolution')):
        with pytest.raises(reader.LAMMPSSavedReadError, match='POSIX'):
            api.read_saved_lammps_run('unused', expected_manifest_sha256='0'*64)


def test_missing_no_follow_support_rejects_before_open(monkeypatch):
    monkeypatch.delattr(reader.os, 'O_NOFOLLOW', raising=False)
    with patch.object(reader.os, 'open', side_effect=AssertionError('no file open')):
        with pytest.raises(reader.LAMMPSSavedReadError, match='POSIX'):
            api.read_saved_lammps_run('unused', expected_manifest_sha256='0'*64)


@pytest.mark.parametrize('changes', [dict(total_bytes=True), dict(total_bytes=0),
                                   dict(total_bytes=21*reader._MIB),
                                   dict(deadline_seconds=float('nan'))])
def test_bad_limits_reject_before_any_platform_io(changes):
    with patch.object(reader.os, 'open', side_effect=AssertionError('no file open')):
        with pytest.raises(reader.LAMMPSSavedReadError):
            api.read_saved_lammps_run('unused', expected_manifest_sha256='0'*64,
                limits=replace(api.LAMMPSSavedReadLimits(), **changes))

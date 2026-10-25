"""Bundled electrothermal fixture storage controls; no native solver calls."""
import base64
from dataclasses import replace
import gzip
import hashlib
import io
import json
from unittest.mock import patch

import pytest

from benchmarks import saved_electrothermal_fixtures as fixtures


def packed(raw):
    destination = io.BytesIO()
    with gzip.GzipFile(fileobj=destination, mode='wb', filename='',
                       compresslevel=9, mtime=0) as stream:
        stream.write(raw)
    return destination.getvalue()


def encoded(compressed):
    return base64.b64encode(compressed) + b'\n'


def spec_for(compressed, raw):
    return fixtures.FixtureSpec(len(compressed), len(raw),
                                hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize('name', fixtures.FIXTURES)
def test_exact_saved_identity_and_deterministic_storage(name):
    raw = fixtures.fixture_bytes(name)
    spec = fixtures.FIXTURES[name]
    assert len(raw) == spec.raw_bytes
    assert hashlib.sha256(raw).hexdigest() == spec.raw_sha256
    assert isinstance(json.loads(raw), dict)
    stored = (fixtures.ROOT / (name + '.gz.b64')).read_bytes()
    compressed = base64.b64decode(stored[:-1], validate=True)
    assert len(stored) == spec.encoded_bytes and len(compressed) == spec.compressed_bytes
    assert compressed[:10] == bytes.fromhex('1f8b08000000000002ff')
    # Deflate output can differ across zlib revisions. Test determinism within
    # this toolchain, not a cross-toolchain compressed-byte identity promise.
    assert packed(raw) == packed(raw)


def test_encoded_file_read_is_bounded_and_unknown_name_is_not_a_path():
    spec = fixtures.FIXTURES['planar_network.json']
    class Source(io.BytesIO):
        def read(self, count=-1):
            assert count == spec.encoded_bytes + 1
            return super().read(count)
    with patch.object(fixtures.Path, 'open', return_value=Source(b'x' * (spec.encoded_bytes + 1))):
        with pytest.raises(ValueError, match='encoded size'):
            fixtures.fixture_bytes('planar_network.json')
    with patch.object(fixtures.Path, 'open', side_effect=AssertionError('must not open')):
        with pytest.raises(KeyError):
            fixtures.fixture_bytes('../anything')


@pytest.mark.parametrize('mutation', ['short', 'long', 'no_lf', 'bad_base64', 'padding_bits'])
def test_encoded_framing_and_base64_rejected(mutation):
    raw = b'fixture'; compressed = packed(raw); spec = spec_for(compressed, raw)
    value = encoded(compressed)
    if mutation == 'short': value = value[:-1]
    elif mutation == 'long': value += b'x'
    elif mutation == 'no_lf': value = value[:-1] + b' '
    elif mutation == 'bad_base64': value = b'!' + value[1:]
    else:
        # Choose a payload whose last base64 quantum has unused bits.
        raw = b'ab'; compressed = packed(raw); spec = spec_for(compressed, raw)
        value = encoded(compressed)
        assert value.endswith(b'==\n')
        alphabet = b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
        position = alphabet.index(value[-4]); value = value[:-4] + bytes([alphabet[position + 1]]) + value[-3:]
    with pytest.raises(ValueError): fixtures._decode(value, spec)


@pytest.mark.parametrize('mutation', ['not_gzip', 'truncated', 'trailing', 'second_member', 'crc', 'wrong_sha', 'raw_short', 'raw_long', 'compressed_size'])
def test_gzip_boundary_size_and_raw_identity_rejected(mutation):
    raw = b'exact frozen bytes'; compressed = packed(raw); spec = spec_for(compressed, raw)
    if mutation == 'not_gzip': compressed = b'x' * len(compressed)
    elif mutation == 'truncated': compressed = compressed[:-1]
    elif mutation == 'trailing': compressed += b'x'
    elif mutation == 'second_member': compressed += packed(b'other')
    elif mutation == 'crc': compressed = compressed[:-8] + bytes([compressed[-8] ^ 1]) + compressed[-7:]
    elif mutation == 'wrong_sha': spec = replace(spec, raw_sha256='0' * 64)
    elif mutation == 'raw_short': spec = replace(spec, raw_bytes=len(raw) - 1)
    elif mutation == 'raw_long': spec = replace(spec, raw_bytes=len(raw) + 1)
    elif mutation == 'compressed_size': spec = replace(spec, compressed_bytes=len(compressed) + 1)
    if mutation in ('truncated', 'trailing', 'second_member'):
        # Admit the altered compressed length to exercise gzip boundary guards.
        spec = replace(spec, compressed_bytes=len(compressed))
    with pytest.raises(ValueError): fixtures._decode(encoded(compressed), spec)


def test_expansion_is_capped_before_full_payload_is_returned():
    compressed = packed(b'x' * 100000)
    spec = spec_for(compressed, b'x' * 31)
    original = fixtures.zlib.decompressobj
    observed = []
    class Observed:
        def __init__(self, **kwargs): self.decoder = original(**kwargs)
        def decompress(self, value, *, max_length):
            observed.append(max_length)
            result = self.decoder.decompress(value, max_length=max_length)
            assert len(result) <= 32
            return result
        def __getattr__(self, name): return getattr(self.decoder, name)
    with patch.object(fixtures.zlib, 'decompressobj', Observed):
        with pytest.raises(ValueError, match='decoded size'):
            fixtures._decode(encoded(compressed), spec)
    assert observed == [32]


def test_fixture_loader_import_is_stdlib_only():
    import builtins
    import importlib.util
    from pathlib import Path
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        assert not name.startswith(('fibernet', 'mcp', 'numpy', 'scipy', 'torch', 'taichi'))
        return original(name, *args, **kwargs)
    path = Path(__file__).resolve().parents[1] / 'benchmarks/saved_electrothermal_fixtures.py'
    spec = importlib.util.spec_from_file_location('_saved_fixture_import_only', path)
    module = importlib.util.module_from_spec(spec)
    with patch.object(builtins, '__import__', side_effect=guard):
        spec.loader.exec_module(module)

"""Bounded test-only access to exact saved JSON bytes, not a runtime wire format.

Storage is one gzip member (empty filename, mtime=0, level=9), encoded as
canonical base64 ASCII with one terminal LF. Limits and raw identities are
fixed here rather than supplied by the file being decoded.
"""
import base64
import binascii
from dataclasses import dataclass
import hashlib
from pathlib import Path
import zlib


@dataclass(frozen=True)
class FixtureSpec:
    compressed_bytes: int
    raw_bytes: int
    raw_sha256: str

    @property
    def encoded_bytes(self):
        return 4 * ((self.compressed_bytes + 2) // 3) + 1


FIXTURES = {
    'projection_native31.json': FixtureSpec(
        36822, 675329,
        'ca338fd42aaf2af8609bb1451abc5c9785b9480db8d75902e65cbb2bf51a8247'),
    'projection_report.json': FixtureSpec(
        15444, 458163,
        '65fd5b7d472ad011df03693c669bc60f1e16e5fa844cdf083ec20bb5ba2357a8'),
}
ROOT = Path(__file__).resolve().parents[1] / 'fixtures'


def _decode(encoded, spec):
    if len(encoded) != spec.encoded_bytes or not encoded.endswith(b'\n'):
        raise ValueError('fixture encoded size or framing mismatch')
    try:
        compressed = base64.b64decode(encoded[:-1], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError('fixture base64 rejected') from exc
    if (len(compressed) != spec.compressed_bytes
            or base64.b64encode(compressed) + b'\n' != encoded):
        raise ValueError('fixture compressed size or encoding mismatch')
    decoder = zlib.decompressobj(wbits=31)
    try:
        raw = decoder.decompress(compressed, max_length=spec.raw_bytes + 1)
    except zlib.error as exc:
        raise ValueError('fixture gzip rejected') from exc
    # No flush(): output allocation is capped even for a valid expansion bomb.
    if (len(raw) != spec.raw_bytes or not decoder.eof
            or decoder.unused_data or decoder.unconsumed_tail):
        raise ValueError('fixture decoded size or gzip boundary mismatch')
    if hashlib.sha256(raw).hexdigest() != spec.raw_sha256:
        raise ValueError('fixture raw identity mismatch')
    return raw


def fixture_bytes(name):
    """Return verified original bytes for one fixed test fixture name."""
    spec = FIXTURES[name]
    with (ROOT / (name + '.gz.b64')).open('rb') as source:
        encoded = source.read(spec.encoded_bytes + 1)
    return _decode(encoded, spec)

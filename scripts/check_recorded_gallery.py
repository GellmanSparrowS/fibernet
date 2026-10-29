"""Validate recorded-gallery payloads without rendering, simulation or learning."""
from pathlib import Path
import hashlib
import json
import re
from PIL import Image
from check_bilingual_homepage import locale_block

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / 'docs/media/new'


def check():
    manifest = json.loads((GALLERY / 'MANIFEST.json').read_text())
    listed = set(manifest['files'])
    actual = {str(p.relative_to(GALLERY)) for p in GALLERY.rglob('*')
              if p.is_file() and p.name != 'MANIFEST.json'
              and '__pycache__' not in p.parts}
    assert listed == actual, 'Gallery payload does not match manifest'
    for relative, entry in manifest['files'].items():
        path = GALLERY / relative
        assert path.resolve().is_relative_to(GALLERY.resolve()), relative
        raw = path.read_bytes()
        assert len(raw) == entry['bytes'], relative
        assert hashlib.sha256(raw).hexdigest() == entry['sha256'], relative
    for relative, entry in manifest['animations'].items():
        image = Image.open(GALLERY / relative)
        assert list(image.size) == entry['dimensions'], relative
        assert image.n_frames == entry['frames'], relative
        elapsed = 0
        for frame in range(image.n_frames):
            image.seek(frame)
            image.load()
            elapsed += image.info.get('duration', 0)
        assert elapsed == entry['duration_ms'], relative
        assert abs(image.height / image.width - 2**0.5) < .004, relative
    destinations = []
    for filename in ('README.md', 'README_CN.md'):
        text = (ROOT / filename).read_text()
        text = locale_block(text, 'en' if filename == 'README.md' else 'zh')
        paths = re.findall(r'\]\((docs/media/new/[^)#]+)(?:#[^)]*)?\)', text)
        assert paths, filename
        assert all((ROOT / path).is_file() for path in paths), filename
        destinations.append(sorted(paths))
    assert destinations[0] == destinations[1], 'Bilingual gallery links differ'
    assert len(manifest['animations']) == 7
    assert len(list((ROOT / 'docs/media').glob('*.gif'))) == 12
    print(json.dumps({'status': 'PASS', 'payload_files': len(listed),
                      'recorded_animations': 7, 'original_animations': 12,
                      'bilingual_links_equal': True,
                      'physical_solver_executed': False}, indent=2))


if __name__ == '__main__':
    check()

"""Validate recorded-gallery payloads without rendering, simulation or learning."""
from pathlib import Path
import hashlib
import json
from PIL import Image
from check_bilingual_homepage import destinations as all_destinations, inline_gifs, locale_block

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
    locale_destinations = []
    homepage = (ROOT / 'README.md').read_text()
    expected_gifs = {'docs/media/new/' + path for path in manifest['animations']}
    for language in ('en', 'zh'):
        text = locale_block(homepage, language)
        paths = [target.split('#', 1)[0] for target in all_destinations(text)
                 if target.startswith('docs/media/new/')]
        assert paths, language
        assert all((ROOT / path).is_file() for path in paths), language
        assert expected_gifs <= set(inline_gifs(text)), language
        locale_destinations.append(sorted(paths))
    assert locale_destinations[0] == locale_destinations[1], 'Bilingual gallery links differ'
    assert locale_block(homepage, 'zh') == locale_block(
        (ROOT / 'README_CN.md').read_text(), 'zh'), 'Standalone Chinese differs'
    assert len(manifest['animations']) == 7
    assert len(list((ROOT / 'docs/media').glob('*.gif'))) == 12
    print(json.dumps({'status': 'PASS', 'payload_files': len(listed),
                      'recorded_animations': 7, 'original_animations': 12,
                      'bilingual_links_equal': True,
                      'physical_solver_executed': False}, indent=2))


if __name__ == '__main__':
    check()

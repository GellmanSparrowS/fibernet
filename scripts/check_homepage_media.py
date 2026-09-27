"""Verify local animated homepage assets without displaying the images.

Run: python scripts/check_homepage_media.py
"""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageChops


class MediaAudit:
    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self.media = self.root / 'docs' / 'media'

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def check(self, name, svg_suffix, frame_field):
        manifest = json.loads((self.media / (name + '.json')).read_text(
            encoding='utf-8'))
        gif_path = self.media / (name + '.gif')
        svg_path = self.media / (name + svg_suffix)
        assert self.digest(gif_path) == manifest['gif_sha256']
        assert self.digest(svg_path) == manifest['svg_sha256']
        render_script = manifest.get('render_script',
                                     'scripts/make_homepage_media.py')
        assert self.digest(self.root / render_script) \
            == manifest['render_script_sha256']
        if 'source_results' in manifest:
            assert self.digest(self.root / manifest['source_results']) \
                == manifest['source_results_sha256']
        with Image.open(gif_path) as animation:
            frame_count = animation.n_frames
            dimensions = animation.size
            assert frame_count == len(manifest[frame_field])
            assert dimensions[0] >= 1000 and dimensions[1] >= 500
            animation.seek(0)
            first = animation.convert('RGB')
            animation.seek(animation.n_frames // 2)
            middle = animation.convert('RGB')
            assert ImageChops.difference(first, middle).getbbox() is not None
        svg = svg_path.read_text(encoding='utf-8')
        assert svg.count('font-family') > 0
        if frame_field == 'amplitudes':
            assert 'Geometry only' in svg
        elif name == 'tensile_recruitment_kagome':
            assert 'Colors do not measure total force flow.' in svg
        elif name == 'continuous_route_kagome':
            assert 'physical printing constraints not certified' in svg
        elif name == 'directional_recruitment_hexagon':
            assert 'Colors do not measure total force flow.' in svg
        else:
            assert 'colors do not measure total force flow' in svg
        print('[media_audit] %s: %d frames, %dx%d, %.1f KB' % (
            name, frame_count, *dimensions,
            gif_path.stat().st_size / 1024))

    def run(self):
        self.check('spectrum_four_topologies', '_peak.svg', 'amplitudes')
        self.check('tensile_recruitment_kagome', '_final.svg',
                   'sampled_frame_indices')
        self.check('tensile_threshold_sensitivity', '_final.svg',
                   'sampled_frame_indices')
        self.check('continuous_route_kagome', '_final.svg',
                   'sampled_edge_counts')
        self.check('directional_recruitment_hexagon', '_final.svg',
                   'sampled_frame_indices')
        self.check('route_preserving_intervention_ring', '_final.svg',
                   'frame_indices')
        self.check('route_preserving_intervention_kagome', '_final.svg',
                   'frame_indices')
        readme = (self.root / 'docs' / 'README_VNEXT.md').read_text(
            encoding='utf-8')
        for asset in ('spectrum_four_topologies.gif',
                      'tensile_recruitment_kagome.gif',
                      'tensile_threshold_sensitivity.gif',
                      'continuous_route_kagome.gif',
                      'directional_recruitment_hexagon.gif',
                      'route_preserving_intervention_ring.gif',
                      'route_preserving_intervention_kagome.gif'):
            assert ('media/' + asset) in readme
        print('[media_audit] PASS')


if __name__ == '__main__':
    MediaAudit().run()

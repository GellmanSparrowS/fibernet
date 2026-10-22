"""Correct an axis name in saved frames; preserve every pixel outside its text box."""
from pathlib import Path
import argparse, hashlib, json, xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROI = (820, 485, 1005, 508)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def outside_hash(array):
    a = array.copy()
    x0, y0, x1, y1 = ROI
    a[y0:y1, x0:x1] = 0
    return hashlib.sha256(a.tobytes()).hexdigest()

def main(root, out, preview=None):
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    names = ['docs/media/tensile_recruitment_kagome.gif', 'docs/media/tensile_recruitment_kagome_final.svg', 'docs/media/tensile_recruitment_kagome.json', 'scripts/make_homepage_media.py', 'fibernet/sim/reduced_beam.py']
    hashes = {n: sha(root / n) for n in names}
    meta = json.loads((root / names[2]).read_text())
    assert hashes[names[0]] == meta['gif_sha256']
    core = (root / names[4]).read_text()
    assert 'strain_levels=(np.array(rec["spans"]) / span0)' in core
    original = Image.open(root / names[0])
    assert original.size == (1150, 552) and original.n_frames == 18
    assert len(meta['sampled_frame_indices']) == original.n_frames
    frames = []
    records = []
    durations = []
    original_rgb = []
    x0, y0, x1, y1 = ROI
    for i in range(original.n_frames):
        original.seek(i)
        before = np.asarray(original.convert('RGB')).copy()
        original_rgb.append(before)
        durations.append(original.info['duration'])
        colors = np.array(sorted((c for n, c in original.convert('RGB').getcolors(1000000))), dtype=np.uint8)
        assert len(colors) <= 256
        keys = (colors[:, 0].astype(np.uint32) << 16) + (colors[:, 1].astype(np.uint32) << 8) + colors[:, 2]
        rawkeys = (before[:, :, 0].astype(np.uint32) << 16) + (before[:, :, 1].astype(np.uint32) << 8) + before[:, :, 2]
        indices = np.searchsorted(keys, rawkeys).astype(np.uint8)
        assert np.array_equal(colors[indices], before)
        patch = Image.new('RGB', (x1 - x0, y1 - y0), 'white')
        draw = ImageDraw.Draw(patch)
        draw.text((569.553488 * 1150 / 720 - x0, 313.396219 * 1150 / 720 - y0), 'Stretch ratio', font=ImageFont.truetype(FONT, 18), fill='black', anchor='ms')
        patch_rgb = np.asarray(patch)
        unique, inverse = np.unique(patch_rgb.reshape(-1, 3), axis=0, return_inverse=True)
        distance = np.sum((unique[:, None, :].astype(np.int32) - colors[None, :, :].astype(np.int32)) ** 2, axis=2)
        nearest = distance.argmin(axis=1).astype(np.uint8)
        indices[y0:y1, x0:x1] = nearest[inverse].reshape(y1 - y0, x1 - x0)
        frame = Image.fromarray(indices, mode='P')
        palette = colors.tolist() + [[0, 0, 0]] * (256 - len(colors))
        frame.putpalette([v for color in palette for v in color])
        after = np.asarray(frame.convert('RGB'))
        assert outside_hash(before) == outside_hash(after)
        changed = np.any(before != after, axis=2)
        ys, xs = np.where(changed)
        bounds = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
        assert x0 <= bounds[0] < bounds[2] <= x1 and y0 <= bounds[1] < bounds[3] <= y1
        records.append({'frame': i, 'saved_trajectory_frame': meta['sampled_frame_indices'][i], 'duration_ms': durations[-1], 'outside_label_rgb_sha256': outside_hash(before), 'changed_pixel_bounds': bounds, 'original_palette_colors': len(colors)})
        frames.append(frame)
    target = out / 'tensile_recruitment_labeled.gif'
    frames[0].save(target, save_all=True, append_images=frames[1:], duration=durations, loop=original.info['loop'], disposal=2, optimize=False, include_color_table=True)
    with Image.open(target) as decoded:
        assert decoded.size == original.size and decoded.n_frames == 18 and (decoded.info['loop'] == original.info['loop'])
        for i, before in enumerate(original_rgb):
            decoded.seek(i)
            after = np.asarray(decoded.convert('RGB'))
            assert decoded.info['duration'] == durations[i]
            assert outside_hash(after) == outside_hash(before)
            assert np.array_equal(after, np.asarray(frames[i].convert('RGB')))
            if preview is not None and i in (0, 8, 17):
                Image.fromarray(after).save(preview / f'tensile_frame_{i:02d}.png')
                for w in [800, 322]:
                    Image.fromarray(after).resize((w, round(552 * w / 1150)), Image.Resampling.LANCZOS).save(preview / f'tensile_frame_{i:02d}_{w}.png')
    raw = (root / names[1]).read_bytes()
    old = b'>Engineering strain</text>'
    new = b'>Stretch ratio</text>'
    assert raw.count(old) == 1
    svg = out / 'tensile_recruitment_labeled_final.svg'
    svg.write_bytes(raw.replace(old, new))
    assert svg.read_bytes().replace(new, old) == raw

    def paths(b):
        return [e.attrib.get('d') for e in ET.fromstring(b).iter() if e.tag.endswith('}path')]
    assert paths(raw) == paths(svg.read_bytes())
    assert hashes == {n: sha(root / n) for n in names}
    receipt = {'kind': 'label_only_saved_trajectory_derivative', 'axis_before': 'Engineering strain', 'axis_after': 'Stretch ratio', 'definition': 'recorded span / initial span', 'definition_source': 'fibernet/sim/reduced_beam.py: strain_levels=(np.array(rec["spans"]) / span0)', 'source_sha256': hashes, 'label_rectangle_px': list(ROI), 'canvas': [1150, 552], 'frames': records, 'gif_frames': 18, 'duration_ms': sum(durations), 'loop': original.info['loop'], 'outside_label_pixels_byte_identical_after_gif_decode': True, 'svg_only_label_string_changed': True, 'svg_path_count': len(paths(raw)), 'original_sources_unchanged': True, 'no_simulation_analysis_or_training_execution': True, 'gif_sha256': sha(target), 'svg_sha256': sha(svg), 'renderer_sha256': sha(__file__), 'font_sha256': sha(FONT)}
    (out / 'tensile_label_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if k not in ['frames', 'source_sha256']}, indent=2))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--preview-dir', type=Path)
    a = p.parse_args()
    main(a.source_root.resolve(), a.output_dir.resolve(), a.preview_dir)

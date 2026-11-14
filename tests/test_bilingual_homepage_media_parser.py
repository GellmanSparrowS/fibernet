"""Regression controls for displayed-image source counting, not visual rendering."""
from pathlib import Path
import importlib.util
import json
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "homepage_media_check", ROOT / "scripts/check_bilingual_homepage.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)

POSITIVE = {
    "markdown": "![x](x.gif)",
    "double_quotes": '<img src="x.gif">',
    "single_quotes": "<img src='x.gif'>",
    "whitespace_equals": '<img src = "x.gif">',
    "tabs_equals": "<img src\t=\t'x.gif'>",
    "uppercase_tag_attribute": '<IMG SRC="x.gif">',
    "self_closing": '<img src="x.gif" />',
    "unquoted_src": '<img src=x.gif>',
    "real_src_with_data_src": '<img data-src="other.gif" src="x.gif">',
    "unmatched_backtick": '` unmatched <img src="x.gif">',
}
NEGATIVE = {
    "data_src_only": '<img data-src="x.gif">',
    "non_img_src": '<div src="x.gif"></div>',
    "commented_html": '<!-- <img src="x.gif"> -->',
    "commented_markdown": '<!-- ![x](x.gif) -->',
    "fenced_html": '```html\n<img src="x.gif">\n```',
    "fenced_markdown": '```markdown\n![x](x.gif)\n```',
    "tilde_fenced_html": '~~~html\n<img src="x.gif">\n~~~',
    "long_fence_short_inner": '````html\n```\n<img src="x.gif">\n````',
    "unclosed_fence": '```html\n<img src="x.gif">',
    "inline_html": '`<img src="x.gif">`',
    "inline_markdown": '`![x](x.gif)`',
    "double_backtick_html": '``a ` <img src="x.gif">``',
    "double_backtick_markdown": '``a ` ![x](x.gif)``',
    "multiline_inline": '`start\n<img src="x.gif">\nend`',
    "html_link_only": '<a href="x.gif">x</a>',
    "markdown_link_only": '[x](x.gif)',
    "markdown_in_attribute": '<div title="![x](x.gif)"></div>',
}


class InlineGifParserTests(unittest.TestCase):
    def test_valid_inline_images(self):
        for name, text in POSITIVE.items():
            with self.subTest(name=name):
                self.assertEqual(CHECK.inline_gifs(text), ['x.gif'])

    def test_non_displayed_images_do_not_count(self):
        for name, text in NEGATIVE.items():
            with self.subTest(name=name):
                self.assertEqual(CHECK.inline_gifs(text), [])

    def test_mixed_syntax_preserves_duplicates(self):
        self.assertEqual(CHECK.inline_gifs('<img src="x.gif"> ![x](x.gif)'),
                         ['x.gif', 'x.gif'])

    def test_code_does_not_hide_following_image(self):
        for code in ['```html\n<img src="hidden.gif">\n```',
                     '~~~\n![x](hidden.gif)\n~~~', '`![x](hidden.gif)`']:
            with self.subTest(code=code):
                self.assertEqual(CHECK.inline_gifs(code + '\n<img src="x.gif">'),
                                 ['x.gif'])

    def test_actual_homepage_gate(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        chinese = (ROOT / 'README_CN.md').read_text(encoding='utf-8')
        english, _ = CHECK.check_locale_structure(readme, chinese)
        image = re.search(r'<img[^>]+src="([^"]+\.gif)"[^>]*>', english)
        self.assertIsNotNone(image)
        target, src = image[0], image[1]
        for name, text in NEGATIVE.items():
            bad = english.replace(target, text.replace('x.gif', src), 1)
            # A fence can hide nested API controls before a later code fence
            # closes it, so either unbalanced-control direction may be found.
            message = ('Unclosed details control|Unmatched details close' if name == 'unclosed_fence'
                       else 'inline animations must match the homepage selection')
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    CHECK.check_locale_structure(readme.replace(english, bad), chinese)
        for name, text in POSITIVE.items():
            good = english.replace(target, text.replace('x.gif', src), 1)
            with self.subTest(name=name):
                CHECK.check_locale_structure(readme.replace(english, good), chinese)
        for name, bad in [('missing', english.replace(target, '', 1)),
                          ('duplicate', english + '\n' + target),
                          ('substituted', english.replace(src, 'unselected.gif', 1))]:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, 'inline animations must match the homepage selection'):
                    CHECK.check_locale_structure(readme.replace(english, bad), chinese)


class NativeDisclosureParserTests(unittest.TestCase):
    def test_outer_bodies_preserve_nested_disclosures(self):
        nested = ('before<details><summary><strong>Branch</strong></summary>'
                  '<details><summary>API</summary>handoff</details></details>after')
        text = ('<details><summary>中文</summary>' + nested + '</details>'
                '<details open><summary>English</summary>' + nested + '</details>')
        self.assertEqual(CHECK.top_level_disclosures(text),
                         [('', '中文', nested), (' open', 'English', nested)])

    def test_commented_and_code_controls_are_not_disclosures(self):
        fake = '<details><summary>not a control</summary>hidden</details>'
        for literal in ('<!-- ' + fake + ' -->', '`' + fake + '`',
                        '```html\n' + fake + '\n```', '~~~html\n' + fake + '\n~~~',
                        '````html\n```\n' + fake + '\n````'):
            text = literal + '\n<details open><summary>English</summary>body</details>'
            with self.subTest(literal=literal):
                self.assertEqual(CHECK.top_level_disclosures(text),
                                 [(' open', 'English', 'body')])

    def test_code_mask_preserves_original_offsets(self):
        original = 'prefix\n```html\n<details>\n```\n`<summary>`\nend'
        masked = CHECK._without_markdown_code(original, preserve_offsets=True)
        self.assertEqual(len(masked), len(original))
        self.assertEqual([i for i, c in enumerate(masked) if c == '\n'],
                         [i for i, c in enumerate(original) if c == '\n'])
        self.assertNotIn('<details>', masked)
        self.assertNotIn('<summary>', masked)

    def test_malformed_nested_controls_reject(self):
        invalid = (
            '</details>', '<details><summary>A</summary>',
            '<details>missing summary</details>', '<summary>A</summary>',
            '<details><summary>A</details>',
            '<details><summary>A<details><summary>B</summary></details></summary></details>',
            '<details><summary>A</summary><summary>B</summary></details>',
            '<details><summary>A</summary><details><summary>B</summary></details>',
            '<details/><summary>A</summary>', '<details><summary/></details>',
        )
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                CHECK.top_level_disclosures(text)

    def test_language_count_and_defaults_remain_strict(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        chinese = (ROOT / 'README_CN.md').read_text(encoding='utf-8')
        self.assertEqual(len(CHECK.top_level_disclosures(readme)), 2)
        for bad in (readme + '<details><summary>Third</summary>x</details>',
                    readme.replace('<details open>\n<summary>English</summary>',
                                   '<details>\n<summary>English</summary>', 1)):
            with self.subTest(change=bad[-80:]), self.assertRaises(ValueError):
                CHECK.check_locale_structure(bad, chinese)

    def test_exact_new_localized_asset_pair(self):
        en = 'docs/media/ui/api-relationship-map-en.svg'
        zh = 'docs/media/ui/api-relationship-map-zh.svg'
        self.assertEqual(CHECK.canonical_destination(en), CHECK.canonical_destination(zh))
        self.assertNotEqual(CHECK.canonical_destination('docs/media/ui/unrelated-zh.svg'),
                            CHECK.canonical_destination('docs/media/ui/unrelated-en.svg'))

    def test_standalone_chinese_drift_still_rejects(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        chinese = (ROOT / 'README_CN.md').read_text(encoding='utf-8')
        altered = chinese.replace('<!-- locale:zh:start -->',
                                  '<!-- locale:zh:start -->changed', 1)
        with self.assertRaisesRegex(ValueError, 'Standalone Chinese differs'):
            CHECK.check_locale_structure(readme, altered)


class RestoredGalleryTests(unittest.TestCase):
    def test_all_originals_are_visible_without_an_extra_gallery_fold(self):
        selection = json.loads((ROOT / 'docs/media/homepage_overview/selection.json').read_text(encoding='utf-8'))
        originals = {record['path'] for record in selection['original_gifs']}
        self.assertEqual(len(originals), 19)
        homepage = (ROOT / 'README.md').read_text(encoding='utf-8')
        for language in ('en', 'zh'):
            body = CHECK.locale_block(homepage, language)
            # Only API documentation is nested; every GIF precedes that fold.
            visible = body.split('<details>', 1)[0]
            with self.subTest(language=language):
                self.assertTrue(originals <= set(CHECK.inline_gifs(visible)))
                self.assertEqual(len(CHECK.inline_gifs(visible)), 22)
                quick_start = '## Quick start details' if language == 'en' else '## 快速开始'
                self.assertLess(body.index(quick_start), body.index('<!-- api-navigation:'))

    def test_brand_quick_start_is_outside_both_language_folds(self):
        homepage = (ROOT / 'README.md').read_text(encoding='utf-8')
        prefix = homepage.split('<details>', 1)[0]
        self.assertIn('<a href="#quick-start">', prefix)
        self.assertEqual(homepage.count('\n## Quick start\n'), 1)
        self.assertIn('\n## Quick start\n', prefix)
        self.assertIn('python -m pip install -e .', prefix)
        self.assertIn('python -m examples.open_source_workflow', prefix)

    def test_rl_compatibility_is_localized_and_not_an_english_footer(self):
        homepage = (ROOT / 'README.md').read_text(encoding='utf-8')
        for language, heading in (('en', '### RL compatibility'), ('zh', '### RL 兼容性')):
            body = CHECK.locale_block(homepage, language)
            with self.subTest(language=language):
                self.assertIn(heading, body)
                self.assertIn('`fibernet.train_rl`', body)
                self.assertIn('`fibernet.rl.train_rl`', body)
        suffix = homepage.split('<!-- locale:en:end -->', 1)[1].strip()
        self.assertEqual(suffix, '</details>')

    def test_scope_labels_preserve_scientific_distinctions(self):
        homepage = (ROOT / 'README.md').read_text(encoding='utf-8')
        required = {
            'en': ('undeformed geometry', 'fixed geometry', 'displacement ×1,000',
                   '×3,000', 'Prescribed actions, no trained policy',
                   'current `train_gnn` does not accept `GINERegressor`',
                   'That trajectory is absent from this media bundle',
                   '1.0042%', '2.2234%', 'Saved interface tour', 'rotating camera'),
            'zh': ('未变形几何', '固定几何', '1,000 倍', '3,000 倍',
                   '给定动作，未训练策略', '当前 `train_gnn` 不接受 `GINERegressor`',
                   '当前媒体包缺少该轨迹', '1.0042%', '2.2234%',
                   '已保存的界面导览', '相机旋转'),
        }
        for language, labels in required.items():
            body = CHECK.locale_block(homepage, language)
            for label in labels:
                with self.subTest(language=language, label=label):
                    self.assertIn(label, body)

    def test_viscoelastic_caption_describes_saved_free_dynamics(self):
        homepage = (ROOT / 'README.md').read_text(encoding='utf-8')
        expected = {
            'en': ('### Viscoelastic dynamics', '### Viscoelastic relaxation',
                   'Initial velocities excite free motion, with zero constant external load and zero numerical drag.'),
            'zh': ('### 黏弹性动力学', '### 黏弹松弛',
                   '初始速度激发自由运动，恒定外载荷和数值阻力均为零。'),
        }
        for language, (heading, incorrect_heading, excitation) in expected.items():
            body = CHECK.locale_block(homepage, language)
            with self.subTest(language=language):
                self.assertIn(heading, body)
                self.assertNotIn(incorrect_heading, body)
                section = body.split(heading, 1)[1].split('\n### ', 1)[0]
                self.assertIn(excitation, section)
                self.assertIn('docs/media/new/viscoelastic/README.md', section)

    def test_named_github_anchors_are_checked_but_literals_are_not(self):
        source = ('<a name="api-en-graph"></a>\n<a id="sample"></a>\n'
                  '<!-- <a name="comment"></a> -->\n'
                  '`<a name="inline"></a>`\n'
                  '```html\n<a name="fenced"></a>\n```')
        self.assertEqual(CHECK.explicit_anchors(source),
                         {'api-en-graph', 'user-content-api-en-graph',
                          'sample', 'user-content-sample'})


if __name__ == '__main__':
    unittest.main()

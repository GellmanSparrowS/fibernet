"""Regression controls for displayed-image source counting, not visual rendering."""
from pathlib import Path
import importlib.util
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
            # An unclosed fence masks the language closing tag, so malformed
            # disclosure structure is rejected before image-count validation.
            message = ('Unclosed details control' if name == 'unclosed_fence'
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


if __name__ == '__main__':
    unittest.main()

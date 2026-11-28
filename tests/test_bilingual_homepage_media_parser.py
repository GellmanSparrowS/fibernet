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
        CHECK.check_locale_structure(readme, chinese)
        visible = readme.rsplit("</details>", 1)[1]
        image = re.search(r'<img[^>]+src="([^"]+\.webp)"[^>]*>', visible)
        self.assertIsNotNone(image)
        target, src = image[0], image[1]
        for name, text in NEGATIVE.items():
            bad = visible.replace(target, text.replace('x.gif', src), 1)
            # A fence can hide nested API controls before a later code fence
            # closes it, so either unbalanced-control direction may be found.
            message = 'Expected all workflow previews outside closed language sections'
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    CHECK.check_locale_structure(readme.replace(visible, bad), chinese)
        for name, text in POSITIVE.items():
            good = visible.replace(target, text.replace('x.gif', src), 1)
            with self.subTest(name=name):
                CHECK.check_locale_structure(readme.replace(visible, good), chinese)
        for name, bad in [('missing', visible.replace(target, '', 1)),
                          ('duplicate', visible + '\n' + target),
                          ('substituted', visible.replace(src, 'unselected.webp', 1))]:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, 'Expected all workflow previews outside closed language sections'):
                    CHECK.check_locale_structure(readme.replace(visible, bad), chinese)


    def test_no_preview_is_hidden_inside_language_disclosure(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        chinese = (ROOT / 'README_CN.md').read_text(encoding='utf-8')
        flow = f'<img src="{CHECK.VISIBLE_MOTION[-1]}">'
        duplicated = readme.replace('<!-- locale:en:start -->', '<!-- locale:en:start -->\n'+flow+'\n', 1)
        with self.assertRaisesRegex(ValueError, 'must not be duplicated'):
            CHECK.check_locale_structure(duplicated, chinese)


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


class PublicHomepageTests(unittest.TestCase):
    def test_shared_visible_animations_and_bilingual_links_to_detailed_docs(self):
        selection = json.loads((ROOT / 'docs/media/homepage_overview/selection.json').read_text(encoding='utf-8'))
        originals = {record['path'] for record in selection['original_gifs']}
        self.assertEqual(len(originals), 19)
        self.assertTrue(all((ROOT / path).is_file() for path in originals))
        homepage = (ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertEqual(CHECK.inline_images(homepage.rsplit('</details>', 1)[1], '.webp'),
                         list(CHECK.VISIBLE_MOTION))
        for language in ('en', 'zh'):
            body = CHECK.locale_block(homepage, language)
            with self.subTest(language=language):
                self.assertLess(len(body.splitlines()), 60)
                self.assertEqual(CHECK.inline_images(body, '.webp'), [])
                self.assertIn('docs/MEDIA_GALLERY.md', CHECK.destinations(body))
                self.assertIn('docs/api/WORKFLOW_MAP.md', CHECK.destinations(body))
                self.assertIn('docs/DEFORMED_TRANSPORT_PREPARATION.md', CHECK.destinations(body))
                self.assertNotIn('api-navigation:', body)
        self.assertEqual(selection['displayed_gifs'], list(CHECK.HISTORICAL_HOMEPAGE_GIFS))

    def test_fast_path_uses_installed_package_before_folding_source_commands(self):
        homepage = (ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertIn('](#quick-start)', homepage.split('<details>', 1)[0])
        self.assertIn('quick-start', CHECK.headings(homepage))
        for language in ('en', 'zh'):
            body = CHECK.locale_block(homepage, language)
            visible, source = body.split('<details>', 1)
            self.assertIn('python -m pip install fibernet', visible)
            self.assertIn('python -m fibernet.examples.basic_usage', visible)
            self.assertNotIn('git clone', visible)
            self.assertIn('python -m pip install -e .', source)
            self.assertIn('python -m examples.open_source_workflow', source)

    def test_public_entrypoints_do_not_link_working_material(self):
        docs = ('README.md', 'README_CN.md', 'docs/README_VNEXT.md', 'docs/RELEASE_4_2_0.md')
        for doc in docs:
            content = (ROOT / doc).read_text(encoding='utf-8')
            with self.subTest(path=doc):
                for name in ('manuscript/', 'handoff/', 'PAPER_POSITIONING_2026-09-24.md'):
                    self.assertNotIn(name, content)

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

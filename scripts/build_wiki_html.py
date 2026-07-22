#!/usr/bin/env python3
"""Convert wiki markdown files to HTML for GitHub Pages.

Usage:
    python scripts/build_wiki_html.py

Reads docs/wiki/*.md and outputs docs/wiki/*.html
"""
import re
import os
from pathlib import Path

DOCS_WIKI = Path(__file__).parent.parent / "docs" / "wiki"

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — FiberNet Wiki</title>
<style>
  :root {{ --bg: #fafafa; --fg: #1a1a2e; --accent: #2c3e50; --link: #2980b9; --border: #e0e0e0; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Georgia', 'Times New Roman', serif; color: var(--fg); background: var(--bg); line-height: 1.7; }}
  .container {{ max-width: 860px; margin: 0 auto; padding: 2rem 1.5rem; }}
  .breadcrumb {{ font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 0.85rem; color: #888; margin-bottom: 1.5rem; }}
  .breadcrumb a {{ color: var(--link); text-decoration: none; }}
  .breadcrumb a:hover {{ text-decoration: underline; }}
  h1 {{ font-size: 2rem; color: var(--accent); margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid var(--accent); }}
  h2 {{ font-size: 1.4rem; color: var(--accent); margin: 2rem 0 0.8rem; padding-bottom: 0.3rem; border-bottom: 1px solid var(--border); }}
  h3 {{ font-size: 1.1rem; color: #333; margin: 1.5rem 0 0.5rem; }}
  p {{ margin-bottom: 0.8rem; }}
  a {{ color: var(--link); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  code {{ font-family: 'Courier New', monospace; background: #f0f0f0; padding: 0.1em 0.3em; border-radius: 2px; font-size: 0.9em; }}
  pre {{ background: #2c3e50; color: #ecf0f1; padding: 1rem; border-radius: 4px; overflow-x: auto; margin: 0.8rem 0; }}
  pre code {{ background: none; padding: 0; color: inherit; font-size: 0.85rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
  th, td {{ border: 1px solid var(--border); padding: 0.5rem 0.8rem; text-align: left; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  ul, ol {{ margin: 0.5rem 0 1rem 1.5rem; }}
  li {{ margin-bottom: 0.3rem; }}
  strong {{ font-weight: 700; }}
  hr {{ border: none; border-top: 1px solid var(--border); margin: 2rem 0; }}
  footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); text-align: center; font-size: 0.8rem; color: #888; }}
</style>
</head>
<body>
<div class="container">
<div class="breadcrumb"><a href="../index.html">FiberNet</a> &rsaquo; Wiki &rsaquo; {title}</div>
{content}
<footer>FiberNet v4.0.5 · MIT License · ML-BioMat Lab, BMG-FDU</footer>
</div>
</body>
</html>"""


def md_to_html(md_text):
    lines = md_text.split('\n')
    html_lines = []
    in_code = False
    in_table = False
    in_list = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('```'):
            if in_code:
                html_lines.append('</code></pre>')
                in_code = False
            else:
                html_lines.append('<pre><code>')
                in_code = True
            continue
        if in_code:
            html_lines.append(stripped.replace('<', '&lt;').replace('>', '&gt;'))
            continue
        if in_list and not stripped.startswith('-') and not stripped.startswith('*'):
            html_lines.append('</ul>')
            in_list = False
        if in_table and '|' not in stripped:
            html_lines.append('</tbody></table>')
            in_table = False
        if not stripped:
            continue
        if stripped == '---':
            html_lines.append('<hr>')
            continue
        m = re.match(r'^(#{1,4})\s+(.*)', stripped)
        if m:
            level = len(m.group(1))
            text = _inline(m.group(2))
            html_lines.append(f'<h{level}>{text}</h{level}>')
            continue
        if '|' in stripped and stripped.startswith('|'):
            cells = [c.strip() for c in stripped.split('|')[1:-1]]
            if not in_table:
                html_lines.append('<table><thead><tr>')
                for c in cells: html_lines.append(f'<th>{_inline(c)}</th>')
                html_lines.append('</tr></thead><tbody>')
                in_table = True
            elif all(re.match(r'^[-:]+$', c) for c in cells):
                continue
            else:
                html_lines.append('<tr>')
                for c in cells: html_lines.append(f'<td>{_inline(c)}</td>')
                html_lines.append('</tr>')
            continue
        if stripped.startswith('- ') or stripped.startswith('* '):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            html_lines.append(f'<li>{_inline(stripped[2:])}</li>')
            continue
        html_lines.append(f'<p>{_inline(stripped)}</p>')
    if in_list: html_lines.append('</ul>')
    if in_table: html_lines.append('</tbody></table>')
    if in_code: html_lines.append('</code></pre>')
    return '\n'.join(html_lines)


def _inline(text):
    text = re.sub(r'\[\[([^\]]+)\]\]', lambda m: f'<a href="{m.group(1).replace(" ", "-")}.html">{m.group(1)}</a>', text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    return text


def main():
    skip = {'_Sidebar.md', '_Footer.md', 'index.md'}
    for md_file in sorted(DOCS_WIKI.glob('*.md')):
        if md_file.name in skip:
            continue
        md_text = md_file.read_text()
        title = md_text.split('\n')[0].lstrip('#').strip()
        html = TEMPLATE.format(title=title, content=md_to_html(md_text))
        (DOCS_WIKI / (md_file.stem + '.html')).write_text(html)
        print(f"  {md_file.name} -> {md_file.stem}.html")
    print("Done.")


if __name__ == '__main__':
    main()

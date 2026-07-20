#!/usr/bin/env python3
"""Convert wiki markdown files to HTML for GitHub Pages.

Usage:
    python scripts/build_wiki_html.py

Reads docs/wiki/*.md and outputs docs/wiki/*.html
"""
import re
import os
import sys
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
  h4 {{ font-size: 1rem; color: #444; margin: 1rem 0 0.4rem; }}
  p {{ margin-bottom: 0.8rem; font-size: 1rem; }}
  a {{ color: var(--link); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  code {{ font-family: 'Courier New', monospace; background: #f0f0f0; padding: 0.1em 0.3em; border-radius: 2px; font-size: 0.9em; }}
  pre {{ background: #2c3e50; color: #ecf0f1; padding: 1rem 1.2rem; border-radius: 4px; overflow-x: auto; margin: 0.8rem 0; line-height: 1.5; }}
  pre code {{ background: none; padding: 0; color: inherit; font-size: 0.85rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
  th, td {{ border: 1px solid var(--border); padding: 0.5rem 0.8rem; text-align: left; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  ul, ol {{ margin: 0.5rem 0 1rem 1.5rem; }}
  li {{ margin-bottom: 0.3rem; font-size: 0.95rem; }}
  strong {{ font-weight: 700; }}
  blockquote {{ border-left: 3px solid var(--accent); padding-left: 1rem; margin: 1rem 0; color: #555; font-style: italic; }}
  hr {{ border: none; border-top: 1px solid var(--border); margin: 2rem 0; }}
  .nav-links {{ display: flex; justify-content: space-between; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border); font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 0.9rem; }}
  footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); text-align: center; font-size: 0.8rem; color: #888; font-family: 'Helvetica Neue', Arial, sans-serif; }}
  @media (max-width: 600px) {{ h1 {{ font-size: 1.5rem; }} pre {{ font-size: 0.8rem; }} }}
</style>
</head>
<body>
<div class="container">
<div class="breadcrumb"><a href="../index.html">FiberNet</a> &rsaquo; <a href="../index.html">Wiki</a> &rsaquo; {title}</div>
{content}
<div class="nav-links">
  <a href="../index.html">&larr; Back to Home</a>
  <a href="https://github.com/GellmanSparrowS/fibernet/wiki">View on GitHub Wiki &rarr;</a>
</div>
<footer>FiberNet v4.0.5 &middot; MIT License &middot; ML-BioMat Lab, BMG-FDU</footer>
</div>
</body>
</html>"""


def md_to_html(md_text):
    """Simple markdown to HTML converter."""
    lines = md_text.split('\n')
    html_lines = []
    in_code_block = False
    in_table = False
    in_list = False
    list_type = None
    code_lang = ""

    for i, line in enumerate(lines):
        # Code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                html_lines.append('</code></pre>')
                in_code_block = False
            else:
                code_lang = line.strip()[3:].strip()
                html_lines.append('<pre><code>')
                in_code_block = True
            continue
        if in_code_block:
            html_lines.append(line.replace('<', '&lt;').replace('>', '&gt;'))
            continue

        # Close open list
        stripped = line.strip()
        if in_list and not stripped.startswith('-') and not stripped.startswith('*') and not re.match(r'^\d+\.', stripped):
            if list_type == 'ul':
                html_lines.append('</ul>')
            else:
                html_lines.append('</ol>')
            in_list = False
            list_type = None

        # Close table
        if in_table and '|' not in stripped:
            html_lines.append('</tbody></table>')
            in_table = False

        # Empty line
        if not stripped:
            html_lines.append('')
            continue

        # Horizontal rule
        if stripped == '---':
            html_lines.append('<hr>')
            continue

        # Headers
        m = re.match(r'^(#{1,4})\s+(.*)', stripped)
        if m:
            level = len(m.group(1))
            text = inline_format(m.group(2))
            html_lines.append(f'<h{level}>{text}</h{level}>')
            continue

        # Table
        if '|' in stripped and stripped.startswith('|'):
            cells = [c.strip() for c in stripped.split('|')[1:-1]]
            if not in_table:
                html_lines.append('<table><thead><tr>')
                for c in cells:
                    html_lines.append(f'<th>{inline_format(c)}</th>')
                html_lines.append('</tr></thead><tbody>')
                in_table = True
            elif all(re.match(r'^[-:]+$', c) for c in cells):
                continue  # separator row
            else:
                html_lines.append('<tr>')
                for c in cells:
                    html_lines.append(f'<td>{inline_format(c)}</td>')
                html_lines.append('</tr>')
            continue

        # Lists
        if stripped.startswith('- ') or stripped.startswith('* '):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
                list_type = 'ul'
            html_lines.append(f'<li>{inline_format(stripped[2:])}</li>')
            continue
        m_list = re.match(r'^(\d+)\.\s+(.*)', stripped)
        if m_list:
            if not in_list:
                html_lines.append('<ol>')
                in_list = True
                list_type = 'ol'
            html_lines.append(f'<li>{inline_format(m_list.group(2))}</li>')
            continue

        # Paragraph
        html_lines.append(f'<p>{inline_format(stripped)}</p>')

    # Close any open blocks
    if in_list:
        html_lines.append('</ul>' if list_type == 'ul' else '</ol>')
    if in_table:
        html_lines.append('</tbody></table>')
    if in_code_block:
        html_lines.append('</code></pre>')

    return '\n'.join(html_lines)


def inline_format(text):
    """Handle inline formatting: bold, code, links, wiki links."""
    # Wiki links [[Page Name]]
    text = re.sub(r'\[\[([^\]]+)\]\]', lambda m: f'<a href="{m.group(1).replace(" ", "-")}.html">{m.group(1)}</a>', text)
    # Inline code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # Bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    # Links [text](url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    return text


def main():
    wiki_dir = DOCS_WIKI
    skip_files = {'_Sidebar.md', '_Footer.md', 'index.md'}

    for md_file in sorted(wiki_dir.glob('*.md')):
        if md_file.name in skip_files:
            continue
        md_text = md_file.read_text()
        title = md_text.split('\n')[0].lstrip('#').strip()
        html_content = md_to_html(md_text)
        html_full = TEMPLATE.format(title=title, content=html_content)
        html_name = md_file.stem + '.html'
        out_path = wiki_dir / html_name
        out_path.write_text(html_full)
        print(f"  {md_file.name} -> {html_name}")

    print(f"\nDone. HTML files in {wiki_dir}")


if __name__ == '__main__':
    main()

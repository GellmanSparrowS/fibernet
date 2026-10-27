#!/usr/bin/env python3
"""Maintain the selective README API guide using the existing module registry.

Standard-library AST inspection only; never imports fibernet or optional runtimes.
Use --check to verify generated content, or --write after updating the overlay.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import html
import json
import math
import re
from pathlib import Path

OVERLAY = "docs/api/navigation-overlay.json"
ASSETS = ("docs/media/ui/api-relationship-map-en.svg", "docs/media/ui/api-relationship-map-zh.svg")

def declaration(tree, spec):
    owner = next((n for n in tree.body
                  if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == spec["sourceSymbol"]), None)
    if owner is None:
        raise ValueError("Source symbol missing: " + spec["registryKey"] + ":" + spec["sourceSymbol"])
    if spec["kind"] == "method":
        method = spec["publicNamespace"].rsplit(".", 1)[-1]
        owner = next((n for n in owner.body
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and n.name == method), None)
        if owner is None:
            raise ValueError("Method missing: " + spec["publicNamespace"])
    return owner.lineno


def derive(repo_root, overlay_path, commit):
    registry_path = repo_root / "code_registry.json"
    raw_registry = registry_path.read_bytes()
    registry = json.loads(raw_registry)
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    url_base = "https://github.com/GellmanSparrowS/fibernet/blob/" + commit + "/"
    nodes = []
    for spec in overlay["nodes"]:
        key = spec["registryKey"]
        source_path = repo_root / key
        raw = source_path.read_bytes()
        tree = ast.parse(raw.decode("utf-8-sig"), filename=key)
        line = declaration(tree, spec)
        record = registry.get(key)
        node = {k: spec[k] for k in ("id", "group", "kind", "label", "title",
                                    "description", "input", "output", "caveat",
                                    "availability")}
        node.update(namespace=spec["publicNamespace"], sourcePath=key,
                    sourceSymbol=spec["sourceSymbol"], sourceLine=line,
                    sourceUrl=url_base + key + "#L" + str(line),
                    sourceSha256=hashlib.sha256(raw).hexdigest(),
                    docsUrl=url_base + spec["docsPath"],
                    publicExportUrl=url_base + spec.get("exportPath", key),
                    registryRef=dict(path="code_registry.json", key=key,
                                     symbol=spec["sourceSymbol"],
                                     entryPresent=record is not None,
                                     symbolPresent=bool(record and
                                         spec["sourceSymbol"] in record.get("symbols", []))))
        if "example" in spec:
            node["example"] = spec["example"]
        nodes.append(node)
    by_id = {node["id"]: node for node in nodes}
    valid_ids = set(by_id) | {g["id"] for g in overlay["groups"]}
    if len(by_id) != len(nodes):
        raise ValueError("Duplicate node IDs")
    edges = []
    for item in overlay["edges"]:
        if item["type"] not in ("containment", "dataflow"):
            raise ValueError("Unknown relation type")
        if item["source"] not in valid_ids or item["target"] not in valid_ids:
            raise ValueError("Unknown relation endpoint")
        edge = dict(item)
        edge["sourceUrls"] = list(dict.fromkeys(
            by_id[x]["sourceUrl"] for x in (item["source"], item["target"]) if x in by_id))
        edge["sourceUrl"] = edge["sourceUrls"][-1]
        edges.append(edge)
    return dict(
        schema="fibernet.api-navigation/1",
        title={"en": "One graph. Connected APIs.", "zh": "一张结构图，串起各类 API。"},
        repository="https://github.com/GellmanSparrowS/fibernet", commit=commit,
        canonicalRegistry={
            "path": "code_registry.json",
            "sha256": hashlib.sha256(raw_registry).hexdigest(),
            "url": url_base + "code_registry.json",
            "rule": "Existing registry keys plus current-source verification; presentation overlay only."
        },
        groups=overlay["groups"], nodes=nodes, edges=edges,
        maintenance={
            "en": "The registry remains the module inventory; this overlay adds homepage selection and bilingual relationships.",
            "zh": "现有清单仍负责模块索引；展示层只补充主页选择和双语关系。"
        }
    )


def text(v,l):return v.get(l,v.get('en','')) if isinstance(v,dict) else str(v or '')
def escape(s):return html.escape(str(s or ''),quote=True)
def entry_anchor(node_id, language):
    """Stable, language-specific README destination; never derived from labels."""
    if language not in ('en', 'zh') or not re.fullmatch(r'[a-z][a-z0-9_]*', node_id):
        raise ValueError('Invalid API entry anchor')
    return 'api-' + language + '-' + node_id

def entry_href(node_id, language):
    """Target GitHub's rendered name while keeping the authored name stable."""
    return '#user-content-' + entry_anchor(node_id, language)

def relationships(data):
    """One deterministic projection of the existing nodes and typed edges."""
    nodes = {n['id']: n for n in data['nodes']}
    groups = {g['id']: i for i, g in enumerate(data['groups'])}
    handoffs = sorted((e for e in data['edges'] if e['type'] == 'dataflow'),
                      key=lambda e: (groups[nodes[e['source']]['group']],
                                     e['source'], e['target'], text(e['label'], 'en')))
    counts = {}
    for edge in handoffs:
        pair = (nodes[edge['source']]['group'], nodes[edge['target']]['group'])
        counts[pair] = counts.get(pair, 0) + 1
    pairs = [(source, target, counts[source, target]) for source, target in
             sorted(counts, key=lambda pair: (groups[pair[0]], groups[pair[1]]))]
    ownership = [e for e in data['edges'] if e['type'] == 'containment'
                 and e['source'] in nodes and e['target'] in nodes]
    return nodes, handoffs, pairs, ownership


def overview(data, l):
    """Static directed chords; interactive navigation remains native Markdown."""
    zh = l == 'zh'
    nodes, handoffs, pairs, _ = relationships(data)
    groups = data['groups']
    cross = [(s, t, count) for s, t, count in pairs if s != t]
    within = {s: count for s, t, count in pairs if s == t}
    cross_count = sum(count for _, _, count in cross)
    within_count = sum(within.values())
    title = '工作流程之间的 API 关系' if zh else 'API handoffs across workflows'
    subtitle = (f'{len(nodes)} 个代表性接口 · {len(handoffs)} 条文档关系 · {len(groups)} 类流程' if zh
                else f'{len(nodes)} selected APIs · {len(handoffs)} documented handoffs · {len(groups)} workflows')
    desc = (f'静态有向弦图。跨组关系 {cross_count} 条，组内关系 {within_count} 条；逐条说明见下方文字表。不是运行时调用图。' if zh
            else f'Static directed chord summary: {cross_count} cross-workflow and {within_count} within-workflow handoffs. The full text list follows below. Not a runtime call graph.')
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="748" viewBox="0 0 1200 748" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><metadata>Source snapshot: {escape(data["commit"])}</metadata>',
           '<defs><marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#357b70"/></marker></defs>',
           '<rect width="1200" height="748" rx="14" fill="#f6f9f8"/><g font-family="Arial,Noto Sans CJK SC,sans-serif" fill="#173d47">',
           f'<text x="32" y="42" font-size="25" font-weight="600">{escape(title)}</text><text x="32" y="70" font-size="15">{escape(subtitle)}</text>',
           '<circle cx="350" cy="390" r="170" fill="none" stroke="#c4d9d4" stroke-width="1.5"/>']
    positions = {}
    for i, group in enumerate(groups):
        angle = -math.pi / 2 + 2 * math.pi * i / len(groups)
        ux, uy = math.cos(angle), math.sin(angle)
        positions[group['id']] = (350 + 170 * ux, 390 + 170 * uy, ux, uy)
    for source, target, count in cross:
        x1, y1, _, _ = positions[source]
        x2, y2, _, _ = positions[target]
        length = math.hypot(x2 - x1, y2 - y1)
        # Opposing directions bend to opposite sides, not onto the same path.
        cx, cy = 350 - (y2 - y1) / length * 42, 390 + (x2 - x1) / length * 42
        start_length, end_length = math.hypot(cx-x1, cy-y1), math.hypot(x2-cx, y2-cy)
        x1, y1 = x1 + (cx-x1)/start_length*8, y1 + (cy-y1)/start_length*8
        x2, y2 = x2 - (x2-cx)/end_length*10, y2 - (y2-cy)/end_length*10
        svg.append(f'<path data-source="{source}" data-target="{target}" data-count="{count}" d="M{x1:.1f} {y1:.1f} Q{cx:.1f} {cy:.1f} {x2:.1f} {y2:.1f}" fill="none" stroke="#357b70" stroke-width="1.7" stroke-opacity="0.75" marker-end="url(#arrow)"/>')
    for group in groups:
        gid = group['id']
        x, y, ux, uy = positions[gid]
        count = within.get(gid, 0)
        if count:
            tx, ty = -uy, ux
            svg.append(f'<path data-source="{gid}" data-target="{gid}" data-count="{count}" d="M{x+tx*7:.1f} {y+ty*7:.1f} C{x+ux*37+tx*31:.1f} {y+uy*37+ty*31:.1f} {x+ux*37-tx*31:.1f} {y+uy*37-ty*31:.1f} {x-tx*7:.1f} {y-ty*7:.1f}" fill="none" stroke="#357b70" stroke-width="1.7" marker-end="url(#arrow)"/>')
        lx, ly = 350 + 230 * ux, 390 + 230 * uy
        anchor = 'middle' if abs(ux) < 0.1 else ('start' if ux > 0 else 'end')
        size = sum(n['group'] == gid for n in nodes.values())
        detail = f'{size} 个接口 · 组内 {count} 条' if zh else f'{size} APIs · {count} within'
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#173d47"/><text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-size="16" font-weight="600">{escape(text(group["title"], l))}</text><text x="{lx:.1f}" y="{ly+21:.1f}" text-anchor="{anchor}" font-size="13">{escape(detail)}</text>')
    heading = '跨组方向与条数' if zh else 'Cross-workflow direction and count'
    svg.append(f'<text x="748" y="133" font-size="18" font-weight="600">{heading}</text>')
    titles = {g['id']: text(g['title'], l) for g in groups}
    for index, (source, target, count) in enumerate(cross):
        label = f'{titles[source]} → {titles[target]} · {count}'
        svg.append(f'<text x="748" y="{166+index*23}" font-size="14">{escape(label)}</text>')
    notes = ([f'跨组 {cross_count} 条 + 组内 {within_count} 条 = {len(handoffs)} 条',
              '箭头表示方向；环线表示同组接口之间的关系。',
              '调用方负责工作流之间的数据衔接。',
              '逐条契约、来源与导航见下方文字表。'] if zh else
             [f'{cross_count} cross-workflow + {within_count} within = {len(handoffs)} handoffs',
              'Arrowheads show direction; loops stay within a workflow.',
              'Caller-managed data handoffs.',
              'Full contracts, sources and navigation are listed below.'])
    for index, note in enumerate(notes):
        svg.append(f'<text x="748" y="{550+index*24}" font-size="13">{escape(note)}</text>')
    footer = ('静态关系总览 · 完整接口契约见下方' if zh else
              'Static relationship overview · Detailed API contracts below')
    svg.append(f'<text x="32" y="714" font-size="12">{escape(footer)}</text></g></svg>')
    result = ''.join(svg)
    if len(result.encode('utf-8')) > 16 * 1024:
        raise ValueError('Static relationship SVG exceeds 16 KiB')
    return result


def handoff_list(data, l):
    """Complete linear alternative: one row per existing directed dataflow edge."""
    zh = l == 'zh'
    nodes, handoffs, _, ownership = relationships(data)
    summary = f'完整数据传递表 · {len(handoffs)} 条' if zh else f'All documented handoffs · {len(handoffs)} relations'
    rows = ['<details>', f'<summary>{summary}</summary>', '',
            ('每行均为有向 dataflow 关系，由调用方完成衔接；不是自动转换或运行时调用记录。来源列链接到两端接口的源码，不能替代逐条契约。' if zh else
             'Each row is a directed dataflow handoff supplied by the caller, not an automatic conversion or a runtime call. Source references identify both endpoint declarations; read the stated contract.'), '',
            ('| 来源接口 | 目标接口 | 数据传递契约 | 源码参考 |' if zh else
             '| From API | To API | Handoff contract | Source references |'),
            '| --- | --- | --- | --- |']
    def node_link(node_id):
        node = nodes[node_id]
        return f'[`{node["namespace"]}`]({entry_href(node_id, l)})'
    used = set()
    for edge in handoffs:
        source, target = edge['source'], edge['target']
        used.update((source, target))
        label = text(edge['label'], l).replace('|', r'\|').replace('\n', ' ')
        refs = f'[{"来源" if zh else "From"}][api-{l}-src-{source}] / [{"目标" if zh else "To"}][api-{l}-src-{target}]'
        rows.append(f'| {node_link(source)} | {node_link(target)} | {label} | {refs} |')
    rows += ['', ('**记录归属（containment，非数据传递）：**' if zh else
                  '**Record ownership (containment, not dataflow):**')]
    for edge in ownership:
        rows.append(f'- {node_link(edge["source"])} → {node_link(edge["target"])} — {text(edge["label"], l)}')
    rows += ['', ('另有 49 条接口与流程分组的归属关系，见下方分支；分组不表示类继承。' if zh else
                  'The 49 API-to-workflow memberships appear in the branches below; grouping is not class inheritance.'), '']
    for node_id in sorted(used):
        rows.append(f'[api-{l}-src-{node_id}]: {nodes[node_id]["sourceUrl"]}')
    return '\n'.join(rows + ['', '</details>', ''])

def section(data, l):
    zh=l=='zh';s=[]
    s += ['## '+('API 关系导航' if zh else 'API relationship map'),'']
    s += [('按 7 类工作流程浏览 49 个代表性接口。展开条目可查看准确命名空间、输入输出及数据传递关系；模块清单见 [`code_registry.json`](code_registry.json)。' if zh else 'Browse 49 representative interfaces across 7 workflows. Expand an entry for its exact namespace, inputs, outputs and documented handoffs; see [`code_registry.json`](code_registry.json) for the module inventory.'),'']
    s += [f'![{"静态有向弦图：49 个代表性接口、50 条文档关系；36 条跨组、14 条组内，完整文字表见下方。" if zh else "Static directed chord summary: 49 selected APIs, 50 documented handoffs; 36 cross-workflow and 14 within-workflow. Full text list below."}](docs/media/ui/api-relationship-map-{l}.svg)','']
    s += [('上图是静态有向弦图，不提供图内点击，也不是运行时调用图。下方原生折叠条目提供接口说明和关系链接。' if zh else 'The image is a static directed chord summary, not clickable navigation or a runtime call graph. Native disclosures below provide interface details and relationship links.'),'']
    s += [handoff_list(data, l), '']
    for g in data['groups']:
        nodes=[n for n in data['nodes'] if n['group']==g['id']]
        s += ['<details>',f'<summary><strong>{escape(text(g["title"],l))}</strong> · {len(nodes)} {"个接口" if zh else "interfaces"}</summary>','']
        for n in nodes:
            s += [f'<a name="{entry_anchor(n["id"], l)}"></a>','']
            s += ['<details>',f'<summary><code>{escape(n["label"])}</code> — {escape(text(n["title"],l))}</summary>','',text(n['description'],l),'',f'- **{"命名空间" if zh else "Namespace"}:** `{n["namespace"]}`',f'- **{"输入" if zh else "Input"}:** {text(n["input"],l)}',f'- **{"输出" if zh else "Output"}:** {text(n["output"],l)}','']
            if n.get('example'):s += ['```python',n['example'],'```','']
            if text(n.get('caveat'),l):s += ['> '+text(n['caveat'],l),'']
            links=[e for e in data['edges'] if e['type']=='dataflow' and n['id'] in (e['source'],e['target'])]
            if links:
                s += ['<details>',f'<summary>{"文档中的数据传递" if zh else "Documented handoffs"} · {len(links)}</summary>','']
                for edge in links:
                    other=next(x for x in data['nodes'] if x['id']==(edge['target'] if edge['source']==n['id'] else edge['source']))
                    direction='→' if edge['source']==n['id'] else '←'
                    s += [f'- {direction} [`{other["namespace"]}`]({entry_href(other["id"], l)}) — {text(edge["label"],l)} · [{"源码" if zh else "Source"}]({other["sourceUrl"]})']
                s += ['','</details>','']
            s += [f'[{"源码" if zh else "Source"}]({n["sourceUrl"]}) · [{"公共导出" if zh else "Public export"}]({n["publicExportUrl"]})'+(f' · [{"文档" if zh else "Documentation"}]({n["docsUrl"]})' if n.get('docsUrl') else ''),'','</details>','']
        s += ['</details>','']
    s += [('分支表示工作流程分组，不表示继承关系。关系链接指向本语言区的接口条目，源码链接单独保留。若目标条目仍被折叠，请展开对应分组；数据传递关系说明需要由调用方完成的衔接。' if zh else 'Branches represent workflows, not inheritance. Relationship links target entries in this language; source links remain separate. If a destination is still hidden, expand its workflow group. Handoffs describe the connections the caller supplies.'),'']
    return '\n'.join(s)


def strip_navigation(readme):
    """Remove only our marked insertions; preserve every surrounding byte."""
    for language in ("en", "zh"):
        begin = "<!-- api-navigation:" + language + ":start -->"
        end = "<!-- api-navigation:" + language + ":end -->"
        if readme.count(begin) != readme.count(end) or readme.count(begin) > 1:
            raise ValueError("Unbalanced or duplicate navigation markers")
        if begin in readme:
            start, finish = readme.index(begin), readme.index(end) + len(end)
            if readme.index(end) < start:
                raise ValueError("Navigation marker order")
            readme = readme[:start] + readme[finish:]
    return readme


def insert_section(document, language, content):
    """Insert generated content in the document's style; preserve other bytes."""
    remainder = document.replace("\r\n", "")
    if "\r" in remainder or ("\r\n" in document and "\n" in remainder):
        raise ValueError("Mixed or unsupported document line endings")
    newline = "\r\n" if "\r\n" in document else "\n"
    anchor = "<!-- locale:" + language + ":start -->"
    if document.count(anchor) != 1:
        raise ValueError("Expected exactly one locale anchor: " + language)
    begin = "<!-- api-navigation:" + language + ":start -->"
    end = "<!-- api-navigation:" + language + ":end -->"
    if begin in document or end in document:
        raise ValueError("Remove the previous navigation block before insertion")
    block = begin + "\n\n" + content + "\n" + end
    return document.replace(anchor, anchor + block.replace("\n", newline), 1)


def build_outputs(repo_root):
    repo_root = Path(repo_root)
    overlay_path = repo_root / OVERLAY
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    commit = overlay["sourceCommit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("sourceCommit must be a full lowercase Git commit")
    data = derive(repo_root, overlay_path, commit)
    readme = strip_navigation((repo_root / "README.md").read_bytes().decode("utf-8"))
    for language in ("en", "zh"):
        readme = insert_section(readme, language, section(data, language))
    chinese = strip_navigation((repo_root / "README_CN.md").read_bytes().decode("utf-8"))
    chinese = insert_section(chinese, "zh", section(data, "zh"))
    return {"README.md": readme, "README_CN.md": chinese,
            ASSETS[0]: overview(data, "en"), ASSETS[1]: overview(data, "zh")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    outputs = build_outputs(args.root)
    stale = []
    for relative, content in outputs.items():
        target = args.root / relative
        raw = content.encode("utf-8")
        if args.write:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        elif not target.exists() or target.read_bytes() != raw:
            stale.append(relative)
    if stale:
        parser.exit(1, "API navigation is stale: " + ", ".join(stale) + "\n")
    print("API navigation: " + ("updated" if args.write else "current"))


if __name__ == "__main__":
    main()

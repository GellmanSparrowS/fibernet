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
def overview(data, l):
    w,h=1200,748
    title='The API, by relationship' if l=='en' else '从关系理解 API'
    subtitle='Workflow branches · exact namespaces in the reference below' if l=='en' else '按研究流程分支 · 准确命名空间见下方可展开索引'
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc"><title id="title">{title}</title><desc id="desc">Seven workflow branches connecting the public FiberNet API. Editorial grouping, not inheritance.</desc><rect width="1200" height="748" rx="14" fill="#f6f9f8"/><g font-family="Arial,Noto Sans CJK SC,sans-serif"><text x="38" y="49" fill="#102a35" font-size="25" font-weight="600">{title}</text><text x="38" y="77" fill="#687e84" font-size="13">{subtitle}</text><path d="M168 140 H215 Q235 140 235 160 V664" fill="none" stroke="#92bfb3" stroke-width="1.5"/><text x="38" y="144" fill="#0d6c62" font-size="23" font-weight="600">FiberNet</text><text x="40" y="167" fill="#748d94" font-size="11">Python API</text>']
    for i,g in enumerate(data['groups']):
        y=140+i*86
        nodes=[n for n in data['nodes'] if n['group']==g['id']]
        labels=[n['label'] for n in nodes[:3]]
        svg.append(f'<path d="M235 {y-16 if i else y} Q235 {y} 254 {y} H278" fill="none" stroke="#92bfb3" stroke-width="1.4"/><circle cx="278" cy="{y}" r="3" fill="#438f7e"/><text x="296" y="{y-3}" fill="#173d47" font-size="17" font-weight="600">{escape(text(g["title"],l))}</text><text x="296" y="{y+18}" fill="#82979d" font-size="11">{len(nodes)} {"interfaces" if l=="en" else "个接口"}</text><path d="M492 {y} H535" stroke="#c4d9d4" stroke-width="1"/>')
        for j,label in enumerate(labels):
            yy=y-19+j*21
            svg.append(f'<path d="M535 {y} C551 {y} 546 {yy-5} 566 {yy-5}" fill="none" stroke="#c4d9d4"/><text x="578" y="{yy}" fill="#31555f" font-family="DejaVu Sans Mono,Consolas,monospace" font-size="12">{escape(label)}</text>')
    svg.append('<text x="38" y="718" fill="#7e9298" font-size="11">code_registry.json + verified public exports · source snapshot '+data['commit'][:12]+'</text></g></svg>')
    return ''.join(svg)

def section(data, l):
    zh=l=='zh';s=[]
    s += ['## '+('API 关系导航' if zh else 'API relationship map'),'']
    s += [('按 7 类工作流程浏览 49 个代表性接口。展开条目可查看准确命名空间、输入输出及数据传递关系；模块清单见 [`code_registry.json`](code_registry.json)。' if zh else 'Browse 49 representative interfaces across 7 workflows. Expand an entry for its exact namespace, inputs, outputs and documented handoffs; see [`code_registry.json`](code_registry.json) for the module inventory.'),'']
    s += [f'![{"API 关系分支概览" if zh else "API workflow branch overview"}](docs/media/ui/api-relationship-map-{l}.svg)','']
    for g in data['groups']:
        nodes=[n for n in data['nodes'] if n['group']==g['id']]
        s += ['<details>',f'<summary><strong>{escape(text(g["title"],l))}</strong> · {len(nodes)} {"个接口" if zh else "interfaces"}</summary>','']
        for n in nodes:
            s += ['<details>',f'<summary><code>{escape(n["label"])}</code> — {escape(text(n["title"],l))}</summary>','',text(n['description'],l),'',f'- **{"命名空间" if zh else "Namespace"}:** `{n["namespace"]}`',f'- **{"输入" if zh else "Input"}:** {text(n["input"],l)}',f'- **{"输出" if zh else "Output"}:** {text(n["output"],l)}','']
            if n.get('example'):s += ['```python',n['example'],'```','']
            if text(n.get('caveat'),l):s += ['> '+text(n['caveat'],l),'']
            links=[e for e in data['edges'] if e['type']=='dataflow' and n['id'] in (e['source'],e['target'])]
            if links:
                s += ['<details>',f'<summary>{"文档中的数据传递" if zh else "Documented handoffs"} · {len(links)}</summary>','']
                for edge in links:
                    other=next(x for x in data['nodes'] if x['id']==(edge['target'] if edge['source']==n['id'] else edge['source']))
                    direction='→' if edge['source']==n['id'] else '←'
                    s += [f'- {direction} [`{other["namespace"]}`]({other["sourceUrl"]}) — {text(edge["label"],l)}']
                s += ['','</details>','']
            s += [f'[{"源码" if zh else "Source"}]({n["sourceUrl"]}) · [{"公共导出" if zh else "Public export"}]({n["publicExportUrl"]})'+(f' · [{"文档" if zh else "Documentation"}]({n["docsUrl"]})' if n.get('docsUrl') else ''),'','</details>','']
        s += ['</details>','']
    s += [('分支表示工作流程分组，不表示继承关系。各条目可独立展开；数据传递关系说明需要由调用方完成的衔接。' if zh else 'Branches represent workflows, not inheritance. Entries expand independently; handoffs describe the connections the caller supplies.'),'']
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


def build_outputs(repo_root):
    repo_root = Path(repo_root)
    overlay_path = repo_root / OVERLAY
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    commit = overlay["sourceCommit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("sourceCommit must be a full lowercase Git commit")
    data = derive(repo_root, overlay_path, commit)
    readme = strip_navigation((repo_root / "README.md").read_text(encoding="utf-8"))
    for language in ("en", "zh"):
        anchor = "<!-- locale:" + language + ":start -->"
        if readme.count(anchor) != 1:
            raise ValueError("Expected exactly one locale anchor: " + language)
        block = ("<!-- api-navigation:" + language + ":start -->\n\n"
                 + section(data, language) + "\n<!-- api-navigation:" + language + ":end -->")
        readme = readme.replace(anchor, anchor + block, 1)
    return {"README.md": readme,
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

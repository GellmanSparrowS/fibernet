"""Read-only README/media checks; never import or run a physical solver.

Run from the repository root: python scripts/check_bilingual_homepage.py
Requires Pillow to decode the GIF assets. Historical SVG/checkpoint hashes may
come from CRLF files; report that distinction instead of rewriting provenance.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def headings(text: str) -> set[str]:
    """GitHub-compatible slugs for this documentation's plain headings."""
    result = set()
    for line in text.splitlines():
        if not re.match(r"^#{1,6} ", line):
            continue
        title = re.sub(r"^#+ ", "", line).strip().lower()
        title = "".join(c for c in title if not unicodedata.category(c).startswith("P") or c in "-_")
        result.add(title.replace(" ", "-"))
    return result


def destinations(text: str) -> list[str]:
    markdown = re.findall(r"\]\(([^)]+)\)", text)
    html = re.findall(r'\b(?:href|src)=["\']([^"\']+)["\']', text, re.I)
    return markdown + html


def canonical_destination(target: str) -> str:
    target = re.sub(r"(?:homepage-)?language-(?:en|zh)-(?:active|inactive)(?:-v2)?\.svg", "language-button.svg", target)
    return (target.replace("manufacturing-zh.png", "manufacturing-en.png")
            .replace("homepage-quickstart-zh-v2.svg", "homepage-quickstart-en-v2.svg"))


def check_links(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    links = destinations(text)
    require("/workspace/" not in text and "/tmp/" not in text, f"Local path in {path.name}")
    require("<script" not in text.lower(), f"JavaScript in {path.name}")
    for target in links:
        parsed = urlsplit(target)
        if parsed.scheme:
            require(parsed.scheme == "https", f"Unexpected external URL: {target}")
            continue
        linked = path.parent / unquote(parsed.path) if parsed.path else path
        require(linked.exists(), f"Missing link in {path.name}: {target}")
        if parsed.fragment and linked.suffix == ".md":
            require(unquote(parsed.fragment) in headings(linked.read_text(encoding="utf-8")),
                    f"Missing anchor in {path.name}: {target}")
    return {"document": str(path.relative_to(ROOT)), "links_checked": len(links)}


def historical_match(path: Path, expected: str) -> str:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() == expected:
        return "byte_exact"
    if path.suffix in {".svg", ".json", ".py"}:
        lf = raw.replace(b"\r\n", b"\n")
        for normalized in (lf, lf.replace(b"\n", b"\r\n")):
            if hashlib.sha256(normalized).hexdigest() == expected:
                return "line_endings_only"
    raise ValueError(f"Historical content changed beyond line endings: {path.relative_to(ROOT)}")


def decode_gif(path: Path) -> dict:
    with Image.open(path) as im:
        require(im.format == "GIF", f"Not a GIF: {path.name}")
        require(im.n_frames > 1, f"Not animated: {path.name}")
        size, frames = im.size, im.n_frames
        duration = 0
        im.seek(0)
        first = im.convert("RGB")
        changed = False
        for index in range(frames):
            im.seek(index)
            frame = im.convert("RGB")
            require(frame.size == size, f"Inconsistent dimensions: {path.name}")
            duration += im.info.get("duration", 0)
            changed |= ImageChops.difference(first, frame).getbbox() is not None
        require(changed, f"All GIF frames are identical: {path.name}")
        require(duration > 0, f"Missing timing: {path.name}")
    return {"asset": path.name, "frames": frames, "size": list(size),
            "duration_ms": duration, "bytes": path.stat().st_size,
            "sha256": digest(path)}


def main() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "README_CN.md").read_text(encoding="utf-8")
    code_pattern = r"```(?:python|bash)\n(.*?)```"
    require(re.findall(code_pattern, english, re.S) == re.findall(code_pattern, chinese, re.S),
            "English and Chinese runnable snippets differ")
    en_links = {canonical_destination(x) for x in destinations(english) if not x.startswith("#")}
    zh_links = {canonical_destination(x) for x in destinations(chinese) if not x.startswith("#")}
    require(en_links == zh_links, f"Language content links differ: {en_links ^ zh_links}")
    require("README_CN.md" in destinations(english) and "README.md" in destinations(chinese),
            "Reciprocal language navigation missing")
    require("fibernet==4.2.0" in english and "4.3.0" in english,
            "Published/development install distinction missing")
    require("pip install fibernet==4.3.0" not in english + chinese,
            "Unpublished package command advertised")
    document_checks = [check_links(ROOT / name) for name in
                       ("README.md", "README_CN.md", "docs/MEDIA_GALLERY.md")]
    for path in sorted((MEDIA / "ui").glob("*.svg")):
        svg = ET.fromstring(path.read_bytes())
        require(svg.attrib.get("role") == "img", f"Accessible role missing: {path.name}")
        require(svg.find("{http://www.w3.org/2000/svg}title") is not None,
                f"Accessible title missing: {path.name}")
        require("<script" not in path.read_text().lower(), f"Script in badge: {path.name}")

    historical = []
    for path in sorted(MEDIA.glob("*.json")):
        if path.stem == "network_workflow_hero":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if "gif_sha256" not in record:
            continue
        suffix = "_peak.svg" if path.stem == "spectrum_four_topologies" else "_final.svg"
        svg = path.with_name(path.stem + suffix)
        historical.append({"manifest": path.name,
                           "gif": historical_match(path.with_suffix(".gif"), record["gif_sha256"]),
                           "svg": historical_match(svg, record["svg_sha256"])})
        ET.fromstring(svg.read_bytes())
        if "source_results" in record:
            historical[-1]["saved_results"] = historical_match(
                ROOT / record["source_results"], record["source_results_sha256"])

    hero_path = MEDIA / "network_workflow_hero.json"
    hero = json.loads(hero_path.read_text(encoding="utf-8"))
    require(hero.get("schema") == "fibernet.saved-media-hero.v1", "Unexpected hero schema")
    for relative, expected in hero["source_sha256"].items():
        require(digest(ROOT / relative) == expected, f"Hero source hash differs: {relative}")
    require(digest(ROOT / hero["render_script"]) == hero["render_script_sha256"],
            "Hero render script hash differs")
    require(hero["right"]["material_interpolation"] is False, "Material interpolation is unexpected")
    mapping = hero["right"]["hero_source_frame_mapping"]
    require(len(mapping) == hero["canvas"]["frames"], "Hero frame mapping length differs")
    require(all(isinstance(i, int) and 0 <= i < hero["right"]["saved_gif_frames"] for i in mapping),
            "Hero has an invalid saved-frame index")
    require(set(mapping) == set(range(hero["right"]["saved_gif_frames"])),
            "Hero does not include every saved material frame")
    require(digest(MEDIA / "network_workflow_hero.gif") == hero["gif_sha256"], "Hero GIF hash differs")
    require(digest(MEDIA / "network_workflow_hero_poster.svg") == hero["svg_sha256"], "Hero SVG hash differs")
    ET.fromstring((MEDIA / "network_workflow_hero_poster.svg").read_bytes())
    gifs = [decode_gif(p) for p in sorted(MEDIA.glob("*.gif"))]
    decoded = {row["asset"]: row for row in gifs}
    tours = json.loads((MEDIA / "module_workflow_tours.json").read_text(encoding="utf-8"))
    require(tours["kind"] == "saved_interface_tour", "Unexpected tour type")
    for item in tours["inputs"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "Tour source hash differs")
    for item in tours["outputs"]:
        actual = decoded[item["file"]]
        for key in ("sha256", "bytes", "frames", "duration_ms"):
            require(actual[key] == item[key], f"Tour {key} differs: {item['file']}")
        require(actual["size"] == [item["width"], item["height"]], "Tour size differs")
    electrical = json.loads((MEDIA / "electrical_torus_field.json").read_text(encoding="utf-8"))
    for relative, item in electrical["sources"].items():
        require(digest(ROOT / relative) == item["sha256"], "Electrical source hash differs")
    require(digest(ROOT / electrical["renderer"]) == electrical["renderer_sha256"],
            "Electrical renderer hash differs")
    require(electrical["field_varies_over_frames"] is False and electrical["synthetic"] is True,
            "Electrical display scope differs")
    actual = decoded["electrical_torus_field.gif"]
    require(actual["frames"] == electrical["frames"] and
            actual["duration_ms"] == electrical["frames"] * electrical["frame_duration_ms"] and
            actual["size"] == [electrical["width"], electrical["height"]],
            "Electrical decoded dimensions or timing differ")
    hero_gif = next(row for row in gifs if row["asset"] == "network_workflow_hero.gif")
    require(hero_gif["bytes"] < 6_000_000, "Hero exceeds its 6 MB delivery budget")
    require(hero_gif["frames"] == hero["canvas"]["frames"] and
            hero_gif["size"] == [hero["canvas"]["width"], hero["canvas"]["height"]],
            "Hero decoded shape differs from manifest")
    require(hero_gif["duration_ms"] == hero["canvas"]["duration_ms"],
            "Hero decoded timing differs from manifest")
    report = {"status": "PASS", "documents": document_checks,
              "bilingual_code_and_link_parity": True, "language_buttons": 4,
              "ui_svg_assets_checked": len(list((MEDIA / "ui").glob("*.svg"))),
              "historical_media": historical, "gifs": gifs,
              "physical_solver_executed": False}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


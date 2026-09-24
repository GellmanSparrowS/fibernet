"""Portable, bounded custom-cell specifications compatible with FiberScope JSON."""
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from fibernet.core.structure_graph import StructureGraph


@dataclass
class CustomCell:
    """One connected simple graph in the 10-unit planar cell coordinate frame."""

    nodes: list
    edges: list
    zh: str = ""
    en: str = ""
    settings: dict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, spec):
        if not isinstance(spec, dict):
            raise ValueError("custom cell must be a JSON object")
        try:
            nodes = [[float(a), float(b)] for a, b in spec["nodes"]]
            raw_edges = spec["edges"]
            edges = [[int(a), int(b)] for a, b in raw_edges]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid custom-cell nodes or edges") from exc
        for raw, converted in zip(raw_edges, edges):
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value != endpoint
                   for value, endpoint in zip(raw, converted)):
                raise ValueError("custom-cell edge endpoints must be integers")
        if not 2 <= len(nodes) <= 48 or not edges or len(edges) > 256:
            raise ValueError("custom cell exceeds node/edge bounds")
        if any(not (math.isfinite(x) and math.isfinite(y)) or
               max(abs(x), abs(y)) > 1e6 for x, y in nodes):
            raise ValueError("custom cell has nonfinite or oversized coordinates")
        seen = set()
        adjacency = [set() for _ in nodes]
        for a, b in edges:
            if not (0 <= a < len(nodes) and 0 <= b < len(nodes)) or a == b:
                raise ValueError("custom cell has invalid edge endpoints")
            pair = (min(a, b), max(a, b))
            if pair in seen:
                raise ValueError("custom cell has duplicate undirected edges")
            seen.add(pair)
            adjacency[a].add(b)
            adjacency[b].add(a)
        reached = {0}
        pending = [0]
        while pending:
            for other in adjacency[pending.pop()]:
                if other not in reached:
                    reached.add(other)
                    pending.append(other)
        if len(reached) != len(nodes):
            raise ValueError("custom cell must be connected")
        settings = spec.get("settings", {})
        if settings is None:
            settings = {}
        if not isinstance(settings, dict):
            raise ValueError("custom-cell settings must be a JSON object")
        return cls(nodes, edges, str(spec.get("zh", "")),
                   str(spec.get("en", "")), dict(settings))

    def to_mapping(self):
        result = {"nodes": self.nodes, "edges": self.edges,
                  "zh": self.zh, "en": self.en}
        if self.settings:
            result["settings"] = self.settings
        return result

    def to_graph(self, scale=10.0, radius=0.1):
        scale = float(scale)
        radius = float(radius)
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(radius) or radius <= 0:
            raise ValueError("scale and radius must be positive and finite")
        graph = StructureGraph(dimension=2, box_size=[scale, scale])
        for x, y in self.nodes:
            graph.add_node([x * scale, y * scale], merge=False)
        for a, b in self.edges:
            graph.add_edge(a, b, radius=radius)
        return graph


class CustomCellRegistry:
    """Read and atomically update the APP's key-to-cell JSON file format."""

    def __init__(self, path, max_bytes=1_000_000, max_cells=256):
        self.path = Path(path)
        self.max_bytes = int(max_bytes)
        self.max_cells = int(max_cells)
        if self.max_bytes < 1024 or self.max_cells < 1:
            raise ValueError("custom-cell registry budgets must be positive")
        self.cells = {}
        self.invalid_keys = []
        self._loaded = False

    def load(self, strict=True):
        if not self.path.exists():
            self.cells = {}
            self.invalid_keys = []
            self._loaded = True
            return self.cells
        if self.path.stat().st_size > self.max_bytes:
            raise MemoryError("custom-cell registry exceeds file budget")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or len(data) > self.max_cells:
            raise ValueError("invalid or oversized custom-cell registry")
        loaded = {}
        invalid = []
        for key, spec in data.items():
            try:
                if not isinstance(key, str):
                    raise ValueError("cell key must be a string")
                loaded[key] = CustomCell.from_mapping(spec)
            except ValueError:
                if strict:
                    raise
                invalid.append(key)
        self.cells = loaded
        self.invalid_keys = invalid
        self._loaded = True
        return dict(self.cells)

    def save(self, key, cell, reserved=()):
        if not self._loaded:
            self.load()
        key = re.sub(r"[^a-z0-9_]+", "_", str(key).lower()).strip("_") or "cell"
        if key in set(reserved):
            raise ValueError("custom-cell key collides with a built-in unit")
        cell = cell if isinstance(cell, CustomCell) else CustomCell.from_mapping(cell)
        if key not in self.cells and len(self.cells) >= self.max_cells:
            raise MemoryError("custom-cell registry exceeds entry budget")
        updated = dict(self.cells)
        updated[key] = cell
        self._write(updated)
        self.cells = updated
        return key

    def delete(self, key):
        if not self._loaded:
            self.load()
        if key not in self.cells:
            return False
        updated = dict(self.cells)
        del updated[key]
        self._write(updated)
        self.cells = updated
        return True

    def replace_all(self, cells):
        """Atomically persist an explicit complete registry snapshot."""
        if not isinstance(cells, dict) or len(cells) > self.max_cells:
            raise ValueError("invalid or oversized custom-cell registry")
        updated = {}
        for key, cell in cells.items():
            if not isinstance(key, str):
                raise ValueError("cell key must be a string")
            updated[key] = (cell if isinstance(cell, CustomCell)
                            else CustomCell.from_mapping(cell))
        self._write(updated)
        self.cells = updated
        self.invalid_keys = []
        self._loaded = True

    def _write(self, cells):
        body = json.dumps({key: cell.to_mapping() for key, cell in cells.items()},
                          ensure_ascii=False, indent=1, allow_nan=False) + "\n"
        if len(body.encode("utf-8")) > self.max_bytes:
            raise MemoryError("custom-cell registry exceeds file budget")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temporary.write_text(body, encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

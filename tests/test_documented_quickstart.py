"""Execute onboarding code as written, and check public model terminology."""
import ast
from pathlib import Path
import runpy
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]


def python_blocks(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks = []
    for index, line in enumerate(lines):
        if line.strip() != ".. code-block:: python":
            continue
        block = []
        for body in lines[index + 1:]:
            if body and not body.startswith(" "):
                break
            block.append(body)
        blocks.append(textwrap.dedent("\n".join(block)).strip())
    return blocks


@pytest.mark.parametrize("document", ["index.rst", "getting_started.rst", "quickstart.rst"])
def test_onboarding_python_blocks(document, tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg", force=True)
    monkeypatch.chdir(tmp_path)
    blocks = python_blocks(ROOT / "docs" / document)
    assert blocks, "No runnable onboarding blocks found"
    namespace = {}
    for index, block in enumerate(blocks):
        exec(compile(block, "%s:block%d" % (document, index), "exec"), namespace)
    if document == "quickstart.rst":
        assert (tmp_path / "demo_output" / "network.png").stat().st_size > 1000
        assert (tmp_path / "demo_output" / "fem_result.json").stat().st_size > 100
        assert namespace["recruitment"].perc_frame == 1


def test_docs_version_matches_library():
    config = runpy.run_path(str(ROOT / "docs" / "conf.py"))
    version = runpy.run_path(str(ROOT / "fibernet" / "version.py"))["__version__"]
    assert config["release"] == version
    assert config["version"] == ".".join(version.split(".")[:2])



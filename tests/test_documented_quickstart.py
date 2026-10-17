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


def test_recruitment_ui_describes_strain_not_force_flow():
    # Inspect the static translation table without importing Qt or the app.
    tree = ast.parse((ROOT / "FiberScope" / "studio" / "i18n.py").read_text(encoding="utf-8"))
    table = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "TEXT"
                         for target in node.targets))
    strings = {ast.literal_eval(key): ast.literal_eval(value)
               for key, value in zip(table.keys, table.values)
               if isinstance(value, ast.Tuple) and all(isinstance(item, ast.Constant)
                                                     for item in value.elts)}
    assert strings["quantile"][1] == "Strain threshold α"
    assert strings["view_perc"][1] == "Tensile recruitment"
    assert strings["legend_active"][1] == "recruited"
    assert "not a fixed 5% strain" in strings["quantile_hint"][1]
    assert "not complete force flow" in strings["explain_perc"][1]
    assert "force flows along continuous paths only" not in strings["explain_perc"][1]

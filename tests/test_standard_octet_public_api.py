"""Focused integration surface assertions; no graph generation."""
import ast
from pathlib import Path
import fibernet
import fibernet.gen
from fibernet.gen import pattern, standard_lattice

def test_additive_public_exports_and_unchanged_registration():
    assert fibernet.octet_fcc_v1 is standard_lattice.octet_fcc_v1
    assert fibernet.gen.octet_fcc_v1 is standard_lattice.octet_fcc_v1
    assert fibernet.__all__.count("octet_fcc_v1") == 1
    assert fibernet.gen.__all__.count("octet_fcc_v1") == 1
    assert "octet_fcc_v1" not in pattern.list_units_3d()

def test_stable_revision_and_public_wording_literals():
    tree = ast.parse(Path(standard_lattice.__file__).read_text())
    values = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert values.count("fibernet.standard-lattice/1.0") == 1
    assert "private-unexecuted-draft-20261026" not in values
    assert "lattice exceeds V1 node/edge allocation budget" in values
    assert "material record exceeds V1 payload budget" in values
    assert "lattice exceeds V1 peak-work allocation budget" in values

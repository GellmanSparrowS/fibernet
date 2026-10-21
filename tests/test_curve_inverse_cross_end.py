"""Public inverse-search candidates, budgets, and explicit topology scope."""
from dataclasses import asdict

import numpy as np
import pytest

from fibernet.ml import CurveInverseDesigner, run_curve_inverse, target_curve
from fibernet.ml.curve_inverse import fast_cfg as library_fast_cfg



def _builder(unit, pert, ld):
    return {"unit": unit, "pert": pert, "ld": ld}


def _evaluator(graph, target):
    values = np.asarray(graph["ld"] or [[0, 0], [0, 0]], float)
    return float(np.sum((values - 0.1) ** 2) + graph["pert"]), graph


def test_inverse_candidates_and_objectives_match_app():
    kw = dict(budget=7, seed=31, fixed_unit="hexagon", evaluator=_evaluator)
    public = run_curve_inverse(_builder, "J", **kw)
    assert public["evaluations"] == 7
    assert target_curve("J").shape == (24,)


def test_public_designer_requires_explicit_topology_scope():
    designer = CurveInverseDesigner(_builder, budget=3)
    with pytest.raises(ValueError, match="unit_keys"):
        designer.run("J", evaluator=_evaluator)
    with pytest.raises(ValueError, match="unknown inverse objective"):
        designer.run("unrecognized", fixed_unit="square", evaluator=_evaluator)



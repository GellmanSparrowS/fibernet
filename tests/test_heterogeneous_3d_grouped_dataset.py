from types import SimpleNamespace

import pytest

from examples.heterogeneous_3d_grouped_dataset import audit_roles, solve_case


def test_three_dimensional_material_variant_has_analytic_response():
    graph, result, features, labels = solve_case((2, 2, 1), 2e9)
    assert (len(graph.nodes), len(result.metadata["edge_ids"])) == (18, 33)
    assert features["parallel_paths"] == 6
    assert labels["cut_force_N"] == pytest.approx(.160849543864)
    assert labels["strain_energy_J"] == pytest.approx(3.21699087727524e-7, rel=1e-8)


def test_related_material_variants_must_stay_in_same_role():
    def row(sample, group):
        return SimpleNamespace(sample_id=sample, metadata={"source_group": group})

    def dataset(*rows):
        return SimpleNamespace(samples=rows)

    a, b, c, d = row("A2", "A"), row("A4", "A"), row("B2", "B"), row("C4", "C")
    original = dataset(a, b, c, d)
    audit_roles(original, (dataset(a, b), dataset(c), dataset(d)))
    with pytest.raises(ValueError, match="source group appears"):
        audit_roles(original, (dataset(a), dataset(b, c), dataset(d)))
    with pytest.raises(ValueError, match="original source group"):
        audit_roles(original, (dataset(a), dataset(row("A4", "X"), c), dataset(d)))
    with pytest.raises(ValueError, match="exactly once"):
        audit_roles(original, (dataset(a), dataset(c), dataset(d)))

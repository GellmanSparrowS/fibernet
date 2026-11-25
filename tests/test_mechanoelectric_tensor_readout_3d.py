import numpy as np

from examples.mechanoelectric_tensor_readout_3d import run_example


def test_diagonal_readouts_identify_signed_normal_and_shear_strains():
    report = run_example()
    assert (report["node_count"], report["member_count"]) == (91, 360)
    assert report["boundary_translation_node_count"] == 56
    assert report["strain_components"][3:] == ["gamma_xy", "gamma_xz", "gamma_yz"]
    assert report["calibration_step_max_abs_difference"] < 2e-8
    matrix = np.asarray(report["gauge_matrix_strain_rows_circuit_columns"])
    assert matrix.shape == (6, 9)
    assert np.linalg.matrix_rank(matrix[:, :3]) == 3
    assert min(report["diagonal_singular_values"]) > .12
    for pads in report["circuits"][3:]:
        assert len(pads["high_node_ids"]) == len(pads["low_node_ids"]) == 4
        assert set(pads["high_node_ids"]).isdisjoint(pads["low_node_ids"])

    cases = report["held_out_mixed_cases"]
    assert len(cases) == 4
    for case in cases:
        assert min(case["imposed_strain"][3:]) < 0 < max(case["imposed_strain"][3:])
        assert case["diagonal_relative_error"] < 2e-4
        assert case["axial_only_relative_error"] > .5
        np.testing.assert_allclose(
            np.einsum("i,ij->j", case["recovered_strain"], matrix[:, 3:]),
            case["fractional_resistance_change"][3:], rtol=1e-12, atol=1e-16,
        )
    assert cases[2]["diagonal_relative_error"] < cases[0]["diagonal_relative_error"]
    assert cases[3]["diagonal_relative_error"] < cases[1]["diagonal_relative_error"]

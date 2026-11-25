import numpy as np

from examples.mechanoelectric_mixed_strain_3d import run_example


def test_three_dc_channels_recover_held_out_signed_normal_strains():
    report = run_example()
    assert (report["node_count"], report["member_count"]) == (91, 360)
    assert report["boundary_translation_node_count"] == 56
    assert report["calibration_step_max_abs_difference"] < 2e-8
    A = np.asarray(report["gauge_matrix_load_rows_readout_columns"])
    assert A.shape == (3, 3)
    assert np.all(np.diag(A) > .5)
    assert np.all(A[~np.eye(3, dtype=bool)] > .1)
    assert 1. < report["gauge_matrix_condition_number"] < 3.

    cases = report["held_out_mixed_cases"]
    assert len(cases) == 4
    for item in cases:
        assert min(item["imposed_normal_strain"]) < 0 < max(item["imposed_normal_strain"])
        assert item["relative_strain_error"] < 1e-4
        assert item["diagonal_only_relative_error"] > .30
        # Reapply an independently written forward map to the recovered strain vector.
        np.testing.assert_allclose(
            np.einsum("i,ij->j", item["recovered_normal_strain"], A),
            item["fractional_resistance_change"], rtol=1e-12, atol=1e-16,
        )
    assert cases[2]["relative_strain_error"] < cases[0]["relative_strain_error"]
    assert cases[3]["relative_strain_error"] < cases[1]["relative_strain_error"]

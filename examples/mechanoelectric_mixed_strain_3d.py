"""Recover three normal strains from DC readouts on a 360-rod 3D network."""

import json
import sys
from contextlib import redirect_stdout

import numpy as np

if __package__:
    from .mechanoelectric_network_3d import build_sample, _length, _resistance
else:
    from mechanoelectric_network_3d import build_sample, _length, _resistance


def run_example():
    """Calibrate and test a fixed-fixture, three-readout inverse problem.

    Beam deformation updates rod lengths but not rod areas, material
    conductivities, contacts or connectivity. Drive only one electrode pair
    at a time. Inputs are the three normal components of the prescribed
    boundary strain; the example does not infer shear or internal strain.
    """
    from fibernet.ml.beam_frame_fem import BeamFrameFEM

    graph, nodes, edges, x, ends, electrodes, sigma, radii = build_sample()
    boundary_ids = set().union(*electrodes)
    boundary = np.asarray([i for i, node in enumerate(nodes) if node in boundary_ids])
    coordinates = x[boundary] - x.mean(axis=0)
    materials = tuple(graph.edges[edge].material for edge in edges)
    length0 = _length(x, ends)
    conductance0 = sigma * np.pi * radii**2 / length0
    fem = BeamFrameFEM()
    inputs = fem.graph_to_fem_input(graph, dim=3)
    np.testing.assert_array_equal(inputs["node_pos"], x)

    def resistance(model, conductance):
        return np.asarray([
            _resistance(model, edges, conductance, pair) for pair in electrodes
        ])

    baseline = resistance(graph, conductance0)

    def signal(strain):
        imposed = coordinates * strain
        result = fem.solve_3d(
            inputs["edge_index"], x, radii, prescribed_disp=dict(zip(boundary, imposed)),
            damping=0., edge_materials=materials,
        )
        u = result["u"][:, :3]
        np.testing.assert_allclose(u[boundary], imposed, rtol=1e-12, atol=3e-15)
        free_residual = result["numerical_diagnostics"]["physical_free_residual"]["values"]
        np.testing.assert_allclose(free_residual, 0., rtol=0., atol=5e-11)
        displaced = graph.copy()
        displaced.set_node_positions(dict(zip(nodes, x + u)))
        conductance = conductance0 * length0 / _length(x + u, ends)
        return resistance(displaced, conductance) / baseline - 1.

    def calibrate(step):
        # Row j is loading direction j; column i is electrical readout i.
        return np.asarray([
            (signal(step * direction) - signal(-step * direction)) / (2. * step)
            for direction in np.eye(3)
        ])

    step = 5e-6
    matrix = calibrate(step)
    step_check = float(np.max(np.abs(matrix - calibrate(4. * step))))
    results = []
    for amplitude in (2e-4, 5e-5):
        for blend in ([1., -.65, .35], [-.36, .57, .91]):
            imposed = amplitude * np.asarray(blend)
            measured = signal(imposed)
            recovered = np.linalg.solve(matrix.T, measured)
            diagonal_only = measured / np.diag(matrix)
            norm = np.linalg.norm(imposed)
            results.append({
                "imposed_normal_strain": imposed.tolist(),
                "fractional_resistance_change": measured.tolist(),
                "recovered_normal_strain": recovered.tolist(),
                "relative_strain_error": float(np.linalg.norm(recovered - imposed) / norm),
                "diagonal_only_relative_error": float(np.linalg.norm(diagonal_only - imposed) / norm),
            })
    return {
        "axes": ["x", "y", "z"],
        "node_count": len(nodes),
        "member_count": len(edges),
        "boundary_translation_node_count": len(boundary),
        "reference_resistance_ohm": baseline.tolist(),
        "gauge_matrix_load_rows_readout_columns": matrix.tolist(),
        "calibration_strain_step": step,
        "calibration_step_max_abs_difference": step_check,
        "gauge_matrix_condition_number": float(np.linalg.cond(matrix)),
        "held_out_mixed_cases": results,
    }


if __name__ == "__main__":
    with redirect_stdout(sys.stderr):
        result = run_example()
    print(json.dumps(result, indent=2, allow_nan=False))

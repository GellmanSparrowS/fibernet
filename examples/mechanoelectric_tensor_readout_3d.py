"""Recover six prescribed small-strain components from a 3D rod network."""

import json
import sys
from contextlib import redirect_stdout

import numpy as np

if __package__:
    from .mechanoelectric_network_3d import _length, _resistance, build_sample
else:
    from mechanoelectric_network_3d import _length, _resistance, build_sample


_PLANES = ((0, 1), (0, 2), (1, 2))
_AXES = ("x", "y", "z")


def run_example():
    """Compare three axial circuits with six opposing diagonal circuits.

    Every load uses the same prescribed-displacement fixture. Engineering
    shear gamma_ij is twice the tensor component epsilon_ij. Contact node
    IDs, per-rod areas and material conductivities remain fixed.
    """
    from fibernet.ml.beam_frame_fem import BeamFrameFEM

    graph, nodes, edges, points, endpoints, axial, conductivity, radii = build_sample()
    boundary_ids = set().union(*axial)
    boundary = np.asarray([i for i, node in enumerate(nodes) if node in boundary_ids])
    centered = points[boundary] - points.mean(axis=0)
    # Nominal lattice coordinates tag pads before deterministic disorder.
    # All pads move with their tagged specimen nodes, not fixed spatial planes.
    nominal_index = np.rint(points[boundary] / .001).astype(int)
    diagonal = []
    labels = list(_AXES)
    for i, j in _PLANES:
        for sign, op in ((1, "+"), (-1, "-")):
            projection = nominal_index[:, i] + sign * nominal_index[:, j]
            low = projection == projection.min()
            high = projection == projection.max()
            diagonal.append({
                nodes[row]: float(upper) for row, lower, upper in zip(boundary, low, high)
                if lower or upper
            })
            labels.append(f"{_AXES[i]}{op}{_AXES[j]}")
    contacts = [*axial, *diagonal]
    if any(set(pins.values()) != {0., 1.} for pins in contacts):
        raise RuntimeError("each circuit requires two disjoint contact groups")

    lengths = _length(points, endpoints)
    conductance = conductivity * np.pi * radii**2 / lengths
    materials = tuple(graph.edges[edge].material for edge in edges)
    fem = BeamFrameFEM()

    def read(model, values):
        return np.asarray([_resistance(model, edges, values, pins) for pins in contacts])

    reference = read(graph, conductance)

    def signal(components):
        strain = np.diag(components[:3])
        for (i, j), gamma in zip(_PLANES, components[3:]):
            strain[i, j] = strain[j, i] = .5 * gamma
        imposed = centered @ strain
        result = fem.solve_3d(
            endpoints.T, points, radii, prescribed_disp=dict(zip(boundary, imposed)),
            damping=0., edge_materials=materials,
        )
        u = result["u"][:, :3]
        np.testing.assert_allclose(u[boundary], imposed, rtol=1e-12, atol=3e-15)
        residual = result["numerical_diagnostics"]["physical_free_residual"]["values"]
        np.testing.assert_allclose(residual, 0., rtol=0., atol=1e-10)
        moved = graph.copy()
        moved.set_node_positions(dict(zip(nodes, points + u)))
        changed_conductance = conductance * lengths / _length(points + u, endpoints)
        return read(moved, changed_conductance) / reference - 1.

    def calibrate(step):
        return np.asarray([
            (signal(step * direction) - signal(-step * direction)) / (2. * step)
            for direction in np.eye(6)
        ])

    matrix = calibrate(5e-6)
    diagonal_matrix = matrix[:, 3:]
    singular_values = np.linalg.svd(diagonal_matrix, compute_uv=False)
    if np.linalg.matrix_rank(diagonal_matrix) != 6:
        raise RuntimeError("the diagonal circuit calibration cannot identify all six strains")
    held_out = []
    for amplitude in (2e-4, 5e-5):
        for pattern in ([1., -.65, .35, .7, -.55, .28],
                        [-.36, .57, .91, -.8, .24, .51]):
            imposed = amplitude * np.asarray(pattern)
            measured = signal(imposed)
            predicted = np.linalg.solve(diagonal_matrix.T, measured[3:])
            axial_only = np.linalg.lstsq(matrix[:, :3].T, measured[:3], rcond=None)[0]
            magnitude = np.linalg.norm(imposed)
            held_out.append({
                "imposed_strain": imposed.tolist(),
                "fractional_resistance_change": measured.tolist(),
                "recovered_strain": predicted.tolist(),
                "diagonal_relative_error": float(np.linalg.norm(predicted - imposed) / magnitude),
                "axial_only_relative_error": float(np.linalg.norm(axial_only - imposed) / magnitude),
            })
    return {
        "strain_components": ["epsilon_xx", "epsilon_yy", "epsilon_zz",
                              "gamma_xy", "gamma_xz", "gamma_yz"],
        "circuit_labels": labels,
        "circuits": [{"low_node_ids": [int(n) for n, p in pins.items() if not p],
                      "high_node_ids": [int(n) for n, p in pins.items() if p]}
                     for pins in contacts],
        "node_count": len(nodes),
        "member_count": len(edges),
        "boundary_translation_node_count": len(boundary),
        "reference_resistance_ohm": reference.tolist(),
        "gauge_matrix_strain_rows_circuit_columns": matrix.tolist(),
        "calibration_step_max_abs_difference": float(np.max(np.abs(matrix - calibrate(2e-5)))),
        "diagonal_singular_values": singular_values.tolist(),
        "held_out_mixed_cases": held_out,
    }


if __name__ == "__main__":
    with redirect_stdout(sys.stderr):
        result = run_example()
    print(json.dumps(result, indent=2, allow_nan=False))

"""Separate stiffness, electrical conductivity and load control on one 3D network."""
from contextlib import redirect_stdout
import json
import sys

import numpy as np

if __package__:
    from .mechanoelectric_network_3d import build_sample, _length, _resistance
else:  # direct script from the source distribution
    from mechanoelectric_network_3d import build_sample, _length, _resistance


def run_example():
    """Return baseline and paired resistance changes for the same network."""
    import fibernet as fn
    from fibernet.ml.beam_frame_fem import BeamFrameFEM

    g, nids, eids, x, ends, electrodes, sigma, radius = build_sample()
    stiffer = g.copy()
    for eid in eids:
        material = g.edges[eid].material
        stiffer.edges[eid].material = fn.Material(
            youngs_modulus=2. * material.youngs_modulus, poissons_ratio=material.poissons_ratio,
        )

    fem = BeamFrameFEM()
    inp = fem.graph_to_fem_input(g, dim=3, pct=16.5 / len(nids))
    np.testing.assert_array_equal(inp["node_pos"], x)
    left, right = (inp["boundaries"][side] for side in ("left", "right"))
    assert set(right) == set(np.flatnonzero(x[:, 0] == x[:, 0].max()))
    displacement = .005 * float(inp["x_range"])
    prescribed = {i: (displacement, 0., 0.) for i in right}

    def solve(graph, *, prescribed_disp=None, forces=None):
        # No numerical diagonal for the exact uniform stiffness scale control.
        # The full original system must solve and satisfy physical equilibrium.
        result = fem.solve_3d(
            inp["edge_index"], inp["node_pos"], inp["radii"],
            fixed_nodes=left, prescribed_disp=prescribed_disp, forces=forces,
            damping=0., edge_materials=tuple(graph.edges[e].material for e in eids),
        )
        residual = result["numerical_diagnostics"]["physical_free_residual"]
        np.testing.assert_allclose(residual["values"], 0., rtol=0., atol=2e-11)
        return result

    def measure(graph, xyz, conductivity_factor=1.):
        state = graph.copy()
        state.set_node_positions(dict(zip(nids, xyz)))
        conductance = (
            conductivity_factor * sigma * np.pi * radius**2 / _length(xyz, ends)
        )
        return np.asarray([_resistance(state, eids, conductance, pins) for pins in electrodes])

    u0 = solve(g, prescribed_disp=prescribed)
    u1 = solve(stiffer, prescribed_disp=prescribed)
    np.testing.assert_allclose(u0["u"], u1["u"], rtol=0., atol=1e-12)
    np.testing.assert_allclose(u1["reactions"], 2.*u0["reactions"], rtol=1e-10, atol=2e-11)
    forces = np.zeros_like(x)
    forces[right] = u0["reactions"][right, :3]
    f0 = solve(g, forces=forces)
    f1 = solve(stiffer, forces=forces)
    np.testing.assert_allclose(f0["u"], u0["u"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(f1["u"], .5*f0["u"], rtol=1e-9, atol=1e-12)

    baseline = measure(g, x)

    def compare(graph, result):
        return (100. * (measure(graph, x + result["u"][:, :3]) / baseline - 1.)).tolist()

    R0 = measure(g, x + u0["u"][:, :3])
    R2 = measure(g, x + u0["u"][:, :3], conductivity_factor=2.)
    np.testing.assert_allclose(2.*R2, R0, rtol=1e-11, atol=0.)
    return {
        "directions": ["x", "y", "z"],
        "member_count": len(eids),
        "reference_axial_grip_strain": .005,
        "same_displacement_change_percent": {"reference_E": compare(g, u0),
                                             "doubled_E": compare(stiffer, u1)},
        "same_force_change_percent": {"reference_E": compare(g, f0),
                                      "doubled_E": compare(stiffer, f1)},
        "uniform_doubled_sigma_loaded_R_ratio": (R2/R0).tolist(),
    }


if __name__ == "__main__":
    with redirect_stdout(sys.stderr):
        result = run_example()
    print(json.dumps(result, indent=2, allow_nan=False))

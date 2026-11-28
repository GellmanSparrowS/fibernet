# SPDX-License-Identifier: MIT
"""Guarded independent reaction and energy recovery for banded Tet4."""
import numpy as np

from fibernet.examples import tet4_mechanoelectric_looped_network as demo
from fibernet.sim import _tet4_recovery, tet4


def _fixture():
    points, cells = demo._lattice()
    bundle = demo._bundle(points, cells, "stiff_half")
    mesh = tet4._validate(bundle.points, bundle.tetrahedra, bundle.node_ids,
                          bundle.cell_ids, bundle.material_ids, bundle.materials,
                          bundle.length_unit)
    _, grip, held, _, length = demo._fixture(bundle)
    imposed = {i: {"x": 0.003 * length} for i in grip}
    return mesh, bundle, held, imposed


def test_banded_recovery_uses_fast_path_on_deformed_welded_network(monkeypatch):
    mesh, bundle, held, imposed = _fixture()
    original = tet4._prescribed_element_response
    def scalar_unexpected(*args):
        raise AssertionError("well-conditioned welded case should use batched recovery")
    monkeypatch.setattr(tet4, "_prescribed_element_response", scalar_unexpected)
    solution = tet4.solve_static_tet4(bundle, forces_by_id={}, fixed_dofs_by_id=held,
                        prescribed_displacements_m_by_id=imposed)
    expected, energy = original(mesh, solution.displacement_m)
    np.testing.assert_allclose(solution.reaction_N[solution.fixed_mask],
                               expected.reshape(-1,3)[solution.fixed_mask],
                               rtol=5e-11, atol=2e-13)
    np.testing.assert_allclose(solution.strain_energy_J, energy, rtol=5e-11)
    np.testing.assert_allclose(solution.half_external_work_J, energy, rtol=5e-11)


def test_near_rigid_rotation_declines_vectorized_recovery():
    mesh, _, _, _ = _fixture()
    x = mesh.points * tet4._LENGTH_TO_M[mesh.length_unit]
    spin = np.array([[0., -.001, .003], [.001, 0., -.002], [-.003, .002, 0.]])
    motion = x @ (spin + np.diag([1e-10, -7e-11, 5e-11])).T + [.003, -.001, .004]
    assert _tet4_recovery.guarded_banded_response(mesh, motion) is None


def test_failed_vectorized_certificate_retries_scalar_without_refactoring(monkeypatch):
    _, bundle, held, imposed = _fixture()
    kwargs = dict(forces_by_id={}, fixed_dofs_by_id=held,
                  prescribed_displacements_m_by_id=imposed)
    reference = tet4.solve_static_tet4(bundle, **kwargs)
    actual_function = _tet4_recovery.guarded_banded_response
    factor = tet4._factor_band
    counts = []
    def invalid_candidate(mesh, u):
        force, energy = actual_function(mesh, u)
        return force, energy * 1.01
    def observe(*args):
        counts.append(1)
        return factor(*args)
    monkeypatch.setattr(_tet4_recovery, "guarded_banded_response", invalid_candidate)
    monkeypatch.setattr(tet4, "_factor_band", observe)
    fallback = tet4.solve_static_tet4(bundle, **kwargs)
    assert counts == [1]
    np.testing.assert_array_equal(fallback.displacement_m, reference.displacement_m)
    np.testing.assert_allclose(fallback.reaction_N, reference.reaction_N, rtol=5e-11, atol=2e-13)
    np.testing.assert_allclose(fallback.strain_energy_J, reference.strain_energy_J, rtol=5e-11)

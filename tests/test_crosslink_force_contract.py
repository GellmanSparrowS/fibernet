"""Endpoint, energy-gradient and compatibility checks for crosslink forces."""

import numpy as np
import pytest

from fibernet.core.crosslinks import (
    BondedCrosslink,
    CovalentBond,
    CrosslinkState,
    IonicBond,
    SpringCrosslink,
    create_crosslink,
)


def _negative_gradient(model, p1, p2, endpoint, step):
    result = np.empty(3)
    for axis in range(3):
        plus = [p1.copy(), p2.copy()]
        minus = [p1.copy(), p2.copy()]
        plus[endpoint][axis] += step
        minus[endpoint][axis] -= step
        result[axis] = -(model.energy(*plus) - model.energy(*minus)) / (2 * step)
    return result


@pytest.mark.parametrize("charge2", [-1.6e-19, 1.6e-19])
@pytest.mark.parametrize("separation", [0.2e-9, 1e-9, 3e-9])
@pytest.mark.parametrize("direction", [[1.0, 0.0, 0.0], [2.0, -3.0, 6.0]])
def test_ionic_force_is_negative_first_endpoint_energy_gradient(
    charge2, separation, direction
):
    bond = IonicBond(charge2=charge2, breaking_force=np.inf)
    direction = np.asarray(direction) / np.linalg.norm(direction)
    p1 = np.array([0.4, -0.7, 0.2]) * 1e-9
    p2 = p1 + separation * direction
    force, state = bond.compute_force(p1, p2, CrosslinkState())
    step = separation * 1e-5
    np.testing.assert_allclose(
        force, _negative_gradient(bond, p1, p2, 0, step), rtol=1e-7, atol=1e-22
    )
    np.testing.assert_allclose(
        -force, _negative_gradient(bond, p1, p2, 1, step), rtol=1e-7, atol=1e-22
    )
    np.testing.assert_array_equal(state.force, force)
    np.testing.assert_array_equal(state.displacement, p2 - p1)
    assert not state.broken


@pytest.mark.parametrize(
    "charge1, charge2, attractive",
    [(1.6e-19, -1.6e-19, True), (1.6e-19, 1.6e-19, False), (-1.6e-19, -1.6e-19, False)],
)
def test_ionic_direction_energy_and_endpoint_reversal(charge1, charge2, attractive):
    bond = create_crosslink(
        "ionic", charge1=charge1, charge2=charge2, breaking_force=np.inf
    )
    p1 = np.array([1.0, -2.0, 0.5]) * 1e-9
    p2 = p1 + np.array([2.0, -3.0, 6.0]) * 1e-10
    dr = p2 - p1
    radius = np.linalg.norm(dr)
    prefactor = charge1 * charge2 / (4 * np.pi * 8.854e-12 * 80.0)
    expected_energy = prefactor * np.exp(-radius / 1e-9) / radius
    expected_magnitude = (
        abs(prefactor) * np.exp(-radius / 1e-9) * (1 + radius / 1e-9) / radius**2
    )
    force, _ = bond.compute_force(p1, p2, CrosslinkState())
    reverse, _ = bond.compute_force(p2, p1, CrosslinkState())
    np.testing.assert_allclose(reverse, -force, rtol=1e-14, atol=0)
    assert (np.dot(force, dr) > 0) == attractive
    assert bond.energy(p1, p2) == pytest.approx(expected_energy, rel=1e-14, abs=0)
    assert np.linalg.norm(force) == pytest.approx(expected_magnitude, rel=1e-14, abs=0)


@pytest.mark.parametrize(
    "model",
    [
        SpringCrosslink(stiffness=2.0, rest_length=1.0),
        BondedCrosslink(stretch_stiffness=2.0, rest_length=1.0),
        CovalentBond(bond_energy=2.0, equilibrium_length=1.0, width=0.7),
    ],
)
@pytest.mark.parametrize("separation", [0.6, 1.4])
def test_existing_elastic_models_share_first_endpoint_convention(model, separation):
    p1 = np.array([0.5, -0.2, 0.3])
    direction = np.array([2.0, -3.0, 6.0]) / 7
    p2 = p1 + separation * direction
    force, _ = model.compute_force(p1, p2, CrosslinkState())
    np.testing.assert_allclose(
        force, _negative_gradient(model, p1, p2, 0, 1e-5), rtol=1e-7, atol=1e-10
    )
    assert (np.dot(force, direction) > 0) == (separation > 1.0)


@pytest.mark.parametrize(
    "threshold_ratio, broken", [(0.5, True), (1.0, False), (2.0, False)]
)
def test_ionic_breaking_threshold_and_transition_force_are_preserved(
    threshold_ratio, broken
):
    p1 = np.zeros(3)
    p2 = np.array([0.5e-9, 0.0, 0.0])
    bond = IonicBond(breaking_force=np.inf)
    reference, _ = bond.compute_force(p1, p2, CrosslinkState())
    bond.F_break = threshold_ratio * np.linalg.norm(reference)
    initial = CrosslinkState()
    force, state = bond.compute_force(p1, p2, initial)
    assert state.broken == broken
    assert not initial.broken
    np.testing.assert_array_equal(force, reference)
    np.testing.assert_array_equal(state.force, reference)
    if broken:
        # The transition reports the threshold-crossing force. Later calls
        # return zero and retain the diagnostic force in the same state object.
        following, following_state = bond.compute_force(p1, p2, state)
        np.testing.assert_array_equal(following, np.zeros(3))
        assert following_state is state
        np.testing.assert_array_equal(following_state.force, reference)


@pytest.mark.parametrize("distance", [0.0, 0.5e-15])
@pytest.mark.parametrize("broken", [False, True])
def test_ionic_near_coincidence_guard_is_unchanged(distance, broken):
    bond = IonicBond()
    force, state = bond.compute_force(
        np.zeros(3), np.array([distance, 0.0, 0.0]), CrosslinkState(broken=broken)
    )
    np.testing.assert_array_equal(force, np.zeros(3))
    assert state.broken == broken


def test_ionic_force_with_zero_charge_remains_zero():
    bond = IonicBond(charge1=0.0)
    force, state = bond.compute_force([0, 0, 0], [1e-9, 0, 0], CrosslinkState())
    np.testing.assert_array_equal(force, np.zeros(3))
    assert not state.broken

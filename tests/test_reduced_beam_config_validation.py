"""Admission contracts for the mutable reduced-model configuration."""

import numpy as np
import pytest

from fibernet import StructureGraph
from fibernet.sim import ReducedBeamConfig, ReducedBeamSolver

REAL_FIELDS = (
    "target_stretch",
    "stiffness",
    "k_bend_frac",
    "k_contact",
    "r_contact",
    "dt",
    "drag",
    "ramp_fraction",
    "grip_pct",
)
COUNT_FIELDS = ("num_steps", "save_interval", "n_increments", "contact_rebuild")


def _bar():
    graph = StructureGraph(dimension=2)
    graph.add_node([0.0, 0.0])
    graph.add_node([1.0, 0.0])
    graph.add_edge(0, 1)
    return graph


@pytest.mark.parametrize("name", REAL_FIELDS + COUNT_FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_config_is_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        ReducedBeamConfig(**{name: value})


@pytest.mark.parametrize("name", REAL_FIELDS + COUNT_FIELDS)
@pytest.mark.parametrize("value", [True, "1", 1 + 0j, [1.0], np.array(1.0)])
def test_numeric_controls_require_real_scalars(name, value):
    with pytest.raises(ValueError, match=name):
        ReducedBeamConfig(**{name: value})


@pytest.mark.parametrize("name", COUNT_FIELDS)
@pytest.mark.parametrize("value", [1.5, 1.0, np.float64(1.0)])
def test_step_controls_require_integers(name, value):
    with pytest.raises(ValueError, match=name):
        ReducedBeamConfig(**{name: value})


@pytest.mark.parametrize(
    "name, value",
    [
        ("target_stretch", 0.0),
        ("target_stretch", -1.0),
        ("r_contact", 0.0),
        ("r_contact", -1.0),
        ("dt", 0.0),
        ("dt", -1.0),
        ("stiffness", -1.0),
        ("k_bend_frac", -1.0),
        ("k_contact", -1.0),
        ("drag", -1.0),
        ("ramp_fraction", -0.1),
        ("ramp_fraction", 1.1),
        ("grip_pct", -0.1),
        ("num_steps", 0),
        ("save_interval", 0),
        ("contact_rebuild", 0),
        ("n_increments", -1),
    ],
)
def test_out_of_range_controls_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        ReducedBeamConfig(**{name: value})


@pytest.mark.parametrize("name", REAL_FIELDS + COUNT_FIELDS)
def test_mutated_config_is_rejected_before_geometry(name):
    config = ReducedBeamConfig()
    setattr(config, name, float("nan"))
    # An invalid graph proves config admission happens before geometry work.
    with pytest.raises(ValueError, match=name):
        ReducedBeamSolver(object(), config)


@pytest.mark.parametrize("use_contact", [False, True])
@pytest.mark.parametrize("n_increments", [0, 1])
def test_mutation_after_solver_construction_cannot_produce_nonfinite_result(
    use_contact, n_increments, monkeypatch
):
    config = ReducedBeamConfig(
        num_steps=1,
        save_interval=1,
        use_contact=use_contact,
        use_bending=False,
        n_increments=n_increments,
    )
    solver = ReducedBeamSolver(_bar(), config)
    config.target_stretch = float("nan")

    def unexpected(*args, **kwargs):
        pytest.fail("invalid config reached physics or progress callback")

    monkeypatch.setattr(solver, "_forces", unexpected)
    monkeypatch.setattr(solver, "_contact_candidates", unexpected)
    with pytest.raises(ValueError, match="target_stretch"):
        solver.run(progress_cb=unexpected)


@pytest.mark.parametrize("target_stretch", [0.8, 1.0, 1.2])
@pytest.mark.parametrize("n_increments", [0, 2])
@pytest.mark.parametrize("ramp_fraction", [0.0, 1.0])
def test_valid_numpy_scalars_keep_analytical_spring_response(
    target_stretch, n_increments, ramp_fraction
):
    config = ReducedBeamConfig(
        target_stretch=np.float64(target_stretch),
        stiffness=np.float64(100.0),
        num_steps=np.int64(2),
        save_interval=np.int32(1),
        n_increments=np.int64(n_increments),
        contact_rebuild=np.int64(1),
        ramp_fraction=ramp_fraction,
        grip_pct=0.0,
        drag=0.0,
        k_bend_frac=0.0,
        k_contact=0.0,
        use_bending=False,
        use_contact=False,
    )
    result = ReducedBeamSolver(_bar(), config).run()
    assert np.isfinite(result.frames_xy).all()
    assert np.isfinite(result.force_curve).all()
    assert result.force_curve[-1] == pytest.approx(100 * (target_stretch - 1))
    assert result.energies["work"][-1] == pytest.approx(50 * (target_stretch - 1) ** 2)


def test_zero_stiffness_remains_supported():
    config = ReducedBeamConfig(
        stiffness=0.0, num_steps=1, save_interval=1, use_contact=False
    )
    result = ReducedBeamSolver(_bar(), config).run()
    np.testing.assert_array_equal(result.force_curve, 0.0)


@pytest.mark.parametrize("name", REAL_FIELDS)
@pytest.mark.parametrize("admission", ["config", "solver", "run"])
def test_extended_precision_scalars_cannot_reach_float64_force_accumulation(
    name, admission
):
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip("longdouble uses float64 on this platform")
    config = ReducedBeamConfig(num_steps=1, save_interval=1)
    value = np.longdouble(getattr(config, name))
    if admission == "config":
        with pytest.raises(ValueError, match=name):
            ReducedBeamConfig(**{name: value})
        return
    if admission == "run":
        solver = ReducedBeamSolver(_bar(), config)
    setattr(config, name, value)
    with pytest.raises(ValueError, match=name):
        if admission == "solver":
            ReducedBeamSolver(_bar(), config)
        else:
            solver.run()


@pytest.mark.parametrize("name", ["use_bending", "use_contact"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, 1, "false", [False]])
def test_feature_switches_require_booleans(name, value):
    with pytest.raises(ValueError, match=name):
        ReducedBeamConfig(**{name: value})


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_supported_numpy_floating_precisions_and_booleans_run(dtype):
    config = ReducedBeamConfig(
        num_steps=1, save_interval=1, stiffness=100.0, k_contact=100.0
    )
    for name in REAL_FIELDS:
        setattr(config, name, dtype(getattr(config, name)))
    config.use_bending = np.bool_(True)
    config.use_contact = np.bool_(True)
    result = ReducedBeamSolver(_bar(), config).run()
    assert all(type(getattr(config, name)) is float for name in REAL_FIELDS)
    assert all(type(getattr(config, name)) is int for name in COUNT_FIELDS)
    assert type(config.use_bending) is type(config.use_contact) is bool
    assert np.isfinite(result.frames_xy).all()
    assert np.isfinite(result.force_curve).all()


@pytest.mark.parametrize("grip_pct", [0.5, 0.75, 1.0, 2.0])
def test_overlapping_grip_bands_keep_legacy_midpoint_fallback(grip_pct):
    graph = StructureGraph(dimension=2)
    for x in (0.0, 0.5, 1.0):
        graph.add_node([x, 0.0])
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    result = ReducedBeamSolver(
        graph,
        ReducedBeamConfig(
            grip_pct=grip_pct, num_steps=1, save_interval=1, use_contact=False
        ),
    ).run()
    np.testing.assert_array_equal(result.left_nodes, [0])
    np.testing.assert_array_equal(result.right_nodes, [1, 2])
    assert np.isfinite(result.frames_xy).all()


def test_large_python_integer_coefficient_does_not_require_numpy_int64():
    result = ReducedBeamSolver(
        _bar(),
        ReducedBeamConfig(
            stiffness=10**20,
            target_stretch=1.2,
            num_steps=1,
            save_interval=1,
            use_contact=False,
            use_bending=False,
        ),
    ).run()
    assert result.force_curve[-1] == pytest.approx(2e19)


def test_real_control_outside_float64_conversion_range_is_rejected():
    with pytest.raises(ValueError, match="stiffness"):
        ReducedBeamConfig(stiffness=10**400)


def test_low_precision_bending_fraction_uses_float64_coefficient_arithmetic():
    graph = StructureGraph(dimension=2)
    for point in ([0.0, 0.0], [0.5, 0.1], [1.0, 0.0]):
        graph.add_node(point)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    results = []
    for fraction in (0.25, np.float16(0.25), np.float32(0.25)):
        config = ReducedBeamConfig(
            num_steps=2,
            save_interval=1,
            use_contact=False,
            target_stretch=1.2,
            k_bend_frac=fraction,
        )
        results.append(ReducedBeamSolver(graph, config).run())
    for result in results:
        assert np.isfinite(result.force_curve).all()
        np.testing.assert_array_equal(result.force_curve, results[0].force_curve)
        np.testing.assert_array_equal(result.frames_xy, results[0].frames_xy)


def test_invalid_mutation_does_not_partially_normalize_other_fields():
    config = ReducedBeamConfig()
    config.stiffness = np.float32(100.0)
    config.use_contact = float("nan")
    with pytest.raises(ValueError, match="use_contact"):
        ReducedBeamSolver(_bar(), config)
    assert type(config.stiffness) is np.float32

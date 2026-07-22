"""Test ML/RL utilities and StructureGraph node manipulation."""

import pytest
pytest.importorskip("sklearn")
import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def test_structure_graph_node_manipulation():
    """Test displace_node, set_node_position, get_internal/boundary_nodes."""
    from fibernet import pattern_2d

    # Generate with explicit zero displacements
    g = pattern_2d(unit="square", box=(10, 10), grid=(3, 3),
                   n_pts_per_side=2, n_internal=0)

    # Check we have both internal and boundary nodes
    internal = g.get_internal_nodes()
    boundary = g.get_boundary_nodes()
    assert len(internal) > 0, "Should have internal nodes in 3x3 grid"
    assert len(boundary) > 0, "Should have boundary nodes"
    assert len(internal) + len(boundary) == g.num_nodes

    # Test displace_node
    old_pos = g.nodes[internal[0]].position.copy()
    g.displace_node(internal[0], [0.5, 1.0])
    new_pos = g.nodes[internal[0]].position
    np.testing.assert_allclose(new_pos[:2], old_pos[:2] + [0.5, 1.0], atol=1e-10)

    # Test set_node_position
    g.set_node_position(internal[1], [5.0, 5.0])
    np.testing.assert_allclose(g.nodes[internal[1]].position[:2], [5.0, 5.0], atol=1e-10)

    # Test set_node_positions (batch)
    positions = {internal[2]: [1.0, 2.0], internal[3]: [3.0, 4.0]}
    g.set_node_positions(positions)
    np.testing.assert_allclose(g.nodes[internal[2]].position[:2], [1.0, 2.0], atol=1e-10)
    np.testing.assert_allclose(g.nodes[internal[3]].position[:2], [3.0, 4.0], atol=1e-10)

    # Test KeyError for invalid node
    try:
        g.displace_node(99999, [0, 0])
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    print("✓ test_structure_graph_node_manipulation PASSED")


def test_parametric_generation():
    """Test RL-style parametric generation with point_displacements."""
    from fibernet import pattern_2d

    n_pts_per_side = 3
    n_sides = 4
    n_disp = n_sides * n_pts_per_side

    # Generate with custom displacements (simulating RL action output)
    np.random.seed(42)
    disps = [(np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5))
             for _ in range(n_disp)]

    g = pattern_2d(unit="square", box=(10, 10), grid=(1, 1),
                   n_pts_per_side=n_pts_per_side,
                   point_displacements=disps,
                   n_internal=0)

    assert g.num_nodes == 4 + n_disp  # 4 corners + 12 intermediate
    assert g.num_edges == n_disp + 4  # connected around the square

    print("✓ test_parametric_generation PASSED")


def test_ml_train_predictor():
    """Test ML train_predictor."""
    from fibernet.ml.utils import train_predictor, cross_validate, compare_models

    # Generate synthetic data with strong linear signal
    np.random.seed(42)
    n_samples = 300
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    true_coefs = np.array([3.0, -2.0, 1.5, 0.5, -1.0])
    y = X @ true_coefs + np.random.randn(n_samples) * 0.2

    # Test train_predictor (ridge should nail linear data)
    model, metrics = train_predictor(X, y, model_type="ridge")
    assert "r2" in metrics
    assert "rmse" in metrics
    assert metrics["r2"] > 0.9, f"R² should be > 0.9, got {metrics['r2']}"

    # Test RF too (needs more data)
    rf_model, rf_metrics = train_predictor(X, y, model_type="rf")
    assert rf_metrics["r2"] > 0.3, f"RF R² should be > 0.3, got {rf_metrics['r2']}"

    # Test cross_validate
    cv = cross_validate(X, y, model_type="ridge", cv=5)
    assert "mean_r2" in cv
    assert "std_r2" in cv
    assert cv["mean_r2"] > 0.9, f"CV R² should be > 0.9, got {cv['mean_r2']}"

    # Test compare_models
    results = compare_models(X, y, model_types=["rf", "ridge"])
    assert "rf" in results
    assert "ridge" in results
    assert results["ridge"]["r2"] > 0.9

    print("✓ test_ml_train_predictor PASSED")


def test_ml_plots():
    """Test ML visualization functions."""
    from fibernet.ml.utils import (
        plot_predictions, plot_feature_importance,
        plot_residuals, plot_learning_curve,
    )
    from fibernet.ml.utils import train_predictor

    np.random.seed(42)
    X = np.random.randn(80, 5)
    y = X @ np.array([1, 2, 0, -1, 0.5]) + np.random.randn(80) * 0.1

    model, metrics = train_predictor(X, y, model_type="rf")
    y_pred = model.predict(X[:20])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test plot_predictions
        fig = plot_predictions(y[:20], y_pred,
                              save_path=f"{tmpdir}/pred.png")
        assert os.path.exists(f"{tmpdir}/pred.png")

        # Test plot_feature_importance
        fig = plot_feature_importance(model, [f"f{i}" for i in range(5)],
                                      save_path=f"{tmpdir}/importance.png")
        assert os.path.exists(f"{tmpdir}/importance.png")

        # Test plot_residuals
        fig = plot_residuals(y[:20], y_pred,
                            save_path=f"{tmpdir}/residuals.png")
        assert os.path.exists(f"{tmpdir}/residuals.png")

    print("✓ test_ml_plots PASSED")


def test_rl_plots():
    """Test RL visualization functions."""
    from fibernet.rl.utils import (
        plot_reward_curve, plot_convergence, plot_action_distribution,
    )

    np.random.seed(42)
    rewards = np.cumsum(np.random.randn(100)) / np.sqrt(np.arange(1, 101))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test plot_reward_curve
        fig = plot_reward_curve(rewards, window=10,
                               save_path=f"{tmpdir}/reward.png")
        assert os.path.exists(f"{tmpdir}/reward.png")

        # Test plot_convergence
        values = np.random.randn(50).cumsum()
        fig = plot_convergence(values, minimize=True,
                              save_path=f"{tmpdir}/convergence.png")
        assert os.path.exists(f"{tmpdir}/convergence.png")

        # Test plot_action_distribution
        actions = [{"grid_x": np.random.randint(2, 5),
                    "radius": np.random.uniform(0.05, 0.3)}
                   for _ in range(100)]
        fig = plot_action_distribution(actions,
                                       save_path=f"{tmpdir}/actions.png")
        assert os.path.exists(f"{tmpdir}/actions.png")

    print("✓ test_rl_plots PASSED")


def test_save_load_agent():
    """Test agent serialization."""
    from fibernet.rl.utils import save_agent, load_agent
    from fibernet.ml.utils import train_predictor

    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X @ [1, 2, 3]
    model, _ = train_predictor(X, y, model_type="ridge")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/agent.pkl"
        save_agent(model, path, metadata={"test": True})
        loaded = load_agent(path)
        assert loaded is not None

    print("✓ test_save_load_agent PASSED")


def test_top_level_imports():
    """Test that top-level imports work."""
    import fibernet

    # Core
    assert hasattr(fibernet, "StructureGraph")
    assert hasattr(fibernet, "pattern_2d")
    assert hasattr(fibernet, "TaichiEngine")

    # ML (if sklearn available)
    if fibernet._HAS_ML:
        assert hasattr(fibernet, "train_predictor")
        assert hasattr(fibernet, "cross_validate")
        assert hasattr(fibernet, "plot_predictions")
        assert hasattr(fibernet, "predict_from_csv")

    # RL
    if fibernet._HAS_RL:
        assert hasattr(fibernet, "plot_reward_curve")
        assert hasattr(fibernet, "plot_convergence")
        assert hasattr(fibernet, "run_bayesian_optimization")

    print("✓ test_top_level_imports PASSED")


def test_predict_from_csv():
    """Test one-line ML from CSV."""
    from fibernet.ml.utils import predict_from_csv
    import pandas as pd

    # Create synthetic CSV
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "id": range(n),
        "feat_a": np.random.randn(n),
        "feat_b": np.random.randn(n),
        "feat_c": np.random.randn(n),
        "max_force": np.random.randn(n) * 100 + 500,
    })
    # Add some correlation
    df["max_force"] = df["feat_a"] * 50 + df["feat_b"] * 30 + np.random.randn(n) * 10

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = f"{tmpdir}/data.csv"
        df.to_csv(csv_path, index=False)

        result = predict_from_csv(
            csv_path, target="max_force",
            model_type="rf",
            output_dir=f"{tmpdir}/results"
        )

        assert "model" in result
        assert "metrics" in result
        assert os.path.exists(f"{tmpdir}/results/metrics.json")
        assert os.path.exists(f"{tmpdir}/results/predictions.png")
        assert os.path.exists(f"{tmpdir}/results/importance.png")

    print("✓ test_predict_from_csv PASSED")


if __name__ == "__main__":
    test_structure_graph_node_manipulation()
    test_parametric_generation()
    test_ml_train_predictor()
    test_ml_plots()
    test_rl_plots()
    test_save_load_agent()
    test_top_level_imports()
    test_predict_from_csv()
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED ✓")

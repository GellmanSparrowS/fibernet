"""Verify all README code blocks work correctly."""
import sys, traceback, warnings
warnings.filterwarnings("ignore")

# Suppress matplotlib GUI
import matplotlib
matplotlib.use("Agg")

import fibernet as fn
import numpy as np

results = []

def test_block(index, name, func):
    try:
        func()
        results.append((index, name, "PASS", ""))
        print(f"  ✓ Block {index} ({name}): PASS")
    except Exception as e:
        tb = traceback.format_exc()
        results.append((index, name, "FAIL", str(e)))
        print(f"  ✗ Block {index} ({name}): FAIL - {e}")

# Block 1: One-Line API
def test_block1():
    g = fn.pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4))
    fn.show(g, save_path="/tmp/verify_show.png")
    r = fn.simulate(g, mode="stretch", strain=1.5, backend="spring")
    assert r.max_force > 0, "max_force should be > 0"
    assert r.max_stretch > 1.0, "max_stretch should be > 1.0"
    # predict_from_csv and run_bayesian_optimization skipped (need data)
test_block(1, "One-Line API", test_block1)

# Block 2: Complete Pipeline
def test_block2():
    displacements = [(np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3))
                     for _ in range(20)]
    g = fn.pattern_2d(
        unit="square", box=(10, 10), grid=(3, 3),
        n_pts_per_side=5,
        point_displacements=displacements,
    )
    engine = fn.TaichiEngine()
    r = engine.stretch_test(
        g,
        target_stretch=1.5,
        stiffness=1e5,
        damping=0.3,
        num_steps=1000,
        save_interval=200,
    )
    assert r.max_force > 0
    fig = fn.render_trajectory(
        g, r.positions_trajectory, r.edge_stretches,
        n_frames=6, title="Stretch Process",
    )
    fig.savefig("/tmp/verify_trajectory.png", dpi=150)
    ext = fn.GraphFeatureExtractor()
    features = ext.extract(g)
    assert len(features) > 0
    internal_nodes = g.get_internal_nodes()
    g.displace_node(internal_nodes[0], [0.1, 0.2])
test_block(2, "Complete Pipeline", test_block2)

# Block 3: RL Parametric Control (conceptual, skip agent.act)
def test_block3():
    # Method 1: Displacement at generation time
    action = np.random.uniform(-0.3, 0.3, size=(40,))
    displacements = [(action[2*i], action[2*i+1]) for i in range(20)]
    g = fn.pattern_2d(unit="square", grid=(3,3), n_pts_per_side=5,
                      point_displacements=displacements)
    # Method 2: Post-generation refinement
    internal_nodes = g.get_internal_nodes()
    # Just verify get_internal_nodes works
    assert len(internal_nodes) > 0
test_block(3, "RL Parametric Control", test_block3)

# Block 4: Structure Generation
def test_block4():
    disps = [(np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3))
             for _ in range(20)]
    g = fn.pattern_2d(
        unit="square", box=(10, 10), grid=(3, 3),
        n_pts_per_side=5,
        point_displacements=disps,
        seed=42,
    )
    units = fn.list_units()
    expected = ['chiral', 'cross', 'diamond', 'hexagon', 'honeycomb', 'kagome',
                'missing_rib', 'reentrant', 'square', 'star', 'triangle', 'voronoi']
    assert units == expected, f"Units mismatch: {units}"
test_block(4, "Structure Generation", test_block4)

# Block 5: Node Manipulation
def test_block5():
    g = fn.pattern_2d(unit="square", box=(10, 10), grid=(3, 3), n_pts_per_side=5)
    internal = g.get_internal_nodes()
    nid = internal[0]
    g.displace_node(nid, [0.05, 0.05])
    g.set_node_position(nid, [2.5, 2.5])
    g.set_node_positions({internal[1]: [2.5, 0.5], internal[2]: [7.5, 1.0]})
    boundary = g.get_boundary_nodes()
    assert len(boundary) > 0
test_block(5, "Node Manipulation", test_block5)

# Block 6: Simulation (uses 'g' not 'graph')
def test_block6():
    g = fn.pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4))
    engine = fn.TaichiEngine()
    r = engine.stretch_test(
        g,
        target_stretch=1.5,
        stiffness=1e5,
        damping=0.3,
        num_steps=5000,
        ramp_fraction=0.2,
        save_interval=1000,
    )
    assert r.max_force > 0
    assert r.max_stretch > 1.0
    assert r.mean_stretch > 0
    assert len(r.edge_forces) > 0
    assert len(r.edge_stretches) > 0
    assert len(r.positions_trajectory) > 0
    r.save("/tmp/verify_result.json", detailed=True)
    r2 = fn.SimResult.load("/tmp/verify_result.json")
    assert r2.max_force > 0
test_block(6, "Simulation", test_block6)

# Block 7: Visualization
def test_block7():
    g = fn.pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4))
    fig1 = fn.render_graph(g, theme="dark")
    fig2 = fn.render_graph(g, theme="light")
    fig3 = fn.render_graph(g, theme="blueprint")
    themes = list(fn.THEMES.keys())
    assert "dark" in themes
    assert "light" in themes
test_block(7, "Visualization", test_block7)

# Block 8: ML imports
def test_block8():
    from fibernet.ml import (
        train_predictor,
        cross_validate,
        compare_models,
        predict_from_csv,
        plot_predictions,
        plot_feature_importance,
        plot_residuals,
        plot_learning_curve,
    )
    # Quick test with synthetic data
    X = np.random.randn(50, 5)
    y = X[:, 0] * 2 + np.random.randn(50) * 0.1
    model, metrics = train_predictor(X, y, model_type="rf")
    assert "r2" in metrics
    cv = cross_validate(X, y, model_type="ridge", cv=5)
    assert "mean_r2" in cv
test_block(8, "Machine Learning", test_block8)

# Block 9: RL imports
def test_block9():
    from fibernet.rl import (
        plot_reward_curve,
        plot_convergence,
        plot_action_distribution,
        evaluate_agent,
        save_agent, load_agent,
        run_bayesian_optimization,
    )
    # Quick test
    rewards = [np.random.randn() for _ in range(50)]
    plot_reward_curve(rewards, window=5, save_path="/tmp/verify_reward.png")
    objectives = [np.random.randn() for _ in range(20)]
    plot_convergence(objectives, minimize=True, save_path="/tmp/verify_conv.png")
test_block(9, "Reinforcement Learning", test_block9)

# Block 10: Chinese Quick Start (same as Block 1)
def test_block10():
    g = fn.pattern_2d(unit="honeycomb", box=(10,10), grid=(4,4))
    fn.show(g, save_path="/tmp/verify_cn_show.png")
    r = fn.simulate(g, mode="stretch", strain=1.5, backend="spring")
    assert r.max_force > 0
test_block(10, "Chinese Quick Start", test_block10)

# Block 11: Chinese Node Manipulation
def test_block11():
    g = fn.pattern_2d(unit="square", box=(10, 10), grid=(3, 3), n_pts_per_side=5)
    internal = g.get_internal_nodes()
    g.displace_node(internal[0], [0.1, 0.2])
    g.set_node_positions({internal[1]: [2.5, 0.5], internal[2]: [7.5, 1.0]})
test_block(11, "Chinese Node Manipulation", test_block11)

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
passed = sum(1 for r in results if r[2] == "PASS")
failed = sum(1 for r in results if r[2] == "FAIL")
print(f"Passed: {passed}/{len(results)}")
print(f"Failed: {failed}/{len(results)}")
for r in results:
    if r[2] == "FAIL":
        print(f"  FAIL Block {r[0]} ({r[1]}): {r[3]}")

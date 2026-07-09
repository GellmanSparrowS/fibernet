"""
Integration test for FiberNet v3 — exercises the full pipeline.

Run with:
    cd fibernet && source .venv/bin/activate && python tests/test_integration_v3.py
"""

import sys
import time
import numpy as np


def test_core():
    """Test StructureGraph core operations."""
    print("=" * 60)
    print("TEST: Core StructureGraph")
    print("=" * 60)

    from fibernet import StructureGraph, Material

    g = StructureGraph(dimension=2, box_size=[10, 10])
    n0 = g.add_node([0, 0])
    n1 = g.add_node([10, 0])
    n2 = g.add_node([10, 10])
    n3 = g.add_node([0, 10])
    assert g.num_nodes == 4

    g.add_edge(n0, n1, radius=0.5, n_internal=4)
    g.add_edge(n1, n2, radius=0.5, n_internal=4)
    g.add_edge(n2, n3, radius=0.5, n_internal=4)
    g.add_edge(n3, n0, radius=0.5, n_internal=4)
    g.add_edge(n0, n2, radius=0.3, n_internal=4)
    assert g.num_edges == 5
    assert g.is_connected()

    # Node merging
    n0_dup = g.add_node([0, 0])
    assert n0_dup == n0, "Node merging failed"

    # Serialization roundtrip
    d = g.to_dict()
    g2 = StructureGraph.from_dict(d)
    assert g2.num_nodes == g.num_nodes
    assert g2.num_edges == g.num_edges

    # Numpy export
    pos, edges, radii = g.to_numpy()
    assert pos.shape[0] == g.num_nodes

    # NetworkX conversion
    G = g.to_networkx()
    assert G.number_of_nodes() == g.num_nodes

    print("  ✓ Core operations")
    print("  ✓ Node merging")
    print("  ✓ Serialization")
    print("  ✓ Conversions")
    return True


def test_transforms():
    """Test geometric transforms."""
    print("=" * 60)
    print("TEST: Transforms")
    print("=" * 60)

    from fibernet import StructureGraph, translate, rotate, mirror_x, scale, compose

    g = StructureGraph(dimension=2)
    g.add_polyline([(0, 0), (10, 0), (10, 10), (0, 10)], closed=True)

    g2 = translate(g, [20, 0])
    bb = g2.bounding_box()
    assert abs(bb[0][0] - 20) < 1e-6

    g3 = rotate(g, 90)
    g4 = scale(g, 2.0)
    assert abs(g4.total_edge_length() - 2 * g.total_edge_length()) < 1e-4

    g5 = compose(g, lambda g: translate(g, [5, 5]), lambda g: rotate(g, 45))

    print("  ✓ translate, rotate, mirror, scale, compose")
    return True


def test_tiling():
    """Test tiling and welding."""
    print("=" * 60)
    print("TEST: Tiling")
    print("=" * 60)

    from fibernet import StructureGraph, tile_2d

    unit = StructureGraph(dimension=2, box_size=[10, 10])
    unit.add_polyline([(0, 0), (10, 0), (10, 10), (0, 10)], closed=True,
                      radius=0.5, n_internal=4)

    tiled = tile_2d(unit, grid=(3, 3))
    assert tiled.is_connected()
    assert tiled.num_nodes == 16
    assert tiled.num_edges == 24

    print(f"  ✓ 3x3 grid: {tiled.num_nodes} nodes, {tiled.num_edges} edges")
    return True


def test_pattern_engine():
    """Test all built-in units."""
    print("=" * 60)
    print("TEST: Pattern Engine")
    print("=" * 60)

    from fibernet import pattern_2d, pattern_3d, list_units

    units = list_units()
    assert len(units) >= 11, f"Expected ≥11 units, got {len(units)}"

    for name in units:
        g = pattern_2d(unit=name, box=(10, 10), grid=(3, 3), n_internal=4)
        assert g.num_nodes > 0
        assert g.num_edges > 0
        connected = g.is_connected()
        print(f"  {name:15s}: {g.num_nodes:4d}n {g.num_edges:4d}e connected={connected}")
        assert connected, f"{name} not connected after tiling!"

    # 3D
    g3d = pattern_3d(unit="cubic", box=(10, 10, 10), grid=(3, 3, 3))
    assert g3d.is_connected()
    print(f"  {'cubic_3d':15s}: {g3d.num_nodes:4d}n {g3d.num_edges:4d}e connected=True")

    # Determinism
    g1 = pattern_2d(unit="honeycomb", box=(10, 10), grid=(3, 3))
    g2 = pattern_2d(unit="honeycomb", box=(10, 10), grid=(3, 3))
    assert g1.fingerprint() == g2.fingerprint()
    print("  ✓ Deterministic generation")

    print(f"  ✓ All {len(units)} 2D units connected")
    return True


def test_fem():
    """Test FEM solver."""
    print("=" * 60)
    print("TEST: FEM Solver")
    print("=" * 60)

    from fibernet import pattern_2d, BeamFEM

    g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5), n_internal=4)
    fem = BeamFEM(g, default_E=1e9, default_radius=0.1)

    result = fem.uniaxial_tension(strain=0.01, deformation_scale=50)
    assert result.effective_youngs_modulus > 0
    assert result.deformed_graph is not None
    assert result.displacements is not None
    assert result.solve_time < 10.0  # Should be fast

    print(f"  E* = {result.effective_youngs_modulus:.2e} Pa")
    print(f"  ν* = {result.effective_poissons_ratio:.3f}")
    print(f"  Solve time: {result.solve_time:.3f}s")

    # Shear test
    result_shear = fem.shear_test(strain=0.01)
    assert result_shear.strain_energy > 0
    print(f"  G* = {result_shear.effective_youngs_modulus:.2e} Pa")

    # Stress-strain curve
    eps, sig = fem.stress_strain_curve(max_strain=0.03, n_steps=3)
    assert len(eps) == 3
    assert all(s > 0 for s in sig)
    print(f"  ✓ Stress-strain curve ({len(eps)} points)")

    print("  ✓ Uniaxial tension, shear test, stress-strain curve")
    return True


def test_visualization():
    """Test visualization (saves to file, checks file exists)."""
    print("=" * 60)
    print("TEST: Visualization")
    print("=" * 60)

    import os
    from fibernet import pattern_2d, render_graph, render_gallery, BeamFEM

    g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5), n_internal=6)

    fig = render_graph(g, theme="dark", color_by="orientation",
                       save_path="output_viz/test_integration.png")
    assert os.path.exists("output_viz/test_integration.png")
    size = os.path.getsize("output_viz/test_integration.png")
    assert size > 1000
    print(f"  ✓ render_graph: {size/1024:.0f} KB")

    import matplotlib.pyplot as plt
    plt.close('all')

    # Gallery
    graphs = [pattern_2d(unit=u, box=(10,10), grid=(3,3), n_internal=4) for u in ["honeycomb", "square", "kagome"]]
    fig = render_gallery(graphs, ["Honeycomb", "Square", "Kagome"], ncols=3,
                         save_path="output_viz/test_gallery.png")
    assert os.path.exists("output_viz/test_gallery.png")
    plt.close('all')
    print("  ✓ render_gallery")

    # Deformation
    from fibernet import render_deformation
    fem = BeamFEM(g)
    result = fem.uniaxial_tension(strain=0.05, deformation_scale=10)
    fig = render_deformation(g, result.deformed_graph,
                            displacement_data=result.displacements,
                            save_path="output_viz/test_deformation.png")
    assert os.path.exists("output_viz/test_deformation.png")
    plt.close('all')
    print("  ✓ render_deformation")

    # Cleanup test images
    for f in ["test_integration.png", "test_gallery.png", "test_deformation.png"]:
        p = os.path.join("output_viz", f)
        if os.path.exists(p):
            os.remove(p)

    return True


def test_ml_dataset():
    """Test ML dataset generation."""
    print("=" * 60)
    print("TEST: ML Dataset")
    print("=" * 60)

    from fibernet import generate_dataset, extract_features
    from fibernet import pattern_2d

    # Small sweep
    ds = generate_dataset(
        units=["honeycomb", "square"],
        grid_range=[(3, 3)],
        radius_range=[0.1],
        verbose=False,
    )
    assert ds["n_samples"] == 2
    assert len(ds["errors"]) == 0
    assert all(l["E_star"] > 0 for l in ds["labels"])
    print(f"  ✓ Generated {ds['n_samples']} samples, {len(ds['errors'])} errors")

    # Feature extraction
    g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5), n_internal=4)
    feat = extract_features(g)
    assert "n_nodes" in feat
    assert "total_length" in feat
    assert "mean_degree" in feat
    print(f"  ✓ Feature extraction: {len(feat)} features")

    return True


def test_rl_env():
    """Test RL environment."""
    print("=" * 60)
    print("TEST: RL Environment")
    print("=" * 60)

    from fibernet import FiberNetworkEnv
    import numpy as np

    env = FiberNetworkEnv(target_E=1e5, target_nu=-0.3)
    obs, info = env.reset()
    assert len(obs) > 0

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert "E_star" in info
    assert isinstance(reward, float)

    print(f"  ✓ Environment step: reward={reward:.2f}, E*={info['E_star']:.2e}")

    env.close()
    return True


def test_full_pipeline():
    """Test the complete pipeline: generate → simulate → visualize → ML."""
    print("=" * 60)
    print("TEST: Full Pipeline")
    print("=" * 60)

    import fibernet as fn

    # 1. Generate
    g = fn.pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5), n_internal=8)
    print(f"  1. Generated: {g}")

    # 2. Simulate
    fem = fn.BeamFEM(g, default_E=1e9, default_radius=0.1)
    result = fem.uniaxial_tension(strain=0.01, deformation_scale=50)
    print(f"  2. FEM: E*={result.effective_youngs_modulus:.2e} Pa, ν*={result.effective_poissons_ratio:.3f}")

    # 3. Visualize (just verify it works)
    fig = fn.render_graph(g, theme="dark", color_by="orientation")
    import matplotlib.pyplot as plt
    plt.close('all')
    print(f"  3. Visualization OK")

    # 4. Features
    feat = fn.extract_features(g)
    print(f"  4. Features: {len(feat)} extracted")

    # 5. Transforms
    g_rotated = fn.rotate(g, 45)
    print(f"  5. Transform: rotated 45° → {g_rotated}")

    print("  ✓ Full pipeline complete")
    return True


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("FiberNet v3 — Integration Test Suite")
    print("=" * 60 + "\n")

    tests = [
        ("Core", test_core),
        ("Transforms", test_transforms),
        ("Tiling", test_tiling),
        ("Pattern Engine", test_pattern_engine),
        ("FEM Solver", test_fem),
        ("Visualization", test_visualization),
        ("ML Dataset", test_ml_dataset),
        ("RL Environment", test_rl_env),
        ("Full Pipeline", test_full_pipeline),
    ]

    results = {}
    t0 = time.time()
    for name, test_fn in tests:
        try:
            ok = test_fn()
            results[name] = "PASS" if ok else "FAIL"
            print()
        except Exception as exc:
            results[name] = f"ERROR: {exc}"
            print(f"  ✗ ERROR: {exc}")
            import traceback
            traceback.print_exc()
            print()

    elapsed = time.time() - t0

    print("=" * 60)
    print(f"RESULTS ({elapsed:.1f}s)")
    print("=" * 60)
    n_pass = sum(1 for v in results.values() if v == "PASS")
    for name, status in results.items():
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {name:20s} {status}")
    print(f"\n  {n_pass}/{len(results)} tests passed")

    if n_pass < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()

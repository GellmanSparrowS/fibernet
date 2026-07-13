"""
FiberNet Easy API — 一行代码完成常见任务

让高中生也能轻松使用 FiberNet：
- show(graph) → 可视化并保存
- simulate(graph, mode) → FEM 模拟
- batch_simulate(configs) → 批量模拟输出 CSV
- train_model(csv_path) → ML 训练
- train_rl(env_config) → RL 训练

Examples
--------
>>> from fibernet import pattern_2d, show, simulate, batch_simulate
>>> g = pattern_2d("honeycomb", box=(10,10), grid=(4,4))
>>> show(g)  # 一行出图
>>> result = simulate(g, mode="tension")  # 一行模拟
>>> batch_simulate([{"unit": "honeycomb"}, {"unit": "kagome"}], "results.csv")  # 一行批量
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from fibernet.core.structure_graph import StructureGraph
from fibernet.gen.pattern import pattern_2d, pattern_3d
from fibernet.sim.accelerated import TaichiEngine, SimResult
from fibernet.viz.render import render_graph, render_deformation


def show(
    graph: StructureGraph,
    *,
    theme: str = "dark",
    save_path: Optional[str] = None,
    title: str = "",
    **kwargs,
) -> None:
    """一行出图：可视化 StructureGraph。

    Parameters
    ----------
    graph : StructureGraph
        要可视化的结构。
    theme : str
        主题: "dark" (紫黑), "light" (白底), "blueprint", "publication"。
    save_path : str, optional
        保存路径，默认不保存。
    title : str
        标题。

    Examples
    --------
    >>> from fibernet import pattern_2d, show
    >>> g = pattern_2d("honeycomb", box=(10,10), grid=(4,4))
    >>> show(g)
    >>> show(g, theme="light", save_path="honeycomb.png")
    """
    fig = render_graph(graph, theme=theme, title=title, **kwargs)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"✓ Saved to {save_path}")
    return fig


def simulate(
    graph: StructureGraph,
    mode: str = "tension",
    strain: float = 0.01,
    backend: str = "spring",
    save_path: Optional[str] = None,
    **kwargs,
) -> SimResult:
    """一行模拟：力学模拟。

    Parameters
    ----------
    graph : StructureGraph
        要模拟的结构。
    mode : str
        模拟模式:
        弹簧后端: "stretch", "dynamics"
        弹簧后端: "stretch" (位移控制拉伸)
    strain : float
        FEM: 施加应变 (无量纲)
        弹簧: target_stretch 倍数 (如 2.0 = 拉到两倍)
    backend : str
        "spring" (TaichiEngine 质点弹簧动力学)
    save_path : str, optional
        JSON 保存路径。

    Returns
    -------
    SimResult
        模拟结果。

    Examples
    --------
    >>> from fibernet import pattern_2d, simulate
    >>> g = pattern_2d("honeycomb", box=(10,10), grid=(4,4))
    >>> r = simulate(g, mode="tension", strain=0.02)  # FEM
    >>> r = simulate(g, mode="stretch", strain=2.0, backend="spring")  # 质点弹簧拉 2 倍
    """
    if backend == "spring":
        engine = TaichiEngine()
        if mode == "stretch":
            result = engine.stretch_test(graph, target_stretch=strain, **kwargs)
        elif mode == "dynamics":
            result = engine.dynamics(graph, **kwargs)
        else:
            raise ValueError(f"Unknown spring mode '{mode}'. Use: stretch, dynamics")


    if save_path:
        result.save(save_path)
        print(f"✓ Saved to {save_path}")

    return result


def batch_simulate(
    configs: List[Dict[str, Any]],
    output: str = "results.csv",
    mode: str = "tension",
    strain: float = 0.01,
) -> str:
    """一行批量模拟：输出 CSV。

    Parameters
    ----------
    configs : list of dict
        每个 dict 包含: unit, box, grid, seed 等参数。
    output : str
        CSV 输出路径。
    mode : str
        模拟模式。
    strain : float
        施加应变。

    Returns
    -------
    str
        CSV 文件路径。

    Examples
    --------
    >>> from fibernet import batch_simulate
    >>> configs = [
    ...     {"unit": "honeycomb", "grid": (4,4)},
    ...     {"unit": "kagome", "grid": (4,4)},
    ...     {"unit": "reentrant", "grid": (3,3)},
    ... ]
    >>> batch_simulate(configs, "results.csv")
    """
    import csv

    rows = []
    for i, cfg in enumerate(configs):
        unit = cfg.get("unit", "honeycomb")
        box = cfg.get("box", (10, 10))
        grid = cfg.get("grid", (4, 4))
        seed = cfg.get("seed", 42)

        try:
            g = pattern_2d(unit=unit, box=box, grid=grid, seed=seed, **cfg.get("kwargs", {}))
            result = simulate(g, mode=mode, strain=strain, backend="fem")

            row = {
                "id": i,
                "unit": unit,
                "grid_x": grid[0] if isinstance(grid, tuple) else grid,
                "grid_y": grid[1] if isinstance(grid, tuple) else grid,
                "n_nodes": g.num_nodes,
                "n_edges": g.num_edges,
                "E_star": result.effective_youngs_modulus,
                "nu_star": result.effective_poissons_ratio,
                "energy": result.strain_energy,
                "mode": mode,
                "strain": strain,
            }
            rows.append(row)
            print(f"  ✓ {i+1}/{len(configs)}: {unit} (E*={row['E_star']:.2e})")

        except Exception as e:
            print(f"  ✗ {i+1}/{len(configs)}: {unit} — {e}")

    if rows:
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"✓ Saved {len(rows)} results to {output}")

    return output


def train_model(
    csv_path: str,
    target: str = "E_star",
    features: Optional[List[str]] = None,
    model_type: str = "rf",
    test_size: float = 0.2,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """一行 ML 训练：从 CSV 预测属性。

    Parameters
    ----------
    csv_path : str
        batch_simulate 输出的 CSV。
    target : str
        目标列名。
    features : list of str, optional
        特征列名，默认自动选择。
    model_type : str
        模型: "rf" (随机森林), "lr" (线性回归), "gb" (梯度提升)。
    test_size : float
        测试集比例。
    save_path : str, optional
        模型保存路径。

    Returns
    -------
    dict
        包含 R2, RMSE 等指标。

    Examples
    --------
    >>> from fibernet import train_model
    >>> metrics = train_model("results.csv", target="E_star")
    >>> print(f"R² = {metrics['r2']:.3f}")
    """
    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
    except ImportError:
        raise ImportError("需要安装 pandas 和 scikit-learn: pip install pandas scikit-learn")

    df = pd.read_csv(csv_path)

    if features is None:
        features = [c for c in df.columns if c not in [target, "id", "unit", "mode"]]

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "lr":
        model = LinearRegression()
    elif model_type == "gb":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use: rf, lr, gb")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metrics = {"r2": r2, "rmse": rmse, "n_samples": len(df), "features": features}
    print(f"✓ R²={r2:.3f}, RMSE={rmse:.2e}, samples={len(df)}")

    if save_path:
        import pickle
        with open(save_path, "wb") as f:
            pickle.dump({"model": model, "features": features, "metrics": metrics}, f)
        print(f"✓ Model saved to {save_path}")

    return metrics


def train_rl(
    env_config: Optional[Dict] = None,
    n_episodes: int = 100,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """一行 RL 训练：训练 RL agent 优化结构。

    Parameters
    ----------
    env_config : dict, optional
        环境配置，默认使用标准配置。
    n_episodes : int
        训练回合数。
    save_path : str, optional
        保存路径。

    Returns
    -------
    dict
        训练统计。

    Examples
    --------
    >>> from fibernet import train_rl
    >>> stats = train_rl(n_episodes=100)
    >>> print(f"Final reward: {stats['final_reward']:.2f}")
    """
    from fibernet.sim.rl_env import FiberNetworkEnv

    env = FiberNetworkEnv(**(env_config or {}))

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        if (ep + 1) % 10 == 0:
            avg = np.mean(rewards[-10:])
            print(f"  Episode {ep+1}/{n_episodes}: avg_reward={avg:.2f}")

    env.close()

    stats = {
        "n_episodes": n_episodes,
        "final_reward": rewards[-1],
        "mean_reward": np.mean(rewards),
        "max_reward": np.max(rewards),
    }

    if save_path:
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Stats saved to {save_path}")

    print(f"✓ Mean reward: {stats['mean_reward']:.2f}, Max: {stats['max_reward']:.2f}")
    return stats

def batch_simulate_from_json(
    json_dir: str,
    output_dir: str = "simulation_results",
    mode: str = "stretch",
    target_stretch: float = 1.5,
    stiffness: float = 1e5,
    damping: float = 0.3,
    num_steps: int = 1000,
    save_interval: int = 200,
    backend: str = "spring",
) -> str:
    """Batch simulate structures from JSON files.
    
    Reads all .json files in json_dir, simulates each, and saves results.
    
    Parameters
    ----------
    json_dir : str
        Directory containing StructureGraph JSON files.
    output_dir : str
        Directory to save results (CSV + individual JSON results).
    mode : str
        Simulation mode: "stretch", "dynamics".
    target_stretch : float
        Target stretch ratio for stretch mode.
    stiffness : float
        Spring stiffness for spring backend.
    damping : float
        Damping ratio.
    num_steps : int
        Number of simulation steps.
    save_interval : int
        Save trajectory every N steps.
    backend : str
        Simulation backend: "spring" (TaichiEngine).
    
    Returns
    -------
    str
        Path to the output CSV file.
    
    Examples
    --------
    >>> from fibernet import batch_simulate_from_json
    >>> csv_path = batch_simulate_from_json(
    ...     "my_structures/",
    ...     output_dir="results/",
    ...     mode="stretch",
    ...     target_stretch=1.5,
    ... )
    >>> print(f"Results saved to {csv_path}")
    
    Notes
    -----
    - Automatically parses JSON files in various formats (StructureGraph, networkx, custom)
    - Saves trajectory data for visualization
    - Supports checkpoint resume (skips already simulated files)
    - Output CSV includes: filename, n_nodes, n_edges, max_force, max_stretch, etc.
    """
    import csv
    import json
    from pathlib import Path
    
    json_path = Path(json_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_files = sorted(json_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return ""
    
    print(f"Found {len(json_files)} JSON files in {json_dir}")
    
    # Checkpoint: load existing results
    csv_file = output_path / "simulation_results.csv"
    existing = set()
    if csv_file.exists():
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            existing = {row["filename"] for row in reader}
    
    results = []
    engine = TaichiEngine() if backend == "spring" else None
    
    for json_file in json_files:
        fname = json_file.name
        if fname in existing:
            print(f"  [{fname}] Already simulated, skipping")
            continue
        
        try:
            # Try loading as StructureGraph
            g = StructureGraph.load_json(str(json_file))
            
            # Run simulation
            if mode == "stretch":
                r = engine.stretch_test(
                    g,
                    target_stretch=target_stretch,
                    stiffness=stiffness,
                    damping=damping,
                    num_steps=num_steps,
                    save_interval=save_interval,
                    auto_steps=False,
                )
                row = {
                    "filename": fname,
                    "n_nodes": g.num_nodes,
                    "n_edges": g.num_edges,
                    "max_force": float(r.max_force),
                    "max_stretch": float(r.max_stretch),
                    "mean_stretch": float(r.mean_stretch),
                    "std_stretch": float(r.std_stretch),
                    "time_seconds": float(r.time_seconds),
                    "success": True,
                }
                results.append(row)
                
                # Save detailed result
                r.save(str(output_path / f"{json_file.stem}_result.json"), detailed=True)
                print(f"  [{fname}] max_force={r.max_force:.0f}, stretch={r.max_stretch:.3f}")
            else:
                raise ValueError(f"Unknown mode: {mode}")
        
        except Exception as e:
            results.append({
                "filename": fname,
                "success": False,
                "error": str(e),
            })
            print(f"  [{fname}] FAILED: {e}")
        
        # Append to CSV after each simulation (checkpoint)
        if results:
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
    
    # Final CSV with all results
    if results:
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        ok = [r for r in results if r.get("success")]
        print(f"\n✓ Simulated {len(ok)}/{len(results)} successfully")
        print(f"  Results: {csv_file}")
    
    return str(csv_file)


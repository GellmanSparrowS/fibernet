#!/usr/bin/env python3
"""
Generate the FiberNet v4.0 tutorial notebook.

Usage:
    python3 generate_tutorial.py           # create .ipynb
    python3 generate_tutorial.py --run     # create + execute notebook
    python3 generate_tutorial.py --test    # run test with 5 samples first
"""

import json
import sys
from pathlib import Path

NB_DIR = Path(__file__).parent


def md(text: str):
    """Create markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.split("\n"),
    }


def code(source: str):
    """Create code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split("\n"),
    }


def build_cells_test():
    """Build cells for small-scale test (5 samples)."""
    cells = []

    # ===== 0. Header =====
    cells.append(md(r"""# FiberNet v4.0 Tutorial — 从生成到优化的完整流水线
## Complete Pipeline: Structure Generation → Simulation → Analysis → ML → RL

**版本 / Version**: `fibernet 4.0.0-dev`

---

## API 速查 / API Quick Reference

| Python API | Description (EN) | 中文说明 |
|---|---|---|
| `pattern_2d(unit, box, grid, n_pts_per_side, point_displacements)` | Generate 2D periodic structure with optional internal point displacements | 生成带内部点位移的2D周期结构 |
| `TaichiEngine().stretch_test(graph, target_stretch, stiffness, damping, num_steps, save_interval)` | Uniaxial stretch simulation with trajectory recording | 单轴拉伸模拟（记录轨迹） |
| `GraphFeatureExtractor().extract(graph)` | Extract 94-dimensional structural features | 提取94维结构特征 |
| `train_predictor(X, y, model_type)` | Train regression model (rf/ridge/gb/svm/mlp), returns (model, metrics) | 训练回归模型 |
| `cross_validate(X, y, model_type, cv)` | K-fold cross-validation with detailed metrics | K折交叉验证 |
| `predict_from_csv(csv_path, target, output_dir)` | One-line ML: load CSV → train → save results | 一行ML：自动训练+出图+保存 |
| `plot_reward_curve(rewards, window)` | Plot reward curve with moving average | 绘制奖励曲线+滑动平均 |
| `run_bayesian_optimization(fn, param_space, n_iter)` | Bayesian optimization for parametric structure design | 贝叶斯优化参数设计 |
| `render_graph(graph, theme)` | Visualize structure (dark/light/blueprint) | 可视化结构 |
| `render_deformation(positions_trajectory, edge_stretches, ...)` | Visualize deformation frames with stress distribution | 可视化形变+应力分布 |
| `g.displace_node(nid, [dx, dy])` | Displace a single node by vector | 按位移向量移动节点 |
| `g.set_node_positions({nid: [x,y], ...})` | Batch-set node positions | 批量设置节点位置 |
| `g.get_internal_nodes()` | Get list of non-boundary node IDs (RL action targets) | 获取非边界节点列表（RL可优化） |

## 安装 / Installation

```bash
pip install fibernet[full]
# or: pip install fibernet[ml,rl] for ML/RL only
```
"""))

    # ===== 1. Setup =====
    cells.append(md(r"""
## 1. Setup / 环境设置

Set output directory for this tutorial.
"""))

    cells.append(code(r'''import os, sys, json, time, warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# --- Configuration ---
N_SAMPLES_TEST = 5  # Change to 2000 for full run
# Output paths
OUT = Path("fibernet_v4_tutorial")
OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT = OUT / "data"
DATA_OUT.mkdir(parents=True, exist_ok=True)
JSON_OUT = DATA_OUT / "json"
JSON_OUT.mkdir(parents=True, exist_ok=True)
IMG_OUT = DATA_OUT / "images"
IMG_OUT.mkdir(parents=True, exist_ok=True)
CSV_OUT = DATA_OUT / "results.csv"
ML_OUT = DATA_OUT / "ml_results"
RL_OUT = DATA_OUT / "rl_results"
(ML_OUT).mkdir(parents=True, exist_ok=True)
(RL_OUT).mkdir(parents=True, exist_ok=True)

print(f"Output directory: {OUT.resolve()}")
print(f"Test mode: {N_SAMPLES_TEST} samples")'''))

    # ===== 2. Import FiberNet =====
    cells.append(md(r"""
## 2. Import & Verify / 导入验证
"""))

    cells.append(code(r'''import fibernet as fn
from fibernet import pattern_2d, TaichiEngine, render_graph, render_deformation, THEMES
from fibernet.analysis.graph_features import GraphFeatureExtractor
from fibernet.ml import train_predictor, cross_validate, compare_models, plot_predictions, plot_feature_importance
from fibernet.rl import plot_reward_curve, plot_convergence, plot_action_distribution, run_bayesian_optimization

print(f"FiberNet v{fn.__version__}")
print(f"Available units: {fn.list_units()}")
print(f"Available themes: {list(THEMES.keys())}"))'''))

    # ===== 3. Generation Demo =====
    cells.append(md(r"""
## 3. Structure Generation / 结构生成

### 3.1 Base Structure (No Displacement)

Generate a base structure with `point_displacements=[(0,0), ...]` — straight edges.
Naming: `square_3x3_pts5_disp0_seed0.json`
"""))

    cells.append(code(r'''# Parameters — these define ALL structures
UNIT = "square"
BOX = (10, 10)
GRID = (3, 3)
N_PTS_PER_SIDE = 5      # ← 每个边上5个内部点
N_DISP_PER_EDGE = N_PTS_PER_SIDE  # displacements per edge
N_SIDES = 4
N_TOTAL_DISP = N_SIDES * N_DISP_PER_EDGE  # 20 total displacement params

print(f"Config: unit={UNIT}, grid={GRID}, n_pts_per_side={N_PTS_PER_SIDE}")
print(f"Total displacement params: {N_TOTAL_DISP} (4 sides × {N_PTS_PER_SIDE} pts)")

# Generate base (zero displacement)
zero_disps = [(0.0, 0.0)] * N_TOTAL_DISP
g_base = pattern_2d(
    unit=UNIT, box=BOX, grid=GRID,
    n_pts_per_side=N_PTS_PER_SIDE,
    point_displacements=zero_disps,
    seed=0,
)
print(f"Base: {g_base.num_nodes} nodes, {g_base.num_edges} edges")
print(f"  Internal nodes (RL targets): {len(g_base.get_internal_nodes())}")
print(f"  Boundary nodes (fixed): {len(g_base.get_boundary_nodes())}")

# Save base JSON
base_name = f"{UNIT}_{GRID[0]}x{GRID[1]}_pts{N_PTS_PER_SIDE}_disp0_seed0"
g_base.save_json(str(JSON_OUT / f"{base_name}.json"))

# Save base visualization
fig = render_graph(g_base, theme="dark", title="Base Structure (No Displacement)")
fig.savefig(str(IMG_OUT / f"{base_name}.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"✓ Saved: {base_name}.json + .png"))'''))

    # ===== 3.2 Parametric Generation =====
    cells.append(md(r"""
### 3.2 Parametric Structures (With Displacements)

Each structure is defined by a 20-dimensional displacement vector `(dx₁,dy₁, ..., dx₁₀,dy₁₀)`.
Values sampled from `U(-0.3, 0.3)` — relative to edge length.

**Naming convention**: `square_3x3_pts5_disp20_seed{i}.json`
- The seed encodes the specific displacement vector
- Enables full traceability: seed → displacements → structure
"""))

    cells.append(code(r'''def generate_parametric(seed, n_disp=N_TOTAL_DISP):
    """Generate a parametric structure with random displacements.
    
    Parameters
    ----------
    seed : int
        Random seed for displacement generation.
    n_disp : int
        Number of displacement pairs (should match 4 × n_pts_per_side).
    
    Returns
    -------
    StructureGraph : The generated structure.
    displacements : list of (dx, dy) tuples — the actual displacements used.
    """
    rng = np.random.default_rng(seed)
    # Generate displacements: fraction of edge length, bounded to [-0.3, 0.3]
    disp_magnitudes = rng.uniform(-0.3, 0.3, size=n_disp * 2)
    displacements = [
        (float(disp_magnitudes[2*i]), float(disp_magnitudes[2*i+1]))
        for i in range(n_disp)
    ]
    
    g = pattern_2d(
        unit=UNIT, box=BOX, grid=GRID,
        n_pts_per_side=N_PTS_PER_SIDE,
        point_displacements=displacements,
        seed=seed,  # seed for any additional randomness in pattern
    )
    return g, displacements


# Test: generate 3 with different seeds
for test_seed in [100, 200, 300]:
    g, disps = generate_parametric(test_seed)
    print(f"seed={test_seed}: {g.num_nodes} nodes, "
          f"disp_rms={np.sqrt(np.mean([d[0]**2+d[1]**2 for d in disps])):.4f}")'''))

    # ===== 3.3 Batch Generation =====
    cells.append(md(r"""
### 3.3 Batch Generate All Structures

Generate `{N_SAMPLES_TEST}` structures (1 base + `{N_SAMPLES_TEST-1}` parametric variants).

Each structure gets:
- **JSON**: `json/square_3x3_pts5_disp{N}seed{i}.json` — full structure data
- **Image**: `images/square_3x3_pts5_disp{N}seed{i}.png` — visualization
- **Metadata**: stored in `displacements_{i}.json` for traceability
"""))

    cells.append(code(r'''N = N_SAMPLES_TEST
metadata_records = []

# Generate base (seed=0, zero displacement)
print(f"[1/{N}] Generating base structure (seed=0, zero displacement)...")
g0, _ = generate_parametric(0)
g0.save_json(str(JSON_OUT / f"{base_name}.json"))
fig = render_graph(g0, theme="dark", title="Base")
fig.savefig(str(IMG_OUT / f"{base_name}.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
metadata_records.append({
    "id": 0, "seed": 0, "name": base_name,
    "is_base": True, "displacements": [[0,0]] * N_TOTAL_DISP,
    "n_nodes": g0.num_nodes, "n_edges": g0.num_edges,
})

# Generate parametric variants
print(f"\nGenerating {N-1} parametric variants...")
for i in range(1, N):
    g, disps = generate_parametric(seed=1000 + i)
    name = f"{UNIT}_{GRID[0]}x{GRID[1]}_pts{N_PTS_PER_SIDE}_disp{N_TOTAL_DISP}_seed{1000+i}"
    
    g.save_json(str(JSON_OUT / f"{name}.json"))
    fig = render_graph(g, theme="dark", title=f"Variant {i} (seed={1000+i})")
    fig.savefig(str(IMG_OUT / f"{name}.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    
    metadata_records.append({
        "id": i, "seed": 1000 + i, "name": name,
        "is_base": False,
        "displacements": disps,
        "n_nodes": g.num_nodes, "n_edges": g.num_edges,
    })
    
    if (i + 1) % 20 == 0 or i == N - 1:
        print(f"  [{i+1}/{N}] Generated")

# Save metadata
with open(str(DATA_OUT / "metadata.json"), "w") as f:
    json.dump(metadata_records, f, indent=2, default=str)

print(f"\n✓ Generated {N} structures")
print(f"  JSON files: {len(list(JSON_OUT.glob('*.json')))}")
print(f"  Image files: {len(list(IMG_OUT.glob('*.png')))}")
print(f"  Metadata: {DATA_OUT / 'metadata.json'}"))'''))

    # ===== 4. Visualization Gallery =====
    cells.append(md(r"""
### 3.4 Structure Gallery

View all generated structures in one panel.
"""))

    cells.append(code(r'''from PIL import Image

# Load and arrange images
n = min(N, 9)  # Show up to 9
ncols, nrows = 3, (n + 2) // 3
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
fig.patch.set_facecolor("#0a0a0f")
axes = axes.flatten()

for idx in range(n):
    rec = metadata_records[idx]
    img_path = IMG_OUT / f"{rec['name']}.png"
    if img_path.exists():
        img = Image.open(img_path)
        axes[idx].imshow(np.asarray(img))
        axes[idx].set_title(f"{rec['name'][:30]}", color="#aaa", fontsize=8)
        axes[idx].axis("off")

for idx in range(n, len(axes)):
    axes[idx].axis("off")

plt.tight_layout()
plt.savefig(str(IMG_OUT / "gallery_all.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"✓ Gallery saved: {IMG_OUT / 'gallery_all.png'}")
plt.close(fig)

# Display inline (Jupyter)
try:
    from IPython.display import display
    display(Image.open(IMG_OUT / "gallery_all.png"))
except ImportError:
    pass'''))

    # ===== 5. Simulation =====
    cells.append(md(r"""
## 4. Stretch Simulation / 拉伸模拟

### 4.1 Single Structure Simulation

Simulate uniaxial stretch with:
- **Relaxation phase**: energy minimization before stretching
- **Stretch phase**: controlled displacement up to `target_stretch=1.5` (1.5× elongation)
- **Trajectory**: saved every `save_interval` steps for visualization
"""))

    cells.append(code(r'''engine = TaichiEngine()

# Simulate the base structure first
r = engine.stretch_test(
    g_base,
    target_stretch=1.5,      # 拉到1.5倍
    stiffness=1e5,            # spring stiffness
    damping=0.3,              # damping ratio
    num_steps=1000,           # total steps
    save_interval=200,        # save trajectory every 200 steps
    auto_steps=False,         # use fixed step count
)

print(f"Simulation Result:")
print(f"  max_force    = {r.max_force:.1f}")
print(f"  max_stretch  = {r.max_stretch:.3f}")
print(f"  mean_stretch = {r.mean_stretch:.3f}")
print(f"  std_stretch  = {r.std_stretch:.3f}")
print(f"  trajectory   = {len(r.positions_trajectory)} frames")
print(f"  edge_forces  = {len(r.edge_forces) if r.edge_forces is not None else 'N/A'} values"))'''))

    # ===== 5.2 Batch Simulation =====
    cells.append(md(r"""
### 4.2 Batch Simulation

Run stretch tests on all generated structures with `tqdm` progress bar.

Each result saved as `{name}_result.json`.
"""))

    cells.append(code(r'''# Batch simulation
sim_results = []

for rec in tqdm(metadata_records, desc="Simulating"):
    g_path = JSON_OUT / f"{rec['name']}.json"
    g = fn.StructureGraph.load_json(str(g_path))
    
    try:
        r = engine.stretch_test(
            g,
            target_stretch=1.5,
            stiffness=1e5,
            damping=0.3,
            num_steps=1000,
            save_interval=200,
            auto_steps=False,
        )
        
        row = {
            "id": rec["id"],
            "name": rec["name"],
            "is_base": rec["is_base"],
            "max_force": float(r.max_force),
            "max_stretch": float(r.max_stretch),
            "mean_stretch": float(r.mean_stretch),
            "std_stretch": float(r.std_stretch),
            "n_nodes": g.num_nodes,
            "n_edges": g.num_edges,
            "time_seconds": float(r.time_seconds),
            "success": True,
        }
        sim_results.append(row)
        
        # Save trajectory data for visualization
        r.save(str(DATA_OUT / f"{rec['name']}_result.json"))
        
    except Exception as e:
        sim_results.append({
            "id": rec["id"], "name": rec["name"],
            "success": False, "error": str(e),
        })

# Save results to CSV
df_sim = pd.DataFrame(sim_results)
df_sim.to_csv(CSV_OUT, index=False)
print(f"\n✓ Simulated {len(sim_results)} structures")
print(f"  Successful: {df_sim['success'].sum()}")
print(f"  Failed: {(~df_sim['success']).sum()}")
print(f"  Results saved to: {CSV_OUT}")

# Show summary stats
if df_sim['success'].sum() > 0:
    ok = df_sim[df_sim['success']]
    print(f"\nSummary of successful simulations:")
    print(f"  max_force:    {ok['max_force'].mean():.0f} ± {ok['max_force'].std():.0f}")
    print(f"  max_stretch:  {ok['max_stretch'].mean():.3f} ± {ok['max_stretch'].std():.3f}")
    print(f"  time:         {ok['time_seconds'].mean():.2f}s")'''))

    # ===== 5.3 Deformation Visualization =====
    cells.append(md(r"""
### 4.3 Deformation Visualization (Stress Distribution)

Visualize the stretch process with stress (edge stretch) distribution across frames.
This uses the `positions_trajectory` saved during simulation.
"""))

    cells.append(code(r'''# Visualize deformation of the first parametric structure
rec = metadata_records[1]
r_path = DATA_OUT / f"{rec['name']}_result.json"
if r_path.exists():
    from fibernet.sim.accelerated import SimResult
    r = SimResult.load(str(r_path))
    
    print(f"Visualizing: {rec['name']}")
    print(f"  Frames: {len(r.positions_trajectory)}")
    print(f"  max_force: {r.max_force:.0f}")
    print(f"  max_stretch: {r.max_stretch:.3f}")
    
    # Use render_deformation to show stress distribution
    fig = render_deformation(
        positions_trajectory=r.positions_trajectory,
        edge_stretches=r.edge_stretches,
        n_frames=min(6, len(r.positions_trajectory)),
        theme="dark",
        title=f"Stretch Process: {rec['name']}",
    )
    fig.savefig(str(IMG_OUT / f"{rec['name']}_deformation.png"), 
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ Saved: {IMG_OUT / f'{rec[\\\"name\\\"]}_deformation.png'}")
    
    # Show inline
    try:
        from IPython.display import display
        display(Image.open(IMG_OUT / f"{rec['name']}_deformation.png"))
    except ImportError:
        pass
else:
    print(f"Result not found: {r_path}")'''))

    # ===== 5.4 All deformations =====
    cells.append(md(r"""
### 4.4 All Structures Deformation

Compare deformation across all structures.
"""))

    cells.append(code(r'''# Generate deformation visualizations for all successful results
deform_count = 0
for rec in tqdm(metadata_records, desc="Rendering deformations"):
    r_path = DATA_OUT / f"{rec['name']}_result.json"
    if not r_path.exists():
        continue
    
    try:
        r = SimResult.load(str(r_path))
        fig = render_deformation(
            positions_trajectory=r.positions_trajectory,
            edge_stretches=r.edge_stretches,
            n_frames=min(4, len(r.positions_trajectory)),
            theme="dark",
            title=rec['name'][:35],
        )
        fig.savefig(str(IMG_OUT / f"{rec['name']}_deform.png"),
                    dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        deform_count += 1
    except Exception as e:
        pass

print(f"✓ Rendered {deform_count} deformation visualizations"))'''))

    # ===== 6. Feature Extraction =====
    cells.append(md(r"""
## 5. Structure Analysis / 结构分析

Extract 94-dimensional feature vectors for each structure.

Features include:
- **34 structural/topological**: nodes, edges, degree, clustering, spectral, etc.
- **18 pore features**: size distribution, shape, spatial uniformity
- **42 contact features**: overlap analysis
"""))

    cells.append(code(r'''ext = GraphFeatureExtractor(canvas_size=256)

feature_records = []
for rec in tqdm(metadata_records, desc="Extracting features"):
    g_path = JSON_OUT / f"{rec['name']}.json"
    g = fn.StructureGraph.load_json(str(g_path))
    
    try:
        features = ext.extract(g)
        record = {"id": rec["id"], "name": rec["name"]}
        for k, v in features.items():
            record[f"feat_{k}"] = float(v) if isinstance(v, (int, float)) else v
        feature_records.append(record)
    except Exception as e:
        print(f"  Warning: feature extraction failed for {rec['name']}: {e}")

df_feat = pd.DataFrame(feature_records)
print(f"✓ Extracted features for {len(df_feat)} structures")
print(f"  Features: {len([c for c in df_feat.columns if c.startswith('feat_')])}")

# Merge with simulation results
df_all = df_sim.merge(df_feat, on=["id", "name"], how="outer")
df_all.to_csv(str(DATA_OUT / "full_results.csv"), index=False)
print(f"✓ Full dataset saved: {DATA_OUT / 'full_results.csv'}")
print(f"  Shape: {df_all.shape} ({len(df_all)} rows, {len(df_all.columns)} columns)")'''))

    # ===== 7. ML =====
    cells.append(md(r"""
## 6. Machine Learning / 机器学习

### 6.1 Prepare Dataset

- **Features**: extracted structural features
- **Target**: `max_force` from simulation
- **Split**: 80% train / 20% test (no data leakage — split *before* any preprocessing)
"""))

    cells.append(code(r'''from sklearn.model_selection import train_test_split

# Get feature columns
feat_cols = [c for c in df_feat.columns if c.startswith("feat_")]
feat_cols = [c for c in feat_cols if df_feat[c].std() > 1e-12]  # remove constant

# Merge features with targets
df_ml = df_feat.merge(df_sim[df_sim['success']][["id", "max_force"]], on="id", how="inner")
print(f"ML dataset: {len(df_ml)} samples")
print(f"Features: {len(feat_cols)}")

X = df_ml[feat_cols].fillna(0).values
y = df_ml["max_force"].values

# CRITICAL: Split BEFORE any normalization
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(df_ml)), test_size=0.2, random_state=42
)
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")'''))

    cells.append(md(r"""
### 6.2 Train Models

Test multiple model types:
- **Ridge**: linear baseline (interpretable)
- **RandomForest**: non-linear ensemble
- **GradientBoosting**: boosted trees
"""))

    cells.append(code(r'''# Compare models
results = compare_models(X_train, y_train, model_types=["ridge", "rf", "gb"])
print("Model Comparison (on training set with test evaluation):")
for name, metrics in results.items():
    print(f"  {name:12s}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.2e}")'''))

    cells.append(md(r"""
### 6.3 Hyperparameter Tuning (Nested CV — No Leakage!)

Use nested cross-validation:
- **Outer loop**: evaluate model performance (5-fold)
- **Inner loop**: tune hyperparameters (3-fold CV on training fold only)

This prevents data leakage — test data never touches the model during tuning.
"""))

    cells.append(code(r'''# Nested CV: outer 5-fold, inner 3-fold for hyperparameter tuning
from sklearn.model_selection import GridSearchCV, KFold

print("Nested Cross-Validation (no data leakage):")
print("=" * 50)

# Ridge: tune alpha
param_grid_ridge = {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

outer_scores = {}
for model_name, pipeline in [
    ("Ridge", Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])),
    ("RF", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ("GB", GradientBoostingRegressor(random_state=42)),
]:
    if model_name == "Ridge":
        param_grid = {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    elif model_name == "RF":
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10, None],
        }
    else:
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1],
        }
    
    # Outer CV
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    outer_test_r2s = []
    
    for train_idx_outer, test_idx_outer in outer_cv.split(X):
        # Inner CV for hyperparameter tuning (only on outer training data)
        inner_model = (
            Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
            if model_name == "Ridge" else
            RandomForestRegressor(random_state=42, n_jobs=-1)
            if model_name == "RF" else
            GradientBoostingRegressor(random_state=42)
        )
        
        grid = GridSearchCV(inner_model, param_grid, cv=3, scoring="r2", n_jobs=1)
        grid.fit(X[train_idx_outer], y[train_idx_outer])
        
        # Evaluate on outer test fold (model never saw this data!)
        y_pred = grid.predict(X[test_idx_outer])
        from sklearn.metrics import r2_score
        outer_test_r2s.append(r2_score(y[test_idx_outer], y_pred))
    
    outer_scores[model_name] = {
        "mean_r2": float(np.mean(outer_test_r2s)),
        "std_r2": float(np.std(outer_test_r2s)),
        "fold_r2s": outer_test_r2s,
    }
    print(f"  {model_name:12s}: R²={np.mean(outer_test_r2s):.4f} ± {np.std(outer_test_r2s):.4f}")

print("\\n✓ Nested CV complete — no data leakage!")'''))

    cells.append(md(r"""
### 6.4 Final Model Training & Visualization

Train final model on full training set (without the test fold), then evaluate.
"""))

    cells.append(code(r'''from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# Train final models
best_ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])
best_rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
best_gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)

for name, model in [("Ridge", best_ridge), ("RF", best_rf), ("GB", best_gb)]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: test R² = {r2:.4f}")'''))

    cells.append(md(r"""
### 6.5 Prediction Plots

Visualize model performance:
"""))

    cells.append(code(r'''# Plot predictions vs actual (RF model)
y_pred = best_rf.predict(X_test)
fig = plot_predictions(y_test, y_pred, title="RF Predictions vs Actual (Test Set)")
fig.savefig(str(ML_OUT / "predictions_rf.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"✓ Saved: {ML_OUT / 'predictions_rf.png'}")

# Show inline
try:
    from IPython.display import display
    display(Image.open(ML_OUT / "predictions_rf.png"))
except ImportError:
    pass'''))

    cells.append(code(r'''# Feature importance (RF model)
fig = plot_feature_importance(best_rf, feat_cols, top_k=15,
                               title="Top 15 Feature Importance (Random Forest)")
fig.savefig(str(ML_OUT / "importance_rf.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"✓ Saved: {ML_OUT / 'importance_rf.png'}")

try:
    from IPython.display import display
    display(Image.open(ML_OUT / "importance_rf.png"))
except ImportError:
    pass'''))

    # ===== 8. RL =====
    cells.append(md(r"""
## 7. Reinforcement Learning / 强化学习

### 7.1 Bayesian Optimization

Optimize the 20 displacement parameters to minimize `max_force` (structural efficiency).

**Action space**: 20 continuous values in `[-0.3, 0.3]` (displacement of each internal point).

**Reward**: `-max_force` (minimize force = maximize structural efficiency).
"""))

    cells.append(code(r'''# Define parametric generation for Bayesian optimization
def evaluate_displacements(params):
    """Generate structure from displacement params and return max_force.
    
    Parameters
    ----------
    params : dict
        Keys: dx0, dy0, dx1, dy1, ..., dx{N-1}, dy{N-1}
        Values: displacement in [-0.3, 0.3]
    
    Returns
    -------
    float : max_force (to minimize)
    """
    n_pairs = N_TOTAL_DISP
    displacements = []
    for i in range(n_pairs):
        dx = float(np.clip(params.get(f"dx{i}", 0), -0.3, 0.3))
        dy = float(np.clip(params.get(f"dy{i}", 0), -0.3, 0.3))
        displacements.append((dx, dy))
    
    try:
        g = pattern_2d(
            unit=UNIT, box=BOX, grid=GRID,
            n_pts_per_side=N_PTS_PER_SIDE,
            point_displacements=displacements,
            seed=99999,
        )
        r = engine.stretch_test(
            g, target_stretch=1.5, stiffness=1e5, damping=0.3,
            num_steps=500, save_interval=500, auto_steps=False,
        )
        return float(r.max_force)
    except Exception:
        return 1e10  # large penalty


# Evaluate base (zero displacement)
base_force = evaluate_displacements({f"{k}{i}": 0.0 
                                      for i in range(N_TOTAL_DISP)
                                      for k in ["dx", "dy"]})
print(f"Base structure (zero displacement): max_force = {base_force:.0f}")'''))

    cells.append(md(r"""
### 7.2 Run Bayesian Optimization

Search for the best displacement pattern.
"""))

    cells.append(code(r'''# Build param space for skopt
from skopt.space import Real
from skopt import gp_minimize

n_pairs = N_TOTAL_DISP
dimensions = []
dim_names = []
for i in range(n_pairs):
    dimensions.append(Real(-0.3, 0.3, name=f"dx{i}"))
    dim_names.append(f"dx{i}")
    dimensions.append(Real(-0.3, 0.3, name=f"dy{i}"))
    dim_names.append(f"dy{i}")

print(f"Param space: {len(dimensions)} dimensions")
print(f"Optimization: 30 iterations (may take a while)...")


# Track progress
n_iter = 30
all_forces = []
all_params_list = []

def _objective(x):
    params = {dim_names[i]: float(x[i]) for i in range(len(x))}
    force = evaluate_displacements(params)
    all_forces.append(force)
    all_params_list.append(params.copy())
    return force


result = gp_minimize(
    _objective,
    dimensions,
    n_calls=n_iter,
    n_initial_points=10,
    random_state=42,
    verbose=True,
)

print(f"\\n✓ Bayesian Optimization Complete")
print(f"  Best force: {result.fun:.0f} (base was {base_force:.0f})")
print(f"  Improvement: {(1 - result.fun / base_force) * 100:.1f}%")'''))

    cells.append(md(r"""
### 7.3 RL Visualization

Plot the optimization progress and action distribution.
"""))

    cells.append(code(r'''# Plot convergence
fig = plot_convergence(all_forces, minimize=True, 
                       title="Bayesian Optimization: Minimizing max_force")
fig.savefig(str(RL_OUT / "convergence.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)

# Plot reward curve (negative of forces)
rewards = [-f for f in all_forces]
fig = plot_reward_curve(rewards, window=5,
                        title="Bayesian Optimization Progress")
fig.savefig(str(RL_OUT / "reward_curve.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)

# Plot action distribution
fig = plot_action_distribution(all_params_list,
                               title="Optimized Displacement Distribution")
fig.savefig(str(RL_OUT / "action_distribution.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)

print(f"✓ RL visualizations saved to {RL_OUT}")
print(f"  convergence.png, reward_curve.png, action_distribution.png")

# Show inline
try:
    from IPython.display import display
    display(Image.open(RL_OUT / "convergence.png"))
    display(Image.open(RL_OUT / "reward_curve.png"))
except ImportError:
    pass'''))

    cells.append(md(r"""
### 7.4 Best Structure Visualization

Visualize the best structure found by Bayesian optimization.
"""))

    cells.append(code(r'''# Generate and visualize best structure
best_disps = []
for i in range(N_TOTAL_DISP):
    best_disps.append((float(result.x[2*i]), float(result.x[2*i+1])))

g_best = pattern_2d(
    unit=UNIT, box=BOX, grid=GRID,
    n_pts_per_side=N_PTS_PER_SIDE,
    point_displacements=best_disps,
    seed=99999,
)
r_best = engine.stretch_test(
    g_best, target_stretch=1.5, stiffness=1e5, damping=0.3,
    num_steps=500, save_interval=500, auto_steps=False,
)

# Save best structure
best_name = f"{UNIT}_{GRID[0]}x{GRID[1]}_pts{N_PTS_PER_SIDE}_optimal"
g_best.save_json(str(JSON_OUT / f"{best_name}.json"))

# Visualize
fig = render_graph(g_best, theme="dark", title=f"Best Structure (force={r_best.max_force:.0f})")
fig.savefig(str(IMG_OUT / f"{best_name}.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)

print(f"Best: max_force={r_best.max_force:.0f}")
print(f"Baseline: max_force={base_force:.0f}")
print(f"Improvement: {(1 - r_best.max_force / base_force) * 100:.1f}%")

try:
    from IPython.display import display
    display(Image.open(IMG_OUT / f"{best_name}.png"))
except ImportError:
    pass'''))

    # ===== 9. Summary =====
    cells.append(md(r"""
## 8. Summary / 总结

### Workflow Completed ✓

| Step | Status | Description |
|------|--------|-------------|
| Generation | ✓ | 1 base + {N-1} parametric variants (n_pts_per_side=5) |
| JSON Save | ✓ | Full structure data with traceable naming |
| Visualization | ✓ | Structure gallery + deformation with stress distribution |
| Simulation | ✓ | Batch stretch test with relaxation |
| Features | ✓ | 94-dim feature extraction |
| ML | ✓ | Train/test split → hyperparameter tuning → evaluation |
| RL | ✓ | Bayesian optimization + convergence visualization |

### Output Files

```
fibernet_v4_tutorial/
├── data/
│   ├── json/               # Structure JSON files
│   ├── images/             # Visualizations
│   ├── results.csv         # Simulation results
│   ├── full_results.csv    # Features + simulation
│   ├── metadata.json       # Generation parameters
│   ├── ml_results/         # ML plots + model
│   └── rl_results/         # RL plots + convergence
└── ...
```

### Next Steps / 下一步

1. Scale to 2000 samples: change `N_SAMPLES_TEST = 2000`
2. Try different units: `voronoi`, `honeycomb`, `triangle`
3. Connect RL to continuous action space via `displace_node()`
4. Use ML model as surrogate for faster RL reward
5. Deploy as PyPI package: `python3 -m build && twine upload dist/*`

---
*Generated with FiberNet v4.0.0-dev*
""".format(N=N_SAMPLES_TEST)))

    return cells


def build_cells_full():
    """Build cells for full 2000-sample run (similar but with checkpointing)."""
    # For now, just return the test cells with N_SAMPLES_TEST=2000
    # User can switch between test and full by changing N in the notebook
    return build_cells_test()


def build_notebook(cells, n_samples):
    """Build full notebook JSON."""
    return {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0",
            },
        },
        "cells": cells,
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "create"

    if mode == "--test":
        # Run with 5 samples first
        cells = build_cells_test()
        nb = build_notebook(cells, 5)
        out_path = NB_DIR / "fibernet_v4_tutorial_test.ipynb"
        with open(out_path, "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"✓ Test notebook: {out_path} (5 samples)")

    elif mode == "--run":
        # Full run notebook
        cells = build_cells_test()  # Uses N_SAMPLES_TEST config cell
        nb = build_notebook(cells, 2000)
        out_path = NB_DIR / "fibernet_v4_tutorial.ipynb"
        with open(out_path, "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"✓ Tutorial notebook: {out_path} (set N_SAMPLES_TEST in cell)")

    else:
        # Default: create test version
        cells = build_cells_test()
        nb = build_notebook(cells, 5)
        out_path = NB_DIR / "fibernet_v4_tutorial_test.ipynb"
        with open(out_path, "w") as f:
            json.dump(nb, indent=1, f, ensure_ascii=False)
        print(f"✓ Test notebook created: {out_path}")


if __name__ == "__main__":
    main()

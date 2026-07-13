#!/usr/bin/env python3
"""Build the FiberNet v4 tutorial notebook."""
import json
from pathlib import Path

NB_DIR = Path(__file__).parent
OUT_PATH = NB_DIR / "fibernet_v4_tutorial.ipynb"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines] if lines else [""]}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.split("\n")}


def build():
    cells = []

    # ─── Cell 0: Title + API Reference ───
    cells.append(md(
        "# FiberNet v4.0 Tutorial \u2014 \u4ece\u751f\u6210\u5230\u4f18\u5316\u7684\u5b8c\u6574\u6d41\u6c34\u7ebf",
        "## Complete Pipeline: Structure Generation \u2192 Simulation \u2192 Analysis \u2192 ML \u2192 RL",
        "",
        "**\u7248\u672c / Version**: `fibernet 4.0.0-dev`",
        "",
        "---",
        "",
        "## API Quick Reference / API \u901f\u67e5",
        "",
        "| API | Description | \u8bf4\u660e |",
        "|-----|-------------|------|",
        "| `pattern_2d(unit, box, grid, n_pts_per_side, point_displacements)` | Generate 2D periodic structure | \u751f\u62102D\u5468\u671f\u7ed3\u6784 |",
        "| `TaichiEngine().stretch_test(graph, target_stretch, ...)` | Uniaxial stretch with trajectory | \u5355\u8f74\u62c9\u4f38\uff08\u5e26\u8f68\u8ff9\uff09 |",
        "| `GraphFeatureExtractor().extract(graph)` | Extract 94-dim features | \u63d0\u53d694\u7ef4\u7279\u5f81 |",
        "| `train_predictor(X, y, model_type)` | Train model \u2192 (model, metrics) | \u8bad\u7ec3\u6a21\u578b |",
        "| `cross_validate(X, y, model_type, cv)` | K-fold CV | K\u6298\u4ea4\u53c9\u9a8c\u8bc1 |",
        "| `render_trajectory(graph, trajectory, stretches, n_frames)` | Multi-frame stress visualization | \u591a\u5e27\u5e94\u529b\u53ef\u89c6\u5316 |",
        "| `run_bayesian_optimization(fn, space, n_iter)` | Bayesian optimization | \u8d1d\u53f6\u65af\u4f18\u5316 |",
        "| `g.displace_node(nid, [dx,dy])` | Displace a node | \u79fb\u52a8\u8282\u70b9 |",
        "| `g.get_internal_nodes()` | Get internal node IDs (RL targets) | \u83b7\u53d6\u5185\u90e8\u8282\u70b9 |",
        "",
        "## Installation",
        "",
        "```bash",
        "pip install fibernet[full]",
        "# or: pip install fibernet[ml,rl] for ML/RL only",
        "```",
    ))

    # ─── Cell 1: Setup ───
    cells.append(md("", "## 1. Setup / \u73af\u5883\u8bbe\u7f6e", ""))

    cells.append(code('''import os, sys, json, time, warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# \u2500\u2500 Configuration \u2500\u2500
# Change N_SAMPLES to 2000 for the full run
N_SAMPLES = 5  # \u2190 Test with 5, change to 2000 for full
N_PTS_PER_SIDE = 5  # 5 internal points per edge

# Output paths
OUT = Path("fibernet_v4_tutorial")
OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT = OUT / "data"
JSON_OUT = DATA_OUT / "json"
IMG_OUT  = DATA_OUT / "images"
ML_OUT   = DATA_OUT / "ml_results"
RL_OUT   = DATA_OUT / "rl_results"

for d in [JSON_OUT, IMG_OUT, ML_OUT, RL_OUT]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Output: {OUT.resolve()}")
print(f"Mode: {'TEST' if N_SAMPLES < 50 else 'FULL'} \u2014 {N_SAMPLES} samples, n_pts_per_side={N_PTS_PER_SIDE}")'''))

    # ─── Cell 2: Import ───
    cells.append(md("", "## 2. Import FiberNet / \u5bfc\u5165\u9a8c\u8bc1", ""))

    cells.append(code('''import fibernet as fn
from fibernet import (
    pattern_2d, TaichiEngine, render_graph, render_trajectory, THEMES,
)
from fibernet.analysis.graph_features import GraphFeatureExtractor
from fibernet.ml import (
    train_predictor, cross_validate, compare_models,
    plot_predictions, plot_feature_importance,
)
from fibernet.rl import (
    plot_reward_curve, plot_convergence, plot_action_distribution,
)
from fibernet.sim.accelerated import SimResult

print(f"FiberNet v{fn.__version__}")
print(f"Units: {fn.list_units()}")
print(f"Themes: {list(THEMES.keys())}")'''))

    # ─── Cell 3: Config + Base ───
    cells.append(md(
        "", "## 3. Structure Generation / \u7ed3\u6784\u751f\u6210", "",
        "### 3.1 Configuration & Base Structure", "",
        "**Naming convention**: `UNIT_GRIDxGRID_ptsN_dispM_seedS.json`",
        "- UNIT: unit type (square, honeycomb, voronoi, ...)",
        "- GRID: grid size (e.g., 3x3)",
        "- N: n_pts_per_side",
        "- M: total displacement params (= 4 \u00d7 N for square)",
        "- S: random seed",
        "",
        "This encoding allows full traceability: any file can be reconstructed from its name.",
    ))

    cells.append(code('''# \u2500\u2500 Generation Parameters \u2500\u2500
UNIT = "voronoi"
BOX = (10, 10)
GRID = (3, 3)
N_PTS = N_PTS_PER_SIDE  # voronoi
N_SEEDS = 8  # voronoi seed count
# Voronoi: different seeds give different topology + auto-displacements  # total displacement params

print(f"Unit: {UNIT}")
print(f"Grid: {GRID}")
print(f"Points per side: {N_PTS}")
print("Total displacement params: N_DISP ({N_SIDES} sides \u00d7 {N_PTS} pts)")
print(f"Internal nodes (RL targets): varies by seed")

# \u2500\u2500 Base Structure (zero displacement) \u2500\u2500
g_base = pattern_2d(
    unit=UNIT, box=BOX, grid=GRID,
    n_pts_per_side=N_PTS,
    seed=0,
    unit_kwargs={"n_seeds": N_SEEDS},
)

base_name = "{UNIT}_{GRID[0]}x{GRID[1]}_pts{N_PTS}_dispN_DISP_seed0"
g_base.save_json(str(JSON_OUT / f"{base_name}.json"))

fig = render_graph(g_base, theme="dark", title="Base Structure (No Displacement)")
fig.savefig(str(IMG_OUT / f"{base_name}.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)

print(f"\\nBase: {g_base.num_nodes} nodes, {g_base.num_edges} edges")
print(f"  Internal: {len(g_base.get_internal_nodes())}")
print(f"  Boundary: {len(g_base.get_boundary_nodes())}")
print(f"\\u2713 Saved: {base_name}.json + .png")'''))

    # ─── Cell 4: Parametric Generation ───
    cells.append(md(
        "", "### 3.2 Parametric Variants + Batch Generation", "",
        "Each variant has a random 20-dim displacement vector sampled from U(-0.3, 0.3).",
        "The seed encodes the displacement vector for full traceability.",
    ))

    cells.append(code('''def generate_parametric(seed, n_disp=N_DISP):
    """Generate structure with random displacements."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(-0.3, 0.3, size=n_disp * 2)
    disps = [(float(raw[2*i]), float(raw[2*i+1])) for i in range(n_disp)]
    
    g = pattern_2d(
        unit=UNIT, box=BOX, grid=GRID,
        n_pts_per_side=N_PTS,
        point_displacements=disps,
        seed=seed,
    )
    return g, disps


# \u2500\u2500 Batch Generation \u2500\u2500
N = N_SAMPLES
metadata = []

# Base already generated
metadata.append({
    "id": 0, "seed": 0, "name": base_name,
    "is_base": True, "n_nodes": g_base.num_nodes, "n_edges": g_base.num_edges,
})

# Generate variants
print(f"Generating {N-1} parametric variants...")
for i in tqdm(range(1, N), desc="Generate"):
    seed = 1000 + i
    g, disps = generate_parametric(seed)
    name = "{UNIT}_{GRID[0]}x{GRID[1]}_pts{N_PTS}_dispN_DISP_seed{seed}"
    
    g.save_json(str(JSON_OUT / f"{name}.json"))
    fig = render_graph(g, theme="dark", title=f"Variant {i} (seed={seed})")
    fig.savefig(str(IMG_OUT / f"{name}.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    
    metadata.append({
        "id": i, "seed": seed, "name": name,
        "is_base": False, "n_nodes": g.num_nodes, "n_edges": g.num_edges,
    })

# Save metadata
with open(str(DATA_OUT / "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\\n\\u2713 Generated {N} structures:")
print(f"  JSON: {len(list(JSON_OUT.glob('*.json')))} files")
print(f"  Images: {len(list(IMG_OUT.glob('*.png')))} files")'''))

    # ─── Cell 5: Gallery ───
    cells.append(md("", "### 3.3 Gallery \u2014 All Generated Structures", ""))

    cells.append(code('''try:
    from PIL import Image
    from IPython.display import display as ipd

    n_show = min(N, 9)
    nc, nr = 3, (n_show + 2) // 3
    fig, axes = plt.subplots(nr, nc, figsize=(4*nc, 4*nr))
    fig.patch.set_facecolor("#0a0a0f")
    axes = axes.flatten()

    for idx in range(n_show):
        rec = metadata[idx]
        img = Image.open(IMG_OUT / f"{rec['name']}.png")
        axes[idx].imshow(np.asarray(img))
        axes[idx].set_title(rec["name"][:30], color="#aaa", fontsize=8)
        axes[idx].axis("off")

    for idx in range(n_show, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(str(IMG_OUT / "gallery.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("\\u2713 Gallery saved")

    try:
        ipd(Image.open(IMG_OUT / "gallery.png"))
    except Exception:
        pass
    plt.close(fig)
except ImportError:
    print("PIL not installed, skipping gallery")'''))

    # ─── Cell 6: Single Simulation ───
    cells.append(md(
        "", "## 4. Simulation / \u6a21\u62df", "",
        "### 4.1 Single Structure Test", "",
        "Uniaxial stretch test with relaxation:",
        "- **Phase 1**: Energy minimization (dynamics relaxation)",
        "- **Phase 2**: Controlled displacement to target stretch ratio",
        "- Trajectory saved every `save_interval` steps",
    ))

    cells.append(code('''engine = TaichiEngine()

# Load base graph
g = fn.StructureGraph.load_json(str(JSON_OUT / f"{base_name}.json"))

# Run stretch test
r = engine.stretch_test(
    g,
    target_stretch=1.5,       # \u62c9\u52301.5\u500d\u957f\u5ea6
    stiffness=1e5,             # \u5f39\u7c27\u521a\u5ea6
    damping=0.3,               # \u963b\u5c3c\u6bd4
    num_steps=1000,            # \u603b\u6b65\u6570
    save_interval=200,         # \u6bcf200\u6b65\u4fdd\u5b58\u8f68\u8ff9
    auto_steps=False,
)

print(f"Simulation Result (base structure):")
print(f"  max_force    = {r.max_force:.1f}")
print(f"  max_stretch  = {r.max_stretch:.3f}")
print(f"  mean_stretch = {r.mean_stretch:.3f}")
print(f"  std_stretch  = {r.std_stretch:.3f}")
print(f"  trajectory   = {len(r.positions_trajectory)} frames")
print(f"  edge_forces  = {len(r.edge_forces)} values")
print(f"  time         = {r.time_seconds:.2f}s")'''))

    # ─── Cell 7: Deformation Vis ───
    cells.append(md(
        "", "### 4.2 Deformation Visualization (Stress Distribution)", "",
        "Multi-frame view of the stretch process with stress (edge stretch) coloring.",
    ))

    cells.append(code('''fig = render_trajectory(
    g,
    r.positions_trajectory,
    r.edge_stretches,
    n_frames=6,
    title=f"Stretch Process: {base_name}",
)
fig.savefig(str(IMG_OUT / f"{base_name}_deform.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\\u2713 Saved: {IMG_OUT}/{base_name}_deform.png")

try:
    from PIL import Image
    from IPython.display import display
    display(Image.open(IMG_OUT / f"{base_name}_deform.png"))
except Exception:
    pass'''))

    # ─── Cell 8: Batch Sim ───
    cells.append(md(
        "", "### 4.3 Batch Simulation", "",
        "Run stretch tests on ALL generated structures.",
        "Uses `tqdm` for hierarchical progress tracking.",
        "Each result saved for checkpoint resume.",
    ))

    cells.append(code('''# \u2500\u2500 Batch Simulation with Checkpoint Resume \u2500\u2500
sim_results = []
ckpt = DATA_OUT / "sim_partial.json"

if ckpt.exists():
    with open(ckpt) as f:
        sim_results = json.load(f)
    done_ids = {r["id"] for r in sim_results}
    print(f"Resuming from checkpoint: {len(sim_results)} already done")
else:
    done_ids = set()

for rec in tqdm(metadata, desc="Simulate", total=len(metadata)):
    if rec["id"] in done_ids:
        continue
    g_path = JSON_OUT / f"{rec['name']}.json"
    try:
        g = fn.StructureGraph.load_json(str(g_path))
        r = engine.stretch_test(
            g, target_stretch=1.5, stiffness=1e5, damping=0.3,
            num_steps=1000, save_interval=200, auto_steps=False,
        )
        row = {
            "id": rec["id"], "name": rec["name"], "is_base": rec["is_base"],
            "max_force": float(r.max_force), "max_stretch": float(r.max_stretch),
            "mean_stretch": float(r.mean_stretch), "std_stretch": float(r.std_stretch),
            "n_nodes": g.num_nodes, "n_edges": g.num_edges,
            "time_seconds": float(r.time_seconds), "success": True,
        }
        sim_results.append(row)
        r.save(str(DATA_OUT / f"{rec['name']}_result.json"), detailed=True)
        with open(str(ckpt), "w") as f:
            json.dump(sim_results, f, indent=2)
    except Exception as e:
        sim_results.append({"id": rec["id"], "name": rec["name"], "success": False, "error": str(e)})

df_sim = pd.DataFrame(sim_results)
df_sim.to_csv(str(DATA_OUT / "sim_results.csv"), index=False)
print(f"\\n\\u2713 Simulation complete: {len(sim_results)} structures")
ok = df_sim[df_sim["success"]]
print(f"  Successful: {len(ok)}")
print(f"  Failed: {(~df_sim['success']).sum()}")
if len(ok) > 0:
    print(f"  max_force: {ok['max_force'].mean():.0f} +/- {ok['max_force'].std():.0f}")
    print(f"  max_stretch: {ok['max_stretch'].mean():.3f} +/- {ok['max_stretch'].std():.3f}")'''))

    # ─── Cell 9: All Deformations ───
    cells.append(md("", "### 4.4 All Structures Deformation", ""))

    cells.append(code('''deform_count = 0
for rec in tqdm(metadata, desc="Deform"):
    r_path = DATA_OUT / f"{rec['name']}_result.json"
    if not r_path.exists():
        continue
    try:
        r = SimResult.load(str(r_path))
        fig = render_trajectory(
            fn.StructureGraph.load_json(str(JSON_OUT / f"{rec['name']}.json")),
            r.positions_trajectory, r.edge_stretches,
            n_frames=4, title=rec["name"][:35],
        )
        fig.savefig(str(IMG_OUT / f"{rec['name']}_deform.png"),
                    dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        deform_count += 1
    except Exception:
        pass
print(f"\\u2713 Rendered {deform_count} deformation visualizations")'''))

    # ─── Cell 10: Features ───
    cells.append(md(
        "", "## 5. Structure Analysis / \u7ed3\u6784\u5206\u6790", "",
        "Extract 94-dimensional feature vectors:",
        "- 34 structural/topological",
        "- 18 pore features",
        "- 42 contact features",
    ))

    cells.append(code('''ext = GraphFeatureExtractor(canvas_size=256)

feat_records = []
for rec in tqdm(metadata, desc="Features"):
    g_path = JSON_OUT / f"{rec['name']}.json"
    g = fn.StructureGraph.load_json(str(g_path))
    try:
        feats = ext.extract(g)
        record = {"id": rec["id"], "name": rec["name"]}
        for k, v in feats.items():
            record[f"feat_{k}"] = float(v) if isinstance(v, (int, float)) else v
        feat_records.append(record)
    except Exception as e:
        tqdm.write(f"  Warning: feat failed for {rec['name']}: {e}")

df_feat = pd.DataFrame(feat_records)
n_feat = len([c for c in df_feat.columns if c.startswith("feat_")])
print(f"\\u2713 Features: {len(df_feat)} samples, {n_feat} dims")

df_all = df_sim.merge(df_feat, on=["id", "name"], how="outer")
df_all.to_csv(str(DATA_OUT / "full_results.csv"), index=False)
print(f"\\u2713 Full dataset: {df_all.shape} (rows, cols)")'''))

    # ─── Cell 11: ML Data ───
    cells.append(md(
        "", "## 6. Machine Learning / \u673a\u5668\u5b66\u4e60", "",
        "### 6.1 Data Preparation (No Leakage!)", "",
        "\u26a0\ufe0f **Critical**: Train/test split happens BEFORE any preprocessing.",
        "This ensures the model never sees test data during tuning.",
    ))

    cells.append(code('''from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Select samples with both features and sim results
ok_ids = set(df_sim[df_sim["success"]]["id"]) & set(df_feat["id"])
df_ml = df_feat[df_feat["id"].isin(ok_ids)].merge(
    df_sim[df_sim["success"]][["id", "max_force"]], on="id"
)

feat_cols = [c for c in df_ml.columns if c.startswith("feat_")]
feat_cols = [c for c in feat_cols if df_ml[c].std() > 1e-12]

X = df_ml[feat_cols].fillna(0).values
y = df_ml["max_force"].values

# \u2500\u2500 CRITICAL: Split BEFORE normalization \u2500\u2500
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"ML Dataset: {len(df_ml)} samples")
print(f"  Features: {len(feat_cols)}")
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")'''))

    # ─── Cell 12: Model Comparison ───
    cells.append(md(
        "", "### 6.2 Model Comparison", "",
        "Test multiple model types on the same train/test split.",
    ))

    cells.append(code('''results = compare_models(X_train, y_train, model_types=["ridge", "rf", "gb"])
print("Model Comparison:")
for name, m in results.items():
    print(f"  {name:12s}: R\\u00b2={m['r2']:.4f}, RMSE={m['rmse']:.2e}")'''))

    # ─── Cell 13: Nested CV ───
    cells.append(md(
        "", "### 6.3 Hyperparameter Tuning (Nested Cross-Validation)", "",
        "- **Outer loop**: 5-fold evaluation",
        "- **Inner loop**: 3-fold hyperparameter tuning (only on outer training fold)",
        "- Test data is NEVER seen during any tuning step",
    ))

    cells.append(code('''from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

print("Nested CV (5 outer \\u00d7 3 inner):")
print("=" * 40)

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

for model_name, make_model, param_grid in [
    ("Ridge", lambda: Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
     {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),
    ("RF", lambda: RandomForestRegressor(random_state=42, n_jobs=-1),
     {"n_estimators": [50, 100], "max_depth": [3, 5, None]}),
    ("GB", lambda: GradientBoostingRegressor(random_state=42),
     {"n_estimators": [50, 100], "max_depth": [3, 5]}),
]:
    outer_r2s = []
    for train_idx, test_idx in outer_cv.split(X):
        model = make_model()
        grid = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=1)
        grid.fit(X[train_idx], y[train_idx])
        y_pred = grid.predict(X[test_idx])
        outer_r2s.append(r2_score(y[test_idx], y_pred))
    print(f"  {model_name}: R\\u00b2={np.mean(outer_r2s):.4f} +/- {np.std(outer_r2s):.4f}")

print("\\n\\u2713 Nested CV complete \u2014 no data leakage!")'''))

    # ─── Cell 14: Final Model + Viz ───
    cells.append(md(
        "", "### 6.4 Final Model + Visualization", "",
        "Train best model on full training set.",
    ))

    cells.append(code('''best_rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Final RF: R\\u00b2={r2:.4f}, RMSE={rmse:.2e}")

# \u2500\u2500 Plots \u2500\u2500
fig = plot_predictions(y_test, y_pred, title="RF Predictions vs Actual (Test)")
fig.savefig(str(ML_OUT / "predictions.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)

fig = plot_feature_importance(best_rf, feat_cols, top_k=15, title="Top 15 Features (RF)")
fig.savefig(str(ML_OUT / "importance.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)

with open(str(ML_OUT / "model.pkl"), "wb") as f:
    pickle.dump({"model": best_rf, "features": feat_cols}, f)

print(f"\\u2713 ML results saved to {ML_OUT}")'''))

    # ─── Cell 15: RL Setup ───
    cells.append(md(
        "", "## 7. Reinforcement Learning / \u5f3a\u5316\u5b66\u4e60", "",
        f"### 7.1 Bayesian Optimization", "",
        'Optimize the N_DISP-dimensional displacement vector to minimize max_force.',
        '',
        'N_DISP = 4 * n_pts_per_side (see Cell 3 for current value).',
        "",
        "**Action space**: N_DISP continuous values in [-0.3, 0.3].",
        "**Reward**: `-max_force` (minimize force = maximize efficiency).",
    ))

    cells.append(code('''# \u2500\u2500 Evaluation Function \u2500\u2500
def evaluate_displacements(params):
    """Generate from params, simulate, return max_force (to minimize)."""
    n_pairs = N_DISP
    disps = []
    for i in range(n_pairs):
        dx = float(np.clip(params.get(f"dx{i}", 0), -0.3, 0.3))
        dy = float(np.clip(params.get(f"dy{i}", 0), -0.3, 0.3))
        disps.append((dx, dy))
    
    try:
        g = pattern_2d(
            unit=UNIT, box=BOX, grid=GRID,
            n_pts_per_side=N_PTS,
            point_displacements=disps, seed=99999,
        )
        r = engine.stretch_test(
            g, target_stretch=1.5, stiffness=1e5, damping=0.3,
            num_steps=500, save_interval=500, auto_steps=False,
        )
        return float(r.max_force)
    except Exception:
        return 1e10

base_force = evaluate_displacements(
    {f"{k}{i}": 0.0 for i in range(N_DISP) for k in ["dx", "dy"]}
)
print(f"Base max_force: {base_force:.0f}")'''))

    # ─── Cell 16: Run BO ───
    cells.append(md(
        "", "### 7.2 Run Optimization", "",
        "30 iterations of Bayesian optimization.",
    ))

    cells.append(code('''from skopt import gp_minimize
from skopt.space import Real

n_pairs = N_DISP
dimensions, dim_names = [], []
for i in range(n_pairs):
    dimensions.append(Real(-0.3, 0.3, name=f"dx{i}"))
    dim_names.append(f"dx{i}")
    dimensions.append(Real(-0.3, 0.3, name=f"dy{i}"))
    dim_names.append(f"dy{i}")

print(f"Param space: {len(dimensions)}D (continuous)")
print(f"Optimization: 30 iterations...")

all_forces = []
all_params_list = []

def _objective(x):
    params = {dim_names[i]: float(x[i]) for i in range(len(x))}
    force = evaluate_displacements(params)
    all_forces.append(force)
    all_params_list.append(params.copy())
    return force

result = gp_minimize(
    _objective, dimensions, n_calls=30, n_initial_points=10,
    random_state=42, verbose=True,
)

print(f"\\n\\u2713 Optimization Complete")
print(f"  Best force: {result.fun:.0f} (base: {base_force:.0f})")
print(f"  Improvement: {(1 - result.fun / max(base_force, 1)) * 100:.1f}%")'''))

    # ─── Cell 17: RL Viz ───
    cells.append(md("", "### 7.3 RL Visualization", ""))

    cells.append(code('''fig = plot_convergence(all_forces, minimize=True,
                       title="Bayesian Optimization: Minimizing max_force")
fig.savefig(str(RL_OUT / "convergence.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)

fig = plot_reward_curve([-f for f in all_forces], window=5,
                        title="Bayesian Optimization Progress")
fig.savefig(str(RL_OUT / "reward_curve.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)

fig = plot_action_distribution(all_params_list,
                               title="Optimized Displacement Distribution")
fig.savefig(str(RL_OUT / "actions.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)

print(f"\\u2713 RL plots saved to {RL_OUT}")'''))

    # ─── Cell 18: Best Structure ───
    cells.append(md("", "### 7.4 Best Structure Visualization", ""))

    cells.append(code('''best_disps = [(float(result.x[2*i]), float(result.x[2*i+1])) for i in range(N_DISP)]
g_best = pattern_2d(
    unit=UNIT, box=BOX, grid=GRID,
    n_pts_per_side=N_PTS,
    point_displacements=best_disps, seed=99999,
)
r_best = engine.stretch_test(
    g_best, target_stretch=1.5, stiffness=1e5, damping=0.3,
    num_steps=500, save_interval=500, auto_steps=False,
)

best_name = f"{UNIT}_{GRID[0]}x{GRID[1]}_pts{N_PTS}_optimal"
g_best.save_json(str(JSON_OUT / f"{best_name}.json"))

fig = render_graph(g_best, theme="dark",
                   title=f"Optimized (force={r_best.max_force:.0f} vs base {base_force:.0f})")
fig.savefig(str(IMG_OUT / f"{best_name}.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)

print(f"\\u2713 Best: force={r_best.max_force:.0f} (base={base_force:.0f}, "
      f"improvement={(1-r_best.max_force/max(base_force,1))*100:.1f}%)")

try:
    from PIL import Image
    from IPython.display import display
    display(Image.open(IMG_OUT / f"{best_name}.png"))
except Exception:
    pass'''))

    # ─── Cell 19: Summary ───
    cells.append(md(
        "", "## 8. Summary / \u603b\u7ed3", "",
        "### Workflow Completed \\u2713", "",
        "| Step | Status | Description |",
        "|------|--------|-------------|",
        "| Generation | \\u2713 | 1 base + N-1 parametric variants (n_pts_per_side=5) |",
        "| JSON Save | \\u2713 | Full traceable naming convention |",
        "| Visualization | \\u2713 | Gallery + trajectory with stress distribution |",
        "| Simulation | \\u2713 | Batch stretch with relaxation + checkpoint resume |",
        "| Features | \\u2713 | 94-dim feature extraction |",
        "| ML | \\u2713 | Train/test split \\u2192 nested CV \\u2192 model evaluation |",
        "| RL | \\u2713 | Bayesian optimization + convergence visualization |",
        "",
        "### Output Files",
        "",
        "```",
        "fibernet_v4_tutorial/data/",
        "\\u251c\\u2500\\u2500 json/                          # Structure JSONs",
        "\\u251c\\u2500\\u2500 images/                        # Visualizations",
        "\\u251c\\u2500\\u2500 sim_results.csv                # Simulation results",
        "\\u251c\\u2500\\u2500 full_results.csv               # Features + simulation",
        "\\u251c\\u2500\\u2500 metadata.json                  # Generation parameters",
        "\\u251c\\u2500\\u2500 ml_results/",
        "\\u2502   \\u251c\\u2500\\u2500 predictions.png",
        "\\u2502   \\u251c\\u2500\\u2500 importance.png",
        "\\u2502   \\u2514\\u2500\\u2500 model.pkl",
        "\\u2514\\u2500\\u2500 rl_results/",
        "    \\u251c\\u2500\\u2500 convergence.png",
        "    \\u251c\\u2500\\u2500 reward_curve.png",
        "    \\u2514\\u2500\\u2500 actions.png",
        "```",
        "",
        "### Next Steps",
        "",
        "1. **Scale to 2000**: Change `N_SAMPLES = 2000` in Cell 1",
        "2. **Different units**: Try `voronoi`, `honeycomb`, `triangle`",
        "3. **RL with continuous actions**: Use `displace_node()` for iterative refinement",
        "4. **ML surrogate for RL**: Use trained RF as fast reward estimator",
        "5. **PyPI release**: `python3 -m build && twine upload dist/*`",
        "",
        "---",
        "*Generated with FiberNet v4.0.0-dev*",
    ))

    # Build notebook
    nb = {
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

    with open(OUT_PATH, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\\u2713 Notebook: {OUT_PATH}")
    print(f"  Cells: {len(cells)}")
    print(f"  Size: {OUT_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build()

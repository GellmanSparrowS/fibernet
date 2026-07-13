#!/usr/bin/env python3
"""
ML Pipeline: Predict max_force from structural features

Usage:
    python3 analysis_scripts/ml_predict_force.py
    python3 analysis_scripts/ml_predict_force.py --target max_stretch

Requires: scikit-learn, pandas, matplotlib
"""
import sys, argparse, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output_data"
CSV_PATH = OUTPUT_DIR / "voronoi_100_results.csv"
ML_OUTPUT = OUTPUT_DIR / "ml_results"
ML_OUTPUT.mkdir(parents=True, exist_ok=True)

# Feature columns (all feat_* columns)
def get_feature_cols(df):
    return [c for c in df.columns if c.startswith("feat_")]


def run_ml(csv_path: str, target: str = "max_force"):
    """Train ML models to predict target from features."""
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} samples, {len(df.columns)} columns")

    # Filter NaN/inf
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    if len(df) < 10:
        print(f"  Not enough data ({len(df)} samples). Need at least 10.")
        return

    feature_cols = get_feature_cols(df)
    # Remove constant features
    feature_cols = [c for c in feature_cols if df[c].std() > 1e-12]
    print(f"  Features: {len(feature_cols)}")

    X = df[feature_cols].fillna(0).values
    y = df[target].values

    # Train/test split
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6))
    fig.patch.set_facecolor('#0a0a0f')

    for idx, (name, model) in enumerate(models.items()):
        print(f"\n{name}:")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

        print(f"  R²: {r2:.4f} (CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f})")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")

        results[name] = {"r2": r2, "rmse": rmse, "mae": mae,
                         "cv_r2_mean": cv_scores.mean(), "cv_r2_std": cv_scores.std()}

        # Scatter plot
        ax = axes[idx] if len(models) > 1 else axes
        ax.scatter(y_test, y_pred, alpha=0.7, c='#b388ff', edgecolors='white', linewidths=0.5)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, '--', color='#ff6644', linewidth=1)
        ax.set_xlabel(f"Actual {target}", color='#d0d0d0')
        ax.set_ylabel(f"Predicted {target}", color='#d0d0d0')
        ax.set_title(f"{name}\nR²={r2:.3f}, RMSE={rmse:.1f}", color='white')
        ax.set_facecolor('#0a0a0f')
        ax.tick_params(colors='#d0d0d0')

    fig.suptitle(f"ML Prediction: {target}", color='white', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(ML_OUTPUT / "ml_scatter.png", dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    # Feature importance (from RandomForest)
    rf = models["RandomForest"]
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_n = 20

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0a0a0f')
    feature_names = [feature_cols[i].replace("feat_", "") for i in sorted_idx[:top_n]]
    ax.barh(range(top_n), importances[sorted_idx[:top_n]], color='#b388ff', alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feature_names, color='#d0d0d0', fontsize=9)
    ax.set_xlabel("Importance", color='#d0d0d0')
    ax.set_title(f"Top {top_n} Feature Importance (predicting {target})", color='white')
    ax.set_facecolor('#0a0a0f')
    ax.tick_params(colors='#d0d0d0')
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(ML_OUTPUT / "feature_importance.png", dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    # Distribution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0a0a0f')
    ax.hist(y, bins=20, color='#b388ff', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax.set_xlabel(target, color='#d0d0d0')
    ax.set_ylabel("Count", color='#d0d0d0')
    ax.set_title(f"Distribution of {target} ({len(y)} samples)", color='white')
    ax.set_facecolor('#0a0a0f')
    ax.tick_params(colors='#d0d0d0')
    fig.tight_layout()
    fig.savefig(ML_OUTPUT / "distribution.png", dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    # Correlation heatmap (top 15 features)
    top_features = [feature_cols[i] for i in sorted_idx[:15]]
    corr = df[top_features + [target]].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#0a0a0f')
    im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    labels = [c.replace("feat_", "") for c in top_features] + [target]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', color='#d0d0d0', fontsize=8)
    ax.set_yticklabels(labels, color='#d0d0d0', fontsize=8)
    ax.set_title(f"Feature Correlation (top 15)", color='white')
    fig.colorbar(im, ax=ax, label='Correlation')
    fig.tight_layout()
    fig.savefig(ML_OUTPUT / "correlation.png", dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    # Save results
    with open(ML_OUTPUT / "ml_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {ML_OUTPUT}/")
    print(f"  ml_scatter.png, feature_importance.png, distribution.png, correlation.png")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="max_force",
                        choices=["max_force", "max_stretch", "mean_stretch", "std_stretch", "energy"])
    parser.add_argument("--csv", default=str(CSV_PATH))
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"CSV not found: {args.csv}")
        print("Run pipeline first: python3 analysis_scripts/pipeline_voronoi_100.py")
        return

    run_ml(args.csv, args.target)


if __name__ == "__main__":
    main()

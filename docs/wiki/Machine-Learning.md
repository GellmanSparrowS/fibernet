# Machine Learning

`fibernet.ml` provides regression, classification, cross-validation, and automated prediction pipelines for structure-property relationships.

## Supported Models

`rf` (Random Forest), `ridge`, `gb` (Gradient Boosting), `svm`, `mlp` (Multi-layer Perceptron). All models work for both regression and classification tasks.

## API Surface

| Function | Purpose |
|----------|---------|
| `train_predictor(X, y, model_type)` | Train → returns `(model, metrics)` |
| `cross_validate(X, y, cv)` | K-fold cross-validation |
| `compare_models(X, y, models)` | Compare multiple models side-by-side |
| `predict_from_csv(csv, target)` | One-line: load → train → evaluate → save |

### Visualization

`plot_predictions`, `plot_feature_importance`, `plot_residuals`, `plot_learning_curve` — each accepts a `save_path` parameter.

## Typical Pattern

1. Generate structures + simulate → collect labels (e.g., `max_force`)
2. Extract features via `GraphFeatureExtractor`
3. Train predictor → evaluate via cross-validation
4. Predict on new structures

## Model Persistence

Standard `joblib.dump()` / `joblib.load()` for serialization. The `predict_from_csv` function auto-saves to `output_dir`.

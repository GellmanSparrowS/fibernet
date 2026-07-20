# Machine Learning

The `fibernet.ml` module provides regression, classification, cross-validation, and automated prediction pipelines for structure-property relationships.

## Supported Models

| Model | Type | Description |
|-------|------|-------------|
| `rf` | Regression/Classification | Random Forest |
| `ridge` | Regression | Ridge regression (L2) |
| `gb` | Regression/Classification | Gradient Boosting |
| `svm` | Regression/Classification | Support Vector Machine |
| `mlp` | Regression/Classification | Multi-layer Perceptron |

## API

### One-Line Prediction

```python
from fibernet.ml import predict_from_csv

result = predict_from_csv(
    "simulation_results.csv",
    target="max_force",
    model_type="rf",
    output_dir="ml_output/",
)
```

This single call performs: CSV loading → train/test split → model training → evaluation → visualization → model saving.

### Manual Training

```python
from fibernet.ml import train_predictor, cross_validate, compare_models

# Train
model, metrics = train_predictor(X_train, y_train, model_type="rf")
print(f"R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}")

# Cross-validation
cv = cross_validate(X, y, model_type="ridge", cv=5)
print(f"CV R² = {cv['mean_r2']:.3f} ± {cv['std_r2']:.3f}")

# Compare multiple models
results = compare_models(X, y, models=["rf", "ridge", "gb", "svm"])
```

### Visualization

```python
from fibernet.ml import (
    plot_predictions,        # predicted vs actual scatter
    plot_feature_importance, # feature importance bar chart
    plot_residuals,          # residual analysis
    plot_learning_curve,     # learning curve
)

plot_predictions(y_true, y_pred, save_path="predictions.png")
plot_feature_importance(model, feature_names, save_path="importance.png")
```

## Metrics

### Regression
- R² (coefficient of determination)
- RMSE (root mean squared error)
- MAE (mean absolute error)

### Classification
- Accuracy
- F1 score
- Confusion matrix

## Typical Workflow

```python
import fibernet as fn
import numpy as np

# 1. Generate structures and simulate
structures = []
forces = []
for seed in range(100):
    g = fn.pattern_2d(unit="honeycomb", box=(10,10), grid=(3,3), seed=seed)
    r = fn.simulate(g, mode="stretch", strain=1.5, backend="spring")
    structures.append(g)
    forces.append(r.max_force)

# 2. Extract features
ext = fn.GraphFeatureExtractor()
X = np.array([ext.extract(g) for g in structures])
y = np.array(forces)

# 3. Train and evaluate
model, metrics = fn.ml.train_predictor(X, y, model_type="rf")

# 4. Predict on new structure
g_new = fn.pattern_2d(unit="honeycomb", box=(10,10), grid=(3,3), seed=999)
f_new = ext.extract(g_new).reshape(1, -1)
predicted_force = model.predict(f_new)[0]
```

## Model Persistence

```python
import joblib

# Save
joblib.dump(model, "force_predictor.pkl")

# Load
model = joblib.load("force_predictor.pkl")
```

"""
Example 15: Machine Learning for Property Prediction

Demonstrates using machine learning to predict fiber network properties:
1. Generate training data (networks + properties)
2. Extract features from networks
3. Train ML models
4. Predict properties for new networks
5. Evaluate model performance

Usage:
    python examples/15_ml_property_prediction.py
"""

import numpy as np
from fibernet import gen
from fibernet.analysis.spatial import compute_spatial_statistics
from fibernet.analysis.homogenization import compute_effective_properties

print("=" * 70)
print("  FiberNet ML Property Prediction Example")
print("=" * 70)

# ============================================================
# Part 1: Generate Training Data
# ============================================================
print("\n[Part 1/5] Generating training data...")
print("-" * 60)

def generate_network_and_features(seed, num_fibers, fiber_length):
    """Generate a network and extract features."""
    net = gen.random_straight_2d(
        num_fibers=num_fibers, 
        fiber_length=fiber_length, 
        seed=seed
    )
    
    # Extract features
    stats = compute_spatial_statistics(net)
    props = compute_effective_properties(net)
    
    features = [
        num_fibers,
        fiber_length,
        stats['nematic_order'],
        stats['anisotropy_index'],
        stats['mean_connectivity'],
        stats['length']['mean'],
        stats['length']['std'],
    ]
    
    # Target: effective modulus
    E = props['elastic'].get('E_x', props['elastic'].get('E', 0))
    
    return np.array(features), E

# Generate training data
n_samples = 50
np.random.seed(42)

X_train = []
y_train = []

for i in range(n_samples):
    seed = np.random.randint(0, 10000)
    num_fibers = np.random.randint(30, 150)
    fiber_length = np.random.uniform(5.0, 15.0)
    
    features, target = generate_network_and_features(seed, num_fibers, fiber_length)
    X_train.append(features)
    y_train.append(target)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"  Generated {n_samples} training samples")
print(f"  Features: {X_train.shape[1]} (n_fibers, length, S, AI, connectivity, L_mean, L_std)")
print(f"  Target range: [{y_train.min():.2e}, {y_train.max():.2e}] Pa")

# ============================================================
# Part 2: Feature Analysis
# ============================================================
print("\n[Part 2/5] Feature analysis...")
print("-" * 60)

feature_names = ['n_fibers', 'length', 'S', 'AI', 'connectivity', 'L_mean', 'L_std']

# Normalize features
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_std[X_std < 1e-10] = 1.0
X_norm = (X_train - X_mean) / X_std

# Correlation with target
correlations = []
for i, name in enumerate(feature_names):
    corr = np.corrcoef(X_norm[:, i], y_train)[0, 1]
    correlations.append(corr)
    print(f"  Correlation({name}, E): {corr:+.3f}")

# Most important features
sorted_features = sorted(zip(feature_names, correlations), key=lambda x: abs(x[1]), reverse=True)
print(f"\n  Most important features:")
for name, corr in sorted_features[:3]:
    print(f"    {name}: |r| = {abs(corr):.3f}")

# ============================================================
# Part 3: Simple Linear Model
# ============================================================
print("\n[Part 3/5] Training linear model...")
print("-" * 60)

# Simple linear regression (manual implementation for demonstration)
# y = X @ w + b

# Split data
n_train = int(0.8 * n_samples)
X_tr = X_norm[:n_train]
y_tr = y_train[:n_train]
X_te = X_norm[n_train:]
y_te = y_train[n_train:]

# Add bias term
X_tr_bias = np.column_stack([X_tr, np.ones(len(X_tr))])
X_te_bias = np.column_stack([X_te, np.ones(len(X_te))])

# Solve least squares
w, _, _, _ = np.linalg.lstsq(X_tr_bias, y_tr, rcond=None)

# Predictions
y_pred = X_te_bias @ w

# Evaluate
mse = np.mean((y_te - y_pred)**2)
rmse = np.sqrt(mse)
r2 = 1 - np.sum((y_te - y_pred)**2) / np.sum((y_te - np.mean(y_te))**2)

print(f"  Training samples: {n_train}")
print(f"  Test samples: {len(X_te)}")
print(f"  RMSE: {rmse:.2e} Pa")
print(f"  R²: {r2:.3f}")

# ============================================================
# Part 4: Predict for New Network
# ============================================================
print("\n[Part 4/5] Predicting for new networks...")
print("-" * 60)

test_cases = [
    {'seed': 999, 'num_fibers': 80, 'fiber_length': 10.0, 'desc': 'Medium random'},
    {'seed': 123, 'num_fibers': 120, 'fiber_length': 8.0, 'desc': 'Dense short'},
    {'seed': 456, 'num_fibers': 50, 'fiber_length': 14.0, 'desc': 'Sparse long'},
]

for case in test_cases:
    features, actual = generate_network_and_features(
        case['seed'], case['num_fibers'], case['fiber_length']
    )
    
    # Normalize
    features_norm = (features - X_mean) / X_std
    features_norm = np.append(features_norm, 1.0)  # Add bias
    
    # Predict
    predicted = features_norm @ w
    
    error = abs(predicted - actual) / actual * 100
    
    print(f"  {case['desc']:<15s}: actual={actual:.2e}, predicted={predicted:.2e}, error={error:.1f}%")

# ============================================================
# Part 5: Cross-Validation
# ============================================================
print("\n[Part 5/5] Cross-validation...")
print("-" * 60)

k_folds = 5
fold_size = n_samples // k_folds
r2_scores = []

for fold in range(k_folds):
    start = fold * fold_size
    end = start + fold_size
    
    X_val = X_norm[start:end]
    y_val = y_train[start:end]
    X_train_cv = np.delete(X_norm, slice(start, end), axis=0)
    y_train_cv = np.delete(y_train, slice(start, end))
    
    X_train_cv_bias = np.column_stack([X_train_cv, np.ones(len(X_train_cv))])
    X_val_bias = np.column_stack([X_val, np.ones(len(X_val))])
    
    w_cv, _, _, _ = np.linalg.lstsq(X_train_cv_bias, y_train_cv, rcond=None)
    y_pred_cv = X_val_bias @ w_cv
    
    r2_cv = 1 - np.sum((y_val - y_pred_cv)**2) / np.sum((y_val - np.mean(y_val))**2)
    r2_scores.append(r2_cv)

print(f"  {k_folds}-fold cross-validation R² scores:")
for i, r2 in enumerate(r2_scores):
    print(f"    Fold {i+1}: {r2:.3f}")

print(f"\n  Mean R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

# ============================================================
print("\n" + "=" * 70)
print("  ML Property Prediction Complete!")
print("=" * 70)
print("""
Summary:
  - Generated training data from fiber network simulations
  - Analyzed feature importance
  - Trained a linear regression model
  - Predicted properties for new networks
  - Validated with k-fold cross-validation

This workflow can be extended with:
  - More sophisticated models (random forest, neural networks)
  - More features (topological, morphological)
  - Hyperparameter optimization
  - Uncertainty quantification
""")

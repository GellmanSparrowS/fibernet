"""
Machine Learning Integration Example
=====================================

This example demonstrates how to use FiberNet's ML integration features
to extract structural features and predict mechanical properties.

Requires: numpy, scipy (core), scikit-learn, tqdm (optional)
"""

import numpy as np
from fibernet import gen, analysis
from fibernet.sim import FiberFEM

print("="*70)
print("FiberNet ML Integration Example")
print("="*70)

# Generate training dataset
print("\n1. Generating training dataset...")
print("-" * 70)

n_samples = 50
dataset = []

for i in range(n_samples):
    # Vary network parameters
    num_fibers = np.random.randint(50, 150)
    fiber_length = np.random.uniform(5.0, 15.0)
    
    net = gen.random_straight_2d(
        num_fibers=num_fibers,
        fiber_length=fiber_length,
        box_size=(50, 50),
        seed=1000 + i
    )
    
    # Extract features
    morph = analysis.MorphologyAnalyzer(net)
    features = {
        'num_fibers': net.num_fibers,
        'mean_length': net.mean_fiber_length,
        'nematic_order': morph.nematic_order_parameter(),
        'porosity': morph.porosity(),
        'tortuosity': np.mean(morph.tortuosity_distribution()),
    }
    
    # Compute target: effective modulus
    try:
        fem = FiberFEM(net, segments_per_fiber=3)
        E_eff = fem.effective_modulus(strain=0.001)
        features['E_eff'] = E_eff
        dataset.append(features)
    except Exception as e:
        print(f"  Sample {i}: Simulation failed - {e}")

print(f"  Successfully generated {len(dataset)} samples")

# Convert to numpy arrays
print("\n2. Preparing feature matrix...")
print("-" * 70)

X = np.array([[d['num_fibers'], d['mean_length'], d['nematic_order'],
               d['porosity'], d['tortuosity']] for d in dataset])
y = np.array([d['E_eff'] for d in dataset])

print(f"  Feature matrix shape: {X.shape}")
print(f"  Target vector shape: {y.shape}")
print(f"  E_eff range: {y.min():.2e} - {y.max():.2e} Pa")

# Feature statistics
print("\n3. Feature statistics...")
print("-" * 70)
feature_names = ['num_fibers', 'mean_length', 'nematic_order', 'porosity', 'tortuosity']
for i, name in enumerate(feature_names):
    print(f"  {name:20s}: mean={X[:,i].mean():.3f}, std={X[:,i].std():.3f}")

# Try to use scikit-learn if available
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    
    print("\n4. Training Random Forest model...")
    print("-" * 70)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n  Model performance:")
    print(f"    R² score: {r2:.3f}")
    print(f"    MAE: {mae:.2e} Pa")
    
    # Feature importance
    print(f"\n5. Feature importance...")
    print("-" * 70)
    importance = model.feature_importances_
    for i, (name, imp) in enumerate(zip(feature_names, importance)):
        print(f"  {name:20s}: {imp:.3f}")
    
    # Make prediction
    print(f"\n6. Making prediction for new network...")
    print("-" * 70)
    new_sample = np.array([[100, 10.0, 0.5, 0.8, 1.1]])
    prediction = model.predict(new_sample)[0]
    print(f"  Input: num_fibers=100, length=10.0, order=0.5, porosity=0.8, tortuosity=1.1")
    print(f"  Predicted E_eff: {prediction:.2e} Pa")
    
except ImportError:
    print("\n4. scikit-learn not available")
    print("-" * 70)
    print("  Install with: pip install scikit-learn")
    print("  Skipping ML model training.")

# Manual correlation analysis (always available)
print("\n7. Manual correlation analysis...")
print("-" * 70)

correlations = []
for i, name in enumerate(feature_names):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    correlations.append((name, corr))
    print(f"  {name:20s}: correlation with E_eff = {corr:.3f}")

print(f"\n8. Summary")
print("-" * 70)
print(f"  Dataset size: {len(dataset)} networks")
print(f"  Features extracted: {len(feature_names)}")
if 'r2' in locals():
    print(f"  Model R² score: {r2:.3f}")
print(f"  Most correlated feature: {max(correlations, key=lambda x: abs(x[1]))[0]}")

print("\n" + "="*70)
print("Example complete!")
print("="*70)
print("\nNext steps:")
print("  - Increase dataset size for better model performance")
print("  - Try different ML models (GradientBoosting, Neural Networks)")
print("  - Add more features (crosslink density, connectivity)")
print("  - Use cross-validation for robust evaluation")

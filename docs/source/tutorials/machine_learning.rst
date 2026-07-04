Machine Learning Tutorial
=========================

Use FiberNet's ML module for structure-property prediction.

Feature Extraction
------------------

Extract structural features from networks:

.. code-block:: python

   from fibernet import gen
   from fibernet.ml.features import FeatureExtractor, extract_features

   # Generate sample network
   net = gen.random_straight_2d(
       num_fibers=50, fiber_length=10, box_size=(30, 30), seed=42
   )

   # Extract features
   features = extract_features(net)
   print(f"Features extracted: {len(features)}")
   print(f"Nematic order: {features['nematic_order']:.3f}")
   print(f"Mean length: {features['mean_length']:.2f}")

Dataset Generation
------------------

Generate a training dataset:

.. code-block:: python

   from fibernet.ml.dataset import generate_dataset

   # Generate 50 networks with varying parameters
   networks, properties, parameters = generate_dataset(
       num_samples=50,
       param_ranges={
           'num_fibers': (20, 100),
           'fiber_length': (5, 20),
           'radius': (0.05, 0.3),
       },
       seed=42
   )

   print(f"Generated {len(networks)} networks")
   print(f"Property range: {properties.min():.4f} to {properties.max():.4f}")

Train Predictor
---------------

Train a machine learning model:

.. code-block:: python

   from fibernet.ml.predictor import PropertyPredictor

   # Create predictor
   predictor = PropertyPredictor(
       model_type='random_forest',
       property_name='modulus'
   )

   # Train on dataset
   predictor.fit(networks, properties)

   # Check training score
   print(f"Training R²: {predictor.model.score(predictor.scaler.transform(X), properties):.4f}")

Predict Properties
------------------

Use trained model for prediction:

.. code-block:: python

   # Generate new network
   test_net = gen.random_straight_2d(
       num_fibers=60, fiber_length=12, box_size=(35, 35), seed=123
   )

   # Predict property
   predicted_value = predictor.predict(test_net)
   print(f"Predicted modulus: {predicted_value:.4e} Pa")

Feature Importance
------------------

Analyze which features matter most:

.. code-block:: python

   importance = predictor.feature_importance()
   
   # Sort by importance
   sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
   
   print("Top 5 features:")
   for name, imp in sorted_imp[:5]:
       print(f"  {name}: {imp:.4f}")

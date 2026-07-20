# Feature Extraction

FiberNet extracts a 94-dimensional feature vector from each `StructureGraph`, capturing structural, topological, and mechanical characteristics suitable for downstream ML/RL tasks.

## Feature Categories

| Category | Dimensions | Description |
|----------|------------|-------------|
| **Structural** | ~34 | Node/edge counts, density, degree statistics, clustering |
| **Spectral** | ~12 | Graph Laplacian eigenvalues, spectral gap |
| **Pore/Contact** | ~18 | Pore size distribution, contact geometry (2D image-based) |
| **Mechanical** | ~30 | Force distribution statistics, energy, anisotropy indicators |

## API

### 2D Feature Extraction

```python
ext = fn.GraphFeatureExtractor()
features = ext.extract(g)  # returns numpy array, shape (94,)
```

### 3D Feature Extraction

```python
from fibernet.analysis.graph_features_3d import GraphFeatureExtractor3D

ext = GraphFeatureExtractor3D()
features = ext.extract(g)  # returns dict with 60 features
```

### Feature Names

```python
names = ext.get_feature_names()
# ['node_count', 'edge_count', 'density', 'mean_degree', ...]
```

## Feature Design

### Structural Features
- Node and edge counts
- Graph density and average degree
- Degree distribution statistics (mean, std, skew, kurtosis)
- Clustering coefficient (local and global)
- Path length statistics

### Spectral Features
- Smallest non-zero Laplacian eigenvalue (algebraic connectivity)
- Spectral gap
- Normalized Laplacian eigenvalues
- Spectral moments

### Pore/Contact Features (2D)
- Voronoi cell area distribution
- Pore size statistics
- Contact angle distribution
- Anisotropy measures

### Mechanical Features
- Force distribution statistics from simulation results
- Energy per edge
- Stretch ratio distribution
- Boundary reaction forces

## Usage in ML Pipeline

```python
# Extract features for a batch of structures
features_list = []
labels = []
for g, sim_result in dataset:
    ext = fn.GraphFeatureExtractor()
    f = ext.extract(g)
    features_list.append(f)
    labels.append(sim_result.max_force)

X = np.array(features_list)
y = np.array(labels)

# Train predictor
model, metrics = fn.ml.train_predictor(X, y, model_type="rf")
```

## Extensibility

The feature extractor is designed to be extensible. New feature categories can be added by subclassing and implementing the `extract()` method. The 3D extractor demonstrates this pattern with a separate feature set optimized for volumetric structures.

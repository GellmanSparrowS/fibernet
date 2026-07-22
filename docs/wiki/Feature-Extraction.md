# Feature Extraction

Extracts numerical feature vectors from `StructureGraph` for downstream ML/RL tasks. Two extractors: 2D (94-dim) and 3D (60-dim).

## Feature Categories

### 2D Extractor (`fn.GraphFeatureExtractor`)

| Category | Approx. Dims | What It Captures |
|----------|-------------|------------------|
| Structural | ~34 | Node/edge counts, density, degree statistics, clustering |
| Spectral | ~12 | Laplacian eigenvalues, spectral gap, spectral moments |
| Pore/Contact | ~18 | Pore size distribution, contact geometry (2D image-based) |
| Mechanical | ~30 | Force distribution, energy, anisotropy (from simulation) |

### 3D Extractor (`GraphFeatureExtractor3D`)

Adapted feature set for volumetric structures. Excludes 2D image-based features; adds volume fraction, surface area, 3D connectivity metrics.

## API

```python
# 2D
ext = fn.GraphFeatureExtractor()
features = ext.extract(g)            # numpy array, shape (94,)
names = ext.get_feature_names()      # human-readable names

# 3D
from fibernet.analysis.graph_features_3d import GraphFeatureExtractor3D
ext3d = GraphFeatureExtractor3D()
features = ext3d.extract(g)          # dict with 60 features
```

## Extensibility

Each extractor is designed to be subclassed. New feature categories can be added by implementing additional extraction methods. The 2D and 3D extractors share a common interface pattern.
